# Double-precision CUDA voting; same metric, votes, and sensitivity as CPU.

import numpy as np
import torch
from time import perf_counter

def histogram_cuda(x, c, schema, tau, squared=False, batch=2048):
    device = 'cuda'
    dtype = torch.float64
    ct = torch.as_tensor(c, dtype=dtype, device=device)
    xt = torch.as_tensor(x, dtype=dtype, device=device)
    mats = {j: torch.as_tensor(a, dtype=dtype, device=device) for j, a in schema.categorical.items()}
    hist = torch.zeros(len(c), dtype=dtype, device=device)
    entropy = torch.zeros((), dtype=dtype, device=device)
    sq = torch.zeros_like(entropy)
    for start in range(0, len(x), batch):
        a = xt[start:start + batch]
        dist = torch.zeros((len(a), len(c)), dtype=dtype, device=device)
        for j in schema.numeric:
            delta = a[:, j, None] - ct[None, :, j]
            dist.add_(delta.square() if squared else delta.abs())
        for j, mat in mats.items():
            dist.add_(mat[a[:, j].long()[:, None], ct[:, j].long()[None, :]])
        dist.div_(schema.p)
        if tau <= 0:
            hist.add_(torch.bincount(dist.argmin(1), minlength=len(c)))
            sq.add_(len(a))
        else:
            v = torch.softmax(-dist / tau, dim=1)
            hist.add_(v.sum(0))
            sq.add_(v.square().sum())
            entropy.sub_((v * v.clamp_min(1e-300).log()).sum())
    return (hist.cpu().numpy(), float(entropy.cpu()) / max(len(x) * np.log(max(len(c), 2)), 1), float(sq.cpu()) / max(len(x), 1))
if __name__ == '__main__':
    import json
    from threadpoolctl import threadpool_limits
    from core import histogram
    from data import schema_for, ROOT
    torch.set_num_threads(1)
    rows = []
    with threadpool_limits(limits=1):
        for name in ['adult', 'bank', 'xor3']:
            s = schema_for(name)
            rng = np.random.default_rng(1923)
            x = s.random(4000, rng)
            c = s.random(512, rng)
            for tau in [0.0, 0.005, 0.05, 0.5]:
                a = histogram(x, c, s, tau)
                b = histogram_cuda(x, c, s, tau)
                error = float(np.max(np.abs(a[0] - b[0])))
                np.testing.assert_allclose(a[0], b[0], rtol=1e-10, atol=1e-09)
                np.testing.assert_allclose(a[1:], b[1:], rtol=1e-10, atol=1e-09)
                rows.append(dict(dataset=name, tau=tau, histogram_max_abs_error=error))
        s = schema_for('adult')
        x = s.random(26048, rng)
        c = s.random(2000, rng)
        for backend, fn in [('cpu', histogram), ('cuda', histogram_cuda)]:
            start = perf_counter()
            fn(x, c, s, 0.05)
            elapsed = perf_counter() - start
            rows.append(dict(backend=backend, n=len(x), m=len(c), tau=0.05, seconds=elapsed))
    (ROOT / 'gpu_validation.json').write_text(json.dumps(dict(torch=torch.__version__, device=torch.cuda.get_device_name(0), rows=rows), indent=2))
    print(json.dumps(rows, indent=2), flush=True)
