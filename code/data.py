# Public benchmark data, frozen schemas, and validation/test separation.

from pathlib import Path
from io import BytesIO
import hashlib, json, zipfile, urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from core import Schema
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'data'

def xor_schema(k=3):
    p = k + 1
    return Schema(f'xor{k}', tuple([f'X{i + 1}' for i in range(k)] + ['Y']), {j: (0.0, 7.0, 1.0) for j in range(k)}, {k: 1 - np.eye(2)}, {j: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5] for j in range(k)})

def xor_data(k=3, n=12000, seed=20260907, eta=0.0):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 8, size=(n, k))
    y = np.bitwise_xor.reduce((raw >= 4).astype(int), axis=1)
    y ^= (rng.random(n) < eta).astype(int)
    return np.column_stack([raw / 7, y])

def download(name, url):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / name
    if not path.exists():
        req = urllib.request.Request(url, headers={'User-Agent': 'PolicyVotingResearch/1.0'})
        with urllib.request.urlopen(req, timeout=120) as response:
            content = response.read()
        path.write_bytes(content)
    return path

def prepare():
    CACHE.mkdir(exist_ok=True)
    adult = download('adult.zip', 'https://archive.ics.uci.edu/static/public/2/adult.zip')
    bank = download('bank.zip', 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip')
    with zipfile.ZipFile(adult) as z:
        cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        frames = []
        for name in ['adult.data', 'adult.test']:
            member = next((n for n in z.namelist() if n.endswith(name)))
            frames.append(pd.read_csv(BytesIO(z.read(member)), names=cols, skipinitialspace=True, comment='|'))
    work = ['Private', 'Federal-gov', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Self-emp-not-inc', 'Without-pay', 'Never-worked', '?']
    adult_schema = Schema('adult', ('age', 'hours.per.week', 'education.num', 'workclass', 'income'), {0: (17, 90, 1), 1: (1, 99, 1), 2: (1, 16, 1)}, {3: 1 - np.eye(9), 4: 1 - np.eye(2)}, {0: [25, 35, 45, 55, 65], 1: [20, 30, 40, 50, 60], 2: [5, 8, 10, 12, 14]})

    def adult_encode(f):
        return np.column_stack([np.clip((f.age - 17) / 73, 0, 1), np.clip((f['hours-per-week'] - 1) / 98, 0, 1), (f['education-num'] - 1) / 15, f.workclass.map({x: i for i, x in enumerate(work)}), f.income.str.contains('>').astype(int)]).astype(float)
    atrain, aval = train_test_split(adult_encode(frames[0]), test_size=0.2, random_state=20260907, stratify=frames[0].income.str.contains('>'))
    atest = adult_encode(frames[1])
    with zipfile.ZipFile(bank) as z:
        if 'bank-full.csv' in z.namelist():
            raw = z.read('bank-full.csv')
        else:
            inner = next((n for n in z.namelist() if n.endswith('bank.zip')))
            with zipfile.ZipFile(BytesIO(z.read(inner))) as zz:
                raw = zz.read('bank-full.csv')
    f = pd.read_csv(BytesIO(raw), sep=';')
    jobs = ['management', 'technician', 'admin.', 'blue-collar', 'services', 'housemaid', 'self-employed', 'entrepreneur', 'retired', 'student', 'unemployed', 'unknown']
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3])
    jm = np.where(groups[:, None] == groups[None, :], 0.5, 1.0)
    np.fill_diagonal(jm, 0)
    em = np.array([[0, 0.5, 1, 1], [0.5, 0, 0.5, 1], [1, 0.5, 0, 1], [1, 1, 1, 0]])
    bschema = Schema('bank', ('age', 'balance', 'education', 'job', 'y'), {0: (18, 100, 1), 1: (-10000, 110000, 1)}, {2: em, 3: jm, 4: 1 - np.eye(2)}, {0: [25, 35, 45, 55, 65], 1: [0, 100, 500, 1500, 5000, 20000]})
    b = np.column_stack([np.clip((f.age - 18) / 82, 0, 1), np.clip((f.balance + 10000) / 120000, 0, 1), f.education.map({'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 3}), f.job.map({x: i for i, x in enumerate(jobs)}), (f.y == 'yes').astype(int)])
    btrain, rest = train_test_split(b, test_size=0.4, random_state=20260907, stratify=b[:, -1])
    bval, btest = train_test_split(rest, test_size=0.5, random_state=20260907, stratify=rest[:, -1])
    for name, train, val, test in [('adult', atrain, aval, atest), ('bank', btrain, bval, btest)]:
        np.savez_compressed(CACHE / f'{name}.npz', train=train, val=val, test=test)
    meta = {'sources': {'adult': str(adult), 'bank': str(bank)}, 'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [adult, bank]}, 'splits': {'adult': [len(atrain), len(aval), len(atest)], 'bank': [len(btrain), len(bval), len(btest)]}, 'adult_missing_workclass': int((frames[0].workclass == '?').sum()), 'bank_balance_clipped': int(((f.balance < -10000) | (f.balance > 110000)).sum()), 'split_seed': 20260907}
    (CACHE / 'manifest.json').write_text(json.dumps(meta, indent=2))
    return meta

def schema_for(name):
    if name.startswith('xor'):
        return xor_schema(int(name[3:]))
    if name == 'adult':
        return Schema('adult', ('age', 'hours.per.week', 'education.num', 'workclass', 'income'), {0: (17, 90, 1), 1: (1, 99, 1), 2: (1, 16, 1)}, {3: 1 - np.eye(9), 4: 1 - np.eye(2)}, {0: [25, 35, 45, 55, 65], 1: [20, 30, 40, 50, 60], 2: [5, 8, 10, 12, 14]})
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3])
    jm = np.where(groups[:, None] == groups[None, :], 0.5, 1.0)
    np.fill_diagonal(jm, 0)
    return Schema('bank', ('age', 'balance', 'education', 'job', 'y'), {0: (18, 100, 1), 1: (-10000, 110000, 1)}, {2: np.array([[0, 0.5, 1, 1], [0.5, 0, 0.5, 1], [1, 0.5, 0, 1], [1, 1, 1, 0]]), 3: jm, 4: 1 - np.eye(2)}, {0: [25, 35, 45, 55, 65], 1: [0, 100, 500, 1500, 5000, 20000]})

def load(name, n=0, data_seed=0):
    s = schema_for(name)
    if name.startswith('xor'):
        k = int(name[3:])
        train = xor_data(k, n or 12000, 20260907 + data_seed)
        val = xor_data(k, 5000, 30260907 + data_seed)
        test = xor_data(k, 10000, 40260907 + data_seed)
    else:
        z = np.load(CACHE / f'{name}.npz')
        train, val, test = (z['train'], z['val'], z['test'])
        if n and n < len(train):
            rng = np.random.default_rng(20260907 + data_seed)
            train = train[rng.choice(len(train), n, replace=False)]
    return (s, train, val, test)
