# Privacy-budget comparison protocol

The paper crossover experiment uses the fixed per-dataset policy and normalized
distance definition, one Hard baseline, and Soft temperatures 0.015, 0.03, and
0.06. Each temperature remains fixed over the 17-budget grid from 1e-5 to 1.
There are twelve paired algorithm runs per cell.

Mechanism decomposition is a separate experiment with temperatures 0.045,
0.065, and 0.02 for Adult, Bank, and XOR3. Its three arms measure Hard error,
Soft error at Hard-level noise, and analytically calibrated Soft error.
Smoothing cost is C = E_M - E_H; noise benefit is N = E_M - E_S;
final gain is G = E_H - E_S = N - C. These terms are not constrained positive.

Fixed configurations in `experiments/configurations/` specify the complete
budgets, policies, temperatures, paired seeds, candidate/output schedule, and
calibration for reproduction. No temperature selection uses target errors.
