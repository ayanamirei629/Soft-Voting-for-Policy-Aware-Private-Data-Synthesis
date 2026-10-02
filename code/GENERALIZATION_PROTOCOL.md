# Policy and training-size experiment protocol

All modeled columns remain in the distance and evaluation. Policies vary the
protected substitution graphs. Distances are normalized additive Gower distances.
Protection across several attributes uses the specified product policy; its
reach is calculated by `generalization_policy.py`.

The fixed configurations specify the complete pilot and confirmation cases,
privacy budgets, temperatures, output sizes, eight-round schedule, and seeds.
The reference output size is 512; candidate populations remain 512 for the first
three rounds and then contain 1,024 candidates under the expansion-two schedule.

Real-data splits are fixed. XOR size comparisons use nested training prefixes
within independent data seeds. Algorithm streams are paired across methods.
Public-selected temperatures are chosen from pilot data before confirmation.

Adult/Bank primary error is pairwise marginal L1 error. XOR primary error is
joint L1 error of the thresholded inputs and parity label. Relative reduction
is computed per paired run, then averaged. Intervals are pointwise Student-t
intervals; Holm correction uses the complete 680-primary-contrast family.
