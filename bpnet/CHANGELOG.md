Changes since Surag started cleanup.

1 March
- removed unused imports
- removed code corresponding to ChromBPNet and attribution priors.
- Removed CustomMeanSquaredError and MultichannelMultinomialNLL
- Removed api directory
- changed losses.py to remove duplication, moved functions from custommodel.py
- changed CLI options and added "bpnet-" prefix
- simplified organization, renamed directories to understandable names (e.g. mseqgen -> generators, genomicsarchsandlosses -> model)
- removed repetition of counts loss function (two functions for mse and poisson loss calculation with identical logic) in custommodel.py