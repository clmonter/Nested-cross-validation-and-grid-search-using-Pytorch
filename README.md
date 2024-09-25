# Nested cross-validation and Grid Search using Pytorch
This repository is used to perform nested cross validation on a single hidden layer MLP using Pytorch. 

The already implemented parameters adjustable with grid search are:
- batch size
- dimensions of the hidden layer
- number of epochs
- learning rate
- L2 Ridge regularization parameter
- Way to apply weights in the loss function, for unbalanced datasets. These options are:
  - non_weighted (no weight is applied).
  - weighted. The inverse of the normalized occurrence of each class.
  - sqrt_weighted. The weighted version, using square root in the denominator (is a softer function than the weighted version).
  - log_weighted. The logaritmic of the weighted version (is a softer function than the weighted version).

The code was initially implemented to perform a comparison between different sets of audio features for acoustic scene classification on a given dataset.
However, the code is adaptable for any different type of problem. It is also adaptable to train different models!
