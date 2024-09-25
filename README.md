# Nested cross-validation and Grid Search using Pytorch
This repository is used to perform nested cross validation using Pytorch.
Let's understand how it works!

The **cross-validation** technique is increasingly used in the world of artificial intelligence. This technique gives more meaningful results around the performance of a model, as it uses different test sets to evaluate it. Instead of using a single test set, we split our data into **outer folds**, so that we evaluate the model 5 times, being able to give statistical descriptors on the results. 

![imagen](https://github.com/user-attachments/assets/87d2aca5-0562-4bb1-b598-f2142fd78b95)

However, a more robust way to provide meaningful results is the **nested cross-validation** technique. To find the best hyperparameters (via grid search), we split each train set inside the outer loop into different **inner folds**. 

![imagen](https://github.com/user-attachments/assets/4d976299-d070-495b-b69c-f962a5507b1c)


In this case, we perform nested cross validation on a single hidden layer MLP.
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
