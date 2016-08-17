# boston-housing-price-prediction-using-regression
This repository is aimed at predicting the housing prices in Boston by generating the optimal regression model.

THe question source has been taken from the following link:
https://docs.google.com/document/d/1_K2pjLJ15c4kRcyNdQ6kBvOjc-pcdBfP6v88JYtt17Y/pub?embedded=true

The objective is to develop an optimal model that can predict the housing prices in Boston given the training data. For this work, the follwing software and associated packages were used:
- Python 3.5
- Jupyter Notebook
- Numpy
- Scikit-Learn
- Matplotlib

### Loading and exploring the data

The data is loaded through the `load_boston()` function of scikit learn. 
```
bostonData = load_boston()
```

The various statistical measures of the data are obtained using the relevant functions in numpy. 
- `The number of houses is 506.`
- `The number of features for each house is 13.`
- `The maximum price among houses is $500,000.00.`
- `The minimum price among houses is $50,000.00.`
- `The average price among houses is $225,328.06.`
- `The standard deviation of prices among houses is $91,880.12.`

### The model complexity curves

The model complexity curves are the curves that are genrated when the training and test error are recorded as the complexity of the model is varied. The following curves depict the generated curves. The parameter is varied from 2 to the limit specified by the user. This is because decision trees require a minimum depth of 2. 

<p align="center">
  <img src="https://github.com/vishnu1729/boston-housing-price-prediction-using-regression/blob/master/decistiontreemodelcomplexity.png" alt="Figure 1. Model Compplexity for Decision Tree Regressor" width="350"/>
</p>


