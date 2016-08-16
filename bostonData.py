
# coding: utf-8

# # Boston Housing Prices Prediction

# The following code demonstrates the application of regression to predict the housing prices in Boston. For this the load_boston()
# is utillized from scikit-learn. 
# The user is free to choose from the following regression algorithms have been used to compare and analyze:
#     * Decision Tree Regression
#     * k-Nearest Neighbor Regression
#     * AdaBoost Regression

# ## All the required libraries

# In[1]:

import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt


# ## Function defitnions

# ### Performance Metric

# Defining the R^2 score as the performance measure. 

# In[2]:

def perfMetric(groundTruth,predicted):
    perfScore = r2_score(groundTruth,predicted)  # compute the R^2 score as a measure of error
    return perfScore


# ### Train and Test Set Generation

# In[3]:

def splitData(X,y):
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size=0.20,train_size=0.80,random_state=42)
    return Xtrain,Ytrain,Xtest,Ytest


# ### Model Complexity Curve Generation

# The model complexity depict the behavior of the model when the complexity of the model is varied. The training error and test error are recorded as the model parameter is progressivley increased.

# In[4]:

def modelComplexity(estimatorType,parameter,Xtrain,Ytrain,Xtest, Ytest):
    paramRange = np.arange(1,parameter)  # the model complexity parameters are varied from 2 to the specified parameter value
    
    # Note: pre-allocation as trainError = testError = np.zeros... does not work as the trainError and testError are always 
    #       held equal by that type of assignment
    trainError = np.zeros(len(range(0,len(paramRange)))) # pre-allocate array to record training error
    testError = np.zeros(len(range(0,len(paramRange))))  # pre-allocate array to record test error
    
    for i in range(len(paramRange)):
        # generate the appropriate estimator with the parameter in the loop
        if estimatorType == 'Decision Tree':
            estimator = DecisionTreeRegressor(max_depth = paramRange[i])
        elif estimatorType == 'kNN':
            estimator = KNeighborsRegressor(n_neighbors = paramRange[i])
        else:
            estimator = AdaBoostRegressor(n_estimators = paramRange[i])
        estimator.fit(Xtrain,Ytrain)
        trainError[i] = perfMetric(Ytrain,estimator.predict(Xtrain)) # record training error for the current parameter value
        testError[i] = perfMetric(Ytest,estimator.predict(Xtest))    # record test error for the current parameter value
    
    # plot the model complexity curves i.e the training error and test error
    plt.figure()
    plt.suptitle('Model complexity curves for %s' % (estimatorType))
    plt.plot(paramRange,testError,lw = 2,label = 'Test Error')
    plt.plot(paramRange,trainError,lw = 2, label = 'Train Error')
    plt.xlabel('Complexity parameter')
    plt.ylabel('R2 Score')
    plt.legend(loc = 'lower right')
    plt.show()


# ### Learning Curve Generation

# The learning curves depict the behavior of the model with varying training set sizes. The training error and test error are recorded as the training set size is prigressively increased.

# In[5]:

def learningCurves(estimatorType,parameter,Xtrain,Ytrain,Xtest,Ytest):
    
    bestParam = parameter      #the parameter given by the user on examination of the model complexity curves 
    if estimatorType == 'Decision Tree':
        estimator = DecisionTreeRegressor(max_depth = bestParam)
        numSamples = np.around(np.linspace(1,Xtrain.shape[0]))
    elif estimatorType == 'kNN':
        estimator = KNeighborsRegressor(n_neighbors = bestParam)
        numSamples = np.around(np.linspace(bestParam,Xtrain.shape[0]))   # the number of training samples is varied from the 
                                                                         # neightbors since scikit learn requires 
                                                                         # n_samples >= n_neighbors
    else:
        estimator = AdaBoostRegressor(n_estimators = bestParam)
        numSamples = np.around(np.linspace(1,Xtrain.shape[0]))    
    
    # numSamples is the array containing number of samples to pass in each iteration
    numSamples = numSamples.astype(int)   # numSamples is converted to int to avoid a Deprecation Warning
    trainError = np.zeros(len(numSamples)) 
    testError = np.zeros(len(numSamples))
    # the number of training samples is progressivley increased
    # the training and test error are recorded in each pass
    for i in range(len(numSamples)):
        estimator.fit(Xtrain[:numSamples[i]],Ytrain[:numSamples[i]])
        trainError[i] = perfMetric(Ytrain[:numSamples[i]],estimator.predict(Xtrain[:numSamples[i]])) 
        testError[i] = perfMetric(Ytest,estimator.predict(Xtest))
    
    # plot the learning curves 
    plt.suptitle('Learning curves for %s at model complexity parameter set to %d' % (estimatorType,bestParam))
    plt.plot(numSamples,testError,label = 'Test Error')
    plt.plot(numSamples,trainError,label = 'Train Error')
    plt.xlabel('Training set size')
    plt.ylabel('R2 Score')
    plt.legend(loc = 'lower right')
    plt.show()    


# ### Model fitting using Grid Search

# Based on the characteristics of the model studied with the above generated curves, a final grid-search with 10-fold cross validation is performed to find the best parameters and the model is fitted for prediction on the unknown test set.

# In[6]:

def fitModel(estimatorType,X,y):
    folds = KFold(X.shape[0],n_folds = 10,shuffle=True, random_state=40) #KFold object holds attirbutes for grid search
    # generate the parameter gris according to the type of the estimator
    if estimatorType == 'Decision Tree':
        model = DecisionTreeRegressor()
        parameters = {"max_depth":list(range(1,10))} # depth parameter for decision tree
    elif estimatorType == 'kNN':
        model = KNeighborsRegressor()
        parameters = {"n_neighbors":list(range(1,10))} # neighbors parameter for kNN
    else:
        model = AdaBoostRegressor()
        parameters = {"n_estimators":list(range(1,10))} # number of base learners for Adaboost
    
    scoreFun = make_scorer(perfMetric)   # make the defined performance metric into a valid scoring function
    paramGrid = GridSearchCV(model,parameters,scoreFun,cv=folds)  # perform grid search
    paramGrid = paramGrid.fit(X,y)
    return paramGrid.best_estimator_     #return the estimator with the best parameters


# # Main program

# ### Acquire data

# In[7]:

bostonData = load_boston()
priceData = bostonData.data
prices = bostonData.target
numHouses = prices.shape[0]
numFeatures = priceData.shape[1]


# ### Statistical data properties

# In[8]:

#data analysis
print("The number of houses is %d" % numHouses)
print("The number of features for each house is %d" % numFeatures)
print("The maximum price among houses is ${:,.2f}".format(np.max(prices)*10000))
print("The minimum price among houses is ${:,.2f}".format(np.min(prices)*10000))
print("The average price among houses is ${:,.2f}".format(np.mean(prices)*10000))
print("The standard deviation of prices among houses is ${:,.2f}".format(np.std(prices)*10000))


# ### Generate model complexity and learning curves

# In[9]:

Xtrain,Ytrain,Xtest,Ytest = splitData(priceData,prices)
estimatorType = 'AdaBoost'
modelComplexity(estimatorType,100,Xtrain,Ytrain,Xtest,Ytest)    # generate model complexity curves
learningCurves(estimatorType,10,Xtrain,Ytrain,Xtest,Ytest)      # generate learning curves


# ### Fit model and use on test set

# In[10]:

regressor = fitModel(estimatorType,priceData,prices)         # fit the best estimator
toPredict = np.array([11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13])  # test sample
toPredict = toPredict.reshape(1,-1)    # one sample is reshaped from (1,) to (1,n_features) to avoid warning
price = regressor.predict(toPredict)   # predict the test sample
print("Recommended selling price is ${:,.2f}".format(price[0]*10000))


# In[11]:

regressor.get_params() # estimated model parameters


# In[ ]:



