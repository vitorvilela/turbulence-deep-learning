# Regression ML
# Authors: Vitor Vilela
# Created at: 21-09-17
# Last modification: 16-10-17
# Accumulated Work Hours: 24


# TODO 
#



# TODO Accomplished



# Load libraries
import numpy
from numpy import arange

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as pyplot

import pandas as pd
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

from sklearn.externals.joblib import dump 
from sklearn.externals.joblib import load


# Load dataset - PROJECT SPECIFICATION

filename = 'LEAIS.csv'
names = ['t', 'i', 'j', 'U', 'V', 'UV', 'TUU', 'TVV', 'TUV']
# Attributes are delimited by whitespace rather than commas
dataset0 = read_csv(filename,  names=names)
#dataset.insert(loc=5, column='UV', value=dataset['U']*dataset['V'])

MESH = 32
dataset0['i'] = dataset0['i']/MESH
dataset0['j'] = dataset0['j']/MESH

dataset0 = dataset0.loc[dataset0['TUU'] > 1e-10]
dataset0 = dataset0.loc[dataset0['TVV'] > 1e-10]
dataset0 = dataset0.loc[dataset0['TUV'] > 1e-10]


dataset0['TUU'] = numpy.log(dataset0['TUU'])
dataset0['TVV'] = numpy.log(dataset0['TVV'])
dataset0['TUV'] = numpy.log(dataset0['TUV'])


CRITICAL = 10
TRANSIENT = 30
#dataset0['t'] > TRANSIENT ? dataset0['t'] = t : dataset0['t'] = TRANSIENT




#dataset = dataset.drop_duplicates(['TUU', 'TVV', 'TUV'])
#dataset = dataset.loc[dataset['TUU'] >= 2.e-2]
#dataset = dataset.loc[dataset['TVV'] >= 5.e-6]
#dataset = dataset.loc[numpy.fabs(dataset['UV']) >= 1.e-4]

#dataset = dataset.loc[numpy.fabs(dataset['U']) >= 1.e-3]
#dataset = dataset.loc[numpy.fabs(dataset['V']) >= 1.e-3]
#dataset = dataset.loc[numpy.fabs(dataset['UV']) >= 1.e-4]

#dataset = dataset0.loc[:,['i','j','U', 'V', 'UV', 'UU', 'VV', 'UV']]

variable = 'TUU'
#varx = 'V'
# (1st) Ranges
#dataset = dataset0.loc[:,[variable]]
# (2nd) Where and When
#dataset = dataset0.loc[:,['t', 'i', 'j', variable]]
# (3rd) Correlation
#dataset = dataset0.loc[dataset0['t'] <= CRITICAL]
#dataset = pd.concat([ dataset.loc[dataset['UV'] < -7.e-5], dataset.loc[dataset['UV'] > 2.e-5] ])
#dataset = pd.concat([ dataset0.loc[dataset0['U'] < -1.e-5], dataset0.loc[dataset0['U'] > 1.e-5] ])
#dataset = dataset.loc[dataset['i'] > 0.9]
#dataset = dataset.loc[dataset['j'] > 0.9]
#dataset = dataset.loc[:,['U', 'V', 'UV', variable]]
dataset = dataset0.loc[:,['i', 'j', 'U', 'V', 'UV', variable]]
#dataset = dataset.loc[:,['i', 'j', 'U', 'V', 'UV', variable]]





#dataset = dataset.loc[dataset['TVV'] > 5.e-6]
#dataset = dataset.loc[dataset0['TUU'] > 0.001]


# Summarize Data

# Descriptive statistics

# Confirming the dimensions of the dataset.
print( '\nDataset shape\n', dataset.shape, '\n')
# Look at the data types of each attribute. Are all attributes numeric?
print( 'Dataset types\n', dataset.dtypes, '\n')
# Take a peek at the first 20 rows of the data. Look at the scales for the attributes.
print( 'Dataset head\n', dataset.head(20), '\n')
set_option('precision', 1)
# Summarize the distribution of each attribute (e.g. min, max, avg, std). How different are the attributes?
print( 'Dataset describe\n', dataset.describe(), '\n')
set_option('precision', 2)
# Have attributes a strong correlation (e.g. > 0.70 or < -0.70) among them? and with the output?
print( 'Dataset correlation\n', dataset.corr(method='pearson'), '\n')


# Data visualizations
# Think about: 
# - Feature selection and removing the most correlated attributes
# - Normalizing the dataset to reduce the effect of differing scales
# - Standardizing the dataset to reduce the effects of differing distributions

# Histograms
# Get a sense of the data distributions (e.g. exponential, bimodal, normal).
fig, ax = pyplot.subplots()
dataset.hist(ax=ax, sharex=False, sharey=False, xlabelsize=12, ylabelsize=12, bins=50)
#alpha=0.5
ax.set_yscale('log')
pyplot.show()

# Density
# Adds more evidence of the distribution. Skewed Gaussian distributions might be helpful later with transforms.
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, fontsize=1)
pyplot.show()

# Box and whisker plots
# Boxplots summarize the distribution of each attribute, drawing a line for the median and a box around the 25th and 75th percentiles.
# The whiskers give an idea of the spread of the data and dots outside of the whiskers show candidate outlier values.
# Outliers: values that are 1.5 times greater than the size of spread of the middle 50% of the data (i.e. 75th - 25th percentile).
dataset.plot(kind='box', subplots=True, layout=(7,1), sharex=False, sharey=False)
pyplot.show()

# Scatter plot matrix
# Visualization of the interaction between variables.
# Higher correlated attributes show good structure in their relationship, maybe not linear, but nice predictable curved relationships.
Axes = scatter_matrix(dataset, alpha=0.2, diagonal='kde')
[pyplot.setp(item.yaxis.get_majorticklabels(), 'size', 12) for item in Axes.ravel()]
[pyplot.setp(item.xaxis.get_majorticklabels(), 'size', 12) for item in Axes.ravel()]
[pyplot.setp(item.yaxis.get_label(), 'size', 12) for item in Axes.ravel()]
[pyplot.setp(item.xaxis.get_label(), 'size', 12) for item in Axes.ravel()]
pyplot.show()


## Scatter plot
#fig, ax = pyplot.subplots()
#dataset.plot(kind='scatter', x=varx, y=variable, fontsize=12, ax=ax)
#ax.set_yscale('log')
#pyplot.show()

# Correlation matrix
# The dark red color shows positive correlation whereas the dark blue color shows negative correlation.
# These suggest candidates for removal to better improve accuracy of models later on.
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()




# 1- Prepare Data

# Split dataset - PROJECT SPECIFICATION
# t, i, j, U, V, UV, TUU, TVV, TUV
# 0  1  2  3  4   5    6    7    8
# 'i', 'j', 'U', 'V', 'UV', 'UU', 'VV', 'UV'
#  0    1    2    3     4     5     6     7
# i, j, U, TUU
# 0  1  2   3  
array = dataset.values
X = array[:,0:5] 
Y = array[:,5]




#X0 = array[:,1:4] 
#Y0 = array[:,6]
#mask = (1.e-5 < numpy.fabs(Y0[:,]))
#X = X0[mask,:]
#Y = Y0[mask,]
#print('Y shape: ', Y.shape)
#print('Y head: ', Y[0:10,])

validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Evaluate Algorithms

# Test options and evaluation metric - PROJECT SPECIFICATION
# 10-fold cross-validation is a good standard test when dataset is not too small (e.g. 500)
num_folds = 10
seed = 7
# 'neg_mean_squared_error' MSE will give a gross idea of how wrong all predictions are (0 is perfect)
# others: 'r2', 'explained_variance'
scoring = 'r2'



## Spot Check Algorithms
## Create a baseline of performance on this problem.
#models = []
## Linear Algorithms. Coefficients do not depend on the independent variables.
#models.append(('LR', LinearRegression()))
#models.append(('LASSO', Lasso()))
#models.append(('EN', ElasticNet()))
## Nonlinear Algorithms.
#models.append(('KNN', KNeighborsRegressor()))
#models.append(('CART', DecisionTreeRegressor()))
#models.append(('SVR', SVR()))

## Evaluate each model in turn
## The algorithms all use default tuning parameters.
## Use for first glance, because the differing scales of the data is probably hurting the skill of all of the algorithms.
#results = []
#names = []
#kfold = KFold(n_splits=num_folds, random_state=seed)
#for name, model in models:
  #cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  #results.append(cv_results)
  #names.append(name)
  #msg = "%s: %e (%e)" % (name, numpy.sqrt(numpy.fabs(cv_results.mean())), numpy.sqrt(numpy.fabs(cv_results.std())))
  #print(msg)

## Compare Algorithms
## See which is more robust, tighter distribution of scores.
#fig = pyplot.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#pyplot.boxplot(results)
#ax.set_xticklabels(names)
#pyplot.show()


#Standardize the dataset

#Pipelines help you prevent data leakage in your test by ensuring that data preparation 
#like standardization is constrained to each fold of your cross-validation procedure.
#(e.g. the training dataset would have been influenced by the scale of the data in the test set). 
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
#pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
#pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
#pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))

results = []
names = []
kfold = KFold(n_splits=num_folds, random_state=seed)
for name, model in pipelines:  
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  #msg depending on the scoring type
  #msg = "%s: %e (%e)" % (name, numpy.sqrt(numpy.fabs(cv_results.mean())), numpy.sqrt(numpy.fabs(cv_results.std())))
  msg = "%s: %e (%e)" % (name, cv_results.mean(), cv_results.std())

  print(msg)

#Compare Algorithms
#See which is more robust, tighter distribution of scores.
#Observe changes against no prepared data.
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()




## KNN Algorithm tuning - PROJECT SPECIFICATION
## Up until now the models were evaluated with default parameters. 
## Use a grid search to try a set of different numbers for the model's parameter
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#k_values = numpy.array([1,3,5,7,9])
#param_grid = dict(n_neighbors=k_values)
#model = KNeighborsRegressor()
#kfold = KFold(n_splits=num_folds, random_state=seed)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
#grid_result = grid.fit(rescaledX, Y_train)

#print("\nBest: %e using %s" % (numpy.sqrt(numpy.fabs(grid_result.best_score_)), grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
    ##print("%e (%e) with: %r" % (numpy.sqrt(numpy.fabs(mean)), numpy.sqrt(numpy.fabs(stdev)), param))
    #print("%e (%e) with: %r" % (mean, stdev, param))



##How sensitive is k-NN classification accuracy to the train/test split proportion?
#t = [0.8, 0.7, 0.6]
#pyplot.figure()

#for s in t:
    #scores = []
    #for i in range(1, 10):
        #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1-s)
        #knnreg = KNeighborsRegressor(n_neighbors = 3).fit(X_train, y_train)
        #scores.append(knnreg.score(X_test, y_test))
    #pyplot.plot(s, numpy.mean(scores), 'bo')

#pyplot.xlabel('Training set proportion (%)')
#pyplot.ylabel('accuracy');
#pyplot.show()



# Ensembles
ensembles = []
# Boosting
# Building multiple models, typically of the same type, each of which learns to fix the prediction errors of a prior model in the sequence of models.
# The models make predictions which may be weighted by their demonstrated accuracy and the results are combined to create a final output prediction.
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
# Bagging 
# Building multiple models, typically of the same type, from different subsamples with replacement of the training dataset.
# The final output prediction is averaged across the predictions of all of the sub-models.
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))

results = []
names = []
for name, model in ensembles:
  kfold = KFold(n_splits=num_folds, random_state=seed)
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  #msg = "%s: %e (%e)" % (name, numpy.sqrt(numpy.fabs(cv_results.mean())), numpy.sqrt(numpy.fabs(cv_results.std())))
  msg = "%s: %e (%e)" % (name, cv_results.mean(), cv_results.std())

  print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()



## Make predictions on validation dataset



#Build the model
scaler = StandardScaler().fit(X_train)
#print('scaler.mean_ ', scaler.mean_[0])
#print('scaler.var_ ', scaler.var_[0])

rescaledX = scaler.transform(X_train)
#print('X_train\n', X_train[1:5,0])
#print('scaler rescaledX\n', rescaledX[1:5,0])
#manual_scaled = (X_train-scaler.mean_[0])/numpy.sqrt(scaler.var_[0])
#print('manual rescaledX\n', manual_scaled[1:5,0])
      
#Save Scaler
scaler_filename = variable + '_scaler.sav'
dump(scaler, scaler_filename)


#model = KNeighborsRegressor(n_neighbors = 3)
#model = LinearRegression()
#model = DecisionTreeRegressor()
#model = GradientBoostingRegressor()
model = ExtraTreesRegressor()
model.fit(rescaledX, Y_train)



# Transform the validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)

print('\n', 'min: ', numpy.amin(Y_validation), 'max: ', numpy.amax(Y_validation))
print('\n', '0.5 of data in range: ', numpy.percentile(Y_validation, 25), ' - ', numpy.percentile(Y_validation, 75))

rmse = numpy.sqrt(mean_squared_error(Y_validation, predictions))
print('\n', 'RMSE: ', numpy.sqrt(mean_squared_error(Y_validation, predictions)), ' Median percentage %: ', 100*rmse/numpy.percentile(Y_validation, 50))
print('\nModel score: ', model.score(rescaledValidationX, Y_validation) )


# Save the model to disk 
filename = variable + '_model.sav'
dump(model, filename)



##Load the model from disk
#loaded_model = load(filename) 

## Transform the validation dataset

## Predict
## U            V               U*V                UU                   VV                  UV
##0.01201016,-0.00221993,-0.0000266616602373,0.0002151054283767,0.0000067757030608,-0.0000372461719900

#rescaledValidationX = scaler.transform(X_validation)
#predictions = loaded_model.predict(rescaledValidationX)
#print('\n', 'RMSE: ', numpy.sqrt(mean_squared_error(Y_validation, predictions)))

#result = loaded_model.score(rescaledValidationX, Y_validation) 
#print('Accuracy: ', result)

#print('\n')

