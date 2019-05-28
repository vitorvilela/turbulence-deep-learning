# Multilayer Perceptron model for a dataset

# Authors: Vitor Vilela,
# Created at: 28-09-17
# Last modification: 04-12-17


# Import section
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from pandas import read_csv
from pandas import set_option
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from keras.models import model_from_json
from sklearn.externals.joblib import dump 
from sklearn.externals.joblib import load
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout


## Load dataset with its respectively outputs
filename = 'LEAIS.csv'
names = ['t', 'i', 'j', 'U', 'V', 'UV', 'TUU', 'TVV', 'TUV']
dataset0 = read_csv(filename,  names=names)

MESH = 32
FINALTIME = 60

dataset0['i'] = dataset0['i']/MESH
dataset0['j'] = dataset0['j']/MESH

dataset0['t'] = dataset0['t']/FINALTIME

dataset0 = dataset0.loc[dataset0['TUU'] > 1e-10]
dataset0 = dataset0.loc[dataset0['TVV'] > 1e-10]
dataset0 = dataset0.loc[dataset0['TUV'] > 1e-10]

dataset0['TUU'] = numpy.log10(dataset0['TUU'])
dataset0['TVV'] = numpy.log10(dataset0['TVV'])
dataset0['TUV'] = numpy.log10(dataset0['TUV'])

dataset = dataset0.loc[:,['t', 'i', 'j', 'U', 'V', 'UV', 'TUU', 'TVV', 'TUV']]


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



## Data visualizations
## Think about: 
## - Feature selection and removing the most correlated attributes
## - Normalizing the dataset to reduce the effect of differing scales
## - Standardizing the dataset to reduce the effects of differing distributions

## Histograms
## Get a sense of the data distributions (e.g. exponential, bimodal, normal).
#dataset.hist(sharex=False, sharey=False, xlabelsize=12, ylabelsize=12, bins=30)
#plt.show()

## Density
## Adds more evidence of the distribution. Skewed Gaussian distributions might be helpful later with transforms.
#dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, fontsize=1)
#plt.show()

## Box and whisker plots
## Boxplots summarize the distribution of each attribute, drawing a line for the median and a box around the 25th and 75th percentiles.
## The whiskers give an idea of the spread of the data and dots outside of the whiskers show candidate outlier values.
## Outliers: values that are 1.5 times greater than the size of spread of the middle 50% of the data (i.e. 75th - 25th percentile).
#dataset.plot(kind='box', subplots=True, layout=(7,1), sharex=False, sharey=False)
#plt.show()

## Scatter plot matrix
## Visualization of the interaction between variables.
## Higher correlated attributes show good structure in their relationship, maybe not linear, but nice predictable curved relationships.
#scatter_matrix(dataset)
#plt.show()

## Correlation matrix
## The dark red color shows positive correlation whereas the dark blue color shows negative correlation.
## These suggest candidates for removal to better improve accuracy of models later on.
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
#fig.colorbar(cax)
#ticks = arange(0,14,1)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
#plt.show()


# 1- Prepare Data

## Split into input (X) and output (Y) variables
# i, j, U, V, UV, TUU, TVV, TUV
# 0  1  2  3  4   5    6    7
# t, i, j, U, V, UV, TUU, TVV, TUV
# 0  1  2  3  4   5    6    7  8
array = dataset.values
X = array[:,0:6] 
Y = array[:,6:9]
#Y = array[:,7]

variable = 'ALL'
ninputs = 6
noutputs = 3




# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

validation_split = 0.3 #0.33 #0.5
batch_size = 2000
epochs = 100
kernel_initializer = 'normal' #'uniform'
loss = 'mean_squared_error' #'binary_crossentropy' #'mean_squared_error'
optimizer = 'adam'
activation = 'relu' #sigmoid #tanh

## Define the model
#def larger_model():
	## Create model
	#model = Sequential()
	#model.add(Dense(ninputs, input_dim=ninputs, kernel_initializer=kernel_initializer, activation=activation))
	##model.add(Dense(800, kernel_initializer=kernel_initializer, activation=activation))
	#model.add(Dense(1200, kernel_initializer=kernel_initializer, activation=activation))
	##model.add(Dense(800, kernel_initializer=kernel_initializer, activation=activation))
	#model.add(Dense(600, kernel_initializer=kernel_initializer, activation=activation))
	#model.add(Dense(noutputs, kernel_initializer=kernel_initializer))
	## Compile model
	#model.compile(loss=loss, optimizer=optimizer)
	#return model
     
          
## Evaluate model with standardized dataset
#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=epochs, batch_size=batch_size, verbose=0)))
#pipeline = Pipeline(estimators)
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(pipeline, X, Y, cv=kfold)

#print("Model: %.3e (%.3e) AVG (STD) of RMSE" % (numpy.sqrt(results.mean()), numpy.sqrt(results.std())))
#print('\n', 'min: ', numpy.amin(Y), 'max: ', numpy.amax(Y))
#print('\n', '50 percent of data is in range: ', numpy.percentile(Y, 25), ' - ', numpy.percentile(Y, 75))
#rmse = numpy.sqrt(results.mean())
#print('\n', 'RMSE: ', rmse, ' Median percentage %: ', 100*rmse/numpy.percentile(Y, 50))




# Scaler
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

#Save Scaler
filename_scaler = variable + '_scaler.sav'
dump(scaler, filename_scaler)



# Create model
model = Sequential()
model.add(Dense(ninputs, input_dim=ninputs, kernel_initializer=kernel_initializer, activation=activation)) #model.add(Dropout(0.2, input_shape=(ninputs,)))
model.add(Dense(600, kernel_initializer=kernel_initializer, activation=activation))
model.add(Dense(1600, kernel_initializer=kernel_initializer, activation=activation))
model.add(Dropout(0.2))
model.add(Dense(600, kernel_initializer=kernel_initializer, activation=activation))
model.add(Dense(noutputs, kernel_initializer=kernel_initializer))
# Compile model
model.compile(loss=loss, optimizer=optimizer)

# Serialize model to JSON 
model_json = model.to_json() 
filename_json = variable + '_model.json'
with open(filename_json, 'w') as json_file:
  json_file.write(model_json)

## Checkpoint
#filepath = 'model-{epoch:02d}-{loss:.4f}.h5'
#checkpoint = ModelCheckpoint(filepath, monitor=loss, verbose=1, save_best_only=True, mode='min') 
#callbacks_list = [checkpoint]

# Fit model
history = model.fit(rescaledX, Y, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=0) #callbacks=callbacks_list


# Serialize weights to HDF5 
filename_h5 = variable + '_model.h5'
model.save_weights(filename_h5) 
print('Saved model to disk')


# Print statistics
print('min: ', numpy.amin(Y), 'max: ', numpy.amax(Y))
print('RMSE: ', numpy.sqrt(history.history['val_loss'][-1]), ' percentage @median: ', 100*rmse/numpy.percentile(Y, 50))


# List all data in history 
#print(history.history.keys())
# Summarize history for accuracy 
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss')
plt.ylabel(loss) 
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right') 
plt.show()






## Using the loaded scaler and model

## Load scaler
#loaded_scaler = load(filename_scaler)
#rescaledValidationX = loaded_scaler.transform(X)

## Load json and create model 
#json_file = open(filename_json, 'r')
#loaded_model_json = json_file.read() 
#json_file.close() 
#loaded_model = model_from_json(loaded_model_json) 

## Evaluate loaded model on test data 
#loaded_model.compile(loss='mean_squared_error', optimizer='adam')

## Load weights into new model 
#loaded_model.load_weights(filename_h5) 
#print('Loaded model from disk')



## Predict
#prediction = loaded_model.predict(rescaledValidationX)

## Print
#print('predict_x')
#print(predict_x)
#print('true_y')
#print(true_y)
#print('prediction')
#print(prediction)
#print('%')
#print(100*(prediction-true_y)/true_y)