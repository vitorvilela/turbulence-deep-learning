# Multilayer Perceptron model for a dataset

# Authors: Vitor Vilela,
# Created at: 28-09-17
# Last modification: 30-09-17
# MH: (Vitor Vilela), (6)


# TODO 
# Type - Description - Responsible - Estimated Work Hours - Priority Sequence


# Import section
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from keras.models import model_from_json


# Load (YOUR) dataset with its respectively outputs
dataframe = read_csv("LEAIS.csv", header=None)

# t, U, V, U*V, UU, VV, UV
dataset = dataframe.values

# Split into input (X) and output (Y) variables
X = dataset[:,0:4]
Y = dataset[:,4:7]

# Test on a single prediction sample X = [U V U*V], Y = [UU, VV, UV]
predict_x = numpy.array([[0.01609, -0.00141, -0.00002264]])
true_y = numpy.array([[0.00034395, 0.00000269, -0.00002686]])

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)



## Define the model
#def larger_model():
	## Create model
	#model = Sequential()
	#model.add(Dense(3, input_dim=3, kernel_initializer='normal', activation='relu'))
	#model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	#model.add(Dense(3, kernel_initializer='normal'))
	## Compile model
	#model.compile(loss='mean_squared_error', optimizer='adam')
	#return model
     
# Evaluate model with standardized dataset
#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=100, verbose=0)))
#pipeline = Pipeline(estimators)
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(estimator, X, Y, cv=kfold)
#print("Model: %.2f (%.2f) AVG (VAR) of MSE" % (results.mean(), results.std()))

#estimator = KerasRegressor(build_fn=larger_model, epochs=50, batch_size=100, verbose=0)

#prediction = estimator.predict(predict_x)



# Create model
model = Sequential()
model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
model.add(Dense(12, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal', activation='relu'))
model.add(Dense(3, kernel_initializer='normal'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# Validation
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Model: %.2f (%.2f) AVG (STD) of RMSE" % (numpy.sqrt(results.mean()), numpy.sqrt(results.std())))

# Fit model
model.fit(X, Y, epochs=50, batch_size=1000, verbose=0)

# Predict
prediction = model.predict(predict_x)

# Print
print('predict_x')
print(predict_x)
print('true_y')
print(true_y)
print('prediction')
print(prediction)
print('%')
print(100*(prediction-true_y)/true_y)


# Serialize model to JSON 
model_json = model.to_json() 
with open('model.json', 'w') as json_file:
  json_file.write(model_json)

# Serialize weights to HDF5 
model.save_weights('model.h5') 
print('Saved model to disk')


# Load json and create model 
json_file = open('model.json', 'r')
loaded_model_json = json_file.read() 
json_file.close() 
loaded_model = model_from_json(loaded_model_json) 

# Evaluate loaded model on test data 
loaded_model.compile(loss='mean_squared_error', optimizer='adam')

# Load weights into new model 
loaded_model.load_weights('model.h5') 
print('Loaded model from disk')



# Predict
prediction = loaded_model.predict(predict_x)

# Print
print('predict_x')
print(predict_x)
print('true_y')
print(true_y)
print('prediction')
print(prediction)
print('%')
print(100*(prediction-true_y)/true_y)