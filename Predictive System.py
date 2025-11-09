import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
# loading the saved model
classifier = pickle.load(open('C:/Users/hp/Downloads/Deploying Machine Learning Model/diabetes_model.sav', 'rb'))
scaler = pickle.load(open('C:/Users/hp/Downloads/Deploying Machine Learning Model/scaler.sav', 'rb'))

input_data = (6,148,72,35,0,33.6,0.627,50)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')