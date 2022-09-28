#!/usr/bin/env python
# coding: utf-8

# In[234]:


###Imports

import pandas as pd
import numpy as np
import tensorflow as tf
import warnings
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from tensorflow import keras
from zipfile import ZipFile
from urllib.request import urlopen
from urllib.error import HTTPError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error
from sklearn.exceptions import DataConversionWarning

### Turning Off Warnings

# to turn off DataConversion warning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# to turn off keras warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# to turn off the chained assignment warning
pd.options.mode.chained_assignment = None

# Let's fix the random number generator's seed to get reproducible results
tf.random.set_seed(20)
np.random.seed(20)


# In[235]:


def get_dataset():        
    
    """Checks existance of dataset if it exists at given URL

    Returns: 
        *Data frame if dataset exists or else prints error message
    """
    try:
        url="https://github.com/askhan131/SciProject/blob/main/dataset.zip?raw=true"
        url = urlopen(url)
        
        # download dataset.zip from URL
        data_files=open('dataset.zip','wb')
        data_files.write(url.read())
        data_files.close()
        
        # Let's store the hour.csv file in dataframe and return it
        with ZipFile("dataset.zip", "r") as f:
            if "hour.csv" in f.namelist():
                zip_file_object = ZipFile('dataset.zip') 
                # making date_id as index column
                hour_dataframe = pd.read_csv(zip_file_object.open('hour.csv'),parse_dates=['date_id'],index_col="date_id")
                return hour_dataframe
            else:
                print("Sorry, there is no file named hour_data.csv in zip file. Please recheck file name")
    except HTTPError:
        print("Sorry, could not connect or find the file at url", url)


# In[236]:


# Let's see data variable below, what hour.csv file contains as dataFrame
data = get_dataset()


# In[237]:


print("Here we have", data.shape[0], "rows and", data.shape[1], "columns in our data file")


# In[238]:


data.head()


# In[239]:


### To get some mathematical statistics from our dataset
data.describe()


# In[240]:


sns.set(font_scale=0.8)
fig, axes = plt.subplots(nrows=3,ncols=2)
fig.set_size_inches(15, 15)

sns.boxplot(data=data,y="cnt",orient="v",ax=axes[0][0])
sns.boxplot(data=data,y="cnt",x="season",orient="v",palette="Blues",ax=axes[0][1])
sns.boxplot(data=data,y="cnt",x="mnth",orient="v",ax=axes[1][0])
sns.boxplot(data=data,y="cnt",x="workingday",orient="v",ax=axes[1][1])
sns.boxplot(data=data,y="cnt",x="hr",orient="v",ax=axes[2][0])
ax = sns.boxplot(data=data,y="cnt",x="ori_temp",orient="v",ax=axes[2][1])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
axes[0][0].set(ylabel='Count',title="Box Plot On RideCount")
axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Seasons")
axes[1][0].set(xlabel='Month', ylabel='Count',title="Box Plot On Count Across Months")
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")
axes[2][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")
axes[2][1].set(xlabel='Temperature', ylabel='Count',title="Box Plot On Count Across Temperature")


# In[241]:


data=data.drop(columns=['serial_id','a_temp','weekday','reg_member','casual_member'])


# In[242]:


# To remove the NULL values from the processed data
data=data.dropna()


# In[243]:


data.head()


# In[244]:


### Let's split the data for training and test set

training_set_size = int(len(data)*0.9)
test_set_size = len(data)-training_set_size
training_set,test_set = data.iloc[0:training_set_size],data.iloc[training_set_size:len(data)]
print("Length of training set:", len(training_set))    
print("Length of test set:",len(test_set))


# In[245]:


columns_to_scale = ['ori_temp','hum','wind_spd']
c_scale_transformer = StandardScaler().fit(training_set[columns_to_scale].to_numpy())
cnt_transformer = StandardScaler().fit(training_set[['cnt']])

training_set.loc[:,columns_to_scale] = c_scale_transformer.transform(training_set[columns_to_scale].to_numpy())
training_set['cnt'] = cnt_transformer.transform(training_set[['cnt']])

test_set.loc[:,columns_to_scale] = c_scale_transformer.transform(test_set[columns_to_scale].to_numpy())
test_set['cnt'] = cnt_transformer.transform(test_set[['cnt']])


# In[246]:


def create_data_sequence(X, y, time_steps=1):
    """ Create data sequence
    
    Arguments:
        * X: time-series data
        * y: Count "cnt" value
        * time_steps: Used to create input sequence of timesteps
    
    Returns:
        * input_sequence: Numpy array of sequences of time-series data
        * output: Numpy array of output i.e. next value for respective sequence
    
    """
    input_sequence, output = [], []
    for i in range(len(X) - time_steps):
        sequence = X.iloc[i:(i + time_steps)].values
        input_sequence.append(sequence)        
        output.append(y.iloc[i + time_steps])
    return np.array(input_sequence), np.array(output)


# In[272]:


time_steps = 8

# training_set_sequence, test_set_sequence are input features for data set, as numpy arrays. 
training_set_sequence, training_set_output = create_data_sequence(training_set, training_set.cnt, time_steps)

# training_set_output and test_set_output are "cnt" values for data set sequences, as numpy arrays.
test_set_sequence, test_set_output = create_data_sequence(test_set, test_set.cnt, time_steps)

# get training and test set sequences as [samples, time_steps, n_features]

print("Training data shape", training_set_sequence.shape, "Training data output shape", training_set_output.shape)
print("Test data shape", test_set_sequence.shape, "Test data output shape", test_set_output.shape)


# In[294]:


def machine_learning_model():
    """Defines machine learning model
    
    Returns:
        * model: LSTM model
    
    """
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=128,input_shape=(training_set_sequence.shape[1], training_set_sequence.shape[2])))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    model.summary()
    return model


# In[310]:


model = machine_learning_model()

history = model.fit(
    training_set_sequence,
    training_set_output, 
    epochs=10, 
    batch_size=64, 
    validation_split=0.1,
    shuffle=False,
)


# In[311]:


fig,ax = plt.subplots()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='test loss')
ax.set_xlabel('EPOCHS')
ax.set_ylabel('Loss value')
plt.legend();


# In[312]:


predict_test_set = model.predict(test_set_sequence)


# In[313]:


model_predictions = cnt_transformer.inverse_transform(predict_test_set)
actual_testset_values = cnt_transformer.inverse_transform(test_set_output)


# In[314]:


fig,ax = plt.subplots()
plt.plot(model_predictions[:100,], label='Predicted ride count')
plt.plot(actual_testset_values[:100,], label='Actual ride count')
ax.set_xlabel('Hours')
ax.set_ylabel('Count')
plt.legend();
plt.show()


# In[315]:


def get_mean_absolute_deviations(predictions,actual_values):
    """ Compute the mean absolute deviations of predictions vs actual test set values
        
        Arguments:
        * predictions: Our Model's predictions
        * actual_values: Test set output
    
    """
    
    # Convert numpy arrays to data frame as pandas as mean absolute deviation function we want to use
    predictions_dataframe = pd.DataFrame(data = predictions.flatten())
    actual_test_set_values_dataframe = pd.DataFrame(data=actual_values.flatten())
        
    print("LSTM model prediction's Mean Absolute Deviation:", predictions_dataframe.mad()[0])
    print("Test set's Mean Absolute Deviation:", actual_test_set_values_dataframe.mad()[0])


# In[316]:


get_mean_absolute_deviations(model_predictions,actual_testset_values)


# In[317]:


def get_prediction(input_sequence):
    """ Gets prediction of bike share count based on input sequence
    
    Arguments:
    * input_sequence: <time_steps> hours of sequence data
    
    Returns:
    * cnt_prediction: Predicted count value of bike share
    """
    
    prediction = model.predict(input_sequence)
    cnt_prediction = cnt_transformer.inverse_transform(prediction)
    return cnt_prediction


# In[318]:


r_data = data.iloc[-8:]
r_data = np.expand_dims(r_data, axis=0)

print("Predicted Bike Share Count for next hour based on last", time_steps,"hours of data is", int(get_prediction(r_data)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




