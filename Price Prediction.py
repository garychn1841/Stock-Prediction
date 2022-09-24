import math
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#read data and sort the column
df = pd.read_csv('F:/Shioaji/Ticks/2022-05-03/2409.csv')
df2 = df[['ts','close','volume','ask_price','ask_volume','bid_price','bid_volume']]

#plot the close price
# df2.close.rolling(120).mean().plot(label='$MA_{120}$',legend=True)
# df2.close.rolling(240).mean().plot(label='$MA_{240}$',legend=True)
# df2.close.rolling(360).mean().plot(label='$MA_{360}$',legend=True)
# plt.show()
# plt.figure(figsize=(16,8))
# plt.plot(df2.ts[:100],df2.close[:100])
# plt.show()

# create a new df for clsoe price
data = df2.filter(['close'])
dataset = data.values
#print(dataset)

#select the training len
training_data_len = math.ceil(len(dataset)*0.7)

#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

#create the scaled traning data set
train_data = scaled_data[0:training_data_len,:]
#print(train_data)

#Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    # if i<=61:
        # print(x_train)
        # print(y_train)
        # print()

#Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
# print(x_train.shape)
# print(y_train.shape)
# print(x_train[0])

#Reshape the data
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
print(x_train.shape)

# #Build the LSTM model
# model = Sequential()
# model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
# model.add(LSTM(50,return_sequences=False))
# model.add(Dense(25))
# model.add(Dense(1))
#
# #compile the model
# model.compile(optimizer='adam',loss='mean_squared_error')
#
# #Train the model
# model.fit(x_train, y_train, batch_size=1, epochs=1)
#
#
# #Creat the test data set
# #creat a new array containing scaled values from index 3793 to end
# test_data = scaled_data[training_data_len-60: , :]
# #creat the data sets x_test and y_test
# x_test = []
# y_test = dataset[training_data_len:, :]
# for i in range(60, len(test_data)):
#     x_test.append(test_data[i-60:i,0])
#
# #Convert the data to a numpy array
# x_test = np.array(x_test)
#
# #Reshape the data
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#
# #Get the model predicted price values
# predictions = model.predict(x_test)
# predictions = scaler.inverse_transform(predictions)
#
# #Get the root mean squared error(RMSE)
# rmse = np.sqrt(np.mean(predictions - y_test)**2)
# print(rmse)
#
# #Plot the data
# train = data[:training_data_len]
# vaild = data[training_data_len:]
# vaild['Predictions'] = predictions
#
# #Visualize the data
# plt.figure(figsize=(16,8))
# plt.title('Model')
# plt.xlabel('Time',fontsize=18)
# plt.ylabel('Close Price', fontsize=18)
# plt.plot(train['close'])
# plt.plot(vaild[['close', 'Predictions']])
# plt.legend(['Train','Val','Predictions'],loc='lower right')
# plt.show()
#
# #Show the valid and prediction prices
# print(vaild)


