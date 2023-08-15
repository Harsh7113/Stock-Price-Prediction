import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import datetime as dt
import yfinance as yf
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
plt.style.use("fivethirtyeight")

#get the stock quote
company = ['AAPL']  #,'AAPL','GOOG']
start = dt.datetime(2013,1,1)
end =  dt.datetime.today()  
df = yf.download(company, start , end)

#visualisation of the Closing price
plt.figure(figsize=(12,6))
plt.title("Close Price History")
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD($)", fontsize=18)
plt.plot(df["Close"])
plt.show()


data=df.filter(["Close"])
dataset=data.values

training_data_len=math.ceil(len(dataset)*.8)
print("\n")
print("Length of the training dataset:",training_data_len)
print("\n")

#scale the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
#print(scaled_data)

#training dataset
train_data=scaled_data[0:training_data_len,:]

x_train=[]
y_train=[]
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=61:
        #print(x_train)
        #print(y_train)
        print()
        
x_train, y_train = np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer="adam",loss="mean_squared_error")
model.fit(x_train,y_train,batch_size=1,epochs=1)

#create the testing dataset
test_data=scaled_data[training_data_len-60:,:]
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range (60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
    
#convert the data to a numpy array
x_test=np.array(x_test)

#reshape the data
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#get the models predicted price values
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)

#get the root mean squared error(RMSE) to check for any errors.
rmse=np.sqrt(np.mean((predictions- y_test)**2))
print(rmse)

train=data[:training_data_len]
valid=data[training_data_len:]
valid["Predictions"]=predictions

#plot the data
plt.figure(figsize=(16,8))
plt.title("Model", fontsize=22)
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD($)", fontsize=18)
plt.plot(train["Close"])
plt.plot(valid[["Close","Predictions"]])
plt.legend(["Train","Val","Predictions"],loc="lower right")
plt.show()

#show the valid and predicted price
print("This Data is the Actual Closing Price Vs The Predicted Price:\n")
print(valid)


apple_quote=yf.download(company, start , end)
#create a new datafreame
new_df = apple_quote.filter(['Close'])

#get the last 60 days closing price values and convert the dataframe to an array
last_60_days=new_df[-60:].values
#scale the data to be values between 0 and 1
last_60_days_scaled=scaler.transform(last_60_days)

#create an empty list
X_test=[]

#append the past 60 days
X_test.append(last_60_days_scaled)

#convert the X_test dataset to a numpy array
X_test=np.array(X_test)

#reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#get the predicted scaled price
pred_price=model.predict(X_test)

#undo the scaling
pred_price=scaler.inverse_transform(pred_price)
print("Predicted price:",pred_price)

#actual price
start_date = dt.datetime(2023,8,11)
end_date = dt.datetime.today()
apple_quote2 = yf.download(company, start_date, end_date)
print(apple_quote2["Close"])


#create a new list to store the predicted prices
future_predictions = []

#use the last 60 days' data from the original dataset
last_60_days = scaled_data[-60:]

for _ in range(15):
    #reshape the data for prediction
    x_test_future = np.array([last_60_days])
    x_test_future = np.reshape(x_test_future, (x_test_future.shape[0], x_test_future.shape[1], 1))
    
    #predict the next day's scaled price
    predicted_scaled_price = model.predict(x_test_future)
    
    #inverse transform the scaled price to get the actual price
    predicted_price = scaler.inverse_transform(predicted_scaled_price)
    
    #append the predicted price to the list
    future_predictions.append(predicted_price[0][0])
    
    #update the last_60_days list for the next iteration
    last_60_days = np.append(last_60_days[1:], predicted_scaled_price, axis=0)

#print the predicted prices for the next 15 days
print("Predicted prices for the next 15 days:")
for i, price in enumerate(future_predictions, start=1):
    print(f"Day {i}: {price}")
