# Recurrent Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import MinMaxScaler
from yahoo_fin import stock_info as si
import datetime as dt
from plotly.offline import plot
import plotly.graph_objs as go


def initialize_variables():
    # Importing the training set
    dataset_train = pd.read_csv('stocks.csv')
    training_set = dataset_train.iloc[:, 1:2].values
    return training_set, dataset_train
    
def scale_dataset(training_set):
    # Feature Scaling
    sc = MinMaxScaler(feature_range = (0, 1))
    return sc.fit_transform(training_set), sc

def create_dataset(training_set_scaled, dataset_train):
    # Creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    for i in range(60, dataset_train.shape[0]): 
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train

def rnn_architecture(X_train, y_train):
    # Building the RNN
    
    # Initialising the RNN
    regressor = Sequential()
    
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    
    # Adding the output layer
    regressor.add(Dense(units = 1))
    
    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    #callback functions
    es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    tb = TensorBoard('logs')
    
    
    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = 50, batch_size = 64, callbacks=[es, rlr,mcp, tb],validation_split=0.2, verbose=1)
    return regressor


def retrieve_predictions(regressor, X_train, stock, days_p, sc):
    # Making the predictions and visualising the results
    # Getting the predicted stock price 
    predicted_stock_price = regressor.predict(X_train[-days_p:])
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    predicted_stock_price = np.array(predicted_stock_price)
    
    price = si.get_live_price(str(stock))
    if(abs(predicted_stock_price[0] - price)> abs(predicted_stock_price[predicted_stock_price.size-1] - price)):
        predicted_stock_price = np.flip(predicted_stock_price, axis = 0)
    return predicted_stock_price

def visualize_results(predicted_stock_price, ddays):
    # Visualising the results
    arr = []
    x = 1
    while x < ddays+1:
        arr.append(x)
        x = x + 1
        
    x = arr
    y = predicted_stock_price
    
    d1 = dt.date.today()
    d2 = d1 + dt.timedelta(days=ddays)
    dd = [d1 + dt.timedelta(days=x) for x in range((d2-d1).days + 1)]
    
    y = np.array(y)
    y = y.ravel()
    y = np.round(y,2)
    
    plot([go.Scatter(x=dd, y=y)],auto_open = True)  
    end = y[len(y)-1]
    return y, end


def rnn_predict(days_p, stock):
    days_p = int(days_p)
    stock = str(stock)
    
    training_set, dataset_train = initialize_variables()
    training_set, sc = scale_dataset(training_set)
    X_train, y_train = create_dataset(training_set, dataset_train)
    regressor = rnn_architecture(X_train, y_train)
    predicted_stock_price= retrieve_predictions(regressor, X_train, stock, days_p, sc)
    y, end = visualize_results(predicted_stock_price, days_p)
    
    if y[0] > end:
        return ( "Loss : " + str(round(float(y[0]-end)/y[0]*100,2)) + "%")
    else:
        return ( "Profit : " + str(round(float(end-y[0])/y[0]*100,2)) + "%")
        # ((float(current)-previous)/previous)*100

