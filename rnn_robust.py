
def rnn_predict(days):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
    from plotly.offline import plot
    import plotly.graph_objs as go
    
    
    	# Importing Training Set
    dataset_train = pd.read_csv('stocks.csv')
    	 
    cols = list(dataset_train)[1:5]
    	 
    	#Preprocess data for training by removing all commas
    	 
    dataset_train = dataset_train[cols].astype(str)
    """for i in cols:
        for j in range(0,len(dataset_train)):
                dataset_train[i][j] = dataset_train[i][j].replace(",","")
    	 """
    dataset_train = dataset_train.astype(float)
    	 
    	 
    training_set = dataset_train.as_matrix() # Using multiple predictors.
    	 
    	# Feature Scaling
    from sklearn.preprocessing import StandardScaler
    	 
    sc = StandardScaler()
    training_set_scaled = sc.fit_transform(training_set)
    	 
    sc_predict = StandardScaler()
    	 
    sc_predict.fit_transform(training_set[:,0:1])
    	 
    	# Creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    ddays = int(days)
    n_future = ddays  # Number of days you want to predict into the future
    n_past = 60  # Number of past days you want to use to predict the future
    	 
    for i in range(n_past, len(training_set_scaled) - n_future + 1):
        X_train.append(training_set_scaled[i - n_past:i, 0:5])
        y_train.append(training_set_scaled[i+n_future-1:i + n_future, 0])
    	 
    X_train, y_train = np.array(X_train), np.array(y_train)
    	 
    	# Part 2 - Building the RNN
    	 
    	# Import Libraries and packages from Keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    	 
    	# Initializing the RNN
    regressor = Sequential()
    	 
    	# Adding first LSTM layer and Drop out Regularization
    regressor.add(LSTM(units=10, return_sequences=True, input_shape=(n_past, 4)))
    regressor.add(Dropout(0.2))
    
    	 
    	# Part 3 - Adding more layers
    	# Adding 2nd layer with some drop out regularization
    regressor.add(LSTM(units=4, return_sequences=False))
    regressor.add(Dropout(0.2))
    
    
    	# Output layer
    regressor.add(Dense(units=1, activation='linear'))
    	 
    	# Compiling the RNN
    regressor.compile(optimizer='adam', loss="mean_squared_error")  # Can change loss to mean-squared-error if you require.
    	 
    	# Fitting RNN to training set using Keras Callbacks. Read Keras callbacks docs for more info.
    	 
    es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    tb = TensorBoard('logs')
    	 
    history = regressor.fit(X_train, y_train, shuffle=True, epochs=100,
    	                        callbacks=[es, rlr,mcp, tb], validation_split=0.2, verbose=1, batch_size=64)
    	 
    	 
    	# creating graph & calculating profits & loss
    	 
    arr = []
    x = 1
    while x < ddays+1:
        arr.append(x)
        x = x + 1
    
    
    
    y = (sc_predict.inverse_transform(regressor.predict(X_train)))
    x = arr
    y = y[:ddays]
    y = np.array(y)
    y = y.ravel()
    y = np.round(y,2)
    
    plot([go.Scatter(x=x, y=y)],auto_open = False)
    
    end = y[len(y)-1]
    
    if y[0] > end:
        return "Loss : " + str(round(float(y[0]-end)/y[0]*100,2)) + "%"
    else:
        return "Profit : " + str(round(float(end-y[0])/y[0]*100,2)) + "%"
    # ((float(current)-previous)/previous)*100
