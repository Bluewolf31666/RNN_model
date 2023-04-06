import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.layers import Dense, Dropout, Flatten,LSTM

class model:
    def LSTM():
        regressor = Sequential()
        regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units = 60, return_sequences = True))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units = 60))
        regressor.add(Dropout(0.1))
        regressor.add(Dense(units = 1))
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        regressor.compile(optimizer='SGD', loss='mean_squared_error')
        return(regressor)