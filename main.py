import pandas as pd
import numpy as np
from collections import deque

from sklearn.preprocessing import StandardScaler, scale

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout, Dense, BatchNormalization

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score


coins = ['BTC', 'BCH', 'ETH', 'LTC', 'XRP']

data = pd.DataFrame()
for coin in coins:
    df = pd.read_csv(f'datasets/Bitstamp_{coin}USD_2021_minute.csv')
    df = df.drop(columns={'date', 'symbol', 'high', 'low', 'Volume USD'})
    df = df.rename(columns={f'Volume {coin}':f'volume_{coin}', 'close':f'close_{coin}', 'open':f'open_{coin}'})
    if len(data) == 0:
        data = df
    else:
        data = pd.merge(data, df, on='unix', how='left')
        
data = data.fillna(method='ffill')
data = data.dropna()
data['date'] = pd.to_datetime(data['unix'], unit='s')
data = data.sort_values(by='unix').reset_index(drop=True)

seq_len = 60 # how long of a preceeding sequence to collect for RNN
future_period_predict = 3 # how far into the future are we trying to predict?
ratio_to_predict = 'BTC-USD'
scaler = StandardScaler()
test_size = 0.3
name = f"{seq_len}-SEQ-{future_period_predict}-PRED"
coin_to_predict = 'BTC'

def calculate_future(data, coin_to_predict, future_period_predict):
    return data[f'close_{coin_to_predict}'].shift(-future_period_predict)

def calculate_target(current, future):
    return np.where(future > current, 1, 0)

def split_data(data, test_size, coin_to_predict):
    train = data[:round(len(data)*test_size)]
    test = data[round(len(data)*test_size):]
    
    train = train.loc[:, ['unix', 
                           f'open_{coin_to_predict}', 
                           f'close_{coin_to_predict}', 
                           f'volume_{coin_to_predict}',
                           'date',
                           'future',
                           'target']]
    
    test = test.loc[:, ['unix', 
                         f'open_{coin_to_predict}', 
                         f'close_{coin_to_predict}', 
                         f'volume_{coin_to_predict}',
                         'date',
                         'future',
                         'target']]
    
    
    return train, test

def preprocess(data):
    data = data.drop(columns={'unix', 'date', 'future'})
    
    for col in data.columns:
        if col != 'target':
            data[col] = scale(data[col].values)

    data = data.dropna()
    
    X = data.drop(columns={'target'})
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    y = data['target']
    y_array = []
    for i in y:
        y_array.append(i)

    return np.array(X), np.array(y_array)

data['future'] = calculate_future(data, coin_to_predict, future_period_predict)
data['target'] = list(map(calculate_target, data[f'close_{coin_to_predict}'], data['future']))
train_data, test_data = split_data(data, test_size, coin_to_predict)
X_train, y_train = preprocess(train_data)
X_test, y_test = preprocess(test_data)

classifier = Sequential()
classifier.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1:])))
classifier.add(Dropout(0.2))
classifier.add(LSTM(units=50, return_sequences=True))
classifier.add(Dropout(0.2))
classifier.add(LSTM(units=50))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()

classifier.fit(X_train, y_train, epochs=10, batch_size=32)
y_pred = classifier.predict(X_test)
y_pred_round = np.where(y_pred > 0.5, 1, 0)
ac=accuracy_score(y_test, y_pred.round())
print('accuracy of the model: ',ac)


