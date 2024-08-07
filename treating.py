import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def dataSpliting(pdSeries, window, N_splits):
    pdSeries = pdSeries
    window = window
    model_data = pd.DataFrame()
    model_data['t'] = [x for x in pdSeries]
    for i in range(1, window+1):
        model_data['t + ' + str(i)] = model_data['t'].shift(-i)

    model_data.dropna(axis=0, inplace=True)
    
    X = model_data.iloc[:,0:-1].values
    y = model_data.iloc[:, -1].values
    
    tscv = TimeSeriesSplit(n_splits=N_splits)
    
    for train_index, test_index in tscv.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    return X_train, y_train, X_test, y_test

def dataSplit(pdSeries, pdLabels, window, forecasting, N_splits):
    pdSeries = pdSeries
    pdLabels = pdLabels
    window = window
    model_data = pd.DataFrame()
    tscv = TimeSeriesSplit(n_splits=N_splits)

    model_data['x0'] = [x for x in pdSeries]
    for i in range(1, window+forecasting):
        model_data['x' + str(i)] = model_data['x0'].shift(-i)

    model_data.dropna(axis=0, inplace=True)

    list_names_window = []
    for n in range(0, window):
        list_names_window.append('x(t-' + str(n)+ ')')
    list_names_window.reverse()
    list_names_window[-1] = 'x(t)'

    list_names_forecasting = []
    for n in range(1, forecasting+1):
        list_names_forecasting.append('x(t+' + str(n)+ ')')

    list_columns = list_names_window + list_names_forecasting
    model_data.columns = list_columns

    X = model_data.iloc[:,0:window].values
    y = model_data.iloc[:,-forecasting:].values
    
    for train_index, test_index in tscv.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    model_data['label'] = pdLabels.iloc[(window-1):-(forecasting-1)].reset_index(drop=True)
    model_data['label'].fillna(0, inplace=True)

    df_train = pd.DataFrame(data = np.column_stack((np.column_stack((X_train, y_train)), model_data['label'].values[0:len(X_train)])), columns=model_data.columns)
    df_test = pd.DataFrame(data = np.column_stack((np.column_stack((X_test, y_test)), model_data['label'].values[len(X_train):])), columns=model_data.columns)
    
    return model_data, df_train, df_test, X_train, y_train, X_test, y_test    

def dataProcessing(pdSeries, window, forecasting):
    model_data = pd.DataFrame()

    model_data['x0'] = [x for x in pdSeries]
    for i in range(1, window+forecasting):
        model_data['x' + str(i)] = model_data['x0'].shift(-i)

    model_data.dropna(axis=0, inplace=True)

    list_names_window = []
    for n in range(0, window):
        list_names_window.append('x(t-' + str(n)+ ')')
    list_names_window.reverse()
    list_names_window[-1] = 'x(t)'

    list_names_forecasting = []
    for n in range(1, forecasting+1):
        list_names_forecasting.append('x(t+' + str(n)+ ')')

    list_columns = list_names_window + list_names_forecasting
    model_data.columns = list_columns

    X = model_data.iloc[:,0:window]
    y = model_data.iloc[:,-forecasting:]
    
    return model_data, X, y



def dataTreating(pdSeries, window):
    model_data = pd.DataFrame()

    model_data['t'] = [x for x in pdSeries]
    for i in range(1, window+1):
        model_data['t + ' + str(i)] = model_data['t'].shift(-i)

    model_data.dropna(axis=0, inplace=True)
    X = model_data.iloc[:,0:-1].values
    y = model_data.iloc[:, -1].values
    
    return model_data, X, y


