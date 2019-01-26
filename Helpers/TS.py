import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from tqdm import tqdm_notebook

from statsmodels.tsa.stattools import adfuller

from sklearn.model_selection import cross_val_score,TimeSeriesSplit

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error,\
    mean_squared_error, mean_squared_log_error



def mean_absolute_percentage_error(y_true, y_pred): 
    """
        INPUT: 
        y_true: Observed value of time series
        y_pred : predicted values
        OUTPUT:
        mean absolute percentage deviation from the true values
    """
    # Drop y_true == 0
    idx = y_true != 0
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred): 
    """
        INPUT: 
        y_true: Observed value of time series
        y_pred : predicted values
        OUTPUT:
        mean absolute percentage deviation from the true values
    """
    return np.mean(np.abs(y_true - y_pred))

def plotMovingAverage(series, window, plot_intervals=False,\
                      scale=1.96, plot_anomalies=False,\
                     figsize = [15,7]):

    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 

    """
    rolling_mean = series.rolling(window=window).mean()
    plt.figure(figsize=figsize)
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")
    sns.despine()

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
        
    plt.plot(series[window:], label="Actual values", alpha = 0.3)
    plt.legend(loc="upper left")
    
    
def plotMovingSDs(series, window,figsize = [15,7]):

    """
        INPUT:
        series - dataframe with timeseries
        window - rolling window size 
        OUTPUT:
        plots the standard deviation on moving windows
    """
    rolling_mean = series.rolling(window=window).mean()
    plt.figure(figsize=figsize)
    plt.title("Moving Standard Deviation \n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")
    plt.plot(series[window:], label="Actual values", alpha = 0.3)
    plt.legend(loc="upper left")
    sns.despine()

    
def plotProcess(df,figsize = [12,4]):
    """
        INPUT:
        df - data frame with timeseries
        OUTPUT:
        performs ADF test and plots the time series with results
    """
    df.plot(figsize = figsize)
    dftest = adfuller(df.values.squeeze(),autolag='AIC')
    p_val = dftest[1]
    s = "Dickey-Fuller p-value: {}".format(p_val,3) + '\n' + \
        'The Series is {} stationary'.format('' if dftest[1] < 0.05 else 'NOT' )
    plt.title("Dickey-Fuller p-value: {}".format(s))
    sns.despine()
    
def compareMAs(df, window_1 = 8, window_2 = 4):
    """
        INPUT:
        df - data frame with timeseries
        OUTPUT:
        computes the moving averages on two window sizes and plots them
    """
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=[12,8])
    df.plot(ax=ax,alpha = 0.2)
    label1 = 'Moving Average over {} Quarters'.format(window_1)
    label2 = 'Moving Average over {} Quarters'.format(window_2)
    df.rolling(window=8).mean().rename(columns = {'gdp':label1}).\
        plot(color = 'red',ax=ax)
    df.rolling(window=4).mean().rename(columns = {'gdp':label2}).\
        plot(color = 'green',ax=ax)
    sns.despine()
    plt.legend()
    
def tsplot(y, lags=None, figsize=(12, 7), style='seaborn-notebook'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        sns.despine()
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        sns.despine()
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        sns.despine()
        plt.tight_layout()

def optimizeSARIMA(df,parameters_list, d, D, s):
    """
        Return dataframe with parameters and corresponding AIC
        df -- time series
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """
    
    results = []
    best_aic = float("inf")

    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model=sm.tsa.statespace.SARIMAX(df, order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table 

def plotSARIMA(series, model, s, d, n_steps):
    """
        Plots model vs predicted values
        
        series - dataset with timeseries
        model - fitted SARIMA model
        s - seasona lag
        d - differencing order
        n_steps - number of steps to predict in the future
        
    """
    # adding model values
    data = series.copy()
    data.columns = ['actual']
    data['arima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['arima_model'][:s+d] = np.NaN
    
    # forecasting on n_steps forward 
    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps)
    forecast = data.arima_model.append(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    error = mean_absolute_error(data['actual'][s+d:], data['arima_model'][s+d:])

    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute  Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label="actual")
    plt.legend()
    sns.despine()
    
def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """
    
    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test


def plotModelResults(model, X_train,\
                     X_test, y_train, y_test, tscv, plot_intervals=False,plot_anomalies=False):
    """
        Plots modelled vs fact values, prediction intervals and anomalies
    
    """
    
    prediction = model.predict(X_test)
    
    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)
    
    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train, 
                                    cv=tscv, 
                                    scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()
        
        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)
        
        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
        
        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(y_test))
            anomalies[y_test<lower] = y_test[y_test<lower]
            anomalies[y_test>upper] = y_test[y_test>upper]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    error = mean_absolute_error(prediction, y_test)
    plt.title("Mean absolute error {0:.2f}%".format(error))
    plt.legend(loc="best")
    sns.despine()
    plt.tight_layout()
    
def plotCoefficients(model,X_train):
    """
        Plots sorted coefficient values of the model
    """
    
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar',color='lightblue')
    plt.grid(False, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');
    sns.despine()
    