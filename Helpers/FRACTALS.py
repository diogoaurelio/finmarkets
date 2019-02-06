
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def fractal(prices, transformer_fn = np.log):
    """
        INPUT:
            prices -- observed prices of a tradeable asset
            transformer_fn -- transforms the input series before analysing
        OUTPUT:
            fractal dimension
        INTUITION OF FRACTAL:
          If the dimension is 1, using a half scale will measure 2^1 half-scales
          If the dimension is 2, using a half scale will measure 2^2 half-scales
          If the dimension is d and scale is f, then the measure would be 1/f^d f-scales
    """
    lnPrices = transformer_fn(prices)
    scale = prices.shape[0]
    pl = np.diff(lnPrices)
    tot_return_in_abs = np.max([0.0001,np.abs(np.sum(pl)) ]) ## doesnt allow this to be zero
    abs_return_total = np.sum(np.abs(pl))
    # fraction below should equal scale if the fractal dimension is 1
    # meaning if dimension is 1, that would mean chopping a length in 
    # 'scale' number of pieces will give me identical length that was broken up to 
    # begin with
    fraction = abs_return_total /(tot_return_in_abs / scale) 
    fractal_d  = np.log(fraction)/np.log(scale)
    return(fractal_d)

def makeFractals(df, pricecol,windows = [14,34,55,89,180], transformer_fn = np.log):
    """
        INPUT:
            df  -- data frame which contains the prices
            pricecol -- string containing the name of the column with price
            windows -- windown over which fractals will be calculated
            transformer_fn -- function which transforms prices before calculating
        OUTPUT:
            return a data frame with price and column extended with fractals
    """
    tmp = df.copy(deep=True)
    fracfn = lambda x : fractal(x,transformer_fn=transformer_fn)
    for j in windows:
        tmp['{}d_fractal'.format(j)] = tmp[pricecol].rolling(j).apply(fracfn)
    return tmp

def priceAndFractals(df , pricecol , periodstr = '2018', fracCols=['34d_fractal']):
    """
        INPUT:
            df -- dataframe with fractal column and prices
            pricecol -- string denoting the price column
            periodstr -- period to consider for visualization
            fracCols -- fractals to show
        OUTPUT:
            plots the fractals and price to visualize
    """
    cols = [pricecol] + fracCols
    ax = df.loc[periodstr,cols].plot(figsize = [15,10],subplots=True);
    return ax

def rescaleRange(X, transformer_fn = lambda x: x):
    '''
        https://en.wikipedia.org/wiki/Rescaled_range
        INPUT:
            X -- a vector containing a time sequence
            transform_fn -- preprocessor
        OUTPUT:
            rescaled range as defined in the link above
    '''
    X = transformer_fn(X)
    m = np.mean(X)
    Y = X - m
    Y = X - m
    Z = np.cumsum(Y)
    R = max(Z) - min(Z)
    S = np.std(X)
    return R/S  

def rescaleRange_Series(X, transformer_fn = lambda x: x):
    '''
        https://en.wikipedia.org/wiki/Rescaled_range
        INPUT:
            X -- a vector containing a time sequence
            transform_fn -- preprocessor
        OUTPUT:
            series of rescaled range as defined in the link above
    '''
    X = transformer_fn(X)
    m = np.mean(X)
    Y = X - m
    Z = np.cumsum(Y)
    R = [max(X[:j+1] - min(X[:j+1])) for j in range(len(Z))]
    S = [np.std(X[:j+1]) for j in range(len(X))]
    rescale_range = [(j , R[j] / S[j]) for j in range(1,len(R))]
    R_S = pd.DataFrame(rescale_range, columns=['N','R_S'])
    return R_S

def calcRS(df, pricecol,windows = [14,34,55,89,180], transformer_fn = np.log):
    """
        https://en.wikipedia.org/wiki/Rescaled_range
        INPUT:
            df  -- data frame which contains the prices
            pricecol -- string containing the name of the column with price
            windows -- windown over which fractals will be calculated
            transformer_fn -- function which transforms prices before calculating
        OUTPUT:
            return a data frame with price and column extended with rescaled range
    """
    tmp = df.copy(deep=True)
    rescalefn = lambda x : rescaleRange(x,transformer_fn=transformer_fn)
    for j in windows:
        tmp['{}d_RS'.format(j)] = tmp[pricecol].rolling(j).apply(rescalefn)
    return tmp

def calcHC(RS , windows = [14,34,55,89,180]):
    '''
        https://en.wikipedia.org/wiki/Hurst_exponent
        INPUT:
             RS -- Expected Risk Ranges corresponding to windows. 
                 RS is a vector and has the same shape as windows
             windows -- window length for which rescaled ranges exist
        OUTPUT:
            Hurst Exponent
    '''
    X = np.log(windows)
    colnames = ['{}d_RS'.format(j) for j in windows]
    y = np.log(RS)
    lr = LinearRegression()
    lr.fit(X.reshape(-1,1),y)
    result = 1.0 if lr.coef_[0] > 1.0 else lr.coef_[0]
    return result

"""
def calcHCFromPrices(df, pricecol,windows = [14,34,55,89,180],\
                     transformer_fn = np.log,\
                     rolling_hc_window = 60):
    '''
        https://en.wikipedia.org/wiki/Hurst_exponent
        https://en.wikipedia.org/wiki/Rescaled_range
        INPUT:
            df -- data frame containing price col
            pricecol -- name of the column containing the price
            windows -- window of periods over which rescaled range will be computed
            transformer_fn -- preprocessor for the variables
            rolling_hc_window -- period over which rescaled ranges will be averaged for 
                estimating Hurst exponent...
        OUTPUT:
            returns a data frame containing rolling hurst components
        
    '''
    tmp = calcRS(df,pricecol,windows, transformer_fn)
    ## For loop below computes the average of Risk Range over the period of rolling_hc_window
    ## This value will be later used to infer hurst exponent
    mean_cols = []
    for j in windows:
        rscol = '{}d_RS'.format(j)
        mean_col_name = '{}d_RS_mean'.format(j)
        tmp[mean_col_name] = tmp[rscol].rolling(rolling_hc_window).mean()
        mean_cols .append(mean_col_name)
    tmp = tmp.dropna() ## Drops the resulting NAs
    mean_rs_vals = tmp.loc[:,mean_cols].values
    res = np.array([calcHC(mean_rs_vals[j],windows) for j in range(tmp.shape[0])])
    tmp['HurstExponent'] = res    
    return tmp
"""

def calcHCFromPrices(df, pricecol,windows = [14,34,55,89,180],\
                     transformer_fn = np.log,\
                     rolling_hc_window = 60):
    '''
        https://en.wikipedia.org/wiki/Hurst_exponent
        https://en.wikipedia.org/wiki/Rescaled_range
        INPUT:
            df -- data frame containing price col
            pricecol -- name of the column containing the price
            windows -- window of periods over which rescaled range will be computed
            transformer_fn -- preprocessor for the variables
            rolling_hc_window -- period over which rescaled ranges will be averaged for 
                estimating Hurst exponent...
        OUTPUT:
            returns a data frame containing rolling hurst components
        
    '''
    tmp = calcRS(df,pricecol,windows, transformer_fn)
    ## For loop below computes the average of Risk Range over the period of rolling_hc_window
    ## This value will be later used to infer hurst exponent
    rscols = ['{}d_RS'.format(j) for j in windows]
    tmp = tmp.dropna()
    rs_vals = tmp.loc[:,rscols].values
    res = np.array([calcHC(rs_vals[j],windows) for j in range(tmp.shape[0])])
    tmp['HurstExponent'] = res    
    return tmp



def computeFractalsAndHurst(df,pricecol,\
                            windows_rs = [90,180, 265, 700],\
                            windows_fractal = [14,34,55,89,180],
                           transformer_fn = np.log,\
                           rolling_hc_window = 60):
    """
        https://en.wikipedia.org/wiki/Hurst_exponent
        https://en.wikipedia.org/wiki/Rescaled_range
        INPUT:
            df -- data frame containing price col
            pricecol -- name of the column containing the price
            windows -- window of periods over which rescaled range will be computed
            transformer_fn -- preprocessor for the variables
            rolling_hc_window -- period over which rescaled ranges will be averaged for 
                estimating Hurst exponent...
        OUTPUT:
            returns a data frame containing rolling hurst components, risk ranges and fractals
    """
    res_1 = calcRS(df,pricecol,windows_rs,transformer_fn)
    #res_2 = calcHCFromPrices(res_1,pricecol,windows,transformer_fn,rolling_hc_window)
    res_2 = calcHCFromPrices(res_1,pricecol,windows_rs,transformer_fn,rolling_hc_window)
    res_3 = makeFractals(res_2, pricecol,windows_fractal , transformer_fn)
    return res_3.dropna()

    