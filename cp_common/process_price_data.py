from cp_common import get_price_data as gpd
import numpy as np
import pandas as pd


def percentage_to_classification(x):
    #returns a number y [-4,4] depending on how much it went up/down
    y = 0
    
    
    if (x > 0.2):
        y = 4
    elif (0.1 <= x <= 0.2):
        y = 3
    elif (0.05 <= x <= 0.1):
        y = 2
    elif (0.03 <= x <= 0.05):
        y = 1
    elif (-0.03 <= x <= 0.03):
        y = 0
    elif (-0.05 <= x <= -0.03):
        y = -1
    elif (-0.1 <= x <= -0.05):
        y = -2
    elif (-0.2 <= x <= -0.1):
        y = -3
    elif (x < -0.2):
        y = -4
    
    return y

def return_data(andrewEncoding=True):
    
    '''
    Returns:
    
    Data from Bitfinex and Coinbase currently. Modify this for others.
    
    mean: pandas mean of training set
    std: pandas std of training set
    pd_Xtrain: Unnormalized X train as pandas
    pd_ytrain: Unnormalized y train as pandas
    pd_Xtest: Unnormalized X test as pandas
    pd_ytest: Unnormalized y test as pandas
    Xtrain: Normalized pandas to numpy with transpose
    ytrain: Normalized pandas to numpy with transpose
    Xtest: Normalized pandas to numpy with transpose
    ytest: Normalized pandas to numpy with transpose
    '''
    
    dfcoinbase = gpd.get_bitcoin_data(['BITFINEX','COINBASE']) #Test file will download data
    dfcoinbase['Percentage Change'] = dfcoinbase.pct_change()['Weighted Price']
    dfcoinbase['Percentage Change'] = dfcoinbase['Percentage Change'].shift(periods=-1)
    df = dfcoinbase.rename(columns={'Percentage Change': 'Percentage Change Next Day'})  #*100 has not been done
    df['Classification'] = df['Percentage Change Next Day'].apply(percentage_to_classification)
    df.drop('Percentage Change Next Day', axis=1, inplace=True)
    
    dfTraining = df[:1500]
    dfTest = df[1500:-1] #Removing last value as it contain Nan
    
    pd_Xtrain = dfTraining.iloc[:,:-1]
    pd_ytrain = dfTraining.iloc[:,-1:]

    pd_Xtest = dfTest.iloc[:,:-1]
    pd_ytest = dfTest.iloc[:,-1:]
    
    
    mean = pd_Xtrain.mean()
    std = pd_Xtrain.std()
    
    pd_XtrainNorm = (pd_Xtrain - mean)/std
    pd_XtestNorm = (pd_Xtest - mean)/std
    
    if (andrewEncoding==True):
        Xtrain = np.array(pd_XtrainNorm).T
        ytrain = np.array(pd_ytrain).T

        Xtest = np.array(pd_XtestNorm).T 
        ytest = np.array(pd_ytest).T
    else:
        Xtrain = np.array(pd_XtrainNorm)
        ytrain = np.array(pd_ytrain)

        Xtest = np.array(pd_XtestNorm) 
        ytest = np.array(pd_ytest)
    
    return mean, std, pd_Xtrain, pd_ytrain, pd_Xtest, pd_ytest, Xtrain, ytrain, Xtest, ytest