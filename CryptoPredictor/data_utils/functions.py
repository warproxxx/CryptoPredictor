import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CryptoScraper import BtcFinex
import math

class BasicFunctions():
    
    def initialize_mini_batch(self, X, y, batchsize=64, random=0):
        '''
        Creates minibatches from values of X and y.

        Parameters:
        X (numpy)
        y (numpy)

        batchsize (optional):
        Default is 64

        random (optional):
        Shuffles before creating minibatch if set to 1

        Returns:
        batches (list):
        list in the format - [[batchX1, batchY1], .... [batchXn, batchYn]]

        '''       
        if (random == 1):
            X, y = shuffle(X,y)

        batches = []

        divide = X.shape[1] // batchsize

        for i in range(1,divide+1):
            batchX = X[:,:64]
            batchY = y[:,:64]

            batches.append((batchX,batchY))
            X = X[:,batchsize:]
            y = y[:,batchsize:] 

        batches.append((X,y))
        return batches

    def single_plot(self, y, title='', Xtitle='', Ytitle='', log=True):
        fig = plt.figure();
        axes = fig.add_axes([0,0,1,1])

        axes.set_xlabel(Xtitle)
        axes.set_ylabel(Ytitle)
        axes.set_title(title)
        
        if (log == True):
            axes.plot(np.arange(1,len(y) + 1), np.log(y))
        else:
            axes.plot(np.arange(1,len(y) + 1), y)

class PriceFunctions():
    def percentage_to_classification(self, x):
        #returns a number y [-4,4] depending on how much it went up/down
        y = 0
        
        if (x >= 0):
            y = 1
        else:
            y = 0
            
        return y
    
    def get_numpy(self, pd_Xtrain, pd_ytrain, pd_Xtest, pd_ytest):
        '''
        Normalized numpy from pandas
        
        Arguments:
        pd_Xtrain: Unnormalized X train as pandas
        pd_ytrain: Unnormalized y train as pandas
        pd_Xtest: Unnormalized X test as pandas
        pd_ytest: Unnormalized y test as pandas
        
        Returns:
        mean: pandas mean of training set
        std: pandas std of training set
        Xtrain: Normalized pandas to numpy
        ytrain: Normalized pandas to numpy
        Xtest: Normalized pandas to numpy
        ytest: Normalized pandas to numpy
        '''
        
        mean = pd_Xtrain.mean()
        std = pd_Xtrain.std()

        pd_XtrainNorm = (pd_Xtrain - mean)/std
        pd_XtestNorm = (pd_Xtest - mean)/std
    
        Xtrain = np.array(pd_XtrainNorm)
        ytrain = np.array(pd_ytrain).astype(np.float32)

        Xtest = np.array(pd_XtestNorm) 
        ytest = np.array(pd_ytest).astype(np.float32)
        
        return mean, std, Xtrain, ytrain, Xtest, ytest
    
    def get_pandas(self, targetdays=24, absolute=True, coin='BTC', data='cached'):
        '''
        Parameters:
        data: (int) (optional)
        'cached' returns cached data. 'download' adds data to cache or redownloads if there is no cache.
        
        targetdays: (int) (optional)
        specify the target number of timeframe from which percentage change is to be calculated. 24 for daily change in hourly. 1 for daily change in daily 
        
        negative: (boolean) (optional)
        If set to true, absolute value of percentage change is returned
        
        Returns:
        
        df: 
        pandas dataframe of data from bitfinex
        '''
        
        targetdays = -1 * targetdays
        
        if (coin == 'BTC'):
            finex = BtcFinex()

            if (data == 'download'):
                finex.loadData()

            df = finex.getCleanData()
            df.set_index('Time', inplace=True)
        
        df['Percentage Change'] = (1 - df['Close']/df.shift(targetdays)['Close'])
        df['Classification'] = df['Percentage Change'].apply(PriceFunctions().percentage_to_classification)
        
        df['Classification'] = df['Classification'].astype(np.float32)
        df['Percentage Change'] = df['Percentage Change'].astype(np.float32)
        
        if (absolute == True):
            df['Percentage Change'] = df['Percentage Change'].abs()
        
        df = df[:targetdays]
        
        return df
    
    def split_traintest(self, df, ratio=0.83):
        '''
        Parameters:
        df: dataframe to split into training and test set
        ratio (int optional): The size of training set in percentage
        
        
        Returns:
        pd_Xtrain: Unnormalized X train as pandas
        pd_ytrain: Unnormalized y train as pandas
        pd_Xtest: Unnormalized X test as pandas
        pd_ytest: Unnormalized y test as pandas
        '''
        
        trainTill = math.floor(ratio * df.shape[0])
        dfTraining = df[:trainTill]
        dfTest = df[trainTill:]
        
        pd_ytrain = dfTraining[['Classification', 'Percentage Change']]
        pd_Xtrain = dfTraining.drop(['Classification', 'Percentage Change'], axis=1)
        
        pd_ytest = dfTest[['Classification', 'Percentage Change']]
        pd_Xtest = dfTest.drop(['Classification', 'Percentage Change'], axis=1)
        
        return pd_Xtrain, pd_ytrain, pd_Xtest, pd_ytest