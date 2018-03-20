import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CryptoScraper import BtcFinex
import math

class BasicFunctions():
    def convert_to_one_hot(self, passedArr, C):
        '''
        Converts the one hot value

        Arguments:
        passedArr -- Numpy array In format [[-4,-3,...4]]
        C -- The number of classifications

        Returns:
        array:
        The value in one hot format.
        Like:
        [[1,0,.........0],
         [0,1,.........0],
         [0,0,.........1]]

        '''   
        reshaped = passedArr.reshape(-1)
        newReshaped = reshaped + 4 #As there are negatives

        digitsLength = np.unique(newReshaped).shape[0]


        returnVal= np.eye(C)[newReshaped]

        return(returnVal)

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
        ytrain = np.array(pd_ytrain)

        Xtest = np.array(pd_XtestNorm) 
        ytest = np.array(pd_ytest)
        
        return mean, std, Xtrain, ytrain, Xtest, ytest
    
    
    
    def get_pandas(self, coin='BTC', data='cached'):
        '''
        Parameters:
        data: 'cached' returns cached data. 'download' adds data to cache or redownloads if there is no cache.
        
        Returns:
        Data from Bitfinex
        
        df: pandas dataframe of data from bitfinex
        '''
        
        if (coin == 'BTC'):
            finex = BtcFinex()

            if (data == 'download'):
                finex.loadData()

            df = finex.getCleanData()
            df.set_index('Time', inplace=True)
        
        df['Percentage Change 24 hours'] = (1 - df['Close']/df.shift(-24)['Close'])
        df['Classification'] = df['Percentage Change 24 hours'].apply(PriceFunctions().percentage_to_classification)
        df.drop('Percentage Change 24 hours', axis=1,inplace=True)

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
        
        dfTest = df[trainTill:-24] #Removing last 24 values as it contain Nan
        
        pd_ytrain = dfTraining['Classification']
        pd_Xtrain = dfTraining.drop('Classification', axis=1)
        
        pd_ytest = dfTraining['Classification']
        pd_Xtest = dfTest.drop('Classification', axis=1)
        
        return pd_Xtrain, pd_ytrain, pd_Xtest, pd_ytest