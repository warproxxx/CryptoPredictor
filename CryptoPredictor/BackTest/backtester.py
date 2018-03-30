import pandas as pd
import json
class Backtester:

    '''
    Backtesting module that does everything

    At the end, in positions, a new column - current net worth should be added
    '''
    
    #add margin rates and margin size too later
    def __init__(self, bars, signals, period=1, bankroll=10000, stake=1000, comission=0.2, slippage=0):
        '''
        Parameters:
        ___________
        bars (dictionary):
        Dictionary of Pandas dataframe containing Open, Close, High and Low Value. The Close value will be used
        
        signals (dictionary):
        Dictionary of coins that contain Numpy array containg direction probablity and magnitude of change.
        
        period (int):
        Number that represents how far in the future the timeframe lies
        1 is used for classification whose target is 1 timeframe away and so on
        
        bankroll (int optional):
        The initial cash to start with
        
        stake (int optional):
        size of each trade
        
        comission (int optional):
        Comission per trade in percentage (out of 100)
        
        slippage (int optional):
        Slippage per trade in percentage (out of 100)
        '''
        self.bars = bars
        self.signals = signals
        self.period = period
        self.bankroll = bankroll
        self.stake = stake
        self.comission = comission
        self.slippage = slippage
        #Type has open or close. Position has long or short. So 
        #long close means closing short
        #long open mean opening long
        #short open meaning opening short
        #short close meaning closing short 
        self.positions = pd.DataFrame(columns=['Date', 'Coin', 'Price', 'Bankroll', 'Amount', 'Type', 'Position', 'Status'])
        
        self.perform_assertion()
    
    def perform_assertion(self):
        '''
        Check if all the values have same timeframes and if they have all required columns
        '''
        shape = self.bars[list(self.bars)[0]].shape
        
        keyList=sorted(self.bars.keys())
        
        for i, key in enumerate(keyList):
            if i > 0:
                try:
                    assert(self.bars[keyList[i]]['Date'].equals(self.bars[keyList[i-1]]['Date']))
                except AssertionError:
                    print("The dates in your dataframe do not match. The dates of {} and {} are different at".format(keyList[i], keyList[i-1]))
                
                try:
                    assert(self.signals[keyList[i]].shape == self.signals[keyList[i-1]].shape)
                except AssertionError:
                    print("The signals don't match at {} and {}".format(keyList[i], keyList[i-1]))
    
    def set_positions(self, df):
        self.positions = df
    
    def get_positions(self):
        return self.positions
        
    def get_avilableamount(self):
        '''
        Checks the dataframe and finds how much percentage is tradable

        Returns:
        ________
        The percentage that is avilable to trade
        '''
        
        avilable = {}
        
        
        activePositions = self.positions[self.positions['Status'] == 'ACTIVE']
        
        tempdf = activePositions.apply(lambda x: x['Amount'] * -1 if x['Position'] == 'SHORT' else x['Amount'], axis=1)
        
        totalValue = activePositions['Bankroll'].iloc[-1] + sum(tempdf)
        
        avilable['long'] = self.positions['Bankroll'].iloc[-1]
        avilable['short'] = totalValue * 2 - avilable['long']
        
        return avilable
        
    def check_validity(self, position, size):
        '''
        Check if a long or short position can be opened currently.
        
        Returns:
        ________
        True if perforamble
        
        The size of long and short that can be performed if not performable
        '''
        
        avilable = self.get_avilableamount()

        if position == 'LONG':
            if (avilable['long'] >= size):
                return True
            else:
                return avilable['long'] 
            
        elif position == 'SHORT':
            if (avilable['short'] >= size):
                return True
            else:
                return avilable['short'] 
        
    def close_reverse_position(self, signal, currprice, date):
        ''' 
        Close all reverse positions. If long signal is generated and short position is open, close it and vice versa
        
        Parameters:
        ___________

        signal: (string)
        'LONG' or 'SHORT'

        currprice: (dict)
        dictionary containing symbols and their price while closing position

        date: (int)
        Date in which the trade takes place
        '''
        
        #perform reverse trade
        #call perform_trade function to perform the reverse trade and change status to inactive
        if (signal == 'LONG'):
            reverse = 'SHORT'
        elif (signal == 'SHORT'):
            reverse = 'LONG'
            
        
        closingsignals = self.positions[(self.positions['Status'] == 'ACTIVE') & (self.positions['Position'] == reverse)]
        
        if (self.positions.shape[0] == 0):
            oldBankroll = self.bankroll
        else:
            oldBankroll = self.positions['Bankroll'].iloc[-1]

        #now for given coin close at the current price
        for coin in currprice:
            requiredcoins = closingsignals[self.positions['Coin'] == coin]

            if(closingsignals[self.positions['Coin'] == coin].shape[0] != 0): #this also happens automatically with the other warning
                perChange = (currprice[coin] - requiredcoins['Price'])/requiredcoins['Price']
                
                if (reverse == 'LONG'):
                    newAmounts = requiredcoins['Amount'] + requiredcoins['Amount'] * perChange
                elif (reverse == 'SHORT'):
                    requiredcoins['Amount'] = requiredcoins['Amount'] * -1
                    newAmounts = requiredcoins['Amount'] + requiredcoins['Amount'] * perChange

                
                closingChange = sum(newAmounts)
                

                newData = pd.Series({'Date': date, 'Coin': coin, 'Price': currprice[coin], 'Bankroll': oldBankroll+closingChange, 'Amount': abs(sum(requiredcoins['Amount'])), 'Type': 'CLOSE', 'Position': signal, 'Status': 'INACTIVE'})
                #also change old ones to inactive. Append newData too on the dataframe at end. And works for long. also check for short
                self.positions = self.positions.append(newData, ignore_index=True)
                oldBankroll = oldBankroll+closingChange

        self.positions.loc[(self.positions['Status'] == 'ACTIVE') & (self.positions['Position'] == reverse), 'Status'] = 'INACTIVE'


    def close_all_positions(self, date, currprice):
        '''
        Just call the close reverse position twice
        '''
        #I think getting the coin names is not required

        self.close_reverse_position('LONG', currprice, date)
        self.close_reverse_position('SHORT', currprice, date)

    def perform_trade(self, date, coin, currprice, amount, tradetype, position):
        '''
        Perform trade and change dataframe to reflect it
        
        Parameters:
        ___________
        
        Date: (int)
        The date in which the trade took place
        
        coin (3 words):
        Symbol like btc
        
        currprice: (dict)
        dictionary containing symbols and their price
        
        amount: (int)
        In money, how many's to open
        
        tradetype: (string)
        Open or Close
        
        
        postion: (string)
        LONG or SHORT
        '''
        
        #Change dataframe that way
        #The trade data should be added to self.positions.
        #The initial bankroll value should be derived from the variable
        #after that from previous +- trade
        #bankroll currently contains new value after position is opened

        self.close_reverse_position(position, currprice, date) #Close reverse positions before performing

        if (self.positions.shape[0] == 0):
            curbankroll = self.bankroll
        else:
            curbankroll = self.positions['Bankroll'].iloc[-1]

        if position == 'LONG':
            newbankroll = curbankroll-amount
        elif position == 'SHORT':
            newbankroll = curbankroll+amount

        ser = pd.Series({'Date': date, 'Coin': coin, 'Price': currprice[coin], 'Bankroll': newbankroll, 'Amount': amount, 'Type': tradetype, 'Position': position, 'Status': 'ACTIVE'})
        self.positions = self.positions.append(ser, ignore_index=True)
    
    def find_best(self):
        '''
        Finds the best coin to buy for that day
        '''
        
        #compare the x of the 3            
        #The one with highet probablity is chosen
        
        shape = len(self.bars[list(self.signals)[0]])
        keyList=sorted(self.signals.keys())
        bests = []
        
        best = {'probablitynorm': 0}
        
        for i in range(shape): #loop through dataframe
            for idx, key in enumerate(keyList): #to loop through dict and compare values of different
                probablity = self.signals[key][i][0]
                
                if probablity < 0.5: #if i feel confused - it is working. I need the smaller number when its smaller and bigger when it is bigger
                    x = 1 - probablity
                else:
                    x = probablity
                    
                if x > best['probablitynorm']:
                    best['coin'] = key
                    best['date'] = self.bars[key]['Date'][i]
                    best['probablity'] = probablity
                    best['probablitynorm'] = x
                    best['percentage'] = self.signals[key][i][1]
                    
                    if (probablity < 0.5):
                        best['position'] = 'SHORT'
                    else:
                        best['position'] = 'LONG'
                    
                    
            bests.append(best)
            best = {'probablitynorm': 0}
        
        return bests
        
    def perform_backtest(self):
        data = self.find_best()

        for dic in data:
            prob = dic['probablity']
            perc = dic['percentage']
            coin = dic['coin']
            date = dic['date']
            pos = dic['position']
            normprob = dic['probablitynorm']

            
            
            #check if performable
            #add conditions and find amount this is doable
            #perform trade closes reverse positions
                

        


        #write pseudocode first:
            #if 1
                #If direction > 2% or confidence >0.5 go long with 20%
                #If direction > 3% or confidence >0.6 go long with 50%
                #If direction > 4% or confidence >0.7 go long with 100%
            #if 0
                #confidence = 1 - confidence
                #then as above but short

            #Next day:
                #If already long and positive, keep going long as strategy until 100%
                #If already long and negative, close position
                #Vice Versa


        #reset best at end