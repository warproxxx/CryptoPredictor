import quandl
import pandas as pd
import numpy as np
from data_utils import BasicFunctions, PriceFunctions

class testBacktester:
    def __init__(self):
        
        np.random.seed(1)
        
        self.bars = {}
        self.signals = {}
        
        for i in ['btc', 'eth']:
            self.bars[i] = pd.read_csv('BackTest/tests/{}_test.csv'.format(i))
            self.bars[i]['Percentage'] = np.random.uniform(low=-0.2, high=0.2, size=self.bars[i].shape[0])
            self.bars[i]['Classification'] = self.bars[i]['Percentage'].apply(PriceFunctions().percentage_to_classification)
            self.bars[i]['Classification'] = self.bars[i]['Classification'].apply(lambda x: np.random.uniform(low=0, high=0.5) if x == 0 else np.random.uniform(low=0.5, high=1)) #adding randomness to classification

            self.signals[i] = np.asarray(self.bars[i][['Classification', 'Percentage']])
            self.bars[i]['Percentage'] = np.absolute(self.bars[i]['Percentage'])

            self.bars[i].drop(['Percentage', 'Classification'], axis=1, inplace=True)
        
    def get_dataframe(self):
        idx = ['Date', 'Coin', 'Price', 'Bankroll', 'Amount', 'Type', 'Position', 'Status']
        
        #bankroll currently contains new value after position is opened
        df = pd.DataFrame(columns=idx)
        df = df.append(pd.Series([0, 'BTC', 1000, 8000, 2000, 'OPEN', 'LONG', 'INACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([1, 'BTC', 1000, 6000, 2000, 'OPEN', 'LONG', 'INACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([2, 'BTC', 1000, 4000, 2000, 'OPEN', 'LONG', 'INACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([3, 'BTC', 1000, 10000, 6000, 'CLOSE', 'SHORT', 'INACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([4, 'BTC', 1000, 0, 10000, 'OPEN', 'LONG', 'INACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([5, 'BTC', 1000, 10000, 10000, 'CLOSE', 'SHORT', 'INACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([6, 'BTC', 1000, 20000, 10000, 'OPEN', 'SHORT', 'INACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([7, 'BTC', 1000, 10000, 10000, 'CLOSE', 'LONG', 'INACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([8, 'BTC', 1000, 9000, 1000, 'OPEN', 'LONG', 'ACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([9, 'BTC', 1000, 8000, 1000, 'OPEN', 'LONG', 'ACTIVE'], index=idx), ignore_index=True)
        
        return df
        
    def test_get_avilableamount(self):
        bt = Backtester(self.bars, self.signals)
        
        bt.set_positions(self.get_dataframe())
        
        assert(bt.get_avilableamount()['long'] == 8000)
        assert(bt.get_avilableamount()['short'] == 12000)
        
        idx = ['Date', 'Coin', 'Price', 'Bankroll', 'Amount', 'Type', 'Position', 'Status']
        
        df = pd.DataFrame(columns=idx)
        df = df.append(pd.Series([0, 'BTC', 1000, 8000, 2000, 'OPEN', 'LONG', 'INACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([1, 'BTC', 1000, 6000, 2000, 'OPEN', 'LONG', 'INACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([2, 'BTC', 1000, 4000, 2000, 'OPEN', 'LONG', 'INACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([3, 'BTC', 1000, 10000, 6000, 'CLOSE', 'SHORT', 'INACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([4, 'BTC', 1000, 20000, 10000, 'OPEN', 'SHORT', 'ACTIVE'], index=idx), ignore_index=True)
        bt.set_positions(df) 
        
        assert(bt.get_avilableamount()['long'] == 20000)
        assert(bt.get_avilableamount()['short'] == 0)
        
        df = pd.DataFrame(columns=idx)
        df = df.append(pd.Series([0, 'BTC', 1000, 12000, 2000, 'OPEN', 'SHORT', 'ACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([1, 'BTC', 1000, 14000, 2000, 'OPEN', 'SHORT', 'ACTIVE'], index=idx), ignore_index=True)
        bt.set_positions(df) 
        
        assert(bt.get_avilableamount()['long'] == 14000)
        assert(bt.get_avilableamount()['short'] == 6000)
        
    def test_check_validity(self):
        bt = Backtester(self.bars, self.signals)
        bt.set_positions(self.get_dataframe())
        
        assert(bt.check_validity('LONG', 8000) == True)
        assert(bt.check_validity('LONG', 2000) == True)
        assert(bt.check_validity('LONG', 8001) != True)
        
        idx = ['Date', 'Coin', 'Price', 'Bankroll', 'Amount', 'Type', 'Position', 'Status']

        df = pd.DataFrame(columns=idx)
        df = df.append(pd.Series([0, 'BTC', 1000, 15000, 5000, 'OPEN', 'SHORT', 'ACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([1, 'BTC', 1000, 19000, 4000, 'OPEN', 'SHORT', 'ACTIVE'], index=idx), ignore_index=True)
        bt.set_positions(df)
        
        assert(bt.check_validity('LONG', 500) == True)
        assert(bt.check_validity('LONG', 19000) == True)
        assert(bt.check_validity('LONG', 19001) != True)

        assert(bt.get_avilableamount()['long'] == 19000)
        assert(bt.get_avilableamount()['short'] == 1000)
        
        df = pd.DataFrame(columns=idx)
        df = df.append(pd.Series([0, 'BTC', 1000, 13000, 3000, 'OPEN', 'SHORT', 'ACTIVE'], index=idx), ignore_index=True)
        df = df.append(pd.Series([1, 'BTC', 1000, 15000, 2000, 'OPEN', 'SHORT', 'ACTIVE'], index=idx), ignore_index=True)
        bt.set_positions(df)
        
        assert(bt.check_validity('LONG', 15000) == True)
        assert(bt.check_validity('SHORT', 5000) == True)
        assert(bt.check_validity('SHORT', 5001) != True)
        
    def test_perform_trade(self):
        bt = Backtester(self.bars, self.signals)
        
        #date, coin, price, amount, tradetype, postion
        bt.perform_trade(1, 'BTC', 1000, 2000, 'Open', 'LONG')
        bt.perform_trade(2, 'BTC', 1005, 3000, 'Open', 'LONG')
        bt.perform_trade(2, 'BTC', 1005, 3000, 'Open', 'SHORT')
        positions = bt.get_positions()
        print(positions)
        
    def test_close_reverse_position(self):
        bt = Backtester(self.bars, self.signals)
        bt.set_positions(self.get_dataframe())
        bt.close_reverse_position(signal='LONG')
        
    def test_find_best(self):
        bt = Backtester(self.bars, self.signals)
        indicators = bt.find_best()
        
        #print(indicators)
        
        assert(indicators[0]['coin'] == 'btc') #0.04917342 of btc is smaller than 0.05737299 of eth
        assert(indicators[0]['position'] == 'SHORT')
        
        assert(indicators[1]['coin'] == 'btc') #0.71055381 of btc is better than 0.47474463 of eth
        assert(indicators[1]['position'] == 'LONG')
        
        assert(indicators[2]['coin'] == 'eth') #0.72495607 of eth is better than 0.47894477 of btc
        assert(indicators[2]['position'] == 'LONG')
        
        assert(indicators[3]['coin'] == 'btc') #0.26658264 of btc is smaller than 0.28919481 of eth
        assert(indicators[3]['position'] == 'SHORT')
        
        assert(indicators[9]['coin'] == 'btc') #0.87507216 of btc is bigger than 0.80857246 of eth
        assert(indicators[9]['position'] == 'LONG')
        
    def test_perform_backtest(self):
        bt = Backtester(self.bars, self.signals)
        bt.perform_backtest()
            
testBacktester().test_check_validity()

testBacktester().test_get_avilableamount()
testBacktester().test_perform_trade()
testBacktester().test_close_reverse_position()

testBacktester().test_perform_backtest()
testBacktester().test_find_best()