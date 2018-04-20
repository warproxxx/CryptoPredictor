from data_utils import BasicFunctions, PriceFunctions
import requests
import json
import pandas as pd

class poloniexDownloader():
    
    def __init__(self, symbol):
        pf = PriceFunctions()
        self.btc = pf.get_pandas(coin='BTC', targetdays=24, absolute=True) #as the data is hourly
        self.btc = self.btc.reset_index()
        self.url = "https://poloniex.com/public?command=returnChartData&currencyPair=BTC_{}&start={}&end=9999999999&period={}" #gives half an hour data
        self.symbol = symbol

    def getData(self):
        firstRes = requests.get(self.url.format(self.symbol, '0', '86400'))
        firstData = firstRes.text
        firstDf = pd.read_json(firstData, convert_dates=False)
        
        finalDate = firstDf['date'][0]
        
        closest = self.btc.iloc[(self.btc['Time']-firstDf['date'][0]).abs().argsort()[0]]['Time']
        res = requests.get(self.url.format(self.symbol, int(closest), 1800))
        data = res.text
        df = pd.read_json(data, convert_dates=False)
        df = df[['date', 'open', 'close', 'high', 'low', 'volume']]
        df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
        
        if ([df['Date']%3600][0][0] != 0): #because keeping 2 values
            df = df[1:]
        
        df = df.reset_index(drop=True)
        
        df = df[df['Date'] <= int(self.btc.iloc[-1]['Time'])] #till the last
        
        return df
    
    def mergeData(self, df, time=2):
        
        retDf = pd.DataFrame(columns=['Date', 'Open', 'Close', 'High', 'Low', 'Volume'])
        
        for i in range(0, df.shape[0], time):

            tempdf = df.iloc[i:i+time]
            retDf = retDf.append({'Date': tempdf.iloc[0]['Date'], 'Open': tempdf.iloc[0]['Open'], 'Close': tempdf.iloc[-1]['Close'], 'High': max(tempdf['High']), 'Low': min(tempdf['Low']), 'Volume': sum(tempdf['Volume'])}, ignore_index=True)
        
        retDf['Date'] = retDf['Date'].astype(int)
        
        return retDf


coins = ['ETH', 'LTC', 'XMR', 'XRP', 'STR', 'DASH', 'DOGE']

for coin in coins:
    ad = poloniexDownloader(coin)
    ret = ad.getData()
    df = ad.mergeData(ret)
    df.to_csv('CryptoScraper/cache/{}.csv'.format(coin), index=False)
    #save df to csv