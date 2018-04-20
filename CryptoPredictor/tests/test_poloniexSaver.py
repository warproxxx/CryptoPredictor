import pandas as pd
from data_utils import PriceFunctions

class TestpoloniexDownloader():

    def test_mergeData(self):
        pf = PriceFunctions()
        self.btc = pf.get_pandas(coin='BTC', targetdays=24, absolute=True) #as the data is hourly
        self.btc = self.btc.reset_index()

        xmr = pd.read_csv('CryptoScraper/cache/XMR.csv')
        dash = pd.read_csv('CryptoScraper/cache/DASH.csv')
        doge = pd.read_csv('CryptoScraper/cache/DOGE.csv')
        eth = pd.read_csv('CryptoScraper/cache/ETH.csv')
        ltc = pd.read_csv('CryptoScraper/cache/LTC.csv')
        str = pd.read_csv('CryptoScraper/cache/STR.csv')
        xrp = pd.read_csv('CryptoScraper/cache/XRP.csv')

        self.btcXmr = self.btc[self.btc['Date'] >= xmr.iloc[0]['Date']]
        self.btcXmr.reset_index(inplace=True, drop=True)

        assert (sum(self.btcXmr['Date'] != xmr['Date']) == 0)

        self.btcDash = self.btc[self.btc['Date'] >= dash.iloc[0]['Date']]
        self.btcDash.reset_index(inplace=True, drop=True)

        assert (sum(self.btcDash['Date'] != dash['Date']) == 0)

        self.btcDoge = self.btc[self.btc['Date'] >= doge.iloc[0]['Date']]
        self.btcDoge.reset_index(inplace=True, drop=True)

        assert (sum(self.btcDoge['Date'] != doge['Date']) == 0)

        self.btcEth = self.btc[self.btc['Date'] >= eth.iloc[0]['Date']]
        self.btcEth.reset_index(inplace=True, drop=True)

        assert (sum(self.btcEth['Date'] != eth['Date']) == 0)

        self.btcLtc = self.btc[self.btc['Date'] >= ltc.iloc[0]['Date']]
        self.btcLtc.reset_index(inplace=True, drop=True)

        assert (sum(self.btcLtc['Date'] != ltc['Date']) == 0)

        self.btcStr = self.btc[self.btc['Date'] >= str.iloc[0]['Date']]
        self.btcStr.reset_index(inplace=True, drop=True)

        assert (sum(self.btcStr['Date'] != str['Date']) == 0)

        self.btcXrp = self.btc[self.btc['Date'] >= xrp.iloc[0]['Date']]
        self.btcXrp.reset_index(inplace=True, drop=True)

        assert (sum(self.btcXrp['Date'] != xrp['Date']) == 0)