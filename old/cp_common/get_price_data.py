import os
import pickle
import quandl
import numpy as np
import pandas as pd

def get_quandl_data(quandl_id, force_download, start_date):
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')

    if (force_download == 0):
        try:
            f = open(cache_path, 'rb')
            df = pickle.load(f)   
            print('Loaded {} from cache'.format(quandl_id))
        except (OSError, IOError) as e:
            print('Downloading {} from Quandl'.format(quandl_id))
            df = quandl.get(quandl_id, returns="pandas", start_date=start_date)
            df.to_pickle(cache_path)
            print('Cached {} at {}'.format(quandl_id, cache_path))
    else:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas",start_date=start_date)
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))

    return df

def get_bitcoin_data(exchanges, force_download=0, start_date="2001-01-01"):
    '''
        Returns daily Bitcoin data in  dataframe combining multiple exchanges supported by quandl.
        Sums the Volume data while averages Open, High, Low, Close and Weighted Price

        Parameters:
        ___________
        exchanges (list):
        List of exchanges. Example [COINBASE] for single. For Multiple ['COINBASE','BITSTAMP','KRAKEN', 'BITFINEX']

        force_download (optional) int:
        If set to 1 data will be force downloaded
        
        start_date (optional):
        Default: 2001-01-01 (YYYY-MM-DD)
        
        Returns:
        ________
        Pandas Dataframe

        '''
    exchange_data = {}

    for exchange in exchanges:
        exchange_code = 'BCHARTS/{}USD'.format(exchange)
        btc_exchange_df = get_quandl_data(exchange_code, force_download, start_date)
        exchange_data[exchange] = btc_exchange_df
        exchange_data[exchange].replace(0, np.nan, inplace=True)

    pd_concat = pd.concat((exchange_data.values()))
    data = pd_concat.groupby(pd_concat.index).mean()[['Open', 'High', 'Low', 'Close', 'Weighted Price']]
    sumData = pd_concat.groupby(pd_concat.index).sum()[['Volume (Currency)', 'Volume (BTC)']]
    df = pd.concat([data, sumData], axis=1)

    return df;

def get_altcoin_data(coins, force_download=0):
    '''
    Gets altcoin data from Poloniex. Do this later

    '''