from data_utils import BasicFunctions, PriceFunctions
import numpy as np
import pandas as pd

def test_initialize_mini_batch():
    bf = BasicFunctions()
    
    np.random.seed(1)
    X = np.random.randn(12288, 148)
    y = np.random.randn(1, 148)

    batches = bf.initialize_mini_batch(X,y,64)
    assert(batches[0][0].shape == (12288, 64)) #testing X
    assert(batches[1][0].shape == (12288, 64))
    assert(batches[2][0].shape == (12288, 20))
    assert(batches[0][1].shape == (1, 64)) #testing y
    assert(batches[1][1].shape == (1, 64))
    assert(batches[2][1].shape == (1, 20))
    
#test for PriceFunctions is not added now because they will be modified when price functions for other cryptos is added