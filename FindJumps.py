import numpy as np
import pandas as pd
from yahoo_fin import options
from yahoo_fin import stock_info as si

symbol = 'TSLA'  # AAPL, TSLA, MSFT, META, CAT, STM, GOOG, NFLX, AMZN

stock_data = si.get_data(symbol, start_date='31/05/2021', end_date='31/05/2023')
print(stock_data.head())
stock_data['Returns'] = stock_data['close'] / stock_data['close'].shift()
stock_data['Log Returns'] = np.log(stock_data['Returns'])
print(stock_data.head())


# %%%%%%%%%%%%%%%%%%%%%%%%       THRESHOLD       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
threshold = -0.94
jumps_df = stock_data[stock_data['Returns'] < threshold]
print(jumps_df)





# %%%%%%%%%%%%%%%%%%%%%%%%       95-TH PERCENTILE       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





# %%%%%%%%%%%%%%%%%%%%%%%%       NORMALITY TEST       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





