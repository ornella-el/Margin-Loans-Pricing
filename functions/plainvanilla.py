import pandas as pd

df = pd.read_csv('../data/nvda_vanilla_1601.csv')
print(df)
print(df.columns)

# Select rows where 'QUOTE_DATE' is equal to '2016-01-20' and 'EXPIRE_DATE' is equal to '2017-01-20
cond1 = df['QUOTE_DATE'] == '2016-01-20'
cond2 = df['EXPIRE_DATE'] == '2017-01-20'

df_filtered = df[cond1 & cond2].copy()

# Select only 20 options to use for calibration (STRIKE around UNDERLYING_LAST)
S0 = df['UNDERLYING_LAST'][0]
bounds = [S0 * 0.5, S0 * 1.5]
cond3 = (df_filtered['STRIKE'] >= bounds[0]) & (df_filtered['STRIKE'] <= bounds[1])
df_filtered1 = df_filtered[cond3].copy

# Select only options whose moneyness K/S0 is in range [0.75, 1.35]
df_filtered['MONEYNESS'] = round(df_filtered['STRIKE'] / df_filtered['UNDERLYING_LAST'],2)

print(df_filtered)

# Select rows where moneyness is in the desired range
moneyness_cond2 = (df_filtered['MONEYNESS'] >= 0.75) & (df_filtered['MONEYNESS'] <= 1.35)
moneyness_cond3 = (df_filtered['MONEYNESS'] >= 0.5) & (df_filtered['MONEYNESS'] <= 1.5)

df_filtered2 = df_filtered[moneyness_cond2]
print(df_filtered2)
df_filtered3 = df_filtered[moneyness_cond3]
print(df_filtered3)

# Create a separate dataframe for call and put options
df_calls_bounds = df_filtered[['QUOTE_DATE', 'UNDERLYING_LAST', 'EXPIRE_DATE', 'C_BID', 'C_ASK', 'STRIKE', 'C_IV']].copy()
print(df_calls_bounds)

# 1. Dataframe with bounds on the STRIKE price
df_puts_bounds = df_filtered[['QUOTE_DATE', 'UNDERLYING_LAST', 'EXPIRE_DATE', 'P_BID', 'P_ASK', 'STRIKE', 'P_IV']].copy()
print(df_puts_bounds)
print(df_calls_bounds.shape, df_puts_bounds.shape)

# 2. Dataframe with only OTM options
df_call_otm = df_calls_bounds[df_calls_bounds['STRIKE'] > df_calls_bounds['UNDERLYING_LAST']]
df_put_otm = df_puts_bounds[df_puts_bounds['STRIKE'] < df_puts_bounds['UNDERLYING_LAST']]


# 3. Dataframe with MONEYNESS constraint [0.75, 1.35]
df_calls_moneyness2 = df_filtered2[['QUOTE_DATE', 'UNDERLYING_LAST', 'EXPIRE_DATE', 'C_BID', 'C_ASK', 'STRIKE', 'C_IV', 'MONEYNESS']].copy()
df_puts_moneyness2 = df_filtered2[['QUOTE_DATE', 'UNDERLYING_LAST', 'EXPIRE_DATE', 'P_BID', 'P_ASK', 'STRIKE', 'P_IV', 'MONEYNESS']].copy()
print(df_calls_moneyness2, df_calls_moneyness2.shape)

# 3. Dataframe with MONEYNESS constraint [0.5, 1.35]
df_calls_moneyness3 = df_filtered3[['QUOTE_DATE', 'UNDERLYING_LAST', 'EXPIRE_DATE', 'C_BID', 'C_ASK', 'STRIKE', 'C_IV', 'MONEYNESS']].copy()
df_puts_moneyness3 = df_filtered3[['QUOTE_DATE', 'UNDERLYING_LAST', 'EXPIRE_DATE', 'P_BID', 'P_ASK', 'STRIKE', 'P_IV', 'MONEYNESS']].copy()
print(df_calls_moneyness3, df_calls_moneyness3.shape)

# export to a CSV file

# df_calls_bounds.to_csv('../data/spx_basket_calls_2016.csv', index=False)
# df_puts_bounds.to_csv('../data/spx_basket_puts_2016.csv', index=False)
df_call_otm.to_csv('../data/nvda_calls_OTM_2016.csv', index=False)
df_put_otm.to_csv('../data/nvda_puts_OTM_2016.csv', index=False)
df_calls_moneyness2.to_csv('../data/OPT16_NVDA_CALLS_75_135.csv', index=False)
df_puts_moneyness2.to_csv('../data/OPT16_NVDA_PUTS_75_135.csv', index=False)
df_calls_moneyness3.to_csv('../data/OPT16_NVDA_CALLS_50_150.csv', index=False)
df_puts_moneyness3.to_csv('../data/OPT16_NVDA_PUTS_50_150.csv', index=False)
