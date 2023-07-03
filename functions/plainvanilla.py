import pandas as pd

df = pd.read_csv('../data/spx_vanilla_1601.csv')
print(df)
print(df.columns)

# Select rows where 'QUOTE_DATE' is equal to '2016-01-20' and 'EXPIRE_DATE' is equal to '2017-01-20
cond1 = df['QUOTE_DATE'] == '2016-01-20'
cond2 = df['EXPIRE_DATE'] == '2017-01-20'

df_filtered = df[cond1 & cond2].copy()

# Select only 20 options to use for calibration (STRIKE around UNDERLYING_LAST)
cond3 = df_filtered['STRIKE_DISTANCE'] <= 1200.0

df_filtered = df_filtered[cond3]

# Create a separate dataframe for call and put options
df_call = df_filtered[['QUOTE_DATE', 'UNDERLYING_LAST', 'EXPIRE_DATE', 'C_BID', 'C_ASK', 'STRIKE']].copy()
print(df_call)

df_put = df_filtered[['QUOTE_DATE', 'UNDERLYING_LAST', 'EXPIRE_DATE', 'P_BID', 'P_ASK', 'STRIKE']].copy()
print(df_put)
print(df_call.shape, df_put.shape)

# export to a CSV file
df_call.to_csv('../data/options_spx_call_2016.csv', index=False)
df_put.to_csv('../data/options_spx_put_2016.csv', index=False)
