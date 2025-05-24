import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Aggregate trade data by minute
def aggregate_trade_data(trade_df):
    trade_df = trade_df.copy()
    # Use the existing Datetime column to create the minute column
    trade_df['minute'] = trade_df['Datetime'].dt.floor('min')
    
    # Initial aggregation
    grouped = trade_df.groupby(['company', 'minute']).agg({
        'price': ['count', 'first', 'max', 'min', 'last'],  # num_trades, o, h, l, c
        'volume': 'sum',  # total_volume
        'buy_order_capacity': 'sum',  # total_buy_cap
        'sell_order_capacity': 'sum',  # total_sell_cap
        'trade_period': 'first',  # trade_period
    }).reset_index()

    # Flatten the column names
    grouped.columns = ['company', 'minute', 'num_trades', 'o', 'h', 'l', 'c', 'total_volume', 
                       'total_buy_cap', 'total_sell_cap', 'trade_period']

    # Compute weighted price
    temp_df = trade_df[['company', 'minute', 'price', 'volume']].copy()
    temp_df = temp_df.groupby(['company', 'minute']).apply(
        lambda x: np.average(x['price'], weights=x['volume']) if x['volume'].sum() > 0 else np.nan
    ).reset_index(name='weighted_price')
    grouped = pd.merge(grouped, temp_df, on=['company', 'minute'], how='left')

    # Compute trade imbalance ratio
    temp_df = trade_df[['company', 'minute', 'aggressor']].copy()
    temp_df = temp_df.groupby(['company', 'minute']).agg(
        lambda x: (x == 'buy').sum() / (x == 'sell').sum() if (x == 'sell').sum() != 0 else np.nan
    ).reset_index(name='trade_imbalance_ratio')
    grouped = pd.merge(grouped, temp_df, on=['company', 'minute'], how='left')

    # Compute trade volume imbalance ratio
    temp_df = trade_df[['company', 'minute', 'volume', 'aggressor']].copy()
    temp_df = temp_df.groupby(['company', 'minute']).apply(
        lambda x: x[x['aggressor'] == 'buy']['volume'].sum() / x[x['aggressor'] == 'sell']['volume'].sum()
        if x[x['aggressor'] == 'sell']['volume'].sum() != 0 else np.nan
    ).reset_index(name='trade_volume_imbalance_ratio')
    grouped = pd.merge(grouped, temp_df, on=['company', 'minute'], how='left')

    # Compute order capacity imbalance ratio
    temp_df = trade_df[['company', 'minute', 'sell_order_capacity', 'buy_order_capacity']].copy()
    temp_df = temp_df.groupby(['company', 'minute']).apply(
        lambda x: ((x['sell_order_capacity'] != x['buy_order_capacity']).sum() / len(x))
        if len(x) > 0 else np.nan
    ).reset_index(name='order_cap_imbalance_ratio')
    grouped = pd.merge(grouped, temp_df, on=['company', 'minute'], how='left')

    return grouped

# Aggregate quote data by minute
def aggregate_quote_data(quote_df):
    quote_df = quote_df.copy()
    # Use the existing Datetime column to create the minute column
    quote_df['minute'] = quote_df['Datetime'].dt.floor('min')
    quote_df['spread'] = quote_df['ask_price'] - quote_df['bid_price']
    
    # Initial aggregation
    grouped = quote_df.groupby(['company', 'minute']).agg({
        'spread': ['mean', 'max', 'min'],  # avg_spread, max_spread, min_spread
        'bid_size': 'sum',  # total_bid_size
        'ask_size': 'sum',  # total_ask_size
    }).reset_index()

    # Flatten the column names
    grouped.columns = ['company', 'minute', 'avg_spread', 'max_spread', 'min_spread', 
                       'total_bid_size', 'total_ask_size']

    # Compute weighted average bid price
    temp_df = quote_df[['company', 'minute', 'bid_price', 'bid_size']].copy()
    temp_df = temp_df.groupby(['company', 'minute']).apply(
        lambda x: np.average(x['bid_price'], weights=x['bid_size']) if x['bid_size'].sum() > 0 else np.nan
    ).reset_index(name='weighted_avg_bid_price')
    grouped = pd.merge(grouped, temp_df, on=['company', 'minute'], how='left')

    # Compute weighted average ask price
    temp_df = quote_df[['company', 'minute', 'ask_price', 'ask_size']].copy()
    temp_df = temp_df.groupby(['company', 'minute']).apply(
        lambda x: np.average(x['ask_price'], weights=x['ask_size']) if x['ask_size'].sum() > 0 else np.nan
    ).reset_index(name='weighted_avg_ask_price')
    grouped = pd.merge(grouped, temp_df, on=['company', 'minute'], how='left')

    return grouped

# Merge aggregated trade and quote data
def merge_data(trade_df, quote_df):
    trade_agg = aggregate_trade_data(trade_df)
    quote_agg = aggregate_quote_data(quote_df)
    return pd.merge(trade_agg, quote_agg, on=['company', 'minute'], how='inner')

# Create target variable for stock return
def create_target(df):
    df = df.sort_values(['company', 'minute'])
    df['next_c'] = df.groupby('company')['c'].shift(-1)
    df['stock_return'] = df['next_c'] / df['c']
    return df.dropna(subset=['stock_return'])

# Preprocess the data for modeling
def preprocess_data(df):
    df = df.copy()
    
    # One-hot encode trade_period if it exists
    if 'trade_period' in df.columns:
        df = pd.get_dummies(df, columns=['trade_period'], prefix='trade_period', dummy_na=False)
    else:
        print("Warning: 'trade_period' column not found. Skipping one-hot encoding.")
    
    # Define numeric columns for scaling
    numeric_cols = ['num_trades', 'o', 'h', 'l', 'c', 'total_volume', 'total_buy_cap', 'total_sell_cap',
                    'weighted_price', 'trade_imbalance_ratio', 'trade_volume_imbalance_ratio', 
                    'order_cap_imbalance_ratio', 'avg_spread', 'max_spread', 'min_spread', 
                    'total_bid_size', 'total_ask_size', 'weighted_avg_bid_price', 'weighted_avg_ask_price']
    
    # Ensure all numeric columns exist in the DataFrame, fill missing ones with NaN
    for col in numeric_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Filling with NaN.")
            df[col] = np.nan
    
    # Fill missing values with the mean of each column
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Scale numeric features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Replace infinite values with NaN
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Fill any new NaNs (from infinite values) with the mean again
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    print("Preprocessing completed successfully.")
    return df