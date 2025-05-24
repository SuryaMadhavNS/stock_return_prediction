import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

def split_data(df, train_ratio=0.8):
    df = df.sort_values(['company', 'minute'])
    companies = df['company'].unique()
    train_dfs, test_dfs = [], []
    
    for company in companies:
        company_df = df[df['company'] == company]
        n = len(company_df)
        train_size = int(n * train_ratio)
        train_dfs.append(company_df.iloc[:train_size])
        test_dfs.append(company_df.iloc[train_size:])
    
    return pd.concat(train_dfs), pd.concat(test_dfs)

def train_model(train_df, test_df):
    features = [col for col in train_df.columns if col not in ['company', 'minute', 'stock_return', 'next_c']]
    X_train = train_df[features]
    y_train = train_df['stock_return']
    X_test = test_df[features]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    test_df['predicted_return'] = model.predict(X_test)
    return test_df

def compute_correlations(test_df):
    correlations = []
    minutes = test_df['minute'].unique()
    
    for minute in sorted(minutes):
        minute_data = test_df[test_df['minute'] == minute]
        if len(minute_data) > 1:
            corr, _ = pearsonr(minute_data['stock_return'], minute_data['predicted_return'])
            correlations.append({'minute': minute, 'correlation': corr})
    
    corr_df = pd.DataFrame(correlations)
    market_open = pd.to_datetime('2025-05-23 09:15:00')
    corr_df['minutes_since_open'] = (corr_df['minute'] - market_open).dt.total_seconds() / 60
    return corr_df