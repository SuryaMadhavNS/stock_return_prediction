It is a model to predict 1-minute stock returns using high-frequency trade and quote data from the NSE (National Stock Exchange of India).
Given trade and quote data for 30+ companies from the NSE. The data contains market data related to these stocks.We would like to develop a model that performs a 1 minute stock return (ratio of time T+1 andtime T price of a stock) using the market data as features

You can consider the following data processing scheme: Aggregate the trade and quote data to
minute level and joined them:
Here are the fields in the aggregated data (1 to 10 are from trade data, 11 onwards from the
quote data)
1. num_trades = total trades that took place in the minute
2. o, h, l, c = OHLC price of the trade in a given minute
3. total_volume = no of shares traded in the minute
4. total_buy_cap = sum of the buy order capacity column for the given minute. (I am not sure
what the buy_order_capacity means in the original trade data)
5. total_sell_cap = sum of the sell order capacity column for the given minute. (I am not sure
what the sell_order _capacity means in the original trade data)
6. trade_period = 'O' for pre opening hours, 'T' for after closing hours, '-' for regular
trading time
7.weighted_price = weighted average price per minute. volume is used as weight
8. trade_imbalance_ratio = no of times buy side was aggresssor / no of times sell side was
aggressor in a minute (please correct if the terminology for the variable is wrong/doubtful)
9. trade_volume_imbalance_ratio = total volume of buy side aggressor / total volume of sell
side aggressor in a minute
10. order_cap_imbalnce_ratio = no of times sell_order_capacity was not equal to
sell_order_capacity / total trades in the minute
11. avg_spread = average(Ask price - bid price) in a minute
12.max_spread = max(Ask price - bid price) in a minute
13.min_spread = min(Ask price - bid price) in a minute
14. total_bid_size = sum of bid size in a minute
15. total_ask_size = sum of ask size
16. weighted_avg_bid_price = weighted average of the bid price over the bid size
17. weighted_avg_ask_price = weighted average of the ask price over the ask size

As this is a time-series prediction task, please split the train-test data appropriately. I used (sklearn) linear regression for simplicity
