from stock_agent import agent
import csv
import polars as pl
import glob
from os import path


# result = agent.invoke({
#     "messages": [{"role": "user", "content": "Should I buy or sell AAPL?"}]
# })
# print(result["messages"][-1].content)

for file in glob.glob("stock_price_csv/*.csv"):
    print(path.basename(file))

stock_price_history = pl.read_csv('stock_price_csv/csv_test_AAPL.csv')

stock_price_begin = stock_price_history.select(pl.first("close"))
stock_price_end = stock_price_history.select(pl.last("close"))

print(stock_price_begin > stock_price_end)
