from stock_agent import agent
import csv
import polars as pl
import glob
from os import path


def call_agent(company):
    result = agent.invoke({
        "messages": [{"role": "user", "content": f"Should I buy or sell {company}?"}]
    })
    recommendation = result["messages"][-1].content

    return recommendation

success_cnt = 0
fail_cnt = 0

for file in glob.glob("stock_price_csv/*.csv")[:5]:
    company = (path.basename(file).split("_")[2]).split(".")[0]

    stock_price_history = pl.read_csv(file)

    stock_price_begin = stock_price_history.select(pl.first("close")).item()
    stock_price_end = stock_price_history.select(pl.last("close")).item()

    recommendation = call_agent(company)

    print(f"{company} | {recommendation}")
    print(f"Begin: {stock_price_begin} | End: {stock_price_end}")
    if ("<Recommendation: SELL>" in recommendation) and (stock_price_begin > stock_price_end):
        print("Success")
        success_cnt = success_cnt + 1
    elif ("<Recommendation: BUY>" in recommendation) and (stock_price_begin < stock_price_end):
        print("Success")
        success_cnt = success_cnt + 1
    else:
        print("Fail")
        fail_cnt = fail_cnt + 1

print("Final results:")
print(success_cnt)
print(fail_cnt)



