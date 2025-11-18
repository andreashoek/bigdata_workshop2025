from stock_agent import agent
import csv
import polars as pl
import glob
from os import path
import pytest


def call_agent(company):
    result = agent.invoke({
        "messages": [{"role": "user", "content": f"Should I buy or sell {company}?"}]
    })
    recommendation = result["messages"][-1].content

    return recommendation


@pytest.mark.parametrize("file", glob.glob("stock_price_csv/*.csv")[:15])
def test_company_recommendation(file):
    company = (path.basename(file).split("_")[2]).split(".")[0]

    stock_price_history = pl.read_csv(file)

    stock_price_begin = stock_price_history.select(pl.first("close")).item()
    stock_price_end = stock_price_history.select(pl.last("close")).item()

    recommendation = call_agent(company)

    print(f"{company} | {recommendation}")
    print(f"Begin: {stock_price_begin} | End: {stock_price_end}")

    if stock_price_begin > stock_price_end:
        assert "<Recommendation: SELL>" in recommendation, f"Expected SELL for {company} (price decreased)"
    elif stock_price_begin < stock_price_end:
        assert "<Recommendation: BUY>" in recommendation, f"Expected BUY for {company} (price increased)"



