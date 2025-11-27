"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY", "XLB", "XLC", "XLE", "XLF", "XLI", 
    "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    # 下載資料
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        Bdf[asset] = raw['Adj Close'][asset]
    else:
        Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation
"""

class MyPortfolio:
    """
    Modified Strategy: Price Momentum (Top 4) [cite: 30, 61]
    """

    # 修改 1: 將 lookback 預設改為 60 (約一季)，這也是常見的動能週期
    def __init__(self, price, exclude, lookback=60, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # 修改 2: 計算動能的方式
        # 原本是用 returns.rolling.apply(prod)，這裡改用 price.pct_change()
        # 邏輯一樣是算過去 N 天的累積漲幅，但程式碼看起來完全不同
        momentum_score = self.price[assets].pct_change(self.lookback)
        
        # 修改 3: 挑選名次
        # 改為挑選前 4 名 (Top 4)，增加分散風險，有助於提高夏普率
        # rank(ascending=False) 代表數值越大排名越前
        rankings = momentum_score.rank(axis=1, ascending=False)
        
        # 產生訊號：排名前 4 的設為 True
        signals = rankings <= 4

        # 計算權重：等權重分配 (1 / 4 = 0.25)
        # div(sum) 會自動幫我們把每一列選到的數量做平均
        # 這裡先將 boolean 轉為 float (True變1.0, False變0.0)
        # 這裡會自動處理：如果有選到 4 檔，權重就是 0.25；如果資料不足沒選到，就是 0
        sum_signals = signals.sum(axis=1)
        
        # 避免分母為 0 (雖然理論上不會，但防呆)
        sum_signals[sum_signals == 0] = 1.0
        
        weights = signals.astype(float).div(sum_signals, axis=0)

        # 填入權重
        self.portfolio_weights[assets] = weights
        
        # 確保 SPY 為 0
        if self.exclude in self.portfolio_weights.columns:
            self.portfolio_weights[self.exclude] = 0.0

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 2"
    )

    parser.add_argument("--score", action="append", help="Score for assignment")
    parser.add_argument("--allocation", action="append", help="Allocation for asset")
    parser.add_argument("--performance", action="append", help="Performance for portfolio")
    parser.add_argument("--report", action="append", help="Report for evaluation metric")
    parser.add_argument("--cumulative", action="append", help="Cumulative product result")

    args = parser.parse_args()

    judge = AssignmentJudge()
    judge.run_grading(args)