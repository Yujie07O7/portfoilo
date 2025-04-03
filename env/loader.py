import yfinance as yf
import pandas as pd
import os

class Loader:
    def __init__(self, djia_year):
        self.djia_year = djia_year
        file_path = os.path.join(os.path.dirname(__file__), f'data/DJIA_{djia_year}/tickers.txt')

        with open(file_path, 'r') as file:
            self.tickers = [line.strip() for line in file.readlines()]  # 讀取 tickers.txt

        print("Tickers loaded:", self.tickers)  # 測試是否正確載入
        self.stocks = []
        # print(self.stocks)

    def download_data(self, start_date, end_date=None):
        for ticker in self.tickers:
            print(f"Downloading data for: {ticker}")
            data = yf.download(ticker, start=start_date, end=end_date)
            data['Ticker'] = ticker  # 確保數據包含股票代號
            self.stocks.append(data)
            data.to_csv(f'env/data/DJIA_2019/ticker_{ticker}.csv')  # 存到 env/data/

    def read_data(self):
        for ticker in self.tickers:
            file_path = f'env/data/DJIA_2019/ticker_{ticker}.csv'
            try:
                data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
                data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
                data['Ticker'] = ticker  # 確保數據包含股票代號
                self.stocks.append(data)
            except FileNotFoundError:
                print(f"Warning: Data file for {ticker} not found, skipping.")

    def load(self, download=False, start_date=None, end_date=None):
        if download:
            self.download_data(start_date, end_date)
        else:
            self.read_data()
        return self.stocks

