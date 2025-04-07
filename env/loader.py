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
    def load_marco_data(self, start_date=None, end_date=None):
        macro = pd.read_csv(f'env/data/DJIA_2019/business.csv', parse_dates=True, index_col='date')
        macro = macro.fillna(method='ffill').fillna(method='bfill')

        # 若 start/end 沒有給，就從股票資料自動推斷
        if start_date is None or end_date is None:
            if len(self.stocks) > 0:
                stock_index = self.stocks[0].index
                if start_date is None:
                    start_date = stock_index[0]
                if end_date is None:
                    end_date = stock_index[-1]
            else:
                raise ValueError("start_date and end_date are None, and no stock data to infer from.")
            macro = macro[(macro.index >= pd.to_datetime(start_date)) & (macro.index <= pd.to_datetime(end_date))]
                    # ✅ 對齊股票時間（可選）
            if len(self.stocks) > 0:
                stock_index = self.stocks[0].index
                macro = macro.loc[macro.index.intersection(stock_index)]
            print(macro.iloc[0])
            print(macro.iloc[-1])
            return macro.values

    def load(self, download=False, start_date=None, end_date=None):
        if download:
            self.download_data(start_date, end_date)
        else:
            self.read_data()
        return self.stocks

