import yfinance as yf
import pandas as pd
import os

class Loader:
    def __init__(self, djia_year=None, start_date=None, end_date=None):
        self.djia_year = djia_year
        self.start_date = start_date
        self.end_date = end_date
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
        start_dates = []
        end_dates = []
        for ticker in self.tickers:
            file_path = f'env/data/DJIA_2019/ticker_{ticker}.csv'
            try:
                data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
                data = data[['DBCReturn(M)', 'SHYReturn(M)', 'SPYReturn(M)', 'DBCSTD', 'SHYSTD', 'SPYSTD', 'SPY-SHYCCOV', 'SHY-DBCCOV', 'DBC-SPYCOV']]
                data['Ticker'] = ticker
                self.stocks.append(data)

                # 加入時間範圍推斷
                start_dates.append(data.index.min())
                end_dates.append(data.index.max())
                print(f"{ticker} loaded | 日期範圍：{data.index.min().date()} ~ {data.index.max().date()} | 筆數：{len(data)}")
                

            except FileNotFoundError:
                print(f"Warning: Data file for {ticker} not found, skipping.")

        # 推斷全體股票的共同區間
        if start_dates and end_dates:
            self.start_date = max(start_dates)
            self.end_date = min(end_dates)
        else:
            self.start_date = None
            self.end_date = None


        
    def load_marco_data(self, start_date=None, end_date=None):
        macro = pd.read_csv('env/data/DJIA_2019/business.csv', parse_dates=True, index_col='date')
        macro = macro.fillna(method='ffill').fillna(method='bfill')
        macro.index = pd.to_datetime(macro.index).normalize()

        # 若沒給日期，就從股票資料推斷
        if start_date is None:
            start_date = self.start_date
            start_date = pd.to_datetime("2006-04-30")
        if end_date is None:
            end_date = self.end_date
            end_date = pd.to_datetime("2025-4-30")
        # ✅ 加入條件檢查，避免 NaT 呼叫 normalize()
        if pd.notna(start_date):
            start_date = pd.to_datetime(start_date).normalize()
        if pd.notna(end_date):
            end_date = pd.to_datetime(end_date).normalize()
        macro.index = pd.to_datetime(macro.index).normalize()
        macro = macro[(macro.index >= start_date) & (macro.index <= end_date)]

        if len(self.stocks) > 0:
            stock_index = self.stocks[0].index.normalize()
            macro = macro.loc[macro.index.intersection(stock_index)]

        if macro.empty:
            raise ValueError("❌ macro data 為空，請確認 business.csv 是否正確、且涵蓋指定時間範圍")

        print("✅ Macro data loaded. 起始:", macro.index[0], "結束:", macro.index[-1])
        return macro.values


    def load(self, download=False, start_date=None, end_date=None):
        if download:
            self.download_data(start_date, end_date)
        else:
            self.read_data()
        return self.stocks

