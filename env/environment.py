import numpy as np
import torch as T
import torch.nn.functional as F
from env.loader import Loader
from finta import TA
import pandas as pd

class PortfolioEnv:

    def __init__(self, start_date=None, end_date=None, action_scale=1, action_interpret='portfolio',
                 state_type='indicators', djia_year=2019):
        self.loader = Loader(djia_year=djia_year)
        self.historical_data = self.loader.load(start_date, end_date)
        self.marco_indicators = self.loader.load_marco_data(start_date, end_date)
        self.n_stocks = len(self.historical_data)
        self.prices = np.zeros(self.n_stocks)
        self.shares = np.zeros(self.n_stocks).astype(np.int64)
        self.balance = 0
        self.current_row = 0
        self.end_row = 0
        self.action_scale = action_scale
        self.action_interpret = action_interpret
        self.state_type = state_type
        self.macro_dim = self.marco_indicators.shape[1]


        # 第一步驟
        self.freerate = 0
        self.windows = 30
        self.returns = []
        
    def state_shape(self):
        if self.action_interpret == 'portfolio' and self.state_type == 'only prices':
            return (self.n_stocks,)
        if self.action_interpret == 'portfolio' and self.state_type == 'indicators':
            return (2 * self.n_stocks,)
        if self.action_interpret == 'transactions' and self.state_type == 'only prices':
            return (2 * self.n_stocks + 1,)
        if self.action_interpret == 'transactions' and self.state_type == 'indicators':
            return (5* self.n_stocks + 3 + self.macro_dim,)  
            
    def action_shape(self):
        if self.action_interpret == 'portfolio':
            return self.n_stocks,
        if self.action_interpret == 'transactions':
            return self.n_stocks,

    def reset(self, start_date=None, end_date=None, initial_balance=1000000):
        index = self.historical_data[0].index
        index = index.drop_duplicates() 
        self.weight_history = []

        if start_date is None:
            self.current_row = 0
        else:
            self.current_row = index.get_indexer([start_date])[0]

        if end_date is None:
            self.end_row = index.size - 1
        else:
            self.end_row = index.get_indexer([end_date])[0]

        self.shares = np.zeros(self.n_stocks).astype(np.int64)
        self.balance = initial_balance
        self.wealth_history = [self.get_wealth()]
        print("current_row:", self.current_row, type(self.current_row))
        print("end_row:", self.end_row, type(self.end_row))
        return self.get_state()

    def get_returns(self):
        returns = np.array([stock['Return'].iloc[self.current_row] for stock in self.historical_data])
        return returns

    def get_state(self):
        if self.current_row >= len(self.historical_data[0]):
            print(f"[Warning] current_row 超出範圍，自動設為最後一筆 index")
            self.current_row = len(self.historical_data[0]) - 1
        if self.action_interpret == 'portfolio' and self.state_type == 'only prices':
            return self.prices.tolist()

        if self.action_interpret == 'portfolio' and self.state_type == 'indicators':
            state = []
            for stock in self.historical_data:
                max_idx = len(stock) - 1
                safe_row = min(self.current_row, max_idx)
                state.extend(stock[['Return', 'STD']].iloc[safe_row])
            # 加入景氣變數
            if hasattr(self, 'macro_indicators'):
                state.extend(self.macro_indicators[self.current_row])
            return np.array(state)
        
        if self.action_interpret == 'transactions' and self.state_type == 'only prices':
            return [self.balance] + self.prices.tolist() + self.shares.tolist()
        
        if self.action_interpret == 'transactions' and self.state_type == 'indicators':
            state = [self.balance] + self.shares.tolist()
            for stock in self.historical_data:
                state.extend(stock[['Return', 'STD']].iloc[self.current_row])
            if hasattr(self, 'macro_indicators'):
                state.extend(self.macro_indicators[self.current_row])
            return np.array(state)

    def is_finished(self):
        return bool(self.current_row >= self.end_row)

    def get_date(self):
        return self.historical_data[0].index[self.current_row]

    def get_wealth(self):
        return self.prices.dot(self.shares) + self.balance
    
    def get_balance(self):
        return self.balance
    
    def get_shares(self):
        return self.shares

    def get_weights(self):
        total_value = self.get_wealth()
        asset_values = self.prices * self.shares
        weights = asset_values / (total_value + 1e-8)
        return weights

    def get_intervals(self, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
        index = self.historical_data[0].index.drop_duplicates()
        size = len(index)

        train_begin = 0
        train_end = int(np.round(train_ratio * size - 1))
        valid_begin = train_end + 1
        valid_end = valid_begin + int(np.round(valid_ratio * size - 1))
        test_begin = valid_end + 1
        test_end = test_begin + int(np.round(test_ratio * size - 1))

        # 保底修正，避免超過 index 長度導致 NaT
        train_end = min(train_end, size - 1)
        valid_end = min(valid_end, size - 1)
        test_end = min(test_end, size - 1)

        intervals = {
            'training': (index[train_begin], index[train_end]),
            'validation': (index[valid_begin], index[valid_end]),
            'testing': (index[test_begin], index[test_end])
        }
        return intervals


    # 第二步驟
    def step(self, action, softmax=True):
        if softmax:
            action = F.softmax(T.tensor(action, dtype=T.float), -1).numpy()
        else:
            action = np.array(action)

        # 獲取當前報酬率
        returns = self.get_returns()

        # 將報酬率套用到每個資產配置比例上，模擬總報酬率
        portfolio_return = np.dot(action, returns)

        # reward 可以放大一點看得比較清楚
        reward = portfolio_return * 10000

        self.returns.append(reward)
        self.current_row += 1
        done = self.is_finished()

        # wealth 模擬：假設初始 100 萬，每次根據報酬率累積
        last_wealth = self.wealth_history[-1]
        new_wealth = last_wealth * (1 + portfolio_return)
        self.wealth_history.append(new_wealth)

        print(f"日期: {self.get_date()}, 報酬率: {returns}, 配置: {action}")
        print(f"Reward: {reward:.2f}, Cumulative Return: {new_wealth - 1000000:.2f}")
        
        if self.current_row >= self.end_row:
            done = True
            return self.get_state(), reward, done, self.get_date(), new_wealth

        self.current_row += 1
        done = self.is_finished()
        return self.get_state(), reward, done, self.get_date(), new_wealth

    def _calculate_sharpe_ratio(self, window_size=30):
        min_window = min(len(self.returns), window_size)
        if min_window < 5:  
            return 0
        recent_returns = np.array(self.returns[-min_window:])
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns) + 1e-8  # 防止除以 0
        sharpe_ratio = (mean_return - self.freerate) / std_return
        return sharpe_ratio

