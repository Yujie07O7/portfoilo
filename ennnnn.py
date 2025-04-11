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
            return (5 * self.n_stocks,)
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
        self.weight_history = []
        if start_date is None:
            self.current_row = 0
        else:
            self.current_row = self.historical_data[0].index.get_loc(start_date)
        if end_date is None:
            self.end_row = self.historical_data[0].index.size - 1
        else:
            self.end_row = self.historical_data[0].index.get_loc(end_date)
        self.prices = self.get_prices()
        self.shares = np.zeros(self.n_stocks).astype(np.int64)
        self.balance = initial_balance
        self.wealth_history = [self.get_wealth()]

        return self.get_state()

    def get_returns(self):
        returns = np.array([stock['Return'].iloc[self.current_row] for stock in self.historical_data])
        return returns

    def get_state(self):

        if self.action_interpret == 'portfolio' and self.state_type == 'only prices':
            return self.prices.tolist()

        if self.action_interpret == 'portfolio' and self.state_type == 'indicators':
            state = []
            for stock in self.historical_data:
                state.extend(stock[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[self.current_row])
            # 加入景氣變數
            if hasattr(self, 'macro_indicators'):
                state.extend(self.macro_indicators[self.current_row])
            return np.array(state)
        
        if self.action_interpret == 'transactions' and self.state_type == 'only prices':
            return [self.balance] + self.prices.tolist() + self.shares.tolist()
        
        if self.action_interpret == 'transactions' and self.state_type == 'indicators':
            state = [self.balance] + self.shares.tolist()
            for stock in self.historical_data:
                state.extend(stock[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[self.current_row])
            if hasattr(self, 'macro_indicators'):
                state.extend(self.macro_indicators[self.current_row])
            return np.array(state)


    def is_finished(self):
        return self.current_row == self.end_row

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
        index = self.historical_data[0].index

        if self.state_type == 'only prices':
            size = len(index)
            train_begin = 0
            train_end = int(np.round(train_ratio * size - 1))
            valid_begin = train_end + 1
            valid_end = valid_begin + int(np.round(valid_ratio * size - 1))
            test_begin = valid_end + 1
            test_end = -1
        
        if self.state_type == 'indicators':
            size = len(index) - 199
            train_begin = 199
            train_end = train_begin + int(np.round(train_ratio * size - 1))
            valid_begin = train_end + 1
            valid_end = valid_begin + int(np.round(valid_ratio * size - 1))
            test_begin = valid_end + 1
            test_end = -1
        
        intervals = {'training': (index[train_begin], index[train_end]),
             'validation': (index[valid_begin], index[valid_end]),
             'testing': (index[test_begin], index[test_end])}

        return intervals

    # 第二步驟，用新的把它改成return把它改成return
    def step(self, action, softmax=True):

        if self.action_interpret == 'portfolio':
            current_wealth = self.get_wealth()
            if softmax:
                action = F.softmax(T.tensor(action, dtype=T.float), -1).numpy()
            else:
                action = np.array(action)
            new_shares = np.floor(current_wealth * action / self.prices)
            actions = new_shares - self.shares
            cost = self.prices.dot(actions)
            self.wealth_history.append(self.get_wealth())  # 每一步累積資產記錄
            self.shares = self.shares + actions.astype(np.int64)
            #balance是手上剩多少資金
            self.balance -= cost
            self.current_row += 1
            new_prices = self.get_prices()
            portfolio_value_before = np.sum(self.prices * self.shares)
            portfolio_value_after = np.sum(new_prices * self.shares)
            reward = (portfolio_value_after - portfolio_value_before) / (portfolio_value_before)*10000 # 加1e-8避免除0
            self.prices = new_prices
            new_wealth = self.get_wealth()
            cumulative_return = new_wealth - 1000000

        #這邊可能要換成算出投資組合的權重加總=1
        elif self.action_interpret == 'transactions':
            actions = np.maximum(np.round(np.array(action) * self.action_scale), -self.shares)
            cost = self.prices.dot(actions)
            if cost > self.balance:
                actions = np.floor(actions * self.balance / cost)
                cost = self.prices.dot(actions)
            self.shares = self.shares + actions.astype(np.int64) 
            #balance是手上剩多少資金
            self.balance -= cost
            self.current_row += 1
            new_prices = self.get_prices()
            
             # 📌 計算 portfolio value：包含股票市值變化前後
            portfolio_value_before = np.sum(self.prices * self.shares)
            portfolio_value_after = np.sum(new_prices * self.shares)
            #這邊*10000是因為要放大reward的數值
            reward = (portfolio_value_after - portfolio_value_before) / (portfolio_value_before)*10000
            self.returns.append(reward)
            self.prices = new_prices
            new_wealth = self.get_wealth()
            cumulative_return = new_wealth - 1000000
        # 數值穩定性保護（避免爆掉），也在這邊加入持股限制
        self.balance = np.clip(self.balance, -1e6, 1e6)
        self.shares = np.clip(self.shares, 2000, 30000)

        # Debug print
        print(f"日期: {self.get_date()}, 當前價格: {self.prices}")
        print(f"持股: {self.shares}, 資金: {self.balance}, 總資產: {new_wealth}")
        print(f"Reward: {reward}, Cumulative Return: {cumulative_return}")
        print(f"投資比例（權重）: {self.get_weights()}")
        return self.get_state(), reward, self.is_finished(), self.get_date(), self.get_wealth()
    def _calculate_sharpe_ratio(self, window_size=30):
        min_window = min(len(self.returns), window_size)
        if min_window < 5:  
            return 0
        recent_returns = np.array(self.returns[-min_window:])
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns) + 1e-8  # 防止除以 0
        sharpe_ratio = (mean_return - self.freerate) / std_return
        return sharpe_ratio