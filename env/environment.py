import numpy as np
import torch as T
import torch.nn.functional as F
from env.loader import Loader
from finta import TA
import pandas as pd

class PortfolioEnv:

    def __init__(self, start_date=None, end_date=None, action_scale=1, action_interpret='transactions',
                 state_type='indicators', djia_year=2019):
        np.set_printoptions(suppress=True, precision=2)
        self.loader = Loader(djia_year=djia_year)
        self.historical_data = self.loader.load(start_date, end_date)
        for stock in self.historical_data:
            stock['MA20'] = TA.SMA(stock, 20)
            stock['MA50'] = TA.SMA(stock, 50)
            stock['MA200'] = TA.SMA(stock, 200)
            stock['ATR'] = TA.ATR(stock)
        self.n_stocks = len(self.historical_data)
        self.prices = np.zeros(self.n_stocks)
        self.shares = np.zeros(self.n_stocks).astype(np.int64)
        self.balance = 0
        self.current_row = 0
        self.end_row = 0
        self.action_scale = action_scale
        self.action_interpret = action_interpret
        self.state_type = state_type
        self.history_log = []
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
            return (6 * self.n_stocks + 1,)  


    def action_shape(self):
        if self.action_interpret == 'portfolio':
            return self.n_stocks + 1,
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

        return self.get_state()

    def get_prices(self):
        prices = np.array([stock['Close'][self.current_row] for stock in self.historical_data])
        print(f"當前價格: {prices}")  # Debug
        return prices


    def clean_state(self, state):
        state = np.array(state, dtype=np.float32)
        state = np.nan_to_num(state, nan=0.0, posinf=1e5, neginf=-1e5)
        return np.clip(state, -1e5, 1e5)

    def get_state(self):

        if self.action_interpret == 'portfolio' and self.state_type == 'only prices':
            state = self.prices.tolist()
            return self.clean_state(state)

        if self.action_interpret == 'portfolio' and self.state_type == 'indicators':
            state = []
            for stock in self.historical_data:
                state.extend(stock[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[self.current_row])
            return self.clean_state(state)
        
        if self.action_interpret == 'transactions' and self.state_type == 'only prices':
            state = [self.balance] + self.prices.tolist() + self.shares.tolist()
            return self.clean_state(state)
        
        if self.action_interpret == 'transactions' and self.state_type == 'indicators':
            state = [self.balance] + self.shares.tolist()
            for stock in self.historical_data:
                state.extend(stock[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[self.current_row])
            return self.clean_state(state)

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

    def buy_hold_history(self, start_date=None, end_date=None):
        if start_date is None:
            start_row = 0
        else:
            start_row = self.historical_data[0].index.get_loc(start_date)
        if end_date is None:
            end_row = self.historical_data[0].index.size - 1
        else:
            end_row = self.historical_data[0].index.get_loc(end_date)
        
        values = [sum([stock['Close'][row] for stock in self.historical_data])
                  for row in range(start_row, end_row + 1)]
        dates = self.historical_data[0].index[start_row:end_row+1]

        return pd.Series(values, index=dates)

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

    # 第二步驟
    def step(self, action, softmax=True):
        previous_wealth = self.get_wealth()
        previous_sharpe = self._calculate_sharpe_ratio()

        if self.action_interpret == 'portfolio':
            if softmax:
                action = F.softmax(T.tensor(action, dtype=T.float), -1).numpy()
            else:
                action = np.array(action)

            new_shares = np.floor(previous_wealth * action[1:] / self.prices)
            actions = new_shares - self.shares
            cost = self.prices.dot(actions)

            self.shares = self.shares + actions.astype(np.int64)
            self.balance -= cost

        #這邊可能要換成算出投資組合的權重加總=1加總=1
        elif self.action_interpret == 'transactions':
            actions = np.clip(action, -1, +1)
            positive = actions > 0
            negative = actions < 0
            actions[negative] = np.ceil(actions[negative] * self.shares[negative])
            if np.sum(self.prices[positive] * actions[positive]) > 0:
                k = (self.balance - np.sum(self.prices[negative] * actions[negative])) 
                np.sum(self.prices[positive] * actions[positive])
            else:   
                k = 0
            actions[positive] = np.floor(actions[positive] * k)

            cost = self.prices.dot(actions)
            self.shares = self.shares + actions
            #balance是手上剩多少資金
            self.balance -= cost

        # 更新價格與資產
        self.current_row += 1
        new_prices = self.get_prices()
        self.prices = new_prices

        new_wealth = self.get_wealth()
        portfolio_return = (new_wealth - previous_wealth) / previous_wealth
        self.returns.append(portfolio_return)

        sharpe_ratio = self._calculate_sharpe_ratio()
        cumulative_return = new_wealth - 1000000
        reward = (sharpe_ratio - previous_sharpe) + (new_wealth - previous_wealth) * 100

        # Debug print
        print(f"日期: {self.get_date()}, 當前價格: {self.prices}")
        print(f"持股: {self.shares}, 資金: {self.balance}, 總資產: {new_wealth}")
        print(f"Sharpe Ratio: {sharpe_ratio}, Reward: {reward}, Cumulative Return: {cumulative_return}")
        # 儲存每個 step 的歷史資料
        self.history_log.append({
            'date': self.get_date(),
            'balance': self.balance,
            'wealth': new_wealth,
            'shares': self.shares.copy(),
            'prices': self.prices.copy(),
            'sharpe_ratio': sharpe_ratio,
            'reward': reward
        })
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

