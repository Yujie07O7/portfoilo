import numpy as np
import torch as T
import torch.nn.functional as F
from env.loader import Loader
from finta import TA
import pandas as pd

class PortfolioEnv:

    def __init__(self, start_date=None, end_date=None, action_scale=1, action_interpret='portfolio',
                 state_type='indicators', djia_year=2019, repeat=0):
        self.loader = Loader(djia_year=djia_year)
        self.historical_data = pd.read_csv("env/data/DJIA_2019/ticker_all.csv", index_col=0, parse_dates=True)
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
        self.macro_dim = self.macro_indicators.shape[1] if hasattr(self, "macro_indicators") else 0
        self.repeat= repeat
        self.repeat = repeat if hasattr(self, "repeat") else 0
        self.history_log = []  # 新增 log 記錄
        # 第一步驟
        self.freerate = 0
        self.windows = 30
        self.returns = []
    def action_shape(self):
        return 3
           
    def state_shape(self):
        if self.action_interpret == 'portfolio' and self.state_type == 'only prices':
            return (self.n_stocks,)
        if self.action_interpret == 'portfolio' and self.state_type == 'indicators':
            return (9 + self.macro_dim,)
        if self.action_interpret == 'transactions' and self.state_type == 'only prices':
            return (2 * self.n_stocks + 1,)
        if self.action_interpret == 'transactions' and self.state_type == 'indicators':
            return (5* self.n_stocks + 3 + self.macro_dim,)  
            

    def reset(self, start_date=None, end_date=None, initial_balance=1000000):
        index = self.historical_data.index.drop_duplicates()
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
        returns = self.historical_data.iloc[self.current_row][["DBCReturn(M)", "SHYReturn(M)", "SPYReturn(M)"]].tolist()
        return returns

    def get_state(self):
        if self.current_row >= len(self.historical_data):
            print(f"[Warning] current_row 超出範圍，自動設為最後一筆 index")
            self.current_row = len(self.historical_data) - 1
        if self.action_interpret == 'portfolio' and self.state_type == 'only prices':
            state = self.historical_data.iloc[self.current_row] [[
                'DBCReturn(M)', 'SHYReturn(M)', 'SPYReturn(M)',
                'DBCSTD', 'SHYSTD', 'SPYSTD',
                'SPY-SHYCOV', 'SHY-DBCCOV', 'DBC-SPYCOV'
            ]].tolist()
            return np.array(state)

        if self.action_interpret == 'portfolio' and self.state_type == 'indicators':
            state = self.historical_data.iloc[self.current_row] [[
                'DBCReturn(M)', 'SHYReturn(M)', 'SPYReturn(M)',
                'DBCSTD', 'SHYSTD', 'SPYSTD',
                'SPY-SHYCOV', 'SHY-DBCCOV', 'DBC-SPYCOV'
            ]].tolist()
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
        return self.historical_data.index[self.current_row]

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
        index = self.historical_data.index.drop_duplicates()
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
    def step(self, action, softmax=True, reward_mode="sharpe"):
        if softmax:
            action = F.softmax(T.tensor(action, dtype=T.float), -1).numpy()
        else:
            action = np.array(action)
        returns = self.get_returns()
        
        # 使用不同的 reward 模式
        # 將報酬率套用到每個資產配置比例上，模擬總報酬率
        #配置在各個資產的投資比例（action），乘上這些資產在這一期的報酬率（returns），並把結果加總起來，得到你整個投資組合的報酬率
        #就是E(rp)=w1*p1+w2*p2+w3*p3
        # reward 可以放大一點看得比較清楚
        if reward_mode == "sharpe":
            sharpe_ratio, portfolio_return = self._calculate_sharpe_ratio(action)
            print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            reward = sharpe_ratio * 10000
        elif reward_mode == "return":
            portfolio_return = np.dot(action, returns)
            reward = portfolio_return * 10000
        self.returns.append(reward)
        done = self.is_finished()

        last_wealth = self.wealth_history[-1]
        new_wealth = last_wealth * (1 + portfolio_return)
        self.wealth_history.append(new_wealth)

        print(f"日期: {self.get_date()}, 報酬率: {returns}, 配置: {action}")
        print(f"Reward: {reward:.2f}, Cumulative Return: {new_wealth - 1000000:.2f}")
        self.history_log.append({
            "round": self.repeat,
            "date": self.get_date(),
            "returns": returns,
            "sharpe_ratio": sharpe_ratio,
            "weights": action.tolist(),
            "reward": reward,
            "wealth": new_wealth
        })
        self.current_row += 1
        done = self.is_finished()
        return self.get_state(), reward, done, self.get_date(), new_wealth

    def _calculate_sharpe_ratio(self, action):
        row = self.historical_data.iloc[self.current_row]
        # 取出報酬率
        returns = row[["DBCReturn(M)", "SHYReturn(M)", "SPYReturn(M)"]].values
        # 取出標準差
        stds = row[["DBCSTD", "SHYSTD", "SPYSTD"]].values
        # 取出共變異數（上三角）
        cov = row[["SPY-SHYCOV", "SHY-DBCCOV", "DBC-SPYCOV"]].values
        cov_matrix = np.zeros((3, 3))
        np.fill_diagonal(cov_matrix, stds**2)
        cov_matrix[0, 1] = cov_matrix[1, 0] = cov[0]  # DBC-HSY
        cov_matrix[0, 2] = cov_matrix[2, 0] = cov[1]  # SHY-SPY
        cov_matrix[1, 2] = cov_matrix[2, 1] = cov[2]  # DBC-SPY
        # 計算 Sharpe Ratio
        portfolio_return = np.dot(action, returns)
        portfolio_variance = np.dot(action, np.dot(cov_matrix, action))
        portfolio_std = np.sqrt(portfolio_variance) + 1e-8
        sharpe_ratio = (portfolio_return - self.freerate) / portfolio_std
        return sharpe_ratio, portfolio_return


