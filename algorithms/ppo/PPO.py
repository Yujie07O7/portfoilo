
from env.environment import PortfolioEnv
from algorithms.ppo.agent import Agent
from plot import add_curve, add_hline, save_plot
import os
import pandas as pd
import matplotlib.pyplot as plt
from pyfolio import timeseries

class PPO:

    def __init__(self, load=False, alpha=0.0005, n_epochs=30,
                 batch_size=64, layer1_size=1024, layer2_size=1024, policy_clip=0.3, t_max=256,gamma=0.95, gae_lambda=0.99,
                state_type='only prices', djia_year=2019, repeat=0, entropy=0):
        self.figure_dir = 'plots/ppo'
        self.checkpoint_dir = 'checkpoints/ppo'
        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.t_max = t_max
        self.repeat = repeat
        self.env = PortfolioEnv(action_scale=1000, state_type=state_type, djia_year=djia_year, repeat=repeat)
        if djia_year == 2019:
            self.intervals = self.env.get_intervals(train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15)
        self.agent = Agent(action_dims=self.env.action_shape(), batch_size=batch_size, alpha=alpha,
                           n_epochs=n_epochs, input_dims=self.env.state_shape(),
                           fc1_dims=layer1_size, fc2_dims=layer2_size, entropy=entropy)
        if load:
            self.agent.load_models(self.checkpoint_dir)

    def train(self, verbose=False):
        training_history = []
        validation_history = []
        iteration = 1
        max_wealth = 0

        while True:
            n_steps = 0
            observation = self.env.reset(*self.intervals['training'])
            done = False
            while not done:
                action, prob, val = self.agent.choose_action(observation)
                observation_, reward, done, info, wealth = self.env.step(action)
                n_steps += 1
                self.env.wealth_history.append(self.env.get_wealth())
                self.agent.remember(observation, action, prob, val, reward, done)
                if n_steps % self.t_max == 0:
                    self.agent.learn()
                observation = observation_

            self.agent.memory.clear_memory()

            print(f"PPO training - Iteration: {iteration},\tCumulative Return: {int(wealth) - 1000000}")
            training_history.append(wealth - 1000000)

            validation_wealth = self.validate(verbose)
            print(f"PPO validating - Iteration: {iteration},\tCumulative Return: {int(validation_wealth) - 1000000}")
            validation_history.append(validation_wealth - 1000000)
            if validation_wealth > max_wealth:
                self.agent.save_models(self.checkpoint_dir)
            max_wealth = max(max_wealth, validation_wealth)
            if validation_history[-5:].count(max_wealth - 1000000) != 1:
                break
            iteration += 1
        print(f"總共訓練了 {iteration} 次迭代")

        self.agent.load_models(self.checkpoint_dir)
        add_curve(training_history, 'PPO')
        save_plot(filename=self.figure_dir + f'/{self.repeat}0_training.png',
                  title=f"Training - {self.intervals['training'][0].date()} to {self.intervals['training'][1].date()}",
                  x_label='Iteration', y_label='Cumulative Return (Dollars)')
        
        add_curve(validation_history, 'PPO')
        save_plot(filename=self.figure_dir + f'/{self.repeat}1_validation.png',
                  title=f"Validation - {self.intervals['validation'][0].date()} to {self.intervals['validation'][1].date()}",
                  x_label='Iteration', y_label='Cumulative Return (Dollars)')
        

    def validate(self, verbose=False):
        observation = self.env.reset(*self.intervals['validation'])
        done = False
        while not done:
            action, prob, val = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            observation = observation_
        return wealth
    
    def test(self, verbose=1):
        return_history = [0]
        n_steps = 0

        observation = self.env.reset(*self.intervals['testing'])
        wealth_history = [self.env.get_wealth()]
        done = False
        while not done:
            action, prob, val = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            n_steps += 1
            self.agent.remember(observation, action, prob, val, reward, done)
            if n_steps % self.t_max == 0:
                self.agent.learn()
            observation = observation_
                
            return_history.append(wealth - 1000000)
            wealth_history.append(wealth)
        self.agent.memory.clear_memory()

        add_curve(return_history, 'PPO')
        save_plot(self.figure_dir + f'/{self.repeat}2_testing.png',
                  title=f"Testing - {self.intervals['testing'][0].date()} to {self.intervals['testing'][1].date()}",
                  x_label='Days', y_label='Cumulative Return (Dollars)')

        returns = pd.Series(wealth_history).pct_change().dropna()
        stats = timeseries.perf_stats(returns)
        stats.to_csv(self.figure_dir + f'/{self.repeat}3_perf.csv')

