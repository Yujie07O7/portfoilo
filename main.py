

import itertools
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from algorithms.ppo.PPO import PPO
import plot
import torch.multiprocessing as mp
import os

def plot_env_history(env):
    import matplotlib.pyplot as plt
    import pandas as pd
    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame(env.history_log)
    df.to_csv("output/history.csv")
def main():

    plot.initialize()
    mp.set_start_method('spawn')

    for i in range(1):
        print(f"---------- round {i} ----------")
        if not os.path.isfile(f'plots/ppo/{i}2_testing.png'):
            ppo = PPO(state_type='indicators', djia_year=2019, repeat=i)
            ppo.train()
            ppo.test()
            plot_env_history(ppo.env)
if __name__ == '__main__':
    main()
