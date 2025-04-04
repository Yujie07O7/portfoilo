
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from algorithms.ppo.PPO import PPO
import plot
import torch.multiprocessing as mp
import os

def main():
    plot.initialize()
    mp.set_start_method('spawn')
    np.set_printoptions(suppress=True)
    for i in range(2):
        print(f"---------- round {i} ----------")
        if not os.path.isfile(f'plots/ppo/{i}2_testing.png'):
            ppo = PPO(state_type='indicators', djia_year=2019, repeat=i)
            ppo.train()
            ppo.test()

if __name__ == '__main__':
    main()
