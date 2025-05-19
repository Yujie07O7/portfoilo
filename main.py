
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
    all_logs=[]
    for i in range(1):
        print(f"---------- round {i} ----------")
        if not os.path.isfile(f'plots/ppo/{i}2_testing.png'):
            ppo = PPO(state_type='indicators', djia_year=2019, repeat=i)
            ppo.train()
            ppo.test()
            all_logs.extend(ppo.env.history_log)

    df = pd.DataFrame(all_logs)
    returns_df = pd.DataFrame(df["returns"].to_list(), columns=["DBC_ret", "SHY_ret", "SPY_ret"])
    weights_df = pd.DataFrame(df["weights"].to_list(), columns=["DBC_weight", "SHY_weight", "SPY_weight"])
    df.drop(columns=["returns", "weights"], inplace=True)
    df = pd.concat([df, returns_df, weights_df], axis=1)
    # 匯出 CSV
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/ppo_rounds_history.csv", index=False)
    print("✅ 所有紀錄已儲存為 output/ppo_rounds_history.csv")
if __name__ == '__main__':
    main()
