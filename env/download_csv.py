import os
import requests

# GitHub 上的 CSV 目錄
base_url = "https://raw.githubusercontent.com/matinaghaei/Portfolio-Management-ActorCriticRL/master/env/data/DJIA_2019/"

# 本機存放路徑
save_dir = "env/data/DJIA_2019/"
os.makedirs(save_dir, exist_ok=True)  # ✅ 確保目錄存在

# 30 支道瓊股票的代碼
tickers = [
    "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DIS", "DWDP", "GS", "HD",
    "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE",
    "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ", "WBA", "WMT", "XOM"
]

# 下載所有 CSV
for ticker in tickers:
    file_url = f"{base_url}ticker_{ticker}.csv"
    file_path = os.path.join(save_dir, f"ticker_{ticker}.csv")

    print(f"正在下載 {file_url} ...")
    response = requests.get(file_url)

    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"✅ 下載成功: {file_path}")
    else:
        print(f"⚠️ 無法下載 {file_url}，請檢查 GitHub 連結是否有效！")

