import requests, json
import pandas as pd

# primary pair (baseline): ethusd, btcusd
# defi pair (protocol):  aaveusd, compusd
# infrastructure (network events): maticusd
# extra: crvusd, yfiusd, sandusd, , sushiusd, dogeusd, maskusd, ctxusd, shibusd

# Many smaller tokens have short or highly volatile histories, or big illiquidity. That can make calibration unstable.

base_url = "https://api.gemini.com/v2"

# List of all available symbols
tokens = ["ethusd", "btcusd", "ltcusd"]

def get_prices(symbol):
    response = requests.get(base_url + f"/candles/{symbol}/1day")
    return response.json()

def create_df(data):
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
    return df

def convert_time(df):
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.sort_values("time")
    return df

# Gather data into dataframes (seperate)
for token in tokens:
    data = get_prices(token)
    df = create_df(data)
    df = convert_time(df)
    df.to_csv(f"/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/data/token_data/{token}.csv", index=False)
    print(df)
    print(f"Saved {token}.csv")

