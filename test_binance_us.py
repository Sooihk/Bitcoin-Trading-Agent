import os
from dotenv import load_dotenv
from binance.client import Client

load_dotenv()  # loads .env from current working directory

api_key = os.getenv("BINANCE_US_KEY")
api_secret = os.getenv("BINANCE_US_SECRET")

if not api_key or not api_secret:
    raise RuntimeError("Missing BINANCE_US_KEY or BINANCE_US_SECRET in environment/.env")

client = Client(api_key, api_secret, tld="us")
print(client.get_symbol_ticker(symbol="BTCUSDT"))

symbols = {s["symbol"] for s in client.get_exchange_info()["symbols"]}
print("BTCUSDT tradable?", "BTCUSDT" in symbols)

