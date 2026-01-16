"""data_feed.py
- Provide OHLCV data for backtesting (load from local/s3)
- Provide OHLCV + latest price for live trading ccxt
- Keep stored  datasets up to date via incremental candle append
"""
from __future__ import annotations

import os # reads environment variables for API keys
import time # time, datetime used to control data time range and pacing requests
from dataclasses import dataclass # to create StorageConfig class
from datetime import datetime, timezone # compute years back start date in UTC
from pathlib import Path # manage file paths
from typing import Optional

#from matplotlib import ticker
import numpy as np 
import pandas as pd
import ccxt # main library to talk to cryptocurrency exchanges
from dotenv import load_dotenv # load .env for secret keys
from config_manager import ConfigManager # custom config manager to load settings from Google Sheets

# ===============================================================================
# STORAGE LAYER FOR DATA FEED
@dataclass
class StorageConfig:
    # settings object, instead Instead of passing 4 separate parameters everywhere, you package storage settings into one object. 
    # This makes code easier to test and evolve.
    data_dir: Path = Path("Data")  # Directory to store local data files
    use_s3: bool = False          # Whether to use S3 for storage
    s3_bucket: Optional[str] = None  # S3 bucket name if using S3
    s3_prefix: str = ''  # S3 prefix/path if using S3

class CandleStore:
    """Persistence layer that knows how tro save/load candle datasets.
    Centralizes:
    1. where files live (Data/)
    2. file namimg rules
    3. read preference (parquet first, csv fallback)
    4. S3 download if missing
    5. S3 upload after save
    6. index normalization (Date UTC, sorted, no duplicates)
    """
    def __init__(self, cfg: StorageConfig):
        self.cfg = cfg
        self.data_dir = Path(self.cfg.data_dir) # ensure data_dir is a Path object
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True) # ensure local data dir exists

        self._s3 = None
        # if s3 is enabled, set up boto3 client and create an S3 client
        if self.cfg.use_s3:
            try:
                import boto3
                self._s3 = boto3.client('s3')
                if not self.cfg.s3_bucket:
                    raise ValueError("S3 bucket name must be provided if use_s3 is True.")
            except Exception as e: # if boto3 isn't installed
                raise RuntimeError(
                    "S3 support requested but boto3 is unavailable or misconfigured. "
                    "Install boto3 and set S3_BUCKET (and optionally S3_PREFIX)."
                ) from e
            
    def _safe_prefix(self) -> str:
        """Takes s3_prefix, strips leading/trailing slashes and ensures S3 prefix ends with a slash if non-empty."""
        p = (self.cfg.s3_prefix or '').strip('/')
        return f"{p}/" if p else ''

    def _s3_key(self, filename: str) -> str:
        """Build full S3 key for a given filename."""
        return f"{self._safe_prefix()}{filename}"
    
    def local_paths(self, base_filename: str) -> dict[str, Path]:
        """Returns where parquet and csv files should be stored locally."""
        return {
            'parquet': self.cfg.data_dir / f'{base_filename}.parquet',
            'csv': self.cfg.data_dir / f'{base_filename}.csv'
        }
    def exists_local(self, base_filename: str) -> bool:
        """Checks if either file exists locally."""
        paths = self.local_paths(base_filename)
        return paths['parquet'].exists() or paths['csv'].exists()
    
    def download_from_s3(self, base_filename: str) -> None:
        """If S3 is enabled, attempts to download parquet and csv files from S3 to local storage."""
        if not self._s3:
            return
        paths = self.local_paths(base_filename)

        # prefer parquet; fallback to csv
        for ext in ('parquet', 'csv'):
            # build exact S3 key and local path
            filename = f'{base_filename}.{ext}'
            key = self._s3_key(filename)
            local_path = paths[ext]
            
            try: 
                self._s3.download_file(self.cfg.s3_bucket, key, str(local_path))
                return
            except Exception:
                continue
    
    def upload_to_s3(self, base_filename: str) -> None:
        """Uploads files locally to S3 if S3 is enabled."""
        if not self._s3:
            return
        paths = self.local_paths(base_filename)
        for ext in ('parquet', 'csv'):
            local_path = paths[ext]
            if not local_path.exists():
                continue
            filename = local_path.name
            key = self._s3_key(filename)
            self._s3.upload_file(str(local_path), self.cfg.s3_bucket, key)

    def load(self, base_filename: str) -> pd.Dataframe:
        """
        Load historical candles from local (or download from S3)
        """
        # If local missing and S3 enabled, try download
        if not self.exists_local(base_filename) and self.cfg.use_s3:
            self.download_from_s3(base_filename)

        # Load parquet if exists, else csv
        paths = self.local_paths(base_filename)
        if paths['parquet'].exists():
            df = pd.read_parquet(paths['parquet'])
            # Depending on engine/version, Parquet may round-trip the datetime index as:
            # (a) an index (ideal)
            # (b) if Date came back as a normal column, convert it to UTC datetime and set as idnex
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                df = df.set_index('Date')
            # (c) handles alternative where parquet stored index with name of "__index_level_0__"
            elif '__index_level_0__' in df.columns:
                df['__index_level_0__'] = pd.to_datetime(df['__index_level_0__'], utc=True)
                df = df.set_index('__index_level_0__')
        # Reads from CSV, parases Date into datetime and sets as index
        elif paths['csv'].exists():
            df = pd.read_csv(paths['csv'], parse_dates=['Date'], index_col='Date')
        else: # Missing case, tells user needs to build history first
            raise FileNotFoundError(f'No local dataset found for {base_filename}.'
                                    'Build historical data with fetch_historical_ohlcv() first.')

        # Normalize index: ensure UTC, sorted, and sets name to 'Date'
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'Date'
        return df.sort_index()
    
    def save(self, df: pd.Dataframe, base_filename: str) -> None:
        """Save to local; optionally upload to S3.

        Normalize the index before both parquet and csv writes to avoid silent Parquet index shape changes.
        """
        paths = self.local_paths(base_filename)
        # Normalize before writing
        df_to_write = df.copy()
        df_to_write.index = pd.to_datetime(df_to_write.index, utc=True)
        df_to_write.index.name = 'Date'

        # Parquet
        try:
            df_to_write.to_parquet(paths['parquet'], index=True)
        except Exception as e:
            print(f"[WARN] Parquet save failed ({type(e).__name__}: {e}).")

        # CSV
        df_to_write.to_csv(paths['csv'], index=True)

        # Optional S3 upload
        if self.cfg.use_s3:
            try:
                self.upload_to_s3(base_filename)
            except Exception as e:
                print(f"[WARN] S3 upload failed ({type(e).__name__}: {e}).")



class DataFeed:
    """ 
    Market data inferface for bot:
    1. Talk to 
    """
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        config = ConfigManager() 
        config.load_google_sheet()  # Load configuration from Google Sheets or local cache

        # set the exchange and trading pair dynamically 
        self.exchange_name = config.get("EXCHANGE_NAME") 
        self.symbol = config.get("TRADING_PAIR")
        self.timeframe = str(config.get("TIMEFRAME", "1d")).strip().lower() # default to 1h if not set
        self.years_back = int(config.get("YEARS_BACK", 4))  # default to 1 year if not set

        # authentication keys from .env
        self.api_key = os.getenv("API_KEY")
        self.api_secret = os.getenv("API_SECRET")

        # initialize exchange class to use
        self.exchange = getattr(ccxt, self.exchange_name)({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True
        })
        self.exchange.load_markets() # load exchange rulebook + list of tradable pairs

    def fetch_historical_ohlcv(
                self, 
                timeframe: str | None = None, 
                years_back: int | None = None, 
                out_dir: str | Path = "Data", 
                base_filename: str | None = None, 
                limit: int = 1000,
        ) -> pd.DataFrame:
        if timeframe is None:
            timeframe = self.timeframe
        if years_back is None:
            years_back = self.years_back
        # Define the date range for how far back to fetch data
        now = datetime.now(timezone.utc)
        start_dt = datetime(now.year - years_back, 1, 1, tzinfo=timezone.utc)
        since_ms = int(start_dt.timestamp() * 1000)
        # Get current timestamp and candlesize in milliseconds
        now_ms = self.exchange.milliseconds()
        cl_ms = int(self.exchange.parse_timeframe(timeframe) * 1000) # candle lenghth in ms
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if base_filename is None: # create base filename if not provided
            base_filename = f"{self.exchange_name}_{self.symbol.replace('/', '')}_{timeframe}_{years_back}Y_to_now"
        all_rows = []
        last_ts = None  # tracks last timestamp to detect no-progress
        while since_ms < now_ms: # Pagination loop, fetching batches until we reach "now"
            batch = self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, since=since_ms, limit=limit)
            if not batch: 
                break  # exit if no data returned
            # open timestamp of last candle in this batch
            batch_last_ts = batch[-1][0]
            # Guard aganist infinite loops if exchange returns 
            if last_ts is not None and batch_last_ts <= last_ts:
                print("No new data returned, ending fetch.")
                break
            # accumulate results
            all_rows.extend(batch)
            last_ts = batch_last_ts
             # move "since" to just after the last candle to avoid duplicates
            # Next request, start at the next candle after the last one I already have.
            since_ms = batch_last_ts + cl_ms
            # respect API rate limits
            time.sleep(self.exchange.rateLimit / 1000)
        if not all_rows:
            raise RuntimeError(f"No OHLCV data returned for {self.symbol} {timeframe}.")
        
        df = pd.DataFrame(all_rows, columns=["Date", "open", "high", "low", "close", "volume"])
        # Clean timestamp and clean types
        df['Date'] = pd.to_datetime(df['Date'], unit='ms', utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        # Clean duplicates + sort + index (pagination can overlap depending on exchange quirks)
        df = (
            df.dropna()
              .drop_duplicates(subset=["Date"])
              .sort_values("Date")
              .set_index("Date")
        )
        # Keep only what you need + add log returns
        df = df[["open", "high", "low", "close", "volume"]].copy()
        # Add log return column, log(close_t) - log(close_{t-1})
        df["log_ret"] = np.log(df["close"]).diff()
        # Save to Parquet + CSV
        parquet_path = out_dir / f"{base_filename}.parquet"
        csv_path = out_dir / f"{base_filename}.csv"
        try:
            df.to_parquet(parquet_path, index=True)
        except Exception as e:
            print(f"[WARN] Parquet save failed ({type(e).__name__}: {e}).")
        df.to_csv(csv_path, index=True)
        return df
    def get_latest_price(self) -> float:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']


if __name__ == "__main__":
    feed = DataFeed()
    df = feed.fetch_historical_ohlcv()
    df = feed.fetch_historical_ohlcv(out_dir="Data")
    print(f"Fetched {feed.timeframe} candles:", df.shape)
    print("Latest price:", feed.get_latest_price())
