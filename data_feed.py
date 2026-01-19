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

# ===============================================================================

class DataFeed:
    """ 
    Market data inferface for bot:
    1. Talk to an exhcnage via CCXT (fetch OHLCV + latest price)
    2. Use CandleStore to persist datasets locally / on S3
    """
    def __init__(self): # constructor
        load_dotenv()  # Load environment variables from .env file
        # load config from Google Sheet
        cfg = ConfigManager()
        cfg.load_google_sheet()  # Load configuration from Google Sheets or local cache

        self.exchange_name = cfg.get('EXCHANGE_NAME')
        self.symbol = cfg.get("TRADING_PAIR")
        # Symbol guard, CCXT expects "BTC/USDT" (with slash), if sheet provides no slash, fixes
        if isinstance(self.symbol, str) and '/' not in self.symbol and self.symbol.endswith('USDT'):
            self.symbol = self.symbol.replace('USDT', '/USDT')
        
        # Timeframes configuation (recommend: 1h and 1d) 
        timeframes_raw = cfg.get("TIMEFRAMES")
        if timeframes_raw:
            self.timeframes = [t.strip().lower() for t in str(timeframes_raw).split(',') if t.strip()]
        else: 
            self.timesframes = [str(cfg.get('TIMEFRAME', '1d')).strip().lower()]
        
        # How much history to download, 4 years is default
        self.years_back = int(cfg.get('YEARS_BACK', 4))

        # Storage wiring
        data_dir = Path(str(cfg.get("DATA_DIR", 'Data')))
        # takes whatever is in the config and converts to bool
        use_s3 = str(cfg.get('USE_S3', '0')).strip().lower() in ('1', 'true', 'yes', 'y')
        # bucket and prefix for s3 object keys
        s3_bucket = cfg.get('S3_BUCKET')
        s3_prefix = str(cfg.get('S3_PREFIX')).strip()
        # create persistence layer, delegate saving/loading to CandleStore
        self.store = CandleStore(StorageConfig(data_dir=data_dir, use_s3=use_s3, s3_bucket=s3_bucket, s3_prefix=s3_prefix))

        # Authentication keys from .env
        self.api_key = os.getenv('API_KEY')
        self.api_secret = os.getenv('API_SECRET')

        # Exchange object from CCXT
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
        except AttributeError as e:
            raise ValueError(
                f"Unknown ccxt exchange '{self.exchange_name}'. Check EXCHANGE_NAME in config."
            ) from e 
        # Instantiate exchange with auth and rate limit enabled
        self.exchange = exchange_class(
            {
                'apiKey': self.api_key,
                'secret': self.api_secret, 
                'enableRateLimit': True
            }
        )
        # Download exchange markets/rules
        self.exchange.load_markets()

    # ----------
    # Helper Methods
    # ----------
    def _base_filename(self, timeframe: str, years_back: Optional[int] = None) -> str:
        """
        Builds a consistent dataset ID used for file names to standardize namimg
        """
        yb = self.years_back if years_back is None else years_back
        # Example: binanceus_BTCUSDT_1h_4Y_to_now
        return f"{self.exchange_name}_{self.symbol.replace('/', '')}_{timeframe}_{yb}Y_to_now"
    
    def _standardize_ohlcv_df(self, rows: list[list[float]]) -> pd.DataFrame:
        """ 
        CCXT returns OHLCV as a list of raw rows. This function turns that into a clean, time-indexed
        DataFrame that the rest of the bot can trust. 
        """
        # convert it into a DataFrame with known schema
        df = pd.DataFrame(rows, columns=['Date', 'open', 'high', 'low', 'close', 'volume'])
        # convert timestamp ms to timezone-aware datetime in UTC
        df['Date'] = pd.to_datetime(df['Date'], unit='ms', utc=True)
        # Coerce OHLCV columns to numeric (some exchanges return strings)
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Remove bad/duplicatw rows and set Date as the index
        df = (
            df.dropna()  # drop rows with NaNs
                .drop_duplicates(subset=['Date'])  # drop duplicate timestamps
                .sort_values('Date')  # sort by Date
                .set_index('Date')  # set Date as index
        )
        # Select and return only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()

        # Basic schema sanity filters (defensive programming for messy exchange data)
        df = df[df["volume"] >= 0]
        df = df[
            (df["high"] >= df["low"]) &
            (df["high"] >= df["open"]) & (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) & (df["low"] <= df["close"])
        ]

        return df

    def ensure_history(
            self, 
            timeframe: str,
            mode: str = 'live',
            ensure_fresh: bool = True, # whether to refresh tail candles
            persist: bool = True, # whether to save to storage after building/updating
    ) -> pd.DataFrame:
        """ 
        Ensures history for this timeframe exists and is up to date. Makes pipeline reliable, by giving
        one canonical way to get the right dataset.
        - tries to load stored history
        - if missing: feteches full history (build)
        - if live and ensure_fresh: incrementally updates tail candles

        Returns up-to-date DataFrame.
        """
        # normalize input
        timeframe = timeframe.strip().lower()
        mode = mode.strip().lower()

        # First try loading from storage (local or S3)
        try:
            df = self.load_historical(timeframe)
        # if missing, build it from scratch by fetching from exchange. Don't have to worry about missing files
        except FileNotFoundError:
            df = self.fetch_historical_ohlcv(timeframe=timeframe, persist=persist)
        
        # In live mode, refresh the tail candles
        if mode == 'live' and ensure_fresh:
            df = self.update_historical(timeframe=timeframe, df=df, persist=persist)
        return df
    
    def fetch_historical_ohlcv(
            self,
            timeframe: Optional[str] = None,
            years_back: Optional[int] = None,
            limit: int = 1000,
            persist: bool = True,
    ) -> pd.DataFrame:
        """
        Run when want durable history file on disk so that:
        - backtests can load instantly without hammering the exchange.
        - live trading can incrementally update instead of re-downloading years of data.
        Fetches full historical OHLCV data from exchange via CCXT. 
        """
        # use default timeframe if none provided
        timeframe = (timeframe or self.timeframes[0]).strip().lower()
        # uses default years_back if none provided
        years_back = self.years_back if years_back is None else years_back

        # Pick a start date: Jan 1st of (current year - years_back)
        now = datetime.now(timezone.utc)
        start_dt = datetime(now.year - years_back, 1, 1, tzinfo=timezone.utc)
        # convert to milliseconds timestamp for CCXT
        since_ms = int(start_dt.timestamp() * 1000)

        now_ms = self.exchange.milliseconds() # exchange's current time in ms
        cl_ms = int(self.exchange.parse_timeframe(timeframe) * 1000)  # candle length in ms
        all_rows = []  # accumulates all candles across pages
        last_ts = None  # tracks last timestamp to detect no-progress

        # Pagination loop, fetch chunks untill you reach 'now'
        while since_ms < now_ms:
            # fetch up to limit candles starting from since_ms
            batch = self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, since=since_ms, limit=limit)
            if not batch:
                break  # exit if no data returned

            # safety valve: if no progress in timestamps, exit to avoid infinite loop
            batch_last_ts = batch[-1][0]
            if last_ts is not None and batch_last_ts <= last_ts:
                print("No new data returned, ending fetch.")
                break
            
            # Accumulate
            all_rows.extend(batch)
            last_ts = batch_last_ts
            # adding +1 ms avoids timestamp alignment issues that can cause gaps if jumped by exact candle length
            since_ms = batch_last_ts + 1 

            # Respect exchange throttling
            time.sleep(self.exchange.rateLimit / 1000)
        if not all_rows:
            raise RuntimeError(f'NO OHLCV data returned for {self.symbol} {timeframe}.')
        # convert raw rows to clean DataFrame
        df = self._standardize_ohlcv_df(all_rows)

        # save to storage if requested
        if persist:
            base = self._base_filename(timeframe, years_back)
            self.store.save(df, base)

        return df

    def load_historical(self, timeframe: str, years_back: Optional[int] = None) -> pd.DataFrame:
        """
        Load stored historical OHLCV data from local/S3.
        Build the file ID and asks CandleStore to load it.
        """
        timerame = timeframe.strip().lower()
        base = self._base_filename(timerame, years_back)
        return self.store.load(base)
    
    def save_historical(self, df: pd.DataFrame, timeframe: str, years_back: Optional[int] = None) -> None:
        """
        Wrapper around store.save(). Keeps namimg consistent.
        """
        timeframe = timeframe.strip().lower()
        base = self._base_filename(timeframe, years_back)
        self.store.save(df, base)

    def update_historical(
            self,
            timeframe: str, 
            df: Optional[pd.DataFrame] = None,
            lookback_candles: int = 3, # refetch last 3 candles and overwrite them to keep tail accurate
            limit: int = 1000,
            persist: bool = True,
    ) -> pd.DataFrame:
        """  
        'live maintenance' function for stored candle dataset to keep the big historical file fresh
        by keeping that file fresh by refetching only the most recetn candles and merging them in safety.

        Prevents bot from trading on stale or partially formed candles without constantly re-downloading years of data.
        """
        timeframe = timeframe.strip().lower()
        cl_ms = int(self.exchange.parse_timeframe(timeframe) * 1000)  # candle length in ms

        # if caller didnt provide df, load from storage
        if df is None:
            df = self.load_historical(timeframe)
        
        # if dataset exists but is empty, rebuild
        if df.empty:
            df = self.fetch_historical_ohlcv(timeframe=timeframe, persist=persist)

        # Find the last candle timestamp and compute overlap window
        last_dt = pd.to_datetime(df.index.max(), utc=True)
        last_ms = int(last_dt.timestamp() * 1000) 

        # Pull back a few candles to safely replace the tail
        since_ms = max(0, last_ms - cl_ms * max(lookback_candles - 1,0))

        # Fetch the recent candles from the exchange
        rows = self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not rows:
            return df
        
        # standardize fetched rows into DataFrame
        new_df = self._standardize_ohlcv_df(rows)

        # Merge old + new and overwrite overlapping timestamps
        combined = pd.concat([df, new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        # persis updated dataset
        if persist: 
            self.save_historical(combined, timeframe)

        return combined

    def get_ohlcv(
            self, 
            timeframe: str,
            lookback: int, 
            mode: str = 'live', 
            ensure_fresh: bool = True,
    ) -> pd.DataFrame:
        """
        High-level method to get OHLCV data for a given timeframe and lookback.
        Unified accessor.

        Always routes through ensure_history() so callers never crash if the dataset
        hasn't been built yet.

        - backtest: loads stored dataset and tail(lookback)
        - live: ensures dataset exists, optionally updates tail candles, then tail(lookback)
        """
        timeframe = timeframe.strip().lower()
        mode = mode.strip().lower()

        # Returns only what is needed: last 'lookback' candles
        df = self.ensure_history(timeframe, mode=mode, ensure_fresh=ensure_fresh, persist=True)
        return df.tail(int(lookback)).copy()
    
    def get_latest_price(self) -> float:
        """Fetches the latest price for the trading pair from the exchange."""
        ticker = self.exchange.fetch_ticker(self.symbol)
        return float(ticker['last'])

if __name__ == "__main__":
    feed = DataFeed()

    # One clean path for both backtest/live: ensure datasets exist and (optionally) refresh.
    for tf in feed.timeframes:
        df_tf = feed.ensure_history(tf, mode="live", ensure_fresh=True, persist=True)
        print(f"[INFO] {tf} rows:", df_tf.shape[0], "last:", df_tf.index.max())

    # Pull last N candles from the first configured timeframe
    tf0 = feed.timeframes[0]
    last_200 = feed.get_ohlcv(tf0, lookback=200, mode="live", ensure_fresh=True)
    print(f"{tf0} candles (tail=200):", last_200.shape)

    print("Latest price:", feed.get_latest_price())
