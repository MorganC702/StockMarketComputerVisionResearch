"""
File:        load_ticker.py
Description: Finds and returns a ticker from a directory of tickers, either upon request or at random.
Author:      Morgan Cooper
Created:     2025-09-01
Updated:     2025-09-01
Usage:       N/A  

Notes:
If `symbol` is provided, load that ticker; otherwise pick randomly.
Returns a single concatenated DataFrame sorted by timestamp.
"""

import sys; sys.path.append("..")
from pathlib import Path
import pandas as pd 
import random

def load_ticker(
    base_dir="../parquet_minute", 
    time_col = "Date",
    symbol_col="Symbol",
    seed: int = 42, 
    symbol: str | None = None,
    verbose: bool = True
) -> pd.DataFrame:
    
    # Convert base_dir to filesytem path object
    base = Path(base_dir)

    # Collect all symbol directories located within base directory
    symbol_dirs = [p for p in base.glob("Symbol=*") if p.is_dir()]
    
    # If no symbol directory exists throw error
    if not symbol_dirs:
        raise FileNotFoundError(f"No Symbol=* directories found under {base.resolve()}")
    
    # If a symbol is not defined, select one at random.
    if symbol is None:
        random.seed(seed)
        sym_path = random.choice(symbol_dirs)
        symbol = sym_path.name.split("=", 1)[1]
    
    # If a symbol is defined collect its file path.
    else:
        sym_path = base / f"Symbol={symbol}"
        if not sym_path.is_dir():
            raise FileNotFoundError(f"{sym_path} not found")

    # Collect all parquet files for this symbol (all years/months) 
    files = sorted(sym_path.glob("year=*/month=*.parquet"))
    if not files:
        raise FileNotFoundError(f"No month parquet files under {sym_path}")

    # Loop through all files for selected ticker, format eahc and add to frames array. 
    frames = []
    for f in files:
        try:
            df_part = pd.read_parquet(f)
            
            # ensure Symbol column exists/consistent
            if symbol_col not in df_part.columns:
                df_part["Symbol"] = symbol
                
            # ensure time column is in datetime format
            if time_col in df_part.columns and not pd.api.types.is_datetime64_any_dtype(df_part[time_col]):
                df_part[time_col] = pd.to_datetime(df_part[time_col], utc=True, errors="coerce")
            frames.append(df_part)
            
        except Exception as e:
            print(f"[LOAD TICKER] Skipping {f}: {e}")

    if not frames:
        raise RuntimeError(f"All reads failed for {symbol}")

    # Combine all files 
    df = pd.concat(frames, ignore_index=True)

    # sort and clean simple issues
    if time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)

    if verbose:
        print(f"[LOAD TICKER] Loaded {symbol}: {len(files)} files -> shape {df.shape}")
        if "Date" in df.columns:
            print(f"[LOAD TICKER] Date range: {df['Date'].min()}  to  {df['Date'].max()}")
    
    return df




