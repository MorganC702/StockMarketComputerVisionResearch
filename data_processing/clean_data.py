"""
File:        clean_data.py
Description: Complete data cleansing function 
Author: Morgan Cooper
Created:     2025-09-01
Updated:     2025-09-01
Usage:       N/A  

Notes:
For this research project the data source is anticipated to be very high quality, 
so this function is simply for insurance or anomolies. 

"""

import pandas as pd
from pandas.util import hash_pandas_object
from typing import Literal, Optional
from time import time
import numpy as np


def clean_data(
    df: pd.DataFrame,
    timestamp_col: str = "Date",
    symbol_col: str = "Symbol",
    drop_duplicate_rows: bool = True,
    drop_duplicate_cols: bool = True,
    drop_constant_columns: bool = True,
    drop_constant_rows: bool = True,
    replace_placeholders: bool = True,
    placeholders=("Null", "null", "NULL", "NaN", "nan", "NAN", "None", "none", "NONE"),
    fill_missing: bool = True,
    convert_numeric: bool = True,
    sort_by: Optional[Literal["index", "timestamp"]] = "timestamp",
    verbose: bool = False,
) -> pd.DataFrame:

    start_total = time()
    df = df.copy()

    if verbose:
        print(f"[---CLEAN---] Starting Shape={df.shape}")
        print(f"[---CLEAN---] Preserving: {symbol_col} and {timestamp_col}")

    # Columns that need to remain un-touched throughout the cleaning process.
    protect = {symbol_col, timestamp_col}


    if drop_duplicate_cols:
        if verbose:
            print("[---CLEAN---] Step 1: Remove Duplicate Columns.")
        t0 = time()
        s1 = df.shape
        col_hash = {}
        
        # Faster + more scalable.... (the dataset has 100s of millions of rows)
        for c in df.columns:
            
            # Create 64bit hash on each row
            h = hash_pandas_object(
                df[c], 
                index=False
            ).to_numpy(dtype="uint64", copy=False)
            
            # Calculate the sum for all hash values in in column c
            h_sum = int(h.sum(dtype=np.uint64))
            
            # Applies bitwise XOR operation
            # XOR helps catch permutations of values like [1, 2, 3] vs [3, 2, 1] that sum would not.
            h_xor = int(np.bitwise_xor.reduce(h)) if h.size else 0
            
            # Store thr sum and XOR results as a tuple
            col_hash[c] = (h_sum, h_xor)

        # Loop through and identify the duplicated from the col_has dict.
        seen, keep = set(), []
        for c, fp in col_hash.items():
            if fp not in seen:
                seen.add(fp)
                keep.append(c)
        df = df[keep]
        s2 = df.shape
        if verbose:
            print(f"[---CLEAN---] ------- Original Column Count: {s1[1]}, "
                  f"After: {s2[1]}, Removed: {s1[1] - s2[1]} in {time() - t0:.5f}s")

    if drop_duplicate_rows:
        if verbose:
            print("[---CLEAN---] Step 2: Remove Duplicate Rows.")
        t0 = time()
        s1 = df.shape
        df = df.drop_duplicates() # Scales fine for rows analysis (intenal row wise hashing)
        s2 = df.shape
        if verbose:
            print(f"[---CLEAN---] ------- Original Row Count: {s1[0]}, "
                  f"After: {s2[0]}, Removed: {s1[0] - s2[0]} in {time() - t0:.5f}s")

    if drop_constant_columns:
        if verbose:
            print("[---CLEAN---] Step 3: Remove Constant Columns.")
        t0 = time()
        s1 = df.shape
        
        # Only operate on numeric columns, since categorical columns like 'Symbol' can be constant.
        num_cols = df.select_dtypes(include=[np.number, "number"])
        if num_cols.shape[1] > 0:
            # Calculate  unique column count
            nunique = num_cols.nunique(dropna=True)
            # If unique columns is <=. 1 remove it.
            constant_cols = nunique.index[nunique <= 1].tolist()
            constant_cols = [c for c in constant_cols if c not in protect]
            if constant_cols:
                df = df.drop(columns=constant_cols)
        s2 = df.shape
        if verbose:
            print(f"[---CLEAN---] ------- Original Column Count: {s1[1]}, "
                  f"After: {s2[1]}, Removed: {s1[1] - s2[1]} in {time() - t0:.5f}s")

    if drop_constant_rows:
        if verbose:
            print('[---CLEAN---] Step 4: Remove Constant Rows.')
        t0 = time()
        s1 = df.shape
        
        # Only operate on numeric columns, since categorical columns like 'Symbol' can be constant.
        num_cols = df.select_dtypes(include=[np.number, "number"])
        if num_cols.shape[1] > 0:
            # Calculate min and max. If they are the same the row is constant. 
            row_min = num_cols.min(axis=1, skipna=True)
            row_max = num_cols.max(axis=1, skipna=True)
            non_empty = num_cols.count(axis=1) > 0
            const_mask = (row_min == row_max) & non_empty
            # Remove const_mask (constant rows)
            df = df.loc[~const_mask]
        s2 = df.shape
        if verbose:
            print(f"[---CLEAN---] ------- Original Row Count: {s1[0]}, "
                  f"After: {s2[0]}, Removed: {s1[0] - s2[0]} in {time() - t0:.5f}s")

    # Replcae placeholder values with NaN
    # Paceholders =  "Null", "null", "NULL", "NaN", "nan", "NAN", "None", "none", "NONE"
    if replace_placeholders:
        if verbose:
            print("[---CLEAN---] Step 5: Replacing Placeholder Values")
        t0 = time()
        str_cols = [c for c in df.select_dtypes(
            include=["object", "string"]).columns if c not in protect]
        if str_cols:
            df[str_cols] = df[str_cols].replace(placeholders, np.nan)
        nulls = int(df.isnull().sum().sum())
        if verbose:
            print(f"[---CLEAN---] ------- Total Nulls After Replacement: "
                  + f"{nulls} in {time() - t0:.5f}s")

    if sort_by:
        if verbose:
            print(f"[---CLEAN---] Step 6: Sorting by {sort_by.capitalize()}.")
        t0 = time()
        if sort_by == "index":
            df = df.sort_index()
        elif sort_by == "timestamp":
            if timestamp_col not in df.columns:
                raise ValueError(f"Column '{timestamp_col}' not found for timestamp sorting.")
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
            df = df.sort_values(by=timestamp_col)
        else:
            raise ValueError(f"Invalid value for sort_by: {sort_by}: 'index' or 'timestamp'.")
        if verbose:
            print(f"[---CLEAN---] ------- Sorted by {sort_by} in {time() - t0:.5f}s")

    # This will ONLY FORWARD FILL the data, to prevent future leaking.
    # An example for when this would be valuable is forward filling a 
    # value that only shows once per month, but has relevence every day. 
    if fill_missing:
        if verbose:
            print("[---CLEAN---] Step 7: Interpolating Missing and NaN Values.")
        t0 = time()
        n1 = int(df.isna().sum().sum())
        numeric_cols = df.select_dtypes(include=[np.number, "number"]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].ffill()
        n2 = int(df.isna().sum().sum())
        if verbose:
            print(f"[---CLEAN---] ------- Initial Nulls: {n1}, After Fill: {n2}, "
                  f"Filled: {n1 - n2} in {time() - t0:.3f}s")

    #  For machine learning, all 'important' and numerical values are in a numeric form (not strings).
    if convert_numeric:
        if verbose:
            print("[---CLEAN---] Step 8: Converting Data to Numerical Values.")
        t0 = time()
        obj_cols = [c for c in df.select_dtypes(include=['object', 'string']).columns if c not in protect]
        for c in obj_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        if verbose:
            print(f"[---CLEAN---] ------- Conversion complete in {time() - t0:.3f}s")

    if verbose:
        print(f"[---CLEAN---] Cleaning complete in {time() - start_total:.3f}s. Final shape: {df.shape}\n")

    return df
