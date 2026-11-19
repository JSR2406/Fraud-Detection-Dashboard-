"""Chronological train/validation/test partitions."""
from dataclasses import dataclass
import pandas as pd
@dataclass
class DataPartitions:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
def chronological_split(df: pd.DataFrame, train_fraction: float=.70, validation_fraction: float=.15) -> DataPartitions:
    if "TransactionDT" not in df: raise ValueError("TransactionDT is required for chronological evaluation.")
    if train_fraction<=0 or validation_fraction<=0 or train_fraction+validation_fraction>=1: raise ValueError("Invalid split fractions.")
    ordered=df.sort_values(["TransactionDT","TransactionID"],kind="mergesort").reset_index(drop=True); n=len(ordered); a=int(n*train_fraction); b=int(n*(train_fraction+validation_fraction))
    if a<2 or b<=a or b>=n: raise ValueError("Dataset too small for requested partitions.")
    parts=DataPartitions(ordered.iloc[:a].copy(),ordered.iloc[a:b].copy(),ordered.iloc[b:].copy())
    if any(p.isFraud.nunique()<2 for p in (parts.train,parts.validation,parts.test)): raise ValueError("Every partition needs both classes.")
    return parts
