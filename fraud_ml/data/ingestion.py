"""Explicit real/demo ingestion and reproducible synthetic records."""
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from fraud_ml.data.validation import validate_dataset
def make_demo_data(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Synthetic UI/pipeline smoke-test records only; not real fraud evidence."""
    rng = np.random.default_rng(seed); fraud = (rng.random(rows) < .035).astype("int8")
    amount = rng.exponential(80, rows) * np.where(fraud == 1, 3.5, 1.0)
    return pd.DataFrame({"TransactionID": np.arange(2987000, 2987000 + rows), "TransactionDT": np.sort(rng.integers(86400, 15811200, rows)), "TransactionAmt": amount.round(2), "ProductCD": rng.choice(["W","H","C","S","R"], rows), "card1": rng.integers(1000,18000,rows), "card2": rng.choice([np.nan,*range(100,600)],rows), "card4": rng.choice(["visa","mastercard","discover","amex"],rows), "card6": rng.choice(["debit","credit"],rows), "P_emaildomain": rng.choice(["gmail.com","yahoo.com","hotmail.com","anonymous.com",None],rows), "dist1": rng.exponential(50,rows), "C1": rng.integers(0,300,rows), "C2": rng.integers(0,300,rows), "C6": rng.integers(0,100,rows), "C11": rng.integers(0,100,rows), "V95": rng.integers(0,5,rows), "V96": rng.integers(0,5,rows), "V97": rng.integers(0,10,rows), "DeviceType": rng.choice(["desktop","mobile",None],rows), "isFraud": fraud})
def _read(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1: raise ValueError(f"Expected exactly one CSV in {path}.")
            with archive.open(names[0]) as stream: return pd.read_csv(stream, nrows=nrows)
    return pd.read_csv(path, compression="infer", nrows=nrows)
def load_ieee_cis(data_dir: str | Path, sample_rows: int | None = None, *, use_pandera: bool = False) -> pd.DataFrame:
    root = Path(data_dir); tx_path = next((p for p in (root/"train_transaction.csv",root/"train_transaction.csv.zip") if p.is_file()),None)
    if tx_path is None: raise FileNotFoundError(f"IEEE-CIS transaction file not found under {root}.")
    tx = _read(tx_path, sample_rows); id_path = next((p for p in (root/"train_identity.csv",root/"train_identity.csv.zip") if p.is_file()),None)
    if id_path:
        identity = _read(id_path)
        if "TransactionID" not in identity or identity.TransactionID.duplicated().any(): raise ValueError("Identity TransactionID must be unique.")
        tx = tx.merge(identity,on="TransactionID",how="left",validate="one_to_one")
    validate_dataset(tx, use_pandera=use_pandera); return tx
def load_dataset(mode: str, data_dir: str | Path = "data", seed: int = 42, sample_rows: int | None = None, *, use_pandera: bool = False) -> tuple[pd.DataFrame,str]:
    if mode == "real": return load_ieee_cis(data_dir,sample_rows,use_pandera=use_pandera), "real"
    if mode == "demo":
        frame = make_demo_data(sample_rows or 5000,seed)
        validate_dataset(frame, use_pandera=use_pandera)
        return frame, "demo — synthetic demonstration only"
    raise ValueError("mode must be explicitly real or demo")
