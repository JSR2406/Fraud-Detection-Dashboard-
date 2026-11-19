"""Row-local features; no cross-transaction/future aggregation."""
import numpy as np
import pandas as pd
NUMERIC=["TransactionAmt","card1","card2","dist1","C1","C2","C6","C11","V95","V96","V97","hour","day","is_night","log_amt","amt_decimal"]
CATEGORICAL=["ProductCD","card4","card6","P_emaildomain","DeviceType"]
RAW_COLUMNS=["TransactionAmt","TransactionDT","ProductCD","card1","card2","card4","card6","P_emaildomain","dist1","C1","C2","C6","C11","V95","V96","V97","DeviceType"]
def engineer_features(frame: pd.DataFrame)->pd.DataFrame:
    out=frame.copy()
    for col in RAW_COLUMNS:
        if col not in out: out[col]=np.nan
    out["TransactionAmt"]=pd.to_numeric(out.TransactionAmt,errors="coerce"); out["log_amt"]=np.log1p(out.TransactionAmt.clip(lower=0)); out["amt_decimal"]=out.TransactionAmt%1
    dt=pd.to_numeric(out.TransactionDT,errors="coerce"); out["hour"]=(dt//3600)%24; out["day"]=(dt//86400)%7; out["is_night"]=((out.hour<6)|(out.hour>=22)).astype(float)
    for col in CATEGORICAL: out[col]=out[col].fillna("unknown").astype(str)
    return out[NUMERIC+CATEGORICAL]
