"""Select operating thresholds from validation outcomes and explicit assumptions."""
import numpy as np
from sklearn.metrics import f1_score, precision_score

def optimize_threshold(y_true, probability, strategy="max_f1", *, min_precision=.5, capacity=None, false_positive_cost=1., missed_fraud_cost=10.):
    y=np.asarray(y_true,dtype=int); p=np.asarray(probability,dtype=float)
    if len(y)==0 or len(y)!=len(p) or not np.isfinite(p).all(): raise ValueError("Threshold inputs must be valid and non-empty.")
    candidates=[]
    for threshold in np.unique(np.r_[0.,p,1.]):
        pred=p>=threshold; precision=precision_score(y,pred,zero_division=0); f1=f1_score(y,pred,zero_division=0)
        if strategy=="max_f1": objective=f1
        elif strategy=="min_precision":
            if precision<min_precision: continue
            objective=float(pred.sum())
        elif strategy=="review_capacity":
            if capacity is None: raise ValueError("capacity is required for review_capacity strategy")
            if pred.sum()>capacity: continue
            objective=float(np.sum(pred & (y==1)))
        elif strategy=="business_cost": objective=-(int(np.sum(pred & (y==0)))*false_positive_cost+int(np.sum(~pred & (y==1)))*missed_fraud_cost)
        else: raise ValueError(f"Unsupported threshold strategy: {strategy}")
        candidates.append((objective,float(threshold),float(precision),float(f1)))
    if not candidates: raise ValueError("No threshold meets the selected constraints.")
    objective,threshold,precision,f1=max(candidates,key=lambda x:(x[0],x[1]))
    return {"threshold":threshold,"strategy":strategy,"precision":precision,"f1":f1,"objective":objective,"assumptions":{"false_positive_cost":false_positive_cost,"missed_fraud_cost":missed_fraud_cost,"capacity":capacity}}
