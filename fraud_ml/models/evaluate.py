"""Fraud-focused ranking, classification, capacity and calibration metrics."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def evaluate_model(y_true: np.ndarray, probability: np.ndarray, threshold: float = .5, k: int | None = None) -> dict:
    y=np.asarray(y_true,dtype=int); p=np.asarray(probability,dtype=float)
    if len(y)!=len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError("Invalid labels or probabilities.")
    pred=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    top=min(k or max(1,int(np.ceil(len(y)*.01))),len(y)); idx=np.argsort(p)[-top:]; fraud_total=max(int(y.sum()),1)
    return {"roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,"average_precision":float(average_precision_score(y,p)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"threshold":float(threshold),"confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]],"false_positive_rate":float(fp/max(fp+tn,1)),"false_negative_rate":float(fn/max(fn+tp,1)),"precision_at_k":float(y[idx].mean()),"recall_at_k":float(y[idx].sum()/fraud_total),"review_capacity":int(top),"brier_score":float(brier_score_loss(y,p))}
