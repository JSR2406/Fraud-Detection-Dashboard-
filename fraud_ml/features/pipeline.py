"""Training-only numeric imputation, scaling and unseen-safe category encoding."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from fraud_ml.features.engineering import NUMERIC,CATEGORICAL
def build_preprocessor(scale_numeric: bool=False)->ColumnTransformer:
    numeric=[("imputer",SimpleImputer(strategy="median",add_indicator=True))]
    if scale_numeric: numeric.append(("scaler",StandardScaler()))
    categorical=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=True))])
    return ColumnTransformer([("numeric",Pipeline(numeric),NUMERIC),("categorical",categorical,CATEGORICAL)],remainder="drop",verbose_feature_names_out=True)
