# XGBoost model training with hyperparameter optimization using Optuna

from pathlib import Path
import pandas as pd, numpy as np, optuna, joblib, datetime as dt
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
import os

try:
    import xgboost as xgb
    xgb.XGBClassifier(tree_method='gpu_hist', device='cuda', n_estimators=1).fit([[1]], [0])
    DEVICE = 'cuda'
    TREE_METHOD = 'gpu_hist'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device_name = "GPU"
except:
    DEVICE = 'cpu'
    TREE_METHOD = 'hist'
    device_name = "CPU"

print(f"Training XGBoost with {device_name}")

DATA_FILE = Path("data/features/ncaab_cumulative_features_v13_sentiment_enhanced.csv")
FEAT_FILE = Path("./config/boruta_features_sentiment.txt")

df = pd.read_csv(DATA_FILE, parse_dates=["date"], low_memory=False).sort_values("date")

feat_names = FEAT_FILE.read_text().splitlines()

if "open_total" in feat_names:
    MODEL_OUT = Path("models/xgb_cover_market.pkl")
    model_type = "market"
else:
    MODEL_OUT = Path("models/xgb_cover_fundamental.pkl")
    model_type = "fundamental"

print(f"Training {model_type} model")

X = df[feat_names].astype('float32')
y = ((df["Actual_Margin"] + df["open_visitor_spread"]) > 0).astype(int)

cut = int(len(df) * 0.80)
X_tr, y_tr = X.iloc[:cut], y.iloc[:cut]
X_te, y_te = X.iloc[cut:], y.iloc[cut:]

print(f"Training samples: {len(X_tr):,}, Features: {X_tr.shape[1]}")

tscv = TimeSeriesSplit(n_splits=5)

# Optuna objective function for hyperparameter optimization
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 50.0, log=True),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": TREE_METHOD,
        "device": DEVICE,
        "max_bin": 256,
        "random_state": 42,
        "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    log_losses = []
    for train_idx, val_idx in tscv.split(X_tr):
        X_train_fold = X_tr.iloc[train_idx]
        y_train_fold = y_tr.iloc[train_idx]
        X_val_fold = X_tr.iloc[val_idx]
        y_val_fold = y_tr.iloc[val_idx]
        
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
        preds = model.predict_proba(X_val_fold)[:, 1]
        log_loss = -np.mean(y_val_fold * np.log(preds + 1e-15) + (1 - y_val_fold) * np.log(1 - preds + 1e-15))
        log_losses.append(log_loss)
    return np.mean(log_losses)

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler, study_name="xgb_optimization")
print("Starting hyperparameter optimization...")
study.optimize(objective, n_trials=50, timeout=3600)

print("Best params:", study.best_params)

best = study.best_params | {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": TREE_METHOD,
    "device": DEVICE,
    "max_bin": 256,
    "random_state": 42,
    "n_jobs": -1,
}
model = XGBClassifier(**best)
model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

preds = model.predict_proba(X_te)[:, 1]
log_loss = -np.mean(y_te * np.log(preds + 1e-15) + (1 - y_te) * np.log(1 - preds + 1e-15))
print(f"Hold-out Log-Loss: {log_loss:0.3f}")

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_OUT)
print(f"{model_type.capitalize()} model saved to {MODEL_OUT}")
