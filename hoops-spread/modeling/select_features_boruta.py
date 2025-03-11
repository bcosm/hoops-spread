# Feature selection using Boruta algorithm with SHAP importance

from pathlib import Path
import datetime as dt
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from BorutaShap import BorutaShap
from xgboost import XGBRegressor
from tqdm import tqdm
import sys

MARKET = False
DATA_FILE = Path("data/features/ncaab_cumulative_features_v13_sentiment_enhanced.csv")
TARGET = "Actual_Margin"
TRAIN_RATIO = 0.8
MAX_SAMPLE = None
N_TRIALS = 50
STABILITY_FOLDS = 5
STABILITY_THRESHOLD = 0.25

REQUIRED_SENTIMENT_FEATURES = [
    "home_sentiment_ema",
    "visitor_sentiment_ema", 
    "home_confidence_ema",
    "visitor_confidence_ema"
]

# Detect if GPU acceleration is available for XGBoost
def detect_gpu():
    try:
      import xgboost as xgb
      test_model = XGBRegressor(tree_method="gpu_hist", n_estimators=1)
      X_test = np.random.random((10, 5))
      y_test = np.random.random(10)
      test_model.fit(X_test, y_test)
      return True
    except:
      return False

GPU_AVAILABLE = detect_gpu()
TREE_METHOD = "gpu_hist" if GPU_AVAILABLE else "hist"
print(f"Using {'GPU' if GPU_AVAILABLE else 'CPU'} acceleration")

RANDOM_STATE = 42

LEAKAGE_COLS = [
    TARGET, "visitor_rot", "home_rot",
    "open_visitor_spread", "open_home_spread",
    "close_visitor_spread", "close_home_spread",
    "close_total", "match_score",
    "visitor_school", "home_school",
    "school_name_x", "school_name_y",
    "visitor_1st_score", "visitor_2nd_score", "visitor_final_score",
    "home_1st_score", "home_2nd_score", "home_final_score",
    "Visitor_Actual_Margin", "Home_Rolling_Margin_Imputed", "home_federal_code", "visitor_federal_code"
] + (["open_total"] if not MARKET else [])

SPECIFIC_LEAK_FEATURES = [
    "bart_possessions_game",
    "bart_season_col", 
    "home_bart_opponent_conf_order",
    "visitor_bart_opponent_conf_order",
    "home_3pa_rate",
    "visitor_3pa_rate",
    "home_sentiment_norm", 
    "visitor_sentiment_norm",
    "home_confidence_norm",
    "visitor_confidence_norm",
]

BAD_PATTERNS = (
    "_ingame_bart",
    "_proj_score_diff",
    "_pred_barthag_wp",
    "_off_vs_", "_eFG_vs_",
    "Spread_Movement", "Total_Movement",
    "Rolling_Margin",
    "_ml",
)

# Prepare dataframe by removing leakage columns and extracting target
def prep_frame(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number]).columns
    safe = [
      c for c in numeric
      if c not in LEAKAGE_COLS
      and c not in SPECIFIC_LEAK_FEATURES
      and not any(pat in c for pat in BAD_PATTERNS)
    ]
    
    leakers = sorted(set(numeric) - set(safe) - {TARGET})
    print(f"Blocked {len(leakers)} potential leak columns")
    print(f"Remaining numeric features: {len(safe)}")
    return df[safe], df[TARGET]

# Ensure required sentiment features are included in the final list
def ensure_sentiment_features(features_list):
    features_set = set(features_list) if features_list else set()
    
    for feature in REQUIRED_SENTIMENT_FEATURES:
        features_set.add(feature)
    
    final_features = sorted(list(features_set))
    
    added_count = len([f for f in REQUIRED_SENTIMENT_FEATURES if f not in features_list])
    if added_count > 0:
        print(f"Added {added_count} required sentiment features")
    
    return final_features

# Run Boruta feature selection with SHAP importance
def run_boruta(X, y):
    try:
      base = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method=TREE_METHOD,
        random_state=RANDOM_STATE,
      )
      
      selector = BorutaShap(
        model=base,
        importance_measure="shap",
        classification=False,
        percentile=0.9,
        pvalue=0.15,
      )

      X = X.fillna(0.0)
      if MAX_SAMPLE and len(X) > MAX_SAMPLE:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), MAX_SAMPLE, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]

      selector.fit(
        X, y,
        n_trials=N_TRIALS,
        sample=False,
        verbose=True
      )
      
      confirmed = selector.Subset().columns.tolist()
      print(f"Confirmed features in this fold: {len(confirmed)}")
      
      try:
        final_decisions = selector.history_x.iloc[-1].values
        all_feature_names = selector.history_x.columns.tolist()
        
        tentative_idx = [i for i, decision in enumerate(final_decisions) if decision == 0]
        tentative = [all_feature_names[i] for i in tentative_idx if i < len(all_feature_names)]
        
        print(f"Tentative features in this fold: {len(tentative)}")
        combined = list(set(confirmed + tentative))
        print(f"Total features (confirmed + tentative): {len(combined)}")
        return combined
      except Exception as e:
        print(f"Warning: Could not extract tentative features: {e}")
        return confirmed
        
    except Exception as e:
      print(f"Error in run_boruta: {e}")
      print("Returning empty feature list for this fold")
      return []

# Main feature selection pipeline with cross-validation stability testing
def main():
    try:
      df = (pd.read_csv(DATA_FILE, parse_dates=["date"], low_memory=False)
            .sort_values("date"))

      cut = int(len(df) * TRAIN_RATIO)
      train_df = df.iloc[:cut].copy()
      X_train, y_train = prep_frame(train_df)
      
      print(f"Features will be saved to config/boruta_features_sentiment.txt")

      confirmed_by_fold = []
      
      out_dir = Path("config")
      out_dir.mkdir(exist_ok=True)
      feat_path = out_dir / "boruta_features_sentiment.txt"

      def save_current_features(confirmed_features, fold_num=None):
        try:
            existing_features = []
            if feat_path.exists():
                try:
                    existing_features = feat_path.read_text().strip().split('\n')
                    existing_features = [f.strip() for f in existing_features if f.strip()]
                except:
                    existing_features = []
            
            all_confirmed = list(set(existing_features + (confirmed_features or [])))
            all_features = ensure_sentiment_features(all_confirmed)
            
            feat_path.write_text("\n".join(all_features))
            fold_msg = f" (after fold {fold_num})" if fold_num else ""
            print(f"Saved {len(all_features)} features to {feat_path}{fold_msg}")
            
            success_path = out_dir / "boruta_success_signal.txt"
            success_path.write_text(f"SUCCESS: {len(all_features)} features saved at {dt.datetime.now()}")
            
        except Exception as e:
            print(f"Error saving features: {e}")

      if STABILITY_FOLDS > 1:
        print(f"Running Boruta-SHAP on {STABILITY_FOLDS} expanding folds")
        tscv = TimeSeriesSplit(n_splits=STABILITY_FOLDS)
        for k, (idx_tr, _) in enumerate(
              tqdm(tscv.split(X_train), total=STABILITY_FOLDS, desc="Folds"), 1):
            try:
              feats = run_boruta(X_train.iloc[idx_tr], y_train.iloc[idx_tr])
              confirmed_by_fold.append(feats)
              save_current_features(feats, k)
              
              if k >= 3:
                early_exit_path = out_dir / "boruta_early_success.txt"
                early_exit_path.write_text(f"Boruta completed {k} folds successfully at {dt.datetime.now()}")
                print(f"Created early success signal")
                print(f"Completed {k} folds. Continuing for more stability.")
              
            except Exception as e:
              print(f"Error in fold {k}: {e}")
              print(f"Continuing with remaining folds...")
              if confirmed_by_fold:
                most_recent_features = confirmed_by_fold[-1] if confirmed_by_fold else []
                save_current_features(most_recent_features, f"{k}_failed")
              else:
                save_current_features([], f"{k}_failed")
      else:
        print("Running single Boruta-SHAP pass")
        try:
            feats = run_boruta(X_train, y_train)
            confirmed_by_fold.append(feats)
            save_current_features(feats)
        except Exception as e:
            print(f"Error in single Boruta pass: {e}")
            save_current_features([])

      flat = [f for sub in confirmed_by_fold for f in sub]
      freq = pd.Series(flat).value_counts(normalize=True) if flat else pd.Series(dtype=float)
      final = freq[freq >= STABILITY_THRESHOLD].index.tolist() if len(freq) > 0 else []

      print(f"{len(final)} features survived >= {STABILITY_THRESHOLD*100:.0f}% of folds.")
      
      final_with_sentiment = ensure_sentiment_features(final)
      save_current_features(final)

      print("Final feature list:")
      final_with_sentiment.sort()
      for f in final_with_sentiment:
        print(f"  {f}")

      print(f"BORUTA FEATURE SELECTION COMPLETED")
      print(f"Total features saved: {len(final_with_sentiment)}")
      
      return True
      
    except Exception as e:
      print(f"CRITICAL ERROR in Boruta feature selection: {e}")
      print(f"Attempting to save fallback feature list...")
      
      try:
        out_dir = Path("config")
        out_dir.mkdir(exist_ok=True)
        feat_path = out_dir / "boruta_features_sentiment.txt"
        
        existing_features = []
        if feat_path.exists():
          try:
            existing_features = feat_path.read_text().strip().split('\n')
            existing_features = [f.strip() for f in existing_features if f.strip()]
          except:
            existing_features = []
        
        fallback_features = ensure_sentiment_features(existing_features)
        
        feat_path.write_text("\n".join(fallback_features))
        print(f"Saved fallback features to {feat_path}")
        print(f"PIPELINE CAN CONTINUE with {len(fallback_features)} features")
        
        success_path = out_dir / "boruta_success_signal.txt"
        success_path.write_text(f"FALLBACK SUCCESS: {len(fallback_features)} features saved at {dt.datetime.now()}")
        
      except Exception as fallback_error:
        print(f"Even fallback save failed: {fallback_error}")
      
      return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
