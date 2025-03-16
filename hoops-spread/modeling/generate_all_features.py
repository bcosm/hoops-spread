
# Feature selection script for filtering out leakage columns and bad patterns

from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import sys

# Generate clean feature list by removing data leakage and problematic patterns
def main():
    try:
        DATA_FILE = Path("data/features/ncaab_cumulative_features_v13_sentiment_enhanced.csv")
        TARGET = "Actual_Margin"
        
        LEAKAGE_COLS = [
            TARGET, "visitor_rot", "home_rot",
            "open_visitor_spread", "open_home_spread",
            "close_visitor_spread", "close_home_spread",
            "close_total", "match_score",
            "visitor_school", "home_school",
            "school_name_x", "school_name_y",
            "visitor_1st_score", "visitor_2nd_score", "visitor_final_score",
            "home_1st_score", "home_2nd_score", "home_final_score",
            "Visitor_Actual_Margin", "Home_Rolling_Margin_Imputed",
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
        
        if not DATA_FILE.exists():
            print(f"ERROR: Data file not found at {DATA_FILE}")
            return False
            
        df = pd.read_csv(DATA_FILE, parse_dates=["date"], low_memory=False)
        
        numeric = df.select_dtypes(include=[np.number]).columns
        safe = [
            c for c in numeric
            if c not in LEAKAGE_COLS
            and not any(pat in c for pat in BAD_PATTERNS)
        ]
        
        additional_sentiment_features = [
            "home_sentiment_ema",
            "home_confidence_ema", 
            "visitor_sentiment_ema",
            "visitor_confidence_ema"
        ]
        
        all_features = list(set(safe + additional_sentiment_features))
        all_features.sort()
        
        print(f"Remaining numeric features: {len(safe)}")
        print(f"Total features for training: {len(all_features)}")
        
        out_dir = Path("config")
        out_dir.mkdir(exist_ok=True)
        feat_path = out_dir / "boruta_features_sentiment.txt"
        
        feat_path.write_text("\n".join(all_features))
        
        success_path = out_dir / "boruta_success_signal.txt"
        success_path.write_text(f"ALL_FEATURES: {len(all_features)} features saved at {dt.datetime.now()}")
        
        print(f"Saved {len(all_features)} features to {feat_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in feature generation: {e}")
        
        try:
            out_dir = Path("config")
            out_dir.mkdir(exist_ok=True)
            feat_path = out_dir / "boruta_features_sentiment.txt"
            
            fallback_features = [
                "home_sentiment_ema",
                "home_confidence_ema", 
                "visitor_sentiment_ema",
                "visitor_confidence_ema"
            ]
            
            feat_path.write_text("\n".join(fallback_features))
            success_path = out_dir / "boruta_success_signal.txt"
            success_path.write_text(f"FALLBACK: {len(fallback_features)} features saved at {dt.datetime.now()}")
            
            print(f"Saved fallback features to {feat_path}")
            
        except Exception as fallback_error:
            print(f"Fallback save failed: {fallback_error}")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
