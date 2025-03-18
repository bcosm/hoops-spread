
# Initial backtesting script for NCAA basketball spread models

from pathlib import Path
import pandas as pd, numpy as np, joblib
from scipy.stats import norm
from tqdm import tqdm
from collections import defaultdict
import sys
import traceback

# Run backtest simulation with betting strategy and performance metrics
def main():
    try:
        MARKET = False
        DATA = Path("data/features/ncaab_cumulative_features_v13_sentiment_enhanced.csv")
        
        model_suffix = "market" if MARKET else "fundamental"
        
        model_name = f"xgb_cover_{model_suffix}.pkl"
        model_file = Path(f"models/{model_name}")
        if not model_file.exists():
            print(f"ERROR: Model file not found at {model_file}")
            return False
            
        MODEL = joblib.load(model_file)
        
        if hasattr(MODEL, 'feature_names_in_'):
            FEATS = list(MODEL.feature_names_in_)
        elif hasattr(MODEL, 'get_booster') and hasattr(MODEL.get_booster(), 'feature_names'):
            FEATS = MODEL.get_booster().feature_names
        else:
            features_file = Path("config/boruta_features_sentiment.txt")
            if not features_file.exists():
                print(f"ERROR: Features file not found at {features_file}")
                return False
                
            FEATS = features_file.read_text().splitlines()
            FEATS = [f.strip() for f in FEATS if f.strip()]
            
            if MARKET:
                if "open_total" not in FEATS:
                    FEATS.append("open_total")
            else:
                FEATS = [f for f in FEATS if f != "open_total"]

        EDGE_THRESHOLD = 0.04
        USE_KELLY = True
        SIGMA_MARGIN = 1.33
        DEC_ODDS = 1.909
        BOOTSTRAPS = 500
        
        def implied_cover_prob(edge):
            return norm.cdf(edge / SIGMA_MARGIN)

        def stake_size(edge_prob):
            if not USE_KELLY:
                return 1.0
            return 1.0

        def profit(row):
            if row.signal == 0 or row.stake == 0:
                return 0.0
            visitor_cover = (row.Actual_Margin + row.market_spread) > 0
            win = visitor_cover if row.signal == 1 else not visitor_cover
            return row.stake * (DEC_ODDS - 1) if win else -row.stake

        if not DATA.exists():
            print(f"ERROR: Data file not found at {DATA}")
            return False
            
        df = pd.read_csv(DATA, parse_dates=["date"], low_memory=False).sort_values("date")

        required_cols = ["date", "Actual_Margin", "open_visitor_spread"] + FEATS
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Missing required columns in data: {missing_cols}")
            return False

        cut = int(len(df) * 0.80)
        test = df.iloc[cut:].copy()
        
        X_test = test[FEATS]
        vig = (DEC_ODDS - 1)
        MODEL.set_params(device="cpu")
        test["cover_prob"] = MODEL.predict_proba(X_test)[:, 1]        
        test["market_spread"] = test["open_visitor_spread"]
        p_break = 1 / (1 + (DEC_ODDS - 1))
        test["edge_prob"] = test.cover_prob - p_break

        test["signal"] = np.where(test.edge_prob >= EDGE_THRESHOLD, 1,
                           np.where(test.edge_prob <= -EDGE_THRESHOLD, -1, 0))

        if USE_KELLY:
            b = DEC_ODDS - 1
            def kelly_stake(row):
                if row.signal == 0:
                    return 0.0
                elif row.signal == 1:
                    p = row.cover_prob
                    f = (b * p - (1 - p)) / b
                    return max(0, 0.5 * f)
                else:
                    p = 1 - row.cover_prob
                    f = (b * p - (1 - p)) / b
                    return max(0, 0.5 * f)
            
            test["stake"] = test.apply(kelly_stake, axis=1)
        else:
            test["stake"] = test.signal.abs().astype(float)

        tqdm.pandas(desc="Calculating P&L")
        test["pl"] = test.progress_apply(profit, axis=1)

        bets = test[test.stake > 0].copy()

        if USE_KELLY and not bets.empty:
            mean_stake = bets['stake'].mean()
            bets['stake'] = bets['stake'] / mean_stake
        n_bet = len(bets)

        if n_bet > 0:
            hit = (bets.pl > 0).mean()
            roi = bets.pl.sum() / bets.stake.sum()
            draw = (bets.pl.cumsum().cummax() - bets.pl.cumsum()).max()
        else:
            hit = 0.0
            roi = 0.0
            draw = 0.0

        print(f"Bets: {n_bet} | Hit rate: {hit:.3%} | ROI: {roi:.3%} | Max drawdown: {draw:.1f}u")

        if n_bet > 0:
            bets_with_edge = bets.copy()
            bets_with_edge['edge_bucket'] = pd.cut(bets_with_edge['edge_prob'], 
                                                   bins=[-1, -0.05, -0.03, 0.03, 0.05, 1], 
                                                   labels=['<-5%', '-5% to -3%', '±3%', '3% to 5%', '>5%'])
            
            edge_analysis = bets_with_edge.groupby('edge_bucket').agg({
                'pl': ['count', 'sum', 'mean'],
                'stake': 'sum'
            }).round(3)
            
            edge_analysis.columns = ['Bets', 'Total_PL', 'Avg_PL', 'Total_Stakes']
            edge_analysis['ROI'] = (edge_analysis['Total_PL'] / edge_analysis['Total_Stakes']).round(4)
            edge_analysis['Hit_Rate'] = bets_with_edge.groupby('edge_bucket')['pl'].apply(lambda x: (x > 0).mean()).round(4)
            
            print(edge_analysis)

        if n_bet > 0:
            bets_yearly = bets.copy()
            bets_yearly['year'] = bets_yearly['date'].dt.year
            yearly_stats = bets_yearly.groupby('year').agg({
                'pl': ['count', 'sum'],
                'stake': 'sum'
            })
            yearly_stats.columns = ['Bets', 'Total_PL', 'Total_Stakes']
            yearly_stats['ROI'] = (yearly_stats['Total_PL'] / yearly_stats['Total_Stakes']).round(4)
            yearly_stats['Hit_Rate'] = bets_yearly.groupby('year')['pl'].apply(lambda x: (x > 0).mean()).round(4)
            
            print(yearly_stats)

        if n_bet > 50:
            bootstrap_rois = []
            
            for i in tqdm(range(BOOTSTRAPS), desc="Bootstrap"):
                sample_bets = bets.sample(n=len(bets), replace=True)
                boot_roi = sample_bets['pl'].sum() / sample_bets['stake'].sum()
                bootstrap_rois.append(boot_roi)
            
            bootstrap_rois = np.array(bootstrap_rois)
            ci_lower = np.percentile(bootstrap_rois, 2.5)
            ci_upper = np.percentile(bootstrap_rois, 97.5)
            
            print(f"Bootstrap ROI 95% CI: [{ci_lower:+.3%}, {ci_upper:+.3%}]")

        try:
            import shap
            
            sample_size = min(1000, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            
            explainer = shap.Explainer(MODEL)
            shap_values = explainer(X_sample)
            
            global_importance = pd.DataFrame({
                'feature': FEATS,
                'importance': np.abs(shap_values.values).mean(0)
            }).sort_values('importance', ascending=False)
            
            print("Top 10 Most Important Features:")
            print(global_importance.head(10))
            
            results_dir = Path("backtest_results")
            results_dir.mkdir(exist_ok=True)
            
            global_importance.to_csv(results_dir / f"global_importance_{model_suffix}.csv", index=False)
            
            if n_bet > 0:
                bet_indices = bets.index.intersection(X_sample.index)
                if len(bet_indices) > 10:
                    bet_shap_values = shap_values.values[X_sample.index.isin(bet_indices)]
                    bet_features_pnl = pd.DataFrame({
                        'feature': FEATS,
                        'avg_shap': np.mean(bet_shap_values, axis=0),
                        'total_contribution': np.sum(bet_shap_values, axis=0)
                    }).sort_values('total_contribution', ascending=False)
                    
                    print("Top Features Contributing to Profitable Bets:")
                    print(bet_features_pnl.head(10))
                    
                    bet_features_pnl.to_csv(results_dir / f"feature_pnl_attribution_{model_suffix}.csv", index=False)
            
        except ImportError:
            pass
        except Exception as e:
            print(f"Error in SHAP analysis: {e}")

        results_dir = Path("backtest_results")
        results_dir.mkdir(exist_ok=True)
        
        results_summary = {
            "model_type": model_suffix,
            "overall_roi": float(roi),
            "hit_rate": float(hit),
            "n_bets": int(n_bet),
            "max_drawdown": float(draw),
            "edge_threshold": EDGE_THRESHOLD,
            "use_kelly": USE_KELLY,
            "staking_method": "Half-Kelly" if USE_KELLY else "Flat 1u"
        }
        
        if n_bet > 50:
            results_summary.update({
                "bootstrap_roi_ci_lower": float(ci_lower),
                "bootstrap_roi_ci_upper": float(ci_upper),
                "bootstrap_roi_mean": float(np.mean(bootstrap_rois)),
                "bootstrap_roi_std": float(np.std(bootstrap_rois))
            })
        
        import json
        with open(results_dir / f"backtest_summary_{model_suffix}.json", "w") as f:
            json.dump(results_summary, f, indent=2)
        
        if n_bet > 0:
            bets_output = bets[['date', 'signal', 'edge_prob', 'cover_prob', 'stake', 'pl']].copy()
            bets_output.to_csv(results_dir / f"bet_history_{model_suffix}.csv", index=False)
        
        return True
    except Exception as e:
        print(f"ERROR in backtesting: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
