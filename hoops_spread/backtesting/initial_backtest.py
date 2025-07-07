# Initial backtesting script for NCAA basketball spread models

from pathlib import Path
import pandas as pd, numpy as np, joblib
from scipy.stats import norm
from tqdm import tqdm
from xgboost import XGBClassifier     
from collections import defaultdict
import sys
import traceback

# Run backtest simulation with betting strategy and performance metrics
def main():
    try:
        MARKET = True
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

        BANKROLL_START = 100.0  
        
        def implied_cover_prob(edge):
            return norm.cdf(edge / SIGMA_MARGIN)

        if not DATA.exists():
            print(f"ERROR: Data file not found at {DATA}")
            return False
            
        df = pd.read_csv(DATA, parse_dates=["date"], low_memory=False).sort_values("date")

        required_cols = ["date",
                         "Actual_Margin",
                         "open_visitor_spread",
                         "close_visitor_spread"] + FEATS
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Missing required columns in data: {missing_cols}")
            return False

        HORIZON  = pd.Timedelta(days=30)   
        MIN_TRAIN = pd.Timedelta(days=365)    

        target = ((df["Actual_Margin"] + df["open_visitor_spread"]) > 0).astype(int)
        walk_chunks = []
        cut_date = df["date"].min() + MIN_TRAIN

        while cut_date + HORIZON <= df["date"].max():
            train_mask = df["date"] <= cut_date
            test_mask  = (df["date"] > cut_date) & (df["date"] <= cut_date + HORIZON)
            if not test_mask.any():
                cut_date += HORIZON
                continue

            refit = XGBClassifier(**MODEL.get_params())
            refit.set_params(device="cpu")
            refit.fit(df.loc[train_mask, FEATS], target[train_mask])

            tmp = df.loc[test_mask].copy()
            tmp["cover_prob"] = refit.predict_proba(tmp[FEATS])[:, 1]
            walk_chunks.append(tmp)

            cut_date += HORIZON

        test = pd.concat(walk_chunks).sort_values("date").reset_index(drop=True)
        vig = (DEC_ODDS - 1)
        test["market_spread"] = test["open_visitor_spread"]
        p_break = 1 / (1 + (DEC_ODDS - 1))
        test["edge_prob"] = test.cover_prob - p_break

        test["signal"] = np.where(test.edge_prob >= EDGE_THRESHOLD, 1,
                              np.where(test.edge_prob <= -EDGE_THRESHOLD, -1, 0))

        test["clv"] = (
            test["open_visitor_spread"] - test["close_visitor_spread"]
        ) * test["signal"]

        if USE_KELLY:
            b = DEC_ODDS - 1
            p = test['cover_prob']
            long_f  = (b * p - (1 - p)) / b          
            short_f = (b * (1 - p) - p) / b       
            stake = np.zeros(len(test))
            mask_long  = test['signal'] == 1
            mask_short = test['signal'] == -1
            stake[mask_long]  = np.maximum(0, 0.5 * long_f[mask_long])
            stake[mask_short] = np.maximum(0, 0.5 * short_f[mask_short])
            test['stake'] = stake
        else:
            test["stake"] = test.signal.abs().astype(float)

        visitor_cover = (test['Actual_Margin'] + test['market_spread']) > 0
        win = (visitor_cover & (test['signal'] == 1)) | (~visitor_cover & (test['signal'] == -1))
        test['pl'] = np.where(win, test['stake'] * (DEC_ODDS - 1), -test['stake'])

        bets = test[test.stake > 0].copy()
        mean_clv = bets["clv"].mean() if not bets.empty else 0.0

        bets["cum_pl"] = bets["pl"].cumsum()
        bets["bankroll"] = BANKROLL_START + bets["cum_pl"]

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

        daily = bets.groupby(bets["date"].dt.normalize()).agg(
            pl    = ('pl',    'sum'),
            stake = ('stake', 'sum')
        )
        daily = daily[daily['stake'] > 0]

        daily['bankroll'] = BANKROLL_START + daily['pl'].cumsum().shift(fill_value=0)
        daily_ret = daily['pl'] / daily['bankroll']

        period_years = ((bets['date'].iloc[-1] - bets['date'].iloc[0]).days
                        / 365.25) if n_bet else 0
        bet_day_freq = len(daily_ret) / period_years if period_years > 0 else 0

        sharpe = (daily_ret.mean() / daily_ret.std(ddof=0) *
                  np.sqrt(bet_day_freq)) if daily_ret.std(ddof=0) > 0 else 0.0

        period_years = ((bets["date"].iloc[-1] - bets["date"].iloc[0]).days /
                        365.25) if n_bet else 0
        final_bankroll = BANKROLL_START + bets["pl"].sum()
        cagr = ((final_bankroll / BANKROLL_START) ** (1 / period_years) - 1) if period_years > 0 else 0.0
        turnover = bets["stake"].sum() / BANKROLL_START / period_years if period_years > 0 else 0.0
        print(f"Bets: {n_bet} | Hit {hit:.3%} | Bet-ROI {roi:.3%} | "
              f"CAGR {cagr:.3%} | Turnover {turnover:.1f}× | "
              f"CLV {mean_clv:+.2f} | Sharpe {sharpe:.2f} | "
              f"Max DD {draw:.1f}u")

        if n_bet > 0:
            bets_with_edge = bets.copy()
            bets_with_edge['edge_bucket'] = pd.cut(bets_with_edge['edge_prob'], 
                                                   bins=[-1, -0.05, -0.03, 0.03, 0.05, 1], 
                                                   labels=['<-5%', '-5% to -3%', '±3%', '3% to 5%', '>5%'])
            
            edge_analysis = bets_with_edge.groupby('edge_bucket', observed=False).agg({
                'pl': ['count', 'sum', 'mean'],
                'stake': 'sum'
            }).round(3)
            
            edge_analysis.columns = ['Bets', 'Total_PL', 'Avg_PL', 'Total_Stakes']
            edge_analysis['ROI'] = (edge_analysis['Total_PL'] / edge_analysis['Total_Stakes']).round(4)
            edge_analysis['Hit_Rate'] = bets_with_edge.groupby('edge_bucket', observed=False)['pl'].apply(lambda x: (x > 0).mean()).round(4)
            
            print(edge_analysis)

        if n_bet > 0:
            bets_yearly = bets.copy()
            bets_yearly['year'] = bets_yearly['date'].dt.year
            yearly_stats = bets_yearly.groupby('year', observed=False).agg({
                'pl': ['count', 'sum'],
                'stake': 'sum'
            })
            yearly_stats.columns = ['Bets', 'Total_PL', 'Total_Stakes']
            yearly_stats['ROI'] = (yearly_stats['Total_PL'] / yearly_stats['Total_Stakes']).round(4)
            yearly_stats['Hit_Rate'] = bets_yearly.groupby('year', observed=False)['pl'].apply(lambda x: (x > 0).mean()).round(4)
            
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
            
            X_test = test[FEATS]
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
            "staking_method": "Half-Kelly" if USE_KELLY else "Flat 1u",
            "mean_clv": float(mean_clv),
            "sharpe": float(sharpe),
            "cagr": float(cagr),
            "turnover": float(turnover)
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
            bets_output = bets[['date', 'signal', 'edge_prob',
                                'cover_prob', 'stake', 'pl',
                                'clv', 'bankroll']].copy()
            bets_output.to_csv(results_dir / f"bet_history_{model_suffix}.csv", index=False)
        
        return True
    except Exception as e:
        print(f"ERROR in backtesting: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)