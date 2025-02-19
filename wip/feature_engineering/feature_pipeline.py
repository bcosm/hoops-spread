# Orchestrates feature engineering pipeline by running scripts in sequence
import subprocess
import sys
from pathlib import Path

# Executes all feature engineering scripts in proper order
def run_pipeline():
    script_dir = Path(__file__).parent
    
    scripts = [
        "cum_pre_game_stats.py",
        "coach_start_date.py",
        "additional_coaching_stats.py", 
        "fix_first_season_and_total_seasons_at_team.py",
        "remove_nan_specific.py",
        "data_nan_cleaner.py",
        "advanced_derived_features.py",
        "fix_duplicate_school_names_inplace.py",
        "fix_duplicate_team_names_inplace.py",
        "verify_state_normalization_inplace.py",
        "clean_team_names.py",
        "verify_team_name_cleanup.py",
        "append_sentiment.py"
    ]
    
    for script in scripts:
        script_path = script_dir / script
        if script_path.exists():
            try:
                result = subprocess.run([sys.executable, str(script_path)], 
                                      cwd=str(Path.cwd()),
                                      capture_output=False, text=True)
                if result.returncode != 0:
                    return False
            except Exception:
                return False
        else:
            return False
    
    return True

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
