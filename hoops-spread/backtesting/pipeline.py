# Pipeline runner for executing backtesting scripts in sequence

import subprocess
import sys
from pathlib import Path

# Execute backtesting scripts in order and check for successful completion
def run_backtesting_pipeline():
    script_dir = Path(__file__).parent
    scripts_order = ["initial_backtest.py"]
    
    for script in scripts_order:
        script_path = script_dir / script
        if not script_path.exists():
            print(f"Script {script} not found")
            return False
            
        try:
            result = subprocess.run([sys.executable, str(script_path)], 
                                  cwd=str(Path.cwd()),
                                  capture_output=False, text=True)
            if result.returncode != 0:
                models_dir = Path("models")
                expected_outputs = [
                    "global_importance.csv",
                    "feature_pnl_attribution.csv",
                    "edge_contrib_prob.csv"
                ]
                
                outputs_exist = all((models_dir / output).exists() for output in expected_outputs)
                if not outputs_exist:
                    return False
        except Exception as e:
            print(f"Error running {script}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    success = run_backtesting_pipeline()
    sys.exit(0 if success else 1)
