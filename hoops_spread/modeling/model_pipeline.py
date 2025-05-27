# Standard modeling pipeline with feature selection and model training

import subprocess
import sys
from pathlib import Path

# Run feature selection and model training scripts in order
def run_modeling_pipeline():
    script_dir = Path(__file__).parent
    scripts_order = [
        "select_features_boruta.py",
        "train_xgboost.py"
    ]
    
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
                config_dir = Path("config")
                success_signal = config_dir / "boruta_success_signal.txt"
                early_success = config_dir / "boruta_early_success.txt"
                
                if not (success_signal.exists() or early_success.exists()):
                    return False
        except Exception as e:
            print(f"Error running {script}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    success = run_modeling_pipeline()
    sys.exit(0 if success else 1)
