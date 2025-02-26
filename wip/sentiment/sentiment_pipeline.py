# Orchestrates complete sentiment analysis pipeline for Reddit college basketball data
import subprocess
import sys
from pathlib import Path

# Executes all sentiment analysis scripts in proper sequence
def run_sentiment_pipeline():  
    script_dir = Path(__file__).parent
    
    scripts = [
        "dedup_school_codes.py",
        "school_aliases.py", 
        "alias_deduper.py",
        "scrape_subs.py",
        "extract_school_mentions_vectorized.py",
        "timeseries_sentiment_analysis_memory_efficient.py"
    ]
    
    for script in scripts:
        script_path = script_dir / script
        if script_path.exists():
            result = subprocess.run([sys.executable, str(script_path)], 
                                  cwd=str(Path.cwd()))
            if result.returncode != 0:
                print(f"Error running {script}")
                return False
        else:
            print(f"Script not found: {script}")
            return False
    
    return True

def analyze_sentiment():
    # TODO: implement
    pass

if __name__ == "__main__":
    success = run_sentiment_pipeline()
    sys.exit(0 if success else 1)
