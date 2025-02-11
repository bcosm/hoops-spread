# Execute the complete data collection pipeline in the correct order
import subprocess
import sys
from pathlib import Path

# Run all data collection scripts in the proper order to build the complete dataset
def run_data_collection_pipeline():   
    script_dir = Path(__file__).parent
    
    scripts_order = [
        "fetch_schedules.py",
        "calculate_elo.py", 
        "fetch_full_bart_torvik.py",
        "assign_school_codes.py",
        "merge_team_stats.py",
        "merge_additional_columns.py",
        "clean_data.py",
        "arena_assignment.py",
        "altitude_and_arena_merge.py"
    ]
    
    for script in scripts_order:
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
    success = run_data_collection_pipeline()
    sys.exit(0 if success else 1)
