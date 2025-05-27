# Command line interface for hoops spread pipelines

import argparse
import sys

from hoops_spread.modeling.model_pipeline import run_modeling_pipeline
from hoops_spread.modeling.model_pipeline_full import run_modeling_pipeline_full_features
from hoops_spread.backtesting.pipeline import run_backtesting_pipeline

# Parse command line arguments and run chosen pipeline
def main():
    parser = argparse.ArgumentParser(description='NCAA Basketball Point Spread Prediction')
    
    parser.add_argument('pipeline', choices=['data', 'features', 'sentiment', 'modeling', 'modeling-full', 'backtest', 'all'])
    
    args = parser.parse_args()
    
    try:
        if args.pipeline == 'modeling':
            success = run_modeling_pipeline()
        elif args.pipeline == 'modeling-full':
            success = run_modeling_pipeline_full_features()
        elif args.pipeline == 'backtest':
            success = run_backtesting_pipeline()
        elif args.pipeline == 'all':
            success = (
                run_modeling_pipeline() and
                run_backtesting_pipeline()
            )
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
