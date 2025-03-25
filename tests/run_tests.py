# Test runner script for executing unittest suites with pattern matching

import unittest
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Discover and run all tests matching the given pattern
def run_tests(test_pattern="test_*.py", verbosity=2):
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern=test_pattern)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()

# Run tests from a specific module by name
def run_specific_module(module_name, verbosity=2):  
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(module_name)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests")
    parser.add_argument('--module', '-m', help='Run specific module')
    parser.add_argument('--pattern', '-p', default='test_*.py', help='Test file pattern')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    verbosity = 2 if args.verbose else 1
    
    if args.module:
        success = run_specific_module(args.module, verbosity)
    else:
        success = run_tests(args.pattern, verbosity)
    
    sys.exit(0 if success else 1)
