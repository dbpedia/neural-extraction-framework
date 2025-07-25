#!/usr/bin/env python3
"""
Wrapper script to run entity-linking integration tests with CLI arguments.
This script provides an easy way to run both quick and comprehensive tests.
"""

import sys
import os
import subprocess
import argparse

def run_quick_test(api_key: str = None):
    """Run the quick test"""
    print("🚀 Running Quick Test...")
    print("=" * 40)
    
    cmd = [sys.executable, "quick_test.py"]
    if api_key:
        cmd.extend(["--api-key", api_key])
    
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    return result.returncode == 0

def run_full_test(api_key: str = None, csv_path: str = None):
    """Run the full test suite"""
    print("🔧 Running Full Test Suite...")
    print("=" * 40)
    
    cmd = [sys.executable, "test_entity_linking_integration.py"]
    if api_key:
        cmd.extend(["--api-key", api_key])
    if csv_path:
        cmd.extend(["--csv-path", csv_path])
    
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    return result.returncode == 0

def main():
    """Main function with CLI argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Run entity-linking integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test with API key
  python run_tests.py quick --api-key YOUR_API_KEY
  
  # Run full test with API key and CSV
  python run_tests.py full -k YOUR_API_KEY --csv-path Data/surface_forms.csv
  
  # Run both tests
  python run_tests.py both -k YOUR_API_KEY
        """
    )
    
    parser.add_argument(
        'test_type',
        choices=['quick', 'full', 'both'],
        help='Type of test to run: quick, full, or both'
    )
    
    parser.add_argument(
        '-k', '--api-key',
        type=str,
        help='Google API key for Gemini LLM (or set GOOGLE_API_KEY environment variable)'
    )
    
    parser.add_argument(
        '--csv-path',
        type=str,
        help='Path to CSV context file (for full test only)'
    )
    
    args = parser.parse_args()
    
    print("🔧 Entity-Linking Integration Test Runner")
    print("=" * 50)
    
    success = True
    
    if args.test_type in ['quick', 'both']:
        success &= run_quick_test(args.api_key)
        print()
    
    if args.test_type in ['full', 'both']:
        success &= run_full_test(args.api_key, args.csv_path)
        print()
    
    print("=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 