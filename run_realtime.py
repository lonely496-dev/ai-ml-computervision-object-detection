#!/usr/bin/env python
"""
Thin wrapper script for src/run_realtime.py.
Allows calling from project root while preserving backward compatibility.

Usage:
    python run_realtime.py [args]
    python run_realtime.py --low_power --frame_step 2
"""
import sys
import os

# Add src/ to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.run_realtime import main

if __name__ == "__main__":
    main()