#!/usr/bin/env python
"""
Thin wrapper script for src/run_on_image.py.
Allows calling from project root while preserving backward compatibility.

Usage:
    python run_on_image.py --input media/sample.jpg
    python run_on_image.py --input media/ --save
"""
import sys
import os

# Add src/ to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.run_on_image import main

if __name__ == "__main__":
    main()
