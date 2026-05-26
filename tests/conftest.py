import sys
import os

# Add the parent directory (src) to sys.path so tests can find the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
