import sys
import os
sys.path.insert(0, os.path.abspath('.'))

try:
    from agents.amadeuszero import MyAgent
    print("Successfully imported MyAgent from agents.amadeuszero")
except Exception as e:
    print(f"Failed to import MyAgent: {e}")
