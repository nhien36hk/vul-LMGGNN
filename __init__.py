import sys
import os

def setup_kaggle_path():
    """Setup path cho Kaggle environment"""
    if os.path.exists('/kaggle/input'):
        kaggle_path = '/kaggle/input/lm-train/LMTrain'
        if kaggle_path not in sys.path:
            sys.path.insert(0, kaggle_path)
            print(f"✅ Added Kaggle path: {kaggle_path}")
    else:
        print("❌ Kaggle input is not found")
setup_kaggle_path()
