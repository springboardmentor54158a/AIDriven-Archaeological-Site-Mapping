"""
Quick launcher for the Streamlit dashboard
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Run dashboard
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.py")
    print(f"Starting dashboard at: {dashboard_path}")
    print("Dashboard will open in your browser at http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])

