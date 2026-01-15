# Quick Start - Dashboard

## With Virtual Environment Activated

Since you have `(venv)` active in your terminal, run:

```powershell
python -m streamlit run dashboard.py
```

## What to Expect

1. **First time**: Streamlit may ask for email - just press **Enter** to skip
2. **Dashboard starts**: You'll see:
   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   ```
3. **Browser opens**: Dashboard should open automatically
4. **If not**: Manually go to `http://localhost:8501`

## Troubleshooting

### If you get "streamlit not found":
```powershell
# Install in venv
pip install streamlit
```

### If models don't load:
- Check model paths in sidebar
- Default paths:
  - Detection: `runs/train/artifacts_quick2/weights/best.pt`
  - Erosion: `erosion_xgboost_regression.pkl`

### If dashboard doesn't open:
- Check firewall settings
- Try different port: `streamlit run dashboard.py --server.port 8502`

## Dashboard Features

✅ Upload images for analysis
✅ View segmentation, detection, and erosion predictions
✅ Batch processing
✅ Interactive configuration

