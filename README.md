# F1 Next Race Predictor (VS Code Setup)

## Quickstart
1. Create a virtual environment (Windows PowerShell):
   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Open this folder in VS Code and select the Python interpreter: **.venv**.
3. Run with the built-in debugger (see launch configurations) or use the terminal.

## CLI usage
```bash
python f1_predict_next_race.py
python f1_predict_next_race.py --use-qual
```

Outputs a CSV in the workspace with the predicted finishing order.
FastF1 cache is stored in `fastf1_cache/` (created on first run).
