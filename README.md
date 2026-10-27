# RiceCast PH — Rice Price Forecasting Project

## Included files
- `app.py` — Streamlit dashboard.
- `rice_forecasting.py` — forecasting and evaluation functions.
- `Rice_Price_Forecasting_Updated_Models.ipynb` — Jupyter Notebook workflow.
- `rice_prices_cleaned.csv` — cleaned historical rice price data.
- `rice_prices_per_series_supervised.csv` — prepared supervised dataset.
- `requirements.txt` — Python dependencies.

## Models
- Naive Forecast baseline (previous month's price)
- Random Forest Regressor
- Extra Trees Regressor
- Ridge Regression
- XGBoost Regressor

Naive Bayes and SARIMA are excluded. The Naive Forecast is a baseline, not a trained regression model.

## Run locally in VS Code
1. Extract this ZIP and open the extracted folder in VS Code.
2. Create a virtual environment: `python -m venv .venv`
3. Activate it in PowerShell: `.\.venv\Scripts\Activate.ps1`
4. Install dependencies: `python -m pip install -r requirements.txt`
5. Start the dashboard: `python -m streamlit run app.py`

Keep the Python files and CSV files together in this folder. Forecast outputs are estimates and should be evaluated against held-out historical data before use.

## Run the notebook
Open `Rice_Price_Forecasting_Updated_Models.ipynb` in Jupyter or VS Code and run cells from top to bottom. Make sure the CSV file(s) required by the notebook are in the same working folder.
