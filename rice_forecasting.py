"""Monthly Philippine rice-price forecasting utilities. No SARIMA or Naive Bayes."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import clone

MONTHS = {name: i for i, name in enumerate(
    ['January','February','March','April','May','June','July','August','September','October','November','December'], 1)}
LAGS = [1, 2, 3, 6, 12]
MODEL_NAMES = ['Naive Forecast', 'Random Forest', 'Extra Trees', 'Ridge Regression', 'XGBoost']


def load_prices(path_or_df):
    """Load the cleaned long-format CSV/DataFrame and create a Date column."""
    df = path_or_df.copy() if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        if 'Month_Num' not in df.columns:
            df['Month_Num'] = df['Month'].map(MONTHS) if 'Month' in df.columns else pd.NA
        df['Date'] = pd.to_datetime(dict(year=df['Year'], month=df['Month_Num'], day=1), errors='coerce')
    required = ['Region', 'Rice_Type', 'Date', 'Price_PHP_per_kg']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"This function needs the cleaned long-format dataset. Missing columns: {', '.join(missing)}")
    df['Price_PHP_per_kg'] = pd.to_numeric(df['Price_PHP_per_kg'], errors='coerce')
    df = df.dropna(subset=required)
    df = df[df['Date'] <= pd.Timestamp('2026-09-01')]
    df = df.sort_values(['Region','Rice_Type','Date']).drop_duplicates(['Region','Rice_Type','Date'], keep='last')
    return df.reset_index(drop=True)


def _features_from_history(region, rice_type, hist, target_date):
    vals = np.asarray(hist, dtype=float)
    row = {'Region': region, 'Rice_Type': rice_type,
           'target_year': int(target_date.year), 'target_month': int(target_date.month),
           'month_sin': float(np.sin(2*np.pi*target_date.month/12)),
           'month_cos': float(np.cos(2*np.pi*target_date.month/12)),
           'time_index': int(target_date.year*12 + target_date.month)}
    for lag in LAGS:
        row[f'lag_{lag}'] = float(vals[-lag])
    for w in [3, 6, 12]:
        row[f'rollmean_{w}'] = float(np.mean(vals[-w:]))
    row['rollstd_3'] = float(np.std(vals[-3:], ddof=0))
    row['change_1'] = float(vals[-1] - vals[-2])
    row['change_12'] = float(vals[-1] - vals[-12])
    return row


def make_supervised(prices):
    """Create one-step-ahead rows using past values only."""
    df = load_prices(prices)
    rows = []
    for (region, rice_type), group in df.groupby(['Region','Rice_Type']):
        group = group.sort_values('Date').reset_index(drop=True)
        vals = group['Price_PHP_per_kg'].astype(float).to_numpy()
        for i in range(12, len(group)):
            target_date = pd.Timestamp(group.loc[i, 'Date'])
            row = _features_from_history(region, rice_type, vals[:i], target_date)
            row['target_date'] = target_date
            row['target_price'] = float(vals[i])
            rows.append(row)
    if not rows:
        raise ValueError('Not enough monthly history to create training rows; each series needs at least 13 observations.')
    return pd.DataFrame(rows).sort_values('target_date').reset_index(drop=True)


def feature_columns(frame=None):
    return ['Region','Rice_Type','target_year','target_month','month_sin','month_cos','time_index',
            'lag_1','lag_2','lag_3','lag_6','lag_12','rollmean_3','rollmean_6','rollmean_12',
            'rollstd_3','change_1','change_12']


def build_model(kind='Random Forest', random_state=42):
    categorical = ['Region','Rice_Type']
    numeric = [c for c in feature_columns() if c not in categorical]
    prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric)
    ])
    if kind == 'Random Forest':
        estimator = RandomForestRegressor(n_estimators=350, min_samples_leaf=2, max_features=0.9,
                                          random_state=random_state, n_jobs=-1)
    elif kind == 'Extra Trees':
        estimator = ExtraTreesRegressor(n_estimators=350, min_samples_leaf=2, max_features=0.9,
                                        random_state=random_state, n_jobs=-1)
    elif kind == 'Ridge Regression':
        estimator = Ridge(alpha=10.0)
        prep = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric)
        ])
    elif kind == 'XGBoost':
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError('XGBoost is not installed. Run: pip install xgboost') from exc
        estimator = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.035,
                                 subsample=0.9, colsample_bytree=0.9, objective='reg:squarederror',
                                 random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f'Unknown model: {kind}. Choose from Random Forest, Extra Trees, Ridge Regression, XGBoost.')
    return Pipeline([('prep', prep), ('model', estimator)])


def split_by_time(supervised, test_months=12):
    frame = supervised.copy()
    frame['target_date'] = pd.to_datetime(frame['target_date'])
    dates = sorted(frame['target_date'].dropna().unique())
    if len(dates) <= test_months:
        raise ValueError('Not enough unique months for this test window.')
    cutoff = pd.Timestamp(dates[-test_months])
    return frame[frame.target_date < cutoff].copy(), frame[frame.target_date >= cutoff].copy(), cutoff


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    nz = np.abs(y_true) > 1e-8
    mape = float(np.mean(np.abs((y_true[nz]-y_pred[nz])/y_true[nz]))*100) if nz.any() else np.nan
    return {'MAE': mae, 'RMSE': rmse, 'MAPE_percent': mape}


def evaluate_models(prices, test_months=12, include_xgboost=True, include_sarima=False):
    """Compare a naive baseline and regression models on a chronological holdout."""
    sup = make_supervised(prices)
    train, test, cutoff = split_by_time(sup, test_months)
    feats = feature_columns()
    results, predictions, fitted = [], [], {}
    naive_pred = test['lag_1'].to_numpy(dtype=float)
    results.append({'Model':'Naive Forecast', **metrics(test.target_price, naive_pred)})
    tmp = test[['Region','Rice_Type','target_date','target_price']].copy()
    tmp['Model'] = 'Naive Forecast'; tmp['Predicted_Price'] = naive_pred; predictions.append(tmp)
    kinds = ['Random Forest', 'Extra Trees', 'Ridge Regression'] + (['XGBoost'] if include_xgboost else [])
    for kind in kinds:
        model = build_model(kind)
        model.fit(train[feats], train.target_price)
        pred = model.predict(test[feats])
        results.append({'Model':kind, **metrics(test.target_price, pred)})
        tmp = test[['Region','Rice_Type','target_date','target_price']].copy()
        tmp['Model'] = kind; tmp['Predicted_Price'] = pred; predictions.append(tmp)
        fitted[kind] = model
    return (pd.DataFrame(results).sort_values('MAE').reset_index(drop=True),
            pd.concat(predictions, ignore_index=True),
            {'train':train, 'test':test, 'cutoff':cutoff, 'supervised':sup, 'fitted_models':fitted})


def forecast_future(prices, region, rice_type, months=6, model_kind='Random Forest'):
    """Train the selected regression model and recursively forecast future months."""
    clean = load_prices(prices)
    series = clean[(clean.Region == region) & (clean.Rice_Type == rice_type)].sort_values('Date')
    if series.empty:
        raise ValueError('No historical data for the selected region and rice type.')
    if len(series) < 13:
        raise ValueError('At least 13 monthly observations are needed to train a model.')
    if model_kind == 'Naive Forecast':
        # This baseline intentionally repeats the last observed value; it is a benchmark, not a learned model.
        last_date = pd.Timestamp(series.Date.max())
        last_price = float(series.Price_PHP_per_kg.iloc[-1])
        rows = [{'Date': last_date + pd.DateOffset(months=i), 'Region':region, 'Rice_Type':rice_type,
                 'Forecast_Price_PHP_per_kg':last_price, 'Horizon_Month':i} for i in range(1, int(months)+1)]
        return pd.DataFrame(rows), None
    sup = make_supervised(clean)
    model = build_model(model_kind)
    feats = feature_columns()
    model.fit(sup[feats], sup.target_price)
    hist = series.Price_PHP_per_kg.astype(float).tolist()
    last_date = pd.Timestamp(series.Date.max())
    out = []
    for step in range(1, int(months)+1):
        target_date = last_date + pd.DateOffset(months=step)
        row = _features_from_history(region, rice_type, hist, target_date)
        X = pd.DataFrame([row], columns=feats)
        pred = max(0.0, float(model.predict(X)[0]))
        out.append({'Date':target_date, 'Region':region, 'Rice_Type':rice_type,
                    'Forecast_Price_PHP_per_kg':pred, 'Horizon_Month':step})
        hist.append(pred)
    return pd.DataFrame(out), model
