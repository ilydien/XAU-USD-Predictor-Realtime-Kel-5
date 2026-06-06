import yfinance as yf
import pandas as pd
import numpy as np
import json
import joblib
import os
import redis
from datetime import datetime, timezone
from kafka import KafkaProducer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

TICKERS = ["DX-Y.NYB", "EURUSD=X", "USDJPY=X", "GBPUSD=X", "^VIX", "^GSPC"]
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
DRAGONFLY_HOST = os.getenv("DRAGONFLY_HOST", "dragonfly")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/dxy_model.pkl")
DATA_PATH = os.getenv("DATA_PATH", "/app/models/training_data.pkl")

TICKER_MAP = {
    "DX-Y.NYB": "dxy",
    "EURUSD=X": "eur",
    "USDJPY=X": "jpy",
    "GBPUSD=X": "gbp",
    "^VIX": "vix",
    "^GSPC": "sp500",
}

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v, default=str).encode(),
    acks=1,
)

r = redis.Redis(host=DRAGONFLY_HOST, port=6379, decode_responses=True)


def extract_ticker(df, ticker):
    try:
        if isinstance(df.columns, pd.MultiIndex):
            tdf = df[ticker].dropna()
        else:
            tdf = df.dropna()
        return tdf if not tdf.empty else None
    except (KeyError, IndexError):
        return None


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


print("[training] Fetching daily data...")
df = yf.download(
    TICKERS, period="2y", interval="1d", group_by="ticker", progress=False
)

ticker_data = {}
for ticker in TICKERS:
    tdf = extract_ticker(df, ticker)
    if tdf is not None:
        ticker_data[ticker] = tdf
        print(f"[training] {ticker}: {len(tdf)} rows")

dxy = ticker_data.get("DX-Y.NYB")
if dxy is None or dxy.empty:
    print("[training] No DXY data fetched, aborting")
    exit(1)

all_dates = dxy.index.sort_values()
rows = []
for date in all_dates:
    row = {"date": date}
    for ticker, tdf in ticker_data.items():
        if date in tdf.index:
            prefix = TICKER_MAP.get(ticker, ticker)
            trow = tdf.loc[date]
            row[f"{prefix}_close"] = float(trow["Close"])
            row[f"{prefix}_high"] = float(trow["High"])
            row[f"{prefix}_low"] = float(trow["Low"])
    rows.append(row)

new_df = pd.DataFrame(rows).set_index("date")
new_df["dxy_volatility"] = new_df["dxy_high"] - new_df["dxy_low"]

if os.path.exists(DATA_PATH):
    train_df = joblib.load(DATA_PATH)
    combined = pd.concat([train_df, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
else:
    combined = new_df.sort_index()
    print("[training] No saved training data, starting fresh")


LAG_COLS = [
    "eur_close", "jpy_close", "gbp_close",
    "vix_close", "sp500_close", "dxy_close",
]
for col in LAG_COLS:
    if col not in combined.columns:
        combined[col] = 0
    for i in range(1, 4):
        combined[f"{col}_lag{i}"] = combined[col].shift(i)

combined["dxy_volatility_lag1"] = combined["dxy_volatility"].shift(1)

for prefix in ["dxy", "eur", "jpy", "gbp", "vix", "sp500"]:
    combined[f"{prefix}_return"] = combined[f"{prefix}_close"].pct_change()

combined["dxy_ma5"] = combined["dxy_close"].rolling(5).mean()
combined["dxy_ma10"] = combined["dxy_close"].rolling(10).mean()
combined["dxy_rsi"] = compute_rsi(combined["dxy_close"])

TECH_COLS = (
    [f"{p}_return" for p in ["dxy", "eur", "jpy", "gbp", "vix", "sp500"]]
    + ["dxy_ma5", "dxy_ma10", "dxy_rsi"]
)

FEATURE_COLS = (
    [f"{c}_lag{i}" for c in LAG_COLS for i in range(1, 4)]
    + ["dxy_volatility_lag1"]
    + TECH_COLS
)

train_data = combined.dropna()

if len(train_data) >= 3:
    X_cols = [c for c in FEATURE_COLS if c in train_data.columns]
    X = train_data[X_cols].values
    y = np.log1p(train_data["dxy_volatility"].values)

    tscv = TimeSeriesSplit(n_splits=min(3, len(train_data) // 2))

    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ])
    ridge_grid = {"model__alpha": [0.01, 0.1, 1, 10, 100, 1000]}
    gs = GridSearchCV(ridge_pipeline, ridge_grid, cv=tscv, scoring="r2")
    gs.fit(X, y)
    ridge_model = gs.best_estimator_

    ridge_cv_r2 = gs.best_score_
    ridge_pred = ridge_model.predict(X)
    ridge_r2 = r2_score(y, ridge_pred)
    ridge_mae = mean_absolute_error(y, ridge_pred)
    print(f"[training] Ridge (alpha={gs.best_params_['model__alpha']}) — In-sample R²: {ridge_r2:.4f} | CV R²: {ridge_cv_r2:.4f} | MAE: {ridge_mae:.4f}")

    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=42)),
    ])
    rf_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5, 10],
    }
    rf_gs = GridSearchCV(rf_pipeline, rf_grid, cv=tscv, scoring="r2", n_jobs=1)
    rf_gs.fit(X, y)
    rf_model = rf_gs.best_estimator_
    rf_cv_r2 = rf_gs.best_score_
    rf_pred = rf_model.predict(X)
    rf_r2 = r2_score(y, rf_pred)
    rf_mae = mean_absolute_error(y, rf_pred)
    print(f"[training] RandomForest {rf_gs.best_params_} — In-sample R²: {rf_r2:.4f} | CV R²: {rf_cv_r2:.4f} | MAE: {rf_mae:.4f}")

    if ridge_cv_r2 >= rf_cv_r2:
        model = ridge_model
        print(f"[training] Using Ridge (CV R²={ridge_cv_r2:.4f})")
    else:
        model = rf_model
        print(f"[training] Using RandomForest (CV R²={rf_cv_r2:.4f})")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(combined, DATA_PATH)

    print(f"[training] Model saved: {type(model).__name__} trained on {len(train_data)} rows")
else:
    print(f"[training] Insufficient data ({len(train_data)} rows), loading existing model")
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        print("[training] No model available, using heuristic")
        model = None

latest = combined.iloc[-1:].copy()
if "dxy_volatility_lag1" not in latest.columns:
    latest["dxy_volatility_lag1"] = combined["dxy_volatility"].iloc[-2] if len(combined) >= 2 else 0

for col in TECH_COLS:
    if col not in latest.columns:
        latest[col] = float(combined[col].iloc[-1]) if col in combined.columns else 0

for col in LAG_COLS:
    for i in range(1, 4):
        lagcol = f"{col}_lag{i}"
        if lagcol not in latest.columns:
            idx = -i
            if len(combined) >= abs(idx):
                latest[lagcol] = combined[col].iloc[idx]
            else:
                latest[lagcol] = 0

if model is not None:
    pred_X = latest[[c for c in FEATURE_COLS if c in latest.columns]].values
    if not np.any(np.isnan(pred_X)):
        predicted_vol = float(np.expm1(model.predict(pred_X)[0]))
    else:
        predicted_vol = float(latest["dxy_volatility"].iloc[0]) if "dxy_volatility" in latest.columns else 0
else:
    predicted_vol = float(latest["dxy_volatility"].iloc[0]) if "dxy_volatility" in latest.columns else 0

actual_vol = float(latest["dxy_volatility"].iloc[0]) if "dxy_volatility" in latest.columns else None
dxy_close = float(latest["dxy_close"].iloc[0]) if "dxy_close" in latest.columns else 0

features = {}
for col in latest.columns:
    if col != "date":
        features[col] = float(latest[col].iloc[0])

result = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "dxy_close": round(dxy_close, 4),
    "predicted_volatility": round(max(predicted_vol, 0), 4),
    "actual_volatility": round(actual_vol, 4) if actual_vol is not None else None,
    "source": "batch",
    "features": {k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()},
}

producer.send("dxy-predictions", value=result)
print(f"[training] Prediction sent: predicted_vol={predicted_vol:.4f} actual_vol={actual_vol}")

updates = {}
for prefix_key in ["eur", "jpy", "gbp", "vix", "sp500", "dxy"]:
    close_val = features.get(f"{prefix_key}_close")
    if close_val is not None:
        updates[f"latest:{prefix_key}:close"] = str(close_val)

updates["latest:dxy:high"] = str(features.get("dxy_high", ""))
updates["latest:dxy:low"] = str(features.get("dxy_low", ""))
updates["latest:dxy:volatility"] = str(features.get("dxy_volatility", ""))
updates["latest:dxy:timestamp"] = result["timestamp"]

with r.pipeline() as pipe:
    for k, v in updates.items():
        pipe.set(k, v)
    pipe.execute()

print(f"[training] Dragonfly cache updated with {len(updates)} keys")
