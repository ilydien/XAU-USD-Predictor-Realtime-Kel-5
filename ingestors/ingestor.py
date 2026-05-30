import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timezone
from kafka import KafkaProducer
import redis
import psycopg2
import time
import os

TICKERS = os.getenv("TICKERS", "XAUUSD=X,DX-Y.NYB,^VIX,^GSPC,CL=F").split(",")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
DRAGONFLY_HOST = os.getenv("DRAGONFLY_HOST", "dragonfly")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://gold:gold@postgres:5432/golddb")
FETCH_INTERVAL = int(os.getenv("FETCH_INTERVAL", "60"))

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v, default=str).encode(),
    acks=1,
)

r = redis.Redis(host=DRAGONFLY_HOST, port=6379, decode_responses=True)


def get_db():
    while True:
        try:
            conn = psycopg2.connect(POSTGRES_DSN)
            return conn, conn.cursor()
        except Exception as e:
            print(f"[ingestor] Waiting for PostgreSQL: {e}")
            time.sleep(3)


conn, cur = get_db()


def fetch_and_publish():
    df = yf.download(
        TICKERS,
        period="1d",
        interval="1m",
        group_by="ticker",
        progress=False,
        auto_adjust=True,
    )

    timestamp = datetime.now(timezone.utc)
    data = {}

    for ticker in TICKERS:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                ticker_df = df[ticker]
            else:
                ticker_df = df

            if ticker_df.empty:
                continue

            last = ticker_df.dropna().iloc[-1]
            data[ticker] = {
                "close": float(last["Close"]),
                "open": float(last["Open"]),
                "high": float(last["High"]),
                "low": float(last["Low"]),
                "volume": int(last["Volume"]) if not pd.isna(last["Volume"]) else 0,
            }
        except (KeyError, IndexError, ValueError, TypeError):
            continue

    if not data or "XAUUSD=X" not in data:
        print(f"[ingestor] No gold data at {timestamp}")
        return

    message = {"timestamp": timestamp.isoformat(), "data": data}

    producer.send("market-data", value=message)
    print(f"[ingestor] Kafka: gold={data['XAUUSD=X']['close']}")

    stream_entry = {"timestamp": timestamp.isoformat()}
    for ticker, ticker_data in data.items():
        stream_entry[ticker] = json.dumps(ticker_data)
    r.xadd("market-data", stream_entry, maxlen=10000)

    r.set("latest:gold:close", data["XAUUSD=X"]["close"])
    r.set("latest:gold:timestamp", timestamp.isoformat())
    if "DX-Y.NYB" in data:
        r.set("latest:dxy:close", data["DX-Y.NYB"]["close"])
    if "^VIX" in data:
        r.set("latest:vix:close", data["^VIX"]["close"])

    for ticker, ticker_data in data.items():
        cur.execute(
            """
            INSERT INTO market_data (timestamp, ticker, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """,
            (
                timestamp,
                ticker,
                ticker_data["open"],
                ticker_data["high"],
                ticker_data["low"],
                ticker_data["close"],
                ticker_data["volume"],
            ),
        )
    conn.commit()
    print(f"[ingestor] PostgreSQL: inserted {len(data)} tickers")


while True:
    try:
        fetch_and_publish()
    except Exception as e:
        print(f"[ingestor] Error: {e}")
        time.sleep(5)
    time.sleep(FETCH_INTERVAL)
