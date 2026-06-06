import json
import os
from datetime import datetime, timezone
from kafka import KafkaProducer
import yfinance as yf

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
WS_TICKERS = os.getenv("WS_TICKERS", "DX-Y.NYB").split(",")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v, default=str).encode(),
    acks=1,
)


def handler(msg):
    ticker = msg.get("id")
    price = msg.get("price")
    ts = msg.get("ts")
    if not ticker or not price:
        return

    data = {
        "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        if isinstance(ts, (int, float))
        else datetime.now(timezone.utc).isoformat(),
        "data": {ticker: {"close": float(price), "source": "ws"}},
    }

    producer.send("market-data", value=data)
    print(f"[ws_listener] {ticker}: {price}")


print(f"[ws_listener] Connecting to WebSocket, tickers={WS_TICKERS}")


def listen_forever():
    while True:
        try:
            ws = yf.WebSocket(verbose=False)
            ws.subscribe(WS_TICKERS)
            ws.listen(handler)
        except Exception as e:
            print(f"[ws_listener] Disconnected: {e}, reconnecting in 5s...")
            import time

            time.sleep(5)


listen_forever()
