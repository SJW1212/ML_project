import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import app

client = TestClient(app)

for url in [
    "/health",
    "/dashboard",
    "/dashboard?model_mode=auto&assets=QQQ,SPY,AAPL&horizon=10D&provider=auto",
    "/models/active",
    "/models/runtime-status?assets=QQQ,SPY,AAPL",
    "/models/artifact-inventory",
]:
    r = client.get(url)
    print(url, r.status_code)
    assert r.status_code == 200

r = client.post("/models/infer", json={"tickers": ["QQQ"], "horizon": "10D", "provider": "auto", "market": "US"})
print("/models/infer", r.status_code, r.json().get("ok"), r.json().get("errors"))
assert r.status_code == 200
print("v0.3 runtime smoke test passed")
