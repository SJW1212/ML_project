from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

for path in ["/health", "/models/active", "/settings/presets", "/validation", "/dashboard"]:
    r = client.get(path)
    print(path, r.status_code)
    assert r.status_code == 200, r.text

payload = client.get("/dashboard", params={"assets":"QQQ,SPY,AAPL,SOXX,NVDA", "horizon":"10D", "capital_mode":"equal"}).json()
print("as_of_date", payload.get("as_of_date"))
print("portfolio_totals", payload.get("portfolio_totals"))
print("signals", len(payload.get("latest_signals", [])))
