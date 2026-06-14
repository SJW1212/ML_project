from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def check_get(path: str):
    res = client.get(path)
    print(path, res.status_code)
    assert res.status_code == 200, res.text
    return res.json()


def main():
    health = check_get("/health")
    assert health["version"] == "0.3.5"
    dashboard = check_get("/dashboard")
    assert dashboard["validation"]["fail_count"] == 0, dashboard["validation"]
    auto = check_get("/dashboard?model_mode=auto&assets=QQQ,SPY,AAPL&horizon=10D&provider=auto")
    assert auto["model_mode"] in {"prediction_file_fallback", "live_inference"}
    active = check_get("/models/active")
    assert "active_model" in active
    runtime = check_get("/models/runtime-status?assets=QQQ,SPY,AAPL")
    assert "runtime_ready" in runtime
    inventory = check_get("/models/artifact-inventory")
    assert "versions" in inventory
    infer_payload = {
        "tickers": ["QQQ"],
        "horizon": "10D",
        "provider": "auto",
        "market": "US",
        "model_version": "candidate_missing_test",
    }
    res = client.post("/models/infer", json=infer_payload)
    print("/models/infer", res.status_code, res.json().get("ok"), res.json().get("errors"))
    assert res.status_code == 200
    assert res.json()["ok"] is False
    print("v0.3.5 runtime smoke test passed")


if __name__ == "__main__":
    main()
