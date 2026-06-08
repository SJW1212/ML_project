from fastapi.testclient import TestClient
from backend.app.main import app


def main():
    client = TestClient(app)
    for url in [
        "/health",
        "/dashboard",
        "/dashboard?model_mode=auto",
        "/models/active",
        "/models/runtime-status",
        "/models/artifact-inventory",
    ]:
        resp = client.get(url)
        assert resp.status_code == 200, (url, resp.status_code, resp.text[:500])
    print("v0.3.6 runtime smoke test passed")


if __name__ == "__main__":
    main()
