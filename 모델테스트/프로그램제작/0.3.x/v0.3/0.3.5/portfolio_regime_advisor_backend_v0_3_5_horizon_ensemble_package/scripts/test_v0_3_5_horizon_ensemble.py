from backend.app.model.horizon_ensemble import HorizonEnsemble


def test_highvol_gate_aware_ensemble_drops_failed_20d():
    ens = HorizonEnsemble()
    probs = {
        "highvol_5D": 0.60,
        "highvol_10D": 0.70,
        "highvol_20D": 0.20,
    }
    gates = {
        "highvol_5D": {"status": "PASS"},
        "highvol_10D": {"status": "PASS"},
        "highvol_20D": {"status": "FAIL"},
    }
    res = ens.combine_family("highvol", probs, gates)
    expected = (0.60 * 0.35 + 0.70 * 0.45) / (0.35 + 0.45)
    assert abs(res.probability - expected) < 1e-9
    assert "highvol_20D" in res.fallback_heads
    assert res.weights["20D"] == 0.0


def test_up_down_strength_ensembles_are_available():
    ens = HorizonEnsemble()
    probs = {
        "up_strength_5D": 0.62,
        "up_strength_10D": 0.58,
        "up_strength_20D": 0.40,
        "down_strength_5D": 0.30,
        "down_strength_10D": 0.35,
        "down_strength_20D": 0.80,
    }
    gates = {
        "up_strength_5D": {"status": "PASS"},
        "up_strength_10D": {"status": "PASS"},
        "up_strength_20D": {"status": "FAIL"},
        "down_strength_5D": {"status": "PASS"},
        "down_strength_10D": {"status": "UNCERTAIN"},
        "down_strength_20D": {"status": "FAIL"},
    }
    out = ens.combine_all(probs, gates, families=["up_strength", "down_strength"])
    assert out["up_strength"].probability > 0.58
    assert out["down_strength"].probability < 0.36
    assert "up_strength_20D" in out["up_strength"].fallback_heads
    assert "down_strength_20D" in out["down_strength"].fallback_heads


if __name__ == "__main__":
    test_highvol_gate_aware_ensemble_drops_failed_20d()
    test_up_down_strength_ensembles_are_available()
    print("v0.3.5 horizon ensemble tests passed")
