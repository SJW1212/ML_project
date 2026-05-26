# Python 3.10+
# 필요 패키지:
# pip install pandas numpy yfinance scikit-learn matplotlib

"""
v3 주요 개선사항 (우선순위 순)
────────────────────────────────────────────────────────────
1순위  라벨 재설계       : direction → 위험조정수익률 기준, 횡보 범위 확대
2순위  데이터 누수 제거   : purge gap 추가, scaler/분위수 학습구간 내부만
3순위  피처 재구성       : 시장 국면 5그룹(추세/변동성/모멘텀/위험회피/시장폭)
4순위  클래스 불균형     : class_weight=balanced_subsample 유지 + SMOTE 선택 가능
5순위  확률 보정        : CalibratedClassifierCV(sigmoid)
6순위  임곗값 튜닝       : 목적별 threshold 분리 (breakdown precision 우선)
7순위  모델 앙상블       : LogisticRegression + HistGradientBoosting + RF + Rule
8순위  국면 결정 로직    : 조합 조건으로 breakdown 오탐 방지
────────────────────────────────────────────────────────────
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, brier_score_loss
)
from sklearn.pipeline import Pipeline


# ═══════════════════════════════════════════════
# 1. 기본 설정
# ═══════════════════════════════════════════════

TARGET_TICKER       = "QQQ"
TARGET_TICKER_LOWER = TARGET_TICKER.lower()
BOND_TICKER         = "IEF"
CASH_TICKER         = "BIL"

DATA_PATH   = f"data/{TARGET_TICKER_LOWER}_features_labeled.csv"
RESULT_DIR  = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

RESULT_IS_CSV   = os.path.join(RESULT_DIR, f"{TARGET_TICKER_LOWER}_v3_IS.csv")
RESULT_OOS_CSV  = os.path.join(RESULT_DIR, f"{TARGET_TICKER_LOWER}_v3_OOS.csv")
SUMMARY_JSON    = os.path.join(RESULT_DIR, f"{TARGET_TICKER_LOWER}_v3_summary.json")

INITIAL_CAPITAL = 100_000_000

# horizon gap: 미래 라벨 누수 방지
DIRECTION_HORIZON = 60   # direction 라벨 미래 거래일
PURGE_GAP         = 20   # 추가 purge gap (라벨 겹침 방지)
TOTAL_GAP         = DIRECTION_HORIZON + PURGE_GAP  # 실제 train 종료 위치

MIN_TRAIN_SIZE      = 1500
TRANSACTION_COST_RATE = 0.001

BACKTEST_START_DATE = "2013-01-01"

# OOS 분리
IS_END_DATE    = "2020-12-31"
OOS_START_DATE = "2021-01-01"

# Ridge 자산배분 가중치 학습
RIDGE_ALPHA              = 1.0
MIN_WEIGHT_TRAIN_MONTHS  = 24

STOCK_WEIGHT_MIN = 0.20
STOCK_WEIGHT_MAX = 0.90

RANDOM_STATE = 42

# ── 6순위: 목적별 임곗값 ──────────────────────────────
THRESHOLDS = {
    "up":        0.45,   # direction 상승: 적당히 낮게 (상승 포착)
    "down":      0.40,   # direction 하락: 낮게 (손실 방어 우선)
    "high_vol":  0.50,   # risk: 기본
    "rebound":   0.60,   # rebound: precision 우선 → 높게
    "breakdown": 0.65,   # breakdown: precision 최우선 → 매우 높게
}

# ── 국면 결정 조합 조건 ────────────────────────────────
BREAKDOWN_COMBO = {
    "p_breakdown": 0.65,
    "p_high_vol":  0.50,
    "p_down":      0.35,
}


# ═══════════════════════════════════════════════
# 2. 유틸 함수
# ═══════════════════════════════════════════════

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"데이터 파일이 없습니다: {path}\n"
            "먼저 step1_make_dataset.py를 실행하세요."
        )
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    for col in ["Date", "Close"]:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼 누락: {col}")
    return df


def make_future_matrix(series: pd.Series, horizon: int) -> pd.DataFrame:
    return pd.concat([series.shift(-i) for i in range(1, horizon + 1)], axis=1)


def ensure_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """미래/과거 타깃 컬럼 생성 (피처로 사용 금지)."""
    df = df.copy()
    if "daily_return" not in df.columns:
        df["daily_return"] = df["Close"].pct_change()

    df["future_return_20d"] = df["Close"].shift(-20) / df["Close"] - 1
    df["future_return_60d"] = df["Close"].shift(-60) / df["Close"] - 1

    future_returns_20 = make_future_matrix(df["daily_return"], 20)
    future_returns_60 = make_future_matrix(df["daily_return"], 60)

    df["future_volatility_20d"] = future_returns_20.std(axis=1)
    df["future_volatility_60d"] = future_returns_60.std(axis=1)

    future_close_20 = make_future_matrix(df["Close"], 20)
    future_close_60 = make_future_matrix(df["Close"], 60)

    df["future_min_return_20d"] = future_close_20.min(axis=1) / df["Close"] - 1
    df["future_max_return_20d"] = future_close_20.max(axis=1) / df["Close"] - 1
    df["future_min_return_60d"] = future_close_60.min(axis=1) / df["Close"] - 1

    df["past_return_40d"] = df["Close"] / df["Close"].shift(40) - 1

    # ── 1순위: 위험조정수익률 ──────────────────────────────
    # 분모 0 방지 및 최소 변동성 보정
    vol_floor = df["future_volatility_20d"].quantile(0.10)
    vol_60_floor = df["future_volatility_60d"].quantile(0.10)

    df["future_risk_adj_return_60d"] = (
        df["future_return_60d"] /
        df["future_volatility_60d"].clip(lower=vol_60_floor)
    )
    df["future_risk_adj_return_20d"] = (
        df["future_return_20d"] /
        df["future_volatility_20d"].clip(lower=vol_floor)
    )

    return df


# ═══════════════════════════════════════════════
# 3순위. 피처 재구성: 시장 국면 5그룹
# ═══════════════════════════════════════════════

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    5개 그룹으로 구성된 시장 국면 피처.
    각 피처는 "현재 장세가 어떤가?"에 답해야 함.
    """
    groups = {
        # A. 추세 피처: 현재 추세가 강한가?
        "trend": [
            "return_5d", "return_20d", "return_60d", "return_120d",
            "price_ma_5_gap", "price_ma_20_gap", "price_ma_60_gap",
            "ma_gap_5_20", "ma_gap_20_60", "ma_gap_60_120",
        ],
        # B. 변동성 피처: 변동성이 커지고 있는가?
        "volatility": [
            "volatility_20d", "volatility_60d",
            "downside_volatility_20d",
            "volatility_ratio_5_20", "volatility_ratio_20_60",
            "drawdown",
        ],
        # C. 모멘텀 둔화 피처: 하락 압력이 누적되는가?
        "momentum": [
            "return_5d_zscore", "return_20d_zscore",
            "trend_slope_20", "trend_slope_60",
            "positive_return_ratio_20",
            "close_to_20d_high", "close_to_60d_high",
            "price_position_20", "price_position_60",
        ],
        # D. 위험회피 피처: 채권/현금 대비 주식 매력도
        "risk_appetite": [
            "volume_ratio_20", "volume_zscore_20", "volume_change",
            "daily_return", "log_return",
        ],
        # E. 추가 피처 (있으면 사용)
        "extra": [
            "return_3d", "return_10d",
        ],
    }

    all_candidates = [col for cols in groups.values() for col in cols]
    feature_cols = [
        col for col in all_candidates
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        raise ValueError("사용 가능한 피처가 없습니다.")

    return feature_cols


# ═══════════════════════════════════════════════
# 1순위. 라벨 재설계
# ═══════════════════════════════════════════════

def make_dynamic_labels(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    [1순위] direction 라벨을 위험조정수익률 기준으로 재설계
    - 단순 수익률 분위수 → 위험조정수익률(return/volatility) 분위수
    - 횡보 범위 확대: 상위 25% / 하위 25% / 중간 50%
    - 절대 조건(> 0) 제거 → 하락장 편향 방지

    [2순위] scaler/분위수는 반드시 train_df 내부만 사용
    """
    result = target_df.copy()

    # ── direction: 위험조정수익률 기준 ──────────────────────
    # 횡보 50%로 확대 (상위 25%, 하위 25%)
    up_q   = train_df["future_risk_adj_return_60d"].quantile(0.75)
    down_q = train_df["future_risk_adj_return_60d"].quantile(0.25)

    result["direction_label"] = "횡보"
    result.loc[result["future_risk_adj_return_60d"] >= up_q,   "direction_label"] = "상승"
    result.loc[result["future_risk_adj_return_60d"] <= down_q, "direction_label"] = "하락"

    # ── risk 라벨 ──────────────────────────────────────────
    vol_q        = train_df["future_volatility_20d"].quantile(0.80)
    min_return_q = train_df["future_min_return_20d"].quantile(0.20)

    result["risk_label"] = "정상"
    result.loc[
        (result["future_volatility_20d"] >= vol_q) |
        (result["future_min_return_20d"] <= min_return_q),
        "risk_label"
    ] = "고변동"

    # ── rebound / breakdown: 위험조정수익률 기준으로 변경 ──────
    past_down_q = train_df["past_return_40d"].quantile(0.30)
    past_up_q   = train_df["past_return_40d"].quantile(0.70)

    # 위험조정 기준으로 반등/급락 필터링 → precision 향상 기대
    future_rebound_q   = train_df["future_risk_adj_return_20d"].quantile(0.70)
    future_breakdown_q = train_df["future_risk_adj_return_20d"].quantile(0.30)

    result["rebound_flag"] = np.where(
        (result["past_return_40d"]           <= past_down_q) &
        (result["future_risk_adj_return_20d"] >= future_rebound_q),
        1, 0
    )
    result["breakdown_flag"] = np.where(
        (result["past_return_40d"]           >= past_up_q) &
        (result["future_risk_adj_return_20d"] <= future_breakdown_q),
        1, 0
    )

    return result


# ═══════════════════════════════════════════════
# 7순위. 모델 앙상블 + 5순위 확률 보정
# ═══════════════════════════════════════════════

def make_ensemble_pipeline(n_classes: int) -> Pipeline:
    """
    LogisticRegression + HistGradientBoosting + RandomForest 앙상블
    → CalibratedClassifierCV(sigmoid)로 확률 보정
    """
    lr = LogisticRegression(
        C=0.5, max_iter=1000, class_weight="balanced",
        solver="lbfgs", random_state=RANDOM_STATE
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, min_samples_leaf=30,
        learning_rate=0.05, random_state=RANDOM_STATE
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=50,
        max_features="sqrt", class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1
    )

    # 앙상블 (soft voting = 확률 평균)
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("hgb", hgb), ("rf", rf)],
        voting="soft",
        weights=[0.35, 0.35, 0.30],
    )

    # 5순위: sigmoid 확률 보정 (샘플 적을 때 안전)
    calibrated = CalibratedClassifierCV(ensemble, method="sigmoid", cv=3)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),          # 2순위: scaler도 학습구간 내부만
        ("model",   calibrated),
    ])

    return pipeline


def fit_model(train_df: pd.DataFrame, feature_cols: list[str], target_col: str):
    clean = train_df.dropna(subset=[target_col]).copy()
    if clean.empty or clean[target_col].nunique() < 2:
        return None

    X = clean[feature_cols]
    y = clean[target_col]
    n_classes = y.nunique()

    pipeline = make_ensemble_pipeline(n_classes)
    try:
        pipeline.fit(X, y)
    except Exception as e:
        print(f"  모델 학습 실패 ({target_col}): {e}")
        return None

    return pipeline


def predict_proba_dict(pipeline, row: pd.DataFrame, feature_cols: list[str]) -> dict:
    if pipeline is None:
        return {}
    try:
        X = row[feature_cols]
        proba = pipeline.predict_proba(X)[0]
        classes = pipeline.classes_
        return {str(classes[i]): float(proba[i]) for i in range(len(classes))}
    except Exception:
        return {}


def get_prob(proba_dict: dict, label, default: float = 0.0) -> float:
    return float(proba_dict.get(str(label), default))


# ═══════════════════════════════════════════════
# 단순 추세 규칙 모델 (앙상블에 10% 가중치)
# ═══════════════════════════════════════════════

def rule_based_direction(row: pd.Series) -> dict:
    """
    단순 이동평균 추세 규칙.
    모델이 불안정할 때 앙상블 안전판 역할.
    """
    ma5_gap  = row.get("price_ma_5_gap",  0)
    ma20_gap = row.get("price_ma_20_gap", 0)
    ma60_gap = row.get("price_ma_60_gap", 0)

    score = (
        0.2 * np.sign(ma5_gap)
        + 0.4 * np.sign(ma20_gap)
        + 0.4 * np.sign(ma60_gap)
    )

    if score > 0.5:
        return {"상승": 0.60, "횡보": 0.30, "하락": 0.10}
    elif score < -0.5:
        return {"상승": 0.10, "횡보": 0.30, "하락": 0.60}
    else:
        return {"상승": 0.25, "횡보": 0.50, "하락": 0.25}


def blend_with_rule(model_proba: dict, rule_proba: dict, rule_weight: float = 0.10) -> dict:
    """모델 확률과 규칙 확률을 블렌딩."""
    keys = set(model_proba) | set(rule_proba)
    blended = {}
    for k in keys:
        mp = model_proba.get(k, 0.0)
        rp = rule_proba.get(k, 0.0)
        blended[k] = (1 - rule_weight) * mp + rule_weight * rp

    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}
    return blended


# ═══════════════════════════════════════════════
# 8순위. 국면 결정 로직 (조합 조건)
# ═══════════════════════════════════════════════

def decide_market_regime(
    p_up: float, p_neutral: float, p_down: float,
    p_high_vol: float, p_rebound: float, p_breakdown: float
) -> str:
    """
    breakdown 오탐 방지를 위해 조합 조건 사용.
    우선순위: 붕괴위험 > 고변동 > 하락 > 반등 > 상승 > 횡보 > 불확실
    """
    # 1. 강한 붕괴위험 (조합 조건 필수)
    if (p_breakdown >= BREAKDOWN_COMBO["p_breakdown"]
            and p_high_vol >= BREAKDOWN_COMBO["p_high_vol"]
            and p_down    >= BREAKDOWN_COMBO["p_down"]):
        return "붕괴위험"

    # 2. 고변동
    if p_high_vol >= 0.60:
        return "고변동"

    # 3. 하락
    if p_down >= THRESHOLDS["down"] and p_up < 0.35:
        return "하락"

    # 4. 반등 (하락 우려 없을 때만)
    if p_rebound >= THRESHOLDS["rebound"] and p_down < 0.35:
        return "반등"

    # 5. 상승 (위험 없을 때만)
    if p_up >= THRESHOLDS["up"] and p_high_vol < 0.50:
        return "상승"

    # 6. 횡보
    if p_neutral >= 0.45:
        return "횡보"

    # 7. 불확실
    return "불확실"


# ═══════════════════════════════════════════════
# 자산배분: Ridge 학습 기반
# ═══════════════════════════════════════════════

def build_prob_feature_vector(
    direction_proba, risk_proba, rebound_proba, breakdown_proba
) -> np.ndarray:
    return np.array([
        get_prob(direction_proba,  "상승"),
        get_prob(direction_proba,  "하락"),
        get_prob(direction_proba,  "횡보"),
        get_prob(risk_proba,       "고변동"),
        get_prob(rebound_proba,    1),
        get_prob(breakdown_proba,  1),
    ])


def fit_allocation_model(monthly_records: list[dict]):
    if len(monthly_records) < MIN_WEIGHT_TRAIN_MONTHS:
        return None, None

    rows, targets = [], []
    for rec in monthly_records:
        prob_vec = build_prob_feature_vector(
            rec["direction_proba"], rec["risk_proba"],
            rec["rebound_proba"],   rec["breakdown_proba"],
        )
        s = rec["stock_return"]
        b = rec["bond_return"]
        c = rec["cash_return"]
        total_abs = abs(s) + abs(b) + abs(c) + 1e-9
        optimal_stock = np.clip(abs(s) / total_abs * np.sign(s + 1), 0.20, 0.90)
        rows.append(prob_vec)
        targets.append(optimal_stock)

    X = np.array(rows)
    y = np.array(targets)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_sc, y)
    return model, scaler


def predict_allocation(
    alloc_model, alloc_scaler,
    direction_proba, risk_proba, rebound_proba, breakdown_proba,
    regime: str
) -> tuple[dict, str]:
    """
    Ridge 학습 비중 기반 + 국면별 보정.
    모델 미준비 시 국면 규칙만 사용.
    """
    # 국면별 기본 주식 비중 (모델 미준비 시 fallback)
    regime_default = {
        "붕괴위험": 0.20,
        "고변동":   0.30,
        "하락":     0.35,
        "반등":     0.55,
        "상승":     0.70,
        "횡보":     0.50,
        "불확실":   0.45,
    }

    if alloc_model is None:
        stock_weight = regime_default.get(regime, 0.50)
        reason = f"국면={regime}, 가중치 모델 미준비 → 국면 기본값 적용"
    else:
        prob_vec = build_prob_feature_vector(
            direction_proba, risk_proba, rebound_proba, breakdown_proba
        ).reshape(1, -1)
        X_sc         = alloc_scaler.transform(prob_vec)
        stock_weight = float(alloc_model.predict(X_sc)[0])

        # 국면에 따라 Ridge 예측값 클리핑 범위 조정
        regime_clip = {
            "붕괴위험": (0.20, 0.35),
            "고변동":   (0.25, 0.45),
            "하락":     (0.25, 0.50),
            "반등":     (0.45, 0.75),
            "상승":     (0.55, 0.90),
            "횡보":     (0.35, 0.65),
            "불확실":   (0.30, 0.60),
        }
        lo, hi = regime_clip.get(regime, (STOCK_WEIGHT_MIN, STOCK_WEIGHT_MAX))
        stock_weight = np.clip(stock_weight, lo, hi)

        reason = (
            f"국면={regime}, Ridge 주식={stock_weight*100:.1f}% "
            f"[clip {lo*100:.0f}~{hi*100:.0f}%]"
        )

    remaining = 1.0 - stock_weight
    p_down      = get_prob(direction_proba, "하락")
    p_high_vol  = get_prob(risk_proba,      "고변동")
    p_breakdown = get_prob(breakdown_proba, 1)

    defensive_score = 0.40 * p_down + 0.50 * p_high_vol + 0.50 * p_breakdown
    cash_ratio      = np.clip(0.30 + 0.40 * defensive_score, 0.25, 0.70)

    cash_weight = remaining * cash_ratio
    bond_weight = remaining - cash_weight
    total       = stock_weight + bond_weight + cash_weight

    allocation = {
        "stock": stock_weight / total,
        "bond":  bond_weight  / total,
        "cash":  cash_weight  / total,
    }
    return allocation, reason


# ═══════════════════════════════════════════════
# 월별 수익률 다운로드
# ═══════════════════════════════════════════════

def download_monthly_returns(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"{ticker} 데이터 다운로드 실패")
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df["Date"]       = pd.to_datetime(df["Date"])
    df["year_month"] = df["Date"].dt.to_period("M")
    me = df.groupby("year_month", as_index=False).tail(1).sort_values("Date").reset_index(drop=True)
    me[f"{ticker}_r"] = me["Close"].shift(-1) / me["Close"] - 1
    return me[["year_month", f"{ticker}_r"]].dropna()


def attach_asset_returns(month_end_df: pd.DataFrame) -> pd.DataFrame:
    start     = str(month_end_df["Date"].min().date())
    stock_ret = download_monthly_returns(TARGET_TICKER, start)
    bond_ret  = download_monthly_returns(BOND_TICKER,   start)
    cash_ret  = download_monthly_returns(CASH_TICKER,   start)

    df = month_end_df.copy()
    df["year_month"] = df["Date"].dt.to_period("M")
    df = (df.merge(stock_ret, on="year_month", how="left")
            .merge(bond_ret,  on="year_month", how="left")
            .merge(cash_ret,  on="year_month", how="left")
            .rename(columns={
                f"{TARGET_TICKER}_r": "stock_next_month_return",
                f"{BOND_TICKER}_r":   "bond_next_month_return",
                f"{CASH_TICKER}_r":   "cash_next_month_return",
            })
            .dropna(subset=["stock_next_month_return",
                            "bond_next_month_return",
                            "cash_next_month_return"])
            .reset_index(drop=True))
    return df


# ═══════════════════════════════════════════════
# 성과 지표
# ═══════════════════════════════════════════════

def calculate_cagr(final_cap, initial_cap, months):
    years = months / 12
    return (final_cap / initial_cap) ** (1 / years) - 1 if years > 0 else np.nan


def calculate_mdd(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def calculate_sharpe(returns: pd.Series, ann=12) -> float:
    std = returns.std()
    return float((returns.mean() / std) * np.sqrt(ann)) if std > 0 else np.nan


def metric_block(df, capital_col, return_col) -> dict:
    fc = float(df[capital_col].iloc[-1])
    m  = len(df)
    return {
        "final_capital": round(fc, 2),
        "total_return":  round(fc / INITIAL_CAPITAL - 1, 6),
        "cagr":          round(calculate_cagr(fc, INITIAL_CAPITAL, m), 6),
        "mdd":           round(calculate_mdd(df[capital_col]), 6),
        "sharpe":        round(calculate_sharpe(df[return_col]), 6),
    }


# ═══════════════════════════════════════════════
# Walk-Forward 실행
# ═══════════════════════════════════════════════

def run_walk_forward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df           = ensure_target_columns(df)
    feature_cols = get_feature_columns(df)

    df["year_month"] = df["Date"].dt.to_period("M")
    month_end_rows = (
        df.groupby("year_month", as_index=False).tail(1)
        .sort_values("Date").reset_index()
        .rename(columns={"index": "original_index"})
    )
    month_end_rows = month_end_rows[
        month_end_rows["Date"] >= pd.to_datetime(BACKTEST_START_DATE)
    ].reset_index(drop=True)
    month_end_rows = attach_asset_returns(month_end_rows)

    is_mask  = month_end_rows["Date"] <= pd.to_datetime(IS_END_DATE)
    oos_mask = month_end_rows["Date"] >= pd.to_datetime(OOS_START_DATE)
    print(f"IS 월 수: {is_mask.sum()}, OOS 월 수: {oos_mask.sum()}")
    print(f"사용 피처 수: {len(feature_cols)}")

    records = []
    strategy_capital        = INITIAL_CAPITAL
    stock_capital           = INITIAL_CAPITAL
    benchmark_60_40_capital = INITIAL_CAPITAL
    static_50_30_20_capital = INITIAL_CAPITAL
    prev_allocation         = {"stock": 0.60, "bond": 0.30, "cash": 0.10}

    monthly_prob_records: list[dict] = []
    alloc_model  = None
    alloc_scaler = None

    required_cols = [
        "future_risk_adj_return_60d", "future_risk_adj_return_20d",
        "future_return_20d", "future_volatility_20d",
        "future_min_return_20d", "past_return_40d",
    ]

    for _, row in month_end_rows.iterrows():
        pred_date    = row["Date"]
        original_idx = int(row["original_index"])

        # 2순위: purge gap 포함한 train 종료 인덱스
        train_end_idx = original_idx - TOTAL_GAP

        if train_end_idx < MIN_TRAIN_SIZE:
            continue

        train_raw = df.iloc[:train_end_idx].copy().dropna(subset=required_cols)
        test_raw  = df.iloc[original_idx: original_idx + 1].copy()

        if len(train_raw) < MIN_TRAIN_SIZE:
            continue

        # 동적 라벨 (분위수는 train_raw 기준)
        train_labeled = make_dynamic_labels(train_raw, train_raw)
        test_labeled  = make_dynamic_labels(train_raw, test_raw)

        # 분류 모델 학습 (앙상블 + 확률 보정)
        direction_pipe = fit_model(train_labeled, feature_cols, "direction_label")
        risk_pipe      = fit_model(train_labeled, feature_cols, "risk_label")
        rebound_pipe   = fit_model(train_labeled, feature_cols, "rebound_flag")
        breakdown_pipe = fit_model(train_labeled, feature_cols, "breakdown_flag")

        # 확률 예측
        direction_proba  = predict_proba_dict(direction_pipe,  test_labeled, feature_cols)
        risk_proba       = predict_proba_dict(risk_pipe,       test_labeled, feature_cols)
        rebound_proba    = predict_proba_dict(rebound_pipe,    test_labeled, feature_cols)
        breakdown_proba  = predict_proba_dict(breakdown_pipe,  test_labeled, feature_cols)

        # 7순위: 규칙 모델 블렌딩
        rule_proba      = rule_based_direction(test_labeled.iloc[0])
        direction_proba = blend_with_rule(direction_proba, rule_proba, rule_weight=0.10)

        # 확률 추출
        p_up        = get_prob(direction_proba, "상승")
        p_down      = get_prob(direction_proba, "하락")
        p_neutral   = get_prob(direction_proba, "횡보")
        p_high_vol  = get_prob(risk_proba,      "고변동")
        p_rebound   = get_prob(rebound_proba,   1)
        p_breakdown = get_prob(breakdown_proba, 1)

        # 8순위: 국면 결정
        regime = decide_market_regime(
            p_up, p_neutral, p_down, p_high_vol, p_rebound, p_breakdown
        )

        # 자산배분 가중치 학습 버퍼 갱신
        if monthly_prob_records and "stock_return" not in monthly_prob_records[-1]:
            last = monthly_prob_records[-1]
            last["stock_return"] = float(row.get("stock_next_month_return", 0))
            last["bond_return"]  = float(row.get("bond_next_month_return",  0))
            last["cash_return"]  = float(row.get("cash_next_month_return",  0))

        monthly_prob_records.append({
            "direction_proba": direction_proba,
            "risk_proba":      risk_proba,
            "rebound_proba":   rebound_proba,
            "breakdown_proba": breakdown_proba,
        })

        complete = [r for r in monthly_prob_records if "stock_return" in r]
        if len(complete) >= MIN_WEIGHT_TRAIN_MONTHS:
            alloc_model, alloc_scaler = fit_allocation_model(complete)

        # 자산배분
        allocation, alloc_reason = predict_allocation(
            alloc_model, alloc_scaler,
            direction_proba, risk_proba, rebound_proba, breakdown_proba,
            regime
        )

        stock_return = float(row["stock_next_month_return"])
        bond_return  = float(row["bond_next_month_return"])
        cash_return  = float(row["cash_next_month_return"])

        strat_return_raw = (
            allocation["stock"] * stock_return
            + allocation["bond"] * bond_return
            + allocation["cash"] * cash_return
        )
        turnover     = sum(abs(allocation[k] - prev_allocation[k]) for k in allocation)
        trade_ratio  = turnover / 2
        tx_cost      = trade_ratio * TRANSACTION_COST_RATE
        strat_return = strat_return_raw - tx_cost

        bm_6040  = 0.60 * stock_return + 0.40 * bond_return
        bm_50302 = 0.50 * stock_return + 0.30 * bond_return + 0.20 * cash_return

        strategy_capital        *= (1 + strat_return)
        stock_capital           *= (1 + stock_return)
        benchmark_60_40_capital *= (1 + bm_6040)
        static_50_30_20_capital *= (1 + bm_50302)

        # 6순위: 임곗값 적용 예측값
        pred_direction = max(direction_proba, key=direction_proba.get) if direction_proba else "N/A"
        pred_risk      = max(risk_proba,      key=risk_proba.get)      if risk_proba      else "N/A"
        pred_rebound   = 1 if p_rebound   >= THRESHOLDS["rebound"]   else 0
        pred_breakdown = 1 if p_breakdown >= THRESHOLDS["breakdown"]  else 0

        actual_direction = test_labeled["direction_label"].iloc[0]
        actual_risk      = test_labeled["risk_label"].iloc[0]
        actual_rebound   = int(test_labeled["rebound_flag"].iloc[0])
        actual_breakdown = int(test_labeled["breakdown_flag"].iloc[0])

        phase = "OOS" if pred_date >= pd.to_datetime(OOS_START_DATE) else "IS"

        records.append({
            "Date": pred_date, "phase": phase, "regime": regime,

            "actual_direction": actual_direction, "pred_direction": pred_direction,
            "actual_risk":      actual_risk,      "pred_risk":      pred_risk,
            "actual_rebound":   actual_rebound,   "pred_rebound":   pred_rebound,
            "actual_breakdown": actual_breakdown, "pred_breakdown": pred_breakdown,

            "prob_up":        p_up,
            "prob_down":      p_down,
            "prob_sideways":  p_neutral,
            "prob_high_vol":  p_high_vol,
            "prob_rebound":   p_rebound,
            "prob_breakdown": p_breakdown,

            "stock_weight": allocation["stock"],
            "bond_weight":  allocation["bond"],
            "cash_weight":  allocation["cash"],
            "allocation_reason": alloc_reason,

            "stock_next_month_return": stock_return,
            "bond_next_month_return":  bond_return,
            "cash_next_month_return":  cash_return,

            "strategy_return_raw":    strat_return_raw,
            "turnover":               turnover,
            "trade_ratio":            trade_ratio,
            "transaction_cost":       tx_cost,
            "strategy_return_after_cost": strat_return,

            "stock_benchmark_return":   stock_return,
            "benchmark_60_40_return":   bm_6040,
            "static_50_30_20_return":   bm_50302,

            "strategy_capital":        strategy_capital,
            "stock_capital":           stock_capital,
            "benchmark_60_40_capital": benchmark_60_40_capital,
            "static_50_30_20_capital": static_50_30_20_capital,

            "alloc_model_ready": alloc_model is not None,
        })

        prev_allocation = allocation

    if not records:
        raise ValueError("Walk-forward 결과가 비어 있습니다.")

    result_df = pd.DataFrame(records)
    is_df     = result_df[result_df["phase"] == "IS"].reset_index(drop=True)
    oos_df    = result_df[result_df["phase"] == "OOS"].reset_index(drop=True)
    return is_df, oos_df


# ═══════════════════════════════════════════════
# 성과 요약
# ═══════════════════════════════════════════════

def make_clf_summary(df: pd.DataFrame) -> dict:
    summary = {}
    for target in ["direction", "risk"]:
        ac = f"actual_{target}"; pc = f"pred_{target}"
        if ac in df.columns:
            summary[target] = {
                "accuracy": round(accuracy_score(df[ac], df[pc]), 6),
                "macro_f1": round(f1_score(df[ac], df[pc], average="macro", zero_division=0), 6),
                "report":   classification_report(df[ac], df[pc], zero_division=0, output_dict=True),
            }
    for flag in ["rebound", "breakdown"]:
        ac = f"actual_{flag}"; pc = f"pred_{flag}"
        if ac in df.columns:
            summary[flag] = {
                "accuracy":          round(accuracy_score(df[ac], df[pc]), 6),
                "macro_f1":          round(f1_score(df[ac], df[pc], average="macro", zero_division=0), 6),
                "positive_precision": round(precision_score(df[ac], df[pc], pos_label=1, zero_division=0), 6),
                "positive_recall":    round(recall_score(df[ac], df[pc],    pos_label=1, zero_division=0), 6),
            }
    return summary


def summarize_phase(df: pd.DataFrame, phase: str) -> dict:
    if df.empty:
        return {"phase": phase, "months": 0}

    if phase == "OOS":
        df = df.copy()
        for col in ["strategy_capital", "stock_capital",
                    "benchmark_60_40_capital", "static_50_30_20_capital"]:
            df[col] = INITIAL_CAPITAL * (df[col] / df[col].iloc[0])

    regime_counts = df["regime"].value_counts().to_dict() if "regime" in df.columns else {}

    return {
        "phase": phase,
        "start": str(df["Date"].iloc[0].date()),
        "end":   str(df["Date"].iloc[-1].date()),
        "months": int(len(df)),
        "regime_counts": regime_counts,
        "average_weights": {
            "avg_stock": round(float(df["stock_weight"].mean()), 6),
            "avg_bond":  round(float(df["bond_weight"].mean()),  6),
            "avg_cash":  round(float(df["cash_weight"].mean()),  6),
        },
        "turnover": {
            "avg_monthly": round(float(df["trade_ratio"].mean()), 6),
            "annual_est":  round(float(df["trade_ratio"].mean() * 12), 6),
        },
        "strategy_after_cost": metric_block(df, "strategy_capital",        "strategy_return_after_cost"),
        "stock_buy_hold":      metric_block(df, "stock_capital",            "stock_benchmark_return"),
        "benchmark_60_40":     metric_block(df, "benchmark_60_40_capital",  "benchmark_60_40_return"),
        "static_50_30_20":     metric_block(df, "static_50_30_20_capital",  "static_50_30_20_return"),
        "classification":      make_clf_summary(df),
    }


def print_summary(s: dict):
    phase = s.get("phase", "?")
    print(f"\n{'='*50}")
    print(f"[{phase}]  {s.get('start')} ~ {s.get('end')}  ({s.get('months')}개월)")
    print(f"{'='*50}")

    rc = s.get("regime_counts", {})
    if rc:
        print("국면 분포:", " | ".join(f"{k}:{v}" for k, v in rc.items()))

    w = s.get("average_weights", {})
    print(f"평균 비중  주식:{w.get('avg_stock',0)*100:.1f}%  채권:{w.get('avg_bond',0)*100:.1f}%  현금:{w.get('avg_cash',0)*100:.1f}%")
    print(f"연간 회전율 추정: {s.get('turnover',{}).get('annual_est',0)*100:.1f}%")

    for name in ["strategy_after_cost", "stock_buy_hold", "benchmark_60_40", "static_50_30_20"]:
        b = s.get(name, {})
        if not b:
            continue
        print(f"\n  [{name}]  CAGR:{b.get('cagr',0)*100:.2f}%  MDD:{b.get('mdd',0)*100:.2f}%  Sharpe:{b.get('sharpe',0):.4f}")

    clf = s.get("classification", {})
    if clf:
        print("\n  [분류 성능]")
        for k, v in clf.items():
            if isinstance(v, dict):
                extra = ""
                if "positive_precision" in v:
                    extra = (f"  pos_prec:{v['positive_precision']*100:.1f}%"
                             f"  pos_recall:{v['positive_recall']*100:.1f}%")
                print(f"    {k}  acc:{v.get('accuracy',0)*100:.1f}%  "
                      f"macro_f1:{v.get('macro_f1',0):.3f}{extra}")


# ═══════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════

def main():
    print("[1] 데이터 로드")
    df = load_dataset(DATA_PATH)
    print(f"크기: {df.shape}  기간: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
    print(f"IS: ~ {IS_END_DATE}  |  OOS: {OOS_START_DATE} ~")

    print("\n[2] Walk-Forward v3 실행")
    is_df, oos_df = run_walk_forward(df)

    print("\n[3] 결과 저장")
    is_df.to_csv(RESULT_IS_CSV,   index=False, encoding="utf-8-sig")
    oos_df.to_csv(RESULT_OOS_CSV, index=False, encoding="utf-8-sig")

    is_summary  = summarize_phase(is_df,  "IS")
    oos_summary = summarize_phase(oos_df, "OOS")

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump({"IS": is_summary, "OOS": oos_summary}, f, ensure_ascii=False, indent=4)

    print_summary(is_summary)
    print_summary(oos_summary)

    # 과적합 자동 진단
    is_cagr  = is_summary.get("strategy_after_cost", {}).get("cagr", 0)
    oos_cagr = oos_summary.get("strategy_after_cost", {}).get("cagr", 0)
    is_f1    = is_summary.get("classification", {}).get("direction", {}).get("macro_f1", 0)
    oos_f1   = oos_summary.get("classification", {}).get("direction", {}).get("macro_f1", 0)
    is_bd_prec  = is_summary.get("classification", {}).get("breakdown", {}).get("positive_precision", 0)
    oos_bd_prec = oos_summary.get("classification", {}).get("breakdown", {}).get("positive_precision", 0)

    print(f"\n{'='*50}")
    print("과적합 진단 및 개선 목표 대비 달성 현황")
    print(f"{'='*50}")
    print(f"IS  CAGR  : {is_cagr*100:.2f}%")
    print(f"OOS CAGR  : {oos_cagr*100:.2f}%")

    if is_cagr > 0 and oos_cagr > 0:
        deg = (is_cagr - oos_cagr) / is_cagr
        mark = "⚠" if deg > 0.40 else ("△" if deg > 0.20 else "✓")
        print(f"성과 저하율: {deg*100:.1f}%  {mark}")

    print(f"\n분류 성능 (IS/OOS)")
    print(f"  direction macro_f1  : {is_f1:.3f} / {oos_f1:.3f}  (목표: 0.47+)")
    print(f"  breakdown pos_prec  : {is_bd_prec*100:.1f}% / {oos_bd_prec*100:.1f}%  (목표: 35%+)")

    print("\n완료")


if __name__ == "__main__":
    main()