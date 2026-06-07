"""
XGBoost v8.6.34 - Volatility-Base + All Strength Combos
====================================================================

목적
- QQQ/IEF/BIL 동적 자산배분 전략
- v8.6.34: 5D/10D/20D 단독 및 5+10/5+20/10+20/5+10+20 조합 신호를 모두 allocation 후보로 사용
- H10/H20 정상/고변동 Stage1 모델을 앙상블
- H10/H20 하락고변동 Down-risk OVR 모델을 앙상블
- 고변동 내부 상승/하락 Stage2 방향 분류는 제거
- Stage1 고변동 확률을 1차 게이트로 사용하고, Down-risk는 방어 보조 신호로 사용
- v8.4 개선: H10 Down-risk 단독 사용을 기본값으로 채택
- v8.4 개선: no-trade band, RISK_OFF threshold, RISK_OFF 비중, 연속 조정 사용 여부를 validation 기반으로 비교 가능
- v8.4 개선: 조건 선택 근거를 condition_search CSV로 저장
- v8.4 개선: regime/probability bin/threshold/turnover/drawdown/feature optimization diagnostics CSV 추가 저장
- v8.4 개선: v8.3 진단 결과를 반영해 c032 계열 조건을 기본값으로 채택
- v8.4 개선: HIGH_VOL 독립 regime 제거, NORMAL/WATCH/RISK_OFF 중심의 3-regime allocation 적용
- v8.4 개선: 극단 위험 구간에서만 추가 방어하는 EXTREME_RISK sub-regime 추가
- v8.4 개선: condition search는 상위 점수 후보 중 turnover/MDD 안정성을 우선하는 stable-top 선택 로직 적용
- v8.5 개선: 보수적 체결 지연(execution_lag_days), select/holdout 성과 분리, raw/EWMA 확률 분리
- v8.5 개선: classification_report 안정화, 라벨 생성 벡터화, max_train_rows 옵션 추가
- 비용/turnover를 고려해 10거래일 단위 리밸런싱 + 긴급 리밸런싱 제한
- 고변동 라벨 quantile 정책은 고정 또는 adaptive nested validation으로 선택 가능

실행 예시
    py xgb_strength_combo_all_v8_6_34.py --speed-profile balanced --h10-down-only --condition-search
    py xgb_strength_combo_all_v8_6_34.py --speed-profile fast --h10-down-only
    py xgb_strength_combo_all_v8_6_34.py --speed-profile full --adaptive-label --h10-down-only

필요 패키지
    pip install pandas numpy yfinance scikit-learn xgboost

중요
- 미래 수익률/변동성 컬럼은 라벨 생성에만 사용하고, 모델 입력 feature에는 사용하지 않습니다.
- walk-forward 예측 시 max(horizons)만큼 purge gap을 둡니다.
- adaptive label policy는 각 retrain 시점의 과거 train 구간 내부에서만 선택합니다.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("xgboost가 설치되어 있지 않습니다. `pip install xgboost`를 실행하세요.") from exc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ============================================================
# Direction-Strength Specialist Labels
# ============================================================

DIRECTION_STRENGTH_LABELS = [
    "NO_STRENGTH_SIGNAL",
    "UP_STRENGTHENING",
    "DOWN_STRENGTHENING",
]
DIRECTION_STRENGTH_LABEL_TO_ID = {name: i for i, name in enumerate(DIRECTION_STRENGTH_LABELS)}
DIRECTION_STRENGTH_ID_TO_LABEL = {i: name for name, i in DIRECTION_STRENGTH_LABEL_TO_ID.items()}


# ============================================================
# 0. CONFIG
# ============================================================

@dataclass(frozen=True)
class LabelPolicy:
    name: str
    vol_q: float = 0.80
    down_q: float = 0.20
    up_q: float = 0.80


@dataclass
class Config:
    target_ticker: str = "QQQ"
    bond_ticker: str = "IEF"
    cash_ticker: str = "BIL"

    start_date: str = "1999-03-10"
    backtest_start_date: str = "2013-01-02"
    end_date: Optional[str] = None

    initial_capital: float = 100_000_000
    transaction_cost_rate: float = 0.001
    # 0: 기존 방식. signal[t]로 Close[t] -> Close[t+1] 수익률 반영.
    # 1: 보수적 방식. signal[t] 생성 후 Close[t+1] -> Close[t+2] 수익률 반영.
    execution_lag_days: int = 1
    # False 권장. True면 BIL 다운로드 실패 시 현금 수익률 0으로 대체.
    allow_cash_download_fallback: bool = False

    horizons: Tuple[int, ...] = (5, 10, 20)
    primary_horizon: int = 10
    min_train_rows: int = 756
    retrain_every_n_days: int = 10
    # None이면 expanding window 전체 사용. 숫자를 주면 최근 N개 학습 샘플만 사용.
    max_train_rows: Optional[int] = None

    random_state: int = 42
    n_jobs: int = -1

    # XGBoost Stage1: normal vs high-vol
    stage1_n_estimators: int = 150
    stage1_learning_rate: float = 0.025
    stage1_max_depth: int = 3
    stage1_min_child_weight: float = 10.0
    stage1_subsample: float = 0.85
    stage1_colsample_bytree: float = 0.80
    stage1_reg_lambda: float = 8.0
    stage1_reg_alpha: float = 0.1

    # XGBoost Down-risk OVR: down-high-vol vs not down-high-vol
    down_n_estimators: int = 100
    down_learning_rate: float = 0.030
    down_max_depth: int = 2
    down_min_child_weight: float = 6.0
    down_subsample: float = 0.90
    down_colsample_bytree: float = 0.85
    down_reg_lambda: float = 10.0
    down_reg_alpha: float = 0.2

    # v8.6 Multi-branch Down-risk ensemble
    # - price_trend: 가격/추세 붕괴 선행 신호
    # - price_volume: 가격 하락 + 거래량 압력 신호
    # - volatility: 변동성/ATR/Range 확인 신호
    # - high_vol: Stage1 고변동 확률을 final down-risk score의 보조 입력으로 사용
    down_price_trend_weight: float = 0.40
    down_price_volume_weight: float = 0.30
    down_volatility_weight: float = 0.20
    down_highvol_weight: float = 0.00
    use_multi_branch_downrisk: bool = True

    # v8.6.5 Overall risk evaluation
    # - 전체 리스크는 고변동 위험과 하락위험을 함께 본다.
    # - Down-risk만 보면 방어 실패/과잉 방어를 전체 국면 관점에서 해석하기 어렵다.
    overall_risk_high_vol_weight: float = 0.35
    overall_risk_down_weight: float = 0.50
    overall_risk_down_minus_up_weight: float = 0.15
    pred_overall_risk_threshold: float = 0.50

    # v8.6.5 Direction model labels
    # future_return_h가 +threshold보다 크면 상승, -threshold보다 작으면 하락, 그 사이를 중립으로 둔다.
    direction_return_threshold: float = 0.005
    direction_decision_margin: float = 0.05
    direction_min_positive: int = 20

    # Adaptive label policy search
    use_adaptive_label_policy: bool = False
    label_search_valid_rows: int = 252
    label_search_stage1_estimators: int = 60
    label_search_down_estimators: int = 40
    label_search_min_positive: int = 20
    label_policy_candidates: Tuple[LabelPolicy, ...] = (
        LabelPolicy("balanced_q80_d20_u80", 0.80, 0.20, 0.80),
        LabelPolicy("sensitive_q75_d25_u75", 0.75, 0.25, 0.75),
        LabelPolicy("strict_q85_d15_u85", 0.85, 0.15, 0.85),
    )
    fixed_label_policy: LabelPolicy = LabelPolicy("fixed_q80_d20_u80", 0.80, 0.20, 0.80)

    # Ensemble weights
    # v8.1: Stage1은 H10/H20을 거의 균등하게 사용하되, Down-risk는 H10 중심으로 사용
    high_vol_weight_h10: float = 0.55
    high_vol_weight_h20: float = 0.45
    down_risk_weight_h10: float = 1.00
    down_risk_weight_h20: float = 0.00

    # Probability smoothing
    use_prob_ewma: bool = True
    prob_ewma_span: int = 7

    # Prediction thresholds for reporting
    pred_high_vol_threshold: float = 0.50
    pred_down_risk_threshold: float = 0.45

    # Allocation gate thresholds
    gate_normal_high_vol_threshold: float = 0.55
    # v8.1: RISK_OFF 진입을 더 어렵게 만들어 과도한 방어와 turnover를 줄임
    # v8.4: c032 계열 기본값. 0.62/0.52가 v8.3 선택값보다 CAGR/MDD/Calmar/turnover 균형이 좋았음.
    gate_high_vol_threshold: float = 0.74
    gate_riskoff_downrisk_threshold: float = 0.74
    gate_watch_downrisk_threshold: float = 0.80

    # v8.4: HIGH_VOL 독립 regime 제거. NORMAL / WATCH / RISK_OFF 중심으로 단순화.
    use_three_regime_allocation: bool = True

    # v8.4: 극단 위험 구간에서만 추가 방어. 일반 RISK_OFF는 58% 주식 유지.
    use_extreme_risk_cut: bool = True
    extreme_high_vol_threshold: float = 0.86
    extreme_downrisk_threshold: float = 0.86
    extreme_stock_weight: float = 0.30
    extreme_bond_weight: float = 0.45
    extreme_cash_weight: float = 0.25

    # Base bucket allocations: low-base regime
    normal_stock_weight: float = 0.72
    normal_bond_weight: float = 0.18
    normal_cash_weight: float = 0.10

    watch_stock_weight: float = 0.62
    watch_bond_weight: float = 0.25
    watch_cash_weight: float = 0.13

    high_vol_stock_weight: float = 0.55
    high_vol_bond_weight: float = 0.30
    high_vol_cash_weight: float = 0.15

    risk_off_stock_weight: float = 0.45
    risk_off_bond_weight: float = 0.37
    risk_off_cash_weight: float = 0.18

    # Small continuous adjustment within bucket
    use_continuous_adjustment: bool = False
    # v8.1: bucket 내부 연속 조정폭을 축소해 평균 주식 비중과 turnover를 개선
    continuous_high_vol_weight: float = 0.025
    continuous_down_risk_weight: float = 0.035
    max_continuous_stock_cut: float = 0.04

    # Trading rules
    rebalance_every_n_days: int = 5
    no_trade_band: float = 0.12
    emergency_high_vol_threshold: float = 0.88
    emergency_combined_high_vol_threshold: float = 0.78
    emergency_combined_down_threshold: float = 0.78
    emergency_cooldown_days: int = 5

    # Optional small rolling allocation threshold optimization
    # 기본값 False: 속도 문제 방지. 켜도 후보 수는 작게 유지.
    use_rolling_gate_optimization: bool = False
    gate_optimize_every_n_days: int = 120
    gate_rolling_window: int = 504
    gate_min_window: int = 252
    gate_score_cagr_weight: float = 1.30
    gate_score_mdd_weight: float = 0.85
    gate_score_turnover_weight: float = 0.45

    result_dir: str = "results_xgb_strength_combo_all_v8_6_34"

    # v8.6.6 Policy Lab
    # base       : v8.6.5 pruned fixed-bucket baseline
    # return_seeking: NORMAL 구간에서만 주식 비중 상향 bonus
    # defensive_risk: Stage1 고변동 중심의 방어형 overlay
    # aggressive_dynamic: Buy & Hold + crash brake 구조
    policy_mode: str = "direction_strength_specialist"
    use_policy_overlay: bool = True
    trend_bull_score_threshold: int = 4
    trend_bear_score_threshold: int = 2
    return_bonus_1: float = 0.03
    return_bonus_2: float = 0.03
    return_bonus_3: float = 0.02
    defensive_max_extra_stock_cut: float = 0.08
    defensive_vol_target_clip_low: float = 0.75
    defensive_vol_target_clip_high: float = 1.02
    aggressive_low_risk_stock_weight: float = 1.00
    aggressive_watch_stock_weight: float = 0.92

    # v8.6.34 base allocation: 기본 상태는 high-vol 확률 기반 연속/계단형 비중으로 산출한다.
    use_vol_probability_base_allocation: bool = True
    vol_base_stock_lt_25: float = 0.78
    vol_base_stock_lt_35: float = 0.74
    vol_base_stock_lt_50: float = 0.68
    vol_base_stock_lt_65: float = 0.60
    vol_base_stock_lt_75: float = 0.52
    vol_base_stock_lt_86: float = 0.42
    vol_base_stock_ge_86: float = 0.30
    vol_base_bond_ratio_of_defensive: float = 0.65

    # v8.6.34 execution: Tier3/Full은 schedule/no-trade-band에 묶이지 않고 즉시 목표 비중으로 점프한다.
    force_strong_offensive_rebalance: bool = True
    force_tier3_rebalance: bool = True
    force_full_stock_rebalance: bool = True
    disable_tier2_signal: bool = True

    # v8.6.34 Separate PortfolioPolicyModel
    # - 1단계 확률 모델 결과를 입력으로 받아 후보 포트폴리오 클래스를 별도 예측한다.
    # - 기본 실행에서는 진단용으로만 생성하고, --policy-mode portfolio_model이면 실제 allocation에 사용한다.
    enable_portfolio_policy_model: bool = False
    portfolio_policy_horizon: int = 20
    portfolio_policy_min_train_rows: int = 756
    portfolio_policy_max_train_rows: Optional[int] = 1260
    portfolio_policy_retrain_every_n_days: int = 10
    portfolio_policy_n_estimators: int = 120
    portfolio_policy_learning_rate: float = 0.035
    portfolio_policy_max_depth: int = 2
    portfolio_policy_min_child_weight: float = 8.0
    portfolio_policy_subsample: float = 0.85
    portfolio_policy_colsample_bytree: float = 0.85
    portfolio_policy_reg_lambda: float = 10.0
    portfolio_policy_reg_alpha: float = 0.2
    portfolio_utility_vol_penalty: float = 0.50
    portfolio_utility_mdd_penalty: float = 0.80
    portfolio_utility_turnover_penalty: float = 0.001
    portfolio_model_min_confidence: float = 0.0
    portfolio_model_force_rebalance: bool = False

    # v8.6.34 Volatility-Base Strong-Override Multi-Horizon Upside Strength Trigger
    # - 기존 binary up_h10/up_h20은 allocation에 직접 쓰지 않는다.
    # - 5D/10D/20D UP_STRENGTHENING specialist를 별도로 학습한다.
    # - 5D는 단기 반등, 10D는 추세 확인, 20D는 중기 지속성으로 해석한다.
    # - 세 확률의 term-structure score로 공격 비중을 결정한다.
    use_direction_strength_specialist: bool = True
    direction_strength_horizon: int = 20  # legacy / bear specialist 기준 horizon
    multi_strength_horizons: Tuple[int, ...] = (5, 10, 20)
    up_strength_weight_5d: float = 0.00
    up_strength_weight_10d: float = 0.20
    up_strength_weight_20d: float = 0.80
    # v8.6.34: prob_bear_down_strengthening 제거. 하락 강화도 상승 강화와 동일하게 5D/10D/20D term structure로 계산한다.
    down_strength_weight_5d: float = 0.00
    down_strength_weight_10d: float = 0.20
    down_strength_weight_20d: float = 0.80
    direction_strength_ret_eps_k: float = 0.20
    direction_strength_eps: float = 0.0
    direction_strength_method: str = "score_delta"
    upside_strength_train_filter: str = "major_only"
    bear_strength_train_filter: str = "bear_stress"
    direction_strength_feature_set: str = "horizon_5_10_20_pruned"
    direction_strength_min_train_rows: int = 300
    direction_strength_max_train_rows: Optional[int] = 1260

    # v8.6.34 Horizon별 학습 기간 비율 실험
    # - train_rows = horizon * multiplier 방식으로 5D/10D/20D마다 학습창을 다르게 둔다.
    # - 너무 작은 학습창은 불안정하므로 horizon_train_min_rows를 하한으로 둔다.
    # - 기본값은 5D≈3년, 10D≈4년, 20D≈5년 수준의 1차 가설이다.
    use_horizon_train_window: bool = True
    horizon_train_min_rows: int = 504
    horizon_train_max_rows_cap: Optional[int] = None
    horizon_train_multiplier_5d: float = 150.0
    horizon_train_multiplier_10d: float = 100.0
    horizon_train_multiplier_20d: float = 63.0
    direction_strength_use_horizon_train_window: bool = True

    direction_strength_n_estimators: int = 160
    direction_strength_learning_rate: float = 0.025
    direction_strength_max_depth: int = 2
    direction_strength_min_child_weight: float = 8.0
    direction_strength_subsample: float = 0.85
    direction_strength_colsample_bytree: float = 0.80
    direction_strength_reg_lambda: float = 10.0
    direction_strength_reg_alpha: float = 0.2

    # v8.6.21 controlled consensus thresholds
    # up_strength_score = 0.00*P5D + 0.20*P10D + 0.80*P20D
    # 핵심: 5D는 trigger에서 제외, 10D는 confirmation, 20D는 primary signal로 쓴다.
    # Tier 2는 약화하고, 100%는 score/p20/p10/high-vol 조건이 모두 강할 때만 허용한다.
    up_strength_bonus_threshold_1: float = 0.30
    up_strength_bonus_threshold_2: float = 0.38  # v8.6.34: Tier2 활성화 기본값
    up_strength_bonus_threshold_3: float = 0.45
    up_strength_confirm_10d_threshold_2: float = 0.32
    up_strength_confirm_20d_threshold_2: float = 0.34
    up_strength_confirm_10d_threshold_3: float = 0.38
    up_strength_confirm_20d_threshold_3: float = 0.38
    up_strength_pred_threshold_5d: float = 0.99  # v8.6.34: 5D 단독 trigger 비활성화
    up_strength_pred_threshold_10d: float = 0.27
    up_strength_pred_threshold_20d: float = 0.25
    up_strength_single_5d_stock_weight: float = 0.82
    up_strength_single_10d_stock_weight: float = 0.82
    up_strength_single_20d_stock_weight: float = 0.80
    up_strength_pair_5d_10d_stock_weight: float = 0.84
    up_strength_pair_5d_20d_stock_weight: float = 0.86
    up_strength_pair_10d_20d_stock_weight: float = 0.88  # Tier2 목표 주식 비중
    up_strength_all3_base_stock_weight: float = 0.96
    up_strength_all3_strong_stock_weight: float = 1.00
    up_strength_all3_strong_score_threshold: float = 0.45
    up_strength_all3_strong_5d_threshold: float = 0.30
    up_strength_all3_strong_10d_threshold: float = 0.38
    up_strength_all3_strong_20d_threshold: float = 0.38
    up_strength_all3_strong_high_vol_threshold: float = 0.55
    bear_down_strength_cut_threshold_1: float = 0.55
    bear_down_strength_cut_threshold_2: float = 0.65
    direction_strength_max_stock_bonus: float = 1.00
    direction_strength_max_stock_cut: float = 0.04

    # Down-risk allocation control
    # 0.0이면 allocation gate에서 down 확률을 사실상 사용하지 않고 Stage1 high-vol만 사용한다.
    # 0.2~0.4는 down-risk를 약한 보조 신호로 쓰는 실험용 값이다.
    allocation_downrisk_weight: float = 0.0
    use_bear_specialist_cut: bool = False

    # Multi-horizon upside trigger target weights
    # v8.6.34: 5D/10D/20D 단독과 모든 2-way/3-way 조합을 strength_combo ladder에서 사용한다.
    up_strength_offensive_stock_weight_1: float = 0.82
    up_strength_offensive_stock_weight_2: float = 0.88
    up_strength_offensive_stock_weight_3: float = 0.96
    up_strength_low_vol_threshold_1: float = 0.82
    up_strength_low_vol_threshold_2: float = 0.72
    up_strength_low_vol_threshold_3: float = 0.68
    up_strength_full_stock_high_vol_threshold: float = 0.58
    up_strength_full_stock_score_threshold: float = 0.50
    up_strength_full_stock_10d_threshold: float = 0.38
    up_strength_full_stock_20d_threshold: float = 0.42
    up_strength_disable_5d_trigger: bool = True
    up_strength_require_20d_for_tier2: bool = True
    up_strength_require_20d_for_tier3: bool = True
    up_strength_bear_block_threshold: float = 0.75

    # v8.6.21 diagnostics / stale offensive de-risking
    # 기본값 True: 저비중 기본형에서는 이전 공격 비중이 오래 남으면 전략 의도가 깨진다.
    enable_stale_offensive_decay: bool = True
    stale_offensive_stock_gap_threshold: float = 0.12
    stale_offensive_up_strength_reset_threshold: float = 0.20
    stale_offensive_high_vol_threshold: float = 0.72

    # v8.6.34 ShortMidConfirm
    # - 사후 진단에서 우수했던 5D+10D+high-vol 조합을 별도 보조 신호로 출력한다.
    # - 기본값은 base_upgrade: 기존 Tier가 없을 때 ShortMidConfirm으로 소폭 승격한다.
    # - diagnostic/tier1_upgrade/base_tier1_upgrade/tier2_add/tier2_replace는 실험용이다.
    short_mid_confirm_mode: str = "base_upgrade"  # diagnostic, tier1_upgrade, base_upgrade, base_tier1_upgrade, tier2_add, tier2_replace
    short_mid_action_signal: str = "confirm"  # confirm, strong, loose, all3
    short_mid_p5_threshold: float = 0.32
    short_mid_p10_threshold: float = 0.34
    short_mid_p20_threshold: float = 0.34
    short_mid_high_vol_threshold: float = 0.72
    short_mid_strong_high_vol_threshold: float = 0.68
    short_mid_loose_high_vol_threshold: float = 0.76
    short_mid_use_score_filter: bool = False
    short_mid_score_threshold: float = 0.38
    short_mid_tier1_upgrade_stock_weight: float = 0.84
    short_mid_tier2_stock_weight: float = 0.88
    short_mid_base_upgrade_stock_weight: float = 0.82

    # v8.6.34 Strength Combo Ladder
    # - 5D, 10D, 20D, 5+10, 5+20, 10+20, 5+10+20 조합을 모두 별도 신호로 만든다.
    # - 기본값 max_weight: 만족한 조합 중 가장 높은 목표 주식 비중을 적용한다.
    # - Tier2는 legacy 이름으로 복구하지 않고, combo action으로 별도 기록한다.
    strength_combo_policy_enabled: bool = True
    strength_combo_policy_mode: str = "max_weight"  # off, diagnostic, max_weight
    strength_combo_use_high_vol_filter: bool = True
    strength_combo_high_vol_threshold: float = 0.72
    strength_combo_use_score_filter: bool = False
    strength_combo_score_threshold: float = 0.38
    strength_combo_single_5d_stock_weight: float = 0.80
    strength_combo_single_10d_stock_weight: float = 0.82
    strength_combo_single_20d_stock_weight: float = 0.82
    strength_combo_pair_5d_10d_stock_weight: float = 0.84
    strength_combo_pair_5d_20d_stock_weight: float = 0.86
    strength_combo_pair_10d_20d_stock_weight: float = 0.88
    strength_combo_all3_stock_weight: float = 0.96
    strength_combo_force_all3_rebalance: bool = True

    # v8.6.34 TierWeightOptimizer
    # - 기본 Tier2는 제거했지만, 필요 시 ShortMid/Tier2 포함 후보 비중을 Walk-forward 방식으로 탐색한다.
    # - 기본 실행에서는 시간이 늘어나는 것을 막기 위해 OFF, --optimize-tier-weights로 활성화한다.
    enable_tier_weight_optimizer: bool = False
    tier_weight_opt_train_rows: int = 756
    tier_weight_opt_test_rows: int = 63
    tier_weight_opt_min_train_rows: int = 504
    tier_weight_opt_score_profile: str = "aggressive"
    tier_weight_opt_include_base_lt25: bool = True
    tier_weight_opt_base_lt25_grid: Tuple[float, ...] = (0.76, 0.78, 0.80, 0.82)
    tier_weight_opt_tier1_grid: Tuple[float, ...] = (0.78, 0.80, 0.82, 0.84)
    tier_weight_opt_tier2_grid: Tuple[float, ...] = (0.82, 0.84, 0.86, 0.88, 0.90)
    tier_weight_opt_tier3_grid: Tuple[float, ...] = (0.92, 0.94, 0.96, 0.98)
    tier_weight_opt_full_grid: Tuple[float, ...] = (0.98, 1.00)


# ============================================================
# 1. DATA
# ============================================================

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.get_level_values(0).unique()) <= 6:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)
    return df


def download_ohlcv(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if df.empty:
        raise ValueError(f"{ticker} 데이터를 다운로드하지 못했습니다.")
    df = _flatten_yf_columns(df).copy()
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker} 데이터에 필요한 컬럼이 없습니다: {missing}")
    df = df[required].copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def download_close(ticker: str, start: str, end: Optional[str]) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if df.empty:
        raise ValueError(f"{ticker} 데이터를 다운로드하지 못했습니다.")
    df = _flatten_yf_columns(df).copy()
    if "Close" not in df.columns:
        raise ValueError(f"{ticker} 데이터에 Close 컬럼이 없습니다.")
    s = df["Close"].copy()
    s.name = ticker
    s.index = pd.to_datetime(s.index)
    return s.sort_index()



def build_aligned_forward_returns(
    target_close: pd.Series,
    bond_close: pd.Series,
    cash_close: pd.Series,
    target_index: pd.Index,
    execution_lag_days: int = 1,
) -> pd.DataFrame:
    """
    종가 기반 피처를 사용한 뒤 실제 체결 가능성을 보수적으로 반영하기 위한 forward return 생성.

    execution_lag_days=0:
        기존 방식과 동일하게 signal[t]에 Close[t] -> Close[t+1] 수익률을 연결한다.
    execution_lag_days=1:
        signal[t] 생성 후 다음 거래일부터 체결된다고 보고 Close[t+1] -> Close[t+2] 수익률을 연결한다.

    주의:
    - 마지막 execution_lag_days + 1개 행은 미래 수익률을 알 수 없으므로 NaN이 남는다.
    - 이후 walk-forward 입력 구성에서 dropna되어 백테스트 대상에서 제외된다.
    """
    if execution_lag_days < 0:
        raise ValueError("execution_lag_days는 0 이상의 정수여야 합니다.")

    prices = pd.concat(
        [
            target_close.rename("stock"),
            bond_close.rename("bond"),
            cash_close.rename("cash"),
        ],
        axis=1,
    ).reindex(target_index).ffill()

    shift_n = 1 + int(execution_lag_days)
    out = pd.DataFrame(index=target_index)
    out["stock_next_return"] = prices["stock"].pct_change().shift(-shift_n)
    out["bond_next_return"] = prices["bond"].pct_change().shift(-shift_n)

    if prices["cash"].notna().sum() == 0:
        # 명시적 fallback이 허용된 경우에만 cash_close가 전부 NaN으로 들어온다.
        out["cash_next_return"] = 0.0
    else:
        out["cash_next_return"] = prices["cash"].pct_change().shift(-shift_n)

    return out


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def rolling_rank_last(series: pd.Series, window: int) -> pd.Series:
    def _rank(x: np.ndarray) -> float:
        if np.all(np.isnan(x)):
            return np.nan
        last = x[-1]
        if np.isnan(last):
            return np.nan
        valid = x[~np.isnan(x)]
        if len(valid) == 0:
            return np.nan
        return float((valid <= last).sum() / len(valid))
    return series.rolling(window, min_periods=max(20, window // 4)).apply(_rank, raw=True)


def calc_trend_slope(close: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    def _slope(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        ly = np.log(np.maximum(y, 1e-12))
        return float(np.polyfit(x, ly, 1)[0])
    return close.rolling(window, min_periods=window).apply(_slope, raw=True)


def add_future_targets(df: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    close = df["Close"]
    ret = df["daily_return"]
    future_cols: Dict[str, pd.Series] = {}
    for h in horizons:
        future_high = close.shift(-1).rolling(h).max().shift(-(h - 1))
        future_low = close.shift(-1).rolling(h).min().shift(-(h - 1))
        future_cols[f"future_volatility_{h}d"] = ret.shift(-1).rolling(h).std().shift(-(h - 1))
        future_cols[f"future_return_{h}d"] = close.shift(-h) / close - 1.0
        future_cols[f"future_max_return_{h}d"] = future_high / close - 1.0
        future_cols[f"future_min_return_{h}d"] = future_low / close - 1.0
    return pd.concat([df, pd.DataFrame(future_cols, index=df.index)], axis=1).copy()


def build_features(ohlcv: pd.DataFrame, horizons: Sequence[int]) -> Tuple[pd.DataFrame, List[str]]:
    df = ohlcv.copy()
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"].replace(0, np.nan)

    df["daily_return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))

    for w in [3, 5, 10, 20, 60, 120]:
        df[f"return_{w}d"] = close / close.shift(w) - 1.0

    # v8.6.23: 5D/10D/20D UP_STRENGTHENING 라벨을 각각 예측하기 위한 과거 구간 피처.
    # 미래 구간은 전혀 사용하지 않고, 현재 시점까지의 term-structure만 사용한다.
    df["return_5d_minus_10d"] = df["return_5d"] - df["return_10d"]
    df["return_5d_minus_20d"] = df["return_5d"] - df["return_20d"]
    df["return_10d_minus_20d"] = df["return_10d"] - df["return_20d"]

    for w in [5, 10, 20, 50, 60, 120, 200]:
        ma = close.rolling(w).mean()
        df[f"ma_{w}"] = ma
        df[f"price_ma_{w}_gap"] = close / ma - 1.0

    df["ma_gap_5_20"] = df["ma_5"] / df["ma_20"] - 1.0
    df["ma_gap_20_60"] = df["ma_20"] / df["ma_60"] - 1.0
    df["ma_gap_60_120"] = df["ma_60"] / df["ma_120"] - 1.0
    df["ma_gap_50_200"] = df["ma_50"] / df["ma_200"] - 1.0

    df["trend_slope_5"] = calc_trend_slope(close, 5)
    df["trend_slope_10"] = calc_trend_slope(close, 10)
    df["trend_slope_20"] = calc_trend_slope(close, 20)
    df["trend_slope_60"] = calc_trend_slope(close, 60)
    df["ma200_slope_60"] = calc_trend_slope(df["ma_200"], 60)

    up = (df["daily_return"] > 0).astype(float)
    large_down = (df["daily_return"] <= -0.02).astype(float)
    large_up = (df["daily_return"] >= 0.02).astype(float)
    for w in [5, 10, 20, 60]:
        df[f"positive_return_ratio_{w}"] = up.rolling(w).mean()
    for w in [5, 10, 20]:
        df[f"large_down_day_ratio_{w}"] = large_down.rolling(w).mean()
        df[f"large_up_day_ratio_{w}"] = large_up.rolling(w).mean()

    for w in [5, 10, 20, 60, 120]:
        roll_high = close.rolling(w).max()
        roll_low = close.rolling(w).min()
        denom = (roll_high - roll_low).replace(0, np.nan)
        df[f"drawdown_{w}"] = close / roll_high - 1.0
        if w in [5, 10, 20, 60]:
            df[f"price_position_{w}"] = (close - roll_low) / denom
            df[f"close_to_{w}d_high"] = close / roll_high - 1.0
    df["price_position_5_minus_20"] = df["price_position_5"] - df["price_position_20"]
    df["price_position_10_minus_20"] = df["price_position_10"] - df["price_position_20"]

    df["volume_change"] = volume.pct_change()
    volume_ma20 = volume.rolling(20).mean()
    volume_std20 = volume.rolling(20).std()
    for w in [5, 10, 20]:
        vma = volume.rolling(w).mean()
        vstd = volume.rolling(w).std()
        df[f"volume_ratio_{w}"] = volume / vma
        df[f"volume_zscore_{w}"] = (volume - vma) / vstd.replace(0, np.nan)
    df["volume_ratio_5_minus_20"] = df["volume_ratio_5"] - df["volume_ratio_20"]
    df["volume_ratio_10_minus_20"] = df["volume_ratio_10"] - df["volume_ratio_20"]

    # v8.6: 가격/추세 기반 하락 전조 피처
    for w in [20, 60]:
        prior_low = close.rolling(w).min().shift(1)
        prior_high = close.rolling(w).max().shift(1)
        df[f"breakdown_{w}"] = (close < prior_low).astype(float)
        df[f"failed_rebound_{w}"] = (close < prior_high * 0.97).astype(float)
    df["lower_high_20"] = (close.rolling(5).max() < close.rolling(20).max().shift(5)).astype(float)
    df["trend_consistency_20"] = (df["daily_return"] > 0).astype(float).rolling(20).mean()
    df["trend_consistency_60"] = (df["daily_return"] > 0).astype(float).rolling(60).mean()
    df["bearish_ma_stack"] = ((df["ma_5"] < df["ma_20"]) & (df["ma_20"] < df["ma_60"])).astype(float)

    # v8.6: 가격+거래량 기반 매도 압력 피처
    down_volume = ((df["daily_return"] < 0).astype(float) * volume)
    for w in [5, 10, 20]:
        df[f"down_volume_ratio_{w}"] = down_volume.rolling(w).sum() / volume.rolling(w).sum().replace(0, np.nan)
    df["high_volume_down_day"] = ((df["daily_return"] < 0) & (df["volume_zscore_20"] > 1.0)).astype(float)
    for w in [5, 10, 20]:
        df[f"high_volume_down_ratio_{w}"] = df["high_volume_down_day"].rolling(w).mean()
    df["price_down_volume_up"] = ((df["daily_return"] < 0) & (df["volume_ratio_20"] > 1.2)).astype(float)
    df["weak_rebound_volume"] = ((df["return_5d"] > 0) & (df["volume_ratio_20"] < 0.8)).astype(float)
    df["down_momentum_volume_confirm"] = ((df["return_20d"] < 0) & (df["volume_ratio_20"] > 1.0)).astype(float)
    df["volume_shock_20"] = volume / volume_ma20
    df["volume_shock_rank_252"] = rolling_rank_last(df["volume_shock_20"], 252)
    df["down_volume_shock"] = ((df["daily_return"] < -0.01) & (df["volume_shock_20"] > 1.5)).astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["true_range"] = tr
    df["true_range_pct"] = tr / close
    for w in [5, 10, 14, 20, 60]:
        df[f"atr_{w}"] = tr.rolling(w).mean()
        df[f"atr_pct_{w}"] = df[f"atr_{w}"] / close
    df["atr_ratio_5_20"] = df["atr_5"] / df["atr_20"]
    df["atr_ratio_10_20"] = df["atr_10"] / df["atr_20"]
    df["atr_ratio_14_60"] = df["atr_14"] / df["atr_60"]
    df["atr_ratio_20_60"] = df["atr_20"] / df["atr_60"]
    df["atr_accel_5"] = df["atr_14"] / df["atr_14"].shift(5) - 1.0
    df["atr_rank_252"] = rolling_rank_last(df["atr_pct_20"], 252)

    log_hl = np.log(high / low).replace([np.inf, -np.inf], np.nan)
    log_co = np.log(close / open_).replace([np.inf, -np.inf], np.nan)
    log_oc = np.log(open_ / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    log_ho = np.log(high / open_).replace([np.inf, -np.inf], np.nan)
    log_lo = np.log(low / open_).replace([np.inf, -np.inf], np.nan)

    parkinson_var = (1.0 / (4.0 * np.log(2.0))) * (log_hl ** 2)
    gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    rs_var = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    for w in [5, 10, 20, 60]:
        df[f"realized_vol_{w}"] = df["daily_return"].rolling(w).std()
        df[f"ewma_vol_{w}"] = df["daily_return"].ewm(span=w, adjust=False).std()
        df[f"parkinson_vol_{w}"] = np.sqrt(parkinson_var.rolling(w).mean().clip(lower=0))
        df[f"garman_klass_vol_{w}"] = np.sqrt(gk_var.rolling(w).mean().clip(lower=0))
        df[f"rogers_satchell_vol_{w}"] = np.sqrt(rs_var.rolling(w).mean().clip(lower=0))
        k = 0.34 / (1.34 + (w + 1.0) / max(w - 1.0, 1.0))
        yz_var = log_oc.rolling(w).var() + k * log_co.rolling(w).var() + (1.0 - k) * rs_var.rolling(w).mean()
        df[f"yang_zhang_vol_{w}"] = np.sqrt(yz_var.clip(lower=0))

    df["realized_vol_ratio_5_20"] = df["realized_vol_5"] / df["realized_vol_20"]
    df["realized_vol_ratio_10_20"] = df["realized_vol_10"] / df["realized_vol_20"]
    df["realized_vol_ratio_20_60"] = df["realized_vol_20"] / df["realized_vol_60"]
    df["parkinson_vol_ratio_20_60"] = df["parkinson_vol_20"] / df["parkinson_vol_60"]
    df["yang_zhang_vol_ratio_20_60"] = df["yang_zhang_vol_20"] / df["yang_zhang_vol_60"]
    df["vol_of_vol_20"] = df["realized_vol_20"].rolling(20).std()

    downside_return = df["daily_return"].clip(upper=0)
    for w in [5, 10, 20, 60]:
        df[f"downside_vol_{w}"] = downside_return.rolling(w).std()
    df["semi_vol_5"] = np.sqrt((downside_return ** 2).rolling(5).mean())
    df["semi_vol_10"] = np.sqrt((downside_return ** 2).rolling(10).mean())
    df["semi_vol_20"] = np.sqrt((downside_return ** 2).rolling(20).mean())
    dd20 = close / close.rolling(20).max() - 1.0
    dd60 = close / close.rolling(60).max() - 1.0
    df["ulcer_index_20"] = np.sqrt((dd20 ** 2).rolling(20).mean())
    df["ulcer_index_60"] = np.sqrt((dd60 ** 2).rolling(60).mean())
    df["ulcer_rank_252"] = rolling_rank_last(df["ulcer_index_20"], 252)

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_width_20"] = (4.0 * std20) / ma20
    df["bb_width_rank_252"] = rolling_rank_last(df["bb_width_20"], 252)
    ema20 = close.ewm(span=20, adjust=False).mean()
    df["keltner_width_20"] = (4.0 * df["atr_20"]) / ema20
    squeeze_on = (df["bb_width_20"] < df["keltner_width_20"]).astype(float)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "squeeze_on": squeeze_on,
                    "squeeze_release": ((squeeze_on.shift(1) == 1.0) & (squeeze_on == 0.0)).astype(float),
                },
                index=df.index,
            ),
        ],
        axis=1,
    ).copy()

    df = add_future_targets(df, horizons)

    # v8.6.5 Pruned feature set
    # 기준: v8.6.2/v8.6.3 feature importance와 branch별 역할을 함께 반영.
    # 제거 원칙:
    # - 여러 branch에서 중요도가 0 또는 극히 낮았던 이벤트성 더미 피처 제거
    # - daily_return/log_return/return_3d/price_ma_5_gap처럼 중복·단기 노이즈가 큰 피처 축소
    # - Stage1 고변동 탐지에 필요한 핵심 변동성 피처는 유지
    # - 방향성 Up/Down은 가격·중기추세·거래량 중심으로 축소
    # v8.6.21 Pruned feature set
    # 기준: v8.6.20 feature_optimization_metrics + up_feature_importance 결과 반영.
    # 제거 대상:
    # - 평균/최대 중요도가 낮은 5D volume, z-score, event dummy 계열
    # - 5D/10D 단기 차분 중 중요도가 낮고 중복성이 큰 피처
    # - Stage1/DownRisk 양쪽에서 모두 낮은 피처
    # 목표: 112개 -> 약 82개로 축소해 20D single-label 모델의 노이즈를 줄인다.
    feature_cols = [
        # price / momentum / trend core
        "return_5d", "return_10d", "return_20d", "return_60d", "return_120d",
        "return_5d_minus_20d", "return_10d_minus_20d",
        "price_ma_20_gap", "price_ma_60_gap", "price_ma_120_gap", "price_ma_200_gap",
        "ma_gap_5_20", "ma_gap_20_60", "ma_gap_60_120", "ma_gap_50_200",
        "trend_slope_5", "trend_slope_10", "trend_slope_20", "trend_slope_60", "ma200_slope_60",
        "positive_return_ratio_10", "positive_return_ratio_20", "positive_return_ratio_60",
        "large_down_day_ratio_10", "large_down_day_ratio_20",
        "large_up_day_ratio_10", "large_up_day_ratio_20",
        "drawdown_5", "drawdown_10", "drawdown_20", "drawdown_60", "drawdown_120",
        "price_position_10", "price_position_20", "price_position_60",
        "price_position_5_minus_20", "price_position_10_minus_20",
        "close_to_10d_high", "close_to_20d_high", "close_to_60d_high",
        "trend_consistency_20", "trend_consistency_60", "bearish_ma_stack",

        # volume / selling pressure core
        "volume_ratio_20", "volume_zscore_20",
        "down_volume_ratio_10", "down_volume_ratio_20",
        "high_volume_down_ratio_10", "high_volume_down_ratio_20",
        "volume_shock_rank_252",

        # volatility / risk core
        "true_range_pct",
        "atr_pct_5", "atr_pct_10", "atr_pct_14", "atr_pct_20", "atr_pct_60", "atr_rank_252",
        "atr_ratio_14_60", "atr_ratio_20_60",
        "realized_vol_10", "realized_vol_20", "realized_vol_60", "realized_vol_ratio_10_20",
        "ewma_vol_20", "ewma_vol_60",
        "parkinson_vol_20", "parkinson_vol_60",
        "garman_klass_vol_20", "rogers_satchell_vol_20",
        "yang_zhang_vol_20", "yang_zhang_vol_60",
        "downside_vol_10", "downside_vol_20", "downside_vol_60",
        "semi_vol_10", "semi_vol_20",
        "ulcer_index_20", "ulcer_index_60", "ulcer_rank_252",
        "bb_width_20", "bb_width_rank_252", "keltner_width_20",
        "vol_of_vol_20",
    ]
    return df, [c for c in feature_cols if c in df.columns]


def _keep_existing_features(feature_cols: Sequence[str], candidates: Sequence[str]) -> List[str]:
    available = set(feature_cols)
    return [c for c in candidates if c in available]


REMOVED_LOW_VALUE_FEATURES_V8_6_5 = [
    # near-zero / unstable event features in previous diagnostics
    "log_return", "return_3d", "price_ma_5_gap",
    "breakdown_20", "breakdown_60",
    "failed_rebound_20", "failed_rebound_60",
    "high_volume_down_day", "price_down_volume_up", "weak_rebound_volume",
    "down_momentum_volume_confirm", "down_volume_shock",
    "realized_vol_ratio_20_60", "parkinson_vol_ratio_20_60",
    "yang_zhang_vol_ratio_20_60", "squeeze_on", "squeeze_release",
]


def build_downrisk_feature_sets(feature_cols: Sequence[str]) -> Dict[str, List[str]]:
    """v8.6.5 Pruned branch-specific feature sets.

    전체 feature_cols도 축소했지만, branch별 입력은 한 번 더 축소한다.
    목적은 branch별로 역할이 다른 피처만 남겨서 잡음과 불안정한 split을 줄이는 것이다.
    """
    price_trend_candidates = [
        "return_5d", "return_10d", "return_20d", "return_60d", "return_120d",
        "price_ma_20_gap", "price_ma_60_gap", "price_ma_120_gap", "price_ma_200_gap",
        "ma_gap_5_20", "ma_gap_20_60", "ma_gap_60_120", "ma_gap_50_200",
        "trend_slope_5", "trend_slope_10", "trend_slope_20", "trend_slope_60", "ma200_slope_60",
        "positive_return_ratio_5", "positive_return_ratio_10", "positive_return_ratio_20", "positive_return_ratio_60",
        "large_down_day_ratio_5", "large_down_day_ratio_10", "large_down_day_ratio_20",
        "large_up_day_ratio_5", "large_up_day_ratio_10", "large_up_day_ratio_20",
        "drawdown_5", "drawdown_10", "drawdown_20", "drawdown_60", "drawdown_120",
        "price_position_5", "price_position_10", "price_position_20", "price_position_60",
        "price_position_5_minus_20", "price_position_10_minus_20",
        "close_to_5d_high", "close_to_10d_high", "close_to_20d_high", "close_to_60d_high",
        "trend_consistency_20", "trend_consistency_60", "bearish_ma_stack",
    ]
    price_volume_candidates = price_trend_candidates + [
        "volume_ratio_20", "volume_zscore_20",
        "down_volume_ratio_20", "high_volume_down_ratio_20",
        "volume_shock_20", "volume_shock_rank_252",
    ]
    volatility_candidates = [
        "true_range_pct",
        "atr_pct_5", "atr_pct_10", "atr_pct_14", "atr_pct_20", "atr_pct_60", "atr_rank_252",
        "atr_ratio_5_20", "atr_ratio_10_20", "atr_ratio_14_60", "atr_ratio_20_60", "atr_accel_5",
        "realized_vol_5", "realized_vol_10", "realized_vol_20", "realized_vol_60",
        "realized_vol_ratio_5_20", "realized_vol_ratio_10_20",
        "ewma_vol_5", "ewma_vol_10", "ewma_vol_20", "ewma_vol_60",
        "parkinson_vol_20", "parkinson_vol_60",
        "garman_klass_vol_20", "rogers_satchell_vol_20",
        "yang_zhang_vol_20", "yang_zhang_vol_60",
        "downside_vol_5", "downside_vol_10", "downside_vol_20", "downside_vol_60",
        "semi_vol_5", "semi_vol_10", "semi_vol_20",
        "ulcer_index_20", "ulcer_index_60", "ulcer_rank_252",
        "bb_width_20", "bb_width_rank_252", "keltner_width_20",
        "vol_of_vol_20",
    ]
    return {
        "price_trend": _keep_existing_features(feature_cols, price_trend_candidates),
        "price_volume": _keep_existing_features(feature_cols, price_volume_candidates),
        "volatility": _keep_existing_features(feature_cols, volatility_candidates),
    }


def normalize_downrisk_branch_weights(cfg: Config) -> Dict[str, float]:
    raw = {
        "price_trend": float(cfg.down_price_trend_weight),
        "price_volume": float(cfg.down_price_volume_weight),
        "volatility": float(cfg.down_volatility_weight),
        "high_vol": float(cfg.down_highvol_weight),
    }
    total = sum(max(0.0, v) for v in raw.values())
    if total <= 0:
        return {"price_trend": 0.45, "price_volume": 0.35, "volatility": 0.20, "high_vol": 0.00}
    return {k: max(0.0, v) / total for k, v in raw.items()}


def compute_overall_risk_prob(
    prob_high_vol: object,
    prob_down: object,
    cfg: Config,
    prob_up: Optional[object] = None,
) -> object:
    """
    v8.6.2 전체 리스크 점수.

    핵심 변경:
    - prob_down은 큰 하락위험이 아니라 방향성 하락 확률로 해석한다.
    - 상승 확률(prob_up)이 있으면 max(prob_down - prob_up, 0)을 추가 위험으로 반영한다.
    - scalar와 pandas Series 모두 처리한다.
    """
    hv_w = max(0.0, float(cfg.overall_risk_high_vol_weight))
    dn_w = max(0.0, float(cfg.overall_risk_down_weight))
    gap_w = max(0.0, float(getattr(cfg, "overall_risk_down_minus_up_weight", 0.0)))

    if prob_up is None:
        gap = 0.0
        total = hv_w + dn_w
        if total <= 0:
            hv_w, dn_w, total = 0.35, 0.50, 0.85
        score = (hv_w / total) * prob_high_vol + (dn_w / total) * prob_down
    else:
        gap = prob_down - prob_up
        if hasattr(gap, "clip"):
            gap = gap.clip(0.0, 1.0)
        else:
            gap = float(np.clip(gap, 0.0, 1.0))
        total = hv_w + dn_w + gap_w
        if total <= 0:
            hv_w, dn_w, gap_w, total = 0.35, 0.50, 0.15, 1.0
        score = (hv_w / total) * prob_high_vol + (dn_w / total) * prob_down + (gap_w / total) * gap

    if hasattr(score, "clip"):
        return score.clip(0.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))

def combine_weighted_importance(
    histories: Dict[str, List[Dict[str, float]]],
    weights: Dict[str, float],
) -> Dict[str, float]:
    combined: Dict[str, float] = {}
    for branch, hist in histories.items():
        w = float(weights.get(branch, 0.0))
        mean_imp = mean_importance(hist)
        for feature, imp in mean_imp.items():
            combined[feature] = combined.get(feature, 0.0) + w * float(imp)
    return dict(sorted(combined.items(), key=lambda kv: kv[1], reverse=True))


# ============================================================
# 3. LABEL DESIGN
# ============================================================

def qclip(q: float) -> float:
    return float(np.clip(q, 0.01, 0.99))


def compute_policy_thresholds(train_df: pd.DataFrame, horizon: int, policy: LabelPolicy) -> Dict[str, float]:
    fvol = train_df[f"future_volatility_{horizon}d"]
    fmin = train_df[f"future_min_return_{horizon}d"]
    fmax = train_df[f"future_max_return_{horizon}d"]
    down_loose_q = qclip(policy.down_q + 0.05)
    down_strict_q = qclip(policy.down_q - 0.05)
    up_loose_q = qclip(policy.up_q - 0.05)
    up_strict_q = qclip(policy.up_q + 0.05)
    return {
        "policy_name": policy.name,
        "vol": float(fvol.quantile(policy.vol_q)),
        "down": float(fmin.quantile(policy.down_q)),
        "down_loose": float(fmin.quantile(down_loose_q)),
        "down_strict": float(fmin.quantile(down_strict_q)),
        "up": float(fmax.quantile(policy.up_q)),
        "up_loose": float(fmax.quantile(up_loose_q)),
        "up_strict": float(fmax.quantile(up_strict_q)),
        "vol_q": float(policy.vol_q),
        "down_q": float(policy.down_q),
        "up_q": float(policy.up_q),
    }


def assign_label(row: pd.Series, horizon: int, th: Dict[str, float]) -> str:
    future_vol = row[f"future_volatility_{horizon}d"]
    future_ret = row[f"future_return_{horizon}d"]
    future_max_ret = row[f"future_max_return_{horizon}d"]
    future_min_ret = row[f"future_min_return_{horizon}d"]

    atr_rank = row.get("atr_rank_252", np.nan)
    atr_ratio = row.get("atr_ratio_20_60", np.nan)
    return_20d = row.get("return_20d", np.nan)
    drawdown_60 = row.get("drawdown_60", np.nan)
    price_position_60 = row.get("price_position_60", np.nan)
    positive_ratio_20 = row.get("positive_return_ratio_20", np.nan)
    large_down_ratio_20 = row.get("large_down_day_ratio_20", np.nan)
    ulcer_rank = row.get("ulcer_rank_252", np.nan)
    bb_rank = row.get("bb_width_rank_252", np.nan)

    atr_high = bool(pd.notna(atr_rank) and atr_rank > 0.70)
    atr_extreme = bool(pd.notna(atr_rank) and atr_rank > 0.85)
    atr_expanding = bool(pd.notna(atr_ratio) and atr_ratio > 1.15)
    atr_compressed = bool(pd.notna(atr_rank) and atr_rank < 0.30)

    down_pressure_now = (
        (pd.notna(drawdown_60) and drawdown_60 < -0.08)
        or (pd.notna(return_20d) and return_20d < -0.05)
        or (pd.notna(large_down_ratio_20) and large_down_ratio_20 > 0.20)
        or (pd.notna(ulcer_rank) and ulcer_rank > 0.70)
    )
    up_pressure_now = (
        (pd.notna(return_20d) and return_20d > 0.05)
        and (pd.notna(price_position_60) and price_position_60 > 0.70)
        and (pd.notna(positive_ratio_20) and positive_ratio_20 > 0.55)
        and not (pd.notna(ulcer_rank) and ulcer_rank > 0.70)
    )
    squeeze_or_breakout = (
        (pd.notna(bb_rank) and bb_rank < 0.30)
        and (pd.notna(atr_ratio) and atr_ratio > 1.05)
    )

    down_threshold = th["down"]
    up_threshold = th["up"]

    if atr_high and atr_expanding and down_pressure_now:
        down_threshold = th["down_loose"]
        up_threshold = th["up_loose"]
    elif atr_high and atr_expanding and up_pressure_now:
        down_threshold = th["down_strict"]
        up_threshold = th["up_loose"]
    elif atr_extreme:
        down_threshold = th["down_loose"]
        up_threshold = th["up_loose"]
    elif atr_compressed and squeeze_or_breakout:
        down_threshold = th["down_loose"]
        up_threshold = th["up_loose"]

    is_high_vol = (
        future_vol >= th["vol"]
        or future_min_ret <= down_threshold
        or future_max_ret >= up_threshold
    )
    if not is_high_vol:
        return "정상"

    # Down-risk는 방어 목적상 우선한다.
    severe_down = future_min_ret <= th["down"]
    if severe_down and not up_pressure_now:
        return "하락고변동"

    if future_min_ret <= down_threshold:
        return "하락고변동"

    if future_max_ret >= up_threshold and future_ret > 0:
        return "상승고변동"

    if abs(future_max_ret) >= abs(future_min_ret):
        return "상승고변동"
    return "하락고변동"


def make_labels(df: pd.DataFrame, horizon: int, th: Dict[str, float]) -> pd.Series:
    """
    assign_label(row) 기반 apply를 벡터화한 라벨 생성 함수.

    목적:
    - walk-forward 반복 학습 속도 개선
    - 라벨 로직을 기존 assign_label과 최대한 동일하게 유지

    검증 권장:
    - 변경 직후에는 일부 구간에서 old apply 방식과 crosstab 비교를 수행하는 것이 안전하다.
    """
    idx = df.index

    future_vol = df[f"future_volatility_{horizon}d"]
    future_ret = df[f"future_return_{horizon}d"]
    future_max_ret = df[f"future_max_return_{horizon}d"]
    future_min_ret = df[f"future_min_return_{horizon}d"]

    atr_rank = df.get("atr_rank_252", pd.Series(np.nan, index=idx))
    atr_ratio = df.get("atr_ratio_20_60", pd.Series(np.nan, index=idx))
    return_20d = df.get("return_20d", pd.Series(np.nan, index=idx))
    drawdown_60 = df.get("drawdown_60", pd.Series(np.nan, index=idx))
    price_position_60 = df.get("price_position_60", pd.Series(np.nan, index=idx))
    positive_ratio_20 = df.get("positive_return_ratio_20", pd.Series(np.nan, index=idx))
    large_down_ratio_20 = df.get("large_down_day_ratio_20", pd.Series(np.nan, index=idx))
    ulcer_rank = df.get("ulcer_rank_252", pd.Series(np.nan, index=idx))
    bb_rank = df.get("bb_width_rank_252", pd.Series(np.nan, index=idx))

    atr_high = atr_rank > 0.70
    atr_extreme = atr_rank > 0.85
    atr_expanding = atr_ratio > 1.15
    atr_compressed = atr_rank < 0.30

    down_pressure_now = (
        (drawdown_60 < -0.08)
        | (return_20d < -0.05)
        | (large_down_ratio_20 > 0.20)
        | (ulcer_rank > 0.70)
    )
    up_pressure_now = (
        (return_20d > 0.05)
        & (price_position_60 > 0.70)
        & (positive_ratio_20 > 0.55)
        & ~(ulcer_rank > 0.70)
    )
    squeeze_or_breakout = (bb_rank < 0.30) & (atr_ratio > 1.05)

    cond1 = atr_high & atr_expanding & down_pressure_now
    cond2 = (~cond1) & atr_high & atr_expanding & up_pressure_now
    cond3 = (~cond1) & (~cond2) & atr_extreme
    cond4 = (~cond1) & (~cond2) & (~cond3) & atr_compressed & squeeze_or_breakout

    down_threshold = pd.Series(th["down"], index=idx, dtype=float)
    up_threshold = pd.Series(th["up"], index=idx, dtype=float)

    down_threshold = down_threshold.mask(cond1, th["down_loose"])
    up_threshold = up_threshold.mask(cond1, th["up_loose"])
    down_threshold = down_threshold.mask(cond2, th["down_strict"])
    up_threshold = up_threshold.mask(cond2, th["up_loose"])
    down_threshold = down_threshold.mask(cond3, th["down_loose"])
    up_threshold = up_threshold.mask(cond3, th["up_loose"])
    down_threshold = down_threshold.mask(cond4, th["down_loose"])
    up_threshold = up_threshold.mask(cond4, th["up_loose"])

    is_high_vol = (
        (future_vol >= th["vol"])
        | (future_min_ret <= down_threshold)
        | (future_max_ret >= up_threshold)
    )

    labels = pd.Series("정상", index=idx, dtype=object)

    severe_down = future_min_ret <= th["down"]
    down_first = is_high_vol & severe_down & ~up_pressure_now
    labels.loc[down_first] = "하락고변동"

    down_second = is_high_vol & ~down_first & (future_min_ret <= down_threshold)
    labels.loc[down_second] = "하락고변동"

    up_first = (
        is_high_vol
        & ~down_first
        & ~down_second
        & (future_max_ret >= up_threshold)
        & (future_ret > 0)
    )
    labels.loc[up_first] = "상승고변동"

    remaining = is_high_vol & (labels == "정상")
    up_remaining = remaining & (future_max_ret.abs() >= future_min_ret.abs())
    labels.loc[up_remaining] = "상승고변동"
    labels.loc[remaining & ~up_remaining] = "하락고변동"

    return labels


def make_direction_labels(df: pd.DataFrame, horizon: int, cfg: Config, direction: str) -> pd.Series:
    """방향성 이진 라벨 생성.

    - up: future_return_h > +direction_return_threshold
    - down: future_return_h < -direction_return_threshold
    - 중립 구간은 두 모델 모두 0으로 처리된다.
    """
    ret = df[f"future_return_{horizon}d"].astype(float)
    thr = float(getattr(cfg, "direction_return_threshold", 0.005))
    if direction == "up":
        return (ret > thr).astype(int)
    if direction == "down":
        return (ret < -thr).astype(int)
    raise ValueError(f"unknown direction: {direction}")


def assign_direction_label(row: pd.Series, horizon: int, cfg: Config) -> str:
    ret = float(row[f"future_return_{horizon}d"])
    thr = float(getattr(cfg, "direction_return_threshold", 0.005))
    if ret > thr:
        return "상승"
    if ret < -thr:
        return "하락"
    return "중립"


# ============================================================
# 4. MODEL
# ============================================================

def calc_scale_pos_weight(y_binary: np.ndarray) -> float:
    pos = float(np.sum(y_binary == 1))
    neg = float(np.sum(y_binary == 0))
    if pos <= 0 or neg <= 0:
        return 1.0
    return max(0.1, min(20.0, neg / pos))


def _get_horizon_train_multiplier(cfg: Config, horizon: int) -> float:
    """Return configured train-window multiplier for a horizon.

    Examples:
    - H5, multiplier 150 -> 750 training rows
    - H10, multiplier 100 -> 1000 training rows
    - H20, multiplier 63 -> 1260 training rows
    """
    h = int(horizon)
    if h == 5:
        return float(getattr(cfg, "horizon_train_multiplier_5d", 150.0))
    if h == 10:
        return float(getattr(cfg, "horizon_train_multiplier_10d", 100.0))
    if h == 20:
        return float(getattr(cfg, "horizon_train_multiplier_20d", 63.0))
    return float(getattr(cfg, "horizon_train_multiplier_20d", 63.0))


def horizon_train_rows(cfg: Config, horizon: int, *, for_direction_strength: bool = False) -> Optional[int]:
    """Compute horizon-specific training rows from horizon * multiplier.

    If horizon train-window mode is disabled, returns cfg.max_train_rows for stage1/up/down
    and cfg.direction_strength_max_train_rows for direction-strength.
    """
    if for_direction_strength:
        if not bool(getattr(cfg, "direction_strength_use_horizon_train_window", True)):
            return getattr(cfg, "direction_strength_max_train_rows", None)
    else:
        if not bool(getattr(cfg, "use_horizon_train_window", True)):
            return getattr(cfg, "max_train_rows", None)
    raw_rows = int(round(float(horizon) * _get_horizon_train_multiplier(cfg, int(horizon))))
    min_rows = int(getattr(cfg, "horizon_train_min_rows", 504))
    rows = max(min_rows, raw_rows)
    cap = getattr(cfg, "horizon_train_max_rows_cap", None)
    if cap is not None:
        rows = min(rows, int(cap))
    # Global max_train_rows can still act as a hard cap if explicitly supplied.
    if not for_direction_strength and getattr(cfg, "max_train_rows", None) is not None:
        rows = min(rows, int(getattr(cfg, "max_train_rows")))
    if for_direction_strength and getattr(cfg, "direction_strength_max_train_rows", None) is not None:
        # Keep legacy cap for safety, but default is equal to H20*63=1260.
        rows = min(rows, int(getattr(cfg, "direction_strength_max_train_rows")))
    return int(rows)


def apply_horizon_train_window(train_df: pd.DataFrame, cfg: Config, horizon: int, *, for_direction_strength: bool = False) -> Tuple[pd.DataFrame, int, float]:
    rows = horizon_train_rows(cfg, horizon, for_direction_strength=for_direction_strength)
    if rows is None:
        out = train_df
    else:
        out = train_df.tail(int(rows))
    ratio = float(len(out)) / max(float(horizon), 1.0)
    return out, int(len(out)), ratio


def horizon_train_window_config(cfg: Config) -> Dict[str, object]:
    rows = {str(h): horizon_train_rows(cfg, h, for_direction_strength=False) for h in (5, 10, 20)}
    ratios = {str(h): (float(rows[str(h)]) / float(h) if rows[str(h)] is not None else None) for h in (5, 10, 20)}
    strength_rows = {str(h): horizon_train_rows(cfg, h, for_direction_strength=True) for h in (5, 10, 20)}
    strength_ratios = {str(h): (float(strength_rows[str(h)]) / float(h) if strength_rows[str(h)] is not None else None) for h in (5, 10, 20)}
    return {
        "enabled": bool(getattr(cfg, "use_horizon_train_window", True)),
        "direction_strength_enabled": bool(getattr(cfg, "direction_strength_use_horizon_train_window", True)),
        "min_rows": int(getattr(cfg, "horizon_train_min_rows", 504)),
        "max_rows_cap": getattr(cfg, "horizon_train_max_rows_cap", None),
        "multipliers": {
            "5": float(getattr(cfg, "horizon_train_multiplier_5d", 150.0)),
            "10": float(getattr(cfg, "horizon_train_multiplier_10d", 100.0)),
            "20": float(getattr(cfg, "horizon_train_multiplier_20d", 63.0)),
        },
        "stage1_train_rows_by_horizon": rows,
        "stage1_train_ratio_by_horizon": ratios,
        "direction_strength_train_rows_by_horizon": strength_rows,
        "direction_strength_train_ratio_by_horizon": strength_ratios,
    }


def make_xgb_stage1(cfg: Config, scale_pos_weight: float, n_estimators: Optional[int] = None) -> Pipeline:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators or cfg.stage1_n_estimators,
        learning_rate=cfg.stage1_learning_rate,
        max_depth=cfg.stage1_max_depth,
        min_child_weight=cfg.stage1_min_child_weight,
        subsample=cfg.stage1_subsample,
        colsample_bytree=cfg.stage1_colsample_bytree,
        reg_lambda=cfg.stage1_reg_lambda,
        reg_alpha=cfg.stage1_reg_alpha,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def make_xgb_downrisk(cfg: Config, scale_pos_weight: float, n_estimators: Optional[int] = None) -> Pipeline:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators or cfg.down_n_estimators,
        learning_rate=cfg.down_learning_rate,
        max_depth=cfg.down_max_depth,
        min_child_weight=cfg.down_min_child_weight,
        subsample=cfg.down_subsample,
        colsample_bytree=cfg.down_colsample_bytree,
        reg_lambda=cfg.down_reg_lambda,
        reg_alpha=cfg.down_reg_alpha,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def safe_auc(y_true: np.ndarray, p: np.ndarray, kind: str) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    if kind == "roc":
        return float(roc_auc_score(y_true, p))
    if kind == "pr":
        return float(average_precision_score(y_true, p))
    raise ValueError(kind)


def policy_imbalance_penalty(y_high: np.ndarray, y_down: np.ndarray) -> float:
    hv_rate = float(np.mean(y_high)) if len(y_high) else 0.0
    down_rate = float(np.mean(y_down)) if len(y_down) else 0.0
    # 너무 희소하거나 너무 넓은 라벨 정책을 방지
    return abs(hv_rate - 0.33) + 0.75 * abs(down_rate - 0.16)


def select_label_policy(train_df: pd.DataFrame, horizon: int, feature_cols: List[str], cfg: Config) -> Tuple[LabelPolicy, Dict[str, float]]:
    if not cfg.use_adaptive_label_policy:
        th = compute_policy_thresholds(train_df, horizon, cfg.fixed_label_policy)
        return cfg.fixed_label_policy, th

    valid_rows = min(cfg.label_search_valid_rows, max(126, len(train_df) // 4))
    if len(train_df) < cfg.min_train_rows + valid_rows:
        th = compute_policy_thresholds(train_df, horizon, cfg.fixed_label_policy)
        return cfg.fixed_label_policy, th

    inner_train = train_df.iloc[:-valid_rows].copy()
    inner_valid = train_df.iloc[-valid_rows:].copy()

    best_score = -np.inf
    best_policy = cfg.fixed_label_policy
    best_th = compute_policy_thresholds(train_df, horizon, cfg.fixed_label_policy)

    X_inner = inner_train[feature_cols]
    X_valid = inner_valid[feature_cols]

    for policy in cfg.label_policy_candidates:
        th_inner = compute_policy_thresholds(inner_train, horizon, policy)
        labels_inner = make_labels(inner_train, horizon, th_inner)
        labels_valid = make_labels(inner_valid, horizon, th_inner)

        y_high = (labels_inner != "정상").astype(int).values
        y_high_valid = (labels_valid != "정상").astype(int).values
        y_down = (labels_inner == "하락고변동").astype(int).values
        y_down_valid = (labels_valid == "하락고변동").astype(int).values

        if len(np.unique(y_high)) < 2 or int(y_high.sum()) < cfg.label_search_min_positive:
            continue

        try:
            m_high = make_xgb_stage1(cfg, calc_scale_pos_weight(y_high), cfg.label_search_stage1_estimators)
            m_high.fit(X_inner, y_high)
            p_high = m_high.predict_proba(X_valid)[:, 1]
            high_pr = safe_auc(y_high_valid, p_high, "pr") or 0.0
            high_roc = safe_auc(y_high_valid, p_high, "roc") or 0.5
        except Exception:
            continue

        down_pr = 0.0
        down_roc = 0.5
        if len(np.unique(y_down)) == 2 and int(y_down.sum()) >= cfg.label_search_min_positive:
            try:
                m_down = make_xgb_downrisk(cfg, calc_scale_pos_weight(y_down), cfg.label_search_down_estimators)
                m_down.fit(X_inner, y_down)
                p_down = m_down.predict_proba(X_valid)[:, 1]
                down_pr = safe_auc(y_down_valid, p_down, "pr") or 0.0
                down_roc = safe_auc(y_down_valid, p_down, "roc") or 0.5
            except Exception:
                pass

        penalty = policy_imbalance_penalty(y_high, y_down)
        score = 0.35 * high_pr + 0.25 * down_pr + 0.20 * high_roc + 0.10 * down_roc - 0.10 * penalty
        if score > best_score:
            best_score = float(score)
            best_policy = policy
            best_th = compute_policy_thresholds(train_df, horizon, policy)

    return best_policy, best_th


# ============================================================
# 5. WALK-FORWARD PREDICTION
# ============================================================

def extract_model_importance(pipeline: Pipeline, feature_cols: List[str]) -> Dict[str, float]:
    try:
        imp = np.asarray(pipeline.named_steps["model"].feature_importances_, dtype=float)
        if len(imp) != len(feature_cols):
            return {}
        return {f: float(v) for f, v in zip(feature_cols, imp)}
    except Exception:
        return {}


def mean_importance(history: List[Dict[str, float]]) -> Dict[str, float]:
    if not history:
        return {}
    imp_df = pd.DataFrame(history).fillna(0.0)
    return imp_df.mean(axis=0).sort_values(ascending=False).to_dict()


def ensemble_weights(cfg: Config) -> Tuple[Dict[int, float], Dict[int, float]]:
    # v8.6.21에서 cfg.horizons에는 UP_STRENGTHENING 5D 라벨 생성을 위해 5가 포함될 수 있다.
    # Stage1/Down-risk ensemble은 기존 H10/H20 중심으로 유지하고, H5는 allocation risk score에 반영하지 않는다.
    hv = {5: 0.0, 10: cfg.high_vol_weight_h10, 20: cfg.high_vol_weight_h20}
    dn = {5: 0.0, 10: cfg.down_risk_weight_h10, 20: cfg.down_risk_weight_h20}
    hv = {h: hv.get(h, 0.0) for h in cfg.horizons}
    dn = {h: dn.get(h, 0.0) for h in cfg.horizons}
    hv_sum = sum(hv.values())
    dn_sum = sum(dn.values())
    if hv_sum <= 0:
        hv = {h: 1.0 / max(1, len(cfg.horizons)) for h in cfg.horizons}
        hv_sum = sum(hv.values())
    if dn_sum <= 0:
        dn = {h: 1.0 / max(1, len(cfg.horizons)) for h in cfg.horizons}
        dn_sum = sum(dn.values())
    return {h: v / hv_sum for h, v in hv.items()}, {h: v / dn_sum for h, v in dn.items()}


def get_multi_strength_horizons(cfg: Config) -> Tuple[int, ...]:
    hs = getattr(cfg, "multi_strength_horizons", (5, 10, 20))
    if isinstance(hs, str):
        vals = [int(x.strip()) for x in hs.split(",") if x.strip()]
    else:
        vals = [int(x) for x in hs]
    vals = sorted(set(vals))
    return tuple(vals) if vals else (20,)


def get_up_strength_horizon_weights(cfg: Config) -> Dict[int, float]:
    hs = get_multi_strength_horizons(cfg)
    raw = {
        5: float(getattr(cfg, "up_strength_weight_5d", 0.25)),
        10: float(getattr(cfg, "up_strength_weight_10d", 0.35)),
        20: float(getattr(cfg, "up_strength_weight_20d", 0.40)),
    }
    weights = {h: max(0.0, raw.get(h, 1.0 / max(1, len(hs)))) for h in hs}
    total = sum(weights.values())
    if total <= 0:
        weights = {h: 1.0 / max(1, len(hs)) for h in hs}
    else:
        weights = {h: v / total for h, v in weights.items()}
    return weights


def combine_up_strength_score_from_values(values: Dict[int, float], cfg: Config) -> float:
    weights = get_up_strength_horizon_weights(cfg)
    return float(np.clip(sum(weights.get(h, 0.0) * float(values.get(h, 0.0)) for h in weights), 0.0, 1.0))


def get_down_strength_horizon_weights(cfg: Config) -> Dict[int, float]:
    hs = get_multi_strength_horizons(cfg)
    raw = {
        5: float(getattr(cfg, "down_strength_weight_5d", getattr(cfg, "up_strength_weight_5d", 0.25))),
        10: float(getattr(cfg, "down_strength_weight_10d", getattr(cfg, "up_strength_weight_10d", 0.35))),
        20: float(getattr(cfg, "down_strength_weight_20d", getattr(cfg, "up_strength_weight_20d", 0.40))),
    }
    weights = {h: max(0.0, raw.get(h, 1.0 / max(1, len(hs)))) for h in hs}
    total = sum(weights.values())
    if total <= 0:
        weights = {h: 1.0 / max(1, len(hs)) for h in hs}
    else:
        weights = {h: v / total for h, v in weights.items()}
    return weights


def combine_down_strength_score_from_values(values: Dict[int, float], cfg: Config) -> float:
    weights = get_down_strength_horizon_weights(cfg)
    return float(np.clip(sum(weights.get(h, 0.0) * float(values.get(h, 0.0)) for h in weights), 0.0, 1.0))


def get_down_strength_score_from_row(row: pd.Series, cfg: Config) -> float:
    if "prob_down_strengthening_score" in row.index and not pd.isna(row.get("prob_down_strengthening_score")):
        return float(np.clip(row.get("prob_down_strengthening_score", 0.0), 0.0, 1.0))
    vals: Dict[int, float] = {}
    for h in get_multi_strength_horizons(cfg):
        col = f"prob_down_strengthening_{h}d"
        vals[h] = _row_float(row, col, _row_float(row, "prob_down_strengthening", 0.0) if h == int(getattr(cfg, "direction_strength_horizon", 20)) else 0.0)
    return combine_down_strength_score_from_values(vals, cfg)


def get_up_strength_score_from_row(row: pd.Series, cfg: Config) -> float:
    if "prob_up_strengthening_score" in row.index and not pd.isna(row.get("prob_up_strengthening_score")):
        return float(np.clip(row.get("prob_up_strengthening_score", 0.0), 0.0, 1.0))
    vals: Dict[int, float] = {}
    for h in get_multi_strength_horizons(cfg):
        col = f"prob_up_strengthening_{h}d"
        vals[h] = _row_float(row, col, _row_float(row, "prob_up_strengthening", 0.0) if h == int(getattr(cfg, "direction_strength_horizon", 20)) else 0.0)
    return combine_up_strength_score_from_values(vals, cfg)


def recompute_multi_horizon_strength_score(pred_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    hs = get_multi_strength_horizons(cfg)

    up_vals: Dict[int, pd.Series] = {}
    down_vals: Dict[int, pd.Series] = {}
    for h in hs:
        up_col = f"prob_up_strengthening_{h}d"
        down_col = f"prob_down_strengthening_{h}d"
        if up_col in pred_df.columns:
            up_vals[h] = pred_df[up_col].astype(float).clip(0.0, 1.0)
        if down_col in pred_df.columns:
            down_vals[h] = pred_df[down_col].astype(float).clip(0.0, 1.0)

    if up_vals:
        weights = get_up_strength_horizon_weights(cfg)
        score = pd.Series(0.0, index=pred_df.index, dtype=float)
        for h, s in up_vals.items():
            score = score + weights.get(h, 0.0) * s
        pred_df["prob_up_strengthening_score"] = score.clip(0.0, 1.0)
        stacked = pd.concat(up_vals.values(), axis=1)
        pred_df["prob_up_strengthening_min_multi"] = stacked.min(axis=1).clip(0.0, 1.0)
        pred_df["prob_up_strengthening_max_multi"] = stacked.max(axis=1).clip(0.0, 1.0)
        if 5 in up_vals and 20 in up_vals:
            pred_df["prob_up_strengthening_spread_5d_20d"] = (up_vals[5] - up_vals[20]).clip(-1.0, 1.0)

    if down_vals:
        weights = get_down_strength_horizon_weights(cfg)
        score = pd.Series(0.0, index=pred_df.index, dtype=float)
        for h, s in down_vals.items():
            score = score + weights.get(h, 0.0) * s
        pred_df["prob_down_strengthening_score"] = score.clip(0.0, 1.0)
        stacked = pd.concat(down_vals.values(), axis=1)
        pred_df["prob_down_strengthening_min_multi"] = stacked.min(axis=1).clip(0.0, 1.0)
        pred_df["prob_down_strengthening_max_multi"] = stacked.max(axis=1).clip(0.0, 1.0)
        if 5 in down_vals and 20 in down_vals:
            pred_df["prob_down_strengthening_spread_5d_20d"] = (down_vals[5] - down_vals[20]).clip(-1.0, 1.0)

    return pred_df

def run_walk_forward(df: pd.DataFrame, feature_cols: List[str], cfg: Config) -> pd.DataFrame:
    """Walk-forward prediction with v8.6.2 directional Up/Down + overall risk.

    구조:
    - Stage1: 전체 피처로 정상/고변동 예측
    - Up-model: 가격/추세+거래량 피처로 상승 확률 예측
    - Down-model price_trend: 가격/추세 피처로 하락 확률 예측
    - Down-model price_volume: 가격/추세+거래량 피처로 하락 확률 예측
    - Down-model volatility: 변동성/ATR/Range 피처로 하락 확률 예측
    - Overall-risk: 하락 확률, 고변동 확률, 하락-상승 우위 점수를 종합
    """
    future_cols = []
    for h in cfg.horizons:
        future_cols.extend([
            f"future_volatility_{h}d",
            f"future_return_{h}d",
            f"future_max_return_{h}d",
            f"future_min_return_{h}d",
        ])
    valid_cols = feature_cols + future_cols + ["stock_next_return", "bond_next_return", "cash_next_return"]

    work = df.dropna(subset=valid_cols).copy()
    work = work[work.index >= pd.Timestamp(cfg.backtest_start_date)].copy()
    if len(work) < cfg.min_train_rows:
        raise ValueError("백테스트 가능한 데이터가 부족합니다.")

    all_df = ensure_direction_strength_helper_columns(df.copy())
    multi_strength_horizons = get_multi_strength_horizons(cfg)
    strength_full_by_h: Dict[int, Tuple[pd.Series, pd.Series, pd.DataFrame]] = {}
    for sh in multi_strength_horizons:
        strength_full_by_h[int(sh)] = build_direction_strength_labels(all_df, cfg, horizon=int(sh))
    legacy_h = int(getattr(cfg, "direction_strength_horizon", 20))
    if legacy_h not in strength_full_by_h:
        strength_full_by_h[legacy_h] = build_direction_strength_labels(all_df, cfg, horizon=legacy_h)
    strength_y_full, strength_valid_full, strength_aux_full = strength_full_by_h[legacy_h]
    candidate_positions = [all_df.index.get_loc(idx) for idx in work.index]
    max_gap = max(max(cfg.horizons), max(multi_strength_horizons))

    models: Dict[int, Dict[str, object]] = {}
    last_retrain_k: Optional[int] = None
    prediction_rows: List[Dict[str, object]] = []
    stage1_imp_hist: List[Dict[str, float]] = []
    up_imp_hist: List[Dict[str, float]] = []
    down_imp_hist_by_branch: Dict[str, List[Dict[str, float]]] = {
        "price_trend": [],
        "price_volume": [],
        "volatility": [],
    }
    policy_usage: Dict[str, int] = {}
    strength_models_by_h: Dict[int, Dict[str, object]] = {}
    bear_strength_models: Dict[str, object] = {}

    hv_w, dn_w = ensemble_weights(cfg)
    down_feature_sets = build_downrisk_feature_sets(feature_cols)
    direction_cols = down_feature_sets.get("price_volume") or down_feature_sets.get("price_trend") or feature_cols
    branch_weights = normalize_downrisk_branch_weights(cfg)

    for k, pos in enumerate(candidate_positions):
        date = all_df.index[pos]
        train_end_pos = pos - max_gap
        if train_end_pos < cfg.min_train_rows:
            continue

        need_retrain = (not models) or (last_retrain_k is None) or (k - last_retrain_k >= cfg.retrain_every_n_days)
        if need_retrain:
            train_df = all_df.iloc[:train_end_pos].copy().dropna(subset=valid_cols)
            if (not bool(getattr(cfg, "use_horizon_train_window", True))) and cfg.max_train_rows is not None:
                train_df = train_df.tail(int(cfg.max_train_rows))
            if len(train_df) < int(getattr(cfg, "horizon_train_min_rows", cfg.min_train_rows)):
                continue

            models = {}
            for h in cfg.horizons:
                train_df_h, train_rows_h, train_ratio_h = apply_horizon_train_window(train_df, cfg, int(h), for_direction_strength=False)
                if len(train_df_h) < int(getattr(cfg, "horizon_train_min_rows", 504)):
                    continue
                X_train_h = train_df_h[feature_cols]
                policy, th = select_label_policy(train_df_h, h, feature_cols, cfg)
                labels = make_labels(train_df_h, h, th)
                y_high = (labels != "정상").astype(int).values
                y_up = make_direction_labels(train_df_h, h, cfg, "up").values
                y_down = make_direction_labels(train_df_h, h, cfg, "down").values

                if len(np.unique(y_high)) < 2:
                    continue

                stage1_model = make_xgb_stage1(cfg, calc_scale_pos_weight(y_high))
                stage1_model.fit(X_train_h, y_high)
                imp1 = extract_model_importance(stage1_model, feature_cols)
                if imp1:
                    stage1_imp_hist.append(imp1)

                up_model: Optional[Pipeline] = None
                up_available = False
                if len(np.unique(y_up)) == 2 and int(y_up.sum()) >= int(getattr(cfg, "direction_min_positive", 20)):
                    up_model = make_xgb_downrisk(cfg, calc_scale_pos_weight(y_up))
                    up_model.fit(train_df_h[direction_cols], y_up)
                    up_available = True
                    impu = extract_model_importance(up_model, direction_cols)
                    if impu:
                        up_imp_hist.append(impu)

                down_models: Dict[str, Optional[Pipeline]] = {"price_trend": None, "price_volume": None, "volatility": None}
                down_available: Dict[str, bool] = {"price_trend": False, "price_volume": False, "volatility": False}
                if len(np.unique(y_down)) == 2 and int(y_down.sum()) >= int(getattr(cfg, "direction_min_positive", 20)):
                    for branch, cols in down_feature_sets.items():
                        if not cols:
                            continue
                        m_down = make_xgb_downrisk(cfg, calc_scale_pos_weight(y_down))
                        m_down.fit(train_df_h[cols], y_down)
                        down_models[branch] = m_down
                        down_available[branch] = True
                        impd = extract_model_importance(m_down, cols)
                        if impd:
                            down_imp_hist_by_branch[branch].append(impd)

                models[h] = {
                    "stage1": stage1_model,
                    "up_model": up_model,
                    "up_available": up_available,
                    "down_models": down_models,
                    "down_available": down_available,
                    "thresholds": th,
                    "policy": policy,
                }
                models[h]["train_rows"] = int(train_rows_h)
                models[h]["train_horizon_ratio"] = float(train_ratio_h)
                models[h]["train_window_multiplier"] = float(_get_horizon_train_multiplier(cfg, int(h)))
                policy_usage[f"H{h}:{policy.name}"] = policy_usage.get(f"H{h}:{policy.name}", 0) + 1

            if bool(getattr(cfg, "use_direction_strength_specialist", True)):
                try:
                    strength_train_df = ensure_direction_strength_helper_columns(train_df)
                    strength_models_by_h = {}
                    for sh in multi_strength_horizons:
                        y_strength_train_h, valid_strength_train_h, _ = build_direction_strength_labels(
                            strength_train_df, cfg, horizon=int(sh)
                        )
                        strength_models_by_h[int(sh)] = fit_direction_strength_specialist(
                            strength_train_df,
                            feature_cols,
                            y_strength_train_h,
                            valid_strength_train_h,
                            str(getattr(cfg, "upside_strength_train_filter", "major_only")),
                            cfg,
                            horizon=int(sh),
                        )
                    # v8.6.34: 별도 bear-filter specialist 제거.
                    # DOWN_STRENGTHENING도 5D/10D/20D 공통 direction-strength 모델에서 직접 추출한다.
                    bear_strength_models = {}
                except Exception as exc:
                    warnings.warn(f"multi-horizon direction-strength specialist training skipped: {exc}", RuntimeWarning)
                    strength_models_by_h = {}
                    bear_strength_models = {}

            last_retrain_k = k

        if not models:
            continue

        row_df = all_df.iloc[[pos]]
        X_now_full = row_df[feature_cols]
        out: Dict[str, object] = {"Date": date}
        strength_label_now = strength_aux_full.loc[date, "direction_strength_label"] if date in strength_aux_full.index else ""
        out["actual_direction_strength"] = str(strength_label_now)
        out["direction_strength_label_valid"] = bool(strength_valid_full.loc[date]) if date in strength_valid_full.index else False
        up_strength_probs_by_h: Dict[int, Dict[str, float]] = {}
        if bool(getattr(cfg, "use_direction_strength_specialist", True)):
            for sh in multi_strength_horizons:
                up_strength_probs_by_h[int(sh)] = predict_direction_strength_one(strength_models_by_h.get(int(sh), {}), row_df)
            bear_strength_probs = {name: 0.0 for name in DIRECTION_STRENGTH_LABELS}  # removed in v8.6.34
        else:
            up_strength_probs_by_h = {int(sh): {name: 0.0 for name in DIRECTION_STRENGTH_LABELS} for sh in multi_strength_horizons}
            bear_strength_probs = {name: 0.0 for name in DIRECTION_STRENGTH_LABELS}
        up_strength_probs = up_strength_probs_by_h.get(legacy_h, {name: 0.0 for name in DIRECTION_STRENGTH_LABELS})
        up_strength_values = {int(sh): up_strength_probs_by_h.get(int(sh), {}).get("UP_STRENGTHENING", 0.0) for sh in multi_strength_horizons}
        down_strength_values = {int(sh): up_strength_probs_by_h.get(int(sh), {}).get("DOWN_STRENGTHENING", 0.0) for sh in multi_strength_horizons}
        for sh in multi_strength_horizons:
            spec_h = strength_models_by_h.get(int(sh), {}) if isinstance(strength_models_by_h, dict) else {}
            out[f"strength_train_rows_{int(sh)}d"] = int(spec_h.get("train_rows", 0)) if isinstance(spec_h, dict) else 0
            out[f"strength_train_horizon_ratio_{int(sh)}d"] = float(spec_h.get("train_horizon_ratio", 0.0)) if isinstance(spec_h, dict) else 0.0
        up_strength_score = combine_up_strength_score_from_values(up_strength_values, cfg)
        down_strength_score = combine_down_strength_score_from_values(down_strength_values, cfg)

        prob_high_ens = 0.0
        prob_up_ens = 0.0
        prob_down_ens = 0.0
        prob_down_branch_ens: Dict[str, float] = {"price_trend": 0.0, "price_volume": 0.0, "volatility": 0.0}
        actual_primary_label = "정상"
        actual_primary_risk = "정상"
        actual_primary_direction = "중립"

        for h in cfg.horizons:
            if h not in models:
                continue
            m = models[h]
            out[f"train_rows_h{int(h)}"] = int(m.get("train_rows", 0))
            out[f"train_horizon_ratio_h{int(h)}"] = float(m.get("train_horizon_ratio", 0.0))
            out[f"train_window_multiplier_h{int(h)}"] = float(m.get("train_window_multiplier", _get_horizon_train_multiplier(cfg, int(h))))
            stage1_model = m["stage1"]
            th = m["thresholds"]
            policy = m["policy"]

            p_high = float(stage1_model.predict_proba(X_now_full)[0, 1])  # type: ignore[union-attr]

            up_model = m.get("up_model")
            if up_model is not None and bool(m.get("up_available", False)):
                p_up = float(up_model.predict_proba(row_df[direction_cols])[0, 1])  # type: ignore[union-attr]
            else:
                p_up = 0.0

            branch_probs: Dict[str, float] = {}
            down_models = m.get("down_models", {})
            down_available = m.get("down_available", {})
            for branch, cols in down_feature_sets.items():
                branch_model = down_models.get(branch) if isinstance(down_models, dict) else None
                branch_ok = bool(down_available.get(branch, False)) if isinstance(down_available, dict) else False
                if branch_model is not None and branch_ok and cols:
                    branch_probs[branch] = float(branch_model.predict_proba(row_df[cols])[0, 1])  # type: ignore[union-attr]
                else:
                    branch_probs[branch] = 0.0

            if bool(cfg.use_multi_branch_downrisk):
                p_down = (
                    branch_weights["price_trend"] * branch_probs["price_trend"]
                    + branch_weights["price_volume"] * branch_probs["price_volume"]
                    + branch_weights["volatility"] * branch_probs["volatility"]
                    + branch_weights["high_vol"] * p_high
                )
            else:
                p_down = branch_probs["price_volume"]
            p_down = float(np.clip(p_down, 0.0, 1.0))

            actual_label_h = assign_label(all_df.iloc[pos], h, th)
            actual_risk_h = "고변동" if actual_label_h != "정상" else "정상"
            actual_direction_h = assign_direction_label(all_df.iloc[pos], h, cfg)

            out[f"prob_high_vol_h{h}"] = p_high
            out[f"prob_up_h{h}"] = p_up
            out[f"prob_down_price_trend_h{h}"] = branch_probs["price_trend"]
            out[f"prob_down_price_volume_h{h}"] = branch_probs["price_volume"]
            out[f"prob_down_volatility_h{h}"] = branch_probs["volatility"]
            out[f"prob_down_h{h}"] = p_down
            out[f"prob_down_risk_h{h}"] = p_down  # 호환 컬럼: v8.6.2에서는 방향성 하락 확률로 해석
            out[f"actual_direction_h{h}"] = actual_direction_h
            out[f"actual_split_vol_h{h}"] = actual_label_h
            out[f"actual_risk_h{h}"] = actual_risk_h
            out[f"label_policy_h{h}"] = policy.name  # type: ignore[union-attr]

            prob_high_ens += hv_w.get(h, 0.0) * p_high
            prob_up_ens += dn_w.get(h, 0.0) * p_up
            prob_down_ens += dn_w.get(h, 0.0) * p_down
            for branch in prob_down_branch_ens:
                prob_down_branch_ens[branch] += dn_w.get(h, 0.0) * branch_probs[branch]

            if h == cfg.primary_horizon:
                actual_primary_label = actual_label_h
                actual_primary_risk = actual_risk_h
                actual_primary_direction = actual_direction_h

        prob_high_ens = float(np.clip(prob_high_ens, 0.0, 1.0))
        prob_up_ens = float(np.clip(prob_up_ens, 0.0, 1.0))
        prob_down_ens = float(np.clip(prob_down_ens, 0.0, 1.0))
        for branch in prob_down_branch_ens:
            prob_down_branch_ens[branch] = float(np.clip(prob_down_branch_ens[branch], 0.0, 1.0))
        prob_down_hv = float(np.clip(min(prob_high_ens, prob_down_ens), 0.0, 1.0))
        prob_up_proxy = prob_up_ens
        prob_overall_risk = compute_overall_risk_prob(prob_high_ens, prob_down_ens, cfg, prob_up=prob_up_ens)
        direction_margin = float(getattr(cfg, "direction_decision_margin", 0.05))
        direction_score = float(prob_up_ens - prob_down_ens)
        if direction_score >= direction_margin:
            pred_direction = "상승"
        elif direction_score <= -direction_margin:
            pred_direction = "하락"
        else:
            pred_direction = "중립"

        out.update({
            "actual_risk": actual_primary_risk,
            "actual_split_vol": actual_primary_label,
            "actual_direction": actual_primary_direction,
            "prob_high_vol": prob_high_ens,
            "prob_up": prob_up_ens,
            # Legacy columns use the configured direction_strength_horizon, usually 20D.
            "prob_up_strengthening": up_strength_probs.get("UP_STRENGTHENING", 0.0),
            "prob_up_weakening": 0.0,
            "prob_down_weakening": 0.0,
            "prob_up_strengthening_score": up_strength_score,
            "prob_up_strengthening_min_multi": min(up_strength_values.values()) if up_strength_values else 0.0,
            "prob_up_strengthening_max_multi": max(up_strength_values.values()) if up_strength_values else 0.0,
            "prob_up_strengthening_spread_5d_20d": float(up_strength_values.get(5, 0.0) - up_strength_values.get(20, 0.0)),
            **{f"prob_up_strengthening_{int(sh)}d": up_strength_probs_by_h.get(int(sh), {}).get("UP_STRENGTHENING", 0.0) for sh in multi_strength_horizons},
            **{f"prob_up_weakening_{int(sh)}d": 0.0 for sh in multi_strength_horizons},
            # v8.6.34: 하락 강화도 상승 강화와 동일한 horizon별 term structure로 산출한다.
            "prob_down_strengthening": down_strength_values.get(legacy_h, 0.0),
            "prob_down_strengthening_score": down_strength_score,
            "prob_down_strengthening_min_multi": min(down_strength_values.values()) if down_strength_values else 0.0,
            "prob_down_strengthening_max_multi": max(down_strength_values.values()) if down_strength_values else 0.0,
            "prob_down_strengthening_spread_5d_20d": float(down_strength_values.get(5, 0.0) - down_strength_values.get(20, 0.0)),
            **{f"prob_down_strengthening_{int(sh)}d": up_strength_probs_by_h.get(int(sh), {}).get("DOWN_STRENGTHENING", 0.0) for sh in multi_strength_horizons},
            **{f"prob_down_weakening_{int(sh)}d": 0.0 for sh in multi_strength_horizons},
            **{f"actual_direction_strength_{int(sh)}d": str(strength_full_by_h[int(sh)][2].loc[date, "direction_strength_label"]) if date in strength_full_by_h[int(sh)][2].index else "" for sh in multi_strength_horizons},
            "prob_down_price_trend": prob_down_branch_ens["price_trend"],
            "prob_down_price_volume": prob_down_branch_ens["price_volume"],
            "prob_down_volatility": prob_down_branch_ens["volatility"],
            "prob_down": prob_down_ens,
            "prob_down_risk": prob_down_ens,  # 호환 컬럼
            "prob_overall_risk": prob_overall_risk,
            "prob_normal": 1.0 - prob_high_ens,
            "prob_down_high_vol": prob_down_hv,
            "prob_up_proxy": prob_up_proxy,
            "direction_score": direction_score,
            "pred_direction": pred_direction,
            "pred_risk": "고변동" if prob_high_ens >= cfg.pred_high_vol_threshold else "정상",
            "pred_overall_risk": "위험" if prob_overall_risk >= cfg.pred_overall_risk_threshold else "정상",
            "pred_split_vol": "하락고변동" if (prob_high_ens >= cfg.pred_high_vol_threshold and prob_down_ens >= cfg.pred_down_risk_threshold) else ("상승고변동" if prob_high_ens >= cfg.pred_high_vol_threshold else "정상"),
            "stock_next_return": float(all_df.iloc[pos]["stock_next_return"]),
            "bond_next_return": float(all_df.iloc[pos]["bond_next_return"]),
            "cash_next_return": float(all_df.iloc[pos]["cash_next_return"]),
        })
        prediction_rows.append(out)

    pred_df = pd.DataFrame(prediction_rows).sort_values("Date").reset_index(drop=True)
    if pred_df.empty:
        raise ValueError("walk-forward 예측 결과가 비어 있습니다.")

    if cfg.use_prob_ewma:
        prob_cols = [
            c for c in pred_df.columns
            if c.startswith("prob_high_vol")
            or c.startswith("prob_up")
            or c.startswith("prob_down_strengthening")
            or c.startswith("prob_down_risk")
            or c.startswith("prob_down_h")
            or c.startswith("prob_down_price_trend")
            or c.startswith("prob_down_price_volume")
            or c.startswith("prob_down_volatility")
            or c.startswith("prob_bear_")
        ]
        for col in prob_cols:
            if pred_df[col].dtype.kind in "if" and not col.endswith("_raw"):
                raw_col = f"{col}_raw"
                if raw_col not in pred_df.columns:
                    pred_df[raw_col] = pred_df[col]
                pred_df[col] = pred_df[col].ewm(span=cfg.prob_ewma_span, adjust=False).mean()

        pred_df["prob_high_vol"] = pred_df["prob_high_vol"].clip(0.0, 1.0)
        pred_df["prob_up"] = pred_df["prob_up"].clip(0.0, 1.0)
        pred_df["prob_down_price_trend"] = pred_df["prob_down_price_trend"].clip(0.0, 1.0)
        pred_df["prob_down_price_volume"] = pred_df["prob_down_price_volume"].clip(0.0, 1.0)
        pred_df["prob_down_volatility"] = pred_df["prob_down_volatility"].clip(0.0, 1.0)
        if bool(cfg.use_multi_branch_downrisk):
            pred_df["prob_down"] = (
                branch_weights["price_trend"] * pred_df["prob_down_price_trend"]
                + branch_weights["price_volume"] * pred_df["prob_down_price_volume"]
                + branch_weights["volatility"] * pred_df["prob_down_volatility"]
                + branch_weights["high_vol"] * pred_df["prob_high_vol"]
            ).clip(0.0, 1.0)
        else:
            pred_df["prob_down"] = pred_df["prob_down_price_volume"].clip(0.0, 1.0)
        pred_df["prob_down_risk"] = pred_df["prob_down"]
        pred_df["prob_normal"] = 1.0 - pred_df["prob_high_vol"]
        pred_df["prob_down_high_vol"] = np.minimum(pred_df["prob_high_vol"], pred_df["prob_down"]).clip(0.0, 1.0)
        pred_df["prob_up_proxy"] = pred_df["prob_up"]
        pred_df = recompute_multi_horizon_strength_score(pred_df, cfg)
        pred_df["direction_score"] = pred_df["prob_up"] - pred_df["prob_down"]
        margin = float(getattr(cfg, "direction_decision_margin", 0.05))
        pred_df["pred_direction"] = np.where(
            pred_df["direction_score"] >= margin,
            "상승",
            np.where(pred_df["direction_score"] <= -margin, "하락", "중립"),
        )
        pred_df["prob_overall_risk"] = compute_overall_risk_prob(
            pred_df["prob_high_vol"], pred_df["prob_down"], cfg, prob_up=pred_df["prob_up"]
        )
        pred_df["pred_risk"] = np.where(pred_df["prob_high_vol"] >= cfg.pred_high_vol_threshold, "고변동", "정상")
        pred_df["pred_overall_risk"] = np.where(pred_df["prob_overall_risk"] >= cfg.pred_overall_risk_threshold, "위험", "정상")
        pred_df["pred_split_vol"] = np.where(
            pred_df["pred_risk"] == "정상",
            "정상",
            np.where(pred_df["prob_down"] >= cfg.pred_down_risk_threshold, "하락고변동", "상승고변동"),
        )

    down_weighted_imp = combine_weighted_importance(down_imp_hist_by_branch, branch_weights)
    pred_df.attrs["stage1_feature_importance_mean"] = mean_importance(stage1_imp_hist)
    pred_df.attrs["up_feature_importance_mean"] = mean_importance(up_imp_hist)
    pred_df.attrs["downrisk_feature_importance_mean"] = down_weighted_imp
    pred_df.attrs["downrisk_price_trend_feature_importance_mean"] = mean_importance(down_imp_hist_by_branch["price_trend"])
    pred_df.attrs["downrisk_price_volume_feature_importance_mean"] = mean_importance(down_imp_hist_by_branch["price_volume"])
    pred_df.attrs["downrisk_volatility_feature_importance_mean"] = mean_importance(down_imp_hist_by_branch["volatility"])
    pred_df.attrs["downrisk_branch_weights"] = branch_weights
    pred_df.attrs["downrisk_feature_sets"] = down_feature_sets
    pred_df.attrs["direction_feature_set"] = direction_cols
    pred_df.attrs["policy_usage"] = policy_usage
    return pred_df


# ============================================================
# 6. ALLOCATION / BACKTEST
# ============================================================

def _normalize_weight_tuple(stock: float, bond: float, cash: float) -> Tuple[float, float, float]:
    vals = np.asarray([stock, bond, cash], dtype=float)
    vals = np.clip(vals, 0.0, 1.0)
    total = float(vals.sum())
    if total <= 0:
        return 1.0, 0.0, 0.0
    vals = vals / total
    return float(vals[0]), float(vals[1]), float(vals[2])


def gate_config_from_cfg(cfg: Config) -> Dict[str, float]:
    return {
        "gate_normal_high_vol_threshold": cfg.gate_normal_high_vol_threshold,
        "gate_high_vol_threshold": cfg.gate_high_vol_threshold,
        "gate_riskoff_downrisk_threshold": cfg.gate_riskoff_downrisk_threshold,
        "gate_watch_downrisk_threshold": cfg.gate_watch_downrisk_threshold,
        "use_three_regime_allocation": cfg.use_three_regime_allocation,
        "use_extreme_risk_cut": cfg.use_extreme_risk_cut,
        "extreme_high_vol_threshold": cfg.extreme_high_vol_threshold,
        "extreme_downrisk_threshold": cfg.extreme_downrisk_threshold,
        "extreme_stock_weight": cfg.extreme_stock_weight,
        "extreme_bond_weight": cfg.extreme_bond_weight,
        "extreme_cash_weight": cfg.extreme_cash_weight,
        "normal_stock_weight": cfg.normal_stock_weight,
        "normal_bond_weight": cfg.normal_bond_weight,
        "normal_cash_weight": cfg.normal_cash_weight,
        "watch_stock_weight": cfg.watch_stock_weight,
        "watch_bond_weight": cfg.watch_bond_weight,
        "watch_cash_weight": cfg.watch_cash_weight,
        "high_vol_stock_weight": cfg.high_vol_stock_weight,
        "high_vol_bond_weight": cfg.high_vol_bond_weight,
        "high_vol_cash_weight": cfg.high_vol_cash_weight,
        "risk_off_stock_weight": cfg.risk_off_stock_weight,
        "risk_off_bond_weight": cfg.risk_off_bond_weight,
        "risk_off_cash_weight": cfg.risk_off_cash_weight,
        "no_trade_band": cfg.no_trade_band,
        "name": "default_v8_4_gate",
        "policy_mode": cfg.policy_mode,
    }


def classify_gate(prob_high_vol: float, prob_down_risk: float, g: Dict[str, float]) -> str:
    """
    v8.4 allocation gate.

    기본값은 3-regime 구조다.
    - NORMAL: 고변동 확률이 충분히 낮은 구간
    - WATCH: 정상은 아니지만 RISK_OFF 조건은 아닌 구간
    - RISK_OFF: 고변동 확률과 하락위험 확률이 동시에 높은 구간

    EXTREME_RISK는 RISK_OFF 내부의 추가 방어 sub-regime이다.
    HIGH_VOL은 v8.3 진단에서 표본이 작고 turnover가 높아 기본 구조에서 제거했다.
    """
    ph = float(np.clip(prob_high_vol, 0.0, 1.0))
    pdn = float(np.clip(prob_down_risk, 0.0, 1.0))

    if ph < g["gate_normal_high_vol_threshold"]:
        return "NORMAL"

    if bool(g.get("use_three_regime_allocation", True)):
        if ph >= g["gate_high_vol_threshold"] and pdn >= g["gate_riskoff_downrisk_threshold"]:
            if bool(g.get("use_extreme_risk_cut", True)):
                if ph >= g.get("extreme_high_vol_threshold", 0.75) and pdn >= g.get("extreme_downrisk_threshold", 0.65):
                    return "EXTREME_RISK"
            return "RISK_OFF"
        return "WATCH"

    # 이전 4-regime 구조를 옵션으로 유지
    if ph < g["gate_high_vol_threshold"]:
        if pdn >= g["gate_watch_downrisk_threshold"]:
            return "HIGH_VOL"
        return "WATCH"
    if pdn >= g["gate_riskoff_downrisk_threshold"]:
        return "RISK_OFF"
    return "HIGH_VOL"


def allocation_downrisk_score(prob_high_vol: float, prob_down_risk: float, cfg: Config) -> float:
    """
    v8.6.23: allocation gate에서 사용할 하락 위험 점수.

    기존 모델의 down_h10/down_h20 및 bear specialist는 통합 성능이 약했으므로,
    기본값에서는 down-risk를 gate에 거의 반영하지 않고 Stage1 high-vol을 위험 차단기로 사용한다.

    allocation_downrisk_weight:
    - 0.0: pdn = prob_high_vol  -> RISK_OFF는 사실상 high-vol threshold 중심
    - 0.2~0.4: pdn = (1-w)*prob_high_vol + w*prob_down_risk
    - 1.0: 기존처럼 down-risk 중심
    """
    ph = float(np.clip(prob_high_vol, 0.0, 1.0))
    pdn_raw = float(np.clip(prob_down_risk, 0.0, 1.0))
    w = float(np.clip(getattr(cfg, "allocation_downrisk_weight", 0.0), 0.0, 1.0))
    return float(np.clip((1.0 - w) * ph + w * pdn_raw, 0.0, 1.0))


def base_weight_for_regime(regime: str, g: Dict[str, float]) -> Tuple[float, float, float]:
    if regime == "NORMAL":
        return _normalize_weight_tuple(g["normal_stock_weight"], g["normal_bond_weight"], g["normal_cash_weight"])
    if regime == "WATCH":
        return _normalize_weight_tuple(g["watch_stock_weight"], g["watch_bond_weight"], g["watch_cash_weight"])
    if regime == "HIGH_VOL":
        return _normalize_weight_tuple(g["high_vol_stock_weight"], g["high_vol_bond_weight"], g["high_vol_cash_weight"])
    if regime == "EXTREME_RISK":
        return _normalize_weight_tuple(g["extreme_stock_weight"], g["extreme_bond_weight"], g["extreme_cash_weight"])
    return _normalize_weight_tuple(g["risk_off_stock_weight"], g["risk_off_bond_weight"], g["risk_off_cash_weight"])


def base_weight_from_vol_probability(prob_high_vol: float, cfg: Config) -> Tuple[float, float, float]:
    """
    v8.6.34 기본 포트폴리오.

    이전 NORMAL/WATCH/RISK_OFF 고정 버킷 대신, 기본 상태에서는 Stage1 high-vol 확률만으로
    주식 비중을 계단형으로 산출한다. Tier3/Full override가 없으면 이 비중이 target이 된다.
    """
    ph = float(np.clip(prob_high_vol, 0.0, 1.0))
    if ph < 0.25:
        stock = float(getattr(cfg, "vol_base_stock_lt_25", 0.78))
    elif ph < 0.35:
        stock = float(getattr(cfg, "vol_base_stock_lt_35", 0.74))
    elif ph < 0.50:
        stock = float(getattr(cfg, "vol_base_stock_lt_50", 0.68))
    elif ph < 0.65:
        stock = float(getattr(cfg, "vol_base_stock_lt_65", 0.60))
    elif ph < 0.75:
        stock = float(getattr(cfg, "vol_base_stock_lt_75", 0.52))
    elif ph < 0.86:
        stock = float(getattr(cfg, "vol_base_stock_lt_86", 0.42))
    else:
        stock = float(getattr(cfg, "vol_base_stock_ge_86", 0.30))

    stock = float(np.clip(stock, 0.0, 1.0))
    remain = max(0.0, 1.0 - stock)
    bond_ratio = float(np.clip(getattr(cfg, "vol_base_bond_ratio_of_defensive", 0.65), 0.0, 1.0))
    bond = remain * bond_ratio
    cash = remain * (1.0 - bond_ratio)
    return _normalize_weight_tuple(stock, bond, cash)


def rolling_rank_last_local(series: pd.Series, window: int) -> pd.Series:
    def _rank(x: np.ndarray) -> float:
        if len(x) == 0 or not np.isfinite(x[-1]):
            return np.nan
        return float(np.mean(x <= x[-1]))
    return series.rolling(window, min_periods=max(20, window // 4)).apply(_rank, raw=True)


def ensure_direction_strength_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    """방향·추세강도 라벨/구간 mask에 필요한 time-t helper 컬럼을 보강한다."""
    out = df.copy()
    if "daily_return" not in out.columns:
        out["daily_return"] = out["Close"].pct_change()
    if "realized_vol_20" not in out.columns:
        out["realized_vol_20"] = out["daily_return"].rolling(20).std()
    if "realized_vol_20_rank_252" not in out.columns:
        out["realized_vol_20_rank_252"] = rolling_rank_last_local(out["realized_vol_20"], 252)

    close = out["Close"]
    for w in [20, 60, 120, 200]:
        ma_col = f"ma_{w}"
        if ma_col not in out.columns:
            out[ma_col] = close.rolling(w, min_periods=max(5, w // 4)).mean()
        gap_col = f"price_ma_{w}_gap"
        if gap_col not in out.columns:
            out[gap_col] = close / out[ma_col] - 1.0
    if "return_60d" not in out.columns:
        out["return_60d"] = close.pct_change(60)
    if "return_120d" not in out.columns:
        out["return_120d"] = close.pct_change(120)
    if "ma_gap_20_60" not in out.columns:
        out["ma_gap_20_60"] = out["ma_20"] / out["ma_60"] - 1.0
    if "ma_gap_60_120" not in out.columns:
        out["ma_gap_60_120"] = out["ma_60"] / out["ma_120"] - 1.0
    if "trend_slope_60" not in out.columns:
        out["trend_slope_60"] = close.pct_change(60) / 60.0
    if "ma200_slope_60" not in out.columns:
        out["ma200_slope_60"] = out["ma_200"].pct_change(60) / 60.0
    return out


def build_strength_feature_sets(feature_cols: Sequence[str]) -> Dict[str, List[str]]:
    available = set(feature_cols)
    def keep(cols: Sequence[str]) -> List[str]:
        return [c for c in cols if c in available]

    trend_core = keep([
        "return_10d", "return_20d", "return_60d", "return_120d",
        "price_ma_20_gap", "price_ma_60_gap", "price_ma_120_gap", "price_ma_200_gap",
        "ma_gap_5_20", "ma_gap_20_60", "ma_gap_60_120", "ma_gap_50_200",
        "trend_slope_20", "trend_slope_60", "ma200_slope_60",
        "positive_return_ratio_20", "positive_return_ratio_60",
        "trend_consistency_20", "trend_consistency_60",
        "price_position_20", "price_position_60", "close_to_20d_high", "close_to_60d_high",
        "large_up_day_ratio_20", "large_down_day_ratio_20",
    ])
    down_core = keep([
        "return_5d", "return_10d", "return_20d", "return_60d", "return_120d",
        "drawdown_20", "drawdown_60", "drawdown_120",
        "price_position_20", "price_position_60",
        "close_to_20d_high", "close_to_60d_high",
        "large_down_day_ratio_20", "lower_high_20", "bearish_ma_stack",
        "positive_return_ratio_20", "positive_return_ratio_60",
        "trend_consistency_20", "trend_consistency_60",
        "volume_ratio_20", "volume_zscore_20", "down_volume_ratio_20", "high_volume_down_ratio_20", "volume_shock_20",
    ])
    vol_risk_core = keep([
        "true_range_pct", "atr_pct_14", "atr_pct_20", "atr_pct_60", "atr_rank_252",
        "atr_ratio_14_60", "atr_ratio_20_60", "atr_accel_5",
        "realized_vol_20", "realized_vol_60", "ewma_vol_20", "ewma_vol_60",
        "downside_vol_5", "downside_vol_10", "downside_vol_20", "downside_vol_60",
        "semi_vol_5", "semi_vol_10", "semi_vol_20",
        "ulcer_index_20", "ulcer_index_60", "ulcer_rank_252",
        "bb_width_20", "bb_width_rank_252", "vol_of_vol_20",
        "drawdown_20", "drawdown_60", "drawdown_120", "large_down_day_ratio_20",
    ])
    strength_core = keep([
        "return_20d", "return_60d", "return_120d",
        "price_ma_20_gap", "price_ma_60_gap", "price_ma_120_gap", "price_ma_200_gap",
        "ma_gap_20_60", "ma_gap_60_120", "ma_gap_50_200",
        "trend_slope_20", "trend_slope_60", "ma200_slope_60",
        "positive_return_ratio_60", "trend_consistency_60",
        "price_position_60", "close_to_60d_high",
        "volume_ratio_20", "volume_zscore_20", "volume_shock_rank_252",
        "atr_rank_252", "realized_vol_20", "ewma_vol_20", "ulcer_index_20",
    ])
    compact_mixed = keep([
        "return_20d", "return_60d", "return_120d",
        "price_ma_60_gap", "price_ma_120_gap", "price_ma_200_gap",
        "ma_gap_20_60", "ma_gap_60_120", "ma_gap_50_200",
        "trend_slope_60", "ma200_slope_60", "positive_return_ratio_60",
        "drawdown_20", "drawdown_60", "drawdown_120", "price_position_60", "close_to_60d_high",
        "volume_ratio_20", "volume_zscore_20", "down_volume_ratio_20", "volume_shock_rank_252",
        "atr_pct_14", "atr_pct_20", "atr_rank_252", "ewma_vol_20", "semi_vol_20", "ulcer_index_20",
    ])
    # v8.6.23: 20D single-label specialist 전용 축소 피처셋.
    # v8.6.20에서 낮은 중요도였던 5D volume/z-score/event dummy, 과도한 단기 차분을 제거한다.
    horizon_5_10_20_pruned = keep([
        # medium trend / momentum: 20D 라벨에 가장 직접적인 정보
        "return_5d", "return_10d", "return_20d", "return_60d", "return_120d",
        "return_5d_minus_20d", "return_10d_minus_20d",
        "price_ma_20_gap", "price_ma_60_gap", "price_ma_120_gap", "price_ma_200_gap",
        "ma_gap_5_20", "ma_gap_20_60", "ma_gap_60_120", "ma_gap_50_200",
        "trend_slope_5", "trend_slope_10", "trend_slope_20", "trend_slope_60", "ma200_slope_60",
        "positive_return_ratio_10", "positive_return_ratio_20", "positive_return_ratio_60",
        "large_up_day_ratio_10", "large_up_day_ratio_20",
        "large_down_day_ratio_10", "large_down_day_ratio_20",
        "drawdown_10", "drawdown_20", "drawdown_60", "drawdown_120",
        "price_position_10", "price_position_20", "price_position_60",
        "price_position_5_minus_20", "price_position_10_minus_20",
        "close_to_10d_high", "close_to_20d_high", "close_to_60d_high",
        "trend_consistency_20", "trend_consistency_60", "bearish_ma_stack",

        # volume: v8.6.20에서 비교적 살아남은 20D selling pressure 중심
        "volume_ratio_20", "volume_zscore_20",
        "down_volume_ratio_20", "high_volume_down_ratio_20", "volume_shock_rank_252",

        # selected risk context: 상승 강화 신호의 false positive를 줄이기 위한 최소 변동성 피처
        "atr_pct_10", "atr_pct_14", "atr_pct_20", "atr_rank_252",
        "realized_vol_20", "ewma_vol_20", "semi_vol_20", "ulcer_index_20", "bb_width_20",
    ])
    return {
        "pruned_all": list(feature_cols),
        "trend_core": trend_core,
        "down_core": down_core,
        "vol_risk_core": vol_risk_core,
        "strength_core": strength_core,
        "compact_mixed": compact_mixed,
        "horizon_5_10_20_pruned": horizon_5_10_20_pruned,
        "horizon_5_10_20": horizon_5_10_20_pruned,
    }


def strength_trend_score_components(df: pd.DataFrame) -> pd.DataFrame:
    comps = pd.DataFrame(index=df.index)
    comps["ret60_pos"] = (df.get("return_60d", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    comps["ret120_pos"] = (df.get("return_120d", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    comps["ma60_gap_pos"] = (df.get("price_ma_60_gap", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    comps["ma120_gap_pos"] = (df.get("price_ma_120_gap", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    comps["ma20_60_pos"] = (df.get("ma_gap_20_60", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    comps["slope60_pos"] = (df.get("trend_slope_60", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    return comps


def strength_current_trend_score(df: pd.DataFrame) -> pd.Series:
    return strength_trend_score_components(df).sum(axis=1)


def strength_current_trend_continuous(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ["return_60d", "return_120d", "price_ma_60_gap", "price_ma_120_gap", "ma_gap_20_60", "trend_slope_60"]:
        if col in df.columns:
            s = df[col].replace([np.inf, -np.inf], np.nan)
            scale = s.rolling(252, min_periods=60).std().replace(0, np.nan)
            parts.append((s / scale).clip(-3, 3))
    if not parts:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1)


def expected_horizon_vol_local(df: pd.DataFrame, horizon: int) -> pd.Series:
    vol = df.get("realized_vol_20")
    if vol is None:
        vol = df["Close"].pct_change().rolling(20).std()
    return vol * math.sqrt(max(horizon, 1))


def build_direction_strength_labels(df: pd.DataFrame, cfg: Config, horizon: Optional[int] = None) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """No-sideway 방향·추세강도 라벨.

    v8.6.23 no-sideway 버전에서는 SIDEWAYS / UP_WEAKENING / DOWN_WEAKENING을 별도 타깃으로 학습하지 않는다.
    각 horizon의 유효 샘플은 아래 3개 중 하나로만 압축된다.
    - UP_STRENGTHENING
    - DOWN_STRENGTHENING
    - NO_STRENGTH_SIGNAL

    NO_STRENGTH_SIGNAL은 classifier의 음성/기준 클래스이며 allocation trigger로 직접 쓰지 않는다.
    trend_delta는 label 생성에만 사용하고 feature에는 넣지 않는다.
    """
    h = int(horizon if horizon is not None else getattr(cfg, "direction_strength_horizon", 20))
    ret_col = f"future_return_{h}d"
    if ret_col not in df.columns:
        raise KeyError(f"missing {ret_col}. cfg.horizons에 direction_strength_horizon을 포함해야 합니다.")
    r_h = df[ret_col]
    vol_h = expected_horizon_vol_local(df, h).replace(0, np.nan)
    ret_eps = float(getattr(cfg, "direction_strength_ret_eps_k", 0.20)) * vol_h

    trend_score_t = strength_current_trend_score(df)
    trend_score_f = trend_score_t.shift(-h)
    score_delta = trend_score_f - trend_score_t

    trend_cont_t = strength_current_trend_continuous(df)
    trend_cont_f = trend_cont_t.shift(-h)
    slope_delta = trend_cont_f - trend_cont_t
    method = str(getattr(cfg, "direction_strength_method", "score_delta"))
    if method == "score_delta":
        trend_delta = score_delta
    elif method == "slope_delta":
        trend_delta = slope_delta
    elif method == "hybrid_delta":
        trend_delta = 0.65 * score_delta + 0.35 * slope_delta
    else:
        raise ValueError(f"unknown direction_strength_method: {method}")

    strength_eps = float(getattr(cfg, "direction_strength_eps", 0.0))
    valid = r_h.notna() & vol_h.notna() & trend_delta.notna()
    up_strength = valid & (r_h >= ret_eps) & (trend_delta > strength_eps)
    down_strength = valid & (r_h <= -ret_eps) & (trend_delta < -strength_eps)

    y_label = pd.Series("NO_STRENGTH_SIGNAL", index=df.index, dtype=object)
    y_label.loc[up_strength] = "UP_STRENGTHENING"
    y_label.loc[down_strength] = "DOWN_STRENGTHENING"

    aux = pd.DataFrame({
        "direction_strength_label": y_label,
        "direction_strength_ret_eps": ret_eps,
        "direction_strength_trend_delta": trend_delta,
        "direction_strength_score_delta": score_delta,
        "direction_strength_is_up_strengthening": up_strength.astype(bool),
        "direction_strength_is_down_strengthening": down_strength.astype(bool),
    }, index=df.index)
    y_id = y_label.map(DIRECTION_STRENGTH_LABEL_TO_ID).astype(float)
    return y_id, valid.astype(bool), aux


def mask_by_strength_filter(df: pd.DataFrame, mode: str) -> pd.Series:
    idx = df.index
    if mode == "all":
        return pd.Series(True, index=idx)
    rank = df.get("realized_vol_20_rank_252")
    if rank is None:
        rank = rolling_rank_last_local(df["realized_vol_20"], 252)
    trend_score = strength_current_trend_score(df)
    dd60 = df.get("drawdown_60", pd.Series(np.nan, index=idx))
    dd120 = df.get("drawdown_120", pd.Series(np.nan, index=idx))
    ret60 = df.get("return_60d", pd.Series(np.nan, index=idx))
    price_ma60_gap = df.get("price_ma_60_gap", pd.Series(np.nan, index=idx))
    slope60 = df.get("trend_slope_60", pd.Series(np.nan, index=idx))

    low_vol = rank <= 0.65
    non_extreme = rank <= 0.80
    high_vol = rank >= 0.70
    extreme_vol = rank >= 0.85
    bull_trend = (trend_score >= 4) & (ret60 > 0) & (price_ma60_gap > 0) & (rank <= 0.75)
    bear_stress = ((trend_score <= 2) & ((dd60 <= -0.08) | (dd120 <= -0.12))) | extreme_vol
    recovery = ((dd60 <= -0.05) | (dd120 <= -0.10)) & (ret60 > 0) & (slope60 > 0) & (rank <= 0.80)
    major_only = bull_trend | bear_stress | recovery | high_vol
    masks = {
        "low_vol_only": low_vol,
        "non_extreme_vol": non_extreme,
        "high_vol": high_vol,
        "extreme_vol": extreme_vol,
        "bull_trend": bull_trend,
        "bear_stress": bear_stress,
        "recovery": recovery,
        "major_only": major_only,
    }
    if mode not in masks:
        raise ValueError(f"unknown strength train filter: {mode}")
    return masks[mode].fillna(False).astype(bool)


def make_xgb_direction_strength_model(cfg: Config, n_classes: int) -> Pipeline:
    clf = XGBClassifier(
        n_estimators=int(getattr(cfg, "direction_strength_n_estimators", 160)),
        learning_rate=float(getattr(cfg, "direction_strength_learning_rate", 0.025)),
        max_depth=int(getattr(cfg, "direction_strength_max_depth", 2)),
        min_child_weight=float(getattr(cfg, "direction_strength_min_child_weight", 8.0)),
        subsample=float(getattr(cfg, "direction_strength_subsample", 0.85)),
        colsample_bytree=float(getattr(cfg, "direction_strength_colsample_bytree", 0.80)),
        reg_lambda=float(getattr(cfg, "direction_strength_reg_lambda", 10.0)),
        reg_alpha=float(getattr(cfg, "direction_strength_reg_alpha", 0.2)),
        objective="multi:softprob",
        num_class=int(n_classes),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=int(getattr(cfg, "random_state", 42)),
        n_jobs=int(getattr(cfg, "n_jobs", -1)),
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])


def _multiclass_sample_weights(y_local: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_local.astype(int))
    total = float(len(y_local))
    k = max(1, len(counts))
    weights = {i: total / (k * c) for i, c in enumerate(counts) if c > 0}
    return np.asarray([weights.get(int(v), 1.0) for v in y_local], dtype=float)


def fit_direction_strength_specialist(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    y_strength: pd.Series,
    valid_strength: pd.Series,
    train_filter: str,
    cfg: Config,
    horizon: Optional[int] = None,
) -> Dict[str, object]:
    feature_sets = build_strength_feature_sets(feature_cols)
    fs_name = str(getattr(cfg, "direction_strength_feature_set", "compact_mixed"))
    cols = feature_sets.get(fs_name, feature_sets.get("compact_mixed", feature_cols))
    cols = [c for c in cols if c in train_df.columns]
    if not cols:
        return {"model": None, "cols": [], "available": False, "train_rows": 0, "train_horizon": int(horizon or 0), "train_horizon_ratio": 0.0, "classes": []}
    mask = valid_strength & mask_by_strength_filter(train_df, train_filter)
    idx = train_df.index[mask.values]
    h_for_window = int(horizon if horizon is not None else getattr(cfg, "direction_strength_horizon", 20))
    max_rows = horizon_train_rows(cfg, h_for_window, for_direction_strength=True)
    if max_rows is not None:
        idx = idx[-int(max_rows):]
    min_rows = int(getattr(cfg, "direction_strength_min_train_rows", 300))
    if len(idx) < min_rows:
        return {"model": None, "cols": cols, "available": False, "train_rows": int(len(idx)), "train_horizon": int(h_for_window), "train_horizon_ratio": float(len(idx)) / max(float(h_for_window), 1.0), "classes": []}
    y_global = y_strength.loc[idx].astype(int).values
    orig_classes = np.array(sorted(np.unique(y_global).astype(int)))
    if len(orig_classes) < 2:
        return {"model": None, "cols": cols, "available": False, "train_rows": int(len(idx)), "train_horizon": int(h_for_window), "train_horizon_ratio": float(len(idx)) / max(float(h_for_window), 1.0), "classes": orig_classes.tolist()}
    local_map = {int(c): i for i, c in enumerate(orig_classes)}
    y_local = np.asarray([local_map[int(v)] for v in y_global], dtype=int)
    model = make_xgb_direction_strength_model(cfg, n_classes=len(orig_classes))
    sw = _multiclass_sample_weights(y_local)
    model.fit(train_df.loc[idx, cols], y_local, clf__sample_weight=sw)
    model.named_steps["clf"].original_label_ids_ = orig_classes
    return {"model": model, "cols": cols, "available": True, "train_rows": int(len(idx)), "train_horizon": int(h_for_window), "train_horizon_ratio": float(len(idx)) / max(float(h_for_window), 1.0), "classes": orig_classes.tolist()}


def align_direction_strength_proba(model: Pipeline, proba: np.ndarray) -> np.ndarray:
    clf = model.named_steps.get("clf")
    classes = getattr(clf, "original_label_ids_", getattr(clf, "classes_", np.arange(proba.shape[1])))
    out = np.zeros((proba.shape[0], len(DIRECTION_STRENGTH_LABELS)), dtype=float)
    for j, cls in enumerate(classes):
        if int(cls) in DIRECTION_STRENGTH_ID_TO_LABEL and j < proba.shape[1]:
            out[:, int(cls)] = proba[:, j]
    row_sum = out.sum(axis=1)
    missing = row_sum <= 0
    out[missing, :] = 1.0 / len(DIRECTION_STRENGTH_LABELS)
    out[~missing, :] = out[~missing, :] / np.maximum(row_sum[~missing, None], 1e-12)
    return out


def predict_direction_strength_one(spec: Dict[str, object], row_df: pd.DataFrame) -> Dict[str, float]:
    out = {name: 0.0 for name in DIRECTION_STRENGTH_LABELS}
    if not spec or not bool(spec.get("available", False)):
        return out
    model = spec.get("model")
    cols = spec.get("cols", [])
    if model is None or not cols:
        return out
    p = align_direction_strength_proba(model, model.predict_proba(row_df[list(cols)]))[0]  # type: ignore[union-attr]
    return {name: float(np.clip(p[i], 0.0, 1.0)) for i, name in enumerate(DIRECTION_STRENGTH_LABELS)}


def _row_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        val = row.get(col, default)
        if pd.isna(val):
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def compute_mid_trend_score(row: pd.Series) -> Tuple[int, str]:
    """
    중기 추세 필터.
    주의: 이 값은 독립 예측 모델이 아니라 policy overlay의 조건으로만 사용한다.
    """
    checks = [
        _row_float(row, "return_60d") > 0.0,
        _row_float(row, "return_120d") > 0.0,
        _row_float(row, "price_ma_60_gap") > 0.0,
        _row_float(row, "price_ma_120_gap") > 0.0,
        _row_float(row, "ma_gap_20_60") > 0.0,
        _row_float(row, "trend_slope_60") > 0.0,
    ]
    score = int(sum(bool(x) for x in checks))
    if score >= 4:
        state = "BULL"
    elif score <= 2:
        state = "BEAR"
    else:
        state = "NEUTRAL"
    return score, state


def _redistribute_after_stock_change(
    new_stock: float,
    old_w: Tuple[float, float, float],
    cash_ratio: Optional[float] = None,
) -> Tuple[float, float, float]:
    """주식 비중 변경 후 남은 방어자산을 채권/현금에 배분한다."""
    old_stock, old_bond, old_cash = old_w
    new_stock = float(np.clip(new_stock, 0.0, 1.0))
    remain = max(0.0, 1.0 - new_stock)
    defensive_total = old_bond + old_cash
    if cash_ratio is None:
        if defensive_total <= 0:
            bond_ratio = 0.65
            cash_ratio = 0.35
        else:
            cash_ratio = float(np.clip(old_cash / defensive_total, 0.0, 1.0))
            bond_ratio = 1.0 - cash_ratio
    else:
        cash_ratio = float(np.clip(cash_ratio, 0.0, 1.0))
        bond_ratio = 1.0 - cash_ratio
    return _normalize_weight_tuple(new_stock, remain * bond_ratio, remain * cash_ratio)


def apply_return_seeking_policy(
    base_w: Tuple[float, float, float],
    regime: str,
    row: pd.Series,
    cfg: Config,
) -> Tuple[Tuple[float, float, float], Dict[str, object]]:
    """
    수익률 개선형.
    v8.6.2의 bucket 구조를 보존하되, NORMAL + 저위험 + 상승 추세에서만 92% 초과를 허용한다.
    """
    ph = _row_float(row, "prob_high_vol")
    direction_score = _row_float(row, "direction_score")
    trend_score, trend_state = compute_mid_trend_score(row)
    stock = base_w[0]
    overlay = 0.0

    if regime == "NORMAL" and trend_state == "BULL":
        if ph < 0.35:
            overlay += cfg.return_bonus_1
        if ph < 0.25 and trend_score >= 5:
            overlay += cfg.return_bonus_2
        if ph < 0.18 and trend_score >= 5 and direction_score > 0.08:
            overlay += cfg.return_bonus_3
        stock = min(1.0, stock + overlay)

    w = _redistribute_after_stock_change(stock, base_w)
    meta = {
        "policy_overlay": float(w[0] - base_w[0]),
        "mid_trend_score": trend_score,
        "mid_trend_state": trend_state,
        "policy_note": "NORMAL_low_risk_bull_bonus_only",
    }
    return w, meta


def apply_defensive_risk_policy(
    base_w: Tuple[float, float, float],
    regime: str,
    row: pd.Series,
    cfg: Config,
) -> Tuple[Tuple[float, float, float], Dict[str, object]]:
    """
    방어력 강화형.
    Stage1 고변동 확률을 핵심으로 사용하고, down_volatility는 확인용으로만 쓴다.
    방향성 확률은 직접 사용하지 않는다.
    """
    ph = _row_float(row, "prob_high_vol")
    pdv = _row_float(row, "prob_down_volatility", _row_float(row, "prob_down"))
    trend_score, trend_state = compute_mid_trend_score(row)
    defensive_risk = float(np.clip(0.80 * ph + 0.20 * pdv, 0.0, 1.0))

    stock = base_w[0]
    # 위험 점수가 0.35를 넘은 구간에서만 완만하게 추가 방어.
    extra_cut = max(0.0, defensive_risk - 0.35) * 0.20
    if regime in {"WATCH", "HIGH_VOL"}:
        extra_cut += 0.015
    if trend_state == "BEAR" and ph >= 0.45:
        extra_cut += 0.015
    extra_cut = float(np.clip(extra_cut, 0.0, cfg.defensive_max_extra_stock_cut))
    stock = max(0.35, stock - extra_cut)

    # 위험이 높을수록 방어자산 내 현금 비율 확대.
    cash_ratio = float(np.clip(0.25 + 0.45 * defensive_risk, 0.25, 0.62))
    w = _redistribute_after_stock_change(stock, base_w, cash_ratio=cash_ratio)
    meta = {
        "policy_overlay": float(w[0] - base_w[0]),
        "mid_trend_score": trend_score,
        "mid_trend_state": trend_state,
        "defensive_risk_score": defensive_risk,
        "policy_note": "stage1_highvol_centered_defensive_overlay",
    }
    return w, meta


def apply_aggressive_dynamic_policy(
    base_w: Tuple[float, float, float],
    regime: str,
    row: pd.Series,
    cfg: Config,
) -> Tuple[Tuple[float, float, float], Dict[str, object]]:
    """
    공격적 동적배분형.
    평상시는 Buy & Hold에 가깝게 두고, 고변동/위험 구간에서만 방어한다.
    """
    ph = _row_float(row, "prob_high_vol")
    pdn = _row_float(row, "prob_down", _row_float(row, "prob_down_risk"))
    trend_score, trend_state = compute_mid_trend_score(row)
    stock = base_w[0]

    if ph < 0.45 and trend_state != "BEAR":
        stock = max(stock, cfg.aggressive_low_risk_stock_weight)
    elif ph < 0.55 and trend_score >= 3:
        stock = max(stock, 0.96)
    elif regime == "WATCH" and ph < 0.60 and trend_state != "BEAR":
        stock = max(stock, cfg.aggressive_watch_stock_weight)

    # 위험이 매우 높으면 기존 RISK_OFF/EXTREME 방어를 훼손하지 않는다.
    if ph >= 0.70 and pdn >= 0.52:
        stock = min(stock, base_w[0])

    w = _redistribute_after_stock_change(stock, base_w)
    meta = {
        "policy_overlay": float(w[0] - base_w[0]),
        "mid_trend_score": trend_score,
        "mid_trend_state": trend_state,
        "policy_note": "buy_and_hold_like_when_low_risk_crash_brake",
    }
    return w, meta


def apply_direction_strength_specialist_policy(
    base_w: Tuple[float, float, float],
    regime: str,
    row: pd.Series,
    cfg: Config,
) -> Tuple[Tuple[float, float, float], Dict[str, object]]:
    """
    v8.6.34 Volatility-Base Strong-Override Upside Strength Trigger.

    핵심 해석:
    - 기본 상태는 apply_allocation에서 prob_high_vol 기반 base_w로 이미 산출된다.
    - weak 확률(prob_up/prob_down/prob_bear_down_strengthening)은 allocation에서 제거한다.
    - Tier 2는 10D+20D 상승강화 확인 기반으로 재활성화한다.
    - Tier 1은 약한 보정만 허용한다.
    - Tier 3 / Full 조건에서는 이전 실행 비중, no-trade band, 리밸런싱 주기에 묶이지 않도록 force_rebalance를 반환한다.
    - 5D는 학습/진단에는 남기지만 allocation trigger에서는 제외한다.
    """
    ph = _row_float(row, "prob_high_vol")
    # v8.6.34: weak 확률(prob_up/prob_down/prob_bear_down_strengthening)은 allocation에서 제거한다.
    # 방어/기본 비중은 prob_high_vol만 사용한다.
    pdn_alloc = ph

    p5 = _row_float(row, "prob_up_strengthening_5d", 0.0)
    p10 = _row_float(row, "prob_up_strengthening_10d", 0.0)
    p20 = _row_float(row, "prob_up_strengthening_20d", _row_float(row, "prob_up_strengthening", 0.0))
    pus_score = _row_float(row, "prob_up_strengthening_score", combine_up_strength_score_from_values({5: p5, 10: p10, 20: p20}, cfg))
    trend_score, trend_state = compute_mid_trend_score(row)

    base_stock = float(base_w[0])
    stock = base_stock
    cut = 0.0

    t5 = float(getattr(cfg, "up_strength_pred_threshold_5d", 0.99))
    t10 = float(getattr(cfg, "up_strength_pred_threshold_10d", 0.27))
    t20 = float(getattr(cfg, "up_strength_pred_threshold_20d", 0.25))

    low_vol_1 = float(getattr(cfg, "up_strength_low_vol_threshold_1", 0.82))
    low_vol_3 = float(getattr(cfg, "up_strength_low_vol_threshold_3", 0.68))
    # bear-down-strengthening block 제거: weak 확률로 판정하지 않는다.

    pred5_raw = bool(p5 >= t5)
    pred5 = False if bool(getattr(cfg, "up_strength_disable_5d_trigger", True)) else pred5_raw
    pred10 = bool(p10 >= t10)
    pred20 = bool(p20 >= t20)

    pattern_bits = []
    if pred5:
        pattern_bits.append("5D")
    if pred10:
        pattern_bits.append("10D")
    if pred20:
        pattern_bits.append("20D")
    pred_pattern = "+".join(pattern_bits) if pattern_bits else "NONE"
    pred_count = int(pred5) + int(pred10) + int(pred20)

    target_stock = base_stock
    offensive_tier = 0
    offensive_active = False
    tier1_signal = False
    tier2_signal = False
    tier3_signal = False
    full_stock_signal = False
    strong_all3 = False
    force_rebalance = False
    five_day_only = bool(pred5 and not pred10 and not pred20)

    sm_p5 = float(getattr(cfg, "short_mid_p5_threshold", 0.32))
    sm_p10 = float(getattr(cfg, "short_mid_p10_threshold", 0.34))
    sm_p20 = float(getattr(cfg, "short_mid_p20_threshold", 0.34))
    sm_hv = float(getattr(cfg, "short_mid_high_vol_threshold", 0.72))
    sm_hv_strong = float(getattr(cfg, "short_mid_strong_high_vol_threshold", 0.68))
    sm_hv_loose = float(getattr(cfg, "short_mid_loose_high_vol_threshold", 0.76))
    sm_score_threshold = float(getattr(cfg, "short_mid_score_threshold", 0.38))
    sm_score_ok = True if not bool(getattr(cfg, "short_mid_use_score_filter", False)) else bool(pus_score >= sm_score_threshold)

    short_mid_confirm = bool(sm_score_ok and p5 >= sm_p5 and p10 >= sm_p10 and ph < sm_hv)
    short_mid_strong_confirm = bool(sm_score_ok and p5 >= sm_p5 and p10 >= sm_p10 and ph < sm_hv_strong)
    short_mid_loose_confirm = bool(sm_score_ok and p5 >= sm_p5 and p10 >= sm_p10 and ph < sm_hv_loose)
    short_mid_all3_confirm = bool(sm_score_ok and p5 >= sm_p5 and p10 >= sm_p10 and p20 >= sm_p20 and ph < sm_hv)
    short_mid_mode = str(getattr(cfg, "short_mid_confirm_mode", "base_upgrade")).lower()
    short_mid_action_signal = str(getattr(cfg, "short_mid_action_signal", "confirm")).lower()
    short_mid_action_confirm = {
        "confirm": short_mid_confirm,
        "strong": short_mid_strong_confirm,
        "loose": short_mid_loose_confirm,
        "all3": short_mid_all3_confirm,
    }.get(short_mid_action_signal, short_mid_confirm)
    short_mid_policy_action = "diagnostic_only"

    # v8.6.34: 5D/10D/20D 단독 및 모든 조합을 별도 신호로 계산한다.
    combo_mode = str(getattr(cfg, "strength_combo_policy_mode", "max_weight")).lower()
    combo_enabled = bool(getattr(cfg, "strength_combo_policy_enabled", True)) and combo_mode not in {"off", "none", "false"}
    combo_hv_threshold = float(getattr(cfg, "strength_combo_high_vol_threshold", sm_hv))
    combo_hv_ok = True if not bool(getattr(cfg, "strength_combo_use_high_vol_filter", True)) else bool(ph < combo_hv_threshold)
    combo_score_threshold = float(getattr(cfg, "strength_combo_score_threshold", sm_score_threshold))
    combo_score_ok = True if not bool(getattr(cfg, "strength_combo_use_score_filter", False)) else bool(pus_score >= combo_score_threshold)
    combo_common_ok = bool(combo_enabled and combo_hv_ok and combo_score_ok)

    combo_5d_signal = bool(combo_common_ok and p5 >= sm_p5)
    combo_10d_signal = bool(combo_common_ok and p10 >= sm_p10)
    combo_20d_signal = bool(combo_common_ok and p20 >= sm_p20)
    combo_5d_10d_signal = bool(combo_5d_signal and combo_10d_signal)
    combo_5d_20d_signal = bool(combo_5d_signal and combo_20d_signal)
    combo_10d_20d_signal = bool(combo_10d_signal and combo_20d_signal)
    combo_all3_signal = bool(combo_5d_signal and combo_10d_signal and combo_20d_signal)

    combo_candidates: List[Tuple[str, bool, float, int]] = [
        ("5D", combo_5d_signal, float(getattr(cfg, "strength_combo_single_5d_stock_weight", 0.80)), 1),
        ("10D", combo_10d_signal, float(getattr(cfg, "strength_combo_single_10d_stock_weight", 0.82)), 1),
        ("20D", combo_20d_signal, float(getattr(cfg, "strength_combo_single_20d_stock_weight", 0.82)), 1),
        ("5D+10D", combo_5d_10d_signal, float(getattr(cfg, "strength_combo_pair_5d_10d_stock_weight", 0.84)), 2),
        ("5D+20D", combo_5d_20d_signal, float(getattr(cfg, "strength_combo_pair_5d_20d_stock_weight", 0.86)), 2),
        ("10D+20D", combo_10d_20d_signal, float(getattr(cfg, "strength_combo_pair_10d_20d_stock_weight", 0.88)), 2),
        ("5D+10D+20D", combo_all3_signal, float(getattr(cfg, "strength_combo_all3_stock_weight", 0.96)), 3),
    ]
    active_combo_candidates = [(name, weight, tier) for name, ok, weight, tier in combo_candidates if ok]
    if active_combo_candidates:
        # 동일 비중이면 더 높은 tier/더 많은 horizon 조합을 우선한다.
        best_combo_name, best_combo_stock_weight, best_combo_tier = max(
            active_combo_candidates,
            key=lambda x: (float(x[1]), int(x[2]), len(str(x[0]))),
        )
    else:
        best_combo_name, best_combo_stock_weight, best_combo_tier = "NONE", 0.0, 0
    combo_policy_action = "diagnostic_only"

    # v8.6.34: 강한 상승 override는 regime bucket보다 실제 high-vol cap을 우선한다.
    # ph cap과 bear block을 통과하지 못하면 공격 전환하지 않는다.
    if True:
        tier1_signal = bool(
            pus_score >= float(getattr(cfg, "up_strength_bonus_threshold_1", 0.30))
            and p20 >= 0.30
            and ph < low_vol_1
        )
        if not bool(getattr(cfg, "disable_tier2_signal", True)):
            tier2_signal = bool(
                pus_score >= float(getattr(cfg, "up_strength_bonus_threshold_2", 0.38))
                and p10 >= float(getattr(cfg, "up_strength_confirm_10d_threshold_2", 0.32))
                and p20 >= float(getattr(cfg, "up_strength_confirm_20d_threshold_2", 0.34))
                and ph < float(getattr(cfg, "up_strength_low_vol_threshold_2", 0.72))
            )
        original_tier2_signal = bool(tier2_signal)
        # 기존 Tier2는 기본 비활성화하지만, ShortMid를 Tier2 실험으로 명시하면 별도 적용한다.
        if short_mid_mode == "tier2_replace":
            tier2_signal = bool(short_mid_action_confirm)
            short_mid_policy_action = f"tier2_replace_{short_mid_action_signal}" if short_mid_action_confirm else "tier2_replace_off"
        elif short_mid_mode == "tier2_add":
            if short_mid_action_confirm and not tier2_signal:
                short_mid_policy_action = f"tier2_add_{short_mid_action_signal}"
            tier2_signal = bool(tier2_signal or short_mid_action_confirm)
        tier3_signal = bool(
            pus_score >= float(getattr(cfg, "up_strength_bonus_threshold_3", 0.45))
            and p20 >= float(getattr(cfg, "up_strength_confirm_20d_threshold_3", 0.38))
            and ph < low_vol_3
        )
        full_stock_signal = bool(
            pus_score >= float(getattr(cfg, "up_strength_full_stock_score_threshold", 0.50))
            and p10 >= float(getattr(cfg, "up_strength_full_stock_10d_threshold", 0.38))
            and p20 >= float(getattr(cfg, "up_strength_full_stock_20d_threshold", 0.42))
            and ph < float(getattr(cfg, "up_strength_full_stock_high_vol_threshold", 0.58))
        )

        # 약한 Tier 1: 기본 변동성 비중보다 너무 낮을 때만 80% 수준으로 보정한다.
        if tier1_signal:
            target_stock = max(target_stock, float(getattr(cfg, "up_strength_single_20d_stock_weight", 0.80)))
            offensive_tier = max(offensive_tier, 1)

        # v8.6.34 ShortMidConfirm optional allocation actions.
        # diagnostic 모드에서는 아래 두 동작이 작동하지 않는다.
        if short_mid_mode in {"base_upgrade", "base_tier1_upgrade"} and short_mid_action_confirm and not (tier1_signal or tier2_signal or tier3_signal or full_stock_signal):
            target_stock = max(target_stock, float(getattr(cfg, "short_mid_base_upgrade_stock_weight", 0.82)))
            offensive_tier = max(offensive_tier, 1)
            short_mid_policy_action = f"base_upgrade_{short_mid_action_signal}"

        if short_mid_mode in {"tier1_upgrade", "base_tier1_upgrade"} and tier1_signal and short_mid_action_confirm and not (tier2_signal or tier3_signal or full_stock_signal):
            target_stock = max(target_stock, float(getattr(cfg, "short_mid_tier1_upgrade_stock_weight", 0.84)))
            offensive_tier = max(offensive_tier, 1)
            short_mid_policy_action = f"tier1_upgrade_{short_mid_action_signal}"

        # v8.6.34 Strength Combo Ladder: 5D/10D/20D 단독 및 조합을 모두 사용한다.
        # legacy Tier2와 별개로, 만족한 combo 중 가장 높은 목표 비중을 적용한다.
        if combo_enabled and combo_mode == "max_weight" and active_combo_candidates:
            target_stock = max(target_stock, float(best_combo_stock_weight))
            offensive_tier = max(offensive_tier, int(best_combo_tier))
            combo_policy_action = f"combo_{best_combo_name.replace('+', '_')}"
            if best_combo_tier >= 3 and bool(getattr(cfg, "strength_combo_force_all3_rebalance", True)):
                force_rebalance = True

        # Tier 2: 10D+20D 확인 기반 중간 공격 구간. 88% 제한.
        if tier2_signal:
            tier2_weight = float(getattr(cfg, "up_strength_pair_10d_20d_stock_weight", 0.88))
            if short_mid_mode in {"tier2_add", "tier2_replace"} and short_mid_action_confirm:
                tier2_weight = float(getattr(cfg, "short_mid_tier2_stock_weight", tier2_weight))
            target_stock = max(target_stock, tier2_weight)
            offensive_tier = max(offensive_tier, 2)

        # 강한 Tier 3: 이전 상태와 관계없이 96% 목표.
        if tier3_signal:
            target_stock = max(target_stock, float(getattr(cfg, "up_strength_all3_base_stock_weight", 0.96)))
            offensive_tier = max(offensive_tier, 3)
            if bool(getattr(cfg, "force_tier3_rebalance", True)):
                force_rebalance = True

        # Full: 가장 강한 구간. 100% 목표.
        if full_stock_signal:
            target_stock = max(target_stock, float(getattr(cfg, "up_strength_all3_strong_stock_weight", 1.00)))
            offensive_tier = max(offensive_tier, 3)
            strong_all3 = True
            if bool(getattr(cfg, "force_full_stock_rebalance", True)):
                force_rebalance = True

        # BEAR mid-trend에서는 Tier1/옵션 Tier2만 제한한다. Tier3/Full은 신호 품질이 더 강하므로 유지한다.
        if trend_state == "BEAR" and offensive_tier in {1, 2} and not (tier3_signal or full_stock_signal):
            target_stock = min(target_stock, max(base_stock, 0.80))

        stock = max(stock, target_stock)
        offensive_active = bool(offensive_tier > 0 and stock > base_stock + 1e-12)

    # v8.6.34: bear/down 약한 확률 기반 cut은 제거한다.

    max_bonus = float(getattr(cfg, "direction_strength_max_stock_bonus", 1.00))
    max_cut = float(getattr(cfg, "direction_strength_max_stock_cut", 0.04))
    bonus = max(0.0, stock - base_stock)
    if offensive_tier < 3 and bonus > max_bonus:
        stock = min(stock, base_stock + max_bonus)
        bonus = max_bonus
    cut = float(np.clip(cut, 0.0, max_cut))
    stock = float(np.clip(stock - cut, 0.20, 1.00))

    w = _redistribute_after_stock_change(stock, base_w)
    spread = float(max(p5, p10, p20) - min(p5, p10, p20))
    meta = {
        "policy_overlay": float(w[0] - base_w[0]),
        "mid_trend_score": trend_score,
        "mid_trend_state": trend_state,
        "defensive_risk_score": float(np.clip(ph, 0.0, 1.0)),
        "direction_strength_score": float(pus_score),
        "up_strength_score": float(pus_score),
        "up_strength_5d": float(p5),
        "up_strength_10d": float(p10),
        "up_strength_20d": float(p20),
        "up_strength_pred_5d": bool(pred5),
        "up_strength_pred_5d_raw": bool(pred5_raw),
        "up_strength_pred_10d": bool(pred10),
        "up_strength_pred_20d": bool(pred20),
        "up_strength_consensus_count": int(pred_count),
        "up_strength_consensus_pattern": f"{pred_pattern}:p5={p5:.4f}|p10={p10:.4f}|p20={p20:.4f}|score={pus_score:.4f}|tier={offensive_tier}|force={int(force_rebalance)}",
        "up_strength_consensus_weighted_score": float(pus_score),
        "up_strength_prob_spread": spread,
        "up_strength_consensus_target_stock": float(target_stock),
        "up_strength_strong_all3": bool(strong_all3),
        "up_strength_five_day_only": bool(five_day_only),
        "up_strength_bonus": float(max(0.0, w[0] - base_stock)),
        "bear_strength_cut": float(cut),
        "allocation_downrisk_score": float(pdn_alloc),
        "offensive_active": bool(offensive_active),
        "offensive_tier": int(offensive_tier),
        "tier1_signal": bool(tier1_signal),
        "tier2_signal": bool(tier2_signal),
        "original_tier2_signal": bool(original_tier2_signal),
        "tier3_signal": bool(tier3_signal),
        "full_stock_signal": bool(full_stock_signal),
        "short_mid_confirm": bool(short_mid_confirm),
        "short_mid_strong_confirm": bool(short_mid_strong_confirm),
        "short_mid_loose_confirm": bool(short_mid_loose_confirm),
        "short_mid_all3_confirm": bool(short_mid_all3_confirm),
        "strength_combo_5d_signal": bool(combo_5d_signal),
        "strength_combo_10d_signal": bool(combo_10d_signal),
        "strength_combo_20d_signal": bool(combo_20d_signal),
        "strength_combo_5d_10d_signal": bool(combo_5d_10d_signal),
        "strength_combo_5d_20d_signal": bool(combo_5d_20d_signal),
        "strength_combo_10d_20d_signal": bool(combo_10d_20d_signal),
        "strength_combo_all3_signal": bool(combo_all3_signal),
        "strength_combo_best_name": str(best_combo_name),
        "strength_combo_best_stock_weight": float(best_combo_stock_weight),
        "strength_combo_best_tier": int(best_combo_tier),
        "strength_combo_policy_mode": str(combo_mode),
        "strength_combo_policy_action": str(combo_policy_action),
        "strength_combo_hv_threshold": float(combo_hv_threshold),
        "strength_combo_hv_ok": bool(combo_hv_ok),
        "strength_combo_score_ok": bool(combo_score_ok),
        "short_mid_policy_action": str(short_mid_policy_action),
        "short_mid_mode": str(short_mid_mode),
        "short_mid_action_signal": str(short_mid_action_signal),
        "short_mid_action_confirm": bool(short_mid_action_confirm),
        "short_mid_p5_threshold": float(sm_p5),
        "short_mid_p10_threshold": float(sm_p10),
        "short_mid_p20_threshold": float(sm_p20),
        "short_mid_high_vol_threshold": float(sm_hv),
        "short_mid_strong_high_vol_threshold": float(sm_hv_strong),
        "short_mid_loose_high_vol_threshold": float(sm_hv_loose),
        "force_rebalance": bool(force_rebalance),
        "p20_up_strengthening": float(p20),
        "p20_tier": int(3 if tier3_signal else (1 if tier1_signal else 0)),
        "policy_note": "vol_base_all_strength_combos_enabled",
    }
    return w, meta


# ============================================================
# v8.6.34 Separate PortfolioPolicyModel
# ============================================================

PORTFOLIO_POLICY_CLASSES: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("P0_EXTREME_DEFENSIVE", (0.30, 0.45, 0.25)),
    ("P1_DEFENSIVE", (0.42, 0.38, 0.20)),
    ("P2_CAUTION", (0.52, 0.31, 0.17)),
    ("P3_BALANCED", (0.60, 0.26, 0.14)),
    ("P4_NORMAL", (0.68, 0.21, 0.11)),
    ("P5_OFFENSIVE", (0.78, 0.14, 0.08)),
    ("P6_TIER1", (0.80, 0.13, 0.07)),
    ("P7_TIER3", (0.96, 0.03, 0.01)),
    ("P8_FULL", (1.00, 0.00, 0.00)),
)
PORTFOLIO_POLICY_CLASS_TO_ID = {name: i for i, (name, _) in enumerate(PORTFOLIO_POLICY_CLASSES)}
PORTFOLIO_POLICY_ID_TO_CLASS = {i: name for i, (name, _) in enumerate(PORTFOLIO_POLICY_CLASSES)}


def portfolio_policy_feature_columns(df: pd.DataFrame) -> List[str]:
    """1단계 확률 출력만 2단계 PortfolioPolicyModel 입력으로 사용한다."""
    candidates = [
        "prob_normal",
        "prob_high_vol",
        "prob_overall_risk",
        "prob_up_strengthening",
        "prob_up_strengthening_score",
        "prob_up_strengthening_5d",
        "prob_up_strengthening_10d",
        "prob_up_strengthening_20d",
        "prob_down_strengthening",
        "prob_down_strengthening_score",
        "prob_down_strengthening_5d",
        "prob_down_strengthening_10d",
        "prob_down_strengthening_20d",
    ]
    return [c for c in candidates if c in df.columns]


def portfolio_candidate_weights(class_id: int) -> Tuple[float, float, float]:
    idx = int(np.clip(class_id, 0, len(PORTFOLIO_POLICY_CLASSES) - 1))
    return _normalize_weight_tuple(*PORTFOLIO_POLICY_CLASSES[idx][1])


def infer_portfolio_policy_class_from_row(row: pd.Series, cfg: Config) -> int:
    """모델 학습 전/불충분 구간 fallback. Tier2는 사용하지 않는다."""
    ph = _row_float(row, "prob_high_vol", 0.50)
    p10 = _row_float(row, "prob_up_strengthening_10d", 0.0)
    p20 = _row_float(row, "prob_up_strengthening_20d", _row_float(row, "prob_up_strengthening", 0.0))
    score = _row_float(row, "prob_up_strengthening_score", combine_up_strength_score_from_values({5: 0.0, 10: p10, 20: p20}, cfg))

    # Full / Tier3 / Tier1만 사용. Tier2는 의도적으로 제거한다.
    if score >= float(getattr(cfg, "up_strength_full_stock_score_threshold", 0.50)) and p10 >= float(getattr(cfg, "up_strength_full_stock_10d_threshold", 0.38)) and p20 >= float(getattr(cfg, "up_strength_full_stock_20d_threshold", 0.42)) and ph < float(getattr(cfg, "up_strength_full_stock_high_vol_threshold", 0.58)):
        return PORTFOLIO_POLICY_CLASS_TO_ID["P8_FULL"]
    if score >= float(getattr(cfg, "up_strength_bonus_threshold_3", 0.45)) and p20 >= float(getattr(cfg, "up_strength_confirm_20d_threshold_3", 0.38)) and ph < float(getattr(cfg, "up_strength_low_vol_threshold_3", 0.68)):
        return PORTFOLIO_POLICY_CLASS_TO_ID["P7_TIER3"]
    if score >= float(getattr(cfg, "up_strength_bonus_threshold_1", 0.30)) and p20 >= 0.30 and ph < float(getattr(cfg, "up_strength_low_vol_threshold_1", 0.82)):
        return PORTFOLIO_POLICY_CLASS_TO_ID["P6_TIER1"]
    if ph < 0.25:
        return PORTFOLIO_POLICY_CLASS_TO_ID["P5_OFFENSIVE"]
    if ph < 0.35:
        return PORTFOLIO_POLICY_CLASS_TO_ID["P5_OFFENSIVE"]
    if ph < 0.50:
        return PORTFOLIO_POLICY_CLASS_TO_ID["P4_NORMAL"]
    if ph < 0.65:
        return PORTFOLIO_POLICY_CLASS_TO_ID["P3_BALANCED"]
    if ph < 0.75:
        return PORTFOLIO_POLICY_CLASS_TO_ID["P2_CAUTION"]
    if ph < 0.86:
        return PORTFOLIO_POLICY_CLASS_TO_ID["P1_DEFENSIVE"]
    return PORTFOLIO_POLICY_CLASS_TO_ID["P0_EXTREME_DEFENSIVE"]


def compute_portfolio_policy_labels(pred_df: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    각 날짜 t에서 미래 H일 동안 후보 포트폴리오별 utility를 계산하고 argmax를 라벨로 둔다.
    학습 시에는 i-H 이하의 과거 라벨만 사용해 overlap leakage를 피한다.
    """
    n = len(pred_df)
    h = int(getattr(cfg, "portfolio_policy_horizon", 20))
    labels = np.full(n, -1, dtype=int)
    best_util = np.full(n, np.nan, dtype=float)
    stock_r = pred_df["stock_next_return"].astype(float).fillna(0.0).values
    bond_r = pred_df["bond_next_return"].astype(float).fillna(0.0).values
    cash_r = pred_df["cash_next_return"].astype(float).fillna(0.0).values
    vol_penalty = float(getattr(cfg, "portfolio_utility_vol_penalty", 0.50))
    mdd_penalty = float(getattr(cfg, "portfolio_utility_mdd_penalty", 0.80))
    turnover_penalty = float(getattr(cfg, "portfolio_utility_turnover_penalty", 0.001))

    for i in range(0, max(0, n - h + 1)):
        window = slice(i, i + h)
        ph = float(pred_df.iloc[i].get("prob_high_vol", 0.50))
        base_w = base_weight_from_vol_probability(ph, cfg) if bool(getattr(cfg, "use_vol_probability_base_allocation", True)) else (0.60, 0.26, 0.14)
        utils: List[float] = []
        for _, w in PORTFOLIO_POLICY_CLASSES:
            w = _normalize_weight_tuple(*w)
            r = w[0] * stock_r[window] + w[1] * bond_r[window] + w[2] * cash_r[window]
            eq = np.cumprod(1.0 + np.nan_to_num(r, nan=0.0))
            future_return = float(eq[-1] - 1.0) if len(eq) else 0.0
            future_vol = float(np.std(r, ddof=0) * math.sqrt(max(1, h))) if len(r) else 0.0
            peak = np.maximum.accumulate(eq) if len(eq) else np.array([1.0])
            dd = eq / np.maximum(peak, 1e-12) - 1.0 if len(eq) else np.array([0.0])
            future_mdd = float(abs(np.min(dd))) if len(dd) else 0.0
            turnover_from_base = float(sum(abs(w[j] - base_w[j]) for j in range(3)))
            utility = future_return - vol_penalty * future_vol - mdd_penalty * future_mdd - turnover_penalty * turnover_from_base
            utils.append(float(utility))
        best = int(np.argmax(utils))
        labels[i] = best
        best_util[i] = float(utils[best])
    return labels, best_util


def build_portfolio_policy_model(cfg: Config, num_classes: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=int(num_classes),
        n_estimators=int(getattr(cfg, "portfolio_policy_n_estimators", 120)),
        learning_rate=float(getattr(cfg, "portfolio_policy_learning_rate", 0.035)),
        max_depth=int(getattr(cfg, "portfolio_policy_max_depth", 2)),
        min_child_weight=float(getattr(cfg, "portfolio_policy_min_child_weight", 8.0)),
        subsample=float(getattr(cfg, "portfolio_policy_subsample", 0.85)),
        colsample_bytree=float(getattr(cfg, "portfolio_policy_colsample_bytree", 0.85)),
        reg_lambda=float(getattr(cfg, "portfolio_policy_reg_lambda", 10.0)),
        reg_alpha=float(getattr(cfg, "portfolio_policy_reg_alpha", 0.2)),
        random_state=int(getattr(cfg, "random_state", 42)),
        n_jobs=int(getattr(cfg, "n_jobs", -1)),
        eval_metric="mlogloss",
        tree_method="hist",
    )


def run_portfolio_policy_model(pred_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """1단계 OOS 확률 출력 → 2단계 포트폴리오 클래스 예측."""
    out = pred_df.copy().reset_index(drop=True)
    feature_cols = portfolio_policy_feature_columns(out)
    labels, best_util = compute_portfolio_policy_labels(out, cfg)
    out["portfolio_model_oracle_class_id"] = labels
    out["portfolio_model_oracle_class"] = [PORTFOLIO_POLICY_ID_TO_CLASS.get(int(x), "UNKNOWN") if int(x) >= 0 else "UNKNOWN" for x in labels]
    out["portfolio_model_oracle_utility"] = best_util

    if not feature_cols:
        # 확률 컬럼이 없으면 fallback만 저장한다.
        pred_ids = [infer_portfolio_policy_class_from_row(row, cfg) for _, row in out.iterrows()]
        confidences = [0.0 for _ in pred_ids]
        source = ["fallback_no_features" for _ in pred_ids]
    else:
        X_all = out[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
        pred_ids: List[int] = []
        confidences: List[float] = []
        source: List[str] = []
        last_model: Optional[Pipeline] = None
        last_classes: Optional[np.ndarray] = None
        min_train = int(getattr(cfg, "portfolio_policy_min_train_rows", 756))
        max_train = getattr(cfg, "portfolio_policy_max_train_rows", 1260)
        retrain_every = int(getattr(cfg, "portfolio_policy_retrain_every_n_days", 10))
        h = int(getattr(cfg, "portfolio_policy_horizon", 20))

        for i, row in out.iterrows():
            train_end = i - h
            do_retrain = bool(i == 0 or last_model is None or (i % max(1, retrain_every) == 0))
            if do_retrain and train_end > 0:
                train_idx = np.arange(0, train_end)
                train_idx = train_idx[labels[train_idx] >= 0]
                if max_train is not None and len(train_idx) > int(max_train):
                    train_idx = train_idx[-int(max_train):]
                y_raw = labels[train_idx]
                unique_classes = np.unique(y_raw)
                if len(train_idx) >= min_train and len(unique_classes) >= 2:
                    class_to_local = {int(c): k for k, c in enumerate(unique_classes)}
                    y_local = np.array([class_to_local[int(c)] for c in y_raw], dtype=int)
                    model = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("clf", build_portfolio_policy_model(cfg, num_classes=len(unique_classes))),
                    ])
                    model.fit(X_all.iloc[train_idx], y_local)
                    last_model = model
                    last_classes = unique_classes.astype(int)

            fallback_id = infer_portfolio_policy_class_from_row(row, cfg)
            if last_model is None or last_classes is None:
                pred_ids.append(int(fallback_id))
                confidences.append(0.0)
                source.append("fallback_warmup")
                continue
            proba_local = last_model.predict_proba(X_all.iloc[[i]])[0]
            local_best = int(np.argmax(proba_local))
            class_id = int(last_classes[local_best])
            confidence = float(np.max(proba_local))
            if confidence < float(getattr(cfg, "portfolio_model_min_confidence", 0.0)):
                pred_ids.append(int(fallback_id))
                confidences.append(float(confidence))
                source.append("fallback_low_confidence")
            else:
                pred_ids.append(class_id)
                confidences.append(float(confidence))
                source.append("model")

    out["portfolio_model_class_id"] = pred_ids
    out["portfolio_model_class"] = [PORTFOLIO_POLICY_ID_TO_CLASS.get(int(x), "UNKNOWN") for x in pred_ids]
    out["portfolio_model_confidence"] = confidences
    out["portfolio_model_source"] = source
    weights = [portfolio_candidate_weights(int(x)) for x in pred_ids]
    out["portfolio_model_stock_weight"] = [float(w[0]) for w in weights]
    out["portfolio_model_bond_weight"] = [float(w[1]) for w in weights]
    out["portfolio_model_cash_weight"] = [float(w[2]) for w in weights]
    return out


def apply_portfolio_policy_model(
    base_w: Tuple[float, float, float],
    regime: str,
    row: pd.Series,
    cfg: Config,
) -> Tuple[Tuple[float, float, float], Dict[str, object]]:
    if "portfolio_model_stock_weight" not in row.index:
        class_id = infer_portfolio_policy_class_from_row(row, cfg)
        w = portfolio_candidate_weights(class_id)
        src = "fallback_no_model_columns"
        conf = 0.0
        cls = PORTFOLIO_POLICY_ID_TO_CLASS.get(class_id, "UNKNOWN")
    else:
        w = _normalize_weight_tuple(
            _row_float(row, "portfolio_model_stock_weight", base_w[0]),
            _row_float(row, "portfolio_model_bond_weight", base_w[1]),
            _row_float(row, "portfolio_model_cash_weight", base_w[2]),
        )
        src = str(row.get("portfolio_model_source", "model"))
        conf = _row_float(row, "portfolio_model_confidence", 0.0)
        cls = str(row.get("portfolio_model_class", "UNKNOWN"))
        class_id = int(row.get("portfolio_model_class_id", infer_portfolio_policy_class_from_row(row, cfg)))

    trend_score, trend_state = compute_mid_trend_score(row)
    force_rebalance = bool(getattr(cfg, "portfolio_model_force_rebalance", False) and class_id >= PORTFOLIO_POLICY_CLASS_TO_ID["P7_TIER3"])
    meta = {
        "policy_overlay": float(w[0] - base_w[0]),
        "mid_trend_score": trend_score,
        "mid_trend_state": trend_state,
        "defensive_risk_score": _row_float(row, "prob_high_vol", 0.0),
        "direction_strength_score": _row_float(row, "prob_up_strengthening_score", 0.0),
        "up_strength_score": _row_float(row, "prob_up_strengthening_score", 0.0),
        "up_strength_5d": _row_float(row, "prob_up_strengthening_5d", 0.0),
        "up_strength_10d": _row_float(row, "prob_up_strengthening_10d", 0.0),
        "up_strength_20d": _row_float(row, "prob_up_strengthening_20d", 0.0),
        "up_strength_bonus": float(max(0.0, w[0] - base_w[0])),
        "bear_strength_cut": float(max(0.0, base_w[0] - w[0])),
        "allocation_downrisk_score": _row_float(row, "prob_high_vol", 0.0),
        "offensive_active": bool(w[0] > base_w[0] + 1e-12),
        "offensive_tier": int(3 if class_id >= PORTFOLIO_POLICY_CLASS_TO_ID["P7_TIER3"] else (1 if class_id == PORTFOLIO_POLICY_CLASS_TO_ID["P6_TIER1"] else 0)),
        "tier1_signal": bool(class_id == PORTFOLIO_POLICY_CLASS_TO_ID["P6_TIER1"]),
        "tier2_signal": False,
        "tier3_signal": bool(class_id == PORTFOLIO_POLICY_CLASS_TO_ID["P7_TIER3"]),
        "full_stock_signal": bool(class_id == PORTFOLIO_POLICY_CLASS_TO_ID["P8_FULL"]),
        "force_rebalance": bool(force_rebalance),
        "portfolio_model_active": True,
        "portfolio_model_class": cls,
        "portfolio_model_class_id": int(class_id),
        "portfolio_model_confidence": float(conf),
        "portfolio_model_source": src,
        "policy_note": "separate_portfolio_policy_model",
    }
    return w, meta


def portfolio_policy_summary(pred_df: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    if "portfolio_model_class" not in pred_df.columns:
        return {"enabled": bool(getattr(cfg, "enable_portfolio_policy_model", False)), "available": False}
    dist = pred_df["portfolio_model_class"].value_counts(normalize=True).mul(100).round(2).to_dict()
    source_dist = pred_df.get("portfolio_model_source", pd.Series(index=pred_df.index, data="unknown")).value_counts(normalize=True).mul(100).round(2).to_dict()
    return {
        "enabled": bool(getattr(cfg, "enable_portfolio_policy_model", False)),
        "available": True,
        "used_as_allocation_policy": str(getattr(cfg, "policy_mode", "")) == "portfolio_model",
        "horizon": int(getattr(cfg, "portfolio_policy_horizon", 20)),
        "feature_cols": portfolio_policy_feature_columns(pred_df),
        "class_distribution_pct": dist,
        "source_distribution_pct": source_dist,
        "avg_confidence": float(pred_df.get("portfolio_model_confidence", pd.Series(index=pred_df.index, data=0.0)).astype(float).mean()),
        "avg_model_stock_weight": float(pred_df.get("portfolio_model_stock_weight", pd.Series(index=pred_df.index, data=np.nan)).astype(float).mean()),
        "tier2_removed": bool(getattr(cfg, "disable_tier2_signal", True)),
        "candidate_portfolios": {name: {"stock": w[0], "bond": w[1], "cash": w[2]} for name, w in PORTFOLIO_POLICY_CLASSES},
    }

def apply_policy_overlay(
    base_w: Tuple[float, float, float],
    regime: str,
    row: pd.Series,
    cfg: Config,
) -> Tuple[Tuple[float, float, float], Dict[str, object]]:
    if not getattr(cfg, "use_policy_overlay", True):
        trend_score, trend_state = compute_mid_trend_score(row)
        return base_w, {
            "policy_overlay": 0.0,
            "mid_trend_score": trend_score,
            "mid_trend_state": trend_state,
            "policy_note": "policy_overlay_disabled",
        }
    mode = str(getattr(cfg, "policy_mode", "return_seeking")).lower()
    if mode == "base":
        trend_score, trend_state = compute_mid_trend_score(row)
        return base_w, {
            "policy_overlay": 0.0,
            "mid_trend_score": trend_score,
            "mid_trend_state": trend_state,
            "policy_note": "base_no_policy_overlay",
        }
    if mode == "return_seeking":
        return apply_return_seeking_policy(base_w, regime, row, cfg)
    if mode == "defensive_risk":
        return apply_defensive_risk_policy(base_w, regime, row, cfg)
    if mode == "aggressive_dynamic":
        return apply_aggressive_dynamic_policy(base_w, regime, row, cfg)
    if mode == "portfolio_model":
        return apply_portfolio_policy_model(base_w, regime, row, cfg)
    if mode in {"direction_strength_specialist", "strength_specialist", "ds_specialist"}:
        return apply_direction_strength_specialist_policy(base_w, regime, row, cfg)
    raise ValueError(f"unknown policy_mode: {mode}")


def apply_continuous_adjustment(
    base_w: Tuple[float, float, float],
    prob_high_vol: float,
    prob_down_risk: float,
    cfg: Config,
) -> Tuple[float, float, float]:
    if not cfg.use_continuous_adjustment:
        return base_w
    stock, bond, cash = base_w
    cut = cfg.continuous_high_vol_weight * prob_high_vol + cfg.continuous_down_risk_weight * prob_down_risk
    cut = float(np.clip(cut, 0.0, cfg.max_continuous_stock_cut))
    new_stock = max(0.0, stock - cut)
    defensive_add = stock - new_stock
    defensive_total = bond + cash
    if defensive_total <= 0:
        return _normalize_weight_tuple(new_stock, defensive_add * 0.65, defensive_add * 0.35)
    new_bond = bond + defensive_add * bond / defensive_total
    new_cash = cash + defensive_add * cash / defensive_total
    return _normalize_weight_tuple(new_stock, new_bond, new_cash)


def allocate_from_probs(
    prob_high_vol: float,
    prob_down_risk: float,
    g: Dict[str, float],
    cfg: Config,
    prev_weights: Optional[Tuple[float, float, float]],
) -> Tuple[Tuple[float, float, float], str]:
    regime = classify_gate(prob_high_vol, prob_down_risk, g)
    target = base_weight_for_regime(regime, g)
    target = apply_continuous_adjustment(target, prob_high_vol, prob_down_risk, cfg)

    if prev_weights is not None:
        total_delta = sum(abs(target[i] - prev_weights[i]) for i in range(3))
        if total_delta < g["no_trade_band"]:
            return prev_weights, regime
    return target, regime


def perf_stats(returns: pd.Series, initial_capital: float) -> Dict[str, float]:
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return {"final_capital": initial_capital, "total_return": 0.0, "cagr": 0.0, "mdd": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0}
    equity = initial_capital * (1.0 + r).cumprod()
    final_capital = float(equity.iloc[-1])
    total_return = final_capital / initial_capital - 1.0
    years = len(r) / 252.0
    cagr = (final_capital / initial_capital) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = float(dd.min())
    vol = float(r.std())
    sharpe = float((r.mean() / vol) * math.sqrt(252)) if vol > 0 else 0.0
    downside = r[r < 0]
    down_std = float(downside.std())
    sortino = float((r.mean() / down_std) * math.sqrt(252)) if down_std > 0 else 0.0
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
    return {
        "final_capital": final_capital,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "mdd": mdd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
    }




# ============================================================
# 6.31 Tier Weight Optimizer
# ============================================================

def _parse_grid_arg(value: Optional[str], default: Tuple[float, ...]) -> Tuple[float, ...]:
    if value is None or str(value).strip() == "":
        return tuple(float(x) for x in default)
    vals: List[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        return tuple(float(x) for x in default)
    return tuple(sorted(set(vals)))


def _boolish(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return False
        return bool(value)
    txt = str(value).strip().lower()
    return txt in {"1", "true", "t", "yes", "y"}


def _defensive_split_from_stock(stock: float, cfg: Config) -> Tuple[float, float, float]:
    stock = float(np.clip(stock, 0.0, 1.0))
    defensive = max(0.0, 1.0 - stock)
    bond_ratio = float(np.clip(getattr(cfg, "vol_base_bond_ratio_of_defensive", 0.65), 0.0, 1.0))
    bond = defensive * bond_ratio
    cash = defensive * (1.0 - bond_ratio)
    return _normalize_weight_tuple(stock, bond, cash)


def _tier_weight_base_stock(ph: float, cfg: Config, weights: Dict[str, float]) -> float:
    """v8.6.25+ 계열 high-vol 확률 기반 기본 주식 비중. lt25만 grid 최적화 대상으로 둔다."""
    ph = float(np.clip(ph, 0.0, 1.0))
    if ph < 0.25:
        return float(weights.get("base_lt25", getattr(cfg, "vol_base_stock_lt_25", 0.78)))
    if ph < 0.35:
        return float(getattr(cfg, "vol_base_stock_lt_35", 0.74))
    if ph < 0.50:
        return float(getattr(cfg, "vol_base_stock_lt_50", 0.68))
    if ph < 0.65:
        return float(getattr(cfg, "vol_base_stock_lt_65", 0.60))
    if ph < 0.75:
        return float(getattr(cfg, "vol_base_stock_lt_75", 0.52))
    if ph < 0.86:
        return float(getattr(cfg, "vol_base_stock_lt_86", 0.42))
    return float(getattr(cfg, "vol_base_stock_ge_86", 0.30))


def _tier_weight_signal_weights(row: pd.Series, cfg: Config, weights: Dict[str, float]) -> Tuple[float, float, float]:
    base_stock = _tier_weight_base_stock(float(row.get("prob_high_vol", 0.0)), cfg, weights)
    stock = base_stock
    if _boolish(row.get("tier1_signal", False)):
        stock = max(stock, float(weights["tier1"]))
    if _boolish(row.get("tier2_signal", False)):
        stock = max(stock, float(weights["tier2"]))
    if _boolish(row.get("tier3_signal", False)):
        stock = max(stock, float(weights["tier3"]))
    if _boolish(row.get("full_stock_signal", False)):
        stock = max(stock, float(weights["full"]))
    return _defensive_split_from_stock(stock, cfg)


def _candidate_weight_grid(cfg: Config) -> List[Dict[str, float]]:
    base_grid = tuple(getattr(cfg, "tier_weight_opt_base_lt25_grid", (0.78,)))
    if not bool(getattr(cfg, "tier_weight_opt_include_base_lt25", True)):
        base_grid = (float(getattr(cfg, "vol_base_stock_lt_25", 0.78)),)
    t1_grid = tuple(getattr(cfg, "tier_weight_opt_tier1_grid", (0.80,)))
    t2_grid = tuple(getattr(cfg, "tier_weight_opt_tier2_grid", (0.88,)))
    t3_grid = tuple(getattr(cfg, "tier_weight_opt_tier3_grid", (0.96,)))
    full_grid = tuple(getattr(cfg, "tier_weight_opt_full_grid", (1.00,)))
    candidates: List[Dict[str, float]] = []
    for b in base_grid:
        for t1 in t1_grid:
            for t2 in t2_grid:
                for t3 in t3_grid:
                    for full in full_grid:
                        # Tier order constraint. base_lt25는 시장환경 기본 비중이므로 tier1보다 높아도 허용한다.
                        if not (float(t1) <= float(t2) <= float(t3) <= float(full)):
                            continue
                        candidates.append({
                            "base_lt25": float(b),
                            "tier1": float(t1),
                            "tier2": float(t2),
                            "tier3": float(t3),
                            "full": float(full),
                        })
    return candidates


def _tier_weight_score(metrics: Dict[str, float], annual_turnover: float, profile: str) -> float:
    cagr = float(metrics.get("cagr", 0.0))
    mdd = abs(float(metrics.get("mdd", 0.0)))
    sharpe = float(metrics.get("sharpe", 0.0))
    sortino = float(metrics.get("sortino", 0.0))
    calmar = float(metrics.get("calmar", 0.0))
    profile = str(profile or "aggressive").lower()
    if profile == "balanced":
        return 1.00 * cagr + 0.35 * sharpe + 0.20 * sortino + 0.10 * calmar - 0.75 * mdd - 0.10 * annual_turnover
    if profile == "calmar":
        return 0.75 * cagr + 0.25 * sharpe + 0.55 * calmar - 0.85 * mdd - 0.08 * annual_turnover
    if profile == "sharpe":
        return 0.70 * cagr + 0.70 * sharpe + 0.20 * sortino - 0.50 * mdd - 0.10 * annual_turnover
    # aggressive: 최고 성능 목표. CAGR를 크게 보되, MDD/turnover를 완전히 무시하지 않는다.
    return 1.30 * cagr + 0.25 * sharpe + 0.10 * sortino - 0.60 * mdd - 0.08 * annual_turnover


def simulate_tier_weight_strategy(
    pred_df: pd.DataFrame,
    cfg: Config,
    weights: Dict[str, float],
    initial_weights: Optional[Tuple[float, float, float]] = None,
) -> Tuple[pd.DataFrame, Tuple[float, float, float]]:
    rows: List[Dict[str, object]] = []
    prev_w = initial_weights
    for _, row in pred_df.iterrows():
        signal_w = _tier_weight_signal_weights(row, cfg, weights)
        if prev_w is None:
            w = signal_w
            hold_reason = "initial"
        else:
            force = bool(
                (_boolish(row.get("tier3_signal", False)) and bool(getattr(cfg, "force_tier3_rebalance", True)))
                or (_boolish(row.get("full_stock_signal", False)) and bool(getattr(cfg, "force_full_stock_rebalance", True)))
            )
            rebalance_due = _boolish(row.get("rebalance_due", False)) or _boolish(row.get("emergency_rebalance", False))
            delta = sum(abs(signal_w[i] - prev_w[i]) for i in range(3))
            if force:
                w = signal_w
                hold_reason = "tier_force"
            elif rebalance_due and delta >= float(getattr(cfg, "no_trade_band", 0.12)):
                w = signal_w
                hold_reason = "scheduled"
            else:
                w = prev_w
                hold_reason = "hold"
        turnover = 0.0 if prev_w is None else sum(abs(w[i] - prev_w[i]) for i in range(3))
        gross = (
            w[0] * float(row.get("stock_next_return", 0.0))
            + w[1] * float(row.get("bond_next_return", 0.0))
            + w[2] * float(row.get("cash_next_return", 0.0))
        )
        cost = float(getattr(cfg, "transaction_cost_rate", 0.001)) * turnover
        net = gross - cost
        out = {
            "Date": row.get("Date"),
            "tier_opt_stock_weight": float(w[0]),
            "tier_opt_bond_weight": float(w[1]),
            "tier_opt_cash_weight": float(w[2]),
            "tier_opt_signal_stock_weight": float(signal_w[0]),
            "tier_opt_turnover": float(turnover),
            "tier_opt_transaction_cost": float(cost),
            "tier_opt_return_gross": float(gross),
            "tier_opt_return_net": float(net),
            "tier_opt_hold_reason": hold_reason,
            "tier1_signal": bool(_boolish(row.get("tier1_signal", False))),
            "tier2_signal": bool(_boolish(row.get("tier2_signal", False))),
            "tier3_signal": bool(_boolish(row.get("tier3_signal", False))),
            "full_stock_signal": bool(_boolish(row.get("full_stock_signal", False))),
            "prob_high_vol": float(row.get("prob_high_vol", np.nan)),
            "prob_up_strengthening_score": float(row.get("prob_up_strengthening_score", np.nan)),
            "prob_down_strengthening_score": float(row.get("prob_down_strengthening_score", np.nan)),
            "stock_next_return": float(row.get("stock_next_return", 0.0)),
            "bond_next_return": float(row.get("bond_next_return", 0.0)),
            "cash_next_return": float(row.get("cash_next_return", 0.0)),
            "base_lt25_weight": float(weights["base_lt25"]),
            "tier1_weight": float(weights["tier1"]),
            "tier2_weight": float(weights["tier2"]),
            "tier3_weight": float(weights["tier3"]),
            "full_weight": float(weights["full"]),
        }
        rows.append(out)
        prev_w = w
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df["tier_opt_equity_net"] = float(getattr(cfg, "initial_capital", 100_000_000)) * (1.0 + out_df["tier_opt_return_net"].astype(float)).cumprod()
        out_df["tier_opt_equity_gross"] = float(getattr(cfg, "initial_capital", 100_000_000)) * (1.0 + out_df["tier_opt_return_gross"].astype(float)).cumprod()
    return out_df, prev_w if prev_w is not None else _defensive_split_from_stock(float(weights.get("base_lt25", 0.78)), cfg)


def _evaluate_tier_candidate(pred_df: pd.DataFrame, cfg: Config, weights: Dict[str, float]) -> Dict[str, object]:
    sim_df, _ = simulate_tier_weight_strategy(pred_df, cfg, weights)
    metrics = perf_stats(sim_df["tier_opt_return_net"], float(getattr(cfg, "initial_capital", 100_000_000)))
    annual_turnover = float(sim_df["tier_opt_turnover"].mean() * 252.0) if not sim_df.empty else 0.0
    score = _tier_weight_score(metrics, annual_turnover, str(getattr(cfg, "tier_weight_opt_score_profile", "aggressive")))
    row: Dict[str, object] = dict(weights)
    row.update({
        "score": float(score),
        "annual_turnover": annual_turnover,
        "avg_stock_weight": float(sim_df["tier_opt_stock_weight"].mean()) if not sim_df.empty else 0.0,
        "trade_ratio": float((sim_df["tier_opt_turnover"] > 1e-12).mean()) if not sim_df.empty else 0.0,
    })
    row.update({f"net_{k}": v for k, v in metrics.items()})
    return row


def run_tier_weight_optimizer(pred_df: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    """
    Tier 2 포함 Tier별 목표 비중 optimizer.
    - 학습창에서 후보 비중 조합을 grid search.
    - 선택된 조합을 다음 OOS 구간에 적용.
    - 1단계/신호 모델은 재학습하지 않고, 이미 생성된 OOS 확률/신호만 사용한다.
    """
    required = {"tier1_signal", "tier2_signal", "tier3_signal", "full_stock_signal", "stock_next_return", "bond_next_return", "cash_next_return"}
    missing = sorted(required - set(pred_df.columns))
    if missing:
        raise ValueError(f"TierWeightOptimizer에 필요한 컬럼이 없습니다: {missing}")

    candidates = _candidate_weight_grid(cfg)
    if not candidates:
        raise ValueError("TierWeightOptimizer candidate grid가 비어 있습니다.")

    train_rows = int(getattr(cfg, "tier_weight_opt_train_rows", 756))
    min_train_rows = int(getattr(cfg, "tier_weight_opt_min_train_rows", 504))
    test_rows = int(getattr(cfg, "tier_weight_opt_test_rows", 63))
    n = len(pred_df)

    # 전체 구간 grid는 진단용이다. 채택은 walk-forward selection만 사용한다.
    full_grid_rows = [_evaluate_tier_candidate(pred_df, cfg, cand) for cand in candidates]
    full_grid_df = pd.DataFrame(full_grid_rows).sort_values("score", ascending=False).reset_index(drop=True)

    selected_rows: List[Dict[str, object]] = []
    oos_parts: List[pd.DataFrame] = []
    prev_w: Optional[Tuple[float, float, float]] = None

    start = max(min_train_rows, min(train_rows, n))
    fold_id = 0
    while start < n:
        train_start = max(0, start - train_rows)
        train_df = pred_df.iloc[train_start:start].copy()
        if len(train_df) < min_train_rows:
            break
        test_end = min(n, start + test_rows)
        test_df = pred_df.iloc[start:test_end].copy()
        if test_df.empty:
            break

        train_scores = [_evaluate_tier_candidate(train_df, cfg, cand) for cand in candidates]
        train_score_df = pd.DataFrame(train_scores).sort_values("score", ascending=False).reset_index(drop=True)
        best = train_score_df.iloc[0].to_dict()
        weights = {
            "base_lt25": float(best["base_lt25"]),
            "tier1": float(best["tier1"]),
            "tier2": float(best["tier2"]),
            "tier3": float(best["tier3"]),
            "full": float(best["full"]),
        }
        oos_df, prev_w = simulate_tier_weight_strategy(test_df, cfg, weights, initial_weights=prev_w)
        oos_df["tier_opt_fold"] = fold_id
        oos_parts.append(oos_df)
        oos_metrics = perf_stats(oos_df["tier_opt_return_net"], float(getattr(cfg, "initial_capital", 100_000_000)))
        selected_rows.append({
            "fold": fold_id,
            "train_start": str(train_df.iloc[0].get("Date")),
            "train_end": str(train_df.iloc[-1].get("Date")),
            "test_start": str(test_df.iloc[0].get("Date")),
            "test_end": str(test_df.iloc[-1].get("Date")),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            **weights,
            "train_score": float(best["score"]),
            "train_cagr": float(best.get("net_cagr", 0.0)),
            "train_mdd": float(best.get("net_mdd", 0.0)),
            "train_sharpe": float(best.get("net_sharpe", 0.0)),
            "train_annual_turnover": float(best.get("annual_turnover", 0.0)),
            "oos_cagr_segment": float(oos_metrics.get("cagr", 0.0)),
            "oos_mdd_segment": float(oos_metrics.get("mdd", 0.0)),
            "oos_sharpe_segment": float(oos_metrics.get("sharpe", 0.0)),
            "oos_avg_stock_weight": float(oos_df["tier_opt_stock_weight"].mean()) if not oos_df.empty else 0.0,
            "oos_annual_turnover_segment": float(oos_df["tier_opt_turnover"].mean() * 252.0) if not oos_df.empty else 0.0,
        })
        fold_id += 1
        start = test_end

    selected_df = pd.DataFrame(selected_rows)
    oos_daily_df = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame()
    if not oos_daily_df.empty:
        # 전체 OOS 구간 equity를 다시 누적 계산한다.
        oos_daily_df["tier_opt_equity_net"] = float(getattr(cfg, "initial_capital", 100_000_000)) * (1.0 + oos_daily_df["tier_opt_return_net"].astype(float)).cumprod()
        oos_daily_df["tier_opt_equity_gross"] = float(getattr(cfg, "initial_capital", 100_000_000)) * (1.0 + oos_daily_df["tier_opt_return_gross"].astype(float)).cumprod()
    oos_metrics = perf_stats(oos_daily_df["tier_opt_return_net"], float(getattr(cfg, "initial_capital", 100_000_000))) if not oos_daily_df.empty else {}
    oos_annual_turnover = float(oos_daily_df["tier_opt_turnover"].mean() * 252.0) if not oos_daily_df.empty else 0.0

    summary = {
        "enabled": True,
        "mode": "walk_forward_grid_search",
        "note": "Tier2 포함 Tier별 목표 주식 비중만 최적화한다. 1단계 확률/신호 모델은 기존 OOS 예측을 재사용한다.",
        "candidate_count": int(len(candidates)),
        "fold_count": int(len(selected_df)),
        "train_rows": int(train_rows),
        "min_train_rows": int(min_train_rows),
        "test_rows": int(test_rows),
        "score_profile": str(getattr(cfg, "tier_weight_opt_score_profile", "aggressive")),
        "candidate_grids": {
            "base_lt25": list(map(float, getattr(cfg, "tier_weight_opt_base_lt25_grid", ()))),
            "tier1": list(map(float, getattr(cfg, "tier_weight_opt_tier1_grid", ()))),
            "tier2": list(map(float, getattr(cfg, "tier_weight_opt_tier2_grid", ()))),
            "tier3": list(map(float, getattr(cfg, "tier_weight_opt_tier3_grid", ()))),
            "full": list(map(float, getattr(cfg, "tier_weight_opt_full_grid", ()))),
        },
        "oos_period": {
            "start": str(oos_daily_df.iloc[0]["Date"]) if not oos_daily_df.empty else None,
            "end": str(oos_daily_df.iloc[-1]["Date"]) if not oos_daily_df.empty else None,
            "rows": int(len(oos_daily_df)),
        },
        "oos_performance_after_cost": oos_metrics,
        "oos_annual_turnover": oos_annual_turnover,
        "oos_avg_stock_weight": float(oos_daily_df["tier_opt_stock_weight"].mean()) if not oos_daily_df.empty else 0.0,
        "selected_weight_mean": selected_df[["base_lt25", "tier1", "tier2", "tier3", "full"]].mean().to_dict() if not selected_df.empty else {},
        "selected_weight_mode": {col: float(selected_df[col].mode().iloc[0]) for col in ["base_lt25", "tier1", "tier2", "tier3", "full"] if not selected_df.empty},
        "full_sample_best_diagnostic": full_grid_df.head(1).to_dict("records")[0] if not full_grid_df.empty else {},
    }
    return {
        "summary": summary,
        "selected_folds": selected_df,
        "oos_daily": oos_daily_df,
        "full_grid": full_grid_df,
    }

def simulate_gate_config(pred_df: pd.DataFrame, g: Dict[str, float], cfg: Config) -> Dict[str, float]:
    prev_w: Optional[Tuple[float, float, float]] = None
    rets: List[float] = []
    turnovers: List[float] = []
    stock_weights: List[float] = []
    for _, row in pred_df.iterrows():
        ph = float(row["prob_high_vol"])
        pdn_raw = float(row.get("prob_down", row.get("prob_down_risk", 0.0)))
        pdn_alloc = allocation_downrisk_score(ph, pdn_raw, cfg)
        w, _ = allocate_from_probs(ph, pdn_alloc, g, cfg, prev_w)
        turnover = 0.0 if prev_w is None else sum(abs(w[i] - prev_w[i]) for i in range(3))
        gross = w[0] * row["stock_next_return"] + w[1] * row["bond_next_return"] + w[2] * row["cash_next_return"]
        net = gross - cfg.transaction_cost_rate * turnover
        rets.append(float(net))
        turnovers.append(float(turnover))
        stock_weights.append(float(w[0]))
        prev_w = w
    stats = perf_stats(pd.Series(rets), cfg.initial_capital)
    stats["avg_turnover"] = float(np.mean(turnovers)) if turnovers else 0.0
    stats["avg_stock_weight"] = float(np.mean(stock_weights)) if stock_weights else 0.0
    return stats


def build_small_gate_grid(cfg: Config) -> List[Dict[str, float]]:
    grid: List[Dict[str, float]] = []
    base = gate_config_from_cfg(cfg)
    i = 0
    for nht in [0.35, 0.40]:
        for hht in [0.55, 0.60, 0.65]:
            if hht <= nht:
                continue
            for rdt in [0.50, 0.55]:
                g = dict(base)
                g["gate_normal_high_vol_threshold"] = nht
                g["gate_high_vol_threshold"] = hht
                g["gate_riskoff_downrisk_threshold"] = rdt
                g["gate_watch_downrisk_threshold"] = max(0.60, rdt + 0.10)
                g["name"] = f"gate_{i:03d}_n{nht:.2f}_h{hht:.2f}_d{rdt:.2f}"
                grid.append(g)
                i += 1
    return grid


def gate_score(stats: Dict[str, float], cfg: Config) -> float:
    annual_turnover = stats.get("avg_turnover", 0.0) * 252.0
    return float(
        cfg.gate_score_cagr_weight * stats.get("cagr", 0.0)
        - cfg.gate_score_mdd_weight * abs(stats.get("mdd", 0.0))
        - cfg.gate_score_turnover_weight * annual_turnover
    )




def infer_regime_from_weights(weights: Tuple[float, float, float], g: Dict[str, float]) -> str:
    """실제 실행 비중과 가장 가까운 regime을 역산한다."""
    candidates = ["NORMAL", "WATCH", "HIGH_VOL", "RISK_OFF", "EXTREME_RISK"]
    best_name = "CUSTOM"
    best_dist = float("inf")
    for name in candidates:
        bw = base_weight_for_regime(name, g)
        dist = sum(abs(float(weights[i]) - float(bw[i])) for i in range(3))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name if best_dist <= 0.08 else "CUSTOM"

def apply_allocation(pred_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, int]]:
    pred_df = pred_df.copy().reset_index(drop=True)
    default_g = gate_config_from_cfg(cfg)
    grid = build_small_gate_grid(cfg)
    current_g = default_g
    usage: Dict[str, int] = {}

    prev_w: Optional[Tuple[float, float, float]] = None
    rows: List[Dict[str, object]] = []
    last_emergency_i = -10**9

    for i, row in pred_df.iterrows():
        if cfg.use_rolling_gate_optimization and i >= cfg.gate_min_window and i % cfg.gate_optimize_every_n_days == 0:
            hist = pred_df.iloc[max(0, i - cfg.gate_rolling_window):i].copy()
            best_g = current_g
            best_score = -np.inf
            for cand in grid:
                st = simulate_gate_config(hist, cand, cfg)
                s = gate_score(st, cfg)
                if s > best_score:
                    best_score = s
                    best_g = cand
            current_g = best_g

        ph = float(row["prob_high_vol"])
        pdn_raw = float(row.get("prob_down", row.get("prob_down_risk", 0.0)))
        pdn = allocation_downrisk_score(ph, pdn_raw, cfg)
        raw_emergency = (
            ph >= cfg.emergency_high_vol_threshold
            or (ph >= cfg.emergency_combined_high_vol_threshold and pdn >= cfg.emergency_combined_down_threshold)
        )
        emergency = bool(raw_emergency and (i - last_emergency_i >= cfg.emergency_cooldown_days))
        scheduled = (i % cfg.rebalance_every_n_days == 0)
        rebalance_due = prev_w is None or scheduled or emergency

        signal_regime = classify_gate(ph, pdn, current_g)
        if bool(getattr(cfg, "use_vol_probability_base_allocation", True)):
            base_signal_w = base_weight_from_vol_probability(ph, cfg)
            base_allocation_mode = "vol_probability_base"
        else:
            base_signal_w = apply_continuous_adjustment(base_weight_for_regime(signal_regime, current_g), ph, pdn, cfg)
            base_allocation_mode = "regime_bucket_base"
        signal_w, policy_meta = apply_policy_overlay(base_signal_w, signal_regime, row, cfg)

        strong_offensive_override = bool(
            getattr(cfg, "force_strong_offensive_rebalance", True)
            and bool(policy_meta.get("force_rebalance", False))
            and int(policy_meta.get("offensive_tier", 0)) >= 3
        )
        if strong_offensive_override:
            rebalance_due = True

        hold_reason = "rebalanced"
        trade_executed = False
        if prev_w is None:
            w = signal_w
            executed_regime = signal_regime
            hold_reason = "initial"
            trade_executed = True
        elif not rebalance_due:
            # v8.6.21 optional: offensive target이 사라졌는데 이전 98~100% 비중이 과도하게 남는 문제를 실험적으로 제어한다.
            # 기본값은 OFF다. 켜려면 --enable-stale-offensive-decay를 사용한다.
            stale_gap = float(prev_w[0] - signal_w[0])
            stale_up_prob = float(row.get("prob_up_strengthening_score", row.get("prob_up_strengthening", 0.0)))
            stale_decay = (
                bool(getattr(cfg, "enable_stale_offensive_decay", False))
                and stale_gap >= float(getattr(cfg, "stale_offensive_stock_gap_threshold", 0.055))
                and (
                    stale_up_prob < float(getattr(cfg, "stale_offensive_up_strength_reset_threshold", 0.20))
                    or ph >= float(getattr(cfg, "stale_offensive_high_vol_threshold", 0.55))
                )
            )
            if stale_decay:
                w = signal_w
                executed_regime = signal_regime
                hold_reason = "stale_offensive_decay"
                trade_executed = True
            else:
                w = prev_w
                executed_regime = infer_regime_from_weights(w, current_g)
                hold_reason = "not_rebalance_day"
        else:
            total_delta_to_signal = sum(abs(signal_w[j] - prev_w[j]) for j in range(3))
            if strong_offensive_override:
                w = signal_w
                executed_regime = signal_regime
                hold_reason = "strong_offensive_override"
                trade_executed = True
            elif total_delta_to_signal < current_g["no_trade_band"]:
                w = prev_w
                executed_regime = infer_regime_from_weights(w, current_g)
                hold_reason = "no_trade_band"
            else:
                w = signal_w
                executed_regime = signal_regime
                hold_reason = "emergency" if emergency else "scheduled"
                trade_executed = True

        turnover = 0.0 if prev_w is None else sum(abs(w[j] - prev_w[j]) for j in range(3))
        if turnover > 1e-12:
            trade_executed = True
        gross = w[0] * row["stock_next_return"] + w[1] * row["bond_next_return"] + w[2] * row["cash_next_return"]
        cost = cfg.transaction_cost_rate * turnover
        net = gross - cost

        if emergency and rebalance_due:
            last_emergency_i = i

        out = row.to_dict()
        out.update({
            "signal_regime": signal_regime,
            "allocation_regime": executed_regime,
            "executed_regime": executed_regime,
            "hold_reason": hold_reason,
            "held_by_no_trade_band": bool(hold_reason == "no_trade_band"),
            "held_by_schedule": bool(hold_reason == "not_rebalance_day"),
            "signal_stock_weight": float(signal_w[0]),
            "signal_bond_weight": float(signal_w[1]),
            "signal_cash_weight": float(signal_w[2]),
            "base_allocation_mode": str(base_allocation_mode),
            "base_signal_stock_weight": float(base_signal_w[0]),
            "base_signal_bond_weight": float(base_signal_w[1]),
            "base_signal_cash_weight": float(base_signal_w[2]),
            "strong_offensive_override": bool(strong_offensive_override),
            "stock_weight": float(w[0]),
            "bond_weight": float(w[1]),
            "cash_weight": float(w[2]),
            "turnover": float(turnover),
            "transaction_cost": float(cost),
            "strategy_return_gross": float(gross),
            "strategy_return_net": float(net),
            "rebalance_due": bool(rebalance_due),
            "rebalanced": bool(rebalance_due),
            "trade_executed": bool(trade_executed),
            "emergency_rebalance": bool(emergency and rebalance_due),
            "gate_config": current_g["name"],
            "policy_mode": str(getattr(cfg, "policy_mode", "base")),
            "policy_overlay": float(policy_meta.get("policy_overlay", 0.0)),
            "mid_trend_score": int(policy_meta.get("mid_trend_score", 0)),
            "mid_trend_state": str(policy_meta.get("mid_trend_state", "NEUTRAL")),
            "defensive_risk_score": float(policy_meta.get("defensive_risk_score", np.nan)),
            "policy_note": str(policy_meta.get("policy_note", "")),
            "portfolio_model_active": bool(policy_meta.get("portfolio_model_active", row.get("portfolio_model_active", False))),
            "portfolio_model_policy_class": str(policy_meta.get("portfolio_model_class", row.get("portfolio_model_class", ""))),
            "portfolio_model_policy_confidence": float(policy_meta.get("portfolio_model_confidence", row.get("portfolio_model_confidence", np.nan))),
            "portfolio_model_policy_source": str(policy_meta.get("portfolio_model_source", row.get("portfolio_model_source", ""))),
            "direction_strength_score": float(policy_meta.get("direction_strength_score", np.nan)),
            "up_strength_bonus": float(policy_meta.get("up_strength_bonus", 0.0)),
            "up_strength_score": float(policy_meta.get("up_strength_score", row.get("prob_up_strengthening_score", np.nan))),
            "up_strength_5d": float(policy_meta.get("up_strength_5d", row.get("prob_up_strengthening_5d", np.nan))),
            "up_strength_10d": float(policy_meta.get("up_strength_10d", row.get("prob_up_strengthening_10d", np.nan))),
            "up_strength_20d": float(policy_meta.get("up_strength_20d", row.get("prob_up_strengthening_20d", np.nan))),
            "bear_strength_cut": float(policy_meta.get("bear_strength_cut", 0.0)),
            "allocation_downrisk_score": float(policy_meta.get("allocation_downrisk_score", pdn)),
            "offensive_active": bool(policy_meta.get("offensive_active", False)),
            "offensive_tier": int(policy_meta.get("offensive_tier", 0)),
            "tier1_signal": bool(policy_meta.get("tier1_signal", False)),
            "tier2_signal": bool(policy_meta.get("tier2_signal", False)),
            "original_tier2_signal": bool(policy_meta.get("original_tier2_signal", False)),
            "tier3_signal": bool(policy_meta.get("tier3_signal", False)),
            "full_stock_signal": bool(policy_meta.get("full_stock_signal", False)),
            "short_mid_confirm": bool(policy_meta.get("short_mid_confirm", False)),
            "short_mid_strong_confirm": bool(policy_meta.get("short_mid_strong_confirm", False)),
            "short_mid_loose_confirm": bool(policy_meta.get("short_mid_loose_confirm", False)),
            "short_mid_all3_confirm": bool(policy_meta.get("short_mid_all3_confirm", False)),
            "strength_combo_5d_signal": bool(policy_meta.get("strength_combo_5d_signal", False)),
            "strength_combo_10d_signal": bool(policy_meta.get("strength_combo_10d_signal", False)),
            "strength_combo_20d_signal": bool(policy_meta.get("strength_combo_20d_signal", False)),
            "strength_combo_5d_10d_signal": bool(policy_meta.get("strength_combo_5d_10d_signal", False)),
            "strength_combo_5d_20d_signal": bool(policy_meta.get("strength_combo_5d_20d_signal", False)),
            "strength_combo_10d_20d_signal": bool(policy_meta.get("strength_combo_10d_20d_signal", False)),
            "strength_combo_all3_signal": bool(policy_meta.get("strength_combo_all3_signal", False)),
            "strength_combo_best_name": str(policy_meta.get("strength_combo_best_name", "NONE")),
            "strength_combo_best_stock_weight": float(policy_meta.get("strength_combo_best_stock_weight", 0.0)),
            "strength_combo_best_tier": int(policy_meta.get("strength_combo_best_tier", 0)),
            "strength_combo_policy_mode": str(policy_meta.get("strength_combo_policy_mode", getattr(cfg, "strength_combo_policy_mode", "max_weight"))),
            "strength_combo_policy_action": str(policy_meta.get("strength_combo_policy_action", "")),
            "strength_combo_hv_threshold": float(policy_meta.get("strength_combo_hv_threshold", getattr(cfg, "strength_combo_high_vol_threshold", 0.72))),
            "strength_combo_hv_ok": bool(policy_meta.get("strength_combo_hv_ok", False)),
            "strength_combo_score_ok": bool(policy_meta.get("strength_combo_score_ok", True)),
            "short_mid_policy_action": str(policy_meta.get("short_mid_policy_action", "")),
            "short_mid_mode": str(policy_meta.get("short_mid_mode", getattr(cfg, "short_mid_confirm_mode", "base_upgrade"))),
            "short_mid_action_signal": str(policy_meta.get("short_mid_action_signal", getattr(cfg, "short_mid_action_signal", "confirm"))),
            "short_mid_action_confirm": bool(policy_meta.get("short_mid_action_confirm", False)),
            "force_rebalance_signal": bool(policy_meta.get("force_rebalance", False)),
            "up_strength_pred_5d": bool(policy_meta.get("up_strength_pred_5d", False)),
            "up_strength_pred_10d": bool(policy_meta.get("up_strength_pred_10d", False)),
            "up_strength_pred_20d": bool(policy_meta.get("up_strength_pred_20d", False)),
            "up_strength_consensus_count": int(policy_meta.get("up_strength_consensus_count", 0)),
            "up_strength_consensus_pattern": str(policy_meta.get("up_strength_consensus_pattern", "")),
            "up_strength_consensus_weighted_score": float(policy_meta.get("up_strength_consensus_weighted_score", 0.0)),
            "up_strength_prob_spread": float(policy_meta.get("up_strength_prob_spread", 0.0)),
            "up_strength_consensus_target_stock": float(policy_meta.get("up_strength_consensus_target_stock", 0.0)),
            "up_strength_strong_all3": bool(policy_meta.get("up_strength_strong_all3", False)),
            "up_strength_five_day_only": bool(policy_meta.get("up_strength_five_day_only", False)),
            "base_signal_stock_weight": float(base_signal_w[0]),
            "base_signal_bond_weight": float(base_signal_w[1]),
            "base_signal_cash_weight": float(base_signal_w[2]),
            "signal_executed_stock_gap": float(w[0] - signal_w[0]),
            "abs_signal_executed_stock_gap": float(abs(w[0] - signal_w[0])),
            "executed_more_aggressive_than_signal": bool((w[0] - signal_w[0]) > 1e-12),
            "stale_offensive_hold": bool((w[0] - signal_w[0]) >= float(getattr(cfg, "stale_offensive_stock_gap_threshold", 0.055)) and float(row.get("prob_up_strengthening_score", row.get("prob_up_strengthening", 0.0))) < float(getattr(cfg, "stale_offensive_up_strength_reset_threshold", 0.20))),
        })
        rows.append(out)
        usage[current_g["name"]] = usage.get(current_g["name"], 0) + 1
        prev_w = w

    out_df = pd.DataFrame(rows)
    out_df["strategy_equity_net"] = cfg.initial_capital * (1.0 + out_df["strategy_return_net"]).cumprod()
    out_df["strategy_equity_gross"] = cfg.initial_capital * (1.0 + out_df["strategy_return_gross"]).cumprod()
    return out_df, usage


# ============================================================
# 7. METRICS / SUMMARY
# ============================================================

def binary_cls_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float, pos_name: str) -> Dict[str, object]:
    y_pred = (prob >= threshold).astype(int)
    out: Dict[str, object] = {
        "rows": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, np.clip(prob, 0.0, 1.0))),
        "support_positive": int(np.sum(y_true == 1)),
        "support_negative": int(np.sum(y_true == 0)),
        "pred_positive_ratio": float(np.mean(y_pred)),
        "positive_class": pos_name,
    }
    out["roc_auc"] = safe_auc(y_true, prob, "roc")
    out["pr_auc"] = safe_auc(y_true, prob, "pr")
    return out


def classification_metrics(pred_df: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    for h in cfg.horizons:
        if f"actual_risk_h{h}" in pred_df.columns and f"prob_high_vol_h{h}" in pred_df.columns:
            y = (pred_df[f"actual_risk_h{h}"] == "고변동").astype(int).values
            p = pred_df[f"prob_high_vol_h{h}"].astype(float).clip(0.0, 1.0).values
            metrics[f"stage1_h{h}"] = binary_cls_metrics(y, p, cfg.pred_high_vol_threshold, "고변동")
        if False and f"actual_direction_h{h}" in pred_df.columns and f"prob_up_h{h}" in pred_df.columns:
            y_up = (pred_df[f"actual_direction_h{h}"] == "상승").astype(int).values
            p_up = pred_df[f"prob_up_h{h}"].astype(float).clip(0.0, 1.0).values
            metrics[f"up_h{h}"] = binary_cls_metrics(y_up, p_up, 0.50, "상승")
        if False and f"actual_direction_h{h}" in pred_df.columns and f"prob_down_h{h}" in pred_df.columns:
            y_down = (pred_df[f"actual_direction_h{h}"] == "하락").astype(int).values
            p_down = pred_df[f"prob_down_h{h}"].astype(float).clip(0.0, 1.0).values
            metrics[f"down_h{h}"] = binary_cls_metrics(y_down, p_down, cfg.pred_down_risk_threshold, "하락")

    y_primary = (pred_df["actual_risk"] == "고변동").astype(int).values
    p_ens = pred_df["prob_high_vol"].astype(float).clip(0.0, 1.0).values
    metrics["stage1_ensemble_vs_primary"] = binary_cls_metrics(y_primary, p_ens, cfg.pred_high_vol_threshold, "고변동")

    y_up_primary = (pred_df["actual_direction"] == "상승").astype(int).values
    y_down_primary = (pred_df["actual_direction"] == "하락").astype(int).values
    # v8.6.34: prob_up/prob_down은 약한 출력으로 판정되어 summary 핵심 성능에서 제외한다.

    if "prob_overall_risk" in pred_df.columns:
        p_overall = pred_df["prob_overall_risk"].astype(float).clip(0.0, 1.0).values
        metrics["overall_risk_vs_highvol_primary"] = binary_cls_metrics(
            y_primary,
            p_overall,
            cfg.pred_overall_risk_threshold,
            "전체위험_by_고변동",
        )
        metrics["overall_risk_vs_down_primary"] = binary_cls_metrics(
            y_down_primary,
            p_overall,
            cfg.pred_overall_risk_threshold,
            "전체위험_by_하락",
        )

    for branch, col in []:
        if col in pred_df.columns:
            p_branch = pred_df[col].astype(float).clip(0.0, 1.0).values
            metrics[f"down_{branch}_vs_primary"] = binary_cls_metrics(
                y_down_primary,
                p_branch,
                cfg.pred_down_risk_threshold,
                f"하락_{branch}",
            )

    labels = ["상승", "중립", "하락"]
    if False and "actual_direction" in pred_df.columns and "pred_direction" in pred_df.columns:
        y_true = pd.Categorical(pred_df["actual_direction"], categories=labels).codes
        y_pred = pd.Categorical(pred_df["pred_direction"], categories=labels).codes
        valid = (y_true >= 0) & (y_pred >= 0)
        metrics["final_direction_3state_vs_primary"] = {
            "rows": int(valid.sum()),
            "accuracy": float(accuracy_score(y_true[valid], y_pred[valid])) if valid.any() else 0.0,
            "macro_f1": float(f1_score(y_true[valid], y_pred[valid], average="macro", zero_division=0)) if valid.any() else 0.0,
            "label_support": pred_df["actual_direction"].value_counts().to_dict(),
            "report": classification_report(y_true[valid], y_pred[valid], labels=[0, 1, 2], target_names=labels, output_dict=True, zero_division=0) if valid.any() else {},
        }
    if "actual_direction_strength" in pred_df.columns:
        actual_strength = pred_df["actual_direction_strength"].astype(str)
        if "prob_up_strengthening" in pred_df.columns:
            y_us = (actual_strength == "UP_STRENGTHENING").astype(int).values
            p_us = pred_df["prob_up_strengthening"].astype(float).clip(0.0, 1.0).values
            metrics["direction_strength_up_strengthening"] = binary_cls_metrics(
                y_us, p_us, float(getattr(cfg, "up_strength_bonus_threshold_1", 0.55)), "UP_STRENGTHENING"
            )
        # Multi-horizon UP_STRENGTHENING diagnostics.
        for sh in get_multi_strength_horizons(cfg):
            actual_col = f"actual_direction_strength_{int(sh)}d"
            prob_col = f"prob_up_strengthening_{int(sh)}d"
            if actual_col in pred_df.columns and prob_col in pred_df.columns:
                y_h = (pred_df[actual_col].astype(str) == "UP_STRENGTHENING").astype(int).values
                p_h = pred_df[prob_col].astype(float).clip(0.0, 1.0).values
                metrics[f"direction_strength_up_strengthening_{int(sh)}d"] = binary_cls_metrics(
                    y_h, p_h, float(getattr(cfg, "up_strength_bonus_threshold_1", 0.25)), f"UP_STRENGTHENING_{int(sh)}D"
                )
        if "prob_up_strengthening_score" in pred_df.columns and "actual_direction_strength_20d" in pred_df.columns:
            y_score = (pred_df["actual_direction_strength_20d"].astype(str) == "UP_STRENGTHENING").astype(int).values
            p_score = pred_df["prob_up_strengthening_score"].astype(float).clip(0.0, 1.0).values
            metrics["direction_strength_up_score_vs_20d"] = binary_cls_metrics(
                y_score, p_score, float(getattr(cfg, "up_strength_bonus_threshold_1", 0.25)), "UP_STRENGTHENING_SCORE"
            )
        # v8.6.34: prob_bear_down_strengthening은 allocation/summary 성능에서 제외한다.

    return metrics

def _pct_weight_dict(row: pd.Series, prefix: str) -> Dict[str, float]:
    return {
        "stock": round(float(row[f"{prefix}stock_weight"]) * 100, 2),
        "bond": round(float(row[f"{prefix}bond_weight"]) * 100, 2),
        "cash": round(float(row[f"{prefix}cash_weight"]) * 100, 2),
    }


def build_summary(pred_df: pd.DataFrame, feature_cols: List[str], gate_usage: Dict[str, int], cfg: Config) -> Dict[str, object]:
    perf = {
        "strategy_after_cost": perf_stats(pred_df["strategy_return_net"], cfg.initial_capital),
        "strategy_gross": perf_stats(pred_df["strategy_return_gross"], cfg.initial_capital),
        "stock_buy_hold": perf_stats(pred_df["stock_next_return"], cfg.initial_capital),
        "benchmark_60_40": perf_stats(0.6 * pred_df["stock_next_return"] + 0.4 * pred_df["bond_next_return"], cfg.initial_capital),
        "static_50_30_20": perf_stats(0.5 * pred_df["stock_next_return"] + 0.3 * pred_df["bond_next_return"] + 0.2 * pred_df["cash_next_return"], cfg.initial_capital),
    }
    latest = pred_df.iloc[-1]
    signal_alloc = {
        "stock": round(float(latest.get("signal_stock_weight", latest["stock_weight"])) * 100, 2),
        "bond": round(float(latest.get("signal_bond_weight", latest["bond_weight"])) * 100, 2),
        "cash": round(float(latest.get("signal_cash_weight", latest["cash_weight"])) * 100, 2),
    }
    executed_alloc = {
        "stock": round(float(latest["stock_weight"]) * 100, 2),
        "bond": round(float(latest["bond_weight"]) * 100, 2),
        "cash": round(float(latest["cash_weight"]) * 100, 2),
    }
    return {
        "model_type": "xgb_strength_combo_all_v8_6_34_diagnostics",
        "label_mode": "no_sideway_3class_strength",
        "label_classes": list(DIRECTION_STRENGTH_LABELS),
        "policy_mode": str(getattr(cfg, "policy_mode", "base")),
        "target_ticker": cfg.target_ticker,
        "bond_ticker": cfg.bond_ticker,
        "cash_ticker": cfg.cash_ticker,
        "config": asdict(cfg),
        "removed_weak_probability_outputs": ["prob_up", "prob_down", "prob_bear_down_strengthening"],
        "new_down_strength_probability_outputs": ["prob_down_strengthening_5d", "prob_down_strengthening_10d", "prob_down_strengthening_20d", "prob_down_strengthening_score"],
        "tier2_reenabled": bool(not getattr(cfg, "disable_tier2_signal", False)),
        "tier2_removed": bool(getattr(cfg, "disable_tier2_signal", True)),
        "portfolio_policy_model": portfolio_policy_summary(pred_df, cfg),
        "horizon_train_window": horizon_train_window_config(cfg),
        "period": {"start": str(pred_df["Date"].iloc[0]), "end": str(pred_df["Date"].iloc[-1]), "rows": int(len(pred_df))},
        "feature_count": int(len(feature_cols)),
        "feature_set": "pruned_directional_risk_features",
        "feature_cols": feature_cols,
        "feature_pruning_note": "v8.6.2 기준 저중요도/중복/불안정 피처를 모델 입력에서 제거한 축소 피처셋",
        "removed_low_value_features": REMOVED_LOW_VALUE_FEATURES_V8_6_5,
        "stage1_feature_importance_mean": pred_df.attrs.get("stage1_feature_importance_mean", {}),
        "up_feature_importance_mean": pred_df.attrs.get("up_feature_importance_mean", {}),
        "downrisk_feature_importance_mean": pred_df.attrs.get("downrisk_feature_importance_mean", {}),
        "downrisk_price_trend_feature_importance_mean": pred_df.attrs.get("downrisk_price_trend_feature_importance_mean", {}),
        "downrisk_price_volume_feature_importance_mean": pred_df.attrs.get("downrisk_price_volume_feature_importance_mean", {}),
        "downrisk_volatility_feature_importance_mean": pred_df.attrs.get("downrisk_volatility_feature_importance_mean", {}),
        "downrisk_branch_weights": pred_df.attrs.get("downrisk_branch_weights", {}),
        "downrisk_feature_sets": pred_df.attrs.get("downrisk_feature_sets", {}),
        "direction_feature_set": pred_df.attrs.get("direction_feature_set", []),
        "label_policy_usage": pred_df.attrs.get("policy_usage", {}),
        "average_probabilities": {
            "avg_prob_normal": float(pred_df["prob_normal"].mean()),
            "avg_prob_high_vol": float(pred_df["prob_high_vol"].mean()),
            "avg_prob_up_strengthening_score": float(pred_df["prob_up_strengthening_score"].mean()) if "prob_up_strengthening_score" in pred_df.columns else 0.0,
            "avg_prob_up_strengthening_5d": float(pred_df["prob_up_strengthening_5d"].mean()) if "prob_up_strengthening_5d" in pred_df.columns else 0.0,
            "avg_prob_up_strengthening_10d": float(pred_df["prob_up_strengthening_10d"].mean()) if "prob_up_strengthening_10d" in pred_df.columns else 0.0,
            "avg_prob_up_strengthening_20d": float(pred_df["prob_up_strengthening_20d"].mean()) if "prob_up_strengthening_20d" in pred_df.columns else 0.0,
            "avg_prob_down_strengthening_score": float(pred_df["prob_down_strengthening_score"].mean()) if "prob_down_strengthening_score" in pred_df.columns else 0.0,
            "avg_prob_down_strengthening_5d": float(pred_df["prob_down_strengthening_5d"].mean()) if "prob_down_strengthening_5d" in pred_df.columns else 0.0,
            "avg_prob_down_strengthening_10d": float(pred_df["prob_down_strengthening_10d"].mean()) if "prob_down_strengthening_10d" in pred_df.columns else 0.0,
            "avg_prob_down_strengthening_20d": float(pred_df["prob_down_strengthening_20d"].mean()) if "prob_down_strengthening_20d" in pred_df.columns else 0.0,
            "avg_prob_overall_risk": float(pred_df["prob_overall_risk"].mean()) if "prob_overall_risk" in pred_df.columns else float(pred_df["prob_high_vol"].mean()),
        },
        "average_weights": {
            "avg_stock_weight": float(pred_df["stock_weight"].mean()),
            "avg_bond_weight": float(pred_df["bond_weight"].mean()),
            "avg_cash_weight": float(pred_df["cash_weight"].mean()),
            "min_stock_weight": float(pred_df["stock_weight"].min()),
            "max_stock_weight": float(pred_df["stock_weight"].max()),
        },
        "allocation_regime_distribution_pct": pred_df["allocation_regime"].value_counts(normalize=True).mul(100).round(2).to_dict(),
        "signal_regime_distribution_pct": pred_df["signal_regime"].value_counts(normalize=True).mul(100).round(2).to_dict() if "signal_regime" in pred_df.columns else {},
        "turnover": {
            "avg_daily_trade_ratio": float(pred_df["turnover"].mean()),
            "annual_turnover_estimate": float(pred_df["turnover"].mean() * 252.0),
            "total_transaction_cost_rate_sum": float(pred_df["transaction_cost"].sum()),
            "rebalance_due_ratio": float(pred_df["rebalance_due"].mean()) if "rebalance_due" in pred_df.columns else float(pred_df["rebalanced"].mean()),
            "trade_executed_ratio": float(pred_df["trade_executed"].mean()) if "trade_executed" in pred_df.columns else float((pred_df["turnover"] > 1e-12).mean()),
            "rebalance_ratio": float(pred_df["rebalanced"].mean()),
            "emergency_rebalance_ratio": float(pred_df["emergency_rebalance"].mean()),
        },
        "performance": perf,
        "classification": classification_metrics(pred_df, cfg),
        "gate_config_usage_top10": dict(sorted(gate_usage.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "direction_strength_specialist": {
            "enabled": bool(getattr(cfg, "use_direction_strength_specialist", True)),
            "horizon": int(getattr(cfg, "direction_strength_horizon", 20)),
            "multi_strength_horizons": list(get_multi_strength_horizons(cfg)),
            "up_strength_horizon_weights": get_up_strength_horizon_weights(cfg),
            "down_strength_horizon_weights": get_down_strength_horizon_weights(cfg),
            "ret_eps_k": float(getattr(cfg, "direction_strength_ret_eps_k", 0.20)),
            "strength_method": str(getattr(cfg, "direction_strength_method", "score_delta")),
            "upside_train_filter": str(getattr(cfg, "upside_strength_train_filter", "major_only")),
            "bear_train_filter": str(getattr(cfg, "bear_strength_train_filter", "bear_stress")),
            "feature_set": str(getattr(cfg, "direction_strength_feature_set", "compact_mixed")),
            "avg_up_strength_bonus": float(pred_df.get("up_strength_bonus", pd.Series(0.0, index=pred_df.index)).mean()),
            "avg_up_strength_score": float(pred_df.get("prob_up_strengthening_score", pd.Series(0.0, index=pred_df.index)).astype(float).mean()),
            "avg_up_strength_5d": float(pred_df.get("prob_up_strengthening_5d", pd.Series(0.0, index=pred_df.index)).astype(float).mean()),
            "avg_up_strength_10d": float(pred_df.get("prob_up_strengthening_10d", pd.Series(0.0, index=pred_df.index)).astype(float).mean()),
            "avg_up_strength_20d": float(pred_df.get("prob_up_strengthening_20d", pd.Series(0.0, index=pred_df.index)).astype(float).mean()),
            "avg_down_strength_score": float(pred_df.get("prob_down_strengthening_score", pd.Series(0.0, index=pred_df.index)).astype(float).mean()),
            "avg_down_strength_5d": float(pred_df.get("prob_down_strengthening_5d", pd.Series(0.0, index=pred_df.index)).astype(float).mean()),
            "avg_down_strength_10d": float(pred_df.get("prob_down_strengthening_10d", pd.Series(0.0, index=pred_df.index)).astype(float).mean()),
            "avg_down_strength_20d": float(pred_df.get("prob_down_strengthening_20d", pd.Series(0.0, index=pred_df.index)).astype(float).mean()),
            "avg_bear_strength_cut": float(pred_df.get("bear_strength_cut", pd.Series(0.0, index=pred_df.index)).mean()),
            "up_bonus_activation_rate": float((pred_df.get("up_strength_bonus", pd.Series(0.0, index=pred_df.index)) > 1e-12).mean()),
            "bear_cut_activation_rate": float((pred_df.get("bear_strength_cut", pd.Series(0.0, index=pred_df.index)) > 1e-12).mean()),
            "offensive_activation_rate": float((pred_df.get("offensive_active", pd.Series(False, index=pred_df.index)).astype(bool)).mean()),
            "offensive_tier_1plus_rate": float((pred_df.get("offensive_tier", pd.Series(0, index=pred_df.index)).fillna(0).astype(int) >= 1).mean()),
            "offensive_tier_2plus_rate": float((pred_df.get("offensive_tier", pd.Series(0, index=pred_df.index)).fillna(0).astype(int) >= 2).mean()),
            "offensive_tier_3_rate": float((pred_df.get("offensive_tier", pd.Series(0, index=pred_df.index)).fillna(0).astype(int) >= 3).mean()),
            "strong_all3_rate": float(pred_df.get("up_strength_strong_all3", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "five_day_only_rate": float(pred_df.get("up_strength_five_day_only", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "short_mid_confirm_mode": str(getattr(cfg, "short_mid_confirm_mode", "base_upgrade")),
            "short_mid_action_signal": str(getattr(cfg, "short_mid_action_signal", "confirm")),
            "short_mid_action_confirm_rate": float(pred_df.get("short_mid_action_confirm", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "short_mid_confirm_rate": float(pred_df.get("short_mid_confirm", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "short_mid_strong_confirm_rate": float(pred_df.get("short_mid_strong_confirm", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "short_mid_loose_confirm_rate": float(pred_df.get("short_mid_loose_confirm", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "short_mid_all3_confirm_rate": float(pred_df.get("short_mid_all3_confirm", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "strength_combo_policy_enabled": bool(getattr(cfg, "strength_combo_policy_enabled", True)),
            "strength_combo_policy_mode": str(getattr(cfg, "strength_combo_policy_mode", "max_weight")),
            "strength_combo_5d_rate": float(pred_df.get("strength_combo_5d_signal", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "strength_combo_10d_rate": float(pred_df.get("strength_combo_10d_signal", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "strength_combo_20d_rate": float(pred_df.get("strength_combo_20d_signal", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "strength_combo_5d_10d_rate": float(pred_df.get("strength_combo_5d_10d_signal", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "strength_combo_5d_20d_rate": float(pred_df.get("strength_combo_5d_20d_signal", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "strength_combo_10d_20d_rate": float(pred_df.get("strength_combo_10d_20d_signal", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "strength_combo_all3_rate": float(pred_df.get("strength_combo_all3_signal", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "strength_combo_best_distribution_pct": pred_df.get("strength_combo_best_name", pd.Series("NONE", index=pred_df.index)).astype(str).value_counts(normalize=True).mul(100).round(2).to_dict(),
            "strength_combo_policy_action_distribution_pct": pred_df.get("strength_combo_policy_action", pd.Series("", index=pred_df.index)).astype(str).value_counts(normalize=True).mul(100).round(2).to_dict(),
            "short_mid_policy_action_distribution_pct": pred_df.get("short_mid_policy_action", pd.Series("", index=pred_df.index)).astype(str).value_counts(normalize=True).mul(100).round(2).to_dict(),
            "avg_consensus_count": float(pred_df.get("up_strength_consensus_count", pd.Series(0, index=pred_df.index)).fillna(0).astype(float).mean()),
            "p20_tier_1plus_rate": float((pred_df.get("p20_tier", pd.Series(0, index=pred_df.index)).fillna(0).astype(int) >= 1).mean()),
            "p20_tier_2plus_rate": float((pred_df.get("p20_tier", pd.Series(0, index=pred_df.index)).fillna(0).astype(int) >= 2).mean()),
            "p20_tier_3_rate": float((pred_df.get("p20_tier", pd.Series(0, index=pred_df.index)).fillna(0).astype(int) >= 3).mean()),
            "avg_strength_train_rows_20d": float(pred_df.get("strength_train_rows_20d", pd.Series(0, index=pred_df.index)).astype(float).mean()) if "strength_train_rows_20d" in pred_df.columns else 0.0,
            "avg_strength_train_ratio_20d": float(pred_df.get("strength_train_horizon_ratio_20d", pd.Series(0, index=pred_df.index)).astype(float).mean()) if "strength_train_horizon_ratio_20d" in pred_df.columns else 0.0,
            "allocation_downrisk_weight": float(getattr(cfg, "allocation_downrisk_weight", 0.0)),
            "use_bear_specialist_cut": bool(getattr(cfg, "use_bear_specialist_cut", False)),
            "avg_allocation_downrisk_score": float(pred_df.get("allocation_downrisk_score", pd.Series(np.nan, index=pred_df.index)).astype(float).mean()),
        },
        "target_execution_diagnostics": {
            "avg_signal_executed_stock_gap": float(pred_df.get("signal_executed_stock_gap", pd.Series(0.0, index=pred_df.index)).astype(float).mean()),
            "avg_abs_signal_executed_stock_gap": float(pred_df.get("abs_signal_executed_stock_gap", pd.Series(0.0, index=pred_df.index)).astype(float).mean()),
            "gap_gt_3pct_rate": float((pred_df.get("abs_signal_executed_stock_gap", pd.Series(0.0, index=pred_df.index)).astype(float) >= 0.03).mean()),
            "gap_gt_5pct_rate": float((pred_df.get("abs_signal_executed_stock_gap", pd.Series(0.0, index=pred_df.index)).astype(float) >= 0.05).mean()),
            "executed_more_aggressive_rate": float(pred_df.get("executed_more_aggressive_than_signal", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "stale_offensive_hold_rate": float(pred_df.get("stale_offensive_hold", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "strong_offensive_override_rate": float(pred_df.get("strong_offensive_override", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "full_stock_signal_rate": float(pred_df.get("full_stock_signal", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "tier3_signal_rate": float(pred_df.get("tier3_signal", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "base_vol_probability_enabled": bool(getattr(cfg, "use_vol_probability_base_allocation", True)),
            "stale_offensive_decay_enabled": bool(getattr(cfg, "enable_stale_offensive_decay", False)),
        },
        "latest_prediction": {
            "date": str(latest["Date"]),
            "pred_risk": str(latest["pred_risk"]),
            "pred_direction": str(latest.get("pred_direction", "중립")),
            "pred_overall_risk": str(latest.get("pred_overall_risk", "정상")),
            "prob_normal": round(float(latest["prob_normal"]) * 100, 2),
            "prob_high_vol": round(float(latest["prob_high_vol"]) * 100, 2),
            "prob_up_strengthening": round(float(latest.get("prob_up_strengthening", 0.0)) * 100, 2),
            "prob_up_strengthening_score": round(float(latest.get("prob_up_strengthening_score", 0.0)) * 100, 2),
            "prob_up_strengthening_5d": round(float(latest.get("prob_up_strengthening_5d", 0.0)) * 100, 2),
            "prob_up_strengthening_10d": round(float(latest.get("prob_up_strengthening_10d", 0.0)) * 100, 2),
            "prob_up_strengthening_20d": round(float(latest.get("prob_up_strengthening_20d", latest.get("prob_up_strengthening", 0.0))) * 100, 2),
            "prob_down_strengthening": round(float(latest.get("prob_down_strengthening", 0.0)) * 100, 2),
            "prob_down_strengthening_score": round(float(latest.get("prob_down_strengthening_score", 0.0)) * 100, 2),
            "prob_down_strengthening_5d": round(float(latest.get("prob_down_strengthening_5d", 0.0)) * 100, 2),
            "prob_down_strengthening_10d": round(float(latest.get("prob_down_strengthening_10d", 0.0)) * 100, 2),
            "prob_down_strengthening_20d": round(float(latest.get("prob_down_strengthening_20d", latest.get("prob_down_strengthening", 0.0))) * 100, 2),
            "prob_up_weakening": 0.0,
            "actual_direction_strength": str(latest.get("actual_direction_strength", "")),
            "prob_overall_risk": round(float(latest.get("prob_overall_risk", 0.0)) * 100, 2),
            "policy_mode": str(latest.get("policy_mode", getattr(cfg, "policy_mode", "base"))),
            "removed_weak_probability_outputs": ["prob_up", "prob_down", "prob_bear_down_strengthening"],
            "portfolio_model_enabled": bool(getattr(cfg, "enable_portfolio_policy_model", False)),
            "portfolio_model_class": str(latest.get("portfolio_model_class", "")),
            "portfolio_model_confidence": round(float(latest.get("portfolio_model_confidence", 0.0)) * 100, 2) if pd.notna(latest.get("portfolio_model_confidence", np.nan)) else None,
            "portfolio_model_source": str(latest.get("portfolio_model_source", "")),
            "portfolio_model_allocation": {
                "stock": round(float(latest.get("portfolio_model_stock_weight", 0.0)) * 100, 2) if pd.notna(latest.get("portfolio_model_stock_weight", np.nan)) else None,
                "bond": round(float(latest.get("portfolio_model_bond_weight", 0.0)) * 100, 2) if pd.notna(latest.get("portfolio_model_bond_weight", np.nan)) else None,
                "cash": round(float(latest.get("portfolio_model_cash_weight", 0.0)) * 100, 2) if pd.notna(latest.get("portfolio_model_cash_weight", np.nan)) else None,
            },
            "policy_overlay": round(float(latest.get("policy_overlay", 0.0)) * 100, 2),
            "direction_strength_score": round(float(latest.get("direction_strength_score", 0.0)) * 100, 2) if pd.notna(latest.get("direction_strength_score", np.nan)) else None,
            "up_strength_score": round(float(latest.get("up_strength_score", latest.get("prob_up_strengthening_score", 0.0))) * 100, 2) if pd.notna(latest.get("prob_up_strengthening_score", np.nan)) else None,
            "up_strength_bonus": round(float(latest.get("up_strength_bonus", 0.0)) * 100, 2),
            "bear_strength_cut": round(float(latest.get("bear_strength_cut", 0.0)) * 100, 2),
            "mid_trend_score": int(latest.get("mid_trend_score", 0)),
            "mid_trend_state": str(latest.get("mid_trend_state", "NEUTRAL")),
            "defensive_risk_score": round(float(latest.get("defensive_risk_score", 0.0)) * 100, 2) if pd.notna(latest.get("defensive_risk_score", np.nan)) else None,
            "signal_regime": str(latest.get("signal_regime", latest["allocation_regime"])),
            "allocation_regime": str(latest["allocation_regime"]),
            "executed_regime": str(latest.get("executed_regime", latest["allocation_regime"])),
            "hold_reason": str(latest.get("hold_reason", "unknown")),
            "held_by_no_trade_band": bool(latest.get("held_by_no_trade_band", False)),
            "held_by_schedule": bool(latest.get("held_by_schedule", False)),
            "signal_executed_stock_gap": round(float(latest.get("signal_executed_stock_gap", 0.0)) * 100, 2),
            "stale_offensive_hold": bool(latest.get("stale_offensive_hold", False)),
            "offensive_tier": int(latest.get("offensive_tier", 0)) if pd.notna(latest.get("offensive_tier", np.nan)) else 0,
            "base_allocation_mode": str(latest.get("base_allocation_mode", "")),
            "base_signal_stock_weight": round(float(latest.get("base_signal_stock_weight", 0.0)) * 100, 2),
            "strong_offensive_override": bool(latest.get("strong_offensive_override", False)),
            "tier1_signal": bool(latest.get("tier1_signal", False)),
            "tier2_signal": bool(latest.get("tier2_signal", False)),
            "original_tier2_signal": bool(latest.get("original_tier2_signal", False)),
            "tier3_signal": bool(latest.get("tier3_signal", False)),
            "full_stock_signal": bool(latest.get("full_stock_signal", False)),
            "short_mid_confirm": bool(latest.get("short_mid_confirm", False)),
            "short_mid_strong_confirm": bool(latest.get("short_mid_strong_confirm", False)),
            "short_mid_loose_confirm": bool(latest.get("short_mid_loose_confirm", False)),
            "short_mid_all3_confirm": bool(latest.get("short_mid_all3_confirm", False)),
            "short_mid_policy_action": str(latest.get("short_mid_policy_action", "")),
            "short_mid_action_signal": str(latest.get("short_mid_action_signal", getattr(cfg, "short_mid_action_signal", "confirm"))),
            "short_mid_action_confirm": bool(latest.get("short_mid_action_confirm", False)),
            "up_strength_pred_5d": bool(latest.get("up_strength_pred_5d", False)),
            "up_strength_pred_10d": bool(latest.get("up_strength_pred_10d", False)),
            "up_strength_pred_20d": bool(latest.get("up_strength_pred_20d", False)),
            "up_strength_consensus_count": int(latest.get("up_strength_consensus_count", 0)) if pd.notna(latest.get("up_strength_consensus_count", np.nan)) else 0,
            "up_strength_consensus_pattern": str(latest.get("up_strength_consensus_pattern", "")),
            "up_strength_consensus_target_stock": round(float(latest.get("up_strength_consensus_target_stock", 0.0)) * 100, 2),
            "up_strength_strong_all3": bool(latest.get("up_strength_strong_all3", False)),
            "up_strength_five_day_only": bool(latest.get("up_strength_five_day_only", False)),
            "signal_allocation": signal_alloc,
            "target_allocation": signal_alloc,
            "executed_allocation": executed_alloc,
        },
    }


def add_condition_period_summary(
    summary: Dict[str, object],
    pred_df: pd.DataFrame,
    cfg: Config,
    split_date: str,
) -> None:
    """
    condition search를 사용했을 때 조건 선택 구간과 holdout 구간의 성과를 분리 저장한다.

    이유:
    - 최종 전체 구간 성과만 보면 조건 선택에 사용된 구간과 사후 검증 구간이 섞인다.
    - holdout 성과를 별도 저장해야 과최적화 여부를 확인할 수 있다.
    """
    if pred_df.empty or "Date" not in pred_df.columns:
        summary["condition_period_performance"] = {"error": "pred_df is empty or Date column is missing"}
        return

    dates = pd.to_datetime(pred_df["Date"])
    split_ts = pd.Timestamp(split_date)
    select_df = pred_df[dates <= split_ts].copy()
    holdout_df = pred_df[dates > split_ts].copy()

    def _period_block(df_part: pd.DataFrame) -> Dict[str, object]:
        if df_part.empty:
            return {
                "start": None,
                "end": None,
                "rows": 0,
                "strategy_after_cost": {},
                "stock_buy_hold": {},
                "benchmark_60_40": {},
                "static_50_30_20": {},
            }
        return {
            "start": str(df_part["Date"].iloc[0]),
            "end": str(df_part["Date"].iloc[-1]),
            "rows": int(len(df_part)),
            "strategy_after_cost": perf_stats(df_part["strategy_return_net"], cfg.initial_capital),
            "stock_buy_hold": perf_stats(df_part["stock_next_return"], cfg.initial_capital),
            "benchmark_60_40": perf_stats(
                0.6 * df_part["stock_next_return"] + 0.4 * df_part["bond_next_return"],
                cfg.initial_capital,
            ),
            "static_50_30_20": perf_stats(
                0.5 * df_part["stock_next_return"]
                + 0.3 * df_part["bond_next_return"]
                + 0.2 * df_part["cash_next_return"],
                cfg.initial_capital,
            ),
        }

    summary["condition_period_performance"] = {
        "split_date": split_date,
        "select_period": _period_block(select_df),
        "holdout_period": _period_block(holdout_df),
    }

def print_summary(summary: Dict[str, object]) -> None:
    p = summary["performance"]
    w = summary["average_weights"]
    t = summary["turnover"]
    cls = summary["classification"]
    print("\n==============================")
    print("XGBoost v8.6.23 No-Sideway Multi-Horizon Train-Window Ratio Diagnostics 결과 요약")
    print("Stage1 + Direction-Strength Specialist Allocation")
    print("==============================")
    print(f"기간: {summary['period']['start']} ~ {summary['period']['end']}")
    print(f"거래일 수: {summary['period']['rows']}")
    print(f"피처 수: {summary['feature_count']}")
    print(f"평균 주식 비중: {w['avg_stock_weight'] * 100:.2f}%")
    print(f"평균 채권 비중: {w['avg_bond_weight'] * 100:.2f}%")
    print(f"평균 현금 비중: {w['avg_cash_weight'] * 100:.2f}%")
    print(f"연간 교체율 추정: {t['annual_turnover_estimate'] * 100:.2f}%")
    print(f"리밸런싱 도래 비율: {t['rebalance_ratio'] * 100:.2f}%")
    if 'trade_executed_ratio' in t:
        print(f"실제 거래 발생 비율: {t['trade_executed_ratio'] * 100:.2f}%")
    print(f"긴급 리밸런싱 비율: {t['emergency_rebalance_ratio'] * 100:.2f}%")
    print(f"배분 regime 분포: {summary['allocation_regime_distribution_pct']}")

    for name in ["strategy_after_cost", "strategy_gross", "stock_buy_hold", "benchmark_60_40", "static_50_30_20"]:
        st = p[name]
        print(f"\n[{name}]")
        print(f"최종 자산: {st['final_capital']:,.0f}")
        print(f"총수익률: {st['total_return'] * 100:.2f}%")
        print(f"CAGR: {st['cagr'] * 100:.2f}%")
        print(f"MDD: {st['mdd'] * 100:.2f}%")
        print(f"Sharpe: {st['sharpe']:.4f}")
        print(f"Sortino: {st['sortino']:.6f}")
        print(f"Calmar: {st['calmar']:.6f}")

    print("\n[분류 성능 핵심]")
    for key in [
        "stage1_h10", "stage1_h20", "stage1_ensemble_vs_primary",
        "up_h10", "up_h20", "up_ensemble_vs_primary",
        "down_h10", "down_h20", "down_ensemble_vs_primary",
        "overall_risk_vs_highvol_primary", "overall_risk_vs_down_primary",
        "down_price_trend_vs_primary", "down_price_volume_vs_primary", "down_volatility_vs_primary",
        "direction_strength_up_strengthening", "direction_strength_bear_down_strengthening",
    ]:
        if key in cls:
            m = cls[key]
            print(f"{key:30s} | ROC {m['roc_auc']} | PR {m['pr_auc']} | F1 {m['f1']:.4f} | Recall {m['recall']:.4f}")

    print("\n[최신 예측]")
    print(json.dumps(summary["latest_prediction"], ensure_ascii=False, indent=2))



# ============================================================
# 8. OBJECTIVE CONDITION SEARCH
# ============================================================


def _slice_by_date(df: pd.DataFrame, end_date: Optional[str] = None, start_date: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["Date"])
    if start_date is not None:
        out = out[dates >= pd.Timestamp(start_date)]
        dates = pd.to_datetime(out["Date"])
    if end_date is not None:
        out = out[dates <= pd.Timestamp(end_date)]
    return out.copy()


def allocated_subset_stats(df: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    """성과 지표 + 조건 최적화용 운용 지표를 함께 계산한다."""
    empty = {
        "final_capital": cfg.initial_capital,
        "total_return": 0.0,
        "cagr": 0.0,
        "mdd": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "gross_cagr": 0.0,
        "cost_cagr_drag": 0.0,
        "annual_turnover": 0.0,
        "total_turnover": 0.0,
        "avg_daily_turnover": 0.0,
        "trade_day_ratio": 0.0,
        "avg_trade_size_on_trade": 0.0,
        "max_trade_size": 0.0,
        "total_transaction_cost_rate_sum": 0.0,
        "avg_stock_weight": 0.0,
        "min_stock_weight": 0.0,
        "max_stock_weight": 0.0,
        "std_stock_weight": 0.0,
        "avg_bond_weight": 0.0,
        "avg_cash_weight": 0.0,
        "rebalance_ratio": 0.0,
        "emergency_rebalance_ratio": 0.0,
        "regime_switch_ratio": 0.0,
        "normal_pct": 0.0,
        "watch_pct": 0.0,
        "high_vol_pct": 0.0,
        "risk_off_pct": 0.0,
        "extreme_risk_pct": 0.0,
        "actual_high_vol_rate": 0.0,
        "actual_down_high_vol_rate": 0.0,
        "avg_prob_high_vol": 0.0,
        "avg_prob_down_risk": 0.0,
    }
    if df.empty:
        return empty

    out = perf_stats(df["strategy_return_net"], cfg.initial_capital)
    gross = perf_stats(df["strategy_return_gross"], cfg.initial_capital) if "strategy_return_gross" in df.columns else {}
    out["gross_cagr"] = float(gross.get("cagr", out.get("cagr", 0.0)))
    out["cost_cagr_drag"] = float(out["gross_cagr"] - out.get("cagr", 0.0))

    turnover = df["turnover"].astype(float) if "turnover" in df.columns else pd.Series(index=df.index, data=0.0)
    trade_mask = turnover > 1e-12
    out["annual_turnover"] = float(turnover.mean() * 252.0)
    out["total_turnover"] = float(turnover.sum())
    out["avg_daily_turnover"] = float(turnover.mean())
    out["trade_day_ratio"] = float(trade_mask.mean())
    out["avg_trade_size_on_trade"] = float(turnover[trade_mask].mean()) if trade_mask.any() else 0.0
    out["max_trade_size"] = float(turnover.max()) if len(turnover) else 0.0
    out["total_transaction_cost_rate_sum"] = float(df.get("transaction_cost", pd.Series(index=df.index, data=0.0)).astype(float).sum())

    for col, key in [("stock_weight", "stock"), ("bond_weight", "bond"), ("cash_weight", "cash")]:
        if col in df.columns:
            s = df[col].astype(float)
            out[f"avg_{key}_weight"] = float(s.mean())
            out[f"min_{key}_weight"] = float(s.min())
            out[f"max_{key}_weight"] = float(s.max())
            out[f"std_{key}_weight"] = float(s.std(ddof=0))

    out["rebalance_ratio"] = float(df["rebalanced"].mean()) if "rebalanced" in df.columns else 0.0
    out["emergency_rebalance_ratio"] = float(df["emergency_rebalance"].mean()) if "emergency_rebalance" in df.columns else 0.0
    out["regime_switch_ratio"] = float(df["allocation_regime"].ne(df["allocation_regime"].shift(1)).mean()) if "allocation_regime" in df.columns else 0.0

    if "allocation_regime" in df.columns:
        dist = df["allocation_regime"].value_counts(normalize=True)
        out["normal_pct"] = float(dist.get("NORMAL", 0.0))
        out["watch_pct"] = float(dist.get("WATCH", 0.0))
        out["high_vol_pct"] = float(dist.get("HIGH_VOL", 0.0))
        out["risk_off_pct"] = float(dist.get("RISK_OFF", 0.0))
        out["extreme_risk_pct"] = float(dist.get("EXTREME_RISK", 0.0))

    if "actual_risk" in df.columns:
        out["actual_high_vol_rate"] = float((df["actual_risk"] == "고변동").mean())
    if "actual_split_vol" in df.columns:
        out["actual_down_high_vol_rate"] = float((df["actual_split_vol"] == "하락고변동").mean())
    if "prob_high_vol" in df.columns:
        out["avg_prob_high_vol"] = float(df["prob_high_vol"].astype(float).mean())
    if "prob_down_risk" in df.columns:
        out["avg_prob_down_risk"] = float(df["prob_down_risk"].astype(float).mean())

    return {**empty, **out}


def objective_condition_score(stats: Dict[str, float], score_profile: str = "balanced") -> float:
    cagr = float(stats.get("cagr", 0.0))
    mdd = abs(float(stats.get("mdd", 0.0)))
    sharpe = float(stats.get("sharpe", 0.0))
    calmar = float(stats.get("calmar", 0.0))
    annual_turnover = float(stats.get("annual_turnover", 0.0))

    if score_profile == "cagr":
        return 1.35 * cagr + 0.04 * sharpe - 0.50 * mdd - 0.040 * annual_turnover
    if score_profile == "calmar":
        return 0.70 * cagr + 0.10 * sharpe + 0.35 * calmar - 0.90 * mdd - 0.060 * annual_turnover
    if score_profile == "turnover":
        return 0.80 * cagr + 0.08 * sharpe + 0.12 * calmar - 0.75 * mdd - 0.120 * annual_turnover

    # v8.6.2 balanced: turnover 과다 문제를 줄이기 위해 v8.6.1보다 페널티를 강화한다.
    return 1.00 * cagr + 0.08 * sharpe + 0.14 * calmar - 0.78 * mdd - 0.085 * annual_turnover


def _risk_off_bond_cash_from_stock(stock: float) -> Tuple[float, float]:
    # RISK_OFF에서 방어자산을 IEF:BIL = 2:1 안팎으로 배분
    remain = max(0.0, 1.0 - stock)
    bond = remain * 0.65
    cash = remain * 0.35
    return float(bond), float(cash)


def make_condition_candidate_configs(base_cfg: Config, grid_size: str = "standard") -> List[Tuple[str, Config]]:
    """
    조건을 감으로 하나만 선택하지 않고, 사전에 정의한 후보군을 validation 구간에서 비교한다.
    모델 예측값은 그대로 두고 allocation 조건만 비교하므로 전체 재학습보다 훨씬 빠르다.
    """
    if grid_size == "compact":
        no_trade_list = [0.15, 0.17]
        cont_list = [False]
        riskoff_stock_list = [0.64, 0.68]
        threshold_pairs = [(0.72, 0.72), (0.74, 0.74)]
    elif grid_size == "wide":
        no_trade_list = [0.13, 0.15, 0.17, 0.19]
        cont_list = [False, True]
        riskoff_stock_list = [0.62, 0.66, 0.68, 0.70]
        threshold_pairs = [(0.70, 0.70), (0.72, 0.72), (0.74, 0.74), (0.76, 0.76)]
    else:
        # v8.6.21 standard: v8.6.10의 과도한 방어/turnover를 줄이는 공격형 후보군
        no_trade_list = [0.15, 0.17, 0.19]
        cont_list = [False]
        riskoff_stock_list = [0.64, 0.68, 0.70]
        threshold_pairs = [(0.72, 0.72), (0.74, 0.74), (0.76, 0.76)]

    out: List[Tuple[str, Config]] = []
    idx = 0
    for ntb in no_trade_list:
        for cont in cont_list:
            for ro_stock in riskoff_stock_list:
                ro_bond, ro_cash = _risk_off_bond_cash_from_stock(ro_stock)
                for hv_th, dn_th in threshold_pairs:
                    c = replace(base_cfg)
                    c.no_trade_band = float(ntb)
                    c.use_continuous_adjustment = bool(cont)
                    c.risk_off_stock_weight = float(ro_stock)
                    c.risk_off_bond_weight = float(ro_bond)
                    c.risk_off_cash_weight = float(ro_cash)
                    c.gate_high_vol_threshold = float(hv_th)
                    c.gate_riskoff_downrisk_threshold = float(dn_th)
                    c.gate_watch_downrisk_threshold = max(0.60, float(dn_th) + 0.12)
                    c.use_three_regime_allocation = True
                    c.use_extreme_risk_cut = True
                    c.extreme_high_vol_threshold = 0.86
                    c.extreme_downrisk_threshold = 0.86
                    c.extreme_stock_weight = 0.60
                    c.extreme_bond_weight, c.extreme_cash_weight = _risk_off_bond_cash_from_stock(c.extreme_stock_weight)
                    c.result_dir = base_cfg.result_dir
                    name = (
                        f"c{idx:03d}_ntb{ntb:.2f}_cont{int(cont)}_"
                        f"ro{ro_stock:.2f}_hv{hv_th:.2f}_dn{dn_th:.2f}"
                    )
                    out.append((name, c))
                    idx += 1
    return out


def run_condition_search(
    pred_raw: pd.DataFrame,
    base_cfg: Config,
    split_date: str,
    grid_size: str = "standard",
    score_profile: str = "balanced",
) -> Tuple[Config, pd.DataFrame, Dict[str, object]]:
    """
    objective condition search.

    선택 구간: Date <= split_date
    보류/검증 구간: Date > split_date

    주의:
    - 이 함수는 모델을 다시 학습하지 않는다.
    - walk-forward 예측 확률은 이미 각 시점의 과거 데이터만 사용해 만들어진 값이다.
    - 조건 선택은 split_date 이전 구간에서만 수행하고, split_date 이후 성과는 선택 후 확인용으로만 남긴다.
    """
    candidates = make_condition_candidate_configs(base_cfg, grid_size=grid_size)
    rows: List[Dict[str, object]] = []
    best_score = -np.inf
    best_cfg = base_cfg
    best_name = "base"

    for name, cand_cfg in candidates:
        allocated, _usage = apply_allocation(pred_raw, cand_cfg)
        select_df = _slice_by_date(allocated, end_date=split_date)
        holdout_df = _slice_by_date(allocated, start_date=split_date)
        # start_date는 split_date 포함이므로 중복을 피하기 위해 하루 단위 조건 재조정
        holdout_df = holdout_df[pd.to_datetime(holdout_df["Date"]) > pd.Timestamp(split_date)].copy()
        full_stats = allocated_subset_stats(allocated, cand_cfg)
        select_stats = allocated_subset_stats(select_df, cand_cfg)
        holdout_stats = allocated_subset_stats(holdout_df, cand_cfg)
        score = objective_condition_score(select_stats, score_profile=score_profile)

        row: Dict[str, object] = {
            "candidate": name,
            "select_score": score,
            "score_profile": score_profile,
            "split_date": split_date,
            "no_trade_band": cand_cfg.no_trade_band,
            "use_continuous_adjustment": cand_cfg.use_continuous_adjustment,
            "risk_off_stock_weight": cand_cfg.risk_off_stock_weight,
            "risk_off_bond_weight": cand_cfg.risk_off_bond_weight,
            "risk_off_cash_weight": cand_cfg.risk_off_cash_weight,
            "gate_high_vol_threshold": cand_cfg.gate_high_vol_threshold,
            "gate_riskoff_downrisk_threshold": cand_cfg.gate_riskoff_downrisk_threshold,
            "gate_watch_downrisk_threshold": cand_cfg.gate_watch_downrisk_threshold,
        }
        for prefix, stats in [("select", select_stats), ("holdout", holdout_stats), ("full", full_stats)]:
            for key in [
                "cagr", "mdd", "sharpe", "sortino", "calmar", "gross_cagr", "cost_cagr_drag",
                "annual_turnover", "total_turnover", "avg_daily_turnover", "trade_day_ratio",
                "avg_trade_size_on_trade", "max_trade_size", "total_transaction_cost_rate_sum",
                "avg_stock_weight", "min_stock_weight", "max_stock_weight", "std_stock_weight",
                "avg_bond_weight", "avg_cash_weight", "rebalance_ratio", "emergency_rebalance_ratio",
                "regime_switch_ratio", "normal_pct", "watch_pct", "high_vol_pct", "risk_off_pct", "extreme_risk_pct",
                "actual_high_vol_rate", "actual_down_high_vol_rate", "avg_prob_high_vol", "avg_prob_down_risk",
            ]:
                row[f"{prefix}_{key}"] = stats.get(key, 0.0)
        rows.append(row)

        if score > best_score:
            best_score = score
            best_cfg = cand_cfg
            best_name = name

    report_df = pd.DataFrame(rows).sort_values("select_score", ascending=False).reset_index(drop=True)

    # v8.4 stable-top 선택 로직:
    # 1) select_score 상위 후보군을 만든다.
    # 2) 그 안에서 select 기준 turnover/MDD/Calmar가 더 안정적인 후보를 고른다.
    # 3) 전체/holdout 성과는 선택 근거가 아니라 사후 진단으로만 저장한다.
    top_n = max(3, int(math.ceil(len(report_df) * 0.08)))
    pool = report_df.head(top_n).copy()
    # 너무 공격적인 후보를 줄이기 위한 soft constraint. 통과 후보가 없으면 top pool 전체 사용.
    constrained = pool[
        (pool["select_annual_turnover"] <= 2.00) &
        (pool["select_mdd"] >= -0.32) &
        (pool["select_calmar"] >= 0.45)
    ].copy()
    if constrained.empty:
        constrained = pool
    constrained = constrained.sort_values(
        ["select_calmar", "select_annual_turnover", "select_mdd", "select_cagr"],
        ascending=[False, True, False, False],
    )
    selected_name = str(constrained.iloc[0]["candidate"])
    selected_score = float(constrained.iloc[0]["select_score"])
    cfg_map = {name: cfg for name, cfg in candidates}
    selected_cfg = cfg_map[selected_name]

    meta = {
        "selected_candidate": selected_name,
        "selected_score": selected_score,
        "selection_method": "stable_top_select_score_then_select_calmar_turnover_mdd",
        "top_pool_size": int(top_n),
        "constrained_pool_size": int(len(constrained)),
        "raw_best_score_candidate": best_name,
        "raw_best_score": float(best_score),
        "split_date": split_date,
        "grid_size": grid_size,
        "score_profile": score_profile,
        "candidate_count": int(len(candidates)),
    }
    return selected_cfg, report_df, meta

# ============================================================
# 8. OPTIMIZATION DIAGNOSTICS
# ============================================================

def _annualized_return(x: pd.Series) -> float:
    r = x.dropna().astype(float)
    if len(r) == 0:
        return 0.0
    return float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)


def _annualized_vol(x: pd.Series) -> float:
    r = x.dropna().astype(float)
    return float(r.std() * math.sqrt(252)) if len(r) > 1 else 0.0


def _win_rate(x: pd.Series) -> float:
    r = x.dropna().astype(float)
    return float((r > 0).mean()) if len(r) else 0.0


def build_regime_analysis(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if "allocation_regime" not in pred_df.columns:
        return pd.DataFrame(rows)
    total_n = len(pred_df)
    for regime, g in pred_df.groupby("allocation_regime", dropna=False):
        rr = g["strategy_return_net"].astype(float)
        rows.append({
            "allocation_regime": str(regime),
            "count": int(len(g)),
            "pct": float(len(g) / total_n) if total_n else 0.0,
            "ann_return_est": _annualized_return(rr),
            "ann_vol_est": _annualized_vol(rr),
            "mean_daily_return": float(rr.mean()),
            "win_rate": _win_rate(rr),
            "avg_stock_weight": float(g["stock_weight"].mean()),
            "avg_bond_weight": float(g["bond_weight"].mean()),
            "avg_cash_weight": float(g["cash_weight"].mean()),
            "avg_turnover": float(g["turnover"].mean()),
            "annual_turnover_est": float(g["turnover"].mean() * 252.0),
            "rebalance_ratio": float(g["rebalanced"].mean()),
            "emergency_rebalance_ratio": float(g["emergency_rebalance"].mean()),
            "avg_prob_high_vol": float(g["prob_high_vol"].mean()),
            "avg_prob_down_risk": float(g["prob_down_risk"].mean()),
            "actual_high_vol_rate": float((g["actual_risk"] == "고변동").mean()) if "actual_risk" in g.columns else 0.0,
            "actual_down_high_vol_rate": float((g["actual_split_vol"] == "하락고변동").mean()) if "actual_split_vol" in g.columns else 0.0,
        })
    return pd.DataFrame(rows).sort_values("pct", ascending=False).reset_index(drop=True)


def build_regime_transition_matrix(pred_df: pd.DataFrame) -> pd.DataFrame:
    if "allocation_regime" not in pred_df.columns or len(pred_df) < 2:
        return pd.DataFrame()
    s = pred_df["allocation_regime"].astype(str)
    mat = pd.crosstab(s.shift(1), s, normalize="index").fillna(0.0)
    mat.index.name = "from_regime"
    mat.columns.name = "to_regime"
    return mat.reset_index()


def _bin_series(s: pd.Series, bins: int = 10) -> pd.Series:
    vals = s.astype(float).clip(0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    return pd.cut(vals, bins=edges, include_lowest=True, right=True)


def build_probability_bin_analysis(pred_df: pd.DataFrame, prob_col: str, actual_col: str, positive_value: str, bins: int = 10) -> pd.DataFrame:
    if prob_col not in pred_df.columns or actual_col not in pred_df.columns:
        return pd.DataFrame()
    tmp = pred_df.copy()
    tmp["prob_bin"] = _bin_series(tmp[prob_col], bins=bins)
    tmp["actual_positive"] = (tmp[actual_col] == positive_value).astype(int)
    rows: List[Dict[str, object]] = []
    for b, g in tmp.groupby("prob_bin", observed=False):
        if len(g) == 0:
            continue
        rows.append({
            "prob_col": prob_col,
            "actual_col": actual_col,
            "positive_value": positive_value,
            "prob_bin": str(b),
            "count": int(len(g)),
            "actual_rate": float(g["actual_positive"].mean()),
            "avg_prob": float(g[prob_col].astype(float).mean()),
            "avg_strategy_return_net": float(g["strategy_return_net"].astype(float).mean()),
            "ann_return_est": _annualized_return(g["strategy_return_net"]),
            "avg_stock_weight": float(g["stock_weight"].mean()),
            "avg_turnover": float(g["turnover"].mean()),
        })
    return pd.DataFrame(rows)


def build_threshold_diagnostics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    diagnostics = [
        ("stage1_ensemble", "actual_risk", "고변동", "prob_high_vol"),
        ("overall_risk_highvol", "actual_risk", "고변동", "prob_overall_risk"),
        ("down_strength_score", "actual_direction_strength_20d", "DOWN_STRENGTHENING", "prob_down_strengthening_score"),
        ("up_strength_score", "actual_direction_strength_20d", "UP_STRENGTHENING", "prob_up_strengthening_score"),
    ]
    for h in [10, 20]:
        if f"prob_high_vol_h{h}" in pred_df.columns:
            diagnostics.append((f"stage1_h{h}", f"actual_risk_h{h}", "고변동", f"prob_high_vol_h{h}"))
        if f"prob_down_strengthening_{h}d" in pred_df.columns:
            diagnostics.append((f"down_strength_{h}d", f"actual_direction_strength_{h}d", "DOWN_STRENGTHENING", f"prob_down_strengthening_{h}d"))
        if f"prob_up_strengthening_{h}d" in pred_df.columns:
            diagnostics.append((f"up_strength_{h}d", f"actual_direction_strength_{h}d", "UP_STRENGTHENING", f"prob_up_strengthening_{h}d"))
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    for name, actual_col, positive_value, prob_col in diagnostics:
        if actual_col not in pred_df.columns or prob_col not in pred_df.columns:
            continue
        y_true = (pred_df[actual_col] == positive_value).astype(int).values
        prob = pred_df[prob_col].astype(float).clip(0.0, 1.0).values
        support_pos = int(y_true.sum())
        support_neg = int(len(y_true) - support_pos)
        for th in thresholds:
            y_pred = (prob >= float(th)).astype(int)
            rows.append({
                "model": name,
                "prob_col": prob_col,
                "actual_col": actual_col,
                "positive_value": positive_value,
                "threshold": float(th),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "pred_positive_ratio": float(y_pred.mean()),
                "support_positive": support_pos,
                "support_negative": support_neg,
            })
    return pd.DataFrame(rows)


def build_turnover_diagnostics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    df = pred_df.copy()
    df["trade_occurred"] = df["turnover"].astype(float) > 1e-12
    df["rebalance_type"] = np.where(df["emergency_rebalance"].astype(bool), "emergency", np.where(df["rebalanced"].astype(bool), "scheduled_or_initial", "no_rebalance"))
    for section, group_cols in [
        ("by_rebalance_type", ["rebalance_type"]),
        ("by_regime", ["allocation_regime"]),
        ("by_regime_rebalance_type", ["allocation_regime", "rebalance_type"]),
    ]:
        grouped = df.groupby(group_cols, dropna=False)
        for group_key, g in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row: Dict[str, object] = {"section": section, "count": int(len(g)), "pct": float(len(g) / len(df)) if len(df) else 0.0}
            for col, val in zip(group_cols, group_key):
                row[col] = str(val)
            row.update({
                "trade_day_ratio": float(g["trade_occurred"].mean()),
                "avg_turnover": float(g["turnover"].mean()),
                "annual_turnover_est": float(g["turnover"].mean() * 252.0),
                "total_turnover": float(g["turnover"].sum()),
                "avg_trade_size_on_trade": float(g.loc[g["trade_occurred"], "turnover"].mean()) if g["trade_occurred"].any() else 0.0,
                "max_turnover": float(g["turnover"].max()),
                "cost_sum": float(g["transaction_cost"].sum()),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def build_drawdown_episodes(pred_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if "strategy_equity_net" not in pred_df.columns or pred_df.empty:
        return pd.DataFrame()
    df = pred_df[["Date", "strategy_equity_net", "allocation_regime", "stock_weight", "prob_high_vol", "prob_down_risk"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    equity = df["strategy_equity_net"].astype(float)
    peak = equity.cummax()
    dd = equity / peak - 1.0
    df["drawdown"] = dd
    episodes: List[Dict[str, object]] = []
    in_dd = False
    start_idx = 0
    trough_idx = 0
    min_dd = 0.0
    for i, val in enumerate(dd.values):
        if not in_dd and val < 0:
            in_dd = True
            start_idx = max(0, i - 1)
            trough_idx = i
            min_dd = float(val)
        elif in_dd:
            if val < min_dd:
                min_dd = float(val)
                trough_idx = i
            if val >= -1e-12:
                end_idx = i
                seg = df.iloc[start_idx:end_idx + 1]
                episodes.append({
                    "start_date": str(df.iloc[start_idx]["Date"].date()),
                    "trough_date": str(df.iloc[trough_idx]["Date"].date()),
                    "recovery_date": str(df.iloc[end_idx]["Date"].date()),
                    "depth": min_dd,
                    "duration_days": int(end_idx - start_idx),
                    "days_to_trough": int(trough_idx - start_idx),
                    "avg_stock_weight": float(seg["stock_weight"].mean()),
                    "avg_prob_high_vol": float(seg["prob_high_vol"].mean()),
                    "avg_prob_down_risk": float(seg["prob_down_risk"].mean()),
                    "trough_regime": str(df.iloc[trough_idx]["allocation_regime"]),
                })
                in_dd = False
    if in_dd:
        end_idx = len(df) - 1
        seg = df.iloc[start_idx:end_idx + 1]
        episodes.append({
            "start_date": str(df.iloc[start_idx]["Date"].date()),
            "trough_date": str(df.iloc[trough_idx]["Date"].date()),
            "recovery_date": "not_recovered",
            "depth": min_dd,
            "duration_days": int(end_idx - start_idx),
            "days_to_trough": int(trough_idx - start_idx),
            "avg_stock_weight": float(seg["stock_weight"].mean()),
            "avg_prob_high_vol": float(seg["prob_high_vol"].mean()),
            "avg_prob_down_risk": float(seg["prob_down_risk"].mean()),
            "trough_regime": str(df.iloc[trough_idx]["allocation_regime"]),
        })
    return pd.DataFrame(episodes).sort_values("depth").head(top_n).reset_index(drop=True)


def build_periodic_returns(pred_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    tmp = pred_df.copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"])
    tmp = tmp.set_index("Date")
    cols = {
        "strategy_return_net": "strategy_net",
        "strategy_return_gross": "strategy_gross",
        "stock_next_return": "stock_buy_hold",
        "bond_next_return": "bond",
        "cash_next_return": "cash",
    }
    rows: List[Dict[str, object]] = []
    for period, g in tmp.resample(freq):
        if g.empty:
            continue
        row: Dict[str, object] = {"period": str(period.date())}
        for col, name in cols.items():
            if col in g.columns:
                row[name] = float((1.0 + g[col].astype(float)).prod() - 1.0)
        row["avg_stock_weight"] = float(g["stock_weight"].mean())
        row["turnover_sum"] = float(g["turnover"].sum())
        period_equity = (1.0 + g["strategy_return_net"].astype(float)).cumprod()
        row["max_drawdown_inside_period"] = float((period_equity / period_equity.cummax() - 1.0).min()) if len(period_equity) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_optimization_metrics(summary: Dict[str, object]) -> pd.DataFrame:
    s1 = summary.get("stage1_feature_importance_mean", {}) or {}
    dn = summary.get("downrisk_feature_importance_mean", {}) or {}
    features = sorted(set(s1.keys()) | set(dn.keys()))
    rows: List[Dict[str, object]] = []
    for f in features:
        s1_imp = float(s1.get(f, 0.0))
        dn_imp = float(dn.get(f, 0.0))
        rows.append({
            "feature": f,
            "stage1_importance": s1_imp,
            "downrisk_importance": dn_imp,
            "mean_importance": (s1_imp + dn_imp) / 2.0,
            "max_importance": max(s1_imp, dn_imp),
            "importance_gap_stage1_minus_downrisk": s1_imp - dn_imp,
            "is_low_importance_candidate": bool(max(s1_imp, dn_imp) < 0.001),
            "used_more_by": "stage1" if s1_imp > dn_imp else ("downrisk" if dn_imp > s1_imp else "tie"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["stage1_rank"] = df["stage1_importance"].rank(ascending=False, method="min").astype(int)
    df["downrisk_rank"] = df["downrisk_importance"].rank(ascending=False, method="min").astype(int)
    df["mean_rank"] = df["mean_importance"].rank(ascending=False, method="min").astype(int)
    return df.sort_values(["mean_importance", "max_importance"], ascending=False).reset_index(drop=True)


def build_optimization_diagnostics(pred_df: pd.DataFrame, summary: Dict[str, object]) -> Dict[str, pd.DataFrame]:
    prob_bins = []
    hv_bins = build_probability_bin_analysis(pred_df, "prob_high_vol", "actual_risk", "고변동")
    if not hv_bins.empty:
        prob_bins.append(hv_bins)
    overall_bins = build_probability_bin_analysis(pred_df, "prob_overall_risk", "actual_risk", "고변동")
    if not overall_bins.empty:
        prob_bins.append(overall_bins)
    dn_bins = build_probability_bin_analysis(pred_df, "prob_down", "actual_direction", "하락") if "prob_down" in pred_df.columns else build_probability_bin_analysis(pred_df, "prob_down_risk", "actual_split_vol", "하락고변동")
    if not dn_bins.empty:
        prob_bins.append(dn_bins)
    up_bins = build_probability_bin_analysis(pred_df, "prob_up", "actual_direction", "상승") if "prob_up" in pred_df.columns else pd.DataFrame()
    if not up_bins.empty:
        prob_bins.append(up_bins)
    up_strength_bins = build_probability_bin_analysis(pred_df, "prob_up_strengthening_score", "actual_direction_strength_20d", "UP_STRENGTHENING") if "prob_up_strengthening_score" in pred_df.columns and "actual_direction_strength_20d" in pred_df.columns else pd.DataFrame()
    if not up_strength_bins.empty:
        prob_bins.append(up_strength_bins)
    return {
        "regime_analysis": build_regime_analysis(pred_df),
        "regime_transition_matrix": build_regime_transition_matrix(pred_df),
        "probability_bins": pd.concat(prob_bins, ignore_index=True) if prob_bins else pd.DataFrame(),
        "threshold_diagnostics": build_threshold_diagnostics(pred_df),
        "turnover_diagnostics": build_turnover_diagnostics(pred_df),
        "drawdown_episodes": build_drawdown_episodes(pred_df),
        "monthly_returns": build_periodic_returns(pred_df, "ME"),
        "annual_returns": build_periodic_returns(pred_df, "YE"),
        "feature_optimization_metrics": build_feature_optimization_metrics(summary),
    }


def diagnostics_summary(diagnostics: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    reg = diagnostics.get("regime_analysis", pd.DataFrame())
    if not reg.empty:
        out["regime_count"] = int(len(reg))
        out["best_regime_by_ann_return"] = str(reg.sort_values("ann_return_est", ascending=False).iloc[0]["allocation_regime"])
        out["highest_turnover_regime"] = str(reg.sort_values("annual_turnover_est", ascending=False).iloc[0]["allocation_regime"])
    dd = diagnostics.get("drawdown_episodes", pd.DataFrame())
    if not dd.empty:
        out["worst_drawdown_episode"] = dd.iloc[0].to_dict()
    feat = diagnostics.get("feature_optimization_metrics", pd.DataFrame())
    if not feat.empty:
        out["low_importance_feature_count"] = int(feat["is_low_importance_candidate"].sum())
        out["top10_mean_importance_features"] = feat.head(10)["feature"].tolist()
    return out

# ============================================================
# 8. CLI / MAIN
# ============================================================

def apply_speed_profile(cfg: Config, profile: str) -> Config:
    if profile == "fast":
        cfg.retrain_every_n_days = 20
        cfg.stage1_n_estimators = 100
        cfg.down_n_estimators = 70
        cfg.use_adaptive_label_policy = False
        cfg.use_rolling_gate_optimization = False
        cfg.result_dir = "results_xgb_strength_combo_all_v8_6_34_fast"
    elif profile == "balanced":
        cfg.retrain_every_n_days = 10
        cfg.stage1_n_estimators = 150
        cfg.down_n_estimators = 100
        cfg.use_adaptive_label_policy = False
        cfg.use_rolling_gate_optimization = False
        cfg.result_dir = "results_xgb_strength_combo_all_v8_6_34_balanced"
    elif profile == "full":
        cfg.retrain_every_n_days = 10
        cfg.stage1_n_estimators = 200
        cfg.down_n_estimators = 140
        cfg.use_adaptive_label_policy = True
        cfg.use_rolling_gate_optimization = False
        cfg.result_dir = "results_xgb_strength_combo_all_v8_6_34_full_adaptive_label"
    else:
        raise ValueError(f"알 수 없는 speed profile: {profile}")
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost v8.6.34 Volatility-Base Strong-Override Multi-Horizon Upside Strength Trigger + Objective Condition Search")
    parser.add_argument("--speed-profile", choices=["fast", "balanced", "full"], default="balanced")
    parser.add_argument("--target-ticker", type=str, default=None, help="검증할 위험자산 ticker. 예: QQQ, SPY, NVDA")
    parser.add_argument("--asset-list", type=str, default=None, help="여러 ticker를 콤마로 지정해 순차 백테스트. 예: QQQ,SPY,IWM,NVDA")
    parser.add_argument("--asset-preset", choices=["etf", "mega", "mixed"], default=None, help="여러 종목 검증 preset")
    parser.add_argument("--adaptive-label", action="store_true", help="라벨 quantile 정책을 nested validation으로 선택")
    parser.add_argument("--rolling-gate-opt", action="store_true", help="작은 grid로 allocation gate threshold를 rolling 최적화")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--retrain-every", type=int, default=None)
    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--h10-down-only", action="store_true", help="Down-risk ensemble에서 H20을 제거하고 H10만 사용")
    parser.add_argument("--no-trade-band", type=float, default=None, help="거래 무시 band 직접 지정. 예: 0.07")
    parser.add_argument("--rebalance-every", type=int, default=None, help="리밸런싱 주기 직접 지정. v8.6.34 기본 5거래일")
    parser.add_argument("--emergency-cooldown", type=int, default=None, help="긴급 리밸런싱 cooldown 직접 지정. v8.6.34 기본 5거래일")
    parser.add_argument("--condition-search", action="store_true", help="조건을 validation 구간에서 객관식 후보 비교 후 선택")
    parser.add_argument("--condition-split-date", type=str, default="2021-12-31", help="조건 선택 구간의 마지막 날짜. 이후 구간은 holdout 확인용")
    parser.add_argument("--condition-grid-size", choices=["compact", "standard", "wide"], default="standard")
    parser.add_argument("--score-profile", choices=["balanced", "cagr", "calmar", "turnover"], default="balanced")
    parser.add_argument("--policy-mode", choices=["base", "return_seeking", "defensive_risk", "aggressive_dynamic", "direction_strength_specialist", "strength_specialist", "ds_specialist", "portfolio_model"], default=None, help="v8.6.34 policy branch")
    parser.add_argument("--no-policy-overlay", action="store_true", help="policy overlay를 끄고 v8.6.5 base bucket만 사용")
    parser.add_argument("--no-direction-strength", action="store_true", help="v8.6.34 up-strength offensive overlay 모델 학습/사용 비활성화")
    parser.add_argument("--multi-strength-horizons", type=str, default=None, help="UP_STRENGTHENING specialist horizons. 예: 5,10,20 또는 20")
    parser.add_argument("--up-strength-threshold", type=float, default=None, help="UP_STRENGTHENING stock bonus 1차 threshold")
    parser.add_argument("--bear-down-strength-threshold", type=float, default=None, help="bear specialist DOWN_STRENGTHENING stock cut 1차 threshold")
    parser.add_argument("--allocation-downrisk-weight", type=float, default=None, help="allocation gate에서 down-risk를 반영할 비중. 기본 0.0=Stage1 high-vol 중심")
    parser.add_argument("--enable-bear-cut", action="store_true", help="기본 OFF인 bear specialist stock cut을 활성화")
    parser.add_argument("--offensive-stock-1", type=float, default=None, help="UP_STRENGTHENING 1차 조건 충족 시 목표 주식 비중")
    parser.add_argument("--offensive-stock-2", type=float, default=None, help="UP_STRENGTHENING 2차 조건 충족 시 목표 주식 비중")
    parser.add_argument("--offensive-stock-3", type=float, default=None, help="UP_STRENGTHENING 3차 조건 충족 시 목표 주식 비중")
    parser.add_argument("--up-strength-threshold-2", type=float, default=None, help="UP_STRENGTHENING 2차 공격 threshold")
    parser.add_argument("--up-strength-threshold-3", type=float, default=None, help="UP_STRENGTHENING 3차 공격 threshold")
    parser.add_argument("--low-vol-threshold-1", type=float, default=None, help="공격 overlay 1차 high-vol 상한")
    parser.add_argument("--low-vol-threshold-2", type=float, default=None, help="공격 overlay 2차 high-vol 상한")
    parser.add_argument("--low-vol-threshold-3", type=float, default=None, help="공격 overlay 3차 high-vol 상한")
    parser.add_argument("--base-normal-stock", type=float, default=None, help="저비중 기본 NORMAL 주식 비중")
    parser.add_argument("--base-watch-stock", type=float, default=None, help="저비중 WATCH 주식 비중")
    parser.add_argument("--base-riskoff-stock", type=float, default=None, help="저비중 RISK_OFF 주식 비중")
    parser.add_argument("--base-extreme-stock", type=float, default=None, help="저비중 EXTREME_RISK 주식 비중")
    parser.add_argument("--disable-stale-offensive-decay", action="store_true", help="v8.6.34 기본 ON인 stale offensive decay 비활성화")
    parser.add_argument("--extreme-high-vol-threshold", type=float, default=None, help="EXTREME_RISK 진입 high-vol threshold")
    parser.add_argument("--riskoff-stock", type=float, default=None, help="RISK_OFF 주식 비중 직접 지정")
    parser.add_argument("--watch-stock", type=float, default=None, help="WATCH 주식 비중 직접 지정")
    parser.add_argument("--four-regime", action="store_true", help="v8.3 방식의 NORMAL/WATCH/HIGH_VOL/RISK_OFF 4-regime 구조 사용")
    parser.add_argument("--no-extreme-risk", action="store_true", help="EXTREME_RISK 추가 방어 규칙 비활성화")
    parser.add_argument("--enable-stale-offensive-decay", action="store_true", help="이전 공격 비중이 현재 target보다 과도하게 높고 상승 강화 신호가 약해지면 중간 리밸런싱")
    parser.add_argument("--stale-offensive-gap", type=float, default=None, help="stale offensive decay를 작동시키는 stock gap. 기본 0.09")
    parser.add_argument("--stale-offensive-reset-threshold", type=float, default=None, help="UP_STRENGTHENING 확률이 이 값보다 낮으면 stale offensive로 간주. 기본 0.18")
    parser.add_argument("--stale-offensive-high-vol-threshold", type=float, default=None, help="high-vol 확률이 이 값 이상이면 stale offensive decay 허용. 기본 0.68")
    parser.add_argument("--enable-portfolio-model", action="store_true", help="2단계 PortfolioPolicyModel을 진단용으로 학습/출력")
    parser.add_argument("--portfolio-policy-horizon", type=int, default=None, help="PortfolioPolicyModel utility 라벨 계산 horizon. 기본 20")
    parser.add_argument("--portfolio-policy-min-train-rows", type=int, default=None, help="PortfolioPolicyModel 최소 학습 row 수")
    parser.add_argument("--portfolio-policy-max-train-rows", type=int, default=None, help="PortfolioPolicyModel rolling 학습창 상한")
    parser.add_argument("--portfolio-model-min-confidence", type=float, default=None, help="이 값보다 confidence가 낮으면 fallback 포트폴리오 사용")
    parser.add_argument("--portfolio-model-force-rebalance", action="store_true", help="PortfolioPolicyModel이 P7/P8을 예측하면 schedule/no-trade-band 우회")
    parser.add_argument("--no-diagnostics", action="store_true", help="추가 최적화 진단 CSV 저장을 생략")
    parser.add_argument("--execution-lag-days", type=int, default=None, help="체결 지연 일수. 0=기존 방식, 1=보수적 다음 거래일 체결 가정")
    parser.add_argument("--max-train-rows", type=int, default=None, help="최근 N개 학습 샘플만 사용. horizon train-window mode에서는 hard cap으로 작동")
    parser.add_argument("--no-horizon-train-window", action="store_true", help="v8.6.34 horizon별 train_rows = horizon*multiplier 모드 비활성화")
    parser.add_argument("--horizon-train-min-rows", type=int, default=None, help="horizon별 학습창 최소 row 수. 기본 504")
    parser.add_argument("--horizon-train-max-cap", type=int, default=None, help="horizon별 학습창 상한 cap")
    parser.add_argument("--horizon-train-multiplier", type=float, default=None, help="5D/10D/20D에 동일 multiplier 적용. 예: 100")
    parser.add_argument("--h5-train-multiplier", type=float, default=None, help="5D 학습창 multiplier. train_rows=5*multiplier")
    parser.add_argument("--h10-train-multiplier", type=float, default=None, help="10D 학습창 multiplier. train_rows=10*multiplier")
    parser.add_argument("--h20-train-multiplier", type=float, default=None, help="20D 학습창 multiplier. train_rows=20*multiplier")
    parser.add_argument("--allow-cash-download-fallback", action="store_true", help="BIL 다운로드 실패 시 cash return을 0으로 대체")
    parser.add_argument("--up-strength-weight-5d", type=float, default=None, help="multi-horizon UP score에서 5D 가중치")
    parser.add_argument("--up-strength-weight-10d", type=float, default=None, help="multi-horizon UP score에서 10D 가중치")
    parser.add_argument("--up-strength-weight-20d", type=float, default=None, help="multi-horizon UP score에서 20D 가중치")
    parser.add_argument("--up-confirm-10d-threshold-2", type=float, default=None, help="Tier2 진입용 10D UP_STRENGTHENING 확인 threshold")
    parser.add_argument("--up-confirm-20d-threshold-2", type=float, default=None, help="Tier2 진입용 20D UP_STRENGTHENING 확인 threshold")
    parser.add_argument("--up-confirm-10d-threshold-3", type=float, default=None, help="Tier3 진입용 10D UP_STRENGTHENING 확인 threshold")
    parser.add_argument("--up-confirm-20d-threshold-3", type=float, default=None, help="Tier3 진입용 20D UP_STRENGTHENING 확인 threshold")
    parser.add_argument("--up-pred-threshold-5d", type=float, default=None, help="5D 상승 예측 판정 threshold")
    parser.add_argument("--up-pred-threshold-10d", type=float, default=None, help="10D 상승 예측 판정 threshold")
    parser.add_argument("--up-pred-threshold-20d", type=float, default=None, help="20D 상승 예측 판정 threshold")
    parser.add_argument("--disable-vol-base-allocation", action="store_true", help="v8.6.34 high-vol 확률 기반 기본 비중을 끄고 regime bucket 기본 비중 사용")
    parser.add_argument("--vol-base-bond-ratio", type=float, default=None, help="기본 상태 방어자산 중 채권 비율. 기본 0.65")
    parser.add_argument("--full-stock-score-threshold", type=float, default=None, help="100% 진입용 UP strength score threshold")
    parser.add_argument("--full-stock-10d-threshold", type=float, default=None, help="100% 진입용 10D UP_STRENGTHENING threshold")
    parser.add_argument("--full-stock-20d-threshold", type=float, default=None, help="100% 진입용 20D UP_STRENGTHENING threshold")
    parser.add_argument("--full-stock-high-vol-threshold", type=float, default=None, help="100% 진입용 high-vol 확률 상한")
    parser.add_argument("--disable-strong-override", action="store_true", help="Tier3/Full의 schedule/no-trade-band 우회 즉시 리밸런싱 비활성화")
    parser.add_argument("--disable-tier2", action="store_true", help="기존 Tier2 비활성화. v8.6.34 기본값도 비활성화")
    parser.add_argument("--enable-tier2", action="store_true", help="기존 10D+20D Tier2 조건을 다시 활성화")
    parser.add_argument("--short-mid-mode", choices=["diagnostic", "tier1_upgrade", "base_upgrade", "base_tier1_upgrade", "tier2_add", "tier2_replace"], default=None, help="5D+10D+high-vol ShortMidConfirm 적용 방식. 기본 base_upgrade")
    parser.add_argument("--short-mid-action-signal", choices=["confirm", "strong", "loose", "all3"], default=None, help="ShortMid action에 사용할 신호. confirm=hv<0.72, strong=hv<0.68, loose=hv<0.76, all3=5D+10D+20D+hv<0.72")
    parser.add_argument("--short-mid-p5", type=float, default=None, help="ShortMidConfirm 5D threshold. 기본 0.32")
    parser.add_argument("--short-mid-p10", type=float, default=None, help="ShortMidConfirm 10D threshold. 기본 0.34")
    parser.add_argument("--short-mid-p20", type=float, default=None, help="ShortMid all3 confirm용 20D threshold. 기본 0.34")
    parser.add_argument("--short-mid-high-vol", type=float, default=None, help="ShortMidConfirm high-vol 상한. 기본 0.72")
    parser.add_argument("--short-mid-strong-high-vol", type=float, default=None, help="Strong ShortMidConfirm high-vol 상한. 기본 0.68")
    parser.add_argument("--short-mid-loose-high-vol", type=float, default=None, help="Loose ShortMidConfirm high-vol 상한. 기본 0.76")
    parser.add_argument("--short-mid-use-score", action="store_true", help="ShortMidConfirm에 up_strength_score 조건도 추가")
    parser.add_argument("--short-mid-score", type=float, default=None, help="ShortMidConfirm score threshold. 기본 0.38")
    parser.add_argument("--short-mid-tier1-upgrade-stock", type=float, default=None, help="tier1_upgrade 모드에서 Tier1+ShortMid 목표 주식 비중. 기본 0.84")
    parser.add_argument("--short-mid-tier2-stock", type=float, default=None, help="tier2_add/tier2_replace 모드에서 ShortMid Tier2 목표 주식 비중. 기본 0.88")
    parser.add_argument("--short-mid-base-upgrade-stock", type=float, default=None, help="base_upgrade 모드에서 base+ShortMid 목표 주식 비중. 기본 0.82")
    parser.add_argument("--strength-combo-mode", choices=["off", "diagnostic", "max_weight"], default=None, help="5D/10D/20D 모든 조합 사용 방식. 기본 max_weight")
    parser.add_argument("--disable-strength-combo-policy", action="store_true", help="5D/10D/20D 조합 ladder allocation 비활성화")
    parser.add_argument("--strength-combo-no-high-vol", action="store_true", help="조합 신호에서 high-vol 필터 제거")
    parser.add_argument("--strength-combo-high-vol", type=float, default=None, help="조합 신호 high-vol 상한. 기본 0.72")
    parser.add_argument("--strength-combo-use-score", action="store_true", help="조합 신호에 up_strength_score 조건 추가")
    parser.add_argument("--strength-combo-score", type=float, default=None, help="조합 신호 score threshold. 기본 0.38")
    parser.add_argument("--combo-single-5d-stock", type=float, default=None, help="5D 단독 조합 목표 주식 비중. 기본 0.80")
    parser.add_argument("--combo-single-10d-stock", type=float, default=None, help="10D 단독 조합 목표 주식 비중. 기본 0.82")
    parser.add_argument("--combo-single-20d-stock", type=float, default=None, help="20D 단독 조합 목표 주식 비중. 기본 0.82")
    parser.add_argument("--combo-5d-10d-stock", type=float, default=None, help="5D+10D 조합 목표 주식 비중. 기본 0.84")
    parser.add_argument("--combo-5d-20d-stock", type=float, default=None, help="5D+20D 조합 목표 주식 비중. 기본 0.86")
    parser.add_argument("--combo-10d-20d-stock", type=float, default=None, help="10D+20D 조합 목표 주식 비중. 기본 0.88")
    parser.add_argument("--combo-all3-stock", type=float, default=None, help="5D+10D+20D 조합 목표 주식 비중. 기본 0.96")
    parser.add_argument("--optimize-tier-weights", action="store_true", help="Tier2 포함 Tier별 목표 비중 Walk-forward grid search 실행")
    parser.add_argument("--tier-opt-train-rows", type=int, default=None, help="TierWeightOptimizer 학습창 row 수. 기본 756")
    parser.add_argument("--tier-opt-test-rows", type=int, default=None, help="TierWeightOptimizer OOS 적용 구간 row 수. 기본 63")
    parser.add_argument("--tier-opt-min-train-rows", type=int, default=None, help="TierWeightOptimizer 최소 학습 row 수. 기본 504")
    parser.add_argument("--tier-opt-score-profile", choices=["aggressive", "balanced", "calmar", "sharpe"], default=None, help="TierWeightOptimizer 목적함수 profile")
    parser.add_argument("--tier-opt-no-base-lt25", action="store_true", help="base prob_high_vol<25% 기본 비중은 최적화하지 않고 고정")
    parser.add_argument("--tier-opt-base-lt25-grid", type=str, default=None, help="base_lt25 후보. 예: 0.76,0.78,0.80,0.82")
    parser.add_argument("--tier-opt-tier1-grid", type=str, default=None, help="Tier1 목표 주식 비중 후보. 예: 0.78,0.80,0.82")
    parser.add_argument("--tier-opt-tier2-grid", type=str, default=None, help="Tier2 목표 주식 비중 후보. 예: 0.82,0.84,0.86,0.88")
    parser.add_argument("--tier-opt-tier3-grid", type=str, default=None, help="Tier3 목표 주식 비중 후보. 예: 0.92,0.94,0.96")
    parser.add_argument("--tier-opt-full-grid", type=str, default=None, help="Full 목표 주식 비중 후보. 예: 0.98,1.00")
    return parser.parse_args()



def drop_weak_probability_output_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    v8.6.34 external-output filter.
    내부 호환 계산에는 weak 확률 컬럼이 일부 남을 수 있지만,
    저장되는 predictions.csv에서는 allocation에 쓰지 않는 약한 확률 계열을 제거한다.
    """
    weak_exact = {
        "prob_up",
        "prob_down",
        "prob_down_risk",
        "prob_up_proxy",
        "prob_bear_up_strengthening",
        "prob_bear_up_weakening",
        "prob_bear_down_strengthening",
        "prob_bear_down_weakening",
        "prob_down_high_vol",
        "direction_score",
    }
    weak_prefixes = (
        "prob_up_h",
        "prob_down_h",
        "prob_down_risk_h",
        "prob_down_price_trend",
        "prob_down_price_volume",
        "prob_down_volatility",
        "prob_bear_",
    )
    drop_cols = [
        c for c in pred_df.columns
        if c in weak_exact or any(str(c).startswith(p) for p in weak_prefixes)
    ]
    return pred_df.drop(columns=drop_cols, errors="ignore")

def main() -> None:
    args = parse_args()

    if getattr(args, "asset_list", None) or getattr(args, "asset_preset", None):
        preset_map = {
            "etf": ["QQQ", "SPY", "IWM", "DIA", "XLK", "SMH", "SOXX", "XLY", "XLF", "XLV"],
            "mega": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO"],
            "mixed": ["QQQ", "SPY", "IWM", "SMH", "SOXX", "NVDA", "MSFT", "AAPL", "TSLA"],
        }
        tickers: List[str] = []
        if getattr(args, "asset_preset", None):
            tickers.extend(preset_map[str(args.asset_preset)])
        if getattr(args, "asset_list", None):
            tickers.extend([x.strip().upper() for x in str(args.asset_list).split(",") if x.strip()])
        tickers = list(dict.fromkeys(tickers))
        batch_root = Path(args.result_dir or "results_xgb_strength_combo_all_v8_6_34_multi_asset")
        batch_root.mkdir(parents=True, exist_ok=True)

        # 현재 CLI에서 batch 관련 인자와 result-dir/target-ticker를 제거한 뒤 ticker별로 재호출한다.
        base_args: List[str] = []
        skip_next = False
        value_args = {"--asset-list", "--asset-preset", "--target-ticker", "--result-dir"}
        for a in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if a in value_args:
                skip_next = True
                continue
            if any(a.startswith(k + "=") for k in value_args):
                continue
            base_args.append(a)

        rows: List[Dict[str, object]] = []
        for ticker in tickers:
            safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in ticker).strip("_") or "asset"
            out_dir = batch_root / safe
            cmd = [sys.executable, sys.argv[0], *base_args, "--target-ticker", ticker, "--result-dir", str(out_dir)]
            print(f"\n[BATCH] {ticker} 실행")
            proc = subprocess.run(cmd)
            row: Dict[str, object] = {"ticker": ticker, "returncode": int(proc.returncode), "result_dir": str(out_dir)}
            summary_path = out_dir / f"{safe}_xgb_strength_combo_all_v8_6_34_summary.json"
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    sm = json.load(f)
                perf = sm.get("performance", {}).get("strategy_after_cost", {})
                row.update({
                    "final_capital": perf.get("final_capital"),
                    "cagr": perf.get("cagr"),
                    "mdd": perf.get("mdd"),
                    "sharpe": perf.get("sharpe"),
                    "sortino": perf.get("sortino"),
                    "calmar": perf.get("calmar"),
                    "avg_stock_weight": sm.get("average_weights", {}).get("avg_stock_weight"),
                    "turnover": sm.get("turnover", {}).get("annual_turnover_estimate"),
                    "offensive_activation_rate": sm.get("direction_strength_specialist", {}).get("offensive_activation_rate"),
                    "tier3_rate": sm.get("direction_strength_specialist", {}).get("offensive_tier_3_rate"),
                    "stale_hold_rate": sm.get("target_execution_diagnostics", {}).get("stale_offensive_hold_rate"),
                })
            rows.append(row)
        batch_df = pd.DataFrame(rows)
        batch_path = batch_root / "multi_asset_summary.csv"
        batch_df.to_csv(batch_path, index=False, encoding="utf-8-sig")
        print("\n[MULTI-ASSET 저장 완료]")
        print(f"- {batch_path}")
        return

    cfg = apply_speed_profile(Config(), args.speed_profile)
    if getattr(args, "target_ticker", None):
        cfg.target_ticker = str(args.target_ticker).upper()
    if args.adaptive_label:
        cfg.use_adaptive_label_policy = True
    if args.rolling_gate_opt:
        cfg.use_rolling_gate_optimization = True
    if args.n_jobs is not None:
        cfg.n_jobs = args.n_jobs
    if args.retrain_every is not None:
        cfg.retrain_every_n_days = args.retrain_every
    if args.policy_mode is not None:
        cfg.policy_mode = str(args.policy_mode)
    if args.no_policy_overlay:
        cfg.use_policy_overlay = False
    if getattr(args, "no_direction_strength", False):
        cfg.use_direction_strength_specialist = False
    if getattr(args, "up_strength_threshold", None) is not None:
        cfg.up_strength_bonus_threshold_1 = float(args.up_strength_threshold)
    if getattr(args, "bear_down_strength_threshold", None) is not None:
        cfg.bear_down_strength_cut_threshold_1 = float(args.bear_down_strength_threshold)
    if getattr(args, "allocation_downrisk_weight", None) is not None:
        cfg.allocation_downrisk_weight = float(args.allocation_downrisk_weight)
    if getattr(args, "enable_bear_cut", False):
        cfg.use_bear_specialist_cut = True
    if getattr(args, "offensive_stock_1", None) is not None:
        cfg.up_strength_offensive_stock_weight_1 = float(args.offensive_stock_1)
        cfg.up_strength_single_20d_stock_weight = float(args.offensive_stock_1)
    if getattr(args, "offensive_stock_2", None) is not None:
        cfg.up_strength_offensive_stock_weight_2 = float(args.offensive_stock_2)
        cfg.up_strength_pair_10d_20d_stock_weight = float(args.offensive_stock_2)
    if getattr(args, "offensive_stock_3", None) is not None:
        cfg.up_strength_offensive_stock_weight_3 = float(args.offensive_stock_3)
        cfg.up_strength_all3_base_stock_weight = float(args.offensive_stock_3)
        cfg.up_strength_all3_base_stock_weight = float(args.offensive_stock_3)
    if getattr(args, "up_strength_threshold_2", None) is not None:
        cfg.up_strength_bonus_threshold_2 = float(args.up_strength_threshold_2)
    if getattr(args, "up_strength_threshold_3", None) is not None:
        cfg.up_strength_bonus_threshold_3 = float(args.up_strength_threshold_3)
    if getattr(args, "low_vol_threshold_1", None) is not None:
        cfg.up_strength_low_vol_threshold_1 = float(args.low_vol_threshold_1)
    if getattr(args, "low_vol_threshold_2", None) is not None:
        cfg.up_strength_low_vol_threshold_2 = float(args.low_vol_threshold_2)
    if getattr(args, "low_vol_threshold_3", None) is not None:
        cfg.up_strength_low_vol_threshold_3 = float(args.low_vol_threshold_3)
    if getattr(args, "base_normal_stock", None) is not None:
        cfg.normal_stock_weight = float(args.base_normal_stock)
        rem = max(0.0, 1.0 - cfg.normal_stock_weight); cfg.normal_bond_weight = rem * 2/3; cfg.normal_cash_weight = rem * 1/3
    if getattr(args, "base_watch_stock", None) is not None:
        cfg.watch_stock_weight = float(args.base_watch_stock)
        rem = max(0.0, 1.0 - cfg.watch_stock_weight); cfg.watch_bond_weight = rem * 0.625; cfg.watch_cash_weight = rem * 0.375
    if getattr(args, "base_riskoff_stock", None) is not None:
        cfg.risk_off_stock_weight = float(args.base_riskoff_stock)
        rem = max(0.0, 1.0 - cfg.risk_off_stock_weight); cfg.risk_off_bond_weight = rem * 0.6363636364; cfg.risk_off_cash_weight = rem * 0.3636363636
    if getattr(args, "base_extreme_stock", None) is not None:
        cfg.extreme_stock_weight = float(args.base_extreme_stock)
        rem = max(0.0, 1.0 - cfg.extreme_stock_weight); cfg.extreme_bond_weight = rem * 0.6428571429; cfg.extreme_cash_weight = rem * 0.3571428571
    if getattr(args, "extreme_high_vol_threshold", None) is not None:
        cfg.extreme_high_vol_threshold = float(args.extreme_high_vol_threshold)
        cfg.extreme_downrisk_threshold = float(args.extreme_high_vol_threshold)
    if getattr(args, "riskoff_stock", None) is not None:
        cfg.risk_off_stock_weight = float(args.riskoff_stock)
        cfg.risk_off_bond_weight, cfg.risk_off_cash_weight = _risk_off_bond_cash_from_stock(cfg.risk_off_stock_weight)
    if getattr(args, "watch_stock", None) is not None:
        cfg.watch_stock_weight = float(args.watch_stock)
        remain = max(0.0, 1.0 - cfg.watch_stock_weight)
        cfg.watch_bond_weight = remain * 0.75
        cfg.watch_cash_weight = remain * 0.25
    # policy별 기본 result_dir. 사용자가 --result-dir를 주면 아래에서 덮어쓴다.
    cfg.result_dir = f"results_xgb_strength_combo_all_v8_6_34_{cfg.policy_mode}"
    if cfg.policy_mode == "defensive_risk":
        cfg.gate_normal_high_vol_threshold = 0.32
        cfg.gate_high_vol_threshold = 0.58
        cfg.gate_riskoff_downrisk_threshold = 0.50
        cfg.no_trade_band = max(cfg.no_trade_band, 0.11)
        cfg.risk_off_stock_weight = min(cfg.risk_off_stock_weight, 0.52)
        cfg.risk_off_bond_weight, cfg.risk_off_cash_weight = _risk_off_bond_cash_from_stock(cfg.risk_off_stock_weight)
    elif cfg.policy_mode == "aggressive_dynamic":
        cfg.gate_normal_high_vol_threshold = 0.45
        cfg.gate_high_vol_threshold = 0.68
        cfg.gate_riskoff_downrisk_threshold = 0.54
        cfg.no_trade_band = max(cfg.no_trade_band, 0.11)
    elif cfg.policy_mode in {"direction_strength_specialist", "strength_specialist", "ds_specialist", "portfolio_model"}:
        # v8.6.34: Volatility-Base Strong-Override Upside Strength Trigger
        # - 5D/10D/20D UP_STRENGTHENING specialist를 각각 학습한다.
        # - horizon별 train_rows = horizon * multiplier 구조로 학습 기간을 다르게 둔다.
        # - 하락 예측 모델은 allocation에서 제외하고 Stage1 high-vol만 위험 차단기로 쓴다.
        # - stale offensive hold는 기본 차단한다.
        cfg.gate_normal_high_vol_threshold = 0.55
        cfg.gate_high_vol_threshold = 0.74
        cfg.gate_riskoff_downrisk_threshold = 0.74
        cfg.gate_watch_downrisk_threshold = 0.80
        cfg.rebalance_every_n_days = min(int(getattr(cfg, "rebalance_every_n_days", 5)), 5)
        cfg.no_trade_band = min(float(getattr(cfg, "no_trade_band", 0.12)), 0.12)
        cfg.emergency_cooldown_days = min(int(getattr(cfg, "emergency_cooldown_days", 5)), 5)
        # v8.6.34: 기본 상태는 prob_high_vol 기반 포트폴리오, 강한 신호만 즉시 override한다.
        cfg.use_vol_probability_base_allocation = True
        cfg.vol_base_stock_lt_25 = 0.78
        cfg.vol_base_stock_lt_35 = 0.74
        cfg.vol_base_stock_lt_50 = 0.68
        cfg.vol_base_stock_lt_65 = 0.60
        cfg.vol_base_stock_lt_75 = 0.52
        cfg.vol_base_stock_lt_86 = 0.42
        cfg.vol_base_stock_ge_86 = 0.30
        cfg.vol_base_bond_ratio_of_defensive = 0.65
        # regime bucket은 diagnostics/gate 용도로 유지한다.
        cfg.normal_stock_weight = 0.72
        cfg.normal_bond_weight = 0.18
        cfg.normal_cash_weight = 0.10
        cfg.watch_stock_weight = 0.62
        cfg.watch_bond_weight = 0.25
        cfg.watch_cash_weight = 0.13
        cfg.high_vol_stock_weight = 0.55
        cfg.high_vol_bond_weight = 0.30
        cfg.high_vol_cash_weight = 0.15
        cfg.risk_off_stock_weight = 0.45
        cfg.risk_off_bond_weight = 0.37
        cfg.risk_off_cash_weight = 0.18
        cfg.extreme_high_vol_threshold = 0.86
        cfg.extreme_downrisk_threshold = 0.86
        cfg.extreme_stock_weight = 0.30
        cfg.extreme_bond_weight = 0.45
        cfg.extreme_cash_weight = 0.25
        cfg.emergency_high_vol_threshold = 0.88
        cfg.emergency_combined_high_vol_threshold = 0.78
        cfg.emergency_combined_down_threshold = 0.78
        cfg.allocation_downrisk_weight = float(getattr(cfg, "allocation_downrisk_weight", 0.0))
        cfg.use_bear_specialist_cut = bool(getattr(cfg, "use_bear_specialist_cut", False))
        cfg.overall_risk_high_vol_weight = 1.0
        cfg.overall_risk_down_weight = 0.0
        cfg.overall_risk_down_minus_up_weight = 0.0
        cfg.up_strength_bonus_threshold_1 = float(getattr(cfg, "up_strength_bonus_threshold_1", 0.30))
        cfg.up_strength_bonus_threshold_2 = float(getattr(cfg, "up_strength_bonus_threshold_2", 0.38))
        cfg.up_strength_bonus_threshold_3 = float(getattr(cfg, "up_strength_bonus_threshold_3", 0.45))
        # v8.6.34: 기존 Tier2는 기본 비활성화한다. 필요하면 --enable-tier2로 복구한다.
        cfg.disable_tier2_signal = True
        if cfg.policy_mode == "portfolio_model":
            cfg.enable_portfolio_policy_model = True
        cfg.force_strong_offensive_rebalance = True
        cfg.force_tier3_rebalance = True
        cfg.force_full_stock_rebalance = True
        cfg.up_strength_low_vol_threshold_1 = float(getattr(cfg, "up_strength_low_vol_threshold_1", 0.82))
        cfg.up_strength_low_vol_threshold_2 = float(getattr(cfg, "up_strength_low_vol_threshold_2", 0.72))
        cfg.up_strength_low_vol_threshold_3 = float(getattr(cfg, "up_strength_low_vol_threshold_3", 0.68))
        # bear block 제거: prob_bear_down_strengthening 미사용
        cfg.direction_strength_max_stock_bonus = 1.0
        cfg.enable_stale_offensive_decay = bool(getattr(cfg, "enable_stale_offensive_decay", True))
        # v8.6.34: 공격 전환을 더 오래 허용하되, 과도한 stale hold는 차단한다.
        cfg.stale_offensive_stock_gap_threshold = float(getattr(cfg, "stale_offensive_stock_gap_threshold", 0.12))
        cfg.stale_offensive_up_strength_reset_threshold = float(getattr(cfg, "stale_offensive_up_strength_reset_threshold", 0.20))
        cfg.stale_offensive_high_vol_threshold = float(getattr(cfg, "stale_offensive_high_vol_threshold", 0.72))
        cfg.use_direction_strength_specialist = True
        cfg.multi_strength_horizons = (5, 10, 20)
        cfg.up_strength_weight_5d = 0.00
        cfg.up_strength_weight_10d = 0.20
        cfg.up_strength_weight_20d = 0.80
        cfg.down_strength_weight_5d = 0.00
        cfg.down_strength_weight_10d = 0.20
        cfg.down_strength_weight_20d = 0.80
        cfg.direction_strength_feature_set = "horizon_5_10_20_pruned"
        cfg.up_strength_full_stock_score_threshold = float(getattr(cfg, "up_strength_full_stock_score_threshold", 0.50))
        cfg.up_strength_full_stock_10d_threshold = float(getattr(cfg, "up_strength_full_stock_10d_threshold", 0.38))
        cfg.up_strength_full_stock_20d_threshold = float(getattr(cfg, "up_strength_full_stock_20d_threshold", 0.42))
        cfg.up_strength_full_stock_high_vol_threshold = float(getattr(cfg, "up_strength_full_stock_high_vol_threshold", 0.58))
    elif cfg.policy_mode == "return_seeking":
        cfg.gate_normal_high_vol_threshold = 0.35
        cfg.gate_high_vol_threshold = 0.62
        cfg.gate_riskoff_downrisk_threshold = 0.52

    # CLI direct overrides must be applied after policy defaults.
    if getattr(args, "extreme_high_vol_threshold", None) is not None:
        cfg.extreme_high_vol_threshold = float(args.extreme_high_vol_threshold)
        cfg.extreme_downrisk_threshold = float(args.extreme_high_vol_threshold)
    if getattr(args, "riskoff_stock", None) is not None:
        cfg.risk_off_stock_weight = float(args.riskoff_stock)
        cfg.risk_off_bond_weight, cfg.risk_off_cash_weight = _risk_off_bond_cash_from_stock(cfg.risk_off_stock_weight)
    if getattr(args, "watch_stock", None) is not None:
        cfg.watch_stock_weight = float(args.watch_stock)
        remain = max(0.0, 1.0 - cfg.watch_stock_weight)
        cfg.watch_bond_weight = remain * 0.75
        cfg.watch_cash_weight = remain * 0.25

    # v8.6.21 direct low-base overrides after policy defaults.
    if getattr(args, "base_normal_stock", None) is not None:
        cfg.normal_stock_weight = float(args.base_normal_stock)
        rem = max(0.0, 1.0 - cfg.normal_stock_weight); cfg.normal_bond_weight = rem * 2/3; cfg.normal_cash_weight = rem * 1/3
    if getattr(args, "base_watch_stock", None) is not None:
        cfg.watch_stock_weight = float(args.base_watch_stock)
        rem = max(0.0, 1.0 - cfg.watch_stock_weight); cfg.watch_bond_weight = rem * 0.625; cfg.watch_cash_weight = rem * 0.375
    if getattr(args, "base_riskoff_stock", None) is not None:
        cfg.risk_off_stock_weight = float(args.base_riskoff_stock)
        rem = max(0.0, 1.0 - cfg.risk_off_stock_weight); cfg.risk_off_bond_weight = rem * 0.6363636364; cfg.risk_off_cash_weight = rem * 0.3636363636
    if getattr(args, "base_extreme_stock", None) is not None:
        cfg.extreme_stock_weight = float(args.base_extreme_stock)
        rem = max(0.0, 1.0 - cfg.extreme_stock_weight); cfg.extreme_bond_weight = rem * 0.6428571429; cfg.extreme_cash_weight = rem * 0.3571428571
    if getattr(args, "offensive_stock_3", None) is not None:
        cfg.up_strength_offensive_stock_weight_3 = float(args.offensive_stock_3)
        cfg.up_strength_all3_base_stock_weight = float(args.offensive_stock_3)
    if getattr(args, "up_strength_threshold_2", None) is not None:
        cfg.up_strength_bonus_threshold_2 = float(args.up_strength_threshold_2)
    if getattr(args, "up_strength_threshold_3", None) is not None:
        cfg.up_strength_bonus_threshold_3 = float(args.up_strength_threshold_3)
    if getattr(args, "low_vol_threshold_3", None) is not None:
        cfg.up_strength_low_vol_threshold_3 = float(args.low_vol_threshold_3)
    if getattr(args, "disable_stale_offensive_decay", False):
        cfg.enable_stale_offensive_decay = False

    if args.result_dir:
        cfg.result_dir = args.result_dir
    if getattr(args, "h10_down_only", False):
        cfg.down_risk_weight_h10 = 1.0
        cfg.down_risk_weight_h20 = 0.0
    if getattr(args, "no_trade_band", None) is not None:
        cfg.no_trade_band = float(args.no_trade_band)
    if getattr(args, "rebalance_every", None) is not None:
        cfg.rebalance_every_n_days = int(args.rebalance_every)
    if getattr(args, "emergency_cooldown", None) is not None:
        cfg.emergency_cooldown_days = int(args.emergency_cooldown)
    if getattr(args, "four_regime", False):
        cfg.use_three_regime_allocation = False
    if getattr(args, "no_extreme_risk", False):
        cfg.use_extreme_risk_cut = False
    if getattr(args, "enable_stale_offensive_decay", False):
        cfg.enable_stale_offensive_decay = True
    if getattr(args, "stale_offensive_gap", None) is not None:
        cfg.stale_offensive_stock_gap_threshold = float(args.stale_offensive_gap)
    if getattr(args, "stale_offensive_reset_threshold", None) is not None:
        cfg.stale_offensive_up_strength_reset_threshold = float(args.stale_offensive_reset_threshold)
    if getattr(args, "stale_offensive_high_vol_threshold", None) is not None:
        cfg.stale_offensive_high_vol_threshold = float(args.stale_offensive_high_vol_threshold)
    if getattr(args, "execution_lag_days", None) is not None:
        cfg.execution_lag_days = int(args.execution_lag_days)
    if getattr(args, "max_train_rows", None) is not None:
        cfg.max_train_rows = int(args.max_train_rows)
    if getattr(args, "no_horizon_train_window", False):
        cfg.use_horizon_train_window = False
        cfg.direction_strength_use_horizon_train_window = False
    if getattr(args, "horizon_train_min_rows", None) is not None:
        cfg.horizon_train_min_rows = int(args.horizon_train_min_rows)
    if getattr(args, "horizon_train_max_cap", None) is not None:
        cfg.horizon_train_max_rows_cap = int(args.horizon_train_max_cap)
    if getattr(args, "horizon_train_multiplier", None) is not None:
        m = float(args.horizon_train_multiplier)
        cfg.horizon_train_multiplier_5d = m
        cfg.horizon_train_multiplier_10d = m
        cfg.horizon_train_multiplier_20d = m
    if getattr(args, "h5_train_multiplier", None) is not None:
        cfg.horizon_train_multiplier_5d = float(args.h5_train_multiplier)
    if getattr(args, "h10_train_multiplier", None) is not None:
        cfg.horizon_train_multiplier_10d = float(args.h10_train_multiplier)
    if getattr(args, "h20_train_multiplier", None) is not None:
        cfg.horizon_train_multiplier_20d = float(args.h20_train_multiplier)
    if getattr(args, "allow_cash_download_fallback", False):
        cfg.allow_cash_download_fallback = True
    if getattr(args, "multi_strength_horizons", None) is not None:
        cfg.multi_strength_horizons = tuple(int(x.strip()) for x in str(args.multi_strength_horizons).split(",") if x.strip())
    if getattr(args, "up_strength_weight_5d", None) is not None:
        cfg.up_strength_weight_5d = float(args.up_strength_weight_5d)
    if getattr(args, "up_strength_weight_10d", None) is not None:
        cfg.up_strength_weight_10d = float(args.up_strength_weight_10d)
    if getattr(args, "up_strength_weight_20d", None) is not None:
        cfg.up_strength_weight_20d = float(args.up_strength_weight_20d)
    if getattr(args, "up_confirm_10d_threshold_2", None) is not None:
        cfg.up_strength_confirm_10d_threshold_2 = float(args.up_confirm_10d_threshold_2)
    if getattr(args, "up_confirm_20d_threshold_2", None) is not None:
        cfg.up_strength_confirm_20d_threshold_2 = float(args.up_confirm_20d_threshold_2)
    if getattr(args, "up_confirm_10d_threshold_3", None) is not None:
        cfg.up_strength_confirm_10d_threshold_3 = float(args.up_confirm_10d_threshold_3)
    if getattr(args, "up_confirm_20d_threshold_3", None) is not None:
        cfg.up_strength_confirm_20d_threshold_3 = float(args.up_confirm_20d_threshold_3)
    if getattr(args, "up_pred_threshold_5d", None) is not None:
        cfg.up_strength_pred_threshold_5d = float(args.up_pred_threshold_5d)
    if getattr(args, "up_pred_threshold_10d", None) is not None:
        cfg.up_strength_pred_threshold_10d = float(args.up_pred_threshold_10d)
    if getattr(args, "up_pred_threshold_20d", None) is not None:
        cfg.up_strength_pred_threshold_20d = float(args.up_pred_threshold_20d)
    if getattr(args, "disable_vol_base_allocation", False):
        cfg.use_vol_probability_base_allocation = False
    if getattr(args, "vol_base_bond_ratio", None) is not None:
        cfg.vol_base_bond_ratio_of_defensive = float(args.vol_base_bond_ratio)
    if getattr(args, "full_stock_score_threshold", None) is not None:
        cfg.up_strength_full_stock_score_threshold = float(args.full_stock_score_threshold)
    if getattr(args, "full_stock_10d_threshold", None) is not None:
        cfg.up_strength_full_stock_10d_threshold = float(args.full_stock_10d_threshold)
    if getattr(args, "full_stock_20d_threshold", None) is not None:
        cfg.up_strength_full_stock_20d_threshold = float(args.full_stock_20d_threshold)
    if getattr(args, "full_stock_high_vol_threshold", None) is not None:
        cfg.up_strength_full_stock_high_vol_threshold = float(args.full_stock_high_vol_threshold)
    if getattr(args, "disable_strong_override", False):
        cfg.force_strong_offensive_rebalance = False
        cfg.force_tier3_rebalance = False
        cfg.force_full_stock_rebalance = False
    if getattr(args, "enable_tier2", False):
        cfg.disable_tier2_signal = False
    if getattr(args, "disable_tier2", False):
        cfg.disable_tier2_signal = True
    if getattr(args, "short_mid_mode", None) is not None:
        cfg.short_mid_confirm_mode = str(args.short_mid_mode)
    if getattr(args, "short_mid_action_signal", None) is not None:
        cfg.short_mid_action_signal = str(args.short_mid_action_signal)
    if getattr(args, "short_mid_p5", None) is not None:
        cfg.short_mid_p5_threshold = float(args.short_mid_p5)
    if getattr(args, "short_mid_p10", None) is not None:
        cfg.short_mid_p10_threshold = float(args.short_mid_p10)
    if getattr(args, "short_mid_p20", None) is not None:
        cfg.short_mid_p20_threshold = float(args.short_mid_p20)
    if getattr(args, "short_mid_high_vol", None) is not None:
        cfg.short_mid_high_vol_threshold = float(args.short_mid_high_vol)
    if getattr(args, "short_mid_strong_high_vol", None) is not None:
        cfg.short_mid_strong_high_vol_threshold = float(args.short_mid_strong_high_vol)
    if getattr(args, "short_mid_loose_high_vol", None) is not None:
        cfg.short_mid_loose_high_vol_threshold = float(args.short_mid_loose_high_vol)
    if getattr(args, "short_mid_use_score", False):
        cfg.short_mid_use_score_filter = True
    if getattr(args, "short_mid_score", None) is not None:
        cfg.short_mid_score_threshold = float(args.short_mid_score)
    if getattr(args, "short_mid_tier1_upgrade_stock", None) is not None:
        cfg.short_mid_tier1_upgrade_stock_weight = float(args.short_mid_tier1_upgrade_stock)
    if getattr(args, "short_mid_tier2_stock", None) is not None:
        cfg.short_mid_tier2_stock_weight = float(args.short_mid_tier2_stock)
    if getattr(args, "short_mid_base_upgrade_stock", None) is not None:
        cfg.short_mid_base_upgrade_stock_weight = float(args.short_mid_base_upgrade_stock)
    if getattr(args, "strength_combo_mode", None) is not None:
        cfg.strength_combo_policy_mode = str(args.strength_combo_mode)
        cfg.strength_combo_policy_enabled = str(args.strength_combo_mode).lower() not in {"off", "none", "false"}
    if getattr(args, "disable_strength_combo_policy", False):
        cfg.strength_combo_policy_enabled = False
        cfg.strength_combo_policy_mode = "off"
    if getattr(args, "strength_combo_no_high_vol", False):
        cfg.strength_combo_use_high_vol_filter = False
    if getattr(args, "strength_combo_high_vol", None) is not None:
        cfg.strength_combo_high_vol_threshold = float(args.strength_combo_high_vol)
    if getattr(args, "strength_combo_use_score", False):
        cfg.strength_combo_use_score_filter = True
    if getattr(args, "strength_combo_score", None) is not None:
        cfg.strength_combo_score_threshold = float(args.strength_combo_score)
    if getattr(args, "combo_single_5d_stock", None) is not None:
        cfg.strength_combo_single_5d_stock_weight = float(args.combo_single_5d_stock)
    if getattr(args, "combo_single_10d_stock", None) is not None:
        cfg.strength_combo_single_10d_stock_weight = float(args.combo_single_10d_stock)
    if getattr(args, "combo_single_20d_stock", None) is not None:
        cfg.strength_combo_single_20d_stock_weight = float(args.combo_single_20d_stock)
    if getattr(args, "combo_5d_10d_stock", None) is not None:
        cfg.strength_combo_pair_5d_10d_stock_weight = float(args.combo_5d_10d_stock)
    if getattr(args, "combo_5d_20d_stock", None) is not None:
        cfg.strength_combo_pair_5d_20d_stock_weight = float(args.combo_5d_20d_stock)
    if getattr(args, "combo_10d_20d_stock", None) is not None:
        cfg.strength_combo_pair_10d_20d_stock_weight = float(args.combo_10d_20d_stock)
    if getattr(args, "combo_all3_stock", None) is not None:
        cfg.strength_combo_all3_stock_weight = float(args.combo_all3_stock)
    if getattr(args, "enable_portfolio_model", False):
        cfg.enable_portfolio_policy_model = True
    if str(getattr(cfg, "policy_mode", "")) == "portfolio_model":
        cfg.enable_portfolio_policy_model = True
    if getattr(args, "portfolio_policy_horizon", None) is not None:
        cfg.portfolio_policy_horizon = int(args.portfolio_policy_horizon)
    if getattr(args, "portfolio_policy_min_train_rows", None) is not None:
        cfg.portfolio_policy_min_train_rows = int(args.portfolio_policy_min_train_rows)
    if getattr(args, "portfolio_policy_max_train_rows", None) is not None:
        cfg.portfolio_policy_max_train_rows = int(args.portfolio_policy_max_train_rows)
    if getattr(args, "portfolio_model_min_confidence", None) is not None:
        cfg.portfolio_model_min_confidence = float(args.portfolio_model_min_confidence)
    if getattr(args, "portfolio_model_force_rebalance", False):
        cfg.portfolio_model_force_rebalance = True
    if getattr(args, "optimize_tier_weights", False):
        cfg.enable_tier_weight_optimizer = True
    if getattr(args, "tier_opt_train_rows", None) is not None:
        cfg.tier_weight_opt_train_rows = int(args.tier_opt_train_rows)
    if getattr(args, "tier_opt_test_rows", None) is not None:
        cfg.tier_weight_opt_test_rows = int(args.tier_opt_test_rows)
    if getattr(args, "tier_opt_min_train_rows", None) is not None:
        cfg.tier_weight_opt_min_train_rows = int(args.tier_opt_min_train_rows)
    if getattr(args, "tier_opt_score_profile", None) is not None:
        cfg.tier_weight_opt_score_profile = str(args.tier_opt_score_profile)
    if getattr(args, "tier_opt_no_base_lt25", False):
        cfg.tier_weight_opt_include_base_lt25 = False
    cfg.tier_weight_opt_base_lt25_grid = _parse_grid_arg(getattr(args, "tier_opt_base_lt25_grid", None), cfg.tier_weight_opt_base_lt25_grid)
    cfg.tier_weight_opt_tier1_grid = _parse_grid_arg(getattr(args, "tier_opt_tier1_grid", None), cfg.tier_weight_opt_tier1_grid)
    cfg.tier_weight_opt_tier2_grid = _parse_grid_arg(getattr(args, "tier_opt_tier2_grid", None), cfg.tier_weight_opt_tier2_grid)
    cfg.tier_weight_opt_tier3_grid = _parse_grid_arg(getattr(args, "tier_opt_tier3_grid", None), cfg.tier_weight_opt_tier3_grid)
    cfg.tier_weight_opt_full_grid = _parse_grid_arg(getattr(args, "tier_opt_full_grid", None), cfg.tier_weight_opt_full_grid)
    # direction_strength_horizon 및 multi_strength_horizons가 기존 horizons에 없으면 자동 포함한다.
    required_horizons = set(tuple(cfg.horizons)) | {int(getattr(cfg, "direction_strength_horizon", 20))} | set(get_multi_strength_horizons(cfg))
    cfg.horizons = tuple(sorted(required_horizons))

    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] 데이터 다운로드")
    target = download_ohlcv(cfg.target_ticker, cfg.start_date, cfg.end_date)
    bond_close = download_close(cfg.bond_ticker, cfg.start_date, cfg.end_date)
    try:
        cash_close = download_close(cfg.cash_ticker, cfg.start_date, cfg.end_date)
    except Exception as exc:
        if not cfg.allow_cash_download_fallback:
            raise RuntimeError(
                f"{cfg.cash_ticker} 다운로드 실패. "
                f"현금 수익률을 0으로 대체하면 백테스트가 왜곡될 수 있습니다."
            ) from exc
        warnings.warn(
            f"{cfg.cash_ticker} 다운로드 실패로 cash return을 0으로 대체합니다: {exc}",
            RuntimeWarning,
        )
        cash_close = pd.Series(index=target.index, data=np.nan, name=cfg.cash_ticker)

    print("[2/5] 피처 생성")
    df, feature_cols = build_features(target, cfg.horizons)
    df = ensure_direction_strength_helper_columns(df)
    returns_df = build_aligned_forward_returns(
        target_close=df["Close"],
        bond_close=bond_close,
        cash_close=cash_close,
        target_index=df.index,
        execution_lag_days=cfg.execution_lag_days,
    )
    df = pd.concat(
        [
            df,
            returns_df[["stock_next_return", "bond_next_return", "cash_next_return"]],
        ],
        axis=1,
    ).copy()
    print(f"    피처 수: {len(feature_cols)}")
    print(f"    horizons: {cfg.horizons}")
    print(f"    adaptive_label: {cfg.use_adaptive_label_policy}")
    print(f"    rolling_gate_opt: {cfg.use_rolling_gate_optimization}")
    print(f"    execution_lag_days: {cfg.execution_lag_days}")
    print(f"    max_train_rows: {cfg.max_train_rows}")
    print(f"    horizon_train_window: {horizon_train_window_config(cfg)}")
    print(f"    policy_mode: {cfg.policy_mode}")
    print(f"    direction_strength_specialist: {cfg.use_direction_strength_specialist}")

    print("[3/5] Walk-forward Stage1 + horizon train-window Direction-Strength Specialist 예측")
    pred_raw = run_walk_forward(df, feature_cols, cfg)

    if bool(getattr(cfg, "enable_portfolio_policy_model", False)):
        print("[3.5/5] Separate PortfolioPolicyModel 학습/예측")
        pred_raw = run_portfolio_policy_model(pred_raw, cfg)

    print("[4/5] 배분/백테스트")
    condition_report_df: Optional[pd.DataFrame] = None
    condition_meta: Dict[str, object] = {}

    if getattr(args, "condition_search", False):
        print("    조건 객관화 탐색 실행")
        print(f"    split_date: {args.condition_split_date}")
        print(f"    grid_size: {args.condition_grid_size}")
        print(f"    score_profile: {args.score_profile}")
        selected_cfg, condition_report_df, condition_meta = run_condition_search(
            pred_raw=pred_raw,
            base_cfg=cfg,
            split_date=args.condition_split_date,
            grid_size=args.condition_grid_size,
            score_profile=args.score_profile,
        )
        # 선택된 조건만 최종 전체 구간에 적용한다. 모델 예측은 pred_raw를 재사용한다.
        selected_cfg.result_dir = cfg.result_dir
        cfg = selected_cfg
        print(f"    selected_candidate: {condition_meta.get('selected_candidate')}")

    pred_df, gate_usage = apply_allocation(pred_raw, cfg)
    pred_df.attrs.update(pred_raw.attrs)

    tier_weight_outputs: Dict[str, object] = {}
    if bool(getattr(cfg, "enable_tier_weight_optimizer", False)):
        print("[4.5/5] TierWeightOptimizer: Tier2 포함 Tier별 목표 비중 Walk-forward 최적화")
        tier_weight_outputs = run_tier_weight_optimizer(pred_df, cfg)

    print("[5/5] 결과 저장")
    summary = build_summary(pred_df, feature_cols, gate_usage, cfg)
    if tier_weight_outputs:
        summary["tier_weight_optimizer"] = tier_weight_outputs.get("summary", {})
    if condition_meta:
        summary["condition_search"] = condition_meta
        add_condition_period_summary(
            summary=summary,
            pred_df=pred_df,
            cfg=cfg,
            split_date=str(condition_meta["split_date"]),
        )

    safe_ticker = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(cfg.target_ticker)).strip("_") or "asset"
    file_prefix = f"{safe_ticker}_xgb_strength_combo_all_v8_6_34"
    pred_path = result_dir / f"{file_prefix}_predictions.csv"
    summary_path = result_dir / f"{file_prefix}_summary.json"
    latest_path = result_dir / f"{file_prefix}_latest.json"
    importance_stage1_path = result_dir / f"{file_prefix}_stage1_feature_importance.csv"
    importance_up_path = result_dir / f"{file_prefix}_up_feature_importance.csv"
    importance_down_path = result_dir / f"{file_prefix}_downrisk_feature_importance.csv"
    importance_down_price_trend_path = result_dir / f"{file_prefix}_downrisk_price_trend_feature_importance.csv"
    importance_down_price_volume_path = result_dir / f"{file_prefix}_downrisk_price_volume_feature_importance.csv"
    importance_down_volatility_path = result_dir / f"{file_prefix}_downrisk_volatility_feature_importance.csv"
    condition_search_path = result_dir / f"{file_prefix}_condition_search.csv"
    tier_weight_selected_path = result_dir / f"{file_prefix}_tier_weight_selected_folds.csv"
    tier_weight_oos_daily_path = result_dir / f"{file_prefix}_tier_weight_oos_daily.csv"
    tier_weight_grid_path = result_dir / f"{file_prefix}_tier_weight_full_grid.csv"

    diagnostics: Dict[str, pd.DataFrame] = {}
    if not getattr(args, "no_diagnostics", False):
        diagnostics = build_optimization_diagnostics(pred_df, summary)
        summary["optimization_diagnostics_summary"] = diagnostics_summary(diagnostics)

    export_pred_df = drop_weak_probability_output_columns(pred_df)
    export_pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    tier_weight_paths: List[Path] = []
    if tier_weight_outputs:
        selected_df = tier_weight_outputs.get("selected_folds")
        oos_daily_df = tier_weight_outputs.get("oos_daily")
        full_grid_df = tier_weight_outputs.get("full_grid")
        if isinstance(selected_df, pd.DataFrame) and not selected_df.empty:
            selected_df.to_csv(tier_weight_selected_path, index=False, encoding="utf-8-sig")
            tier_weight_paths.append(tier_weight_selected_path)
        if isinstance(oos_daily_df, pd.DataFrame) and not oos_daily_df.empty:
            oos_daily_df.to_csv(tier_weight_oos_daily_path, index=False, encoding="utf-8-sig")
            tier_weight_paths.append(tier_weight_oos_daily_path)
        if isinstance(full_grid_df, pd.DataFrame) and not full_grid_df.empty:
            full_grid_df.to_csv(tier_weight_grid_path, index=False, encoding="utf-8-sig")
            tier_weight_paths.append(tier_weight_grid_path)
        summary.setdefault("tier_weight_optimizer", {})["output_files"] = {
            "selected_folds": str(tier_weight_selected_path) if tier_weight_selected_path in tier_weight_paths else None,
            "oos_daily": str(tier_weight_oos_daily_path) if tier_weight_oos_daily_path in tier_weight_paths else None,
            "full_grid": str(tier_weight_grid_path) if tier_weight_grid_path in tier_weight_paths else None,
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary["latest_prediction"], f, ensure_ascii=False, indent=2)

    pd.Series(summary.get("stage1_feature_importance_mean", {}), name="importance").to_csv(importance_stage1_path, encoding="utf-8-sig")
    pd.Series(summary.get("up_feature_importance_mean", {}), name="importance").to_csv(importance_up_path, encoding="utf-8-sig")
    pd.Series(summary.get("downrisk_feature_importance_mean", {}), name="importance").to_csv(importance_down_path, encoding="utf-8-sig")
    pd.Series(summary.get("downrisk_price_trend_feature_importance_mean", {}), name="importance").to_csv(importance_down_price_trend_path, encoding="utf-8-sig")
    pd.Series(summary.get("downrisk_price_volume_feature_importance_mean", {}), name="importance").to_csv(importance_down_price_volume_path, encoding="utf-8-sig")
    pd.Series(summary.get("downrisk_volatility_feature_importance_mean", {}), name="importance").to_csv(importance_down_volatility_path, encoding="utf-8-sig")
    if condition_report_df is not None:
        condition_report_df.to_csv(condition_search_path, index=False, encoding="utf-8-sig")

    diagnostic_paths: List[Path] = []
    for name, diag_df in diagnostics.items():
        if diag_df is not None and not diag_df.empty:
            pth = result_dir / f"{file_prefix}_{name}.csv"
            diag_df.to_csv(pth, index=False, encoding="utf-8-sig")
            diagnostic_paths.append(pth)

    print_summary(summary)
    print("\n[저장 완료]")
    print(f"- {pred_path}")
    print(f"- {summary_path}")
    print(f"- {latest_path}")
    print(f"- {importance_stage1_path}")
    print(f"- {importance_up_path}")
    print(f"- {importance_down_path}")
    print(f"- {importance_down_price_trend_path}")
    print(f"- {importance_down_price_volume_path}")
    print(f"- {importance_down_volatility_path}")
    if condition_report_df is not None:
        print(f"- {condition_search_path}")
    if 'tier_weight_paths' in locals():
        for pth in tier_weight_paths:
            print(f"- {pth}")
    for pth in diagnostic_paths:
        print(f"- {pth}")


if __name__ == "__main__":
    main()
