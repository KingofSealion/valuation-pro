"""
valuation_utils.py - Valuation Calculation Utilities
- WACC (Weighted Average Cost of Capital)
- Synthetic Credit Rating
- Adjusted Beta
- Terminal Growth Cap
- Lifecycle Classification
- Growth Decay & Convergence Logic
"""
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


# ==================== Lifecycle Classification ====================

class LifecycleStage(Enum):
    """회사 성장 단계 분류"""
    HYPER_GROWTH = "hyper_growth"    # >20% 성장
    HIGH_GROWTH = "high_growth"       # 10-20% 성장
    STABLE = "stable"                 # <10% 성장


@dataclass
class LifecycleResult:
    """Lifecycle 분류 결과"""
    stage: LifecycleStage
    revenue_growth: float
    projection_years: int
    stage_label: str
    insight: str


def classify_lifecycle(
    revenue_growth: float,
    hyper_threshold: float = 0.20,
    high_threshold: float = 0.10
) -> LifecycleResult:
    """
    회사의 성장 단계를 분류하고 적절한 Projection Period 반환

    Args:
        revenue_growth: 매출 성장률 (decimal, e.g., 0.25 = 25%)
        hyper_threshold: Hyper-Growth 기준 (default 20%)
        high_threshold: High-Growth 기준 (default 10%)

    Returns:
        LifecycleResult with stage, projection_years, insights
    """
    if revenue_growth > hyper_threshold:
        return LifecycleResult(
            stage=LifecycleStage.HYPER_GROWTH,
            revenue_growth=revenue_growth,
            projection_years=10,
            stage_label="Hyper-Growth",
            insight=f"매출 성장률 {revenue_growth*100:.1f}%로 Hyper-Growth 단계입니다. "
                    f"10년 projection으로 점진적 성장 둔화를 반영합니다."
        )
    elif revenue_growth > high_threshold:
        return LifecycleResult(
            stage=LifecycleStage.HIGH_GROWTH,
            revenue_growth=revenue_growth,
            projection_years=7,
            stage_label="High-Growth",
            insight=f"매출 성장률 {revenue_growth*100:.1f}%로 High-Growth 단계입니다. "
                    f"7년 projection이 적절합니다."
        )
    else:
        return LifecycleResult(
            stage=LifecycleStage.STABLE,
            revenue_growth=revenue_growth,
            projection_years=5,
            stage_label="Stable",
            insight=f"매출 성장률 {revenue_growth*100:.1f}%로 Stable 단계입니다. "
                    f"5년 projection으로 충분합니다."
        )


# ==================== Growth Decay Functions ====================

def generate_growth_decay_schedule(
    initial_growth: float,
    terminal_growth: float,
    years: int,
    decay_type: str = 'linear'
) -> List[float]:
    """
    성장률 Decay 스케줄 생성 (Risk-Free Rate으로 수렴)

    Args:
        initial_growth: 첫 해 성장률
        terminal_growth: 최종 목표 성장률 (typically Risk-Free Rate)
        years: projection 기간
        decay_type: 'linear', 'exponential', 'front_loaded'

    Returns:
        연도별 성장률 리스트
    """
    if years <= 1:
        return [initial_growth]

    # Terminal growth보다는 약간 높게 마무리 (버퍼)
    final_growth = max(terminal_growth * 1.2, terminal_growth + 0.005)

    if decay_type == 'linear':
        # 선형 감소: 가장 일반적
        step = (initial_growth - final_growth) / (years - 1)
        return [max(initial_growth - (step * i), final_growth) for i in range(years)]

    elif decay_type == 'exponential':
        # 지수 감소: 초반에 빠르게 감소
        if initial_growth <= final_growth:
            return [initial_growth] * years
        ratio = (final_growth / initial_growth) ** (1 / (years - 1))
        return [max(initial_growth * (ratio ** i), final_growth) for i in range(years)]

    elif decay_type == 'front_loaded':
        # 앞쪽 가중: 처음 3년은 빠르게, 이후 완만하게
        schedule = []
        fast_decay_years = min(3, years // 2)
        mid_point = (initial_growth + final_growth) / 2

        # 처음 빠른 구간
        if fast_decay_years > 1:
            fast_step = (initial_growth - mid_point) / fast_decay_years
            for i in range(fast_decay_years):
                schedule.append(initial_growth - fast_step * i)

        # 나머지 완만한 구간
        remaining = years - len(schedule)
        if remaining > 0:
            slow_step = (mid_point - final_growth) / remaining
            for i in range(remaining):
                schedule.append(mid_point - slow_step * i)

        return [max(g, final_growth) for g in schedule[:years]]

    else:
        return [initial_growth] * years


# ==================== Margin Convergence Functions ====================

# 섹터별 Target EBITDA Margin (Mature 기업 기준)
SECTOR_TARGET_MARGINS = {
    'Technology': 0.30,
    'Consumer Cyclical': 0.15,
    'Communication Services': 0.30,
    'Healthcare': 0.25,
    'Financials': 0.35,
    'Consumer Defensive': 0.18,
    'Industrials': 0.15,
    'Energy': 0.25,
    'Utilities': 0.30,
    'Real Estate': 0.55,
    'Materials': 0.18,
    'Default': 0.20
}


def generate_margin_convergence_schedule(
    current_margin: float,
    target_margin: float,
    years: int,
    convergence_speed: float = 0.5
) -> List[float]:
    """
    EBITDA Margin 수렴 스케줄 생성

    Args:
        current_margin: 현재 EBITDA 마진
        target_margin: 목표 마진 (섹터 평균)
        years: projection 기간
        convergence_speed: 수렴 속도 (0.0=변화없음, 1.0=즉시 수렴)

    Returns:
        연도별 마진 리스트

    Note:
        Q1 답변에 따라 현재 마진을 우선 사용하고,
        섹터 평균으로 점진적 수렴
    """
    if years <= 1:
        return [current_margin]

    schedule = []
    gap = target_margin - current_margin

    for i in range(years):
        # 연도별로 gap의 일정 비율씩 수렴
        progress = (i + 1) / years * convergence_speed
        margin = current_margin + gap * progress
        schedule.append(margin)

    return schedule


def get_target_margin(sector: str, current_margin: float) -> Tuple[float, str]:
    """
    목표 마진 결정 (Q1 답변 반영: 회사 Historical 우선)

    Returns:
        (target_margin, source_description)
    """
    sector_target = SECTOR_TARGET_MARGINS.get(sector, SECTOR_TARGET_MARGINS['Default'])

    # 현재 마진이 섹터 평균과 비슷하면 유지
    if abs(current_margin - sector_target) < 0.05:  # 5%p 이내
        return current_margin, "현재 마진 유지 (섹터 평균과 유사)"

    # 현재 마진이 높으면 점진적 하락 예상 (경쟁 심화)
    if current_margin > sector_target + 0.10:  # 10%p 이상 높음
        target = (current_margin + sector_target) / 2  # 중간값으로 수렴
        return target, f"고마진 → 섹터 평균({sector_target*100:.0f}%)으로 수렴 예상"

    # 현재 마진이 낮으면 점진적 개선 가정 (운영 효율화)
    if current_margin < sector_target - 0.10:
        target = (current_margin + sector_target) / 2
        return target, f"저마진 → 섹터 평균({sector_target*100:.0f}%)으로 개선 예상"

    return sector_target, f"섹터 평균 ({sector_target*100:.0f}%)"


# ==================== CapEx Convergence (Q3: Option B - Gradual) ====================

def generate_capex_convergence_schedule(
    current_capex_pct: float,
    current_da_pct: float,
    years: int,
    target_capex_ratio: float = 1.05  # CapEx = D&A × 1.05 (Steady State)
) -> List[float]:
    """
    CapEx 수렴 스케줄 생성 (Q3 답변: 점진적 수렴)

    Steady State에서 CapEx ≈ D&A × 105-110%
    (유지 투자 + 소폭 성장 투자)

    Args:
        current_capex_pct: 현재 CapEx/Revenue 비율
        current_da_pct: 현재 D&A/Revenue 비율
        years: projection 기간
        target_capex_ratio: 목표 CapEx/D&A 비율 (default 1.05)

    Returns:
        연도별 CapEx/Revenue 비율 리스트
    """
    if years <= 1:
        return [current_capex_pct]

    # 목표 CapEx = D&A × target_ratio
    target_capex_pct = current_da_pct * target_capex_ratio

    # 선형 보간 (Linear Interpolation)
    schedule = []
    step = (target_capex_pct - current_capex_pct) / (years - 1)

    for i in range(years):
        capex = current_capex_pct + step * i
        # 최소/최대 제한
        capex = max(min(capex, 0.25), 0.02)  # 2% ~ 25%
        schedule.append(capex)

    return schedule


# ==================== Tax Rate Normalization ====================

# 국가별/지역별 법정 법인세율
STATUTORY_TAX_RATES = {
    'US': 0.21,
    'EU': 0.25,  # 평균
    'UK': 0.25,
    'JP': 0.30,
    'KR': 0.25,
    'CN': 0.25,
    'Default': 0.21  # US 기준
}


def normalize_tax_rate(
    current_effective_tax_rate: float,
    country: str = 'US',
    years: int = 5
) -> Tuple[List[float], str]:
    """
    실효세율을 법정세율로 정상화 (Tax Shield 소멸 반영)

    Args:
        current_effective_tax_rate: 현재 실효 세율
        country: 국가 코드
        years: projection 기간

    Returns:
        (연도별 세율 리스트, 설명)
    """
    statutory_rate = STATUTORY_TAX_RATES.get(country, STATUTORY_TAX_RATES['Default'])

    # 현재 세율이 법정세율에 가깝거나 높으면 유지
    if current_effective_tax_rate >= statutory_rate * 0.9:
        return [current_effective_tax_rate] * years, f"세율 유지 ({current_effective_tax_rate*100:.1f}%)"

    # 현재 세율이 낮으면 점진적 정상화
    schedule = []
    step = (statutory_rate - current_effective_tax_rate) / years

    for i in range(years):
        tax = current_effective_tax_rate + step * (i + 1)
        schedule.append(min(tax, statutory_rate))

    insight = f"세율 정상화: {current_effective_tax_rate*100:.1f}% → {statutory_rate*100:.1f}%"
    return schedule, insight


# ==================== Smart Defaults Generator ====================

@dataclass
class SmartDefaults:
    """Context-Aware Smart Default 값들"""
    lifecycle: LifecycleResult
    projection_years: int
    growth_schedule: List[float]
    margin_schedule: List[float]
    capex_schedule: List[float]
    tax_schedule: List[float]
    terminal_growth: float
    insights: List[str]


def generate_smart_defaults(
    financial_data: Dict,
    risk_free_rate: float,
    sector: str = 'Default'
) -> SmartDefaults:
    """
    재무 데이터 기반 Smart Default 값 생성

    Args:
        financial_data: get_stock_data() 결과
        risk_free_rate: 무위험 이자율
        sector: 섹터 이름

    Returns:
        SmartDefaults with all schedules and insights
    """
    insights = []

    # 1. Revenue Growth 추출
    revenue_growth = financial_data.get('revenue_growth', 0) or 0
    if revenue_growth == 0:
        # Historical CAGR 사용
        historical = financial_data.get('historical_financials', [])
        if len(historical) >= 2:
            revenues = [h.get('revenue', 0) for h in historical if h.get('revenue', 0) > 0]
            if len(revenues) >= 2:
                start, end = revenues[-1], revenues[0]  # 역순
                n = len(revenues) - 1
                if start > 0 and end > 0:
                    revenue_growth = (end / start) ** (1 / n) - 1

    # 2. Lifecycle Classification (Q2 반영)
    lifecycle = classify_lifecycle(revenue_growth)
    projection_years = lifecycle.projection_years
    insights.append(lifecycle.insight)

    # 3. Growth Decay Schedule (Risk-Free Rate으로 수렴)
    terminal_growth = min(risk_free_rate, 0.03)  # 최대 3%
    growth_schedule = generate_growth_decay_schedule(
        initial_growth=revenue_growth,
        terminal_growth=terminal_growth,
        years=projection_years,
        decay_type='linear'
    )
    insights.append(f"성장률: {revenue_growth*100:.1f}% → {terminal_growth*100:.1f}% (Risk-Free Rate 수렴)")

    # 4. EBITDA Margin Convergence (Q1 반영: 회사 Historical 우선)
    current_margin = 0
    revenue = financial_data.get('revenue', 0)
    ebitda = financial_data.get('ebitda', 0)
    if revenue > 0 and ebitda > 0:
        current_margin = ebitda / revenue
    else:
        current_margin = SECTOR_TARGET_MARGINS.get(sector, 0.20)

    target_margin, margin_source = get_target_margin(sector, current_margin)
    margin_schedule = generate_margin_convergence_schedule(
        current_margin=current_margin,
        target_margin=target_margin,
        years=projection_years,
        convergence_speed=0.3  # 느린 수렴
    )
    insights.append(f"EBITDA 마진: {current_margin*100:.1f}% → {target_margin*100:.1f}% ({margin_source})")

    # 5. CapEx Convergence (Q3 반영: 점진적 수렴)
    current_capex_pct = 0.05  # Default
    current_da_pct = 0.05  # Default

    if revenue > 0:
        operating_cf = financial_data.get('operating_cf', 0)
        fcf = financial_data.get('fcf', 0)
        if operating_cf > 0 and fcf > 0:
            current_capex_pct = (operating_cf - fcf) / revenue

    capex_schedule = generate_capex_convergence_schedule(
        current_capex_pct=current_capex_pct,
        current_da_pct=current_da_pct,
        years=projection_years,
        target_capex_ratio=1.05
    )
    insights.append(f"CapEx: {current_capex_pct*100:.1f}% → D&A × 105% (Steady State 수렴)")

    # 6. Tax Rate Normalization
    current_tax_rate = financial_data.get('tax_rate', 0.21)
    tax_schedule, tax_insight = normalize_tax_rate(
        current_effective_tax_rate=current_tax_rate,
        country='US',
        years=projection_years
    )
    insights.append(tax_insight)

    return SmartDefaults(
        lifecycle=lifecycle,
        projection_years=projection_years,
        growth_schedule=growth_schedule,
        margin_schedule=margin_schedule,
        capex_schedule=capex_schedule,
        tax_schedule=tax_schedule,
        terminal_growth=terminal_growth,
        insights=insights
    )


# ==================== Credit Rating & Spreads ====================

# Synthetic Rating: Interest Coverage Ratio (ICR) → Credit Spread
# Source: Damodaran's Corporate Finance framework
CREDIT_SPREAD_TABLE = {
    # (ICR_min, ICR_max): (rating, spread)
    (-999, 0.5): ('D', 0.12),      # Default risk
    (0.5, 0.8): ('C', 0.10),
    (0.8, 1.25): ('CC', 0.09),
    (1.25, 1.5): ('CCC', 0.08),
    (1.5, 2.0): ('B-', 0.07),
    (2.0, 2.5): ('B', 0.065),
    (2.5, 3.0): ('B+', 0.06),
    (3.0, 4.0): ('BB', 0.05),
    (4.0, 4.5): ('BB+', 0.045),
    (4.5, 6.0): ('BBB', 0.035),    # Investment grade threshold
    (6.0, 7.5): ('A-', 0.025),
    (7.5, 9.5): ('A', 0.02),
    (9.5, 12.5): ('A+', 0.015),
    (12.5, 999): ('AA/AAA', 0.01),
}


def get_synthetic_rating(interest_coverage_ratio: float) -> Tuple[str, float]:
    """
    이자보상배율(ICR)을 통한 합성 신용등급 산출

    Args:
        interest_coverage_ratio: EBIT / Interest Expense

    Returns:
        (credit_rating, credit_spread)

    Example:
        >>> get_synthetic_rating(5.5)
        ('BBB', 0.035)
    """
    if interest_coverage_ratio is None or np.isnan(interest_coverage_ratio):
        return ('BB', 0.05)  # Default to junk bond

    for (icr_min, icr_max), (rating, spread) in CREDIT_SPREAD_TABLE.items():
        if icr_min <= interest_coverage_ratio < icr_max:
            return (rating, spread)

    # Fallback
    return ('BB', 0.05)


# ==================== Cost of Debt ====================

def calculate_cost_of_debt(
    interest_expense: float,
    total_debt: float,
    ebit: float,
    risk_free_rate: float,
    tax_rate: float = 0.21
) -> Dict:
    """
    타인자본비용(Cost of Debt) 계산

    Method 1: 실제 이자비용 사용 (우선)
    Method 2: Synthetic Rating (ICR 기반)

    Returns after-tax cost: Kd × (1 - Tax Rate)
    """
    result = {
        'method': None,
        'kd_pretax': 0,
        'kd_aftertax': 0,
        'credit_rating': None,
        'credit_spread': 0,
        'icr': None,
        'note': ''
    }

    # Method 1: 실제 이자비용 기반
    if interest_expense > 0 and total_debt > 0:
        kd_pretax = interest_expense / total_debt

        # Sanity check: 이자율이 0.5%~15% 범위 내인지
        if 0.005 <= kd_pretax <= 0.15:
            result['method'] = 'actual'
            result['kd_pretax'] = kd_pretax
            result['kd_aftertax'] = kd_pretax * (1 - tax_rate)
            result['note'] = f'Actual interest rate: {kd_pretax*100:.2f}%'
            return result

    # Method 2: Synthetic Rating (ICR 기반)
    if ebit > 0 and interest_expense > 0:
        icr = ebit / interest_expense
        rating, spread = get_synthetic_rating(icr)
        kd_pretax = risk_free_rate + spread

        result['method'] = 'synthetic'
        result['kd_pretax'] = kd_pretax
        result['kd_aftertax'] = kd_pretax * (1 - tax_rate)
        result['credit_rating'] = rating
        result['credit_spread'] = spread
        result['icr'] = icr
        result['note'] = f'Synthetic rating {rating} (ICR={icr:.2f}x) → Spread {spread*100:.1f}%'
        return result

    # Fallback: 섹터 평균 (보수적)
    kd_pretax = risk_free_rate + 0.05  # Risk-free + 5% spread
    result['method'] = 'fallback'
    result['kd_pretax'] = kd_pretax
    result['kd_aftertax'] = kd_pretax * (1 - tax_rate)
    result['note'] = 'Insufficient data → Using Rf + 5% spread'

    return result


# ==================== Cost of Equity ====================

def adjust_beta(raw_beta: float, adjustment_factor: float = 0.67) -> float:
    """
    Blume's Adjusted Beta: 베타의 평균 회귀 속성 반영

    Adjusted Beta = Raw Beta × (1 - α) + Market Beta × α
    where α = adjustment factor (default 0.67, Blume's original)

    Args:
        raw_beta: 과거 데이터 기반 베타
        adjustment_factor: 평균 회귀 가중치 (0.67 = Bloomberg/Blume 방식)

    Returns:
        조정된 베타

    Example:
        >>> adjust_beta(1.5)  # High beta stock
        1.335  # Pulls toward 1.0
    """
    market_beta = 1.0
    adjusted = raw_beta * (1 - adjustment_factor) + market_beta * adjustment_factor

    # Sanity check: 베타가 음수이거나 5를 초과하면 제한
    adjusted = max(min(adjusted, 5.0), 0.1)

    return adjusted


def calculate_cost_of_equity(
    beta: float,
    risk_free_rate: float,
    market_risk_premium: float = 0.055,
    use_adjusted_beta: bool = True
) -> Dict:
    """
    자기자본비용(Cost of Equity) 계산 - CAPM

    Ke = Rf + β × MRP

    Args:
        beta: 기업의 베타 (시장 대비 변동성)
        risk_free_rate: 무위험 이자율 (10Y Treasury)
        market_risk_premium: 시장 위험 프리미엄 (default 5.5%)
        use_adjusted_beta: Blume's Adjusted Beta 사용 여부

    Returns:
        {'ke': float, 'beta_used': float, 'beta_raw': float, 'note': str}
    """
    beta_raw = beta if beta and beta > 0 else 1.0
    beta_used = adjust_beta(beta_raw) if use_adjusted_beta else beta_raw

    ke = risk_free_rate + beta_used * market_risk_premium

    # Sanity check: CoE는 최소 Rf + 2%, 최대 25%
    ke = max(ke, risk_free_rate + 0.02)
    ke = min(ke, 0.25)

    note = ''
    if use_adjusted_beta:
        note = f'Adjusted Beta (Blume): {beta_raw:.3f} → {beta_used:.3f}'
    else:
        note = f'Raw Beta: {beta_used:.3f}'

    return {
        'ke': ke,
        'beta_used': beta_used,
        'beta_raw': beta_raw,
        'note': note
    }


# ==================== WACC ====================

def calculate_wacc(
    market_cap: float,
    total_debt: float,
    cash: float,
    cost_of_equity: float,
    cost_of_debt_aftertax: float,
    include_cash: bool = False
) -> Dict:
    """
    WACC (Weighted Average Cost of Capital) 계산

    WACC = (E/V) × Ke + (D/V) × Kd × (1-T)

    Args:
        market_cap: 시가총액 (E)
        total_debt: 총 부채 (D)
        cash: 현금 (선택적으로 차감 가능)
        cost_of_equity: 자기자본비용 (Ke)
        cost_of_debt_aftertax: 세후 타인자본비용 (Kd × (1-T))
        include_cash: Net Debt 사용 여부 (True면 D - Cash)

    Returns:
        {'wacc': float, 'weight_equity': float, 'weight_debt': float, ...}
    """
    # Net Debt 옵션
    net_debt = total_debt - cash if include_cash else total_debt
    net_debt = max(net_debt, 0)  # 음수 방지

    # Enterprise Value = E + D
    ev = market_cap + net_debt

    if ev <= 0:
        return {
            'wacc': cost_of_equity,  # 부채 없으면 Ke만 사용
            'weight_equity': 1.0,
            'weight_debt': 0.0,
            'note': 'No debt → WACC = Cost of Equity'
        }

    weight_equity = market_cap / ev
    weight_debt = net_debt / ev

    wacc = weight_equity * cost_of_equity + weight_debt * cost_of_debt_aftertax

    # Sanity check: WACC는 최소 3%, 최대 20%
    wacc = max(min(wacc, 0.20), 0.03)

    return {
        'wacc': wacc,
        'weight_equity': weight_equity,
        'weight_debt': weight_debt,
        'ev': ev,
        'note': f'E/V={weight_equity*100:.1f}%, D/V={weight_debt*100:.1f}%'
    }


# ==================== Complete WACC Calculation ====================

def calculate_full_wacc(
    financial_data: Dict,
    risk_free_rate: float,
    market_risk_premium: float = 0.055,
    use_adjusted_beta: bool = True,
    include_cash_in_debt: bool = False
) -> Dict:
    """
    전체 WACC 계산 (원스톱 함수)

    Args:
        financial_data: get_stock_data()로 수집한 재무 데이터
        risk_free_rate: 무위험 이자율 (10Y Treasury)
        market_risk_premium: 시장 위험 프리미엄
        use_adjusted_beta: Adjusted Beta 사용 여부
        include_cash_in_debt: Net Debt 사용 여부

    Returns:
        {
            'wacc': float,
            'cost_of_equity': dict,
            'cost_of_debt': dict,
            'weights': dict,
            'calculation_log': list  # 계산 과정 로그
        }
    """
    log = []

    # 1. Cost of Equity
    beta = financial_data.get('beta', 1.0)
    coe_result = calculate_cost_of_equity(
        beta=beta,
        risk_free_rate=risk_free_rate,
        market_risk_premium=market_risk_premium,
        use_adjusted_beta=use_adjusted_beta
    )
    log.append(f"📊 Cost of Equity: {coe_result['ke']*100:.2f}% ({coe_result['note']})")

    # 2. Cost of Debt
    interest_expense = financial_data.get('interest_expense', 0)
    total_debt = financial_data.get('total_debt', 0)
    ebit = financial_data.get('ebit', 0)
    tax_rate = financial_data.get('tax_rate', 0.21)

    cod_result = calculate_cost_of_debt(
        interest_expense=interest_expense,
        total_debt=total_debt,
        ebit=ebit,
        risk_free_rate=risk_free_rate,
        tax_rate=tax_rate
    )
    log.append(f"💰 Cost of Debt (After-tax): {cod_result['kd_aftertax']*100:.2f}% ({cod_result['note']})")

    # 3. WACC
    market_cap = financial_data.get('market_cap', 0)
    cash = financial_data.get('cash', 0)

    wacc_result = calculate_wacc(
        market_cap=market_cap,
        total_debt=total_debt,
        cash=cash,
        cost_of_equity=coe_result['ke'],
        cost_of_debt_aftertax=cod_result['kd_aftertax'],
        include_cash=include_cash_in_debt
    )
    log.append(f"⚖️ WACC: {wacc_result['wacc']*100:.2f}% ({wacc_result['note']})")

    return {
        'wacc': wacc_result['wacc'],
        'cost_of_equity': coe_result,
        'cost_of_debt': cod_result,
        'weights': {
            'equity': wacc_result['weight_equity'],
            'debt': wacc_result['weight_debt']
        },
        'calculation_log': log
    }


# ==================== Terminal Growth Validation ====================

def validate_terminal_growth(
    terminal_growth: float,
    risk_free_rate: float,
    max_allowed: Optional[float] = None
) -> Dict:
    """
    영구 성장률(Terminal Growth) 검증

    Rule: 영구 성장률은 경제 성장률(≈ Risk-free rate)을 초과할 수 없음

    Args:
        terminal_growth: 사용자 입력 영구 성장률
        risk_free_rate: 무위험 이자율 (경제 성장률 proxy)
        max_allowed: 커스텀 상한 (None이면 Rf 사용)

    Returns:
        {'valid': bool, 'adjusted_tg': float, 'warning': str}
    """
    cap = max_allowed if max_allowed is not None else risk_free_rate

    if terminal_growth > cap:
        return {
            'valid': False,
            'adjusted_tg': cap,
            'warning': f'⚠️ Terminal Growth ({terminal_growth*100:.1f}%) > Economic Growth ({cap*100:.1f}%). Capped at {cap*100:.1f}%'
        }

    if terminal_growth < 0:
        return {
            'valid': False,
            'adjusted_tg': 0.02,
            'warning': f'⚠️ Terminal Growth cannot be negative. Reset to 2.0%'
        }

    return {
        'valid': True,
        'adjusted_tg': terminal_growth,
        'warning': None
    }


# ==================== Reinvestment Check (ROIC-based) ====================

def check_reinvestment_feasibility(
    revenue_growth: float,
    roic: float,
    threshold: float = 0.50
) -> Dict:
    """
    재투자율 실현 가능성 검증 (선택적 기능)

    Required Reinvestment Rate = Revenue Growth / ROIC

    If RRR > 50%, the growth assumption may be unrealistic.

    Args:
        revenue_growth: 예상 매출 성장률
        roic: Return on Invested Capital
        threshold: 비현실적 재투자율 기준 (default 50%)

    Returns:
        {'feasible': bool, 'required_reinvestment': float, 'warning': str}
    """
    if roic is None or roic <= 0:
        return {
            'feasible': True,
            'required_reinvestment': None,
            'warning': '⚠️ ROIC 데이터 부족 → 재투자율 검증 불가'
        }

    required_reinvestment = revenue_growth / roic

    if required_reinvestment > threshold:
        return {
            'feasible': False,
            'required_reinvestment': required_reinvestment,
            'warning': f'⚠️ 성장률 {revenue_growth*100:.1f}% 달성에 필요한 재투자율: {required_reinvestment*100:.0f}% (ROIC {roic*100:.1f}% 가정). 비현실적일 수 있음.'
        }

    return {
        'feasible': True,
        'required_reinvestment': required_reinvestment,
        'warning': None
    }
