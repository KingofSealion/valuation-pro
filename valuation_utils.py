"""
valuation_utils.py - Valuation Calculation Utilities
- WACC (Weighted Average Cost of Capital)
- Synthetic Credit Rating
- Adjusted Beta
- Terminal Growth Cap
"""
import numpy as np
from typing import Dict, Tuple, Optional


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
