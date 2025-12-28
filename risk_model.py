"""
risk_model.py - Quality & Risk Scorecard
Value Trap 감지를 위한 정량적 리스크 진단 모델

5가지 핵심 지표:
1. Earnings Quality: OCF vs Net Income (분식회계/재고 누적 감지)
2. Capital Efficiency: ROIC vs WACC (가치 파괴 여부)
3. Growth Momentum: TTM Growth vs 3Y CAGR (성장 둔화 감지)
4. Market Sentiment: Earnings Surprise + Forward EPS Gap
5. Leverage Risk: Debt/EBITDA (과도한 부채)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class RiskLevel(Enum):
    """리스크 수준"""
    LOW = "low"          # 0-1 flags: Green
    MODERATE = "moderate"  # 2 flags: Yellow
    HIGH = "high"        # 3-5 flags: Red


@dataclass
class RiskFlag:
    """개별 리스크 플래그"""
    name: str
    triggered: bool
    value: Optional[float]
    threshold: float
    message: str
    severity: str  # "warning", "danger", "ok"


@dataclass
class RiskScorecard:
    """종합 리스크 평가 결과"""
    ticker: str
    risk_level: RiskLevel
    flags_triggered: int
    total_flags: int
    flags: List[RiskFlag]
    summary: str
    recommendation: str


# ==================== Individual Risk Metrics ====================

def calculate_earnings_quality(
    operating_cf: float,
    net_income: float,
    threshold: float = 0.8
) -> RiskFlag:
    """
    1. Earnings Quality (이익의 질)

    OCF/NI < 0.8 이면 경고 (이익이 현금으로 뒷받침되지 않음)
    - 분식회계 가능성
    - 재고/매출채권 누적
    - 비현금 이익 과대계상

    Args:
        operating_cf: 영업활동현금흐름
        net_income: 순이익
        threshold: 경고 기준 (default 0.8)

    Returns:
        RiskFlag with triggered status
    """
    if net_income <= 0:
        # 적자 기업은 이 지표 적용 불가
        return RiskFlag(
            name="Earnings Quality",
            triggered=False,
            value=None,
            threshold=threshold,
            message="N/A (Net Income <= 0)",
            severity="ok"
        )

    ratio = operating_cf / net_income if net_income != 0 else 0
    triggered = ratio < threshold

    if ratio < 0.5:
        severity = "danger"
        message = f"OCF/NI = {ratio:.2f}x (Critical: Cash flow severely lags earnings)"
    elif ratio < threshold:
        severity = "warning"
        message = f"OCF/NI = {ratio:.2f}x (Warning: Earnings quality concern)"
    else:
        severity = "ok"
        message = f"OCF/NI = {ratio:.2f}x (Healthy: Earnings backed by cash)"

    return RiskFlag(
        name="Earnings Quality",
        triggered=triggered,
        value=ratio,
        threshold=threshold,
        message=message,
        severity=severity
    )


def calculate_roic(
    ebit: float,
    tax_rate: float,
    total_equity: float,
    total_debt: float
) -> Optional[float]:
    """
    ROIC (Return on Invested Capital) 계산

    ROIC = NOPAT / Invested Capital
    NOPAT = EBIT × (1 - Tax Rate)
    Invested Capital = Total Equity + Total Debt

    Note: 현금을 차감하지 않는 이유:
    - yfinance의 totalCash는 단기투자까지 포함하여 과대계상
    - Finviz/GuruFocus 등 주요 사이트와 동일한 방식
    - 현금 차감 시 Invested Capital이 너무 작아져 ROIC 과대 계산
    """
    if ebit <= 0:
        return None

    nopat = ebit * (1 - tax_rate)
    invested_capital = total_equity + total_debt

    if invested_capital <= 0:
        return None

    return nopat / invested_capital


def calculate_capital_efficiency(
    roic: Optional[float],
    wacc: float,
    threshold: float = 0.0  # ROIC should be > WACC
) -> RiskFlag:
    """
    2. Capital Efficiency (자본 효율성)

    ROIC < WACC 이면 경고 (가치 파괴)
    - 투자 대비 수익이 자본비용을 커버하지 못함
    - 사업이 주주가치를 파괴하고 있음

    Args:
        roic: Return on Invested Capital
        wacc: Weighted Average Cost of Capital
        threshold: ROIC - WACC 최소 마진 (default 0)

    Returns:
        RiskFlag with triggered status
    """
    if roic is None:
        return RiskFlag(
            name="Capital Efficiency",
            triggered=False,
            value=None,
            threshold=wacc,
            message="N/A (ROIC calculation not available)",
            severity="ok"
        )

    spread = roic - wacc
    triggered = spread < threshold

    if spread < -0.03:  # ROIC가 WACC보다 3%p 이상 낮음
        severity = "danger"
        message = f"ROIC {roic*100:.1f}% < WACC {wacc*100:.1f}% (Value Destruction)"
    elif spread < threshold:
        severity = "warning"
        message = f"ROIC {roic*100:.1f}% ≈ WACC {wacc*100:.1f}% (Marginal returns)"
    else:
        severity = "ok"
        message = f"ROIC {roic*100:.1f}% > WACC {wacc*100:.1f}% (Value Creation)"

    return RiskFlag(
        name="Capital Efficiency",
        triggered=triggered,
        value=roic,
        threshold=wacc,
        message=message,
        severity=severity
    )


def calculate_growth_momentum(
    ttm_revenue_growth: float,
    cagr_3y: float,
    deceleration_threshold: float = 0.5  # TTM이 CAGR의 50% 미만이면 경고
) -> RiskFlag:
    """
    3. Growth Momentum (성장 모멘텀)

    TTM Revenue Growth가 3Y CAGR 대비 급격히 꺾이면 경고
    - 성장 둔화/역성장 전환 신호
    - 시장 포화 또는 경쟁력 약화

    Args:
        ttm_revenue_growth: 최근 12개월 매출 성장률
        cagr_3y: 3년 매출 CAGR
        deceleration_threshold: TTM/CAGR 비율 하한 (default 0.5)

    Returns:
        RiskFlag with triggered status
    """
    if cagr_3y <= 0:
        # 과거 CAGR이 0 이하면 이미 문제
        if ttm_revenue_growth < 0:
            return RiskFlag(
                name="Growth Momentum",
                triggered=True,
                value=ttm_revenue_growth,
                threshold=0,
                message=f"TTM Growth {ttm_revenue_growth*100:.1f}% (Negative growth)",
                severity="danger"
            )
        return RiskFlag(
            name="Growth Momentum",
            triggered=False,
            value=ttm_revenue_growth,
            threshold=0,
            message=f"TTM Growth {ttm_revenue_growth*100:.1f}% (Baseline CAGR ≤ 0)",
            severity="ok"
        )

    # TTM이 음수이거나 CAGR 대비 급격히 하락
    ratio = ttm_revenue_growth / cagr_3y if cagr_3y > 0 else 0

    # 역성장 전환은 무조건 경고
    if ttm_revenue_growth < 0 and cagr_3y > 0.05:
        return RiskFlag(
            name="Growth Momentum",
            triggered=True,
            value=ttm_revenue_growth,
            threshold=cagr_3y * deceleration_threshold,
            message=f"TTM {ttm_revenue_growth*100:.1f}% vs 3Y CAGR {cagr_3y*100:.1f}% (Growth reversal)",
            severity="danger"
        )

    triggered = ratio < deceleration_threshold

    if ratio < 0.3:
        severity = "danger"
        message = f"TTM {ttm_revenue_growth*100:.1f}% vs 3Y CAGR {cagr_3y*100:.1f}% (Severe deceleration)"
    elif triggered:
        severity = "warning"
        message = f"TTM {ttm_revenue_growth*100:.1f}% vs 3Y CAGR {cagr_3y*100:.1f}% (Growth slowing)"
    else:
        severity = "ok"
        message = f"TTM {ttm_revenue_growth*100:.1f}% vs 3Y CAGR {cagr_3y*100:.1f}% (Momentum intact)"

    return RiskFlag(
        name="Growth Momentum",
        triggered=triggered,
        value=ttm_revenue_growth,
        threshold=cagr_3y * deceleration_threshold,
        message=message,
        severity=severity
    )


def calculate_market_sentiment(
    earnings_surprises: List[Dict],  # [{quarter, actual, estimate, surprise_pct}, ...]
    forward_eps: float,
    trailing_eps: float,
    miss_threshold: int = 2  # 최근 4분기 중 2회 이상 miss면 경고
) -> RiskFlag:
    """
    4. Market Sentiment (시장 심리)

    A) Earnings Surprise: 최근 4분기 중 miss 횟수
    B) Forward EPS Gap: Forward EPS < Trailing EPS면 역성장 예상

    둘 중 하나라도 부정적이면 경고

    Args:
        earnings_surprises: 분기별 실적 서프라이즈 리스트
        forward_eps: 향후 12개월 예상 EPS
        trailing_eps: 최근 12개월 EPS
        miss_threshold: miss 허용 횟수 (default 2)

    Returns:
        RiskFlag with triggered status
    """
    issues = []

    # A) Earnings Surprise 분석
    if earnings_surprises:
        misses = sum(1 for e in earnings_surprises if e.get('surprise_pct', 0) < 0)
        total = len(earnings_surprises)

        if misses >= miss_threshold:
            issues.append(f"Missed {misses}/{total} quarters")

    # B) Forward EPS Gap 분석
    if trailing_eps > 0 and forward_eps > 0:
        eps_growth = (forward_eps - trailing_eps) / trailing_eps
        if eps_growth < -0.05:  # 5% 이상 역성장 예상
            issues.append(f"Forward EPS decline expected ({eps_growth*100:.1f}%)")
    elif forward_eps <= 0 and trailing_eps > 0:
        issues.append("Forward EPS unavailable or negative")

    triggered = len(issues) > 0

    if len(issues) >= 2:
        severity = "danger"
        message = " | ".join(issues)
    elif triggered:
        severity = "warning"
        message = issues[0] if issues else "Minor concern"
    else:
        # 긍정적 신호
        beats = sum(1 for e in earnings_surprises if e.get('surprise_pct', 0) > 0) if earnings_surprises else 0
        total = len(earnings_surprises) if earnings_surprises else 0

        if trailing_eps > 0 and forward_eps > 0:
            eps_growth = (forward_eps - trailing_eps) / trailing_eps
            severity = "ok"
            message = f"Beat {beats}/{total} quarters | Forward EPS +{eps_growth*100:.1f}%"
        else:
            severity = "ok"
            message = f"Beat {beats}/{total} quarters"

    return RiskFlag(
        name="Market Sentiment",
        triggered=triggered,
        value=None,
        threshold=miss_threshold,
        message=message,
        severity=severity
    )


def calculate_leverage_risk(
    total_debt: float,
    ebitda: float,
    threshold: float = 4.0  # Debt/EBITDA > 4x면 경고
) -> RiskFlag:
    """
    5. Leverage Risk (레버리지 위험)

    Debt/EBITDA > 4x면 경고
    - 과도한 부채 부담
    - 금리 상승 시 이자 부담 급증
    - 재무적 유연성 저하

    Args:
        total_debt: 총 부채
        ebitda: EBITDA
        threshold: 경고 기준 배수 (default 4.0x)

    Returns:
        RiskFlag with triggered status
    """
    if ebitda <= 0:
        if total_debt > 0:
            return RiskFlag(
                name="Leverage Risk",
                triggered=True,
                value=None,
                threshold=threshold,
                message="Debt exists but EBITDA ≤ 0 (Cannot service debt)",
                severity="danger"
            )
        return RiskFlag(
            name="Leverage Risk",
            triggered=False,
            value=None,
            threshold=threshold,
            message="N/A (EBITDA ≤ 0, No debt)",
            severity="ok"
        )

    ratio = total_debt / ebitda
    triggered = ratio > threshold

    if ratio > 6.0:
        severity = "danger"
        message = f"Debt/EBITDA = {ratio:.1f}x (Highly leveraged)"
    elif ratio > threshold:
        severity = "warning"
        message = f"Debt/EBITDA = {ratio:.1f}x (Elevated leverage)"
    elif ratio > 2.0:
        severity = "ok"
        message = f"Debt/EBITDA = {ratio:.1f}x (Moderate leverage)"
    else:
        severity = "ok"
        message = f"Debt/EBITDA = {ratio:.1f}x (Conservative)"

    return RiskFlag(
        name="Leverage Risk",
        triggered=triggered,
        value=ratio,
        threshold=threshold,
        message=message,
        severity=severity
    )


# ==================== Aggregate Scorecard ====================

def generate_risk_scorecard(
    ticker: str,
    financial_data: Dict,
    wacc: float,
    earnings_surprises: List[Dict] = None
) -> RiskScorecard:
    """
    종합 Risk Scorecard 생성

    Args:
        ticker: 종목 코드
        financial_data: get_stock_data() 결과
        wacc: WACC (from valuation_utils)
        earnings_surprises: get_earnings_history() 결과

    Returns:
        RiskScorecard with all flags and summary
    """
    flags = []

    # 1. Earnings Quality
    operating_cf = financial_data.get('operating_cf', 0) or 0
    net_income = financial_data.get('net_income', 0) or 0
    flags.append(calculate_earnings_quality(operating_cf, net_income))

    # 2. Capital Efficiency (ROIC vs WACC)
    ebit = financial_data.get('ebit', 0) or 0
    tax_rate = financial_data.get('tax_rate', 0.21)
    total_equity = financial_data.get('total_equity', 0) or 0
    total_debt = financial_data.get('total_debt', 0) or 0

    roic = calculate_roic(ebit, tax_rate, total_equity, total_debt)
    flags.append(calculate_capital_efficiency(roic, wacc))

    # 3. Growth Momentum
    ttm_revenue_growth = financial_data.get('revenue_growth', 0) or 0

    # 3Y Revenue CAGR 계산
    historical = financial_data.get('historical_financials', [])
    cagr_3y = 0
    if len(historical) >= 3:
        revenues = [h.get('revenue', 0) for h in historical if h.get('revenue', 0) > 0]
        if len(revenues) >= 3:
            start, end = revenues[-1], revenues[0]  # 역순 (최신이 앞)
            n = min(3, len(revenues) - 1)
            if start > 0 and end > 0:
                cagr_3y = (end / start) ** (1 / n) - 1

    flags.append(calculate_growth_momentum(ttm_revenue_growth, cagr_3y))

    # 4. Market Sentiment
    forward_eps = financial_data.get('forward_eps', 0) or 0
    trailing_eps = financial_data.get('eps', 0) or 0
    flags.append(calculate_market_sentiment(
        earnings_surprises or [],
        forward_eps,
        trailing_eps
    ))

    # 5. Leverage Risk
    ebitda = financial_data.get('ebitda', 0) or 0
    flags.append(calculate_leverage_risk(total_debt, ebitda))

    # 종합 판정
    flags_triggered = sum(1 for f in flags if f.triggered)
    total_flags = len(flags)

    if flags_triggered <= 1:
        risk_level = RiskLevel.LOW
        summary = "Low Risk - Fundamentals appear healthy"
        recommendation = "Valuation-based decision is reasonable."
    elif flags_triggered == 2:
        risk_level = RiskLevel.MODERATE
        summary = "Moderate Risk - Some concerns detected"
        recommendation = "Review flagged items before investing. Consider margin of safety."
    else:
        risk_level = RiskLevel.HIGH
        summary = "High Risk - Multiple warning signs"
        recommendation = "Potential Value Trap. Cheap valuation may reflect real problems."

    return RiskScorecard(
        ticker=ticker,
        risk_level=risk_level,
        flags_triggered=flags_triggered,
        total_flags=total_flags,
        flags=flags,
        summary=summary,
        recommendation=recommendation
    )


def get_risk_color(risk_level: RiskLevel) -> Tuple[str, str]:
    """
    Risk Level에 따른 색상 반환

    Returns:
        (background_color, text_color)
    """
    if risk_level == RiskLevel.LOW:
        return ("#22c55e", "#166534")  # Green
    elif risk_level == RiskLevel.MODERATE:
        return ("#f59e0b", "#92400e")  # Yellow/Amber
    else:
        return ("#ef4444", "#991b1b")  # Red


def get_risk_emoji(risk_level: RiskLevel) -> str:
    """Risk Level에 따른 이모지"""
    if risk_level == RiskLevel.LOW:
        return "🟢"
    elif risk_level == RiskLevel.MODERATE:
        return "🟡"
    else:
        return "🔴"


def get_flag_icon(flag: RiskFlag) -> str:
    """Flag 상태에 따른 아이콘"""
    if flag.triggered:
        if flag.severity == "danger":
            return "🚨"
        return "⚠️"
    return "✅"
