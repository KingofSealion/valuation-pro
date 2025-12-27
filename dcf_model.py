"""
dcf_model.py - Wall Street Style DCF Model (v4)
- 3Y CAGR 기반 Base Growth
- Bull/Bear는 Revenue Growth + Exit Multiple만 조절
- WACC 자동 계산 (Synthetic Rating, Adjusted Beta)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from valuation_utils import (
    calculate_full_wacc,
    validate_terminal_growth,
    check_reinvestment_feasibility,
    classify_lifecycle,
    generate_growth_decay_schedule,
    generate_margin_convergence_schedule,
    generate_capex_convergence_schedule,
    normalize_tax_rate,
    generate_smart_defaults,
    get_target_margin,
    SmartDefaults,
    LifecycleStage,
    SECTOR_TARGET_MARGINS
)
from data_fetcher import get_risk_free_rate

# 섹터별 기본값 + 성장률 Cap
SECTOR_DEFAULTS = {
    'Technology': {
        'ebitda_margin': 0.30,
        'da_pct': 0.06,
        'capex_pct': 0.06,
        'nwc_pct': 0.05,
        'exit_multiple': 22,  # Tech 기업 프리미엄 반영
        'growth_cap': 0.40,   # 고성장 Tech 기업 반영 (NVIDIA 등)
    },
    'Consumer Cyclical': {
        'ebitda_margin': 0.15,
        'da_pct': 0.08,
        'capex_pct': 0.08,
        'nwc_pct': -0.05,
        'exit_multiple': 14,
        'growth_cap': 0.20,
    },
    'Communication Services': {
        'ebitda_margin': 0.35,
        'da_pct': 0.08,
        'capex_pct': 0.08,
        'nwc_pct': 0.02,
        'exit_multiple': 12,
        'growth_cap': 0.15,
    },
    'Healthcare': {
        'ebitda_margin': 0.25,
        'da_pct': 0.04,
        'capex_pct': 0.04,
        'nwc_pct': 0.15,
        'exit_multiple': 14,
        'growth_cap': 0.20,
    },
    'Financials': {
        'ebitda_margin': 0.40,
        'da_pct': 0.02,
        'capex_pct': 0.02,
        'nwc_pct': 0.0,
        'exit_multiple': 10,
        'growth_cap': 0.10,
    },
    'Consumer Defensive': {
        'ebitda_margin': 0.18,
        'da_pct': 0.04,
        'capex_pct': 0.04,
        'nwc_pct': 0.08,
        'exit_multiple': 14,
        'growth_cap': 0.08,
    },
    'Industrials': {
        'ebitda_margin': 0.15,
        'da_pct': 0.05,
        'capex_pct': 0.05,
        'nwc_pct': 0.15,
        'exit_multiple': 11,
        'growth_cap': 0.12,
    },
    'Energy': {
        'ebitda_margin': 0.25,
        'da_pct': 0.08,
        'capex_pct': 0.10,
        'nwc_pct': 0.05,
        'exit_multiple': 6,
        'growth_cap': 0.10,
    },
    'Utilities': {
        'ebitda_margin': 0.35,
        'da_pct': 0.08,
        'capex_pct': 0.12,
        'nwc_pct': 0.02,
        'exit_multiple': 12,
        'growth_cap': 0.05,
    },
    'Real Estate': {
        'ebitda_margin': 0.60,
        'da_pct': 0.10,
        'capex_pct': 0.05,
        'nwc_pct': 0.02,
        'exit_multiple': 18,
        'growth_cap': 0.08,
    },
    'Materials': {
        'ebitda_margin': 0.20,
        'da_pct': 0.06,
        'capex_pct': 0.07,
        'nwc_pct': 0.12,
        'exit_multiple': 8,
        'growth_cap': 0.10,
    },
    'Default': {
        'ebitda_margin': 0.20,
        'da_pct': 0.05,
        'capex_pct': 0.05,
        'nwc_pct': 0.10,
        'exit_multiple': 12,
        'growth_cap': 0.15,
    }
}


@dataclass
class DCFAssumptions:
    """DCF 가정값"""
    revenue_growth_rates: List[float]
    ebitda_margin: float
    da_pct: float
    capex_pct: float
    nwc_pct: float
    tax_rate: float
    terminal_growth: float
    exit_multiple: float
    wacc: float
    scenario: str = "Base"


class WallStreetDCF:
    """월가 스타일 DCF 모델 (v4 - WACC 자동화)"""

    def __init__(self, financial_data: dict):
        self.data = financial_data
        self.sector = financial_data.get('sector', 'Default')
        self.sector_defaults = SECTOR_DEFAULTS.get(self.sector, SECTOR_DEFAULTS['Default'])
        self.historical = self._extract_historical()

        self.actual_fcf = financial_data.get('fcf', 0)
        self.actual_fcf_margin = self.actual_fcf / financial_data.get('revenue', 1) if financial_data.get('revenue', 0) > 0 else 0

        # WACC 계산 (자동)
        self.risk_free_rate = get_risk_free_rate()
        self.wacc_data = None  # calculate_auto_wacc()로 계산
    
    def calculate_auto_wacc(
        self,
        market_risk_premium: float = 0.055,
        use_adjusted_beta: bool = True,
        include_cash_in_debt: bool = False
    ) -> Dict:
        """
        WACC 자동 계산 (Synthetic Rating + Adjusted Beta)

        Returns:
            {
                'wacc': float,
                'cost_of_equity': dict,
                'cost_of_debt': dict,
                'calculation_log': list
            }
        """
        wacc_result = calculate_full_wacc(
            financial_data=self.data,
            risk_free_rate=self.risk_free_rate,
            market_risk_premium=market_risk_premium,
            use_adjusted_beta=use_adjusted_beta,
            include_cash_in_debt=include_cash_in_debt
        )

        self.wacc_data = wacc_result
        return wacc_result

    def _extract_historical(self) -> pd.DataFrame:
        """과거 재무 데이터 추출"""
        hist_list = self.data.get('historical_financials', [])
        
        if not hist_list:
            return pd.DataFrame([{
                'year': 'TTM',
                'revenue': self.data.get('revenue', 0),
                'ebitda': self.data.get('ebitda', 0),
                'fcf': self.data.get('fcf', 0),
            }])
        
        df = pd.DataFrame(hist_list)
        
        if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
            cash = df.get('cash', 0) if 'cash' in df.columns else 0
            df['nwc'] = df['current_assets'] - cash - df['current_liabilities']
        else:
            df['nwc'] = 0
        
        if 'ebitda' in df.columns and 'revenue' in df.columns:
            df['ebitda_margin'] = df['ebitda'] / df['revenue']
        
        return df.sort_values('year', ascending=True) if 'year' in df.columns else df
    
    def calculate_cagr(self, values: list, years: int) -> float:
        """CAGR 계산"""
        if len(values) < 2:
            return 0.0
        
        # 최근 n년 데이터
        if len(values) > years:
            values = values[-years-1:]  # years+1개 필요 (시작점 포함)
        
        start = values[0]
        end = values[-1]
        n = len(values) - 1
        
        if start <= 0 or end <= 0 or n <= 0:
            return 0.0
        
        return (end / start) ** (1 / n) - 1
    
    def get_historical_averages(self) -> dict:
        """
        과거 평균값 계산 (★ 회사 실제 데이터 우선)
        - 섹터 기본값은 데이터 없을 때만 사용
        - 현재 거래 배수 참고하여 Exit Multiple 산정
        """
        df = self.historical
        defaults = self.sector_defaults

        # Revenue 리스트 추출
        if 'revenue' in df.columns:
            revenues = df['revenue'].dropna().tolist()
        else:
            revenues = [self.data.get('revenue', 0)]

        # 3Y, 5Y CAGR 계산
        cagr_3y = self.calculate_cagr(revenues, 3)
        cagr_5y = self.calculate_cagr(revenues, 5)

        # 가중 평균 (3Y × 70% + 5Y × 30%)
        if cagr_3y > 0 and cagr_5y > 0:
            blended_growth = cagr_3y * 0.7 + cagr_5y * 0.3
        elif cagr_3y > 0:
            blended_growth = cagr_3y
        elif cagr_5y > 0:
            blended_growth = cagr_5y
        else:
            # yfinance에서 제공하는 성장률 사용
            blended_growth = self.data.get('revenue_growth', 0.05)

        # ★ 성장률: 고성장 기업 반영 (절대 상한만 적용)
        # Decay를 통해 점진적 감소하므로 초기 성장률은 높게 유지
        base_growth = min(blended_growth, 0.80)  # 절대 상한 80%
        base_growth = max(base_growth, 0.02)  # 최소 2%
        sector_cap = defaults.get('growth_cap', 0.15)

        # ★ EBITDA 마진: 회사 실제 데이터 우선
        revenue = self.data.get('revenue', 0)
        ebitda = self.data.get('ebitda', 0)

        if revenue > 0 and ebitda > 0:
            # 현재 TTM 마진 사용 (가장 정확)
            current_margin = ebitda / revenue
            # 과거 데이터가 있으면 평균과 블렌딩
            if 'ebitda_margin' in df.columns:
                hist_margins = df['ebitda_margin'].dropna()
                if len(hist_margins) > 0:
                    hist_avg = hist_margins.mean()
                    # 최근 마진에 가중치 (70% 현재, 30% 과거)
                    avg_margin = current_margin * 0.7 + hist_avg * 0.3
                else:
                    avg_margin = current_margin
            else:
                avg_margin = current_margin
        elif 'ebitda_margin' in df.columns:
            margin = df['ebitda_margin'].dropna()
            avg_margin = margin.mean() if len(margin) > 0 else defaults['ebitda_margin']
        else:
            avg_margin = defaults['ebitda_margin']

        # ★ D&A %: 실제 감가상각비 데이터 우선 (핵심 수정!)
        # 주의: EBITDA - Operating Income은 주식보상비용 등을 포함하므로 부정확함

        # 1순위: 과거 재무제표의 실제 D&A 데이터
        if 'depreciation' in df.columns:
            dep_values = df['depreciation'].dropna()
            rev_values = df['revenue'].dropna()
            if len(dep_values) > 0 and len(rev_values) > 0:
                # 최근 데이터 가중 평균
                da_pcts = []
                for i, (dep, rev) in enumerate(zip(dep_values, rev_values)):
                    if rev > 0 and dep > 0:
                        da_pcts.append(dep / rev)
                if da_pcts:
                    # 최근 데이터에 가중치
                    if len(da_pcts) >= 3:
                        avg_da = da_pcts[0] * 0.5 + da_pcts[1] * 0.3 + sum(da_pcts[2:]) / len(da_pcts[2:]) * 0.2
                    else:
                        avg_da = sum(da_pcts) / len(da_pcts)
                else:
                    avg_da = defaults['da_pct']
            else:
                avg_da = defaults['da_pct']
        # 2순위: da_pct 컬럼이 있으면 사용
        elif 'da_pct' in df.columns:
            da_pcts = df['da_pct'].dropna()
            avg_da = da_pcts.mean() if len(da_pcts) > 0 else defaults['da_pct']
        else:
            avg_da = defaults['da_pct']

        # D&A 최소값 보장 (대부분 기업은 최소 3% 이상)
        avg_da = max(avg_da, 0.03)

        # ★ CapEx %: 실제 CapEx 데이터 우선 (핵심 수정!)
        # 1순위: 과거 재무제표의 실제 CapEx 데이터
        if 'capex' in df.columns:
            capex_values = df['capex'].dropna()
            rev_values = df['revenue'].dropna()
            if len(capex_values) > 0 and len(rev_values) > 0:
                capex_pcts = []
                for capex_val, rev in zip(capex_values, rev_values):
                    if rev > 0 and capex_val > 0:
                        capex_pcts.append(capex_val / rev)
                if capex_pcts:
                    # 최근 데이터에 가중치
                    if len(capex_pcts) >= 3:
                        avg_capex = capex_pcts[0] * 0.5 + capex_pcts[1] * 0.3 + sum(capex_pcts[2:]) / len(capex_pcts[2:]) * 0.2
                    else:
                        avg_capex = sum(capex_pcts) / len(capex_pcts)
                else:
                    avg_capex = defaults['capex_pct']
            else:
                avg_capex = defaults['capex_pct']
        # 2순위: Operating CF - FCF 방식
        elif self.data.get('operating_cf', 0) > 0 and self.data.get('fcf', 0) > 0 and revenue > 0:
            op_cf = self.data.get('operating_cf', 0)
            fcf = self.data.get('fcf', 0)
            avg_capex = (op_cf - fcf) / revenue
        # 3순위: capex_pct 컬럼
        elif 'capex_pct' in df.columns:
            capex_pcts = df['capex_pct'].dropna()
            avg_capex = capex_pcts.mean() if len(capex_pcts) > 0 else defaults['capex_pct']
        else:
            avg_capex = defaults['capex_pct']

        # CapEx sanity check: D&A와 비교 (일반적으로 CapEx ≈ D&A × 0.8 ~ 1.5)
        # 너무 높은 CapEx는 UFCF를 왜곡시킴
        if avg_capex > avg_da * 2.5:
            # CapEx가 D&A의 2.5배 초과시 조정 (maintenance capex 개념)
            avg_capex = min(avg_capex, avg_da * 2.0)

        # ★ NWC %: 실제 데이터 우선
        current_assets = self.data.get('current_assets', 0)
        current_liabilities = self.data.get('current_liabilities', 0)
        cash = self.data.get('cash', 0)

        if current_assets > 0 and revenue > 0:
            # NWC = Current Assets - Cash - Current Liabilities
            nwc = current_assets - cash - current_liabilities
            avg_nwc = nwc / revenue
        elif 'nwc' in df.columns and 'revenue' in df.columns:
            nwc_pct = (df['nwc'] / df['revenue']).dropna()
            avg_nwc = nwc_pct.mean() if len(nwc_pct) > 0 else defaults['nwc_pct']
        else:
            avg_nwc = defaults['nwc_pct']

        # ★ Exit Multiple: 현재 거래 배수 기반 (핵심!)
        current_ev_ebitda = self.data.get('ev_ebitda', 0)
        if current_ev_ebitda and current_ev_ebitda > 0:
            # 현재 배수의 70% 사용 (보수적 할인)
            # 하지만 섹터 평균보다는 높게 유지
            suggested_exit = max(
                current_ev_ebitda * 0.70,  # 현재의 70%
                defaults.get('exit_multiple', 12)  # 섹터 최소
            )
            # 상한 40x
            suggested_exit = min(suggested_exit, 40)
        else:
            suggested_exit = defaults.get('exit_multiple', 12)

        # 실제 FCF 마진
        fcf = self.data.get('fcf', 0)
        actual_fcf_margin = fcf / revenue if revenue > 0 and fcf > 0 else 0

        # ★ UFCF 예상 마진 사전 계산 (sanity check용)
        # UFCF ≈ (EBITDA margin - D&A) * (1-tax) + D&A - CapEx
        # tax rate는 아직 모르므로 21% 가정
        est_ebit_margin = avg_margin - avg_da
        est_nopat_margin = est_ebit_margin * 0.79 if est_ebit_margin > 0 else est_ebit_margin
        est_ufcf_margin = est_nopat_margin + avg_da - avg_capex

        # ★ 실제 FCF 마진과 비교하여 자동 조정
        # 예상 UFCF가 실제 FCF와 너무 다르면 (10%p 이상 차이) 경고 및 조정
        if actual_fcf_margin > 0 and abs(est_ufcf_margin - actual_fcf_margin) > 0.10:
            # 차이가 크면 CapEx 조정 (가장 불확실한 변수)
            # 목표: 예상 UFCF ≈ 실제 FCF의 ±5%p 내
            target_ufcf = actual_fcf_margin
            # CapEx = NOPAT + D&A - target_UFCF
            adjusted_capex = est_nopat_margin + avg_da - target_ufcf
            # 합리적 범위 내에서만 조정
            if 0.02 <= adjusted_capex <= avg_da * 2.0:
                avg_capex = adjusted_capex

        # 최종 경계값 적용
        final_da = min(max(avg_da, 0.03), 0.20)  # 3%~20%
        final_capex = min(max(avg_capex, 0.02), 0.20)  # 2%~20%
        final_nwc = min(max(avg_nwc, -0.20), 0.25)  # -20%~25%

        return {
            # ★ CAGR 정보
            'cagr_3y': cagr_3y,
            'cagr_5y': cagr_5y,
            'blended_growth': blended_growth,
            'base_growth': base_growth,
            'sector_cap': sector_cap,

            # ★ 회사 실제 데이터 기반 가정값
            'avg_ebitda_margin': min(max(avg_margin, 0.05), 0.80),
            'avg_da_pct': final_da,
            'avg_capex_pct': final_capex,
            'avg_nwc_pct': final_nwc,
            'actual_fcf_margin': actual_fcf_margin,

            # ★ 디버그 정보 (UI에서 표시 가능)
            'est_ufcf_margin': est_ufcf_margin,

            # ★ Exit Multiple (현재 배수 기반)
            'current_ev_ebitda': current_ev_ebitda,
            'suggested_exit_multiple': suggested_exit,

            'sector_defaults': defaults,
        }
    
    def generate_growth_schedule(
        self,
        initial_growth: float,
        terminal_growth: float = 0.025,
        years: int = 5,
        decay_type: str = 'linear'
    ) -> List[float]:
        """
        성장률 스케줄 생성 (점진적 decay)

        Args:
            initial_growth: 첫 해 성장률
            terminal_growth: 최종 연도 목표 성장률 (terminal growth에 수렴)
            years: 예측 기간
            decay_type: 'linear', 'exponential', 'step'

        Returns:
            각 연도별 성장률 리스트
        """
        if years <= 1:
            return [initial_growth]

        # 최종 연도 성장률은 terminal growth보다 약간 높게 (버퍼)
        final_growth = max(terminal_growth * 1.5, 0.03)  # 최소 3%

        if decay_type == 'linear':
            # 선형 감소
            step = (initial_growth - final_growth) / (years - 1)
            return [initial_growth - (step * i) for i in range(years)]

        elif decay_type == 'exponential':
            # 지수 감소 (초반에 빠르게 감소)
            if initial_growth <= final_growth:
                return [initial_growth] * years
            ratio = (final_growth / initial_growth) ** (1 / (years - 1))
            return [initial_growth * (ratio ** i) for i in range(years)]

        elif decay_type == 'step':
            # 단계적 감소 (2년씩 유지)
            rates = []
            current = initial_growth
            step = (initial_growth - final_growth) / ((years + 1) // 2)
            for i in range(years):
                rates.append(current)
                if (i + 1) % 2 == 0:
                    current = max(current - step, final_growth)
            return rates

        else:
            return [initial_growth] * years

    def build_projections(self, assumptions: DCFAssumptions, years: int = 5) -> pd.DataFrame:
        """Revenue → UFCF Bottom-up 빌드 (Mid-year convention 적용)"""
        projections = []

        base_revenue = self.data.get('revenue', 0)
        if base_revenue == 0:
            return pd.DataFrame()

        # 현재 NWC를 기준으로 시작 (실제 데이터 우선)
        current_nwc = self.data.get('working_capital', base_revenue * assumptions.nwc_pct)
        prev_nwc = current_nwc

        for i in range(years):
            year = i + 1

            if i < len(assumptions.revenue_growth_rates):
                growth = assumptions.revenue_growth_rates[i]
            else:
                growth = assumptions.revenue_growth_rates[-1]

            if i == 0:
                revenue = base_revenue * (1 + growth)
            else:
                revenue = projections[-1]['revenue'] * (1 + growth)

            ebitda = revenue * assumptions.ebitda_margin
            da = revenue * assumptions.da_pct
            ebit = ebitda - da

            # EBIT이 음수면 세금 없음 (NOL 가정)
            if ebit > 0:
                nopat = ebit * (1 - assumptions.tax_rate)
            else:
                nopat = ebit  # 손실 시 세금 없음

            capex = revenue * assumptions.capex_pct

            nwc = revenue * assumptions.nwc_pct
            delta_nwc = nwc - prev_nwc
            prev_nwc = nwc

            ufcf = nopat + da - capex - delta_nwc

            # Mid-year convention: 현금흐름이 연중에 발생한다고 가정
            discount_period = year - 0.5

            projections.append({
                'year': year,
                'discount_period': discount_period,  # mid-year
                'revenue_growth': growth,
                'revenue': revenue,
                'ebitda': ebitda,
                'ebitda_margin': ebitda / revenue if revenue > 0 else 0,
                'da': da,
                'ebit': ebit,
                'nopat': nopat,
                'capex': capex,
                'delta_nwc': delta_nwc,
                'ufcf': ufcf,
                'ufcf_margin': ufcf / revenue if revenue > 0 else 0
            })

        return pd.DataFrame(projections)

    def build_projections_with_convergence(
        self,
        growth_schedule: List[float],
        margin_schedule: List[float],
        capex_schedule: List[float],
        tax_schedule: List[float],
        da_pct: float,
        nwc_pct: float,
        wacc: float,
        terminal_growth: float,
        exit_multiple: float
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Context-Aware Projection with Convergence Schedules

        각 연도별로 다른 가정값을 적용하여 보다 현실적인 projection 생성

        Args:
            growth_schedule: 연도별 성장률 (decay 반영)
            margin_schedule: 연도별 EBITDA 마진 (수렴 반영)
            capex_schedule: 연도별 CapEx % (수렴 반영)
            tax_schedule: 연도별 세율 (정상화 반영)
            da_pct: D&A % (고정)
            nwc_pct: NWC % (고정)
            wacc: 할인율
            terminal_growth: 영구 성장률
            exit_multiple: Exit EV/EBITDA 배수

        Returns:
            (projections DataFrame, valuation dict)
        """
        projections = []
        years = len(growth_schedule)

        base_revenue = self.data.get('revenue', 0)
        if base_revenue == 0:
            return pd.DataFrame(), {}

        current_nwc = self.data.get('working_capital', base_revenue * nwc_pct)
        prev_nwc = current_nwc

        for i in range(years):
            year = i + 1
            growth = growth_schedule[i]
            ebitda_margin = margin_schedule[i] if i < len(margin_schedule) else margin_schedule[-1]
            capex_pct = capex_schedule[i] if i < len(capex_schedule) else capex_schedule[-1]
            tax_rate = tax_schedule[i] if i < len(tax_schedule) else tax_schedule[-1]

            if i == 0:
                revenue = base_revenue * (1 + growth)
            else:
                revenue = projections[-1]['revenue'] * (1 + growth)

            ebitda = revenue * ebitda_margin
            da = revenue * da_pct
            ebit = ebitda - da

            if ebit > 0:
                nopat = ebit * (1 - tax_rate)
            else:
                nopat = ebit

            capex = revenue * capex_pct
            nwc = revenue * nwc_pct
            delta_nwc = nwc - prev_nwc
            prev_nwc = nwc

            ufcf = nopat + da - capex - delta_nwc
            discount_period = year - 0.5

            projections.append({
                'year': year,
                'discount_period': discount_period,
                'revenue_growth': growth,
                'revenue': revenue,
                'ebitda': ebitda,
                'ebitda_margin': ebitda_margin,
                'da': da,
                'da_pct': da_pct,
                'ebit': ebit,
                'tax_rate': tax_rate,
                'nopat': nopat,
                'capex': capex,
                'capex_pct': capex_pct,
                'delta_nwc': delta_nwc,
                'ufcf': ufcf,
                'ufcf_margin': ufcf / revenue if revenue > 0 else 0
            })

        df = pd.DataFrame(projections)

        # Discount factors
        df['discount_factor'] = [1 / ((1 + wacc) ** dp) for dp in df['discount_period']]
        df['pv_ufcf'] = df['ufcf'] * df['discount_factor']
        sum_pv_fcf = df['pv_ufcf'].sum()

        # Terminal Value calculations
        final = df.iloc[-1]
        final_ufcf = final['ufcf']
        final_ebitda = final['ebitda']
        final_year = final['year']

        valuation = {}

        # Perpetuity Growth Method
        spread = wacc - terminal_growth
        if spread > 0.02:
            tv_perpetuity = final_ufcf * (1 + terminal_growth) / spread
            pv_tv_perpetuity = tv_perpetuity / ((1 + wacc) ** final_year)
            ev_perpetuity = sum_pv_fcf + pv_tv_perpetuity

            valuation['perpetuity'] = {
                'terminal_value': tv_perpetuity,
                'pv_terminal_value': pv_tv_perpetuity,
                'enterprise_value': ev_perpetuity,
                'tv_pct': pv_tv_perpetuity / ev_perpetuity if ev_perpetuity > 0 else 0,
                'implied_multiple': tv_perpetuity / final_ebitda if final_ebitda > 0 else 0
            }

        # Exit Multiple Method
        if final_ebitda > 0:
            tv_exit = final_ebitda * exit_multiple
            pv_tv_exit = tv_exit / ((1 + wacc) ** final_year)
            ev_exit = sum_pv_fcf + pv_tv_exit

            valuation['exit_multiple'] = {
                'terminal_value': tv_exit,
                'pv_terminal_value': pv_tv_exit,
                'enterprise_value': ev_exit,
                'tv_pct': pv_tv_exit / ev_exit if ev_exit > 0 else 0,
                'exit_multiple_used': exit_multiple
            }

        # Equity Value calculation
        cash = self.data.get('cash', 0)
        debt = self.data.get('total_debt', 0)
        minority_interest = self.data.get('minority_interest', 0)
        preferred_stock = self.data.get('preferred_stock', 0)
        shares = self.data.get('shares_outstanding', 0)

        for method in ['perpetuity', 'exit_multiple']:
            if method in valuation:
                ev = valuation[method]['enterprise_value']
                equity = ev - debt - minority_interest - preferred_stock + cash
                per_share = equity / shares if shares > 0 else 0

                valuation[method]['equity_value'] = equity
                valuation[method]['per_share_value'] = per_share
                valuation[method]['cash'] = cash
                valuation[method]['total_debt'] = debt

        valuation['sum_pv_fcf'] = sum_pv_fcf
        valuation['current_price'] = self.data.get('current_price', 0)

        return df, valuation

    def get_smart_defaults(self) -> Dict:
        """
        Context-Aware Smart Defaults 생성

        Returns:
            {
                'lifecycle': LifecycleResult,
                'projection_years': int,
                'growth_schedule': list,
                'margin_schedule': list,
                'capex_schedule': list,
                'tax_schedule': list,
                'terminal_growth': float,
                'insights': list,
                'assumptions': dict  # UI용 기본값
            }
        """
        smart = generate_smart_defaults(
            financial_data=self.data,
            risk_free_rate=self.risk_free_rate,
            sector=self.sector
        )

        # Historical Averages도 포함
        hist_avg = self.get_historical_averages()

        return {
            'lifecycle': smart.lifecycle,
            'projection_years': smart.projection_years,
            'growth_schedule': smart.growth_schedule,
            'margin_schedule': smart.margin_schedule,
            'capex_schedule': smart.capex_schedule,
            'tax_schedule': smart.tax_schedule,
            'terminal_growth': smart.terminal_growth,
            'insights': smart.insights,
            'assumptions': {
                'initial_growth': smart.growth_schedule[0] if smart.growth_schedule else 0,
                'ebitda_margin': hist_avg.get('avg_ebitda_margin', 0.20),
                'da_pct': hist_avg.get('avg_da_pct', 0.05),
                'capex_pct': hist_avg.get('avg_capex_pct', 0.05),
                'nwc_pct': hist_avg.get('avg_nwc_pct', 0.05),
                'exit_multiple': hist_avg.get('suggested_exit_multiple', 12),
                'tax_rate': self.data.get('tax_rate', 0.21),
            },
            'historical_averages': hist_avg
        }
    
    def sanity_check(self, projections: pd.DataFrame) -> dict:
        """FCF Sanity Check - 개선된 버전"""
        if projections.empty:
            return {'pass': False, 'message': 'No projections', 'warnings': []}

        warnings = []
        actual_fcf_margin = self.actual_fcf_margin

        # Year 1 예측 마진
        projected_margin = projections.iloc[0]['ufcf_margin']
        year1_growth = projections.iloc[0]['revenue_growth']

        # ★ 허용 범위 개선: 기본 5%p + 성장률 반영
        # 고성장 기업도 실제 FCF와 너무 다르면 안 됨
        base_tolerance = 0.05  # 기본 5%p
        growth_adjustment = min(year1_growth * 0.3, 0.10)  # 최대 10%p 추가
        tolerance = base_tolerance + growth_adjustment

        diff = abs(projected_margin - actual_fcf_margin)

        # ★ UFCF 마진이 너무 낮으면 명확한 경고
        if projected_margin < 0.05 and actual_fcf_margin > 0.10:
            warnings.append(f"⚠️ UFCF 마진이 너무 낮음: {projected_margin*100:.1f}% (실제 FCF: {actual_fcf_margin*100:.1f}%)")

        # ★ 경고 조건 (더 구체적)
        if diff > tolerance:
            direction = "낮음" if projected_margin < actual_fcf_margin else "높음"
            warnings.append(f"예측 FCF가 실제보다 {direction}: {projected_margin*100:.1f}% vs {actual_fcf_margin*100:.1f}%")

        # EBITDA 마진 체크
        if 'ebitda_margin' in projections.columns:
            proj_ebitda_margin = projections.iloc[0]['ebitda_margin']
            if proj_ebitda_margin > 0.60:
                warnings.append(f"EBITDA 마진이 높음: {proj_ebitda_margin*100:.1f}%")
            elif proj_ebitda_margin < 0.05:
                warnings.append(f"EBITDA 마진이 낮음: {proj_ebitda_margin*100:.1f}%")

        # ★ 5년 평균 UFCF 마진 계산 (TV 비중 예측용)
        avg_ufcf_margin = projections['ufcf_margin'].mean()

        # 전체 통과 여부 결정 (더 관대하게)
        is_pass = diff <= tolerance or (projected_margin > 0.05 and diff <= 0.10)

        return {
            'pass': is_pass,
            'message': f'예측 {projected_margin*100:.1f}% vs 실제 {actual_fcf_margin*100:.1f}% (허용: ±{tolerance*100:.1f}%p)',
            'actual': actual_fcf_margin,
            'projected': projected_margin,
            'avg_ufcf_margin': avg_ufcf_margin,
            'diff': diff,
            'tolerance': tolerance,
            'warnings': warnings
        }
    
    def calculate_terminal_value(self, projections: pd.DataFrame, assumptions: DCFAssumptions) -> dict:
        """Terminal Value 계산 (Mid-year convention 적용)"""
        if projections.empty:
            return {}

        final = projections.iloc[-1]
        final_ufcf = final['ufcf']
        final_ebitda = final['ebitda']
        final_year = final['year']

        result = {}

        # WACC vs Terminal Growth 검증 (최소 2%p 차이 필요)
        spread = assumptions.wacc - assumptions.terminal_growth
        min_spread = 0.02  # 최소 2%p

        # Perpetuity Growth Method
        if spread > min_spread:
            # Gordon Growth: TV = FCF_n * (1+g) / (WACC - g)
            # TV는 Year n 말 기준 가치
            tv = final_ufcf * (1 + assumptions.terminal_growth) / spread

            # Mid-year convention: TV는 연말에 발생하므로 full year로 할인
            pv_tv = tv / ((1 + assumptions.wacc) ** final_year)

            implied_multiple = tv / final_ebitda if final_ebitda > 0 else 0

            result['perpetuity'] = {
                'terminal_value': tv,
                'pv_terminal_value': pv_tv,
                'implied_multiple': implied_multiple,
                'spread': spread,
                'warning': None if implied_multiple < 30 else f"Implied multiple {implied_multiple:.1f}x가 너무 높음"
            }
        elif spread > 0:
            # Spread가 너무 작으면 경고와 함께 계산
            tv = final_ufcf * (1 + assumptions.terminal_growth) / spread
            pv_tv = tv / ((1 + assumptions.wacc) ** final_year)
            implied_multiple = tv / final_ebitda if final_ebitda > 0 else 0

            result['perpetuity'] = {
                'terminal_value': tv,
                'pv_terminal_value': pv_tv,
                'implied_multiple': implied_multiple,
                'spread': spread,
                'warning': f"WACC-Growth spread ({spread*100:.1f}%p)가 너무 작아 TV가 과대평가될 수 있음"
            }

        # Exit Multiple Method
        if final_ebitda > 0:
            tv = final_ebitda * assumptions.exit_multiple
            # TV는 Year n 말 기준이므로 full year로 할인
            pv_tv = tv / ((1 + assumptions.wacc) ** final_year)

            result['exit_multiple'] = {
                'terminal_value': tv,
                'pv_terminal_value': pv_tv,
                'exit_multiple': assumptions.exit_multiple,
                'warning': None
            }
        else:
            # 음수 EBITDA인 경우 Revenue multiple 사용
            final_revenue = final['revenue']
            # 섹터별 적정 Revenue multiple (보수적)
            rev_multiple = self.sector_defaults.get('exit_multiple', 12) * 0.3  # EBITDA multiple의 30%
            tv = final_revenue * rev_multiple
            pv_tv = tv / ((1 + assumptions.wacc) ** final_year)

            result['exit_multiple'] = {
                'terminal_value': tv,
                'pv_terminal_value': pv_tv,
                'exit_multiple': rev_multiple,
                'method': 'revenue_multiple',
                'warning': f"EBITDA가 음수라 Revenue multiple ({rev_multiple:.1f}x) 사용"
            }

        return result
    
    def calculate_dcf(self, assumptions: DCFAssumptions, years: int = 5) -> dict:
        """전체 DCF 계산 (Mid-year convention 적용)"""
        projections = self.build_projections(assumptions, years)

        if projections.empty:
            return {'error': 'No projections'}

        sanity = self.sanity_check(projections)

        # Mid-year convention 적용된 discount_period 사용
        projections['discount_factor'] = [
            1 / ((1 + assumptions.wacc) ** dp)
            for dp in projections['discount_period']
        ]
        projections['pv_ufcf'] = projections['ufcf'] * projections['discount_factor']

        sum_pv_fcf = projections['pv_ufcf'].sum()

        tv_results = self.calculate_terminal_value(projections, assumptions)

        results = {}
        warnings = sanity.get('warnings', [])

        for method in ['perpetuity', 'exit_multiple']:
            if method not in tv_results:
                continue

            tv = tv_results[method]
            pv_tv = tv['pv_terminal_value']
            ev = sum_pv_fcf + pv_tv

            # EV가 음수면 문제 있음
            if ev <= 0:
                warnings.append(f"{method}: EV가 음수 또는 0 (${ev:,.0f})")
                continue

            cash = self.data.get('cash', 0)
            debt = self.data.get('total_debt', 0)
            minority_interest = self.data.get('minority_interest', 0)
            preferred_stock = self.data.get('preferred_stock', 0)

            # Equity Value = EV - Net Debt - Minority - Preferred + Cash
            equity = ev - debt - minority_interest - preferred_stock + cash

            shares = self.data.get('shares_outstanding', 0)
            per_share = equity / shares if shares > 0 else 0

            tv_pct = pv_tv / ev if ev > 0 else 0

            # TV가 EV의 85% 이상이면 경고
            if tv_pct > 0.85:
                warnings.append(f"{method}: TV가 EV의 {tv_pct*100:.0f}%로 너무 높음 (projection period FCF가 낮음)")

            # TV 관련 경고 추가
            if tv.get('warning'):
                warnings.append(tv['warning'])

            results[method] = {
                'sum_pv_fcf': sum_pv_fcf,
                'pv_terminal_value': pv_tv,
                'tv_pct_of_ev': tv_pct,
                'enterprise_value': ev,
                'cash': cash,
                'total_debt': debt,
                'equity_value': equity,
                'shares_outstanding': shares,
                'per_share_value': per_share,
            }

        return {
            'scenario': assumptions.scenario,
            'projections': projections,
            'valuations': results,
            'sanity_check': sanity,
            'warnings': warnings,
            'current_price': self.data.get('current_price', 0)
        }
    
    def run_scenarios(
        self,
        base_assumptions: dict,
        wacc: float,
        tax_rate: float = 0.21,
        bull_growth_factor: float = 1.20,  # 성장률 +20%
        bear_growth_factor: float = 0.70,  # 성장률 -30%
        bull_multiple_factor: float = 1.15,  # Multiple +15%
        bear_multiple_factor: float = 0.85,  # Multiple -15%
    ) -> dict:
        """
        Bull / Base / Bear 시나리오
        ★ 비율 기반 조정 (고성장 기업에도 적절한 범위 유지)
        """
        scenarios = {}

        base_growth = base_assumptions['revenue_growth']
        base_exit = base_assumptions['exit_multiple']
        terminal_growth = base_assumptions.get('terminal_growth', 0.025)

        # ===== BASE =====
        base = DCFAssumptions(
            revenue_growth_rates=base_growth,
            ebitda_margin=base_assumptions['ebitda_margin'],
            da_pct=base_assumptions['da_pct'],
            capex_pct=base_assumptions['capex_pct'],
            nwc_pct=base_assumptions['nwc_pct'],
            tax_rate=tax_rate,
            terminal_growth=terminal_growth,
            exit_multiple=base_exit,
            wacc=wacc,
            scenario="Base"
        )
        scenarios['base'] = self.calculate_dcf(base)

        # ===== BULL =====
        # 비율 기반 성장률 조정 (절대 상한 80%)
        bull_growth = [min(g * bull_growth_factor, 0.80) for g in base_growth]
        bull_exit = min(base_exit * bull_multiple_factor, 35)  # Multiple 상한 35x

        bull = DCFAssumptions(
            revenue_growth_rates=bull_growth,
            ebitda_margin=base_assumptions['ebitda_margin'],
            da_pct=base_assumptions['da_pct'],
            capex_pct=base_assumptions['capex_pct'],
            nwc_pct=base_assumptions['nwc_pct'],
            tax_rate=tax_rate,
            terminal_growth=terminal_growth,
            exit_multiple=bull_exit,
            wacc=wacc,
            scenario="Bull"
        )
        scenarios['bull'] = self.calculate_dcf(bull)

        # ===== BEAR =====
        # 비율 기반 성장률 조정 (최소 0%)
        bear_growth = [max(g * bear_growth_factor, 0.0) for g in base_growth]
        bear_exit = max(base_exit * bear_multiple_factor, 5.0)  # Multiple 하한 5x

        bear = DCFAssumptions(
            revenue_growth_rates=bear_growth,
            ebitda_margin=base_assumptions['ebitda_margin'],
            da_pct=base_assumptions['da_pct'],
            capex_pct=base_assumptions['capex_pct'],
            nwc_pct=base_assumptions['nwc_pct'],
            tax_rate=tax_rate,
            terminal_growth=terminal_growth,
            exit_multiple=bear_exit,
            wacc=wacc,
            scenario="Bear"
        )
        scenarios['bear'] = self.calculate_dcf(bear)
        
        # Summary
        current = self.data.get('current_price', 0)
        summary = {
            'current_price': current,
            # Base
            'base_growth': base_growth[0] if base_growth else 0,
            'base_exit': base_exit,
            'base_perpetuity': scenarios['base']['valuations'].get('perpetuity', {}).get('per_share_value', 0),
            'base_exit_val': scenarios['base']['valuations'].get('exit_multiple', {}).get('per_share_value', 0),
            # Bull
            'bull_growth': bull_growth[0] if bull_growth else 0,
            'bull_exit': bull_exit,
            'bull_perpetuity': scenarios['bull']['valuations'].get('perpetuity', {}).get('per_share_value', 0),
            'bull_exit_val': scenarios['bull']['valuations'].get('exit_multiple', {}).get('per_share_value', 0),
            # Bear
            'bear_growth': bear_growth[0] if bear_growth else 0,
            'bear_exit': bear_exit,
            'bear_perpetuity': scenarios['bear']['valuations'].get('perpetuity', {}).get('per_share_value', 0),
            'bear_exit_val': scenarios['bear']['valuations'].get('exit_multiple', {}).get('per_share_value', 0),
        }
        
        # Weighted (25/50/25)
        if all(summary.get(k, 0) > 0 for k in ['bull_perpetuity', 'base_perpetuity', 'bear_perpetuity']):
            summary['weighted_perpetuity'] = (
                summary['bull_perpetuity'] * 0.25 +
                summary['base_perpetuity'] * 0.50 +
                summary['bear_perpetuity'] * 0.25
            )
        
        if all(summary.get(k, 0) > 0 for k in ['bull_exit_val', 'base_exit_val', 'bear_exit_val']):
            summary['weighted_exit'] = (
                summary['bull_exit_val'] * 0.25 +
                summary['base_exit_val'] * 0.50 +
                summary['bear_exit_val'] * 0.25
            )
        
        scenarios['summary'] = summary
        return scenarios
    
    def sensitivity_analysis(
        self,
        base_assumptions: DCFAssumptions,
        wacc_range: List[float] = None,
        growth_range: List[float] = None,
        exit_range: List[float] = None
    ) -> dict:
        """2-Way Sensitivity"""
        if wacc_range is None:
            w = base_assumptions.wacc
            wacc_range = [w - 0.02, w - 0.01, w, w + 0.01, w + 0.02]
        
        if growth_range is None:
            g = base_assumptions.terminal_growth
            growth_range = [g - 0.01, g - 0.005, g, g + 0.005, g + 0.01]
        
        if exit_range is None:
            e = base_assumptions.exit_multiple
            exit_range = [e - 4, e - 2, e, e + 2, e + 4]
        
        # WACC vs Terminal Growth
        wacc_growth_table = []
        for wacc in wacc_range:
            row = {'WACC': f"{wacc*100:.1f}%"}
            for tg in growth_range:
                if wacc > tg:
                    temp = DCFAssumptions(
                        revenue_growth_rates=base_assumptions.revenue_growth_rates,
                        ebitda_margin=base_assumptions.ebitda_margin,
                        da_pct=base_assumptions.da_pct,
                        capex_pct=base_assumptions.capex_pct,
                        nwc_pct=base_assumptions.nwc_pct,
                        tax_rate=base_assumptions.tax_rate,
                        terminal_growth=tg,
                        exit_multiple=base_assumptions.exit_multiple,
                        wacc=wacc,
                    )
                    result = self.calculate_dcf(temp)
                    val = result['valuations'].get('perpetuity', {}).get('per_share_value', 0)
                    row[f"g={tg*100:.1f}%"] = f"${val:.2f}"
                else:
                    row[f"g={tg*100:.1f}%"] = "N/A"
            wacc_growth_table.append(row)
        
        # WACC vs Exit Multiple
        wacc_exit_table = []
        for wacc in wacc_range:
            row = {'WACC': f"{wacc*100:.1f}%"}
            for em in exit_range:
                temp = DCFAssumptions(
                    revenue_growth_rates=base_assumptions.revenue_growth_rates,
                    ebitda_margin=base_assumptions.ebitda_margin,
                    da_pct=base_assumptions.da_pct,
                    capex_pct=base_assumptions.capex_pct,
                    nwc_pct=base_assumptions.nwc_pct,
                    tax_rate=base_assumptions.tax_rate,
                    terminal_growth=base_assumptions.terminal_growth,
                    exit_multiple=em,
                    wacc=wacc,
                )
                result = self.calculate_dcf(temp)
                val = result['valuations'].get('exit_multiple', {}).get('per_share_value', 0)
                row[f"{em:.0f}x"] = f"${val:.2f}"
            wacc_exit_table.append(row)
        
        return {
            'wacc_vs_growth': pd.DataFrame(wacc_growth_table),
            'wacc_vs_exit': pd.DataFrame(wacc_exit_table)
        }

    def reverse_dcf(
        self,
        base_assumptions: dict,
        wacc: float,
        tax_rate: float = 0.21,
        years: int = 5
    ) -> dict:
        """
        Reverse DCF: 현재 주가가 암시하는 성장률 계산
        - 시장이 어떤 성장률을 가정하고 있는지 역산
        """
        current_price = self.data.get('current_price', 0)
        shares = self.data.get('shares_outstanding', 0)

        if current_price <= 0 or shares <= 0:
            return {'error': 'Missing price or shares data'}

        # 현재 시가총액에서 implied EV 계산
        market_cap = current_price * shares
        cash = self.data.get('cash', 0)
        debt = self.data.get('total_debt', 0)
        implied_ev = market_cap - cash + debt

        # Binary search로 implied growth 찾기
        low_growth = 0.0
        high_growth = 1.0  # 100%
        tolerance = 0.001  # 0.1%p

        implied_growth = None

        for _ in range(50):  # 최대 50번 반복
            mid_growth = (low_growth + high_growth) / 2

            # 이 성장률로 DCF 계산
            growth_schedule = self.generate_growth_schedule(
                initial_growth=mid_growth,
                terminal_growth=base_assumptions.get('terminal_growth', 0.025),
                years=years,
                decay_type='linear'
            )

            test_assumptions = DCFAssumptions(
                revenue_growth_rates=growth_schedule,
                ebitda_margin=base_assumptions['ebitda_margin'],
                da_pct=base_assumptions['da_pct'],
                capex_pct=base_assumptions['capex_pct'],
                nwc_pct=base_assumptions['nwc_pct'],
                tax_rate=tax_rate,
                terminal_growth=base_assumptions.get('terminal_growth', 0.025),
                exit_multiple=base_assumptions['exit_multiple'],
                wacc=wacc,
            )

            result = self.calculate_dcf(test_assumptions, years)
            calculated_ev = result['valuations'].get('exit_multiple', {}).get('enterprise_value', 0)

            if calculated_ev <= 0:
                low_growth = mid_growth
                continue

            diff_pct = (calculated_ev - implied_ev) / implied_ev

            if abs(diff_pct) < tolerance:
                implied_growth = mid_growth
                break
            elif calculated_ev < implied_ev:
                low_growth = mid_growth
            else:
                high_growth = mid_growth

        # 결과 평가
        if implied_growth is not None:
            # 실현 가능성 평가
            cagr_3y = self.calculate_cagr(
                self.historical['revenue'].dropna().tolist() if 'revenue' in self.historical.columns else [self.data.get('revenue', 0)],
                3
            )

            if implied_growth > 0.50:
                feasibility = "매우 공격적 (50%+ 성장 필요)"
                rating = "High Risk"
            elif implied_growth > cagr_3y * 1.5:
                feasibility = f"공격적 (과거 3Y CAGR {cagr_3y*100:.1f}%의 1.5배 이상)"
                rating = "Aggressive"
            elif implied_growth > cagr_3y:
                feasibility = f"약간 공격적 (과거 3Y CAGR {cagr_3y*100:.1f}% 초과)"
                rating = "Slightly Aggressive"
            elif implied_growth > cagr_3y * 0.7:
                feasibility = "합리적 범위"
                rating = "Reasonable"
            else:
                feasibility = "보수적 (현재 주가 저평가 가능)"
                rating = "Conservative"

            return {
                'implied_growth': implied_growth,
                'implied_growth_pct': f"{implied_growth*100:.1f}%",
                'current_price': current_price,
                'implied_ev': implied_ev,
                'historical_cagr_3y': cagr_3y,
                'feasibility': feasibility,
                'rating': rating,
                'assumptions_used': {
                    'exit_multiple': base_assumptions['exit_multiple'],
                    'wacc': wacc,
                    'ebitda_margin': base_assumptions['ebitda_margin'],
                }
            }
        else:
            return {
                'error': 'Could not calculate implied growth',
                'message': '현재 주가를 정당화하려면 100% 이상의 성장률이 필요할 수 있음'
            }


def create_football_field_data(
    current_price: float,
    dcf_scenarios: dict,
    analyst_targets: Tuple[float, float, float] = None,
    week_52_range: Tuple[float, float] = None
) -> List[dict]:
    """Football Field Chart 데이터"""
    data = []
    
    if week_52_range and week_52_range[0] > 0:
        data.append({
            'category': '52-Week Range',
            'low': week_52_range[0],
            'high': week_52_range[1],
            'mid': (week_52_range[0] + week_52_range[1]) / 2
        })
    
    if analyst_targets and analyst_targets[0] > 0:
        data.append({
            'category': 'Analyst Targets',
            'low': analyst_targets[0],
            'high': analyst_targets[2],
            'mid': analyst_targets[1]
        })
    
    summary = dcf_scenarios.get('summary', {})
    
    if summary.get('bear_perpetuity', 0) > 0 and summary.get('bull_perpetuity', 0) > 0:
        data.append({
            'category': 'DCF (Perpetuity)',
            'low': summary['bear_perpetuity'],
            'high': summary['bull_perpetuity'],
            'mid': summary.get('base_perpetuity', 0)
        })
    
    if summary.get('bear_exit_val', 0) > 0 and summary.get('bull_exit_val', 0) > 0:
        data.append({
            'category': 'DCF (Exit Multiple)',
            'low': summary['bear_exit_val'],
            'high': summary['bull_exit_val'],
            'mid': summary.get('base_exit_val', 0)
        })
    
    return data
