"""
dcf_model.py - Wall Street Style DCF Model (v3)
- 3Y CAGR 기반 Base Growth
- Bull/Bear는 Revenue Growth + Exit Multiple만 조절
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# 섹터별 기본값 + 성장률 Cap
SECTOR_DEFAULTS = {
    'Technology': {
        'ebitda_margin': 0.30,
        'da_pct': 0.06,
        'capex_pct': 0.06,
        'nwc_pct': 0.05,
        'exit_multiple': 18,
        'growth_cap': 0.25,
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
    """월가 스타일 DCF 모델 (v3)"""
    
    def __init__(self, financial_data: dict):
        self.data = financial_data
        self.sector = financial_data.get('sector', 'Default')
        self.sector_defaults = SECTOR_DEFAULTS.get(self.sector, SECTOR_DEFAULTS['Default'])
        self.historical = self._extract_historical()
        
        self.actual_fcf = financial_data.get('fcf', 0)
        self.actual_fcf_margin = self.actual_fcf / financial_data.get('revenue', 1) if financial_data.get('revenue', 0) > 0 else 0
    
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
        과거 평균값 계산
        ★ 핵심: 3Y CAGR × 0.7 + 5Y CAGR × 0.3
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
            blended_growth = 0.05  # 기본값
        
        # 섹터 Cap 적용
        growth_cap = defaults.get('growth_cap', 0.15)
        base_growth = max(0, min(blended_growth, growth_cap))
        
        # EBITDA 마진
        if 'ebitda_margin' in df.columns:
            margin = df['ebitda_margin'].dropna()
            avg_margin = margin.mean() if len(margin) > 0 else defaults['ebitda_margin']
        else:
            revenue = self.data.get('revenue', 0)
            ebitda = self.data.get('ebitda', 0)
            if revenue > 0 and ebitda > 0:
                avg_margin = ebitda / revenue
            else:
                avg_margin = defaults['ebitda_margin']
        
        # D&A %
        ebitda = self.data.get('ebitda', 0)
        op_income = self.data.get('operating_income', 0)
        revenue = self.data.get('revenue', 1)
        if ebitda > 0 and op_income > 0 and revenue > 0:
            avg_da = (ebitda - op_income) / revenue
        else:
            avg_da = defaults['da_pct']
        
        # CapEx %
        op_cf = self.data.get('operating_cf', 0)
        fcf = self.data.get('fcf', 0)
        if op_cf > 0 and revenue > 0:
            avg_capex = (op_cf - fcf) / revenue if fcf > 0 else defaults['capex_pct']
        else:
            avg_capex = defaults['capex_pct']
        
        # NWC %
        if 'nwc' in df.columns and 'revenue' in df.columns:
            nwc_pct = (df['nwc'] / df['revenue']).dropna()
            avg_nwc = nwc_pct.mean() if len(nwc_pct) > 0 else defaults['nwc_pct']
        else:
            avg_nwc = defaults['nwc_pct']
        
        # 실제 FCF 마진
        actual_fcf_margin = self.actual_fcf / self.data.get('revenue', 1) if self.data.get('revenue', 0) > 0 else 0
        
        return {
            # ★ CAGR 정보
            'cagr_3y': cagr_3y,
            'cagr_5y': cagr_5y,
            'base_growth': base_growth,
            'growth_cap': growth_cap,
            
            # 기타 가정값
            'avg_ebitda_margin': min(max(avg_margin, 0.05), 0.70),
            'avg_da_pct': min(max(avg_da, 0.01), 0.15),
            'avg_capex_pct': min(max(avg_capex, 0.01), 0.20),
            'avg_nwc_pct': min(max(avg_nwc, -0.20), 0.30),
            'actual_fcf_margin': actual_fcf_margin,
            'sector_defaults': defaults,
        }
    
    def build_projections(self, assumptions: DCFAssumptions, years: int = 5) -> pd.DataFrame:
        """Revenue → UFCF Bottom-up 빌드"""
        projections = []
        
        base_revenue = self.data.get('revenue', 0)
        if base_revenue == 0:
            return pd.DataFrame()
        
        prev_nwc = base_revenue * assumptions.nwc_pct
        
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
            nopat = ebit * (1 - assumptions.tax_rate)
            capex = revenue * assumptions.capex_pct
            
            nwc = revenue * assumptions.nwc_pct
            delta_nwc = nwc - prev_nwc
            prev_nwc = nwc
            
            ufcf = nopat + da - capex - delta_nwc
            
            projections.append({
                'year': year,
                'revenue_growth': growth,
                'revenue': revenue,
                'ebitda': ebitda,
                'da': da,
                'ebit': ebit,
                'nopat': nopat,
                'capex': capex,
                'delta_nwc': delta_nwc,
                'ufcf': ufcf,
                'ufcf_margin': ufcf / revenue
            })
        
        return pd.DataFrame(projections)
    
    def sanity_check(self, projections: pd.DataFrame) -> dict:
        """FCF Sanity Check"""
        if projections.empty:
            return {'pass': False, 'message': 'No projections'}
        
        actual = self.actual_fcf_margin
        projected = projections.iloc[0]['ufcf_margin']
        diff = abs(projected - actual)
        
        return {
            'pass': diff <= 0.05,
            'message': f'Projected {projected*100:.1f}% vs Actual {actual*100:.1f}%',
            'actual': actual,
            'projected': projected,
            'diff': diff
        }
    
    def calculate_terminal_value(self, projections: pd.DataFrame, assumptions: DCFAssumptions) -> dict:
        """Terminal Value 계산"""
        if projections.empty:
            return {}
        
        final = projections.iloc[-1]
        final_ufcf = final['ufcf']
        final_ebitda = final['ebitda']
        final_year = final['year']
        
        result = {}
        
        # Perpetuity
        if assumptions.wacc > assumptions.terminal_growth:
            tv = final_ufcf * (1 + assumptions.terminal_growth) / (assumptions.wacc - assumptions.terminal_growth)
            pv_tv = tv / ((1 + assumptions.wacc) ** final_year)
            
            result['perpetuity'] = {
                'terminal_value': tv,
                'pv_terminal_value': pv_tv,
                'implied_multiple': tv / final_ebitda if final_ebitda > 0 else 0
            }
        
        # Exit Multiple
        tv = final_ebitda * assumptions.exit_multiple
        pv_tv = tv / ((1 + assumptions.wacc) ** final_year)
        
        result['exit_multiple'] = {
            'terminal_value': tv,
            'pv_terminal_value': pv_tv,
            'exit_multiple': assumptions.exit_multiple
        }
        
        return result
    
    def calculate_dcf(self, assumptions: DCFAssumptions, years: int = 5) -> dict:
        """전체 DCF 계산"""
        projections = self.build_projections(assumptions, years)
        
        if projections.empty:
            return {'error': 'No projections'}
        
        sanity = self.sanity_check(projections)
        
        projections['discount_factor'] = [1 / ((1 + assumptions.wacc) ** yr) for yr in projections['year']]
        projections['pv_ufcf'] = projections['ufcf'] * projections['discount_factor']
        
        sum_pv_fcf = projections['pv_ufcf'].sum()
        
        tv_results = self.calculate_terminal_value(projections, assumptions)
        
        results = {}
        
        for method in ['perpetuity', 'exit_multiple']:
            if method not in tv_results:
                continue
            
            tv = tv_results[method]
            ev = sum_pv_fcf + tv['pv_terminal_value']
            
            cash = self.data.get('cash', 0)
            debt = self.data.get('total_debt', 0)
            equity = ev + cash - debt
            
            shares = self.data.get('shares_outstanding', 0)
            per_share = equity / shares if shares > 0 else 0
            
            tv_pct = tv['pv_terminal_value'] / ev if ev > 0 else 0
            
            results[method] = {
                'sum_pv_fcf': sum_pv_fcf,
                'pv_terminal_value': tv['pv_terminal_value'],
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
            'current_price': self.data.get('current_price', 0)
        }
    
    def run_scenarios(
        self,
        base_assumptions: dict,
        wacc: float,
        tax_rate: float = 0.21,
        bull_growth_delta: float = 0.05,  # +5%p
        bear_growth_delta: float = 0.05,  # -5%p
        bull_multiple_delta: float = 2.0,  # +2x
        bear_multiple_delta: float = 2.0,  # -2x
    ) -> dict:
        """
        Bull / Base / Bear 시나리오
        ★ Revenue Growth와 Exit Multiple만 조절
        """
        scenarios = {}
        
        base_growth = base_assumptions['revenue_growth']
        base_exit = base_assumptions['exit_multiple']
        
        # ===== BASE =====
        base = DCFAssumptions(
            revenue_growth_rates=base_growth,
            ebitda_margin=base_assumptions['ebitda_margin'],
            da_pct=base_assumptions['da_pct'],
            capex_pct=base_assumptions['capex_pct'],
            nwc_pct=base_assumptions['nwc_pct'],
            tax_rate=tax_rate,
            terminal_growth=base_assumptions['terminal_growth'],
            exit_multiple=base_exit,
            wacc=wacc,
            scenario="Base"
        )
        scenarios['base'] = self.calculate_dcf(base)
        
        # ===== BULL =====
        # Revenue Growth만 +delta, Exit Multiple만 +delta
        bull_growth = [min(g + bull_growth_delta, 0.40) for g in base_growth]
        bull = DCFAssumptions(
            revenue_growth_rates=bull_growth,
            ebitda_margin=base_assumptions['ebitda_margin'],  # 동일
            da_pct=base_assumptions['da_pct'],  # 동일
            capex_pct=base_assumptions['capex_pct'],  # 동일
            nwc_pct=base_assumptions['nwc_pct'],  # 동일
            tax_rate=tax_rate,
            terminal_growth=base_assumptions['terminal_growth'],  # 동일
            exit_multiple=base_exit + bull_multiple_delta,
            wacc=wacc,  # 동일
            scenario="Bull"
        )
        scenarios['bull'] = self.calculate_dcf(bull)
        
        # ===== BEAR =====
        bear_growth = [max(g - bear_growth_delta, 0.0) for g in base_growth]
        bear = DCFAssumptions(
            revenue_growth_rates=bear_growth,
            ebitda_margin=base_assumptions['ebitda_margin'],  # 동일
            da_pct=base_assumptions['da_pct'],  # 동일
            capex_pct=base_assumptions['capex_pct'],  # 동일
            nwc_pct=base_assumptions['nwc_pct'],  # 동일
            tax_rate=tax_rate,
            terminal_growth=base_assumptions['terminal_growth'],  # 동일
            exit_multiple=max(base_exit - bear_multiple_delta, 4.0),
            wacc=wacc,  # 동일
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
            'bull_exit': base_exit + bull_multiple_delta,
            'bull_perpetuity': scenarios['bull']['valuations'].get('perpetuity', {}).get('per_share_value', 0),
            'bull_exit_val': scenarios['bull']['valuations'].get('exit_multiple', {}).get('per_share_value', 0),
            # Bear
            'bear_growth': bear_growth[0] if bear_growth else 0,
            'bear_exit': max(base_exit - bear_multiple_delta, 4.0),
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
