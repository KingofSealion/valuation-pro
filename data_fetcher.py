"""
data_fetcher.py - 미국 주식 데이터 수집 모듈 (개선판)
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_ttm_fcf(stock) -> float:
    """
    Quarterly 데이터를 사용해 TTM (Trailing Twelve Months) FCF 계산
    yfinance info.freeCashflow는 종종 outdated되어 있어서 직접 계산
    """
    try:
        qcf = stock.quarterly_cashflow
        if qcf is None or qcf.empty:
            # quarterly 데이터가 없으면 info에서 가져오기
            return stock.info.get('freeCashflow', 0) or 0

        # Free Cash Flow 행이 있으면 직접 사용
        if 'Free Cash Flow' in qcf.index:
            fcf_row = qcf.loc['Free Cash Flow'].head(4)
            ttm_fcf = fcf_row.sum()
            return float(ttm_fcf) if pd.notna(ttm_fcf) else 0

        # 없으면 Operating Cash Flow - Capital Expenditure로 계산
        if 'Operating Cash Flow' in qcf.index and 'Capital Expenditure' in qcf.index:
            op_cf = qcf.loc['Operating Cash Flow'].head(4).sum()
            capex = qcf.loc['Capital Expenditure'].head(4).sum()  # 이미 음수
            ttm_fcf = op_cf + capex
            return float(ttm_fcf) if pd.notna(ttm_fcf) else 0

        # 둘 다 없으면 info에서 가져오기
        return stock.info.get('freeCashflow', 0) or 0
    except Exception:
        return stock.info.get('freeCashflow', 0) or 0


def get_stock_data(ticker: str) -> tuple:
    """
    미국 주식 종합 데이터 수집
    
    Returns:
        (data_dict, success_bool)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info or info.get('regularMarketPrice') is None:
            return {'error': f'No data found for {ticker}'}, False
        
        # 재무제표 가져오기
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # 기본 정보
        data = {
            'ticker': ticker.upper(),
            'name': info.get('longName', info.get('shortName', ticker)),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            
            # 가격 정보
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
            'market_cap': info.get('marketCap', 0),
            'enterprise_value': info.get('enterpriseValue', 0),
            'shares_outstanding': info.get('sharesOutstanding', 0),
            
            # 52주 정보
            '52w_high': info.get('fiftyTwoWeekHigh', 0),
            '52w_low': info.get('fiftyTwoWeekLow', 0),
            
            # 손익계산서 (TTM)
            'revenue': info.get('totalRevenue', 0),
            'gross_profit': info.get('grossProfits', 0),
            'operating_income': info.get('operatingIncome', 0),
            'ebitda': info.get('ebitda', 0),
            'net_income': info.get('netIncomeToCommon', 0),
            'eps': info.get('trailingEps', 0) or 0,
            'forward_eps': info.get('forwardEps', 0) or 0,
            
            # 재무상태표
            'total_assets': info.get('totalAssets', 0),
            'total_equity': info.get('totalStockholderEquity', 0),
            'total_debt': info.get('totalDebt', 0),
            'net_debt': info.get('netDebt', 0),
            'cash': info.get('totalCash', 0),
            'current_assets': info.get('totalCurrentAssets', 0),
            'current_liabilities': info.get('totalCurrentLiabilities', 0),
            
            # 현금흐름 (TTM은 quarterly 데이터로 계산)
            'operating_cf': info.get('operatingCashflow', 0),
            'fcf': calculate_ttm_fcf(stock),  # quarterly 기반 TTM FCF
            
            # 비율
            'gross_margin': info.get('grossMargins', 0) or 0,
            'operating_margin': info.get('operatingMargins', 0) or 0,
            'profit_margin': info.get('profitMargins', 0) or 0,
            'roe': info.get('returnOnEquity', 0) or 0,
            'roa': info.get('returnOnAssets', 0) or 0,
            
            # 성장률
            'revenue_growth': info.get('revenueGrowth', 0) or 0,
            'earnings_growth': info.get('earningsGrowth', 0) or 0,
            
            # 밸류에이션
            'pe_ratio': info.get('trailingPE', 0) or 0,
            'forward_pe': info.get('forwardPE', 0) or 0,
            'pb_ratio': info.get('priceToBook', 0) or 0,
            'ev_ebitda': info.get('enterpriseToEbitda', 0) or 0,
            
            # 베타
            'beta': info.get('beta', 1.0) or 1.0,
            
            # 배당
            'dividend_rate': info.get('dividendRate', 0) or 0,
            'dividend_yield': info.get('dividendYield', 0) or 0,
            
            # 애널리스트
            'target_high': info.get('targetHighPrice', 0) or 0,
            'target_low': info.get('targetLowPrice', 0) or 0,
            'target_mean': info.get('targetMeanPrice', 0) or 0,
            'num_analysts': info.get('numberOfAnalystOpinions', 0) or 0,
        }
        
        # 과거 재무 데이터 추출
        data['historical_financials'] = extract_historical(income_stmt, balance_sheet, cash_flow)
        
        # 주가 히스토리
        try:
            data['price_history'] = stock.history(period='2y')
        except:
            data['price_history'] = pd.DataFrame()
        
        return data, True
        
    except Exception as e:
        return {'error': str(e)}, False


def extract_historical(income_stmt, balance_sheet, cash_flow) -> list:
    """과거 5년 재무 데이터 추출"""
    historical = []
    
    if income_stmt is None or income_stmt.empty:
        return historical
    
    # 최대 5년치
    for col in income_stmt.columns[:5]:
        year = col.year if hasattr(col, 'year') else str(col)[:4]
        
        row = {'year': year}
        
        # 손익계산서
        row['revenue'] = safe_get(income_stmt, 'Total Revenue', col)
        row['gross_profit'] = safe_get(income_stmt, 'Gross Profit', col)
        row['operating_income'] = safe_get(income_stmt, 'Operating Income', col)
        row['net_income'] = safe_get(income_stmt, 'Net Income', col)
        row['ebitda'] = safe_get(income_stmt, 'EBITDA', col)
        
        # EBITDA가 없으면 계산
        if row['ebitda'] == 0 and row['operating_income'] > 0:
            depreciation = safe_get(cash_flow, 'Depreciation And Amortization', col) if cash_flow is not None else 0
            row['ebitda'] = row['operating_income'] + depreciation
        
        # 재무상태표
        if balance_sheet is not None and not balance_sheet.empty:
            row['total_assets'] = safe_get(balance_sheet, 'Total Assets', col)
            row['total_equity'] = safe_get(balance_sheet, 'Stockholders Equity', col)
            if row['total_equity'] == 0:
                row['total_equity'] = safe_get(balance_sheet, 'Total Equity Gross Minority Interest', col)
            row['total_debt'] = safe_get(balance_sheet, 'Total Debt', col)
            row['cash'] = safe_get(balance_sheet, 'Cash And Cash Equivalents', col)
            row['current_assets'] = safe_get(balance_sheet, 'Current Assets', col)
            row['current_liabilities'] = safe_get(balance_sheet, 'Current Liabilities', col)
            row['inventory'] = safe_get(balance_sheet, 'Inventory', col)
            row['receivables'] = safe_get(balance_sheet, 'Receivables', col)
        
        # 현금흐름표
        if cash_flow is not None and not cash_flow.empty:
            row['operating_cf'] = safe_get(cash_flow, 'Operating Cash Flow', col)
            row['capex'] = abs(safe_get(cash_flow, 'Capital Expenditure', col))
            row['depreciation'] = safe_get(cash_flow, 'Depreciation And Amortization', col)
            row['fcf'] = row['operating_cf'] - row['capex'] if row['operating_cf'] > 0 else 0
        
        # 마진 계산
        if row['revenue'] > 0:
            row['gross_margin'] = row['gross_profit'] / row['revenue']
            row['operating_margin'] = row['operating_income'] / row['revenue']
            row['ebitda_margin'] = row['ebitda'] / row['revenue']
            row['net_margin'] = row['net_income'] / row['revenue']
            row['da_pct'] = row.get('depreciation', 0) / row['revenue']
            row['capex_pct'] = row.get('capex', 0) / row['revenue']
        
        historical.append(row)
    
    return historical


def safe_get(df, row_name, col):
    """DataFrame에서 안전하게 값 추출"""
    try:
        # 여러 가능한 행 이름 시도
        possible_names = [row_name]
        
        # 변형 이름들 추가
        if 'Cash Flow' in row_name:
            possible_names.append(row_name.replace('Cash Flow', 'CashFlow'))
        if ' And ' in row_name:
            possible_names.append(row_name.replace(' And ', ' & '))
        
        for name in possible_names:
            if name in df.index:
                val = df.loc[name, col]
                return float(val) if pd.notna(val) else 0
        
        return 0
    except:
        return 0


def get_risk_free_rate() -> float:
    """미국 10년 국채 금리"""
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period='5d')
        if not hist.empty:
            return hist['Close'].iloc[-1] / 100
        return 0.045
    except:
        return 0.045


def get_market_data() -> dict:
    """시장 데이터 (S&P500 등)"""
    try:
        spy = yf.Ticker("SPY")
        info = spy.info
        
        return {
            'sp500_pe': info.get('trailingPE', 22),
            'market_return': 0.10,  # 역사적 평균
        }
    except:
        return {'sp500_pe': 22, 'market_return': 0.10}


def get_sector_ev_ebitda(sector: str) -> float:
    """섹터별 평균 EV/EBITDA 배수"""
    sector_multiples = {
        'Technology': 18,
        'Healthcare': 14,
        'Financials': 10,
        'Consumer Cyclical': 12,
        'Consumer Defensive': 14,
        'Communication Services': 10,
        'Industrials': 11,
        'Energy': 6,
        'Utilities': 12,
        'Real Estate': 18,
        'Materials': 8,
    }
    return sector_multiples.get(sector, 12)


def get_peers(sector: str, exclude: str) -> list:
    """동종 업계 Peer 리스트"""
    sector_tickers = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'ADBE', 'AMD'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY'],
        'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'SPGI'],
        'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'LOW', 'TJX', 'BKNG'],
        'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MDLZ', 'CL', 'EL'],
        'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS'],
        'Industrials': ['UNP', 'HON', 'UPS', 'BA', 'CAT', 'GE', 'RTX', 'DE', 'LMT'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY'],
    }
    
    peers = sector_tickers.get(sector, [])
    return [p for p in peers if p.upper() != exclude.upper()][:8]
