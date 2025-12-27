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


def get_interest_expense(stock) -> float:
    """
    이자비용(Interest Expense) 추출

    Returns:
        양수 값 (실제 비용)
    """
    try:
        income_stmt = stock.income_stmt
        if income_stmt is None or income_stmt.empty:
            return 0

        # TTM (가장 최근 컬럼)
        latest_col = income_stmt.columns[0]

        # 방법 1: Interest Expense 행
        if 'Interest Expense' in income_stmt.index:
            val = income_stmt.loc['Interest Expense', latest_col]
            if pd.notna(val):
                return abs(float(val))  # 양수로 변환

        # 방법 2: Interest Expense Non Operating (일부 기업)
        if 'Interest Expense Non Operating' in income_stmt.index:
            val = income_stmt.loc['Interest Expense Non Operating', latest_col]
            if pd.notna(val):
                return abs(float(val))

        # 방법 3: Net Interest Income (금융기관의 경우 음수)
        if 'Net Interest Income' in income_stmt.index:
            val = income_stmt.loc['Net Interest Income', latest_col]
            if pd.notna(val) and val < 0:
                return abs(float(val))

        return 0
    except Exception:
        return 0


def get_ebit(stock, info: dict) -> float:
    """
    EBIT (Operating Income) 추출

    Method 1: Income Statement에서 직접 추출
    Method 2: Revenue * Operating Margin
    Method 3: EBITDA * 0.8 (추정)

    Returns:
        EBIT 값 (양수)
    """
    try:
        income_stmt = stock.income_stmt
        if income_stmt is not None and not income_stmt.empty:
            latest_col = income_stmt.columns[0]

            # Method 1: Operating Income 행
            if 'Operating Income' in income_stmt.index:
                val = income_stmt.loc['Operating Income', latest_col]
                if pd.notna(val) and val > 0:
                    return float(val)

            # EBIT 행
            if 'EBIT' in income_stmt.index:
                val = income_stmt.loc['EBIT', latest_col]
                if pd.notna(val) and val > 0:
                    return float(val)

        # Method 2: Revenue * Operating Margin
        revenue = info.get('totalRevenue', 0)
        op_margin = info.get('operatingMargins', 0)
        if revenue > 0 and op_margin > 0:
            return revenue * op_margin

        # Method 3: EBITDA * 0.8 (보수적 추정)
        ebitda = info.get('ebitda', 0)
        if ebitda > 0:
            return ebitda * 0.80

        return 0
    except Exception:
        return 0


def get_tax_rate(stock, info: dict) -> float:
    """
    실효 세율(Effective Tax Rate) 계산

    Method 1: Tax Provision / Pretax Income
    Method 2: 법인세율 (default 21%)

    Returns:
        세율 (0.0 ~ 1.0)
    """
    try:
        income_stmt = stock.income_stmt
        if income_stmt is None or income_stmt.empty:
            return 0.21  # Default US corporate tax rate

        latest_col = income_stmt.columns[0]

        # Tax Provision 추출
        tax_provision = 0
        if 'Tax Provision' in income_stmt.index:
            val = income_stmt.loc['Tax Provision', latest_col]
            if pd.notna(val):
                tax_provision = float(val)

        # Pretax Income 추출
        pretax_income = 0
        if 'Pretax Income' in income_stmt.index:
            val = income_stmt.loc['Pretax Income', latest_col]
            if pd.notna(val):
                pretax_income = float(val)

        # 실효 세율 계산
        if pretax_income > 0 and tax_provision > 0:
            effective_rate = tax_provision / pretax_income

            # Sanity check: 0% ~ 40% 범위
            if 0 <= effective_rate <= 0.40:
                return effective_rate

        # Fallback: 법인세율
        return 0.21
    except Exception:
        return 0.21


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
            'ebit': get_ebit(stock, info),  # Operating Income = EBIT
            'net_income': info.get('netIncomeToCommon', 0),
            'eps': info.get('trailingEps', 0) or 0,
            'forward_eps': info.get('forwardEps', 0) or 0,

            # WACC 계산용 데이터
            'interest_expense': get_interest_expense(stock),
            'tax_rate': get_tax_rate(stock, info),
            
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
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', 0) or 0,
            
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


def get_peer_group_data(peer_tickers: list, max_workers: int = 5) -> list:
    """
    Peer Group 데이터 병렬 수집

    Args:
        peer_tickers: Peer 티커 리스트
        max_workers: 동시 요청 수 (기본 5)

    Returns:
        [{ticker, name, price, market_cap, pe_ratio, forward_pe, pb_ratio, ev_ebitda, ...}, ...]
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def fetch_peer(ticker: str) -> dict:
        """단일 Peer 데이터 수집"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or info.get('regularMarketPrice') is None:
                return {'ticker': ticker, 'error': True, 'error_msg': 'No data'}

            return {
                'ticker': ticker.upper(),
                'name': info.get('shortName', ticker),
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'market_cap': info.get('marketCap', 0),

                # Valuation Multiples
                'pe_ratio': info.get('trailingPE', 0) or 0,
                'forward_pe': info.get('forwardPE', 0) or 0,
                'pb_ratio': info.get('priceToBook', 0) or 0,
                'ps_ratio': info.get('priceToSalesTrailing12Months', 0) or 0,
                'ev_ebitda': info.get('enterpriseToEbitda', 0) or 0,
                'ev_revenue': info.get('enterpriseToRevenue', 0) or 0,

                # Growth & Margins
                'revenue_growth': info.get('revenueGrowth', 0) or 0,
                'earnings_growth': info.get('earningsGrowth', 0) or 0,
                'profit_margin': info.get('profitMargins', 0) or 0,
                'operating_margin': info.get('operatingMargins', 0) or 0,

                # Other
                'beta': info.get('beta', 1.0) or 1.0,
                'dividend_yield': info.get('dividendYield', 0) or 0,

                'error': False
            }
        except Exception as e:
            return {'ticker': ticker, 'error': True, 'error_msg': str(e)}

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(fetch_peer, ticker): ticker for ticker in peer_tickers}

        for future in as_completed(future_to_ticker):
            result = future.result()
            if not result.get('error'):
                results.append(result)

    # 시가총액 순 정렬
    results.sort(key=lambda x: x.get('market_cap', 0), reverse=True)

    return results


def calculate_peer_relative_valuation(target_data: dict, peer_data: list) -> dict:
    """
    Peer 대비 상대가치 분석

    Args:
        target_data: 타겟 기업 데이터 (get_stock_data 결과)
        peer_data: Peer 그룹 데이터 (get_peer_group_data 결과)

    Returns:
        {
            'peer_avg': {pe, pb, ev_ebitda, ...},
            'peer_median': {pe, pb, ev_ebitda, ...},
            'implied_values': {pe_based, pb_based, ev_ebitda_based, ...},
            'premium_discount': {pe, pb, ev_ebitda, ...}  # 양수=프리미엄, 음수=디스카운트
        }
    """
    if not peer_data:
        return {'error': 'No peer data'}

    # Peer 평균/중앙값 계산
    def calc_stats(key):
        values = [p[key] for p in peer_data if p.get(key, 0) > 0]
        if not values:
            return {'avg': 0, 'median': 0}
        return {
            'avg': sum(values) / len(values),
            'median': sorted(values)[len(values) // 2]
        }

    pe_stats = calc_stats('pe_ratio')
    forward_pe_stats = calc_stats('forward_pe')
    pb_stats = calc_stats('pb_ratio')
    ev_ebitda_stats = calc_stats('ev_ebitda')
    ps_stats = calc_stats('ps_ratio')

    # 타겟 기업 데이터
    target_price = target_data.get('current_price', 0)
    target_eps = target_data.get('eps', 0)

    # BVPS: total_equity 기반 또는 Price/PB ratio 역산
    if target_data.get('total_equity', 0) > 0 and target_data.get('shares_outstanding', 0) > 0:
        target_bvps = target_data.get('total_equity', 0) / target_data.get('shares_outstanding', 1)
    elif target_price > 0 and target_data.get('pb_ratio', 0) > 0:
        target_bvps = target_price / target_data.get('pb_ratio')
    else:
        target_bvps = 0

    target_revenue_ps = target_data.get('revenue', 0) / target_data.get('shares_outstanding', 1) if target_data.get('shares_outstanding', 0) > 0 else 0

    # Implied Value 계산
    implied = {}
    if target_eps > 0 and pe_stats['avg'] > 0:
        implied['pe_based'] = target_eps * pe_stats['avg']
    if target_eps > 0 and forward_pe_stats['avg'] > 0:
        implied['forward_pe_based'] = target_eps * forward_pe_stats['avg']
    if target_bvps > 0 and pb_stats['avg'] > 0:
        implied['pb_based'] = target_bvps * pb_stats['avg']
    if target_revenue_ps > 0 and ps_stats['avg'] > 0:
        implied['ps_based'] = target_revenue_ps * ps_stats['avg']

    # Premium/Discount 계산
    premium = {}
    target_pe = target_data.get('pe_ratio', 0)
    target_pb = target_data.get('pb_ratio', 0)
    target_ev_ebitda = target_data.get('ev_ebitda', 0)

    if target_pe > 0 and pe_stats['avg'] > 0:
        premium['pe'] = (target_pe / pe_stats['avg'] - 1) * 100
    if target_pb > 0 and pb_stats['avg'] > 0:
        premium['pb'] = (target_pb / pb_stats['avg'] - 1) * 100
    if target_ev_ebitda > 0 and ev_ebitda_stats['avg'] > 0:
        premium['ev_ebitda'] = (target_ev_ebitda / ev_ebitda_stats['avg'] - 1) * 100

    return {
        'peer_count': len(peer_data),
        'peer_avg': {
            'pe': pe_stats['avg'],
            'forward_pe': forward_pe_stats['avg'],
            'pb': pb_stats['avg'],
            'ev_ebitda': ev_ebitda_stats['avg'],
            'ps': ps_stats['avg']
        },
        'peer_median': {
            'pe': pe_stats['median'],
            'forward_pe': forward_pe_stats['median'],
            'pb': pb_stats['median'],
            'ev_ebitda': ev_ebitda_stats['median'],
            'ps': ps_stats['median']
        },
        'implied_values': implied,
        'premium_discount': premium,
        'current_price': target_price
    }
