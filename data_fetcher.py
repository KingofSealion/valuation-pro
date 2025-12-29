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


def get_minority_interest(balance_sheet) -> float:
    """
    소수지분(Minority Interest) 추출

    EV에서 Equity로 변환 시 차감해야 하는 항목

    Returns:
        양수 값
    """
    try:
        if balance_sheet is None or balance_sheet.empty:
            return 0

        latest_col = balance_sheet.columns[0]

        # 가능한 행 이름들
        possible_names = [
            'Minority Interest',
            'MinorityInterest',
            'Non Controlling Interest',
            'Noncontrolling Interest'
        ]

        for name in possible_names:
            if name in balance_sheet.index:
                val = balance_sheet.loc[name, latest_col]
                if pd.notna(val):
                    return abs(float(val))

        return 0
    except Exception:
        return 0


def get_preferred_stock(balance_sheet) -> float:
    """
    우선주(Preferred Stock) 추출

    EV에서 Equity로 변환 시 차감해야 하는 항목

    Returns:
        양수 값
    """
    try:
        if balance_sheet is None or balance_sheet.empty:
            return 0

        latest_col = balance_sheet.columns[0]

        # 가능한 행 이름들
        possible_names = [
            'Preferred Stock',
            'PreferredStock',
            'Preferred Securities',
            'Redeemable Preferred Stock'
        ]

        for name in possible_names:
            if name in balance_sheet.index:
                val = balance_sheet.loc[name, latest_col]
                if pd.notna(val):
                    return abs(float(val))

        return 0
    except Exception:
        return 0


def get_total_equity(info: dict, balance_sheet) -> float:
    """
    총 자본(Total Stockholders Equity) 추출

    info.get('totalStockholderEquity')가 0 또는 None인 경우,
    Balance Sheet에서 직접 가져온다.

    이슈: ADBE 등 일부 종목에서 info에서 0이 반환되는 버그 존재

    Returns:
        Total Equity 값
    """
    # 방법 1: info에서 가져오기
    equity_from_info = info.get('totalStockholderEquity', 0) or 0

    if equity_from_info > 0:
        return equity_from_info

    # 방법 2: Balance Sheet에서 직접 추출
    try:
        if balance_sheet is None or balance_sheet.empty:
            return 0

        latest_col = balance_sheet.columns[0]

        # 가능한 행 이름들 (우선순위)
        possible_names = [
            'Stockholders Equity',
            'Total Equity Gross Minority Interest',
            'Common Stock Equity',
            'Total Stockholders Equity',
            'Stockholder Equity'
        ]

        for name in possible_names:
            if name in balance_sheet.index:
                val = balance_sheet.loc[name, latest_col]
                if pd.notna(val) and val > 0:
                    return float(val)

        return 0
    except Exception:
        return 0


def get_cash_only(balance_sheet) -> float:
    """
    현금 및 현금성 자산만 추출 (단기투자 제외)

    ROIC 계산 시 Invested Capital에서 차감할 현금은
    'Cash And Cash Equivalents'만 사용해야 정확함.
    yfinance의 info.totalCash는 단기투자까지 포함하여 과대계상됨.

    Returns:
        Cash and Cash Equivalents 값
    """
    try:
        if balance_sheet is None or balance_sheet.empty:
            return 0

        latest_col = balance_sheet.columns[0]

        # 가능한 행 이름들
        possible_names = [
            'Cash And Cash Equivalents',
            'Cash',
            'CashAndCashEquivalents'
        ]

        for name in possible_names:
            if name in balance_sheet.index:
                val = balance_sheet.loc[name, latest_col]
                if pd.notna(val):
                    return float(val)

        return 0
    except Exception:
        return 0


def get_operating_cashflow(info: dict, cash_flow) -> float:
    """
    영업활동현금흐름(Operating Cash Flow) 추출

    info.get('operatingCashflow')가 0 또는 None인 경우,
    Cashflow Statement에서 직접 가져온다.

    이슈: ADBE 등 일부 종목에서 info에서 None 반환되는 버그 존재

    Returns:
        Operating Cash Flow 값
    """
    # 방법 1: info에서 가져오기
    ocf_from_info = info.get('operatingCashflow', 0) or 0

    if ocf_from_info != 0:
        return ocf_from_info

    # 방법 2: Cashflow Statement에서 직접 추출
    try:
        if cash_flow is None or cash_flow.empty:
            return 0

        latest_col = cash_flow.columns[0]

        # 가능한 행 이름들
        possible_names = [
            'Operating Cash Flow',
            'Cash Flow From Operating Activities',
            'Net Cash Provided By Operating Activities'
        ]

        for name in possible_names:
            if name in cash_flow.index:
                val = cash_flow.loc[name, latest_col]
                if pd.notna(val):
                    return float(val)

        return 0
    except Exception:
        return 0


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
            'total_equity': get_total_equity(info, balance_sheet),  # Balance Sheet fallback
            'total_debt': info.get('totalDebt', 0),
            'net_debt': info.get('netDebt', 0),
            'cash': info.get('totalCash', 0),
            'current_assets': info.get('totalCurrentAssets', 0),
            'current_liabilities': info.get('totalCurrentLiabilities', 0),

            # EV → Equity 변환용 (Minority Interest, Preferred Stock)
            'minority_interest': get_minority_interest(balance_sheet),
            'preferred_stock': get_preferred_stock(balance_sheet),
            
            # 현금흐름 (TTM은 quarterly 데이터로 계산)
            'operating_cf': get_operating_cashflow(info, cash_flow),  # Cashflow fallback
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

            # EPS Growth 계산 (Forward EPS - Trailing EPS)
            trailing_eps = info.get('trailingEps', 0) or 0
            forward_eps = info.get('forwardEps', 0) or 0
            eps_growth = 0
            if trailing_eps > 0 and forward_eps > 0:
                eps_growth = (forward_eps - trailing_eps) / trailing_eps

            # PEG Ratio 계산
            pe_ratio = info.get('trailingPE', 0) or 0
            peg_ratio = 0
            if pe_ratio > 0 and eps_growth > 0:
                peg_ratio = pe_ratio / (eps_growth * 100)

            return {
                'ticker': ticker.upper(),
                'name': info.get('shortName', ticker),
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'market_cap': info.get('marketCap', 0),

                # Valuation Multiples
                'pe_ratio': pe_ratio,
                'forward_pe': info.get('forwardPE', 0) or 0,
                'pb_ratio': info.get('priceToBook', 0) or 0,
                'ps_ratio': info.get('priceToSalesTrailing12Months', 0) or 0,
                'ev_ebitda': info.get('enterpriseToEbitda', 0) or 0,
                'ev_revenue': info.get('enterpriseToRevenue', 0) or 0,

                # EPS & PEG (Relative Valuation용)
                'eps': trailing_eps,
                'forward_eps': forward_eps,
                'eps_growth': eps_growth,
                'peg_ratio': peg_ratio,

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


def get_historical_valuation(ticker: str, years: int = 5) -> dict:
    """
    Historical PE/PB Band 계산 (5년 기준)

    Returns:
        {
            'pe': {'current', 'avg', 'high', 'low', 'percentile', 'history': [...]},
            'pb': {'current', 'avg', 'high', 'low', 'percentile', 'history': [...]},
            'forward_pe': {...},
            'current_price': float,
            'ttm_eps': float,
            'bvps': float
        }
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # 현재 가격 및 EPS
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        ttm_eps = info.get('trailingEps', 0) or 0
        forward_eps = info.get('forwardEps', 0) or 0
        book_value = info.get('bookValue', 0) or 0  # BVPS

        # 현재 멀티플
        current_pe = current_price / ttm_eps if ttm_eps > 0 else 0
        current_forward_pe = current_price / forward_eps if forward_eps > 0 else 0
        current_pb = current_price / book_value if book_value > 0 else 0

        # 과거 주가 가져오기
        hist = stock.history(period=f'{years}y')
        if hist.empty:
            return {'error': 'No price history'}

        # 연간 EPS 가져오기
        income_stmt = stock.income_stmt
        if income_stmt is None or income_stmt.empty:
            return {'error': 'No income statement'}

        # EPS 히스토리 구축 (회계연도 종료일 기준)
        # [(fiscal_year_end_date, eps_value), ...] 형태로 저장 후 정렬
        eps_list = []
        eps_row_name = 'Basic EPS' if 'Basic EPS' in income_stmt.index else 'Diluted EPS'
        if eps_row_name in income_stmt.index:
            for col in income_stmt.columns:
                eps_val = income_stmt.loc[eps_row_name, col]
                if pd.notna(eps_val) and eps_val > 0:
                    # col은 회계연도 종료일 (예: 2024-01-31)
                    eps_list.append((col, float(eps_val)))
        # 날짜 오름차순 정렬
        eps_list.sort(key=lambda x: x[0])

        # Book Value 히스토리 (Balance Sheet) - 회계연도 종료일 기준
        balance_sheet = stock.balance_sheet
        bvps_list = []
        shares = info.get('sharesOutstanding', 0)

        if balance_sheet is not None and not balance_sheet.empty and shares > 0:
            for col in balance_sheet.columns:
                equity = None
                for name in ['Stockholders Equity', 'Total Equity Gross Minority Interest', 'Common Stock Equity']:
                    if name in balance_sheet.index:
                        val = balance_sheet.loc[name, col]
                        if pd.notna(val):
                            equity = float(val)
                            break
                if equity:
                    bvps_list.append((col, equity / shares))
        bvps_list.sort(key=lambda x: x[0])

        # 주어진 날짜에 해당하는 EPS 찾기 (해당 날짜 이전의 가장 최근 회계연도 EPS)
        def find_eps_for_date(price_date, eps_list):
            """주가 날짜에 해당하는 EPS 반환 (회계연도 종료일 이후의 EPS 사용)"""
            result_eps = None
            for fy_end_date, eps_val in eps_list:
                # 회계연도 종료일이 주가 날짜보다 이전이면 해당 EPS 사용 가능
                if fy_end_date.tz_localize(None) <= price_date.tz_localize(None):
                    result_eps = eps_val
                else:
                    break
            return result_eps

        def find_bvps_for_date(price_date, bvps_list):
            """주가 날짜에 해당하는 BVPS 반환"""
            result_bvps = None
            for fy_end_date, bvps_val in bvps_list:
                if fy_end_date.tz_localize(None) <= price_date.tz_localize(None):
                    result_bvps = bvps_val
                else:
                    break
            return result_bvps

        # PE 히스토리 계산 (일별)
        pe_history = []
        pb_history = []

        # 일별 데이터로 계산
        for date_idx, row in hist.iterrows():
            close_price = row['Close']
            date_str = date_idx.strftime('%Y-%m-%d')

            # 해당 날짜에 적용할 EPS 찾기 (회계연도 종료일 기준)
            eps_for_period = find_eps_for_date(date_idx, eps_list)

            if eps_for_period and eps_for_period > 0:
                pe = close_price / eps_for_period
                if 0 < pe < 500:  # 이상치 제거 (상한 완화)
                    pe_history.append({
                        'date': date_str,
                        'price': float(close_price),
                        'eps': eps_for_period,
                        'pe': pe
                    })

            # PB 계산
            bvps_for_period = find_bvps_for_date(date_idx, bvps_list)
            if bvps_for_period and bvps_for_period > 0:
                pb = close_price / bvps_for_period
                if 0 < pb < 100:
                    pb_history.append({
                        'date': date_str,
                        'price': float(close_price),
                        'bvps': bvps_for_period,
                        'pb': pb
                    })

        # PE 통계
        if pe_history:
            pe_values = [p['pe'] for p in pe_history]
            pe_avg = sum(pe_values) / len(pe_values)
            pe_high = max(pe_values)
            pe_low = min(pe_values)

            # Percentile 계산 (현재 PE가 과거 대비 몇 %ile인지)
            if current_pe > 0:
                below_count = sum(1 for p in pe_values if p < current_pe)
                pe_percentile = (below_count / len(pe_values)) * 100
            else:
                pe_percentile = 50

            pe_result = {
                'current': current_pe,
                'avg': pe_avg,
                'high': pe_high,
                'low': pe_low,
                'percentile': pe_percentile,
                'vs_avg_pct': ((current_pe / pe_avg) - 1) * 100 if pe_avg > 0 else 0,
                'history': pe_history
            }
        else:
            pe_result = {'current': current_pe, 'avg': 0, 'high': 0, 'low': 0, 'percentile': 50, 'vs_avg_pct': 0, 'history': []}

        # PB 통계
        if pb_history:
            pb_values = [p['pb'] for p in pb_history]
            pb_avg = sum(pb_values) / len(pb_values)
            pb_high = max(pb_values)
            pb_low = min(pb_values)

            if current_pb > 0:
                below_count = sum(1 for p in pb_values if p < current_pb)
                pb_percentile = (below_count / len(pb_values)) * 100
            else:
                pb_percentile = 50

            pb_result = {
                'current': current_pb,
                'avg': pb_avg,
                'high': pb_high,
                'low': pb_low,
                'percentile': pb_percentile,
                'vs_avg_pct': ((current_pb / pb_avg) - 1) * 100 if pb_avg > 0 else 0,
                'history': pb_history
            }
        else:
            pb_result = {'current': current_pb, 'avg': 0, 'high': 0, 'low': 0, 'percentile': 50, 'vs_avg_pct': 0, 'history': []}

        # Forward PE (현재만)
        forward_pe_result = {
            'current': current_forward_pe,
            'vs_trailing': ((current_forward_pe / current_pe) - 1) * 100 if current_pe > 0 and current_forward_pe > 0 else 0
        }

        return {
            'pe': pe_result,
            'pb': pb_result,
            'forward_pe': forward_pe_result,
            'current_price': current_price,
            'ttm_eps': ttm_eps,
            'forward_eps': forward_eps,
            'bvps': book_value,
            'data_points': len(pe_history)
        }

    except Exception as e:
        return {'error': str(e)}


def get_earnings_history(ticker: str) -> list:
    """
    Earnings Surprise 히스토리 가져오기 (최근 4분기)

    yfinance의 stock.earnings 또는 stock.quarterly_earnings에서 추출

    Returns:
        [
            {'quarter': '2024Q3', 'actual': 1.52, 'estimate': 1.48, 'surprise_pct': 2.7},
            {'quarter': '2024Q2', 'actual': 1.40, 'estimate': 1.42, 'surprise_pct': -1.4},
            ...
        ]
    """
    try:
        stock = yf.Ticker(ticker)

        # 방법 1: quarterly_earnings 사용
        quarterly = stock.quarterly_earnings
        if quarterly is not None and not quarterly.empty:
            results = []
            for idx in quarterly.index[:4]:  # 최근 4분기
                row = quarterly.loc[idx]

                # 컬럼 이름이 다를 수 있음
                actual = None
                estimate = None

                if 'Reported EPS' in row.index:
                    actual = float(row['Reported EPS']) if pd.notna(row['Reported EPS']) else None
                elif 'Actual' in row.index:
                    actual = float(row['Actual']) if pd.notna(row['Actual']) else None

                if 'EPS Estimate' in row.index:
                    estimate = float(row['EPS Estimate']) if pd.notna(row['EPS Estimate']) else None
                elif 'Estimate' in row.index:
                    estimate = float(row['Estimate']) if pd.notna(row['Estimate']) else None

                if actual is not None:
                    surprise_pct = 0
                    if estimate and estimate != 0:
                        surprise_pct = ((actual - estimate) / abs(estimate)) * 100

                    quarter_str = str(idx)
                    if hasattr(idx, 'strftime'):
                        quarter_str = idx.strftime('%Y-Q') + str((idx.month - 1) // 3 + 1)

                    results.append({
                        'quarter': quarter_str,
                        'actual': actual,
                        'estimate': estimate,
                        'surprise_pct': surprise_pct
                    })

            if results:
                return results

        # 방법 2: earnings_history 사용 (일부 종목)
        try:
            history = stock.earnings_history
            if history is not None and not history.empty:
                results = []
                for idx, row in history.head(4).iterrows():
                    actual = row.get('epsActual', row.get('Actual', None))
                    estimate = row.get('epsEstimate', row.get('Estimate', None))

                    if actual is not None and pd.notna(actual):
                        actual = float(actual)
                        estimate = float(estimate) if estimate and pd.notna(estimate) else None

                        surprise_pct = 0
                        if estimate and estimate != 0:
                            surprise_pct = ((actual - estimate) / abs(estimate)) * 100

                        quarter_str = str(idx) if not hasattr(idx, 'strftime') else idx.strftime('%Y-%m')

                        results.append({
                            'quarter': quarter_str,
                            'actual': actual,
                            'estimate': estimate,
                            'surprise_pct': surprise_pct
                        })

                if results:
                    return results
        except Exception:
            pass

        # 방법 3: earnings 사용 (연간 데이터라도 최근 값 추출)
        try:
            earnings = stock.earnings
            if earnings is not None and not earnings.empty:
                results = []
                # 최근 연도 데이터
                for idx in earnings.index[-2:]:  # 최근 2개년
                    row = earnings.loc[idx]
                    actual = None

                    if 'Earnings' in row.index:
                        actual = float(row['Earnings']) if pd.notna(row['Earnings']) else None
                    elif isinstance(row, (int, float)):
                        actual = float(row)

                    if actual is not None:
                        results.append({
                            'quarter': f"FY{idx}",
                            'actual': actual,
                            'estimate': None,
                            'surprise_pct': 0  # 연간은 surprise 계산 어려움
                        })

                return results
        except Exception:
            pass

        return []

    except Exception:
        return []


def get_revenue_cagr(historical_financials: list, years: int = 3) -> float:
    """
    Historical financials에서 Revenue CAGR 계산

    Args:
        historical_financials: get_stock_data()['historical_financials']
        years: CAGR 계산 기간

    Returns:
        CAGR (decimal, e.g., 0.15 = 15%)
    """
    if not historical_financials or len(historical_financials) < 2:
        return 0

    revenues = [h.get('revenue', 0) for h in historical_financials if h.get('revenue', 0) > 0]

    if len(revenues) < 2:
        return 0

    # historical_financials는 최신이 먼저 (역순)
    n = min(years, len(revenues) - 1)
    end_revenue = revenues[0]  # 최신
    start_revenue = revenues[n]  # n년 전

    if start_revenue <= 0 or end_revenue <= 0:
        return 0

    cagr = (end_revenue / start_revenue) ** (1 / n) - 1
    return cagr


def get_analyst_estimates(ticker: str) -> dict:
    """
    Analyst EPS 예상치 가져오기 (FY0, FY1, 그리고 FY2/FY3 추정)

    Returns:
        {
            'fy0_eps': float,  # 현재 회계연도 EPS 예상치
            'fy1_eps': float,  # 다음 회계연도 EPS 예상치
            'fy1_growth': float,  # FY0 → FY1 성장률
            'num_analysts': int,  # 애널리스트 수
            'trailing_eps': float,  # TTM EPS (참고용)
            'forward_eps': float,  # Forward EPS (info 기준)
        }
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        result = {
            'fy0_eps': 0,
            'fy1_eps': 0,
            'fy1_growth': 0,
            'num_analysts': 0,
            'trailing_eps': info.get('trailingEps', 0) or 0,
            'forward_eps': info.get('forwardEps', 0) or 0,
        }

        # Earnings Estimate 가져오기
        try:
            earnings_est = stock.earnings_estimate
            if earnings_est is not None and not earnings_est.empty:
                # FY0 (0y)
                if '0y' in earnings_est.index:
                    row = earnings_est.loc['0y']
                    result['fy0_eps'] = float(row['avg']) if pd.notna(row['avg']) else 0
                    result['num_analysts'] = int(row['numberOfAnalysts']) if pd.notna(row['numberOfAnalysts']) else 0

                # FY1 (+1y)
                if '+1y' in earnings_est.index:
                    row = earnings_est.loc['+1y']
                    result['fy1_eps'] = float(row['avg']) if pd.notna(row['avg']) else 0
                    result['fy1_growth'] = float(row['growth']) if pd.notna(row['growth']) else 0

                    # num_analysts 업데이트 (더 많은 쪽으로)
                    if pd.notna(row['numberOfAnalysts']):
                        result['num_analysts'] = max(result['num_analysts'], int(row['numberOfAnalysts']))

        except Exception:
            pass

        # FY0가 없으면 forward_eps 사용
        if result['fy0_eps'] == 0 and result['forward_eps'] > 0:
            result['fy0_eps'] = result['forward_eps']

        # FY1이 없으면 FY0에 성장률 적용 또는 forward_eps 사용
        if result['fy1_eps'] == 0:
            if result['fy0_eps'] > 0 and result['fy1_growth'] > 0:
                result['fy1_eps'] = result['fy0_eps'] * (1 + result['fy1_growth'])
            elif result['forward_eps'] > 0:
                result['fy1_eps'] = result['forward_eps']

        return result

    except Exception as e:
        return {
            'fy0_eps': 0,
            'fy1_eps': 0,
            'fy1_growth': 0,
            'num_analysts': 0,
            'trailing_eps': 0,
            'forward_eps': 0,
            'error': str(e)
        }
