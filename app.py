"""
Stock Valuation Pro - Multi-Method Valuation Dashboard
- Tab 1: DCF Valuation (절대가치)
- Tab 2: Relative Valuation (상대가치)
- Tab 3: Summary (Football Field Chart)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import (
    get_stock_data as _get_stock_data, get_peers,
    get_peer_group_data as _get_peer_group_data,
    get_historical_valuation as _get_historical_valuation,
    get_earnings_history as _get_earnings_history,
    get_analyst_estimates as _get_analyst_estimates
)
from dcf_model import WallStreetDCF
from risk_model import (
    generate_risk_scorecard, get_risk_color, get_risk_emoji, get_flag_icon,
    RiskLevel
)

# Caching으로 Rate Limit 방지 (10분간 캐시)
@st.cache_data(ttl=600, show_spinner=False)
def get_stock_data(ticker: str):
    return _get_stock_data(ticker)

@st.cache_data(ttl=600, show_spinner=False)
def get_peer_group_data(peer_tickers: tuple):
    return _get_peer_group_data(list(peer_tickers))

@st.cache_data(ttl=600, show_spinner=False)
def get_historical_valuation(ticker: str, years: int = 5):
    return _get_historical_valuation(ticker, years)

@st.cache_data(ttl=600, show_spinner=False)
def get_earnings_history(ticker: str):
    return _get_earnings_history(ticker)

@st.cache_data(ttl=600, show_spinner=False)
def get_analyst_estimates(ticker: str):
    return _get_analyst_estimates(ticker)

st.set_page_config(page_title="Stock Valuation Pro", page_icon="📊", layout="wide")

# CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-box {
        background: rgba(102, 126, 234, 0.1);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .result-box {
        background: rgba(16, 185, 129, 0.1);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid rgba(16, 185, 129, 0.5);
    }
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        font-size: 0.85rem;
    }
    .guide-text {
        font-size: 0.75rem;
        color: #888;
        line-height: 1.4;
    }
    .premium { color: #ef4444; }
    .discount { color: #10b981; }
    /* Column height equalization for cards */
    [data-testid="column"] > div {
        height: 100%;
    }
    [data-testid="column"] > div > div {
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Stock Valuation Pro</div>', unsafe_allow_html=True)

# ===== Sidebar =====
with st.sidebar:
    st.header("🔧 Stock Selection")
    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
    fetch_btn = st.button("📥 Fetch Data", type="primary", use_container_width=True)

    if fetch_btn:
        with st.spinner(f"Fetching {ticker}..."):
            data, success = get_stock_data(ticker)
            if success:
                st.session_state['stock_data'] = data
                st.session_state['ticker'] = ticker
                # Reset peer data when new stock is loaded
                if 'peer_data' in st.session_state:
                    del st.session_state['peer_data']
                # Reset historical valuation cache
                if 'hist_val' in st.session_state:
                    del st.session_state['hist_val']
                if 'hist_val_key' in st.session_state:
                    del st.session_state['hist_val_key']
                # Reset Valuation Simulator & Scenario widget values
                keys_to_reset = [
                    f"sim_target_pe_{ticker}",
                    f"sim_fy1_eps_{ticker}",
                    f"sim_fy2_eps_{ticker}",
                    f"sim_fy3_eps_{ticker}",
                    f"scenario_pe_bull_{ticker}",
                    f"scenario_eps_bull_{ticker}",
                    f"scenario_pe_base_{ticker}",
                    f"scenario_eps_base_{ticker}",
                    f"scenario_pe_bear_{ticker}",
                    f"scenario_eps_bear_{ticker}",
                ]
                for key in keys_to_reset:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success(f"✅ {ticker} loaded!")
            else:
                st.error(f"Error: {data.get('error', 'Failed')}")

# ===== 메인 =====
if 'stock_data' not in st.session_state:
    st.info("👈 Enter a stock ticker and click 'Fetch Data' to start")
    st.markdown("""
    ### 📖 How to Use
    1. **Enter Ticker** (예: MSFT, AAPL, GOOGL)
    2. **Fetch Data** 클릭
    3. **3개 탭**에서 다양한 밸류에이션 확인
    """)
    st.stop()

data = st.session_state['stock_data']
ticker = st.session_state.get('ticker', 'N/A')

# 기본 정보 헤더
col1, col2, col3, col4 = st.columns(4)
col1.metric("Stock", ticker)
col2.metric("Current Price", f"${data.get('current_price', 0):.2f}")
col3.metric("Market Cap", f"${data.get('market_cap', 0)/1e9:.1f}B")
col4.metric("Sector", data.get('sector', 'N/A'))

st.divider()

# ===== 3-Tab 구조 =====
tab1, tab2, tab3 = st.tabs(["📊 DCF Valuation", "📈 Relative Valuation", "🎯 Summary"])

# ============================================================
# TAB 1: DCF Valuation
# ============================================================
with tab1:
    # ===== Value Trap Risk Badge (상단 표시) =====
    # 사전 계산: Risk Scorecard
    from risk_model import generate_risk_scorecard, RiskLevel, get_risk_emoji

    # WACC 계산 (Risk Scorecard에 필요)
    _temp_dcf = WallStreetDCF(data)
    _temp_wacc_result = _temp_dcf.calculate_auto_wacc()
    _temp_wacc = _temp_wacc_result['wacc']

    # Earnings History (캐싱)
    if 'earnings_history' not in st.session_state or st.session_state.get('earnings_ticker') != ticker:
        _earnings_hist = get_earnings_history(ticker)
        st.session_state['earnings_history'] = _earnings_hist
        st.session_state['earnings_ticker'] = ticker
    else:
        _earnings_hist = st.session_state['earnings_history']

    # Risk Scorecard 생성
    risk_scorecard = generate_risk_scorecard(
        ticker=ticker,
        financial_data=data,
        wacc=_temp_wacc,
        earnings_surprises=_earnings_hist
    )

    # Badge 색상
    if risk_scorecard.risk_level == RiskLevel.LOW:
        badge_bg, badge_text = "#dcfce7", "#166534"
        badge_emoji = "🟢"
    elif risk_scorecard.risk_level == RiskLevel.MODERATE:
        badge_bg, badge_text = "#fef3c7", "#92400e"
        badge_emoji = "🟡"
    else:
        badge_bg, badge_text = "#fee2e2", "#991b1b"
        badge_emoji = "🔴"

    # Badge 표시
    badge_col1, badge_col2 = st.columns([3, 1])
    with badge_col1:
        st.subheader("📈 Historical Free Cash Flow")
    with badge_col2:
        st.markdown(f"""
        <div style="background:{badge_bg}; color:{badge_text}; padding:8px 16px; border-radius:20px; text-align:center; font-weight:bold;">
            {badge_emoji} {risk_scorecard.risk_level.value.upper()} RISK ({risk_scorecard.flags_triggered}/{risk_scorecard.total_flags})
        </div>
        """, unsafe_allow_html=True)

    # session_state에 저장 (Tab 3에서 재사용)
    st.session_state['risk_scorecard'] = risk_scorecard

    historical = data.get('historical_financials', [])

    # FCF & Revenue 데이터 수집
    all_financial_data = []
    for h in historical:
        year = h.get('year', '')
        fcf = h.get('fcf', 0)
        if fcf == 0:
            op_cf = h.get('operating_cf', 0)
            capex = h.get('capex', 0)
            fcf = op_cf - capex if op_cf > 0 else 0
        revenue = h.get('revenue', 0)
        if fcf != 0 or revenue > 0:
            all_financial_data.append({'year': str(year), 'fcf': fcf, 'revenue': revenue})

    all_financial_data = sorted(all_financial_data, key=lambda x: x['year'])

    # TTM 데이터 추가
    ttm_fcf = data.get('fcf', 0)
    ttm_revenue = data.get('revenue', 0)
    if ttm_fcf and ttm_fcf != 0:
        all_financial_data.append({'year': 'TTM', 'fcf': ttm_fcf, 'revenue': ttm_revenue})

    available_years = len(all_financial_data)

    if available_years == 0:
        st.error("⚠️ 재무 데이터가 없습니다.")
        st.stop()

    # 모든 데이터 사용
    fcf_data = all_financial_data
    years_list = [fd['year'] for fd in fcf_data]
    st.caption(f"💡 {available_years} years ({years_list[0]} ~ {years_list[-1]})")

    # DCF 모델 초기화 (성장률 계산에 필요)
    dcf_model = WallStreetDCF(data)

    # ===== Smart Defaults & Lifecycle Classification =====
    smart_defaults = dcf_model.get_smart_defaults()
    lifecycle = smart_defaults['lifecycle']

    # Lifecycle 표시 (Insight Card)
    lifecycle_colors = {
        'Hyper-Growth': ('#ef4444', '#fef2f2'),  # 빨강
        'High-Growth': ('#f59e0b', '#fffbeb'),   # 노랑
        'Stable': ('#10b981', '#ecfdf5')          # 초록
    }
    lc_color, lc_bg = lifecycle_colors.get(lifecycle.stage_label, ('#6b7280', '#f9fafb'))

    st.markdown(f"""
    <div style="background: {lc_bg}; padding: 16px 20px; border-radius: 10px;
                border-left: 5px solid {lc_color}; margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 0.85rem; color: #666;">Company Stage</span>
                <h3 style="margin: 5px 0; color: {lc_color};">{lifecycle.stage_label}</h3>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 2rem; font-weight: bold; color: {lc_color};">
                    {lifecycle.projection_years}Y
                </span>
                <br><span style="font-size: 0.75rem; color: #888;">Projection Period</span>
            </div>
        </div>
        <p style="font-size: 0.85rem; color: #555; margin-top: 10px; margin-bottom: 0;">
            {lifecycle.insight}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Smart Insights Expander
    with st.expander("💡 Smart Default Insights", expanded=False):
        for insight in smart_defaults['insights']:
            st.markdown(f"• {insight}")

        st.markdown("---")
        st.markdown("**Convergence Schedules (연도별 수렴)**")
        col_a, col_b = st.columns(2)
        with col_a:
            # Growth Schedule
            growth_sch = smart_defaults['growth_schedule']
            growth_df = pd.DataFrame({
                'Year': [f"Y{i+1}" for i in range(len(growth_sch))],
                'Growth': [f"{g*100:.1f}%" for g in growth_sch]
            })
            st.markdown("**Growth Decay**")
            st.dataframe(growth_df.T, use_container_width=True)
        with col_b:
            # CapEx Schedule
            capex_sch = smart_defaults['capex_schedule']
            capex_df = pd.DataFrame({
                'Year': [f"Y{i+1}" for i in range(len(capex_sch))],
                'CapEx%': [f"{c*100:.1f}%" for c in capex_sch]
            })
            st.markdown("**CapEx Convergence**")
            st.dataframe(capex_df.T, use_container_width=True)

    st.divider()

    # 성장률 계산 (FCF & Revenue)
    for i in range(1, len(fcf_data)):
        curr_year = fcf_data[i]['year']
        prev_year = fcf_data[i-1]['year']
        # TTM vs FY는 기간 겹침으로 성장률 비교 무의미 → "-" 표시
        is_ttm_vs_fy = (curr_year == 'TTM' and prev_year != 'TTM')

        # FCF Growth
        prev_fcf = fcf_data[i-1]['fcf']
        curr_fcf = fcf_data[i]['fcf']
        if is_ttm_vs_fy:
            fcf_data[i]['fcf_growth'] = None
        elif prev_fcf > 0 and curr_fcf > 0:
            fcf_data[i]['fcf_growth'] = (curr_fcf - prev_fcf) / prev_fcf
        else:
            fcf_data[i]['fcf_growth'] = None

        # Revenue Growth
        prev_rev = fcf_data[i-1]['revenue']
        curr_rev = fcf_data[i]['revenue']
        if is_ttm_vs_fy:
            fcf_data[i]['rev_growth'] = None
        elif prev_rev > 0 and curr_rev > 0:
            fcf_data[i]['rev_growth'] = (curr_rev - prev_rev) / prev_rev
        else:
            fcf_data[i]['rev_growth'] = None

    # CAGR 계산 함수
    def calc_cagr(data_list, key, years):
        """최근 n년 CAGR 계산"""
        if len(data_list) < years + 1:
            years = len(data_list) - 1
        if years < 1:
            return None
        start_val = data_list[-(years+1)][key]
        end_val = data_list[-1][key]
        if start_val > 0 and end_val > 0:
            return (end_val / start_val) ** (1 / years) - 1
        return None

    # FCF CAGR (3Y, 5Y)
    fcf_cagr_3y = calc_cagr(fcf_data, 'fcf', 3)
    fcf_cagr_5y = calc_cagr(fcf_data, 'fcf', 5)
    historical_fcf_cagr = fcf_cagr_5y if fcf_cagr_5y else (fcf_cagr_3y or 0.10)

    # Revenue CAGR (3Y, 5Y)
    rev_cagr_3y = calc_cagr(fcf_data, 'revenue', 3)
    rev_cagr_5y = calc_cagr(fcf_data, 'revenue', 5)

    # 2. Revenue Growth (Quarterly YoY from yfinance)
    revenue_growth = data.get('revenue_growth', 0) or 0

    # 3. Historical Revenue CAGR (WallStreetDCF에서 계산)
    hist_avgs = dcf_model.get_historical_averages()
    revenue_cagr = hist_avgs.get('blended_growth', revenue_growth)

    avg_growth = historical_fcf_cagr  # 기본값

    # Revenue & FCF 테이블 - Annual/Quarterly 전환
    view_mode = st.radio(
        "View",
        ["Annual", "Quarterly"],
        horizontal=True,
        label_visibility="collapsed",
        key="financial_table_view"
    )

    if view_mode == "Annual":
        # Annual 테이블 (기존)
        table_data = {
            '': ['Revenue (M)', 'Rev Growth', 'FCF (M)', 'FCF Growth']
        }
        for fd in fcf_data:
            rev_g = fd.get('rev_growth')
            fcf_g = fd.get('fcf_growth')
            rev_g_str = f"{rev_g*100:.1f}%" if rev_g is not None else "-"
            fcf_g_str = f"{fcf_g*100:.1f}%" if fcf_g is not None else "-"
            table_data[fd['year']] = [
                f"{fd['revenue']/1e6:,.0f}" if fd['revenue'] > 0 else "-",
                rev_g_str,
                f"{fd['fcf']/1e6:,.0f}" if fd['fcf'] != 0 else "-",
                fcf_g_str
            ]
        st.dataframe(pd.DataFrame(table_data).set_index('').T, use_container_width=True)
    else:
        # Quarterly 테이블 (신규)
        quarterly_data = data.get('quarterly_financials', [])
        if not quarterly_data:
            st.warning("Quarterly 데이터가 없습니다.")
        else:
            # YoY, QoQ 성장률 계산
            for i, qd in enumerate(quarterly_data):
                # QoQ: 이전 분기 대비
                if i > 0:
                    prev = quarterly_data[i-1]
                    if prev['revenue'] > 0 and qd['revenue'] > 0:
                        qd['rev_qoq'] = (qd['revenue'] - prev['revenue']) / prev['revenue']
                    if prev['fcf'] != 0 and qd['fcf'] != 0:
                        qd['fcf_qoq'] = (qd['fcf'] - prev['fcf']) / abs(prev['fcf'])

                # YoY: 4분기 전 대비
                if i >= 4:
                    prev_yr = quarterly_data[i-4]
                    if prev_yr['revenue'] > 0 and qd['revenue'] > 0:
                        qd['rev_yoy'] = (qd['revenue'] - prev_yr['revenue']) / prev_yr['revenue']
                    if prev_yr['fcf'] != 0 and qd['fcf'] != 0:
                        qd['fcf_yoy'] = (qd['fcf'] - prev_yr['fcf']) / abs(prev_yr['fcf'])

            # 최근 6분기만 표시
            display_quarters = quarterly_data[-6:] if len(quarterly_data) >= 6 else quarterly_data

            table_data = {
                '': ['Revenue (M)', 'Rev YoY', 'Rev QoQ', 'FCF (M)', 'FCF YoY', 'FCF QoQ']
            }
            for qd in display_quarters:
                yoy_str = f"{qd['rev_yoy']*100:.1f}%" if qd.get('rev_yoy') is not None else "-"
                qoq_str = f"{qd['rev_qoq']*100:.1f}%" if qd.get('rev_qoq') is not None else "-"
                fcf_yoy_str = f"{qd['fcf_yoy']*100:.1f}%" if qd.get('fcf_yoy') is not None else "-"
                fcf_qoq_str = f"{qd['fcf_qoq']*100:.1f}%" if qd.get('fcf_qoq') is not None else "-"

                table_data[qd['quarter']] = [
                    f"{qd['revenue']/1e6:,.0f}" if qd['revenue'] > 0 else "-",
                    yoy_str,
                    qoq_str,
                    f"{qd['fcf']/1e6:,.0f}" if qd['fcf'] != 0 else "-",
                    fcf_yoy_str,
                    fcf_qoq_str
                ]

            st.dataframe(pd.DataFrame(table_data).set_index('').T, use_container_width=True)

    # CAGR 요약
    st.markdown("**CAGR Summary**")
    cagr_cols = st.columns(4)
    with cagr_cols[0]:
        st.metric("Revenue 3Y", f"{rev_cagr_3y*100:.1f}%" if rev_cagr_3y else "N/A")
    with cagr_cols[1]:
        st.metric("Revenue 5Y", f"{rev_cagr_5y*100:.1f}%" if rev_cagr_5y else "N/A")
    with cagr_cols[2]:
        st.metric("FCF 3Y", f"{fcf_cagr_3y*100:.1f}%" if fcf_cagr_3y else "N/A")
    with cagr_cols[3]:
        st.metric("FCF 5Y", f"{fcf_cagr_5y*100:.1f}%" if fcf_cagr_5y else "N/A")

    base_fcf = fcf_data[-1]['fcf']
    base_year_str = fcf_data[-1]['year']
    base_year = datetime.now().year if base_year_str == 'TTM' else int(base_year_str)

    st.divider()

    # ===== DCF 가정값 =====
    st.subheader("⚙️ DCF Assumptions")

    wacc_result = dcf_model.calculate_auto_wacc()
    auto_wacc = wacc_result['wacc'] * 100
    smart_growth_pct = smart_defaults['assumptions']['initial_growth'] * 100
    fcf_cagr_pct = historical_fcf_cagr * 100
    revenue_qtr_yoy_pct = revenue_growth * 100  # yfinance revenueGrowth = Quarterly YoY
    revenue_cagr_pct = revenue_cagr * 100

    # Smart Default 버튼 - 클릭 시 모든 값을 추천안으로 세팅
    if st.button("🤖 Smart Default 적용", help="모든 값을 추천안으로 자동 세팅"):
        # 내부 상태
        st.session_state['_growth_rate_value'] = smart_growth_pct
        st.session_state['_proj_years'] = lifecycle.projection_years
        st.session_state['_apply_decay'] = True
        st.session_state['_tv_method'] = "Both"
        st.session_state['_use_auto_wacc'] = True
        # 위젯 key 직접 설정
        st.session_state['growth_rate_input'] = round(smart_growth_pct, 2)
        st.session_state['proj_years_select'] = lifecycle.projection_years
        st.session_state['apply_decay_check'] = True
        st.session_state['tv_method_radio'] = "Both"
        st.session_state['auto_wacc_toggle'] = True
        st.rerun()

    st.caption(f"🤖 Smart Default: {lifecycle.stage_label} | Growth: {smart_growth_pct:.1f}% | {lifecycle.projection_years}Y | Decay ON | WACC: {auto_wacc:.1f}%")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # 초기값 설정
        if '_growth_rate_value' not in st.session_state:
            st.session_state['_growth_rate_value'] = smart_growth_pct

        # Growth Rate 직접 입력
        growth_rate = st.number_input(
            "Growth Rate (%)",
            min_value=-50.0,
            max_value=150.0,
            value=round(min(max(st.session_state.get('_growth_rate_value', smart_growth_pct), -50.0), 150.0), 2),
            step=1.0,
            format="%.2f",
            key="growth_rate_input"
        )
        st.session_state['_growth_rate_value'] = growth_rate

        if growth_rate > 50:
            st.warning(f"⚠️ High Growth ({growth_rate:.1f}%)")

        # 참고 데이터 표시
        st.markdown(f"""
        <div class="guide-text">
        💡 <b>Reference:</b><br>
        • FCF CAGR: {fcf_cagr_pct:.1f}%<br>
        • Revenue Qtr YoY: {revenue_qtr_yoy_pct:.1f}%<br>
        • Revenue CAGR (5Y): {revenue_cagr_pct:.1f}%
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Projection Period 옵션
        proj_year_options = [5, 7, 10]

        # 초기값 설정
        if '_proj_years' not in st.session_state:
            st.session_state['_proj_years'] = lifecycle.projection_years
        if '_apply_decay' not in st.session_state:
            st.session_state['_apply_decay'] = True

        current_proj = st.session_state.get('_proj_years', lifecycle.projection_years)
        default_proj_idx = proj_year_options.index(current_proj) if current_proj in proj_year_options else 0

        selected_proj_years = st.selectbox(
            "Projection Years",
            options=proj_year_options,
            index=default_proj_idx,
            format_func=lambda x: f"{x}Y",
            key="proj_years_select"
        )
        st.session_state['_proj_years'] = selected_proj_years

        apply_decay = st.checkbox(
            "Apply Growth Decay",
            value=st.session_state.get('_apply_decay', True),
            key="apply_decay_check",
            help="성장률을 Terminal Growth로 점진적 감소"
        )
        st.session_state['_apply_decay'] = apply_decay

        st.markdown(f"""
        <div class="guide-text">
        💡 <b>Lifecycle 기준:</b><br>
        • Hyper-Growth: 10Y<br>
        • High-Growth: 7Y<br>
        • Stable: 5Y
        </div>
        """, unsafe_allow_html=True)

    with col3:
        rf_rate = dcf_model.risk_free_rate

        tv_options = ["Both", "Perpetuity Growth", "Exit Multiple"]

        # 초기값 설정
        if '_tv_method' not in st.session_state:
            st.session_state['_tv_method'] = "Both"

        current_tv = st.session_state.get('_tv_method', 'Both')
        tv_default_idx = tv_options.index(current_tv) if current_tv in tv_options else 0

        # Terminal Value Method 선택
        tv_method = st.radio(
            "Terminal Value Method",
            options=tv_options,
            index=tv_default_idx,
            horizontal=True,
            key="tv_method_radio",
            help="Both: 두 방식 평균"
        )
        st.session_state['_tv_method'] = tv_method

        # Perpetuity Growth Rate 입력 (Perpetuity Growth 또는 Both일 때만)
        if tv_method in ["Both", "Perpetuity Growth"]:
            perpetual_growth = st.number_input(
                "Perpetual Growth Rate (%)",
                min_value=0.0,
                max_value=5.0,
                value=2.5,
                step=0.1,
                format="%.1f",
                key="perp_growth",
                help="Terminal Value 이후 영구 성장률 (GDP 수준 권장)"
            )
            if perpetual_growth / 100 > rf_rate:
                st.warning(f"⚠️ Risk-Free Rate({rf_rate*100:.1f}%) 초과")
        else:
            perpetual_growth = 2.5  # 기본값

        # Exit Multiple 관련 변수 계산
        current_ev_ebitda = data.get('ev_ebitda', 0) or 0
        current_fcf = data.get('fcf', 0) or 0
        current_ebitda = data.get('ebitda', 0) or 0
        sector_avg_multiple = dcf_model.sector_defaults.get('exit_multiple', 15)

        # FCF/EBITDA 비율 계산 (Fair Multiple용)
        if current_ebitda > 0 and current_fcf > 0:
            fcf_to_ebitda = current_fcf / current_ebitda
            fcf_to_ebitda = max(0.3, min(0.8, fcf_to_ebitda))
        else:
            fcf_to_ebitda = 0.6

        # Fair Multiple (Gordon Growth 기반)
        wacc_decimal = auto_wacc / 100
        g_decimal = perpetual_growth / 100
        if wacc_decimal > g_decimal:
            fair_multiple = fcf_to_ebitda / (wacc_decimal - g_decimal)
            fair_multiple = max(5.0, min(25.0, fair_multiple))
        else:
            fair_multiple = 10.0

        # Target Multiple 결정
        if selected_proj_years <= 5:
            base_target = sector_avg_multiple
        elif selected_proj_years >= 10:
            base_target = fair_multiple
        else:
            blend = (selected_proj_years - 5) / 5
            base_target = sector_avg_multiple - (sector_avg_multiple - fair_multiple) * blend

        # Growth Decay 반영
        if apply_decay:
            target_multiple = base_target
            decay_note = "Decay ON"
        else:
            target_multiple = (base_target + fair_multiple) / 2
            decay_note = "Decay OFF → Fair 조정"

        # Exit Multiple 결정
        if current_ev_ebitda > 0:
            if current_ev_ebitda > target_multiple:
                default_exit_multiple = target_multiple
            else:
                default_exit_multiple = current_ev_ebitda
        else:
            default_exit_multiple = target_multiple

        # Exit Multiple 입력 (Exit Multiple 또는 Both일 때만)
        if tv_method in ["Both", "Exit Multiple"]:
            exit_multiple = st.number_input(
                "Exit EV/EBITDA Multiple",
                min_value=3.0,
                max_value=60.0,
                value=float(round(default_exit_multiple, 1)),
                step=0.5,
                format="%.1f",
                key=f"exit_mult_{selected_proj_years}_{apply_decay}",
                help=f"{decay_note}"
            )

            # 구조화된 Caption
            st.caption(f"""
**현재:** {current_ev_ebitda:.1f}x | **섹터:** {sector_avg_multiple}x | **Fair:** {fair_multiple:.1f}x
**{selected_proj_years}Y Target:** {target_multiple:.1f}x ({decay_note})
""")
        else:
            exit_multiple = default_exit_multiple  # 기본값

        # 가이드 (Exit Multiple 관련일 때만)
        if tv_method in ["Both", "Exit Multiple"]:
            if apply_decay:
                guide_text = f"Decay ON → 섹터({sector_avg_multiple}x) 기준"
            else:
                guide_text = f"Decay OFF → Fair({fair_multiple:.1f}x)에 가깝게"

            st.markdown(f"""
            <div class="guide-text">
            💡 <b>Exit Multiple:</b><br>
            • 5Y+Decay → 섹터({sector_avg_multiple}x)<br>
            • 10Y or No Decay → Fair({fair_multiple:.1f}x)<br>
            • {guide_text}
        </div>
        """, unsafe_allow_html=True)

    with col4:
        # 초기값 설정
        if '_use_auto_wacc' not in st.session_state:
            st.session_state['_use_auto_wacc'] = True

        use_auto_wacc = st.checkbox(
            "Auto WACC",
            value=st.session_state.get('_use_auto_wacc', True),
            key="auto_wacc_toggle"
        )
        st.session_state['_use_auto_wacc'] = use_auto_wacc

        if use_auto_wacc:
            discount_rate = st.number_input(
                "Discount Rate (WACC) (%)",
                min_value=3.0, max_value=20.0,
                value=round(auto_wacc, 2),
                step=0.5, format="%.2f",
                disabled=True,
                key=f"wacc_auto_display"
            )
            discount_rate = auto_wacc
        else:
            discount_rate = st.number_input(
                "Discount Rate (WACC) (%)",
                min_value=3.0, max_value=20.0,
                value=st.session_state.get('_manual_wacc_value', 8.0),
                step=0.5, format="%.1f",
                key="wacc_manual_input"
            )
            st.session_state['_manual_wacc_value'] = discount_rate

        # Risk Scorecard에서 사용할 WACC 저장 (decimal 형태)
        st.session_state['calculated_wacc'] = discount_rate / 100

        st.markdown(f"""
        <div class="guide-text">
        💡 <b>Guideline:</b><br>
        • 대형 우량주: 7~9%<br>
        • 성장주/중소형: 10~12%<br>
        • 고위험/신흥시장: 12~15%
        </div>
        """, unsafe_allow_html=True)

    # WACC 상세
    with st.expander("📊 WACC Calculation Details"):
        coe = wacc_result['cost_of_equity']
        cod = wacc_result['cost_of_debt']
        weights = wacc_result['weights']

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Cost of Equity (Ke)**")
            st.write(f"• Beta: {coe['beta_raw']:.3f} → {coe['beta_used']:.3f} (Adjusted)")
            st.write(f"• Risk-Free Rate: {rf_rate*100:.2f}%")
            st.write(f"• **Ke = {coe['ke']*100:.2f}%**")

        with col_b:
            st.markdown("**Cost of Debt (Kd)**")
            if cod['credit_rating']:
                st.write(f"• Synthetic Rating: {cod['credit_rating']}")
                st.write(f"• ICR: {cod['icr']:.2f}x")
            st.write(f"• **Kd (After-tax) = {cod['kd_aftertax']*100:.2f}%**")

        st.markdown(f"**WACC = {wacc_result['wacc']*100:.2f}%** (E/V={weights['equity']*100:.0f}%, D/V={weights['debt']*100:.0f}%)")

    # 변환
    growth_dec = growth_rate / 100
    perp_dec = perpetual_growth / 100
    disc_dec = discount_rate / 100

    # Market Implied Growth는 DCF 결과 섹션 이후에 통합 표시 (중복 제거)
    current_price = data.get('current_price', 0)

    if disc_dec <= perp_dec:
        st.error("⚠️ Discount Rate > Perpetual Growth Rate 필요!")
        st.stop()

    st.divider()

    # DCF 계산 - 사용자 선택 Projection Period & Decay 옵션 사용
    projection_years = selected_proj_years
    st.subheader(f"📊 Future FCF Projections ({projection_years} Years)")

    if base_fcf <= 0:
        st.error("⚠️ Base FCF가 0 이하입니다.")
        st.stop()

    projections = []
    # Decay 적용 여부에 따라 Growth Schedule 생성
    use_decay_schedule = apply_decay

    if use_decay_schedule:
        # 선택한 projection_years에 맞게 growth schedule 생성
        from valuation_utils import generate_growth_decay_schedule
        growth_schedule = generate_growth_decay_schedule(
            initial_growth=growth_dec,
            terminal_growth=perp_dec,
            years=projection_years,
            decay_type='linear'
            )
    else:
        growth_schedule = None

    for i in range(projection_years):
        year = base_year + i + 1

        if use_decay_schedule and i < len(growth_schedule):
            # 연도별 다른 성장률 적용
            year_growth = growth_schedule[i]
        else:
            year_growth = growth_dec

        if i == 0:
            fcf = base_fcf * (1 + year_growth)
        else:
            fcf = projections[-1]['fcf'] * (1 + year_growth)

        # Mid-year Convention 적용: 현금흐름이 연중 발생한다고 가정
        pv = fcf / ((1 + disc_dec) ** (i + 0.5))
        projections.append({'year': year, 'fcf': fcf, 'pv': pv, 'growth': year_growth})

    final_fcf = projections[-1]['fcf']
    final_year_ebitda = data.get('ebitda', 0) or 0
    if final_year_ebitda > 0 and len(projections) > 0:
        # 마지막 해 EBITDA 추정: FCF 기반 역산 (대략적 추정)
        # FCF ≈ EBITDA × (1 - Tax) × (1 - Reinvestment Rate)
        # 간략화: EBITDA 성장 = FCF 성장으로 가정
        ebitda_growth_factor = final_fcf / base_fcf if base_fcf > 0 else 1
        final_year_ebitda = final_year_ebitda * ebitda_growth_factor

    # === Terminal Value 계산 ===
    # 1. Perpetuity Growth Method
    tv_perpetuity = final_fcf * (1 + perp_dec) / (disc_dec - perp_dec) if disc_dec > perp_dec else 0
    pv_tv_perpetuity = tv_perpetuity / ((1 + disc_dec) ** projection_years)

    # 2. Exit Multiple Method (EV/EBITDA)
    exit_mult_dec = exit_multiple  # 이미 숫자로 받음 (UI에서)
    tv_exit_multiple = final_year_ebitda * exit_mult_dec if final_year_ebitda > 0 else 0
    pv_tv_exit_multiple = tv_exit_multiple / ((1 + disc_dec) ** projection_years)

    # TV Method에 따른 최종 TV 선택
    if tv_method == "Perpetuity Growth":
        tv = tv_perpetuity
        pv_tv = pv_tv_perpetuity
    elif tv_method == "Exit Multiple":
        tv = tv_exit_multiple
        pv_tv = pv_tv_exit_multiple
    else:  # "Both" - 평균 사용
        tv = (tv_perpetuity + tv_exit_multiple) / 2 if tv_exit_multiple > 0 else tv_perpetuity
        pv_tv = (pv_tv_perpetuity + pv_tv_exit_multiple) / 2 if pv_tv_exit_multiple > 0 else pv_tv_perpetuity

    # Smart Default 모드에서는 연도별 성장률도 표시
    if use_decay_schedule:
        proj_table = {'': ['Year', 'Growth', 'Future FCF', 'PV of FCF']}
        for i, p in enumerate(projections):
            proj_table[str(i+1)] = [
                str(p['year']),
                f"{p['growth']*100:.1f}%",
                f"${p['fcf']/1e6:,.0f}M",
                f"${p['pv']/1e6:,.0f}M"
            ]
        proj_table['TV'] = ['Terminal', f"{perp_dec*100:.1f}%", f"${tv/1e6:,.0f}M", f"${pv_tv/1e6:,.0f}M"]
    else:
        proj_table = {'': ['Year', 'Future FCF', 'PV of FCF']}
        for i, p in enumerate(projections):
            proj_table[str(i+1)] = [str(p['year']), f"${p['fcf']/1e6:,.0f}M", f"${p['pv']/1e6:,.0f}M"]
        proj_table['TV'] = ['Terminal Value', f"${tv/1e6:,.0f}M", f"${pv_tv/1e6:,.0f}M"]

    st.dataframe(pd.DataFrame(proj_table).set_index('').T, use_container_width=True)

    # Smart Default 모드일 때 Growth Decay 시각화
    if use_decay_schedule:
        with st.expander("📉 Growth Decay Visualization", expanded=False):
            years = [f"Y{i+1}" for i in range(len(growth_schedule))]
            growth_pcts = [g * 100 for g in growth_schedule]

            fig_decay = go.Figure()
            fig_decay.add_trace(go.Scatter(
                x=years, y=growth_pcts,
                mode='lines+markers',
                name='Growth Rate',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            ))
            fig_decay.add_hline(
                y=perp_dec * 100,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Terminal Growth: {perp_dec*100:.1f}%"
            )
            fig_decay.update_layout(
                title="Growth Rate Decay to Terminal Growth",
                xaxis_title="Year",
                yaxis_title="Growth Rate (%)",
                height=250,
                margin=dict(t=40, b=40)
            )
            st.plotly_chart(fig_decay, use_container_width=True)

    # 결과 계산 - 각 방식별로 계산
    sum_pv_fcf = sum(p['pv'] for p in projections)
    cash = data.get('cash', 0)
    debt = data.get('total_debt', 0)
    shares = data.get('shares_outstanding', 1)
    current_price = data.get('current_price', 0)

    # Minority Interest, Preferred Stock 차감 (EV → Equity)
    minority_interest = data.get('minority_interest', 0) or 0
    preferred_stock = data.get('preferred_stock', 0) or 0

    # 1. Perpetuity Growth Method 결과
    sum_pv_perpetuity = sum_pv_fcf + pv_tv_perpetuity
    equity_perpetuity = sum_pv_perpetuity + cash - debt - minority_interest - preferred_stock
    dcf_price_perpetuity = equity_perpetuity / shares if shares > 0 else 0

    # 2. Exit Multiple Method 결과
    sum_pv_exit = sum_pv_fcf + pv_tv_exit_multiple
    equity_exit = sum_pv_exit + cash - debt - minority_interest - preferred_stock
    dcf_price_exit = equity_exit / shares if shares > 0 else 0

    # 3. Blended (Both) 결과
    if tv_method == "Perpetuity Growth":
        dcf_price = dcf_price_perpetuity
        sum_pv = sum_pv_perpetuity
        equity = equity_perpetuity
    elif tv_method == "Exit Multiple":
        dcf_price = dcf_price_exit
        sum_pv = sum_pv_exit
        equity = equity_exit
    else:  # "Both"
        dcf_price = (dcf_price_perpetuity + dcf_price_exit) / 2 if dcf_price_exit > 0 else dcf_price_perpetuity
        sum_pv = (sum_pv_perpetuity + sum_pv_exit) / 2 if sum_pv_exit > 0 else sum_pv_perpetuity
        equity = (equity_perpetuity + equity_exit) / 2 if equity_exit > 0 else equity_perpetuity

    # Margin of Safety 계산 및 등급 부여
    mos_pct = (dcf_price / current_price - 1) * 100 if current_price > 0 else 0

    def get_mos_grade(mos):
        """Margin of Safety 등급 체계"""
        if mos >= 30:
            return "🟢 STRONG BUY", "#10b981", "High MoS - Low downside risk"
        elif mos >= 15:
            return "🟢 BUY", "#22c55e", "Attractive valuation"
        elif mos >= 5:
            return "🟡 HOLD/ACCUMULATE", "#84cc16", "Modest upside"
        elif mos >= -10:
            return "🟡 FAIR VALUE", "#f59e0b", "Priced appropriately"
        elif mos >= -25:
            return "🟠 EXPENSIVE", "#f97316", "Limited upside"
        else:
            return "🔴 AVOID", "#ef4444", "Significant overvaluation"

    verdict, color, verdict_desc = get_mos_grade(mos_pct)

    # 결과를 session_state에 저장 (Tab 3에서 사용)
    tv_pct = pv_tv / sum_pv * 100 if sum_pv > 0 else 0
    st.session_state['dcf_result'] = {
        'dcf_price': dcf_price,
        'dcf_price_perpetuity': dcf_price_perpetuity,
        'dcf_price_exit': dcf_price_exit if dcf_price_exit > 0 else None,
        'sum_pv': sum_pv,
        'pv_tv': pv_tv,
        'tv_pct': tv_pct,
        'mos_pct': mos_pct,
        'verdict': verdict,
        'tv_method': tv_method
    }

    # ===== Implied Market Growth 계산 (DCF 가정 동일 적용) =====
    def find_implied_growth():
        """Binary Search로 현재 주가를 정당화하는 초기 성장률 역산"""
        if current_price <= 0 or shares <= 0:
            return None
        target_equity = current_price * shares

        low, high = -0.30, 1.50
        tolerance = 0.005

        for _ in range(50):
            mid = (low + high) / 2

            # FCF Projection (사용자 DCF 가정과 동일)
            pv_sum = 0
            prev_fcf = base_fcf
            for i in range(projection_years):
                if use_decay_schedule and growth_schedule and i < len(growth_schedule):
                    # Decay 적용: mid를 초기값으로 같은 비율로 decay
                    ratio = growth_schedule[i] / growth_dec if growth_dec != 0 else 1
                    year_growth = mid * ratio
                else:
                    year_growth = mid

                if i == 0:
                    fcf_i = base_fcf * (1 + year_growth)
                else:
                    fcf_i = prev_fcf * (1 + year_growth)
                prev_fcf = fcf_i
                pv_i = fcf_i / ((1 + disc_dec) ** (i + 0.5))
                pv_sum += pv_i

            # Terminal Value (사용자 TV Method와 동일)
            if tv_method == "Exit Multiple":
                # Exit Multiple: EBITDA 기반
                ebitda_growth = prev_fcf / base_fcf if base_fcf > 0 else 1
                implied_ebitda = (data.get('ebitda', 0) or 0) * ebitda_growth
                tv_calc = implied_ebitda * exit_multiple if implied_ebitda > 0 else 0
            elif tv_method == "Both":
                # Perpetuity + Exit 평균
                tv_perp = prev_fcf * (1 + perp_dec) / (disc_dec - perp_dec) if disc_dec > perp_dec else 0
                ebitda_growth = prev_fcf / base_fcf if base_fcf > 0 else 1
                implied_ebitda = (data.get('ebitda', 0) or 0) * ebitda_growth
                tv_exit = implied_ebitda * exit_multiple if implied_ebitda > 0 else 0
                tv_calc = (tv_perp + tv_exit) / 2 if tv_exit > 0 else tv_perp
            else:  # Perpetuity Growth
                tv_calc = prev_fcf * (1 + perp_dec) / (disc_dec - perp_dec) if disc_dec > perp_dec else 0

            pv_tv_calc = tv_calc / ((1 + disc_dec) ** projection_years)
            ev_calc = pv_sum + pv_tv_calc
            eq_calc = ev_calc + cash - debt - minority_interest - preferred_stock

            diff_pct = (eq_calc - target_equity) / target_equity if target_equity > 0 else 0
            if abs(diff_pct) < tolerance:
                return mid
            elif eq_calc < target_equity:
                low = mid
            else:
                high = mid

        return mid if abs(diff_pct) < 0.10 else None

    implied_growth = find_implied_growth()

    st.divider()

    # 결과 표시
    st.subheader("💰 DCF Valuation Result")

    # 두 방식 비교 테이블 (Both 선택 시)
    if tv_method == "Both" and dcf_price_exit > 0:
        st.markdown("##### 📊 Valuation Comparison")

        diff_perp = (dcf_price_perpetuity / current_price - 1) * 100 if current_price > 0 else 0
        diff_exit = (dcf_price_exit / current_price - 1) * 100 if current_price > 0 else 0

        comparison_df = pd.DataFrame({
            'Method': [
                f'Perpetuity Growth (g={perp_dec*100:.1f}%)',
                f'Exit Multiple ({exit_multiple:.1f}x EBITDA)',
                '**Blended Average**'
            ],
            'Fair Value': [
                f'${dcf_price_perpetuity:.2f}',
                f'${dcf_price_exit:.2f}',
                f'**${dcf_price:.2f}**'
            ],
            'vs Current': [
                f'{diff_perp:+.1f}%',
                f'{diff_exit:+.1f}%',
                f'**{mos_pct:+.1f}%**'
            ]
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        summary_df = pd.DataFrame({
            'Item': ['Enterprise Value', 'Cash', 'Debt', 'Equity Value', 'Shares'],
            'Value': [
                f"${sum_pv/1e9:.2f}B",
                f"${cash/1e9:.2f}B",
                f"${debt/1e9:.2f}B",
                f"${equity/1e9:.2f}B",
                f"{shares/1e6:,.0f}M"
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        if tv_pct > 75:
            st.warning(f"⚠️ Terminal Value = {tv_pct:.0f}% (높음)")

    with col2:
        # Implied Growth 표시 문자열
        if implied_growth is not None:
            ig_str = f'<p style="font-size:0.85em; color:#888;">Implied Growth: {implied_growth*100:.1f}%</p>'
        else:
            ig_str = '<p style="font-size:0.85em; color:#888;">Implied Growth: N/A</p>'

        st.markdown(f"""
        <div class="result-box">
            <h2 style="margin:0;">DCF Fair Value</h2>
            <h1 style="margin:10px 0; color:#667eea;">${dcf_price:.2f}</h1>
            <hr>
            <p><b>Current:</b> ${current_price:.2f}</p>
            <p><b>Margin of Safety:</b> <span style="color:{color}; font-weight:bold;">{mos_pct:+.1f}%</span></p>
            {ig_str}
            <h3 style="color:{color}; margin-top:10px;">{verdict}</h3>
            <p style="font-size:0.85em; color:#6b7280;">{verdict_desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # ===== Sensitivity Analysis =====
    st.divider()
    st.subheader("📊 Sensitivity Analysis (WACC × Terminal Growth)")

    # WACC vs Perpetual Growth (현업 표준)
    wacc_range = [disc_dec - 0.02, disc_dec - 0.01, disc_dec, disc_dec + 0.01, disc_dec + 0.02]
    growth_range = [perp_dec - 0.01, perp_dec - 0.005, perp_dec, perp_dec + 0.005, perp_dec + 0.01]

    # 음수 방지
    wacc_range = [max(w, 0.03) for w in wacc_range]
    growth_range = [max(g, 0.0) for g in growth_range]

    def calc_dcf_value_full(wacc_val, tgr_val, fcf_growth_val, use_schedule=False, schedule=None):
        """주어진 WACC, Terminal Growth, FCF Growth로 DCF 가치 계산 (Mid-year Convention 적용)"""
        if wacc_val <= tgr_val:
            return None

        # FCF 프로젝션 PV (Mid-year Convention)
        pv_sum = 0
        prev_fcf = base_fcf
        for i in range(projection_years):
            if use_schedule and schedule and i < len(schedule):
                g = schedule[i]
            else:
                g = fcf_growth_val

            if i == 0:
                fcf_i = base_fcf * (1 + g)
            else:
                fcf_i = prev_fcf * (1 + g)
            prev_fcf = fcf_i

            # Mid-year Convention: (i + 0.5) 적용
            pv_i = fcf_i / ((1 + wacc_val) ** (i + 0.5))
            pv_sum += pv_i

        # Terminal Value
        tv_calc = prev_fcf * (1 + tgr_val) / (wacc_val - tgr_val)
        pv_tv_calc = tv_calc / ((1 + wacc_val) ** projection_years)

        # Equity Value (Minority Interest, Preferred Stock 차감)
        ev_calc = pv_sum + pv_tv_calc
        equity_calc = ev_calc + cash - debt - minority_interest - preferred_stock
        price_calc = equity_calc / shares if shares > 0 else 0
        return price_calc

    def calc_dcf_value(wacc_val, tgr_val):
        """Smart Default 모드면 schedule 사용"""
        return calc_dcf_value_full(wacc_val, tgr_val, growth_dec, use_decay_schedule, growth_schedule)

    # Heatmap 데이터 생성
    z_values = []
    z_text = []
    for wacc_val in wacc_range:
        row_values = []
        row_text = []
        for tgr_val in growth_range:
            val = calc_dcf_value(wacc_val, tgr_val)
            if val is not None:
                row_values.append(val)
                row_text.append(f"${val:.0f}")
            else:
                row_values.append(None)
                row_text.append("N/A")
        z_values.append(row_values)
        z_text.append(row_text)

    # X, Y 라벨
    x_labels = [f"{g*100:.1f}%" for g in growth_range]
    y_labels = [f"{w*100:.1f}%" for w in wacc_range]

    # Base Case 라벨 (중앙)
    base_x_label = x_labels[2]
    base_y_label = y_labels[2]

    # Plotly Heatmap
    fig_sens = go.Figure()

    # Heatmap
    fig_sens.add_trace(go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        text=z_text,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "white"},
        colorscale='RdYlGn',
        colorbar=dict(title="Fair Value", tickformat="$,.0f"),
        hovertemplate="WACC: %{y}<br>Terminal Growth: %{x}<br>Fair Value: %{text}<extra></extra>",
        xgap=2,
        ygap=2
    ))

    # Base Case 마커를 Scatter로 표시 (더 안정적)
    fig_sens.add_trace(go.Scatter(
        x=[base_x_label],
        y=[base_y_label],
        mode='markers',
        marker=dict(
            symbol='square',
            size=45,
            color='rgba(0,0,0,0)',
            line=dict(color='blue', width=3)
        ),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig_sens.update_layout(
        title=dict(text="DCF Fair Value Matrix", font=dict(size=16)),
        xaxis_title="Terminal Growth Rate",
        yaxis_title="WACC",
        xaxis=dict(type='category', categoryorder='array', categoryarray=x_labels),
        yaxis=dict(type='category', categoryorder='array', categoryarray=y_labels),
        height=320,
        margin=dict(t=50, b=60, l=70, r=80)
    )

    st.plotly_chart(fig_sens, use_container_width=True)

    # 범례 설명
    st.caption(f"◼ **Base Case**: WACC={disc_dec*100:.1f}%, TGR={perp_dec*100:.1f}% → **${dcf_price:.2f}**")

    # ===== Bull / Base / Bear Scenario Table =====
    st.divider()
    st.subheader("🎯 Bull / Base / Bear Scenarios")

    # 시나리오 파라미터 정의
    # Base: 현재 설정값
    # Bull: 성장률 +20%, WACC -1%p, Exit Multiple +2x
    # Bear: 성장률 -30%, WACC +1%p, Exit Multiple -2x

    bull_growth = growth_dec * 1.20  # 20% 상향
    bear_growth = growth_dec * 0.70  # 30% 하향

    bull_wacc = max(disc_dec - 0.01, 0.04)  # 1%p 하향
    bear_wacc = disc_dec + 0.01  # 1%p 상향

    bull_perp = min(perp_dec + 0.005, rf_rate)  # 0.5%p 상향 (Rf 이하)
    bear_perp = max(perp_dec - 0.005, 0.01)  # 0.5%p 하향

    bull_exit = exit_multiple + 2.0  # 2x 상향
    bear_exit = max(exit_multiple - 2.0, 5.0)  # 2x 하향 (최소 5x)

    def calc_scenario_price(wacc_val, tgr_val, growth_val, exit_mult):
        """시나리오별 DCF 가치 계산"""
        if wacc_val <= tgr_val:
            return None, None

        # FCF 프로젝션
        pv_sum = 0
        prev_fcf = base_fcf
        for i in range(projection_years):
            if i == 0:
                fcf_i = base_fcf * (1 + growth_val)
            else:
                # Decay 적용 (선형 감소)
                decay_rate = growth_val - (growth_val - tgr_val) * (i / (projection_years - 1)) if projection_years > 1 else growth_val
                fcf_i = prev_fcf * (1 + decay_rate)
            prev_fcf = fcf_i
            pv_i = fcf_i / ((1 + wacc_val) ** (i + 0.5))  # Mid-year
            pv_sum += pv_i

        # Terminal Value - Perpetuity
        tv_perp = prev_fcf * (1 + tgr_val) / (wacc_val - tgr_val)
        pv_tv_perp = tv_perp / ((1 + wacc_val) ** projection_years)

        # Terminal Value - Exit Multiple
        ebitda_growth = prev_fcf / base_fcf if base_fcf > 0 else 1
        final_ebitda = (data.get('ebitda', 0) or 0) * ebitda_growth
        tv_exit = final_ebitda * exit_mult if final_ebitda > 0 else 0
        pv_tv_exit = tv_exit / ((1 + wacc_val) ** projection_years)

        # Blended
        if tv_method == "Perpetuity Growth":
            pv_tv = pv_tv_perp
        elif tv_method == "Exit Multiple":
            pv_tv = pv_tv_exit
        else:
            pv_tv = (pv_tv_perp + pv_tv_exit) / 2 if pv_tv_exit > 0 else pv_tv_perp

        ev = pv_sum + pv_tv
        eq = ev + cash - debt - minority_interest - preferred_stock
        price = eq / shares if shares > 0 else 0

        upside = (price / current_price - 1) * 100 if current_price > 0 else 0
        return price, upside

    # 시나리오별 계산
    bull_price, bull_upside = calc_scenario_price(bull_wacc, bull_perp, bull_growth, bull_exit)
    base_price, base_upside = dcf_price, mos_pct  # 이미 계산된 값
    bear_price, bear_upside = calc_scenario_price(bear_wacc, bear_perp, bear_growth, bear_exit)

    # 확률 가중 기대값 (간단한 가중치: Bull 25%, Base 50%, Bear 25%)
    if bull_price and bear_price:
        expected_price = bull_price * 0.25 + base_price * 0.50 + bear_price * 0.25
        expected_upside = (expected_price / current_price - 1) * 100 if current_price > 0 else 0
    else:
        expected_price = base_price
        expected_upside = base_upside

    # 시나리오 테이블
    scenario_col1, scenario_col2 = st.columns([2, 1])

    with scenario_col1:
        scenario_data = {
            'Scenario': ['🐻 Bear', '📊 Base', '🐂 Bull', '⚖️ Expected'],
            'Growth': [
                f'{bear_growth*100:.1f}%',
                f'{growth_dec*100:.1f}%',
                f'{bull_growth*100:.1f}%',
                '-'
            ],
            'WACC': [
                f'{bear_wacc*100:.1f}%',
                f'{disc_dec*100:.1f}%',
                f'{bull_wacc*100:.1f}%',
                '-'
            ],
            'Fair Value': [
                f'${bear_price:.2f}' if bear_price else 'N/A',
                f'${base_price:.2f}',
                f'${bull_price:.2f}' if bull_price else 'N/A',
                f'${expected_price:.2f}'
            ],
            'Upside': [
                f'{bear_upside:+.1f}%' if bear_upside else 'N/A',
                f'{base_upside:+.1f}%',
                f'{bull_upside:+.1f}%' if bull_upside else 'N/A',
                f'{expected_upside:+.1f}%'
            ]
        }
        scenario_df = pd.DataFrame(scenario_data)
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    with scenario_col2:
        # Risk/Reward 비율
        if bull_price and bear_price and current_price > 0:
            upside_potential = bull_price - current_price
            downside_risk = current_price - bear_price
            risk_reward = upside_potential / downside_risk if downside_risk > 0 else float('inf')

            st.markdown("##### Risk/Reward")
            if risk_reward > 2:
                rr_color, rr_label = "#10b981", "Favorable"
            elif risk_reward > 1:
                rr_color, rr_label = "#f59e0b", "Balanced"
            else:
                rr_color, rr_label = "#ef4444", "Unfavorable"

            st.markdown(f"""
            <div style="background:{rr_color}20; padding:15px; border-radius:8px; border-left:4px solid {rr_color};">
                <h2 style="margin:0; color:{rr_color};">{risk_reward:.2f}x</h2>
                <p style="margin:5px 0 0 0; color:{rr_color};">{rr_label}</p>
                <hr style="margin:10px 0;">
                <small>Upside: +${upside_potential:.2f}</small><br>
                <small>Downside: -${downside_risk:.2f}</small>
            </div>
            """, unsafe_allow_html=True)

    st.caption("💡 **Expected Value** = Bull(25%) + Base(50%) + Bear(25%) 가중 평균")

# ============================================================
# TAB 2: Relative Valuation
# ============================================================
with tab2:
    current_price = data.get('current_price', 0)
    trailing_eps = data.get('eps', 0)
    forward_eps = data.get('forward_eps', 0)

    # ===== Section 1: Historical Valuation =====
    hist_header_col1, hist_header_col2 = st.columns([3, 1])
    with hist_header_col1:
        st.subheader("📊 Historical Valuation")
    with hist_header_col2:
        hist_period = st.selectbox(
            "Period",
            options=[5, 3],
            format_func=lambda x: f"{x}Y",
            key="hist_val_period",
            label_visibility="collapsed"
        )

    # Historical 데이터 가져오기 (period 포함)
    hist_val_key = f"{ticker}_{hist_period}"
    if 'hist_val' not in st.session_state or st.session_state.get('hist_val_key') != hist_val_key:
        with st.spinner("Loading historical valuation data..."):
            hist_val = get_historical_valuation(ticker, hist_period)
            st.session_state['hist_val'] = hist_val
            st.session_state['hist_val_key'] = hist_val_key
    else:
        hist_val = st.session_state['hist_val']

    if 'error' not in hist_val and hist_val.get('data_points', 0) > 0:
        pe_data = hist_val['pe']
        pb_data = hist_val['pb']
        fwd_pe_data = hist_val['forward_pe']

        # PEG 계산 - Forward P/E + Analyst FY1 Growth (Finviz/Nasdaq 방식)
        forward_pe = data.get('forward_pe', 0) or 0

        # Analyst FY1 성장률 가져오기
        analyst_est = get_analyst_estimates(ticker)
        eps_growth_rate = analyst_est.get('fy1_growth', 0) or 0

        # fy1_growth는 소수점 형태 (0.05 = 5%), % 변환 필요
        if eps_growth_rate and abs(eps_growth_rate) < 1:
            eps_growth_rate = eps_growth_rate * 100

        if eps_growth_rate > 0 and forward_pe > 0:
            peg_ratio = forward_pe / eps_growth_rate
        else:
            peg_ratio = None

        # PE Band 카드
        col1, col2, col3 = st.columns(3)

        with col1:
            # PE 분석
            pe_vs_avg = pe_data['vs_avg_pct']
            if pe_vs_avg < -10:
                pe_status = "Below Average"
                pe_color = "#22c55e"
            elif pe_vs_avg > 10:
                pe_status = "Above Average"
                pe_color = "#ef4444"
            else:
                pe_status = "Near Average"
                pe_color = "#f59e0b"

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1));
                        padding: 20px; border-radius: 12px; border-left: 5px solid #667eea; height: 100%;">
                <h4 style="margin:0 0 15px 0;">P/E Ratio Band ({hist_period}Y)</h4>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span>Low: <b>{pe_data['low']:.1f}x</b></span>
                    <span>Avg: <b>{pe_data['avg']:.1f}x</b></span>
                    <span>High: <b>{pe_data['high']:.1f}x</b></span>
                </div>
                <div style="background: #e5e7eb; border-radius: 10px; height: 24px; position: relative; margin: 15px 0;">
                    <div style="position: absolute; left: 50%; transform: translateX(-50%); width: 3px;
                                height: 100%; background: #667eea; border-radius: 3px;"></div>
                    <div style="position: absolute; left: {min(max((pe_data.get('annual', pe_data['current']) - pe_data['low']) / (pe_data['high'] - pe_data['low']) * 100, 0), 100):.0f}%;
                                transform: translateX(-50%); width: 16px; height: 24px;
                                background: {pe_color}; border-radius: 4px;"></div>
                </div>
                <div style="text-align: center; margin-top: 10px;">
                    <span style="font-size: 1.5rem; font-weight: bold; color: {pe_color};">{pe_data.get('annual', pe_data['current']):.1f}x</span>
                    <span style="font-size: 0.85rem; color: #888;">(Annual)</span>
                    <br><span style="font-size: 0.85rem; color: #666;">{pe_data['percentile']:.0f}th Percentile</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Forward PE vs Trailing PE
            if fwd_pe_data['current'] > 0:
                fwd_vs_trailing = fwd_pe_data['vs_trailing']
                if fwd_vs_trailing < -10:
                    fwd_msg = "Growth Expected (Fwd PE lower)"
                    fwd_color = "#22c55e"
                elif fwd_vs_trailing > 10:
                    fwd_msg = "Earnings Decline Expected"
                    fwd_color = "#ef4444"
                else:
                    fwd_msg = "Stable Earnings Expected"
                    fwd_color = "#6b7280"

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(5,150,105,0.1));
                            padding: 20px; border-radius: 12px; border-left: 5px solid #10b981; height: 100%;">
                    <h4 style="margin:0 0 15px 0;">Trailing vs Forward P/E</h4>
                    <div style="display: flex; justify-content: space-around; text-align: center;">
                        <div>
                            <div style="font-size: 0.85rem; color: #666;">TTM P/E</div>
                            <div style="font-size: 1.8rem; font-weight: bold; color: #667eea;">{pe_data['current']:.1f}x</div>
                            <div style="font-size: 0.75rem; color: #999;">Annual: {pe_data.get('annual', pe_data['current']):.1f}x</div>
                        </div>
                        <div style="font-size: 2rem; color: #ccc; align-self: center;">→</div>
                        <div>
                            <div style="font-size: 0.85rem; color: #666;">Forward P/E</div>
                            <div style="font-size: 1.8rem; font-weight: bold; color: #10b981;">{fwd_pe_data['current']:.1f}x</div>
                        </div>
                    </div>
                    <div style="text-align: center; margin-top: 15px;">
                        <span style="color: {fwd_color}; font-size: 0.9rem;">{fwd_msg}<br>({fwd_vs_trailing:+.1f}%)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Forward P/E 데이터 없음")

        with col3:
            # PEG 분석 카드
            if peg_ratio is not None:
                # PEG 해석
                if peg_ratio < 0.5:
                    peg_status = "Significantly Undervalued"
                    peg_color = "#059669"  # 진한 초록
                    peg_emoji = "🔥"
                elif peg_ratio < 1:
                    peg_status = "Undervalued (GARP)"
                    peg_color = "#22c55e"  # 초록
                    peg_emoji = "✅"
                elif peg_ratio <= 1.5:
                    peg_status = "Fair Valued"
                    peg_color = "#f59e0b"  # 노랑
                    peg_emoji = "⚖️"
                elif peg_ratio <= 2:
                    peg_status = "Modestly Overvalued"
                    peg_color = "#f97316"  # 주황
                    peg_emoji = "⚠️"
                else:
                    peg_status = "Overvalued vs Growth"
                    peg_color = "#ef4444"  # 빨강
                    peg_emoji = "🚨"

                # PE가 높아도 PEG가 낮으면 긍정적 메시지
                if pe_data['current'] > pe_data['avg'] and peg_ratio < 1:
                    insight_msg = "High P/E but strong growth justifies premium"
                    insight_color = "#059669"
                elif pe_data['current'] < pe_data['avg'] and peg_ratio < 1:
                    insight_msg = "Low P/E + Low PEG = Strong Value"
                    insight_color = "#059669"
                elif peg_ratio > 2:
                    insight_msg = "Growth doesn't justify current valuation"
                    insight_color = "#ef4444"
                else:
                    insight_msg = f"Fwd P/E {forward_pe:.1f}x ÷ Growth {eps_growth_rate:.1f}%"
                    insight_color = "#6b7280"

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(168,85,247,0.1), rgba(139,92,246,0.1));
                            padding: 20px; border-radius: 12px; border-left: 5px solid #a855f7; height: 100%;">
                    <h4 style="margin:0 0 15px 0;">PEG Ratio</h4>
                    <div style="text-align: center;">
                        <div style="font-size: 2.5rem; font-weight: bold; color: {peg_color};">{peg_ratio:.2f}x</div>
                        <div style="font-size: 1.1rem; color: {peg_color}; margin: 5px 0;">{peg_emoji} {peg_status}</div>
                    </div>
                    <div style="background: #f3f4f6; padding: 10px; border-radius: 8px; margin-top: 12px;">
                        <div style="font-size: 0.8rem; color: #666; text-align: center;">
                            <span style="color: {insight_color}; font-weight: 500;">{insight_msg}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # PEG 계산 불가 (성장률 음수 등)
                if eps_growth_rate <= 0:
                    reason = "Negative or zero EPS growth"
                else:
                    reason = "P/E data unavailable"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(168,85,247,0.1), rgba(139,92,246,0.1));
                            padding: 20px; border-radius: 12px; border-left: 5px solid #a855f7; height: 100%;">
                    <h4 style="margin:0 0 15px 0;">PEG Ratio</h4>
                    <div style="text-align: center; padding: 20px 0;">
                        <div style="font-size: 1.2rem; color: #888;">N/A</div>
                        <div style="font-size: 0.85rem; color: #666; margin-top: 5px;">{reason}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.write("")  # 간격 추가

        # PE Band Chart (Trailing P/E only)
        if pe_data['history']:
            with st.expander(f"📈 P/E Trend Chart ({hist_period}Y)", expanded=False):
                dates = [p['date'] for p in pe_data['history']]
                pe_values = [p['pe'] for p in pe_data['history']]

                fig_pe = go.Figure()
                fig_pe.add_trace(go.Scatter(
                    x=dates, y=pe_values,
                    mode='lines',
                    name='Trailing P/E',
                    line=dict(color='#667eea', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)'
                ))
                # Average line
                fig_pe.add_hline(y=pe_data['avg'], line_dash="dash", line_color="#f59e0b",
                                annotation_text=f"{hist_period}Y Avg: {pe_data['avg']:.1f}x")
                # Current marker (TTM 기준)
                fig_pe.add_hline(y=pe_data['current'], line_dash="dot", line_color="#22c55e",
                                annotation_text=f"Current (TTM): {pe_data['current']:.1f}x", annotation_position="bottom right")

                fig_pe.update_layout(
                    title=f"{ticker} Historical P/E (Annual EPS)",
                    xaxis_title="",
                    yaxis_title="P/E Ratio",
                    height=300,
                    margin=dict(t=40, b=40),
                    showlegend=False
                )
                st.plotly_chart(fig_pe, use_container_width=True)

        # PEG 인사이트 박스
        if peg_ratio is not None:
            if peg_ratio < 1:
                peg_insight = f"💡 **PEG Insight**: Forward PEG {peg_ratio:.2f}x < 1 indicates stock may be **undervalued relative to growth**. Forward P/E ({forward_pe:.1f}x) justified by growth rate ({eps_growth_rate:.1f}%)."
                insight_type = "success"
            elif peg_ratio <= 1.5:
                peg_insight = f"💡 **PEG Insight**: Forward PEG {peg_ratio:.2f}x ≈ 1 suggests **fair valuation**. Forward P/E ({forward_pe:.1f}x) reflects expected earnings growth appropriately."
                insight_type = "info"
            else:
                peg_insight = f"⚠️ **PEG Insight**: Forward PEG {peg_ratio:.2f}x > 1.5 suggests stock may be **overvalued relative to growth**. Forward P/E ({forward_pe:.1f}x) not fully justified by growth rate ({eps_growth_rate:.1f}%)."
                insight_type = "warning"

            if insight_type == "success":
                st.success(peg_insight)
            elif insight_type == "warning":
                st.warning(peg_insight)
            else:
                st.info(peg_insight)

        # session_state에 저장
        if pe_data['avg'] > 0 and trailing_eps > 0:
            st.session_state['hist_pe_fair_value'] = pe_data['avg'] * trailing_eps
        else:
            st.session_state['hist_pe_fair_value'] = 0

    else:
        st.warning("⚠️ Historical valuation 데이터를 가져올 수 없습니다.")
        pe_data = {'current': data.get('pe_ratio', 0), 'avg': 0, 'high': 0, 'low': 0}

    st.divider()

    # ===== Section 2: Peer Comparison =====
    st.subheader("🏢 Peer Comparison")

    # Peer 자동 선정
    default_peers = get_peers(data.get('sector', 'Technology'), ticker)

    col1, col2 = st.columns([3, 1])
    with col1:
        use_custom = st.checkbox("Custom Peer Group", value=False, key="custom_peers")

    if use_custom:
        custom_input = st.text_input(
            "Enter tickers (comma-separated)",
            value=", ".join(default_peers[:5]),
            key="peer_input"
        )
        peer_tickers = [t.strip().upper() for t in custom_input.split(',') if t.strip()]
    else:
        peer_tickers = default_peers[:6]
        st.caption(f"Auto-selected peers ({data.get('sector', 'N/A')}): {', '.join(peer_tickers)}")

    # Peer 데이터 가져오기
    fetch_peers = st.button("📥 Fetch Peer Data", type="primary", key="fetch_peers")

    if fetch_peers or 'peer_data' in st.session_state:
        if fetch_peers:
            with st.spinner(f"Fetching {len(peer_tickers)} peers..."):
                peer_data = get_peer_group_data(tuple(peer_tickers))
                st.session_state['peer_data'] = peer_data
        else:
            peer_data = st.session_state['peer_data']

        if not peer_data:
            st.warning("⚠️ Peer 데이터를 가져올 수 없습니다.")
        else:
            # EPS Growth 계산
            if trailing_eps > 0 and forward_eps > 0:
                target_eps_growth = (forward_eps - trailing_eps) / trailing_eps
            else:
                target_eps_growth = data.get('earnings_growth', 0) or 0

            # PEG Ratio 계산
            target_pe = data.get('pe_ratio', 0)
            if target_pe > 0 and target_eps_growth > 0:
                target_peg = target_pe / (target_eps_growth * 100)
            else:
                target_peg = 0

            # Peer 데이터에 EPS Growth, PEG 추가
            for p in peer_data:
                p_trailing = p.get('eps', 0)
                p_forward = p.get('forward_eps', 0)
                if p_trailing > 0 and p_forward > 0:
                    p['eps_growth'] = (p_forward - p_trailing) / p_trailing
                else:
                    p['eps_growth'] = p.get('earnings_growth', 0) or 0

                p_pe = p.get('pe_ratio', 0)
                if p_pe > 0 and p.get('eps_growth', 0) > 0:
                    p['peg_ratio'] = p_pe / (p['eps_growth'] * 100)
                else:
                    p['peg_ratio'] = 0

            # Peer 평균 계산
            peer_pes = [p.get('pe_ratio', 0) for p in peer_data if p.get('pe_ratio', 0) > 0]
            peer_fwd_pes = [p.get('forward_pe', 0) for p in peer_data if p.get('forward_pe', 0) > 0]
            peer_pegs = [p.get('peg_ratio', 0) for p in peer_data if p.get('peg_ratio', 0) > 0]

            avg_peer_pe = sum(peer_pes) / len(peer_pes) if peer_pes else 0
            avg_peer_fwd_pe = sum(peer_fwd_pes) / len(peer_fwd_pes) if peer_fwd_pes else 0
            avg_peer_peg = sum(peer_pegs) / len(peer_pegs) if peer_pegs else 0

            # 비교 테이블
            st.markdown("**Valuation Multiples Comparison**")

            target_row = {
                'ticker': f"**{ticker}**",
                'pe_ratio': data.get('pe_ratio', 0),
                'forward_pe': data.get('forward_pe', 0),
                'peg_ratio': target_peg,
                'ev_ebitda': data.get('ev_ebitda', 0),
            }
            all_data = [target_row] + peer_data

            df = pd.DataFrame(all_data)
            display_df = df[['ticker', 'pe_ratio', 'forward_pe', 'peg_ratio', 'ev_ebitda']].copy()
            display_df.columns = ['Ticker', 'P/E', 'Fwd P/E', 'PEG', 'EV/EBITDA']

            display_df['P/E'] = display_df['P/E'].apply(lambda x: f"{x:.1f}x" if x > 0 else "-")
            display_df['Fwd P/E'] = display_df['Fwd P/E'].apply(lambda x: f"{x:.1f}x" if x > 0 else "-")
            display_df['PEG'] = display_df['PEG'].apply(lambda x: f"{x:.2f}" if x > 0 else "-")
            display_df['EV/EBITDA'] = display_df['EV/EBITDA'].apply(lambda x: f"{x:.1f}x" if x > 0 else "-")

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Peer 기반 Premium/Discount
            st.markdown("**vs Peer Average**")
            prem_cols = st.columns(4)
            with prem_cols[0]:
                if target_pe > 0 and avg_peer_pe > 0:
                    pe_prem = (target_pe / avg_peer_pe - 1) * 100
                    st.metric(f"P/E (Peer Avg: {avg_peer_pe:.1f}x)", f"{target_pe:.1f}x", f"{pe_prem:+.1f}%")
            with prem_cols[1]:
                target_fwd_pe = data.get('forward_pe', 0)
                if target_fwd_pe > 0 and avg_peer_fwd_pe > 0:
                    fwd_pe_prem = (target_fwd_pe / avg_peer_fwd_pe - 1) * 100
                    st.metric(f"Fwd P/E (Avg: {avg_peer_fwd_pe:.1f}x)", f"{target_fwd_pe:.1f}x", f"{fwd_pe_prem:+.1f}%")
            with prem_cols[2]:
                if target_peg > 0 and avg_peer_peg > 0:
                    peg_prem = (target_peg / avg_peer_peg - 1) * 100
                    st.metric(f"PEG (Avg: {avg_peer_peg:.2f})", f"{target_peg:.2f}", f"{peg_prem:+.1f}%")
            with prem_cols[3]:
                # Peer 기반 Implied Price
                if avg_peer_pe > 0 and trailing_eps > 0:
                    peer_implied = avg_peer_pe * trailing_eps
                    peer_upside = (peer_implied / current_price - 1) * 100 if current_price > 0 else 0
                    st.metric("Peer P/E Implied", f"${peer_implied:.2f}", f"{peer_upside:+.1f}%")
                    st.session_state['peer_result'] = {
                        'peer_fair_value': peer_implied,
                        'peer_avg_pe': avg_peer_pe,
                        'premium_discount': (target_pe / avg_peer_pe - 1) * 100 if avg_peer_pe > 0 else 0
                    }

    st.divider()

    # ===== Section 3: Valuation Simulator =====
    st.subheader("🎛️ Valuation Simulator")
    st.caption("💡 Target Price = Target P/E × Forward EPS (주가는 미래 실적을 반영)")

    # Analyst EPS 예상치 가져오기
    analyst_est = get_analyst_estimates(ticker)
    fy1_eps_default = analyst_est.get('fy1_eps', 0) or forward_eps
    fy1_eps_low = analyst_est.get('fy1_eps_low', 0)
    fy1_eps_high = analyst_est.get('fy1_eps_high', 0)
    fy1_growth = analyst_est.get('fy1_growth', 0) or 0.10
    num_analysts = analyst_est.get('num_analysts', 0)

    # FY2, FY3 기본값 계산
    fy2_eps_default = fy1_eps_default * (1 + fy1_growth) if fy1_eps_default > 0 else 0
    fy3_eps_default = fy2_eps_default * (1 + fy1_growth) if fy2_eps_default > 0 else 0

    # Target P/E 기본값: Historical Average (단순화)
    pe_default = pe_data.get('avg', 0) or fwd_pe_data.get('current', 0) or 20.0

    # P/E max_value 동적 설정 (기본값보다 높으면 여유 있게)
    pe_max = max(200.0, pe_default * 2.0)

    # 입력 영역
    st.markdown(f"**Assumptions** (Analyst: {num_analysts}, Est. Growth: {fy1_growth*100:.1f}%)")

    input_cols = st.columns(4)
    with input_cols[0]:
        target_pe = st.number_input(
            "Target P/E",
            min_value=5.0,
            max_value=pe_max,
            value=round(pe_default, 1),
            step=0.5,
            format="%.1f",
            key=f"sim_target_pe_{ticker}"
        )
        st.caption(f"**Default:** Hist {hist_period}Y Avg ({pe_data['avg']:.1f}x)")
        st.caption(f"Current: {fwd_pe_data['current']:.1f}x (Fwd) | {pe_data['current']:.1f}x (TTM)")

    with input_cols[1]:
        fy1_eps_input = st.number_input(
            "FY1 EPS ($)",
            min_value=0.0,
            max_value=500.0,
            value=round(fy1_eps_default, 2),
            step=0.1,
            format="%.2f",
            key=f"sim_fy1_eps_{ticker}"
        )
        # Analyst Est. 캡션 (Bull / Base / Bear)
        if fy1_eps_low > 0 and fy1_eps_high > 0:
            st.caption(f"Analyst: \\${fy1_eps_high:.2f}(Bull), \\${fy1_eps_default:.2f}(Base), \\${fy1_eps_low:.2f}(Bear)")
        else:
            st.caption(f"Analyst: \\${fy1_eps_default:.2f}")

    with input_cols[2]:
        fy2_eps_input = st.number_input(
            "FY2 EPS ($)",
            min_value=0.0,
            max_value=500.0,
            value=round(fy2_eps_default, 2),
            step=0.1,
            format="%.2f",
            key=f"sim_fy2_eps_{ticker}"
        )
        st.caption("Projected")

    with input_cols[3]:
        fy3_eps_input = st.number_input(
            "FY3 EPS ($)",
            min_value=0.0,
            max_value=500.0,
            value=round(fy3_eps_default, 2),
            step=0.1,
            format="%.2f",
            key=f"sim_fy3_eps_{ticker}"
        )
        st.caption("Projected")

    # Target Price 계산
    def calc_upside_color(upside):
        if upside > 15:
            return "#22c55e"
        elif upside < -15:
            return "#ef4444"
        else:
            return "#f59e0b"

    fy1_target = target_pe * fy1_eps_input
    fy2_target = target_pe * fy2_eps_input
    fy3_target = target_pe * fy3_eps_input

    fy1_upside = (fy1_target / current_price - 1) * 100 if current_price > 0 else 0
    fy2_upside = (fy2_target / current_price - 1) * 100 if current_price > 0 else 0
    fy3_upside = (fy3_target / current_price - 1) * 100 if current_price > 0 else 0

    fy1_color = calc_upside_color(fy1_upside)
    fy2_color = calc_upside_color(fy2_upside)
    fy3_color = calc_upside_color(fy3_upside)

    # 결과 테이블
    st.markdown("**Target Price by Year**")
    st.markdown(f"""
    <table style="width:100%; border-collapse:collapse; text-align:center; font-size:0.95em;">
        <thead>
            <tr style="background:#f8f9fa; border-bottom:2px solid #dee2e6;">
                <th style="padding:12px;">Year</th>
                <th style="padding:12px;">EPS</th>
                <th style="padding:12px;">× P/E</th>
                <th style="padding:12px;">Target Price</th>
                <th style="padding:12px;">Upside</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom:1px solid #dee2e6;">
                <td style="padding:12px; font-weight:600;">FY1 (1Y)</td>
                <td style="padding:12px;">${fy1_eps_input:.2f}</td>
                <td style="padding:12px;">{target_pe:.1f}x</td>
                <td style="padding:12px; font-weight:600;">${fy1_target:.2f}</td>
                <td style="padding:12px; color:{fy1_color}; font-weight:600;">{fy1_upside:+.1f}%</td>
            </tr>
            <tr style="border-bottom:1px solid #dee2e6;">
                <td style="padding:12px; font-weight:600;">FY2 (2Y)</td>
                <td style="padding:12px;">${fy2_eps_input:.2f}</td>
                <td style="padding:12px;">{target_pe:.1f}x</td>
                <td style="padding:12px; font-weight:600;">${fy2_target:.2f}</td>
                <td style="padding:12px; color:{fy2_color}; font-weight:600;">{fy2_upside:+.1f}%</td>
            </tr>
            <tr>
                <td style="padding:12px; font-weight:600;">FY3 (3Y)</td>
                <td style="padding:12px;">${fy3_eps_input:.2f}</td>
                <td style="padding:12px;">{target_pe:.1f}x</td>
                <td style="padding:12px; font-weight:600;">${fy3_target:.2f}</td>
                <td style="padding:12px; color:{fy3_color}; font-weight:600;">{fy3_upside:+.1f}%</td>
            </tr>
        </tbody>
    </table>
    <p style="text-align:right; color:#888; font-size:0.8em; margin-top:8px;">
        Current Price: ${current_price:.2f} |
        <span style="color:#22c55e;">■</span> &gt;15% Undervalued |
        <span style="color:#f59e0b;">■</span> Fair Value |
        <span style="color:#ef4444;">■</span> &lt;-15% Overvalued
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    # ===== Section 4: Bull / Base / Bear Scenarios =====
    st.subheader("🎯 Bull / Base / Bear Scenarios")

    # PE 기본값: Historical Percentile 기반 (p75 / avg / p25)
    pe_bull_default = pe_data.get('p75', 0) or pe_data.get('avg', 20) * 1.15
    pe_base_default = pe_data.get('avg', 0) or 20.0
    pe_bear_default = pe_data.get('p25', 0) or pe_data.get('avg', 20) * 0.85

    # EPS 기본값 (Base = Analyst Avg, Bull = High, Bear = Low)
    eps_base_default = analyst_est.get('fy1_eps', 0) or forward_eps
    eps_bull_default = analyst_est.get('fy1_eps_high', 0) or eps_base_default
    eps_bear_default = analyst_est.get('fy1_eps_low', 0) or eps_base_default

    # fallback: high/low가 0이면 base에서 ±10% 적용
    if eps_bull_default <= eps_base_default:
        eps_bull_default = eps_base_default * 1.10
    if eps_bear_default <= 0 or eps_bear_default >= eps_base_default:
        eps_bear_default = eps_base_default * 0.90

    if pe_base_default > 0 and eps_base_default > 0:
        # 캡션: 기본값 설명
        st.caption(f"💡 P/E 기본값 (Hist {hist_period}Y): Bull={pe_bull_default:.1f}x (상위 25%), Base={pe_base_default:.1f}x (avg), Bear={pe_bear_default:.1f}x (하위 25%)")
        st.caption(f"💡 EPS 기본값 (FY1 Analyst): Bull=\\${eps_bull_default:.2f} (High), Base=\\${eps_base_default:.2f} (Avg), Bear=\\${eps_bear_default:.2f} (Low)")

        # 입력 UI - 상단에 6개 컬럼으로 배치
        input_cols = st.columns(6)
        with input_cols[0]:
            pe_bull = st.number_input("🐂 Bull P/E", min_value=1.0, max_value=500.0, value=round(pe_bull_default, 1), step=0.5, format="%.1f", key=f"scenario_pe_bull_{ticker}")
        with input_cols[1]:
            eps_bull = st.number_input("🐂 Bull EPS", min_value=0.0, max_value=500.0, value=round(eps_bull_default, 2), step=0.1, format="%.2f", key=f"scenario_eps_bull_{ticker}")
        with input_cols[2]:
            pe_base = st.number_input("📊 Base P/E", min_value=1.0, max_value=500.0, value=round(pe_base_default, 1), step=0.5, format="%.1f", key=f"scenario_pe_base_{ticker}")
        with input_cols[3]:
            eps_base = st.number_input("📊 Base EPS", min_value=0.0, max_value=500.0, value=round(eps_base_default, 2), step=0.1, format="%.2f", key=f"scenario_eps_base_{ticker}")
        with input_cols[4]:
            pe_bear = st.number_input("🐻 Bear P/E", min_value=1.0, max_value=500.0, value=round(pe_bear_default, 1), step=0.5, format="%.1f", key=f"scenario_pe_bear_{ticker}")
        with input_cols[5]:
            eps_bear = st.number_input("🐻 Bear EPS", min_value=0.0, max_value=500.0, value=round(eps_bear_default, 2), step=0.1, format="%.2f", key=f"scenario_eps_bear_{ticker}")

        # 계산
        bull_price = pe_bull * eps_bull
        base_price = pe_base * eps_base
        bear_price = pe_bear * eps_bear

        bull_upside = (bull_price / current_price - 1) * 100 if current_price > 0 else 0
        base_upside = (base_price / current_price - 1) * 100 if current_price > 0 else 0
        bear_upside = (bear_price / current_price - 1) * 100 if current_price > 0 else 0

        # 결과 테이블
        st.markdown("---")
        scenario_data = {
            'Scenario': ['🐂 Bull', '📊 Base', '🐻 Bear'],
            'P/E': [f'{pe_bull:.1f}x', f'{pe_base:.1f}x', f'{pe_bear:.1f}x'],
            'EPS': [f'${eps_bull:.2f}', f'${eps_base:.2f}', f'${eps_bear:.2f}'],
            'Target Price': [f'${bull_price:.2f}', f'${base_price:.2f}', f'${bear_price:.2f}'],
            'Upside': [f'{bull_upside:+.1f}%', f'{base_upside:+.1f}%', f'{bear_upside:+.1f}%']
        }
        scenario_df = pd.DataFrame(scenario_data)
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)

        # Base case를 relative_fair_value로 저장 (Summary 탭용)
        st.session_state['relative_fair_value'] = base_price
    else:
        st.info("시나리오 계산을 위한 데이터가 부족합니다.")

# ============================================================
# TAB 3: Summary (Football Field Chart)
# ============================================================
with tab3:
    # ===== Risk Scorecard Banner =====
    # WACC 값 가져오기 (Tab 1에서 계산된 값 또는 기본값)
    # Risk Scorecard (Tab 1에서 이미 계산됨, 재사용)
    if 'risk_scorecard' in st.session_state:
        risk_scorecard = st.session_state['risk_scorecard']
    else:
        # Fallback: Tab 1을 거치지 않은 경우
        wacc_for_risk = st.session_state.get('calculated_wacc', 0.10)
        earnings_surprises = get_earnings_history(ticker)
        risk_scorecard = generate_risk_scorecard(
            ticker=ticker,
            financial_data=data,
            wacc=wacc_for_risk,
            earnings_surprises=earnings_surprises
        )

    # Risk Level에 따른 색상
    bg_color, text_color = get_risk_color(risk_scorecard.risk_level)
    risk_emoji = get_risk_emoji(risk_scorecard.risk_level)

    # Risk Banner
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {bg_color}22, {bg_color}11);
                padding: 20px; border-radius: 12px; border: 2px solid {bg_color};
                margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 1.4rem; font-weight: bold; color: {bg_color};">
                    {risk_emoji} VALUE TRAP RISK: {risk_scorecard.risk_level.name}
                </span>
                <span style="font-size: 0.9rem; color: #666; margin-left: 15px;">
                    ({risk_scorecard.flags_triggered}/{risk_scorecard.total_flags} flags triggered)
                </span>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.85rem; color: {text_color};">{risk_scorecard.summary}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 개별 Risk Flags 표시
    with st.expander("📋 Risk Assessment Details", expanded=False):
        flag_cols = st.columns(5)

        for i, flag in enumerate(risk_scorecard.flags):
            with flag_cols[i % 5]:
                icon = get_flag_icon(flag)

                if flag.severity == "danger":
                    flag_bg = "#fee2e2"
                    flag_border = "#ef4444"
                elif flag.severity == "warning":
                    flag_bg = "#fef3c7"
                    flag_border = "#f59e0b"
                else:
                    flag_bg = "#dcfce7"
                    flag_border = "#22c55e"

                st.markdown(f"""
                <div style="background: {flag_bg}; padding: 12px; border-radius: 8px;
                            border-left: 4px solid {flag_border}; margin-bottom: 10px; min-height: 100px;">
                    <div style="font-size: 0.8rem; font-weight: bold; color: #374151;">
                        {icon} {flag.name}
                    </div>
                    <div style="font-size: 0.75rem; color: #6b7280; margin-top: 5px;">
                        {flag.message}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 권고사항
        if risk_scorecard.risk_level == RiskLevel.HIGH:
            st.error(f"⚠️ **Recommendation**: {risk_scorecard.recommendation}")
        elif risk_scorecard.risk_level == RiskLevel.MODERATE:
            st.warning(f"💡 **Recommendation**: {risk_scorecard.recommendation}")
        else:
            st.success(f"✅ **Recommendation**: {risk_scorecard.recommendation}")

    st.divider()

    # ===== Valuation Summary Section =====
    st.subheader("🎯 Valuation Summary - Football Field Chart")

    current_price = data.get('current_price', 0)

    # 데이터 수집
    valuation_ranges = []

    # 1. 52주 범위
    low_52 = data.get('52w_low', 0)
    high_52 = data.get('52w_high', 0)
    if low_52 > 0 and high_52 > 0:
        valuation_ranges.append({
            'category': '52-Week Range',
            'low': low_52,
            'mid': (low_52 + high_52) / 2,
            'high': high_52,
            'color': '#3b82f6'
        })

    # 2. 애널리스트 목표가
    target_low = data.get('target_low', 0)
    target_mean = data.get('target_mean', 0)
    target_high = data.get('target_high', 0)
    if target_low > 0 and target_high > 0:
        valuation_ranges.append({
            'category': 'Analyst Targets',
            'low': target_low,
            'mid': target_mean,
            'high': target_high,
            'color': '#8b5cf6'
        })

    # 3. DCF 결과 (Tab 1에서)
    if 'dcf_result' in st.session_state:
        dcf_price = st.session_state['dcf_result']['dcf_price']
        # Bull/Bear 시나리오 (±20%)
        valuation_ranges.append({
            'category': 'DCF Valuation',
            'low': dcf_price * 0.8,
            'mid': dcf_price,
            'high': dcf_price * 1.2,
            'color': '#10b981'
        })

    # 4. Relative Valuation 결과 (Tab 2 Bull/Base/Bear에서)
    if 'relative_fair_value' in st.session_state and st.session_state['relative_fair_value'] > 0:
        rel_price = st.session_state['relative_fair_value']
        valuation_ranges.append({
            'category': 'Relative Valuation',
            'low': rel_price * 0.85,
            'mid': rel_price,
            'high': rel_price * 1.15,
            'color': '#f59e0b'
        })

    if not valuation_ranges:
        st.warning("⚠️ 먼저 Tab 1 (DCF)와 Tab 2 (Relative Valuation)를 완료해주세요.")
        st.stop()

    # Football Field Chart
    fig = go.Figure()

    for i, item in enumerate(valuation_ranges):
        # Range bar
        fig.add_trace(go.Bar(
            name=item['category'],
            y=[item['category']],
            x=[item['high'] - item['low']],
            base=[item['low']],
            orientation='h',
            marker=dict(color=item['color'], opacity=0.6),
            text=[f"${item['mid']:.0f}"],
            textposition='inside',
            hovertemplate=f"Low: ${item['low']:.2f}<br>Mid: ${item['mid']:.2f}<br>High: ${item['high']:.2f}<extra></extra>"
        ))

        # Mid point marker
        fig.add_trace(go.Scatter(
            x=[item['mid']],
            y=[item['category']],
            mode='markers',
            marker=dict(size=12, color='white', line=dict(color=item['color'], width=2)),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Current price line
    fig.add_vline(
        x=current_price,
        line=dict(color='red', width=3, dash='dash'),
        annotation_text=f"Current: ${current_price:.2f}",
        annotation_position="top"
    )

    fig.update_layout(
        title=f"{ticker} Valuation Range Comparison",
        xaxis_title="Price ($)",
        yaxis_title="",
        height=400,
        showlegend=False,
        barmode='overlay'
    )

    st.plotly_chart(fig, use_container_width=True)

    # 요약 테이블
    st.divider()
    st.subheader("📋 Valuation Summary Table")

    summary_data = []
    for item in valuation_ranges:
        upside = (item['mid'] / current_price - 1) * 100 if current_price > 0 else 0
        summary_data.append({
            'Method': item['category'],
            'Low': f"${item['low']:.2f}",
            'Mid': f"${item['mid']:.2f}",
            'High': f"${item['high']:.2f}",
            'Upside/Downside': f"{upside:+.1f}%"
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # 종합 의견
    st.divider()

    if 'dcf_result' in st.session_state:
        dcf_price = st.session_state['dcf_result']['dcf_price']
        avg_target = target_mean if target_mean > 0 else dcf_price

        # 평균 적정가 (DCF + Relative Valuation + Analyst Targets)
        fair_values = [item['mid'] for item in valuation_ranges if item['category'] in ['DCF Valuation', 'Relative Valuation', 'Analyst Targets']]
        if fair_values:
            avg_fair = sum(fair_values) / len(fair_values)
            upside = (avg_fair / current_price - 1) * 100 if current_price > 0 else 0

            if upside > 20:
                verdict, color = "Strong Buy", "#10b981"
            elif upside > 5:
                verdict, color = "Buy", "#22c55e"
            elif upside > -5:
                verdict, color = "Hold", "#f59e0b"
            elif upside > -20:
                verdict, color = "Sell", "#f97316"
            else:
                verdict, color = "Strong Sell", "#ef4444"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Avg. Fair Value", f"${avg_fair:.2f}")
            with col3:
                st.markdown(f"""
                <div style="text-align:center; padding:10px; background:{color}22; border-radius:10px; border:2px solid {color};">
                    <h2 style="color:{color}; margin:0;">{verdict}</h2>
                    <p style="margin:5px 0;">{upside:+.1f}% Upside</p>
                </div>
                """, unsafe_allow_html=True)

st.divider()
st.caption(f"⚠️ 무료 API 사용 등으로 인해 수치가 정확하지 않으므로, 상세 수치 확인은 Finviz를 활용하고 이 앱은 간단한 계산에만 활용할 것. | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
