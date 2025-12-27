"""
Stock Valuation Pro - Multi-Method Valuation Dashboard
- Tab 1: DCF Valuation (절대가치)
- Tab 2: Peer Comparison (상대가치)
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
    get_peer_group_data as _get_peer_group_data, calculate_peer_relative_valuation
)
from dcf_model import WallStreetDCF

# Caching으로 Rate Limit 방지 (10분간 캐시)
@st.cache_data(ttl=600, show_spinner=False)
def get_stock_data(ticker: str):
    return _get_stock_data(ticker)

@st.cache_data(ttl=600, show_spinner=False)
def get_peer_group_data(peer_tickers: tuple):
    return _get_peer_group_data(list(peer_tickers))

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
tab1, tab2, tab3 = st.tabs(["📊 DCF Valuation", "📈 Peer Comparison", "🎯 Summary"])

# ============================================================
# TAB 1: DCF Valuation
# ============================================================
with tab1:
    st.subheader("📈 Historical Free Cash Flow")

    historical = data.get('historical_financials', [])

    # FCF 데이터 수집
    all_fcf_data = []
    for h in historical:
        year = h.get('year', '')
        fcf = h.get('fcf', 0)
        if fcf == 0:
            op_cf = h.get('operating_cf', 0)
            capex = h.get('capex', 0)
            fcf = op_cf - capex if op_cf > 0 else 0
        if fcf != 0:
            all_fcf_data.append({'year': str(year), 'fcf': fcf})

    all_fcf_data = sorted(all_fcf_data, key=lambda x: x['year'])

    # TTM FCF
    ttm_fcf = data.get('fcf', 0)
    if ttm_fcf and ttm_fcf != 0:
        all_fcf_data.append({'year': 'TTM', 'fcf': ttm_fcf})

    available_years = len(all_fcf_data)

    if available_years == 0:
        st.error("⚠️ FCF 데이터가 없습니다.")
        st.stop()

    # 기간 선택
    col1, col2 = st.columns([1, 3])
    with col1:
        max_years = min(available_years, 10)
        year_options = list(range(3, max_years + 1)) if max_years >= 3 else [max_years]
        selected_years = st.selectbox("Period", options=year_options, index=len(year_options)-1, format_func=lambda x: f"{x}Y", key="dcf_period")

    with col2:
        years_list = [fd['year'] for fd in all_fcf_data]
        st.caption(f"💡 Available: {available_years} years ({years_list[0]} ~ {years_list[-1]})")

    fcf_data = all_fcf_data[-selected_years:]

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

    # 성장률 계산
    growth_rates = []
    for i in range(1, len(fcf_data)):
        prev = fcf_data[i-1]['fcf']
        curr = fcf_data[i]['fcf']
        if prev > 0 and curr > 0:
            g = (curr - prev) / prev
            growth_rates.append(g)
            fcf_data[i]['growth'] = g
        else:
            fcf_data[i]['growth'] = None

    # 1. Historical FCF CAGR (복합성장률 - 단순평균보다 안정적)
    first_fcf = fcf_data[0]['fcf']
    last_fcf = fcf_data[-1]['fcf']
    n_years = len(fcf_data) - 1
    if first_fcf > 0 and last_fcf > 0 and n_years > 0:
        historical_fcf_cagr = (last_fcf / first_fcf) ** (1 / n_years) - 1
    else:
        historical_fcf_cagr = np.mean(growth_rates) if growth_rates else 0.10

    # 2. Revenue Growth (TTM - DCF에 적합)
    revenue_growth = data.get('revenue_growth', 0) or 0

    # 3. Historical Revenue CAGR (WallStreetDCF에서 계산)
    hist_avgs = dcf_model.get_historical_averages()
    revenue_cagr = hist_avgs.get('blended_growth', revenue_growth)

    avg_growth = historical_fcf_cagr  # 기본값

    # FCF 테이블
    table_data = {'': ['Year', 'FCF (in thousands)', 'Growth']}
    for fd in fcf_data:
        g = fd.get('growth')
        g_str = f"{g*100:.1f}%" if g is not None else "-"
        table_data[fd['year']] = [fd['year'], f"{fd['fcf']/1e3:,.0f}", g_str]

    st.dataframe(pd.DataFrame(table_data).set_index('').T, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <b>Average Growth Rate:</b> <span style="font-size:1.3rem; color:#667eea;">{avg_growth*100:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

    base_fcf = fcf_data[-1]['fcf']
    base_year_str = fcf_data[-1]['year']
    base_year = datetime.now().year if base_year_str == 'TTM' else int(base_year_str)

    st.divider()

    # ===== DCF 가정값 =====
    st.subheader("⚙️ DCF Assumptions")

    wacc_result = dcf_model.calculate_auto_wacc()
    auto_wacc = wacc_result['wacc'] * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        # 성장률 옵션들 (DCF에 적합한 지표들)
        fcf_cagr_pct = historical_fcf_cagr * 100
        revenue_ttm_pct = revenue_growth * 100
        revenue_cagr_pct = revenue_cagr * 100
        smart_growth_pct = smart_defaults['assumptions']['initial_growth'] * 100

        growth_source = st.radio(
            "Growth Rate Source",
            options=["Smart Default", "FCF CAGR", "Revenue Growth", "Revenue CAGR", "Manual"],
            index=0,
            horizontal=True,
            key="growth_source"
        )

        if growth_source == "Smart Default":
            default_growth = smart_growth_pct
            st.caption(f"🤖 Context-Aware: {smart_growth_pct:.1f}% (Lifecycle-based decay 적용)")
        elif growth_source == "FCF CAGR":
            default_growth = fcf_cagr_pct
            st.caption(f"📊 Historical FCF CAGR: {fcf_cagr_pct:.1f}%")
        elif growth_source == "Revenue Growth":
            default_growth = revenue_ttm_pct
            st.caption(f"📊 TTM Revenue Growth: {revenue_ttm_pct:.1f}%")
        elif growth_source == "Revenue CAGR":
            default_growth = revenue_cagr_pct
            st.caption(f"📊 3Y Revenue CAGR: {revenue_cagr_pct:.1f}%")
        else:
            default_growth = 10.0
            st.caption("✏️ Enter your estimate")

        growth_rate = st.number_input(
            "Growth Rate (%)",
            min_value=-50.0,
            max_value=150.0,
            value=round(min(max(default_growth, -50.0), 150.0), 2),
            step=1.0,
            format="%.2f",
            disabled=(growth_source != "Manual"),
            key="growth_rate"
        )

        if growth_source != "Manual":
            growth_rate = default_growth

        if growth_rate > 50:
            st.warning(f"⚠️ High Growth ({growth_rate:.1f}%)")

        # Growth Rate 가이드라인
        st.markdown(f"""
        <div class="guide-text">
        💡 <b>Available Data:</b><br>
        • 🤖 Smart Default: {smart_growth_pct:.1f}%<br>
        • FCF CAGR: {fcf_cagr_pct:.1f}%<br>
        • Revenue TTM: {revenue_ttm_pct:.1f}%<br>
        • Revenue CAGR: {revenue_cagr_pct:.1f}%
        </div>
        """, unsafe_allow_html=True)

    with col2:
        rf_rate = dcf_model.risk_free_rate
        perpetual_growth = st.number_input(
            "Perpetual Growth Rate (%)",
            min_value=0.0,
            max_value=5.0,
            value=2.5,
            step=0.1,
            format="%.1f",
            key="perp_growth"
        )
        if perpetual_growth / 100 > rf_rate:
            st.warning(f"⚠️ Risk-Free Rate({rf_rate*100:.1f}%) 초과!")
        st.markdown(f"""
        <div class="guide-text">
        💡 <b>Guideline:</b><br>
        • 장기 GDP 성장률 수준 (2~3%)<br>
        • Risk-Free Rate ({rf_rate*100:.1f}%) 이하 권장<br>
        • 인플레이션 고려 시 1.5~2.5%
        </div>
        """, unsafe_allow_html=True)

    with col3:
        use_auto_wacc = st.checkbox("Auto WACC", value=True, key="auto_wacc")

        if use_auto_wacc:
            discount_rate = st.number_input(
                "Discount Rate (WACC) (%)",
                min_value=3.0, max_value=20.0,
                value=round(auto_wacc, 2),
                step=0.5, format="%.2f",
                disabled=True,
                key="wacc_input"
            )
            discount_rate = auto_wacc
        else:
            discount_rate = st.number_input(
                "Discount Rate (WACC) (%)",
                min_value=3.0, max_value=20.0,
                value=8.0, step=0.5, format="%.1f",
                key="wacc_manual"
            )
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

    # Market Implied FCF Growth 계산 (사용자 입력값 반영)
    def calc_market_implied_growth(wacc_val, tgr_val):
        """현재 주가가 암시하는 FCF Growth Rate"""
        current_price = data.get('current_price', 0)
        shares = data.get('shares_outstanding', 1)
        cash = data.get('cash', 0)
        debt = data.get('total_debt', 0)
        # Lifecycle 기반 projection 기간 사용
        proj_years = lifecycle.projection_years if growth_source == "Smart Default" else 10

        if current_price <= 0 or base_fcf <= 0 or wacc_val <= tgr_val:
            return None

        low, high = -0.5, 2.0
        for _ in range(50):
            mid = (low + high) / 2
            pv_sum = 0
            prev_fcf = base_fcf
            for i in range(proj_years):
                if i == 0:
                    fcf_i = base_fcf * (1 + mid)
                else:
                    fcf_i = prev_fcf * (1 + mid)
                prev_fcf = fcf_i
                pv_sum += fcf_i / ((1 + wacc_val) ** (i + 1))

            tv = prev_fcf * (1 + tgr_val) / (wacc_val - tgr_val)
            pv_tv = tv / ((1 + wacc_val) ** proj_years)
            equity_val = pv_sum + pv_tv + cash - debt
            fair_price = equity_val / shares if shares > 0 else 0
            if fair_price < current_price:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    market_implied = calc_market_implied_growth(disc_dec, perp_dec)
    current_price = data.get('current_price', 0)

    # Market Implied 표시
    if market_implied is not None:
        implied_pct = market_implied * 100
        diff_vs_assumption = implied_pct - growth_rate
        if diff_vs_assumption > 5:
            implied_color = "#ef4444"  # 빨강 - 시장이 더 높은 성장 기대
            implied_msg = "시장이 더 높은 성장을 반영 중"
        elif diff_vs_assumption < -5:
            implied_color = "#22c55e"  # 초록 - 시장이 더 낮은 성장 기대
            implied_msg = "시장이 더 낮은 성장을 반영 중"
        else:
            implied_color = "#6b7280"  # 회색 - 비슷
            implied_msg = "가정과 유사"

        st.markdown(f"""
        <div style="background: linear-gradient(90deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1));
                    padding: 12px 16px; border-radius: 8px; margin: 10px 0;
                    border-left: 4px solid #667eea;">
            <span style="font-size: 0.9rem;">⭐ <b>Market Implied FCF Growth:</b></span>
            <span style="font-size: 1.2rem; font-weight: bold; color: {implied_color}; margin-left: 8px;">{implied_pct:.1f}%</span>
            <span style="font-size: 0.8rem; color: #888; margin-left: 12px;">
                (WACC={disc_dec*100:.1f}%, TGR={perp_dec*100:.1f}% 기준 | 현재가 ${current_price:.0f})
            </span>
            <br><span style="font-size: 0.75rem; color: {implied_color};">→ {implied_msg} (Your assumption: {growth_rate:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)

    if disc_dec <= perp_dec:
        st.error("⚠️ Discount Rate > Perpetual Growth Rate 필요!")
        st.stop()

    st.divider()

    # DCF 계산 - Lifecycle 기반 Projection Period
    projection_years = lifecycle.projection_years if growth_source == "Smart Default" else 10
    st.subheader(f"📊 Future FCF Projections ({projection_years} Years)")

    if base_fcf <= 0:
        st.error("⚠️ Base FCF가 0 이하입니다.")
        st.stop()

    projections = []
    # Smart Default 모드: Growth Decay Schedule 사용
    use_decay_schedule = (growth_source == "Smart Default")
    growth_schedule = smart_defaults['growth_schedule'] if use_decay_schedule else None

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

        pv = fcf / ((1 + disc_dec) ** (i + 1))
        projections.append({'year': year, 'fcf': fcf, 'pv': pv, 'growth': year_growth})

    final_fcf = projections[-1]['fcf']
    tv = final_fcf * (1 + perp_dec) / (disc_dec - perp_dec)
    pv_tv = tv / ((1 + disc_dec) ** projection_years)

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

    # 결과 계산
    sum_pv_fcf = sum(p['pv'] for p in projections)
    sum_pv = sum_pv_fcf + pv_tv
    cash = data.get('cash', 0)
    debt = data.get('total_debt', 0)
    equity = sum_pv + cash - debt
    shares = data.get('shares_outstanding', 1)
    dcf_price = equity / shares if shares > 0 else 0
    current_price = data.get('current_price', 0)

    # 결과를 session_state에 저장 (Tab 3에서 사용)
    st.session_state['dcf_result'] = {
        'dcf_price': dcf_price,
        'sum_pv': sum_pv,
        'pv_tv': pv_tv,
        'tv_pct': pv_tv / sum_pv * 100 if sum_pv > 0 else 0
    }

    st.divider()

    # 결과 표시
    st.subheader("💰 DCF Valuation Result")

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

        tv_pct = pv_tv / sum_pv * 100 if sum_pv > 0 else 0
        if tv_pct > 75:
            st.warning(f"⚠️ Terminal Value = {tv_pct:.0f}% (높음)")

    with col2:
        diff = (dcf_price / current_price - 1) * 100 if current_price > 0 else 0
        if diff > 15:
            verdict, color = "🟢 UNDERVALUED", "#10b981"
        elif diff > -15:
            verdict, color = "🟡 FAIR VALUE", "#f59e0b"
        else:
            verdict, color = "🔴 OVERVALUED", "#ef4444"

        st.markdown(f"""
        <div class="result-box">
            <h2 style="margin:0;">DCF Fair Value</h2>
            <h1 style="margin:10px 0; color:#667eea;">${dcf_price:.2f}</h1>
            <hr>
            <p><b>Current:</b> ${current_price:.2f} | <b>Diff:</b> <span style="color:{color};">{diff:+.1f}%</span></p>
            <h3 style="color:{color};">{verdict}</h3>
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
        """주어진 WACC, Terminal Growth, FCF Growth로 DCF 가치 계산"""
        if wacc_val <= tgr_val:
            return None

        # FCF 프로젝션 PV
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

            pv_i = fcf_i / ((1 + wacc_val) ** (i + 1))
            pv_sum += pv_i

        # Terminal Value
        tv_calc = prev_fcf * (1 + tgr_val) / (wacc_val - tgr_val)
        pv_tv_calc = tv_calc / ((1 + wacc_val) ** projection_years)

        # Equity Value
        ev_calc = pv_sum + pv_tv_calc
        equity_calc = ev_calc + cash - debt
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

# ============================================================
# TAB 2: Peer Comparison
# ============================================================
with tab2:
    st.subheader("🏢 Peer Group Selection")

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
            st.success(f"✅ {len(peer_data)} peers loaded")

            # Peer Comparison Table
            st.subheader("📊 Valuation Multiples Comparison")

            # EPS Growth 계산 (Forward EPS / Trailing EPS - 1)
            trailing_eps = data.get('eps', 0)
            forward_eps = data.get('forward_eps', 0)
            if trailing_eps > 0 and forward_eps > 0:
                target_eps_growth = (forward_eps - trailing_eps) / trailing_eps
            else:
                target_eps_growth = data.get('earnings_growth', 0) or 0

            # PEG Ratio 계산 (P/E / EPS Growth%)
            target_pe = data.get('pe_ratio', 0)
            if target_pe > 0 and target_eps_growth > 0:
                target_peg = target_pe / (target_eps_growth * 100)
            else:
                target_peg = 0

            # 타겟 기업 데이터 추가
            target_row = {
                'ticker': f"**{ticker}**",
                'name': data.get('name', ticker),
                'price': data.get('current_price', 0),
                'market_cap': data.get('market_cap', 0),
                'pe_ratio': data.get('pe_ratio', 0),
                'forward_pe': data.get('forward_pe', 0),
                'eps_growth': target_eps_growth,
                'peg_ratio': target_peg,
                'pb_ratio': data.get('pb_ratio', 0),
                'ev_ebitda': data.get('ev_ebitda', 0),
            }

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

            all_data = [target_row] + peer_data

            # DataFrame 생성
            df = pd.DataFrame(all_data)
            display_df = df[['ticker', 'name', 'price', 'market_cap', 'pe_ratio', 'forward_pe', 'eps_growth', 'peg_ratio', 'ev_ebitda']].copy()
            display_df.columns = ['Ticker', 'Company', 'Price', 'Market Cap', 'P/E', 'Fwd P/E', 'EPS Gr', 'PEG', 'EV/EBITDA']

            # 포맷팅
            display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}" if x > 0 else "-")
            display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: f"${x/1e9:.1f}B" if x > 0 else "-")
            display_df['P/E'] = display_df['P/E'].apply(lambda x: f"{x:.1f}x" if x > 0 else "-")
            display_df['Fwd P/E'] = display_df['Fwd P/E'].apply(lambda x: f"{x:.1f}x" if x > 0 else "-")
            display_df['EPS Gr'] = display_df['EPS Gr'].apply(lambda x: f"{x*100:.1f}%" if x != 0 else "-")
            display_df['PEG'] = display_df['PEG'].apply(lambda x: f"{x:.2f}" if x > 0 else "-")
            display_df['EV/EBITDA'] = display_df['EV/EBITDA'].apply(lambda x: f"{x:.1f}x" if x > 0 else "-")

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # PEG Ratio 가이드
            st.caption("💡 **PEG Ratio**: P/E ÷ EPS Growth%. PEG < 1 = 저평가, PEG > 2 = 고평가 (일반적 기준)")

            st.divider()

            # 상대가치 분석
            st.subheader("💹 Relative Valuation Analysis")

            relative = calculate_peer_relative_valuation(data, peer_data)

            if 'error' not in relative:
                # Peer 평균 PEG 계산
                peer_pegs = [p.get('peg_ratio', 0) for p in peer_data if p.get('peg_ratio', 0) > 0]
                avg_peg = sum(peer_pegs) / len(peer_pegs) if peer_pegs else 0

                # Peer 평균 EPS Growth 계산
                peer_eps_growths = [p.get('eps_growth', 0) for p in peer_data if p.get('eps_growth', 0) > 0]
                avg_eps_growth = sum(peer_eps_growths) / len(peer_eps_growths) if peer_eps_growths else 0

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Peer Average Multiples**")
                    avg = relative['peer_avg']
                    avg_df = pd.DataFrame({
                        'Multiple': ['P/E', 'Forward P/E', 'EPS Growth', 'PEG Ratio', 'EV/EBITDA'],
                        'Peer Avg': [
                            f"{avg['pe']:.1f}x" if avg['pe'] > 0 else "-",
                            f"{avg['forward_pe']:.1f}x" if avg['forward_pe'] > 0 else "-",
                            f"{avg_eps_growth*100:.1f}%" if avg_eps_growth > 0 else "-",
                            f"{avg_peg:.2f}" if avg_peg > 0 else "-",
                            f"{avg['ev_ebitda']:.1f}x" if avg['ev_ebitda'] > 0 else "-"
                        ],
                        f'{ticker}': [
                            f"{data.get('pe_ratio', 0):.1f}x" if data.get('pe_ratio', 0) > 0 else "-",
                            f"{data.get('forward_pe', 0):.1f}x" if data.get('forward_pe', 0) > 0 else "-",
                            f"{target_eps_growth*100:.1f}%" if target_eps_growth > 0 else "-",
                            f"{target_peg:.2f}" if target_peg > 0 else "-",
                            f"{data.get('ev_ebitda', 0):.1f}x" if data.get('ev_ebitda', 0) > 0 else "-"
                        ]
                    })
                    st.dataframe(avg_df, use_container_width=True, hide_index=True)

                with col2:
                    st.markdown("**Implied Fair Value**")
                    implied = relative['implied_values']
                    premium = relative['premium_discount']

                    # PEG 기반 적정가 계산 (Peer Avg PEG × Target EPS Growth × Target EPS)
                    peg_implied = 0
                    if avg_peg > 0 and target_eps_growth > 0 and trailing_eps > 0:
                        # Fair P/E = Avg PEG × EPS Growth%
                        fair_pe = avg_peg * (target_eps_growth * 100)
                        peg_implied = fair_pe * trailing_eps

                    implied_df = pd.DataFrame({
                        'Method': ['P/E Based', 'PEG Based', 'P/B Based'],
                        'Implied Price': [
                            f"${implied.get('pe_based', 0):.2f}" if implied.get('pe_based', 0) > 0 else "-",
                            f"${peg_implied:.2f}" if peg_implied > 0 else "-",
                            f"${implied.get('pb_based', 0):.2f}" if implied.get('pb_based', 0) > 0 else "-"
                        ],
                        'vs Current': [
                            f"{((implied.get('pe_based', 0) / current_price - 1) * 100):+.1f}%" if implied.get('pe_based', 0) > 0 and current_price > 0 else "-",
                            f"{((peg_implied / current_price - 1) * 100):+.1f}%" if peg_implied > 0 and current_price > 0 else "-",
                            f"{((implied.get('pb_based', 0) / current_price - 1) * 100):+.1f}%" if implied.get('pb_based', 0) > 0 and current_price > 0 else "-"
                        ]
                    })
                    st.dataframe(implied_df, use_container_width=True, hide_index=True)

                # 프리미엄/디스카운트 표시
                st.divider()
                st.markdown("**Premium / Discount vs Peers**")

                prem_cols = st.columns(4)
                if 'pe' in premium:
                    with prem_cols[0]:
                        pe_prem = premium['pe']
                        st.metric("P/E", f"{pe_prem:+.1f}%", delta=None)
                # PEG 프리미엄/디스카운트
                if target_peg > 0 and avg_peg > 0:
                    with prem_cols[1]:
                        peg_prem = (target_peg / avg_peg - 1) * 100
                        st.metric("PEG", f"{peg_prem:+.1f}%", delta=None)
                if 'pb' in premium:
                    with prem_cols[2]:
                        pb_prem = premium['pb']
                        st.metric("P/B", f"{pb_prem:+.1f}%", delta=None)
                if 'ev_ebitda' in premium:
                    with prem_cols[3]:
                        ev_prem = premium['ev_ebitda']
                        st.metric("EV/EBITDA", f"{ev_prem:+.1f}%", delta=None)

                # Peer 기반 적정주가를 session_state에 저장
                peer_fair_value = implied.get('pe_based', 0) if implied.get('pe_based', 0) > 0 else implied.get('pb_based', 0)
                st.session_state['peer_result'] = {
                    'peer_fair_value': peer_fair_value,
                    'peer_avg_pe': avg['pe'],
                    'premium_discount': premium.get('pe', 0)
                }

# ============================================================
# TAB 3: Summary (Football Field Chart)
# ============================================================
with tab3:
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

    # 4. Peer Comparison 결과 (Tab 2에서)
    if 'peer_result' in st.session_state and st.session_state['peer_result']['peer_fair_value'] > 0:
        peer_price = st.session_state['peer_result']['peer_fair_value']
        valuation_ranges.append({
            'category': 'Peer-Based (P/E)',
            'low': peer_price * 0.9,
            'mid': peer_price,
            'high': peer_price * 1.1,
            'color': '#f59e0b'
        })

    if not valuation_ranges:
        st.warning("⚠️ 먼저 Tab 1 (DCF)와 Tab 2 (Peer)를 완료해주세요.")
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

        # 평균 적정가
        fair_values = [item['mid'] for item in valuation_ranges if 'DCF' in item['category'] or 'Peer' in item['category']]
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
st.caption(f"⚠️ For educational purposes only | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
