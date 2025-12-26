"""
Stock Valuation Pro - Simple DCF Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import get_stock_data

st.set_page_config(page_title="DCF Valuation", page_icon="📊", layout="wide")

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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Discounted Cash Flow Analysis Model</div>', unsafe_allow_html=True)

# ===== Sidebar =====
with st.sidebar:
    st.header("🔧 Stock Selection")
    ticker = st.text_input("Stock Ticker", value="MSFT").upper()
    fetch_btn = st.button("📥 Fetch Data", type="primary", use_container_width=True)

    if fetch_btn:
        with st.spinner(f"Fetching {ticker}..."):
            data, success = get_stock_data(ticker)
            if success:
                st.session_state['stock_data'] = data
                st.session_state['ticker'] = ticker
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
    3. **가정값 조절** → 실시간 재계산
    """)
    st.stop()

data = st.session_state['stock_data']
ticker = st.session_state.get('ticker', 'N/A')

# 기본 정보
col1, col2, col3, col4 = st.columns(4)
col1.metric("Stock", ticker)
col2.metric("Current Price", f"${data.get('current_price', 0):.2f}")
col3.metric("Market Cap", f"${data.get('market_cap', 0)/1e9:.1f}B")
col4.metric("Sector", data.get('sector', 'N/A'))

st.divider()

# ===== Historical FCF (TTM 포함) =====
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

# 오래된순 정렬
all_fcf_data = sorted(all_fcf_data, key=lambda x: x['year'])

# TTM FCF - 항상 추가 (가장 최신 데이터)
ttm_fcf = data.get('fcf', 0)

if ttm_fcf and ttm_fcf != 0:
    # TTM 항상 추가 (가장 최신 rolling 12개월 데이터)
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
    selected_years = st.selectbox("Period", options=year_options, index=len(year_options)-1, format_func=lambda x: f"{x}Y")

with col2:
    years_list = [fd['year'] for fd in all_fcf_data]
    st.caption(f"💡 Available: {available_years} years ({years_list[0]} ~ {years_list[-1]})")

# 선택된 기간 데이터
fcf_data = all_fcf_data[-selected_years:]

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

avg_growth = np.mean(growth_rates) if growth_rates else 0.10

# 테이블 표시 (thousands 단위 - 스크린샷 형식)
table_data = {
    '': ['Year', 'FCF (in thousands)', 'Growth']
}
for fd in fcf_data:
    g = fd.get('growth')
    g_str = f"{g*100:.1f}%" if g is not None else "-"
    table_data[fd['year']] = [
        fd['year'],
        f"{fd['fcf']/1e3:,.0f}",  # thousands 단위 (스크린샷 형식)
        g_str
    ]

st.dataframe(pd.DataFrame(table_data).set_index('').T, use_container_width=True)

# Raw FCF 값 표시 (디버깅용)
with st.expander("🔍 Raw Data (디버깅용)"):
    st.write(f"**TTM FCF (from yfinance):** ${ttm_fcf:,.0f}")
    st.write(f"**TTM FCF (in thousands):** {ttm_fcf/1e3:,.0f}")
    st.write(f"**Base FCF for projection:** ${fcf_data[-1]['fcf']:,.0f}")

# 평균 성장률
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

# ===== DCF 가정값 (입력 칸) =====
st.subheader("⚙️ DCF Assumptions")

col1, col2, col3 = st.columns(3)

with col1:
    growth_rate = st.number_input(
        "Growth Rate (%)",
        min_value=-20.0,
        max_value=50.0,
        value=round(avg_growth * 100, 2),
        step=0.5,
        format="%.2f"
    )
    st.markdown("""
    <div class="guide-text">
    💡 <b>가이드라인:</b><br>
    • 성숙 기업: 3-7%<br>
    • 성장 기업: 10-20%<br>
    • 고성장: 20%+
    </div>
    """, unsafe_allow_html=True)

with col2:
    perpetual_growth = st.number_input(
        "Perpetual Growth Rate (%)",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.1,
        format="%.1f"
    )
    st.markdown("""
    <div class="guide-text">
    💡 <b>가이드라인:</b><br>
    • 일반적: 2-3%<br>
    • GDP 성장률 수준<br>
    • 보수적 2%, 낙관적 3%
    </div>
    """, unsafe_allow_html=True)

with col3:
    discount_rate = st.number_input(
        "Discount Rate (%)",
        min_value=3.0,
        max_value=20.0,
        value=8.0,
        step=0.5,
        format="%.1f"
    )
    st.markdown("""
    <div class="guide-text">
    💡 <b>가이드라인:</b><br>
    • 대형 우량주: 6-8%<br>
    • 일반 기업: 8-10%<br>
    • 고위험: 10-15%
    </div>
    """, unsafe_allow_html=True)

# 변환
growth_dec = growth_rate / 100
perp_dec = perpetual_growth / 100
disc_dec = discount_rate / 100

# 검증
if disc_dec <= perp_dec:
    st.error("⚠️ Discount Rate는 Perpetual Growth Rate보다 커야 합니다!")
    st.stop()

st.divider()

# ===== DCF 계산 (10년 고정) =====
st.subheader("📊 Future Free Cash Flow Projections (10 Years)")

projection_years = 10

if base_fcf <= 0:
    st.error("⚠️ Base FCF가 0 이하입니다.")
    st.stop()

# 미래 FCF 계산
projections = []
for i in range(projection_years):
    year = base_year + i + 1
    fcf = base_fcf * ((1 + growth_dec) ** (i + 1))
    pv = fcf / ((1 + disc_dec) ** (i + 1))
    projections.append({'year': year, 'fcf': fcf, 'pv': pv})

# Terminal Value
final_fcf = projections[-1]['fcf']
tv = final_fcf * (1 + perp_dec) / (disc_dec - perp_dec)
pv_tv = tv / ((1 + disc_dec) ** projection_years)

# 테이블
proj_table = {
    '': ['Year', 'Future FCF', 'PV of FCF']
}
for i, p in enumerate(projections):
    proj_table[str(i+1)] = [
        str(p['year']),
        f"${p['fcf']/1e6:,.0f}M",
        f"${p['pv']/1e6:,.0f}M"
    ]
proj_table['TV'] = ['Terminal Value', f"${tv/1e6:,.0f}M", f"${pv_tv/1e6:,.0f}M"]

st.dataframe(pd.DataFrame(proj_table).set_index('').T, use_container_width=True)

# 계산
sum_pv_fcf = sum(p['pv'] for p in projections)  # FCF의 PV 합계
sum_pv = sum_pv_fcf + pv_tv  # 총 Enterprise Value
cash = data.get('cash', 0)
debt = data.get('total_debt', 0)
equity = sum_pv + cash - debt
shares = data.get('shares_outstanding', 1)
dcf_price = equity / shares if shares > 0 else 0
current_price = data.get('current_price', 0)

# 계산 과정 디버깅
with st.expander("📐 Calculation Details"):
    st.write(f"**Base FCF (TTM/Latest):** ${base_fcf/1e3:,.0f}K = ${base_fcf/1e9:.2f}B")
    st.write(f"**Growth Rate:** {growth_dec*100:.2f}%")
    st.write(f"**Discount Rate:** {disc_dec*100:.2f}%")
    st.write(f"**Perpetual Growth:** {perp_dec*100:.2f}%")
    st.write("---")
    st.write(f"**Sum of PV (10yr FCF):** ${sum_pv_fcf/1e9:.2f}B")
    st.write(f"**Terminal Value:** ${tv/1e9:.2f}B")
    st.write(f"**PV of Terminal Value:** ${pv_tv/1e9:.2f}B")
    st.write(f"**Enterprise Value (Sum):** ${sum_pv/1e9:.2f}B")
    st.write("---")
    st.write(f"**+ Cash:** ${cash/1e9:.2f}B")
    st.write(f"**- Debt:** ${debt/1e9:.2f}B")
    st.write(f"**= Equity Value:** ${equity/1e9:.2f}B")
    st.write("---")
    st.write(f"**Shares Outstanding:** {shares/1e9:.3f}B ({shares/1e6:,.0f}M)")
    st.write(f"**DCF Price:** ${equity/1e9:.2f}B / {shares/1e9:.3f}B = **${dcf_price:.2f}**")

st.divider()

# ===== 결과 =====
st.subheader("💰 Valuation Summary")

col1, col2 = st.columns(2)

with col1:
    summary_df = pd.DataFrame({
        'Item': ['Sum of PV(FCF)', 'Cash & Equivalents', 'Total Debt', 'Equity Value', 'Shares Outstanding'],
        'Value': [
            f"${sum_pv/1e6:,.0f}M",
            f"${cash/1e6:,.0f}M",
            f"${debt/1e6:,.0f}M",
            f"${equity/1e6:,.0f}M",
            f"{shares/1e6:,.0f}M"
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # TV 비중
    tv_pct = pv_tv / sum_pv * 100 if sum_pv > 0 else 0
    if tv_pct > 75:
        st.markdown(f'<div class="warning-box">⚠️ Terminal Value = {tv_pct:.0f}% (70% 이하 권장)</div>', unsafe_allow_html=True)
    else:
        st.info(f"Terminal Value = {tv_pct:.0f}% of total")

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
        <h2 style="margin:0;">DCF Price per Share</h2>
        <h1 style="margin:10px 0; color:#667eea;">${dcf_price:.2f}</h1>
        <hr>
        <p><b>Current Price:</b> ${current_price:.2f}</p>
        <p><b>Difference:</b> <span style="color:{color};">{diff:+.1f}%</span></p>
        <h3 style="color:{color};">{verdict}</h3>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ===== 민감도 분석 =====
st.subheader("🎯 Sensitivity Analysis (Growth Rate vs Discount Rate)")

g_range = [growth_dec + x for x in [-0.02, -0.01, 0, 0.01, 0.02]]
d_range = [disc_dec + x for x in [-0.01, -0.005, 0, 0.005, 0.01]]

sens = []
for d in d_range:
    row = {'WACC': f"{d*100:.1f}%"}
    for g in g_range:
        pv_sum = sum(base_fcf * ((1+g)**(i+1)) / ((1+d)**(i+1)) for i in range(projection_years))
        final = base_fcf * ((1+g)**projection_years)
        if d > perp_dec:
            t = final * (1+perp_dec) / (d - perp_dec)
            pv_t = t / ((1+d)**projection_years)
        else:
            pv_t = 0
        eq = pv_sum + pv_t + cash - debt
        pr = eq / shares if shares > 0 else 0
        row[f"g={g*100:.0f}%"] = f"${pr:.0f}"
    sens.append(row)

st.dataframe(pd.DataFrame(sens), use_container_width=True, hide_index=True)

st.divider()
st.caption(f"⚠️ For educational purposes only | {datetime.now().strftime('%Y-%m-%d %H:%M')}")