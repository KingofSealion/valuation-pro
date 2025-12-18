"""
Stock Valuation Pro - Wall Street Edition (v3)
- 3Y CAGR 기반 Base Growth
- Bull/Bear는 2개 변수만 조절 (Revenue Growth, Exit Multiple)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import get_stock_data, get_risk_free_rate
from dcf_model import WallStreetDCF, DCFAssumptions, create_football_field_data, SECTOR_DEFAULTS

st.set_page_config(page_title="DCF Valuation Pro", page_icon="💰", layout="wide")

# 다크/라이트 모드 호환 CSS
st.markdown("""
<style>
    /* 메인 타이틀 - 테마 색상 사용 */
    .main-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: var(--text-color);
    }

    /* 정보 박스 - 파란색 계열 (테마 호환) */
    .info-box {
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 10px 0;
        background: rgba(59, 130, 246, 0.1);
        color: inherit;
    }

    /* 경고 박스 - 노란색 계열 (테마 호환) */
    .warning-box {
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 10px 0;
        background: rgba(245, 158, 11, 0.1);
        color: inherit;
    }

    /* 성공 박스 - 초록색 계열 (테마 호환) */
    .success-box {
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 10px 0;
        background: rgba(16, 185, 129, 0.1);
        color: inherit;
    }

    /* 에러 박스 - 빨간색 계열 (테마 호환) */
    .error-box {
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ef4444;
        margin: 10px 0;
        background: rgba(239, 68, 68, 0.1);
        color: inherit;
    }

    /* 강조 텍스트 */
    .highlight-green { color: #10b981; font-weight: bold; }
    .highlight-red { color: #ef4444; font-weight: bold; }
    .highlight-blue { color: #3b82f6; font-weight: bold; }
    .highlight-yellow { color: #f59e0b; font-weight: bold; }

    /* 다크모드에서 테이블 가독성 향상 */
    .stDataFrame {
        border-radius: 8px;
    }

    /* 메트릭 카드 스타일 */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">💰 Wall Street DCF Valuation</p>', unsafe_allow_html=True)
st.caption("3Y CAGR 기반 | Mid-year Convention | Bull/Bear = 비율 기반 조정")

# Sidebar
with st.sidebar:
    st.header("📊 Setup")
    ticker = st.text_input("Ticker", value="AAPL").upper()
    st.divider()
    projection_years = st.selectbox("Projection Years", [5, 7, 10], index=0)
    tax_rate = st.slider("Tax Rate", 0.15, 0.30, 0.21, 0.01, format="%.0f%%")
    st.divider()
    analyze_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

if analyze_btn:
    with st.spinner(f"Fetching {ticker}..."):
        data, success = get_stock_data(ticker)
    
    if not success:
        st.error(f"Error: {data.get('error')}")
        st.stop()
    
    # 기본 정보
    st.header(f"🏢 {data['name']} ({ticker})")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${data['current_price']:.2f}")
    col2.metric("Market Cap", f"${data['market_cap']/1e9:.1f}B")
    col3.metric("52W", f"${data['52w_low']:.0f} - ${data['52w_high']:.0f}")
    col4.metric("Sector", data['sector'])
    
    st.divider()
    
    # DCF 모델 초기화
    dcf_model = WallStreetDCF(data)
    hist_avg = dcf_model.get_historical_averages()
    sector_defaults = hist_avg.get('sector_defaults', {})
    
    # 탭
    tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Assumptions", "📈 DCF Model", "🎯 Sensitivity", "🏈 Football Field"])
    
    # ==================== TAB 1: Assumptions ====================
    with tab1:
        st.subheader("⚙️ DCF Assumptions")
        
        # ===== WACC =====
        st.markdown("### 💵 WACC")
        
        w1, w2, w3 = st.columns(3)
        
        with w1:
            risk_free = st.number_input("Risk-Free (%)", value=get_risk_free_rate()*100, step=0.1, format="%.2f") / 100
            beta = st.number_input("Beta", value=data.get('beta', 1.0), step=0.05, format="%.2f")
            erp = st.number_input("ERP (%)", value=5.5, step=0.1) / 100
        
        with w2:
            cost_of_debt = st.number_input("Cost of Debt (%)", value=5.0, step=0.1) / 100
            
            debt = data.get('total_debt', 0)
            mcap = data.get('market_cap', 0)
            total = debt + mcap
            
            d_wt = debt / total if total > 0 else 0.2
            e_wt = mcap / total if total > 0 else 0.8
            
            st.metric("Debt Weight", f"{d_wt*100:.1f}%")
            st.metric("Equity Weight", f"{e_wt*100:.1f}%")
        
        with w3:
            coe = risk_free + beta * erp
            wacc = (e_wt * coe) + (d_wt * cost_of_debt * (1 - tax_rate))
            
            st.metric("Cost of Equity", f"{coe*100:.2f}%")
            st.metric("**WACC**", f"{wacc*100:.2f}%")
        
        st.divider()
        
        # ===== Revenue Growth (★ 핵심) =====
        st.markdown("### 📈 Revenue Growth")
        
        cagr_3y = hist_avg.get('cagr_3y', 0)
        cagr_5y = hist_avg.get('cagr_5y', 0)
        base_growth = hist_avg.get('base_growth', 0.05)
        sector_cap = hist_avg.get('sector_cap', 0.15)
        
        blended_growth = hist_avg.get('blended_growth', 0)
        st.markdown(f"""
        <div class="info-box">
        <b>📊 Base Growth (회사 실제 데이터 기반)</b><br>
        • 3Y CAGR: <b>{cagr_3y*100:.1f}%</b><br>
        • 5Y CAGR: <b>{cagr_5y*100:.1f}%</b><br>
        • 가중평균: <b>{blended_growth*100:.1f}%</b><br>
        → <span class="highlight-green">Base Growth: {base_growth*100:.1f}%</span> (상한 80%)
        </div>
        """, unsafe_allow_html=True)
        
        # Base Growth 수동 조절 옵션
        use_custom_growth = st.checkbox("Base Growth 수동 설정", value=False)
        
        if use_custom_growth:
            base_growth = st.slider("Base Growth (%)", 0.0, 30.0, base_growth*100, 0.5) / 100
        
        # 연도별 성장률 (점진적 감소)
        st.markdown("**연도별 Growth (점진적 감소)**")
        
        decay = st.slider("연간 감소율", 0.0, 30.0, 10.0, 5.0) / 100
        
        revenue_growth = []
        growth_display = []
        for i in range(projection_years):
            g = base_growth * ((1 - decay) ** i)
            revenue_growth.append(g)
            growth_display.append(f"Y{i+1}: {g*100:.1f}%")
        
        st.code(" → ".join(growth_display))
        
        st.divider()
        
        # ===== Margins (고정값) =====
        st.markdown("### 📊 Margins & Ratios")
        
        m1, m2, m3, m4 = st.columns(4)
        
        ebitda_margin = m1.number_input("EBITDA Margin (%)", value=hist_avg['avg_ebitda_margin']*100, step=0.5) / 100
        da_pct = m2.number_input("D&A (%)", value=hist_avg['avg_da_pct']*100, step=0.1) / 100
        capex_pct = m3.number_input("CapEx (%)", value=hist_avg['avg_capex_pct']*100, step=0.1) / 100
        nwc_pct = m4.number_input("NWC (%)", value=hist_avg['avg_nwc_pct']*100, step=0.5) / 100
        
        st.divider()

        # ===== Terminal Value =====
        st.markdown("### 🎯 Terminal Value")

        # 현재 거래 배수 및 추천 배수 표시
        current_ev_ebitda = hist_avg.get('current_ev_ebitda', 0)
        suggested_exit = hist_avg.get('suggested_exit_multiple', sector_defaults.get('exit_multiple', 12))

        if current_ev_ebitda and current_ev_ebitda > 0:
            st.markdown(f"""
            <div class="info-box">
            <b>📊 Exit Multiple 기준</b><br>
            • 현재 EV/EBITDA: <b>{current_ev_ebitda:.1f}x</b><br>
            • 섹터 평균: <b>{sector_defaults.get('exit_multiple', 12):.0f}x</b><br>
            → <span class="highlight-blue">추천 Exit (현재의 70%): {suggested_exit:.1f}x</span>
            </div>
            """, unsafe_allow_html=True)

        tv1, tv2 = st.columns(2)

        terminal_growth = tv1.number_input("Perpetual Growth (%)", value=2.5, min_value=1.0, max_value=4.0, step=0.1) / 100
        exit_multiple = tv2.number_input("Exit EV/EBITDA", value=float(suggested_exit), min_value=4.0, max_value=50.0, step=0.5)

        st.divider()
        
        # ===== Bull/Bear 조절 (★ 핵심: 2개만!) =====
        st.markdown("### 🎭 Bull / Bear 조절")

        st.markdown("""
        <div class="warning-box">
        💡 <b>Bull/Bear는 2개 변수만 조절합니다:</b><br>
        1. <b>Revenue Growth</b> (성장 스토리) - 비율 조정<br>
        2. <b>Exit Multiple</b> (시장 센티먼트) - 비율 조정
        </div>
        """, unsafe_allow_html=True)

        bb1, bb2 = st.columns(2)

        with bb1:
            st.markdown("**Revenue Growth 조정**")
            bull_growth_factor = st.slider("Bull: Base × ", 1.0, 1.5, 1.20, 0.05)
            bear_growth_factor = st.slider("Bear: Base × ", 0.5, 1.0, 0.70, 0.05)

        with bb2:
            st.markdown("**Exit Multiple 조정**")
            bull_multiple_factor = st.slider("Bull: Base × ", 1.0, 1.5, 1.15, 0.05)
            bear_multiple_factor = st.slider("Bear: Base × ", 0.5, 1.0, 0.85, 0.05)

        # 시나리오 미리보기
        st.markdown("**시나리오 미리보기**")

        bull_g = min(base_growth * bull_growth_factor, 0.80)
        bear_g = max(base_growth * bear_growth_factor, 0.0)
        bull_m = min(exit_multiple * bull_multiple_factor, 35)
        bear_m = max(exit_multiple * bear_multiple_factor, 5)

        preview_df = pd.DataFrame({
            'Scenario': ['🐻 Bear', '📊 Base', '🐂 Bull'],
            'Revenue Growth Y1': [
                f"{bear_g*100:.1f}%",
                f"{base_growth*100:.1f}%",
                f"{bull_g*100:.1f}%"
            ],
            'Exit Multiple': [
                f"{bear_m:.1f}x",
                f"{exit_multiple:.1f}x",
                f"{bull_m:.1f}x"
            ]
        })

        st.dataframe(preview_df, use_container_width=True, hide_index=True)

        # 저장
        st.session_state['assumptions'] = {
            'revenue_growth': revenue_growth,
            'ebitda_margin': ebitda_margin,
            'da_pct': da_pct,
            'capex_pct': capex_pct,
            'nwc_pct': nwc_pct,
            'terminal_growth': terminal_growth,
            'exit_multiple': exit_multiple,
            'wacc': wacc,
            'tax_rate': tax_rate,
            'bull_growth_factor': bull_growth_factor,
            'bear_growth_factor': bear_growth_factor,
            'bull_multiple_factor': bull_multiple_factor,
            'bear_multiple_factor': bear_multiple_factor,
        }
        st.session_state['dcf_model'] = dcf_model
        st.session_state['stock_data'] = data
        
        st.success("✅ Go to 'DCF Model' tab")
    
    # ==================== TAB 2: DCF Model ====================
    with tab2:
        st.subheader("📈 DCF Model Output")
        
        if 'assumptions' not in st.session_state:
            st.warning("Set assumptions first")
            st.stop()
        
        a = st.session_state['assumptions']
        dcf_model = st.session_state['dcf_model']

        scenarios = dcf_model.run_scenarios(
            base_assumptions=a,
            wacc=a['wacc'],
            tax_rate=a['tax_rate'],
            bull_growth_factor=a['bull_growth_factor'],
            bear_growth_factor=a['bear_growth_factor'],
            bull_multiple_factor=a['bull_multiple_factor'],
            bear_multiple_factor=a['bear_multiple_factor'],
        )
        
        st.session_state['scenarios'] = scenarios

        # Sanity Check & Warnings
        sanity = scenarios['base'].get('sanity_check', {})
        warnings_list = scenarios['base'].get('warnings', [])

        if sanity.get('pass'):
            st.markdown(f'<div class="success-box">✅ FCF Sanity Check: {sanity.get("message")}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-box">⚠️ {sanity.get("message")}</div>', unsafe_allow_html=True)

        # 모델 경고 표시
        if warnings_list:
            warnings_html = "<br>".join([f"• {w}" for w in warnings_list])
            st.markdown(f'<div class="warning-box">⚠️ <b>Model Warnings:</b><br>{warnings_html}</div>', unsafe_allow_html=True)
        
        # Projections
        st.markdown("### 📊 Base Case Projections")
        
        proj = scenarios['base']['projections']
        
        display_df = pd.DataFrame({
            'Year': [f"Y{int(r['year'])}" for _, r in proj.iterrows()],
            'Growth': [f"{r['revenue_growth']*100:.1f}%" for _, r in proj.iterrows()],
            'Revenue': [f"${r['revenue']/1e9:.1f}B" for _, r in proj.iterrows()],
            'EBITDA': [f"${r['ebitda']/1e9:.1f}B" for _, r in proj.iterrows()],
            'UFCF': [f"${r['ufcf']/1e9:.1f}B" for _, r in proj.iterrows()],
            'PV': [f"${r['pv_ufcf']/1e9:.1f}B" for _, r in proj.iterrows()],
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Valuation Summary
        st.markdown("### 💰 Valuation")
        
        v = scenarios['base']['valuations']
        summary = scenarios['summary']
        current = data['current_price']
        
        v1, v2, v3 = st.columns(3)
        
        with v1:
            st.markdown("**Perpetuity**")
            perp = v.get('perpetuity', {})
            if perp:
                st.write(f"PV(FCF): ${perp.get('sum_pv_fcf', 0)/1e9:.1f}B")
                st.write(f"PV(TV): ${perp.get('pv_terminal_value', 0)/1e9:.1f}B ({perp.get('tv_pct_of_ev', 0)*100:.0f}%)")
                st.write(f"EV: ${perp.get('enterprise_value', 0)/1e9:.1f}B")
                st.metric("Price", f"${perp.get('per_share_value', 0):.2f}")
        
        with v2:
            st.markdown("**Exit Multiple**")
            exit_m = v.get('exit_multiple', {})
            if exit_m:
                st.write(f"PV(FCF): ${exit_m.get('sum_pv_fcf', 0)/1e9:.1f}B")
                st.write(f"PV(TV): ${exit_m.get('pv_terminal_value', 0)/1e9:.1f}B ({exit_m.get('tv_pct_of_ev', 0)*100:.0f}%)")
                st.write(f"EV: ${exit_m.get('enterprise_value', 0)/1e9:.1f}B")
                st.metric("Price", f"${exit_m.get('per_share_value', 0):.2f}")
        
        with v3:
            st.markdown("**Summary**")
            p1 = perp.get('per_share_value', 0) if perp else 0
            p2 = exit_m.get('per_share_value', 0) if exit_m else 0
            blended = (p1 + p2) / 2 if p1 > 0 and p2 > 0 else max(p1, p2)
            upside = (blended / current - 1) * 100 if current > 0 else 0
            
            st.metric("Current", f"${current:.2f}")
            st.metric("Fair Value", f"${blended:.2f}", f"{upside:+.1f}%")
            
            if upside > 15:
                st.success("🟢 UNDERVALUED")
            elif upside > -15:
                st.warning("🟡 FAIR")
            else:
                st.error("🔴 OVERVALUED")
        
        # ===== Scenario Comparison (★ 핵심) =====
        st.markdown("### 🎭 Scenario Comparison")
        
        st.markdown("""
        <div class="info-box">
        <b>변경된 변수만 표시:</b> Revenue Growth Y1, Exit Multiple
        </div>
        """, unsafe_allow_html=True)
        
        scenario_df = pd.DataFrame({
            'Scenario': ['🐻 BEAR', '📊 BASE', '🐂 BULL'],
            'Rev Growth Y1': [
                f"{summary.get('bear_growth', 0)*100:.1f}%",
                f"{summary.get('base_growth', 0)*100:.1f}%",
                f"{summary.get('bull_growth', 0)*100:.1f}%"
            ],
            'Exit Multiple': [
                f"{summary.get('bear_exit', 0):.1f}x",
                f"{summary.get('base_exit', 0):.1f}x",
                f"{summary.get('bull_exit', 0):.1f}x"
            ],
            'Perpetuity': [
                f"${summary.get('bear_perpetuity', 0):.2f}",
                f"${summary.get('base_perpetuity', 0):.2f}",
                f"${summary.get('bull_perpetuity', 0):.2f}"
            ],
            'Exit Method': [
                f"${summary.get('bear_exit_val', 0):.2f}",
                f"${summary.get('base_exit_val', 0):.2f}",
                f"${summary.get('bull_exit_val', 0):.2f}"
            ],
            'Upside': [
                f"{(summary.get('bear_perpetuity', 0)/current-1)*100:+.0f}%" if current > 0 else "N/A",
                f"{(summary.get('base_perpetuity', 0)/current-1)*100:+.0f}%" if current > 0 else "N/A",
                f"{(summary.get('bull_perpetuity', 0)/current-1)*100:+.0f}%" if current > 0 else "N/A"
            ]
        })
        
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)
        
        # Weighted
        if summary.get('weighted_perpetuity'):
            wp = summary['weighted_perpetuity']
            we = summary.get('weighted_exit', 0)
            st.info(f"**Probability-Weighted (25/50/25):** Perpetuity ${wp:.2f} ({(wp/current-1)*100:+.1f}%) | Exit ${we:.2f} ({(we/current-1)*100:+.1f}%)")

        # ===== Reverse DCF (★ 시장 기대 성장률) =====
        st.markdown("### 🔄 Reverse DCF")
        st.caption("현재 주가가 암시하는 성장률 (시장이 기대하는 성장률)")

        reverse_result = dcf_model.reverse_dcf(
            base_assumptions=a,
            wacc=a['wacc'],
            tax_rate=a['tax_rate'],
            years=5
        )

        if 'error' not in reverse_result:
            implied_g = reverse_result['implied_growth']
            rating = reverse_result['rating']
            feasibility = reverse_result['feasibility']

            # 등급에 따른 색상
            if rating == "Conservative":
                rating_color = "highlight-green"
            elif rating == "Reasonable":
                rating_color = "highlight-blue"
            elif rating == "Slightly Aggressive":
                rating_color = "highlight-yellow"
            else:
                rating_color = "highlight-red"

            st.markdown(f"""
            <div class="info-box">
            <b>📊 시장이 가정하는 성장률</b><br>
            • Implied Growth (Y1): <span class="{rating_color}">{implied_g*100:.1f}%</span><br>
            • 과거 3Y CAGR: <b>{reverse_result['historical_cagr_3y']*100:.1f}%</b><br>
            • 평가: <span class="{rating_color}">{feasibility}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(reverse_result.get('message', 'Reverse DCF 계산 불가'))

    # ==================== TAB 3: Sensitivity ====================
    with tab3:
        st.subheader("🎯 Sensitivity Analysis")
        
        if 'assumptions' not in st.session_state:
            st.warning("Run analysis first")
            st.stop()
        
        a = st.session_state['assumptions']
        dcf_model = st.session_state['dcf_model']
        
        base_dcf = DCFAssumptions(
            revenue_growth_rates=a['revenue_growth'],
            ebitda_margin=a['ebitda_margin'],
            da_pct=a['da_pct'],
            capex_pct=a['capex_pct'],
            nwc_pct=a['nwc_pct'],
            tax_rate=a['tax_rate'],
            terminal_growth=a['terminal_growth'],
            exit_multiple=a['exit_multiple'],
            wacc=a['wacc'],
        )
        
        sens = dcf_model.sensitivity_analysis(base_dcf)
        
        s1, s2 = st.columns(2)
        
        with s1:
            st.markdown("**WACC vs Terminal Growth**")
            st.dataframe(sens['wacc_vs_growth'], use_container_width=True, hide_index=True)
        
        with s2:
            st.markdown("**WACC vs Exit Multiple**")
            st.dataframe(sens['wacc_vs_exit'], use_container_width=True, hide_index=True)
    
    # ==================== TAB 4: Football Field ====================
    with tab4:
        st.subheader("🏈 Football Field Chart")
        
        if 'scenarios' not in st.session_state:
            st.warning("Run analysis first")
            st.stop()
        
        scenarios = st.session_state['scenarios']
        data = st.session_state['stock_data']
        
        ff_data = create_football_field_data(
            current_price=data['current_price'],
            dcf_scenarios=scenarios,
            analyst_targets=(data.get('target_low', 0), data.get('target_mean', 0), data.get('target_high', 0)),
            week_52_range=(data['52w_low'], data['52w_high'])
        )
        
        if ff_data:
            fig = go.Figure()

            # 테마 호환 색상 (밝은 색상으로 다크/라이트 모두 가시성 확보)
            colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']

            for i, row in enumerate(ff_data):
                fig.add_trace(go.Bar(
                    y=[row['category']],
                    x=[row['high'] - row['low']],
                    base=[row['low']],
                    orientation='h',
                    marker_color=colors[i % len(colors)],
                    marker_line=dict(width=1, color='rgba(255,255,255,0.3)'),
                    text=f"${row['low']:.0f} - ${row['high']:.0f}",
                    textposition='inside',
                    textfont=dict(color='white', size=12),
                    name=row['category']
                ))

                fig.add_trace(go.Scatter(
                    x=[row['mid']],
                    y=[row['category']],
                    mode='markers',
                    marker=dict(size=14, color='#fbbf24', symbol='diamond',
                               line=dict(width=2, color='white')),
                    showlegend=False,
                    hovertemplate=f"Mid: ${row['mid']:.2f}<extra></extra>"
                ))

            fig.add_vline(x=data['current_price'], line_dash="dash", line_color="#ef4444", line_width=2,
                         annotation_text=f"Current: ${data['current_price']:.2f}",
                         annotation_font_color="#ef4444")

            fig.update_layout(
                title=dict(text="Valuation Range", font=dict(size=16)),
                xaxis_title="Share Price ($)",
                height=400,
                showlegend=False,
                margin=dict(l=150, r=50, t=50, b=50),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(128,128,128,0.2)', zerolinecolor='rgba(128,128,128,0.2)'),
                yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
                font=dict(color='gray')
            )

            st.plotly_chart(fig, use_container_width=True)

if not analyze_btn:
    st.info("👈 Enter ticker and click 'Run Analysis'")

    st.markdown("""
    ### ✨ Key Features

    **1. Base Growth 자동 계산**
    - 3Y CAGR × 70% + 5Y CAGR × 30% 가중평균
    - 섹터 Cap 초과 시 초과분의 50% 반영 (고성장 기업 대응)

    **2. Mid-year Convention**
    - 현금흐름이 연중에 발생한다고 가정
    - 보다 정확한 현재가치 계산

    **3. Bull/Bear 비율 조정**
    | Scenario | Rev Growth | Exit Multiple |
    |----------|------------|---------------|
    | Bull | Base × 1.2 | Base × 1.15 |
    | Base | Base | Base |
    | Bear | Base × 0.7 | Base × 0.85 |

    **4. 개선된 경고 시스템**
    - WACC-Growth spread 검증
    - TV 비중 경고
    - FCF Sanity Check
    """)

st.divider()
st.caption(f"⚠️ Educational only | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
