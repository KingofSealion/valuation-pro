# 💰 Stock Valuation Pro

**Context-Aware DCF Valuation with Smart Defaults**

Investment Banking Style DCF | Lifecycle-Based Projection | Peer Comparison

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Structure

```
valuation_pro/
├── app.py              # Main Streamlit application (3-Tab UI)
├── dcf_model.py        # Wall Street DCF engine
├── valuation_utils.py  # WACC, Lifecycle, Convergence utilities
├── data_fetcher.py     # Yahoo Finance data collector
└── requirements.txt
```

## ✨ Key Features

### 🤖 Context-Aware Smart Defaults

회사 상황에 맞는 지능형 기본값 자동 설정:

| Feature | Description |
|---------|-------------|
| **Lifecycle Classification** | Hyper-Growth (>20%), High-Growth (10-20%), Stable (<10%) |
| **Projection Period** | 10Y / 7Y / 5Y (Lifecycle 기반 자동 설정) |
| **Growth Decay** | Risk-Free Rate로 점진적 수렴 |
| **Margin Convergence** | 섹터 평균으로 수렴 |
| **CapEx Convergence** | D&A × 105% (Steady State) |
| **Tax Normalization** | 21% 법정세율로 정상화 |

### 📊 Proper DCF Structure (Full Model)

```
Revenue
× EBITDA Margin (연도별 수렴)
─────────────────
= EBITDA
- D&A
─────────────────
= EBIT
× (1 - Tax Rate) (연도별 정상화)
─────────────────
= NOPAT
+ D&A
- CapEx (연도별 수렴)
- ΔNWC
─────────────────
= Unlevered FCF    ← This is what you discount
```

### 💵 WACC Auto-Calculation

- **Cost of Equity (CAPM)**: Rf + β × MRP
- **Adjusted Beta**: Blume's method (mean reversion)
- **Synthetic Rating**: ICR 기반 신용등급 산출
- **Cost of Debt**: 실제 이자비용 or Synthetic Spread

### 🎯 Terminal Value (Dual Methods)

1. **Perpetuity Growth**: `FCF × (1+g) / (WACC - g)`
2. **Exit Multiple**: `EBITDA × EV/EBITDA`

### 📈 Peer Comparison (Relative Valuation)

- **EPS Growth**: Forward EPS / Trailing EPS - 1
- **PEG Ratio**: P/E ÷ EPS Growth%
- **Implied Fair Value**: Peer Avg 기반 적정가
- **Premium/Discount**: Peer 대비 프리미엄/디스카운트

## 📋 3-Tab Structure

| Tab | Description |
|-----|-------------|
| **DCF Valuation** | Smart Defaults, Growth Decay, Sensitivity Analysis |
| **Peer Comparison** | EPS Growth, PEG Ratio, Relative Valuation |
| **Summary** | Football Field Chart, Buy/Hold/Sell 판단 |

## 🎯 Growth Rate Sources

| Source | Description | Use Case |
|--------|-------------|----------|
| **Smart Default** | Lifecycle 기반 + Decay Schedule | 권장 (Context-Aware) |
| FCF CAGR | Historical FCF 복합성장률 | 안정적 FCF 기업 |
| Revenue Growth | TTM 매출 성장률 | 최근 트렌드 반영 |
| Revenue CAGR | 3~5Y 매출 복합성장률 | 장기 평균 |
| Manual | 사용자 직접 입력 | 특수 상황 |

## 🔄 Convergence Logic

### Growth Decay (Risk-Free Rate 수렴)
```
Year 1: 25.0%  ──┐
Year 2: 22.5%    │
Year 3: 20.0%    │ Linear Decay
Year 4: 17.5%    │
...              │
Year N: 3.0%   ──┘ (≈ Risk-Free Rate)
```

### CapEx Convergence (Steady State)
```
Current: 8% of Revenue  ──┐
                          │ Linear Interpolation
Final: D&A × 105%       ──┘ (Maintenance + Growth CapEx)
```

## ⚠️ Limitations

- Data from Yahoo Finance (free, may be delayed)
- Historical data required for Smart Defaults
- Small-caps may have data gaps

## 📖 How to Use

1. **Enter Ticker** (e.g., AAPL, MSFT, NVDA)
2. **Review Lifecycle** - 자동 분류된 성장 단계 확인
3. **Choose Growth Source** - Smart Default 권장
4. **Adjust WACC/TGR** - 필요시 조정
5. **Check Sensitivity** - WACC vs TGR 민감도 분석
6. **Compare with Peers** - Relative Valuation 확인
7. **Review Summary** - Football Field에서 종합 판단

---

*Built for Quantimental Investors*
