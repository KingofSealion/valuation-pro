# 💰 Wall Street DCF Valuation

**Proper Investment Banking Style DCF Model**

Revenue → EBITDA → UFCF Bottom-up Build | Perpetuity & Exit Multiple Terminal Value

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Structure

```
valuation_pro/
├── app.py              # Main Streamlit application
├── dcf_model.py        # Wall Street DCF engine
├── data_fetcher.py     # Yahoo Finance data collector
└── requirements.txt
```

## ✨ Features

### 📊 Proper DCF Structure

```
Revenue
× EBITDA Margin
─────────────────
= EBITDA
- D&A
─────────────────
= EBIT
× (1 - Tax Rate)
─────────────────
= NOPAT
+ D&A
- CapEx
- ΔNWC
─────────────────
= Unlevered FCF    ← This is what you discount
```

### 💵 WACC Calculation (CAPM)

- Risk-Free Rate (10Y Treasury)
- Beta
- Equity Risk Premium
- Cost of Debt
- Capital Structure (D/E)

### 🎯 Terminal Value (Both Methods)

1. **Perpetuity Growth**: `FCF × (1+g) / (WACC - g)`
2. **Exit Multiple**: `EBITDA × EV/EBITDA`

### 🎭 Scenario Analysis

- **Bull Case**: +25% growth, +2% margin, -1% WACC
- **Base Case**: Your assumptions
- **Bear Case**: -25% growth, -2% margin, +1% WACC
- **Probability Weighted**: 25% / 50% / 25%

### 🎯 Sensitivity Analysis

- WACC vs Terminal Growth (Perpetuity method)
- WACC vs Exit Multiple

### 🏈 Football Field Chart

Visual comparison of:
- 52-Week Range
- Analyst Targets
- DCF (Perpetuity) - Bear to Bull
- DCF (Exit Multiple) - Bear to Bull

## 📋 Tabs

| Tab | Description |
|-----|-------------|
| Historical Data | 5-year revenue, EBITDA, margins trend |
| Assumptions | WACC, growth rates, margins, terminal value |
| DCF Model | Year-by-year projections, valuation bridge |
| Sensitivity | 2-way sensitivity tables |
| Football Field | Visual valuation comparison |

## ⚠️ Limitations

- Data from Yahoo Finance (free, delayed)
- No consensus estimates (forward EPS only)
- Small-caps may have data gaps

## 📖 How to Use

1. Enter ticker symbol (e.g., AAPL, MSFT, NVDA)
2. Review **Historical Data** for context
3. Adjust **Assumptions** based on your view
4. Check **DCF Model** for implied price
5. Validate with **Sensitivity Analysis**
6. Compare methods in **Football Field**

---

*Built for Quantimental Investors*
