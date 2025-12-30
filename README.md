# Stock Valuation Pro

**Multi-Method Stock Valuation Dashboard**

DCF Valuation | Relative Valuation | Risk Analysis | Football Field Summary

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure

```
valuation_pro/
├── app.py              # Main Streamlit application (3-Tab UI)
├── dcf_model.py        # Wall Street DCF engine
├── valuation_utils.py  # WACC, Lifecycle, Convergence utilities
├── data_fetcher.py     # Yahoo Finance data collector
├── risk_model.py       # Risk scorecard & analysis
└── requirements.txt
```

## Key Features

### 1. DCF Valuation (Absolute Value)

| Feature | Description |
|---------|-------------|
| **Lifecycle Classification** | Hyper-Growth (>20%), High-Growth (10-20%), Stable (<10%) |
| **Smart Defaults** | Projection period, growth decay, margin convergence |
| **WACC Auto-Calculation** | CAPM + Synthetic Rating + Blume's Adjusted Beta |
| **Terminal Value** | Perpetuity Growth & Exit Multiple (dual methods) |
| **Sensitivity Analysis** | WACC vs Terminal Growth matrix |

#### DCF Model Structure

```
Revenue
× EBITDA Margin (sector convergence)
= EBITDA
- D&A
= EBIT
× (1 - Tax Rate)
= NOPAT
+ D&A
- CapEx (steady state convergence)
- ΔNWC
= Unlevered FCF → Discount to Present Value
```

#### WACC Calculation

```
WACC = (E/V) × Ke + (D/V) × Kd × (1-T)
```

| Component | Method |
|-----------|--------|
| Cost of Equity (Ke) | CAPM: Rf + β × MRP |
| Beta | Blume's Adjusted Beta (mean reversion) |
| Cost of Debt (Kd) | Actual Interest Rate or Synthetic Rating (ICR-based) |
| Risk-Free Rate | 10Y Treasury (^TNX) real-time |
| Tax Rate | Effective tax rate from financials |

### 2. Relative Valuation

| Feature | Description |
|---------|-------------|
| **Historical P/E Band** | 5Y Low/Avg/High with percentile |
| **Forward P/E Analysis** | Trailing vs Forward comparison |
| **PEG Ratio** | Forward P/E ÷ Analyst FY1 Growth (Finviz/Nasdaq style) |
| **P/B Ratio Band** | Book value based valuation |
| **Peer Comparison** | Multi-metric peer group analysis |

#### PEG Ratio Calculation

```
PEG = Forward P/E / FY1 EPS Growth Rate (%)
```
- Uses analyst consensus growth estimates
- Comparable to Finviz/Nasdaq methodology

### 3. Risk Scorecard

| Category | Metrics |
|----------|---------|
| **Valuation Risk** | P/E vs 5Y Avg, PEG Ratio |
| **Financial Risk** | Debt/Equity, Interest Coverage |
| **Quality Risk** | ROE, Operating Margin stability |
| **Growth Risk** | Revenue growth volatility |

### 4. Summary (Football Field)

- Visual range comparison of all valuation methods
- DCF (Bull/Base/Bear), Relative Valuation, Analyst Targets
- Average Fair Value calculation
- Buy/Hold/Sell recommendation

## 3-Tab Structure

| Tab | Content |
|-----|---------|
| **DCF Valuation** | Lifecycle analysis, FCF projection, sensitivity matrix |
| **Relative Valuation** | P/E Band, PEG, Peer comparison, Bull/Base/Bear scenarios |
| **Summary** | Football Field chart, risk scorecard, final verdict |

## Financial Table

- **Annual View**: Revenue, EBITDA, Net Income with YoY growth
- **Quarterly View**: YoY and QoQ growth rates
- Toggle between views with radio button

## Data Sources

- **Yahoo Finance (yfinance)**: Price, financials, analyst estimates
- **10Y Treasury (^TNX)**: Risk-free rate
- Real-time data with 10-minute caching

## Limitations

- Data from Yahoo Finance (free tier, may have delays)
- Rate limiting on Streamlit Cloud (shared IP issue)
- Some small-caps may have incomplete data

## How to Use

1. **Enter Ticker** (e.g., AAPL, MSFT, NVDA)
2. **DCF Tab**: Review lifecycle, adjust assumptions if needed
3. **Relative Tab**: Check P/E Band position, PEG ratio
4. **Summary Tab**: See Football Field, make final decision

---

*Built for value investors*
