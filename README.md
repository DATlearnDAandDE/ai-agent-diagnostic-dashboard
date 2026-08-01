# AI Agent Diagnostic Intelligence Dashboard

> 🧠 **Báo cáo phân tích chuyên sâu chi phí & hiệu năng hoạt động của AI Agent**  
> Dữ liệu Telemetry 05/2026 – 08/2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## 📊 Tính năng

Dashboard phân tích theo **4 cấp độ** (4-Level Analytics Framework):

| Tab | Cấp độ | Nội dung |
|-----|--------|----------|
| 📊 Cấp 1 | **Descriptive** | Tổng quan hiệu năng, chi phí, phân tích theo domain & động lực học |
| 🔍 Cấp 2 | **Diagnostic** | Looping Pattern, System Prompt Impact, Anomaly Detection |
| 🔮 Cấp 3 | **Predictive** | Risk Matrix, Magic Quadrant, Radar Chart đa chiều |
| 💡 Cấp 4 | **Prescriptive** | Smart Routing Matrix, Circuit Breaker, Micro-Tasking Pipeline |

## 🚀 Deploy locally

```bash
# Clone repo
git clone <your-repo-url>
cd <repo-name>

# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run validate_dashboard.py
```

## 📁 Cấu trúc thư mục

```
code/
├── validate_dashboard.py       # Main app
├── processed_agentic_traces.csv # Dữ liệu
├── requirements.txt
└── .streamlit/
    └── config.toml
```

## 🛠 Tech Stack

- **Frontend**: Streamlit 1.59
- **Visualization**: Plotly 6.9 (Bento Grid Layout)
- **Data**: Pandas, NumPy, SciPy
- **Design**: Custom CSS — Glassmorphism + Bento Grid

---
*Data Source: OpenTelemetry + LangSmith | Q2-Q3 2026 | Confidential*
