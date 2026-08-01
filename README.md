# AI Agent Diagnostic Intelligence Dashboard

> 🧠 **Báo cáo phân tích chuyên sâu chi phí & hiệu năng hoạt động của AI Agent**  
> Dữ liệu Telemetry 05/2026 – 08/2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## 📊 Tính năng

Dashboard **chính** là file `validate_dashboard.py` — phân tích theo **4 cấp độ** (4-Level Analytics Framework):

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

# Chạy dashboard chính
streamlit run validate_dashboard.py
```

## 📁 Cấu trúc thư mục

```
code/
├── validate_dashboard.py        # 🎯 DASHBOARD CHÍNH (4 tabs, 14+ biểu đồ)
├── app_dashboard.py             # Dashboard phụ — Bento Grid tóm tắt
├── processed_agentic_traces.csv # Dữ liệu Telemetry
├── requirements.txt             # Python dependencies
└── .streamlit/
    └── config.toml              # Cấu hình Streamlit
```

## ☁️ Deploy trên Streamlit Community Cloud

1. Push repo lên GitHub
2. Vào [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Điền:
   - **Repository**: `<github-username>/<repo-name>`
   - **Branch**: `main`
   - **Main file path**: `validate_dashboard.py`  ← **File chính**
4. Nhấn **Deploy!**

## 🛠 Tech Stack

- **Frontend**: Streamlit 1.59
- **Visualization**: Plotly 6.9 (Bento Grid Layout + 4 Analytical Tabs)
- **Data**: Pandas 3.0, NumPy 2.4, SciPy
- **Design**: Custom CSS — Glassmorphism + Bento Grid (Light Theme)

---
*Data Source: OpenTelemetry + LangSmith | Q2-Q3 2026 | Confidential*
