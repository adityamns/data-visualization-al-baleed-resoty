# 🏨 Al Baleed Resort - Interactive Dashboard

**Comprehensive Analytics Dashboard for Hotel Review Analysis**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

---


## ⚡ Quick Start

### Windows
Double-click `run_dashboard.bat`

### Mac/Linux
```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

### Manual
```bash
pip install -r requirements.txt
streamlit run streamlit_dashboard.py
```

Dashboard akan otomatis membuka di browser: `http://localhost:8501`

---

## 📊 Dashboard Features

### 🎯 6 Interactive Tabs

1. **📊 Overview** - Executive summary, key metrics, trends
2. **📈 Descriptive** - Service aspects, trip types, correlations
3. **🔬 Diagnostic** - Correlation heatmap, word clouds, root cause
4. **🤖 Predictive** - ML model, feature importance, predictions
5. **💼 Prescriptive** - Recommendations, action plans
6. **📝 Raw Data** - Interactive data explorer

### 🔍 Sidebar Filters (Real-time!)

- 📅 Date Range
- 👥 Trip Type (Couples, Family, Friends, Solo, Business)
- ⭐ Rating (1-5 stars)
- 🌴 Season (High Season/Khareef vs Low Season)

---

## 🌐 Deploy to Cloud (FREE!)

### Step 1: GitHub
1. Create repo: https://github.com/new
2. Upload files: `streamlit_dashboard.py`, `requirements.txt`, dataset

### Step 2: Streamlit Cloud
1. Go to: https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repo
5. Deploy! ✨

**Your dashboard is now live with a public URL!**

---

## 📦 Files Included

```
📁 Dashboard Package
├── streamlit_dashboard.py    # Main app
├── requirements.txt           # Dependencies
├── DEPLOYMENT_GUIDE.md        # Full guide
├── README.md                  # This file
├── run_dashboard.bat          # Windows launcher
├── run_dashboard.sh           # Mac/Linux launcher
└── .streamlit/
    └── config.toml            # Streamlit config
```

---

## 🎯 Key Insights from Dashboard

### Top 3 Service Aspects (from ML Model)
Feature importance analysis reveals the most critical factors affecting guest satisfaction.

### Sentiment Analysis
- **88%+ Positive** reviews (4-5 stars)
- Word clouds show common themes
- Bi-gram analysis identifies key phrases

### Actionable Recommendations
Department-specific action plans for:
- General Manager
- Front Office
- Housekeeping
- Maintenance
- Marketing

---

## 💡 Use Cases

### For Students (Tugas/Thesis)
- ✅ Professional presentation tool
- ✅ Interactive demo for defense
- ✅ Portfolio project
- ✅ Shareable via URL

### For Hotel Management
- ✅ Real-time monitoring
- ✅ Data-driven decisions
- ✅ Department KPIs
- ✅ Trend analysis

### For Data Analysts
- ✅ EDA (Exploratory Data Analysis)
- ✅ ML model insights
- ✅ Correlation studies
- ✅ Text mining

---

## 🔧 Requirements

- Python 3.8+
- 8GB RAM (recommended)
- Modern browser (Chrome, Firefox, Safari)

---

## 📸 Screenshots

*Upload your dataset → Instant analytics!*

```
┌─────────────────────────────────────┐
│  📊 Overview                        │
│  ├─ Key Metrics (5 cards)          │
│  ├─ Rating Distribution (Pie)      │
│  ├─ Sentiment Analysis (Bar)       │
│  └─ Monthly Trend (Line)           │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  🤖 Predictive Analytics            │
│  ├─ Feature Importance (Bar)       │
│  ├─ Model Accuracy: 85%+           │
│  └─ Confusion Matrix               │
└─────────────────────────────────────┘
```

---

## 🆘 Troubleshooting

### Port already in use
```bash
streamlit run streamlit_dashboard.py --server.port 8502
```

### Module not found
```bash
pip install -r requirements.txt
```

### Upload size too large
Edit `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200
```

---

## 🎓 Academic Context

**Course**: Data Analytics / Business Intelligence / Data Science  
**Project**: Hotel Review Analysis & Predictive Modeling  
**Framework**: 6-Phase Analytics (Descriptive → Prescriptive)  
**Tech Stack**: Python, Streamlit, Plotly, Scikit-learn

---

## 📚 Documentation

- **Full Deployment Guide**: See `DEPLOYMENT_GUIDE.md`
- **Streamlit Docs**: https://docs.streamlit.io
- **Plotly Docs**: https://plotly.com/python/

---

## 🎉 Ready to Impress!

Your dashboard is **production-ready** and **presentation-ready**!

**What makes it special:**
- ✅ Fully interactive with real-time filters
- ✅ Machine learning insights (feature importance!)
- ✅ Professional UI/UX
- ✅ Cloud deployable (free!)
- ✅ Mobile responsive
- ✅ Actionable business recommendations

---

## 🏆 Pro Tips

1. **For Presentation**: Start with Overview tab → Show filters → Demo ML insights
2. **For Report**: Export key charts as PNG (right-click Plotly charts)
3. **For Portfolio**: Add custom domain and include in CV/LinkedIn
4. **For Collaboration**: Share URL with team for real-time feedback

---

## 📞 Support

Need help or want to customize?
- Check `DEPLOYMENT_GUIDE.md` for detailed instructions
- Streamlit docs: https://docs.streamlit.io
- Community: https://discuss.streamlit.io

---

**Made with ❤️ for Al Baleed Resort Analytics**

*Streamlit • Plotly • Scikit-learn • Python*
