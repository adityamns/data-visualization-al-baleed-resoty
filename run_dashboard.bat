@echo off
echo ========================================
echo   Al Baleed Resort Dashboard Launcher
echo ========================================
echo.

echo Installing dependencies...
pip install streamlit pandas numpy plotly scikit-learn openpyxl wordcloud matplotlib seaborn statsmodels

echo.
echo Starting dashboard...
streamlit run streamlit_dashboard.py

pause
