#!/bin/bash

echo "========================================"
echo "  Al Baleed Resort Dashboard Launcher"
echo "========================================"
echo

echo "Installing dependencies..."
pip3 install streamlit pandas numpy plotly scikit-learn openpyxl wordcloud matplotlib seaborn statsmodels

echo
echo "Starting dashboard..."
python3 -m streamlit run streamlit_dashboard.py
