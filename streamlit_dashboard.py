"""
🏨 Al Baleed Resort Salalah - Interactive Dashboard
Comprehensive Analytics Dashboard for Hotel Review Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
import time

# Deep Learning imports for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Al Baleed Resort Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== DATA LOADING & CACHING ====================
@st.cache_data
def load_and_process_data(file_path):
    """Load and process the dataset with all feature engineering"""
    
    # Load data
    df = pd.read_excel(file_path)
    
    # Data Cleaning
    df['User Location'].fillna('Unknown', inplace=True)
    df.dropna(subset=['Stay Date'], inplace=True)
    
    # Convert dates
    df['Stay Date'] = pd.to_datetime(df['Stay Date'], format='%d/%m/%Y', errors='coerce')
    df['Created Date'] = pd.to_datetime(df['Created Date'], format='%d/%m/%Y', errors='coerce')
    df['Published Date'] = pd.to_datetime(df['Published Date'], format='%d/%m/%Y', errors='coerce')
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Strip whitespace
    text_cols = ['Hotel Name', 'User Location', 'Review Title', 'Review Text', 'Trip Type', 'Language']
    for col in text_cols:
        df[col] = df[col].str.strip()
    
    # Feature Engineering
    df['Year'] = df['Stay Date'].dt.year
    df['Month'] = df['Stay Date'].dt.month
    df['Month_Name'] = df['Stay Date'].dt.strftime('%B')
    df['Day_Name'] = df['Stay Date'].dt.strftime('%A')
    df['Is_Weekend'] = df['Stay Date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Seasonality (Salalah Khareef Season: June-September)
    def get_season(month):
        if month in [6, 7, 8, 9]:
            return 'High Season (Khareef)'
        else:
            return 'Low Season'
    
    df['Season'] = df['Month'].apply(get_season)
    
    # Review length
    df['Review_Length_Chars'] = df['Review Text'].str.len()
    df['Review_Length_Words'] = df['Review Text'].str.split().str.len()
    
    # Sentiment
    def get_sentiment(rating):
        if rating >= 4:
            return 'Positive'
        elif rating == 3:
            return 'Neutral'
        else:
            return 'Negative'
    
    df['Sentiment'] = df['Rating'].apply(get_sentiment)
    
    # Service scores
    service_cols = ['Value', 'Rooms', 'Location', 'Cleanliness', 'Service', 'Sleep Quality']
    df['Avg_Service_Score'] = df[service_cols].mean(axis=1)
    df['Service_Score_Variance'] = df[service_cols].var(axis=1)
    
    return df


@st.cache_data
def get_filtered_data(df, date_range, trip_types, ratings, seasons):
    """Filter data based on user selection"""
    filtered_df = df.copy()
    
    # Date filter
    if date_range:
        filtered_df = filtered_df[
            (filtered_df['Stay Date'] >= pd.to_datetime(date_range[0])) &
            (filtered_df['Stay Date'] <= pd.to_datetime(date_range[1]))
        ]
    
    # Trip type filter
    if trip_types:
        filtered_df = filtered_df[filtered_df['Trip Type'].isin(trip_types)]
    
    # Rating filter
    if ratings:
        filtered_df = filtered_df[filtered_df['Rating'].isin(ratings)]
    
    # Season filter
    if seasons:
        filtered_df = filtered_df[filtered_df['Season'].isin(seasons)]
    
    return filtered_df


# ==================== MAIN APP ====================
def main():
    
    # Header
    st.markdown('<h1 class="main-header">🏨 Al Baleed Resort Salalah</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive Analytics Dashboard - TripAdvisor Reviews Analysis</p>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("📁 Upload Dataset (Excel)", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Load data
        with st.spinner('Loading and processing data...'):
            df = load_and_process_data(uploaded_file)
        
        st.sidebar.success(f'✅ Data loaded: {len(df):,} reviews')
        
        # Sidebar Filters
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Filters")
        
        # Date range filter
        min_date = df['Stay Date'].min().date()
        max_date = df['Stay Date'].max().date()
        date_range = st.sidebar.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Trip type filter
        all_trip_types = df['Trip Type'].unique().tolist()
        trip_types = st.sidebar.multiselect(
            "👥 Trip Type",
            options=all_trip_types,
            default=all_trip_types
        )
        
        # Rating filter
        all_ratings = sorted(df['Rating'].unique().tolist())
        ratings = st.sidebar.multiselect(
            "⭐ Rating",
            options=all_ratings,
            default=all_ratings
        )
        
        # Season filter
        all_seasons = df['Season'].unique().tolist()
        seasons = st.sidebar.multiselect(
            "🌴 Season",
            options=all_seasons,
            default=all_seasons
        )
        
        # Apply filters
        filtered_df = get_filtered_data(df, date_range, trip_types, ratings, seasons)
        
        st.sidebar.markdown("---")
        st.sidebar.info(f"📊 Showing {len(filtered_df):,} of {len(df):,} reviews")
        
        # Main Dashboard Tabs
        tab1, tab2, tab3, tab4, tab6 = st.tabs([
            "📊 Overview",
            "📈 Descriptive Analytics",
            "🔬 Diagnostic Analytics",
            "🤖 Predictive Analytics",
            # "💼 Prescriptive Analytics",
            "📝 Raw Data"
        ])
        
        # ==================== TAB 1: OVERVIEW ====================
        with tab1:
            show_overview(filtered_df)
        
        # ==================== TAB 2: DESCRIPTIVE ANALYTICS ====================
        with tab2:
            show_descriptive_analytics(filtered_df)
        
        # ==================== TAB 3: DIAGNOSTIC ANALYTICS ====================
        with tab3:
            show_diagnostic_analytics(filtered_df)
        
        # ==================== TAB 4: PREDICTIVE ANALYTICS ====================
        with tab4:
            show_predictive_analytics(filtered_df)
        
        # ==================== TAB 5: PRESCRIPTIVE ANALYTICS ====================
        # with tab5:
        #     show_prescriptive_analytics(filtered_df)
        
        # ==================== TAB 6: RAW DATA ====================
        with tab6:
            show_raw_data(filtered_df)
    
    else:
        st.info("👈 Please upload the dataset using the sidebar")
        st.markdown("""
        ### 📖 Instructions:
        1. Click on the **Browse files** button in the sidebar
        2. Upload your Excel file (`Al_Baleed_Resort.xlsx`)
        3. The dashboard will automatically process and display the analytics
        
        ### 📊 Dashboard Features:
        - **Interactive filters** (Date, Trip Type, Rating, Season)
        - **6-Phase Analytics** (Overview → Prescriptive)
        - **Real-time visualizations** with Plotly
        - **Machine Learning insights** with feature importance
        - **Actionable recommendations** for hotel management
        """)


# ==================== OVERVIEW TAB ====================
def show_overview(df):
    st.header("📊 Executive Summary")
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Reviews",
            f"{len(df):,}",
            delta=None
        )
    
    with col2:
        avg_rating = df['Rating'].mean()
        st.metric(
            "Average Rating",
            f"{avg_rating:.2f} ⭐",
            delta=f"{(avg_rating - 5) * 100:.1f}%" if avg_rating < 5 else "Perfect!"
        )
    
    with col3:
        positive_pct = (len(df[df['Rating'] >= 4]) / len(df)) * 100
        st.metric(
            "Positive Reviews",
            f"{positive_pct:.1f}%",
            delta=f"{positive_pct - 80:.1f}%" if positive_pct > 80 else None
        )
    
    with col4:
        avg_review_length = df['Review_Length_Words'].mean()
        st.metric(
            "Avg Review Length",
            f"{avg_review_length:.0f} words"
        )
    
    with col5:
        date_range = (df['Stay Date'].max() - df['Stay Date'].min()).days
        st.metric(
            "Data Range",
            f"{date_range} days"
        )
    
    st.markdown("---")
    
    # Charts in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating Distribution
        rating_counts = df['Rating'].value_counts().sort_index()
        fig = go.Figure(data=[
            go.Pie(
                labels=rating_counts.index,
                values=rating_counts.values,
                hole=0.4,
                marker_colors=['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#1f77b4']
            )
        ])
        fig.update_layout(
            title="Rating Distribution",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment Distribution
        sentiment_counts = df['Sentiment'].value_counts()
        fig = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map={'Positive': '#2ca02c', 'Neutral': '#ffbb78', 'Negative': '#d62728'},
            labels={'x': 'Sentiment', 'y': 'Count'},
            title='Sentiment Distribution'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Trip Type Distribution
    st.subheader("👥 Trip Type Analysis")
    trip_counts = df['Trip Type'].value_counts()
    fig = px.bar(
        x=trip_counts.index,
        y=trip_counts.values,
        color=trip_counts.index,
        labels={'x': 'Trip Type', 'y': 'Number of Reviews'},
        title='Distribution of Trip Types'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly Trend
    st.subheader("📅 Monthly Rating Trend")
    monthly_data = df.groupby('Month_Name').agg({
        'Rating': 'mean',
        'Review Text': 'count'
    }).reindex(['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'])
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=monthly_data.index, y=monthly_data['Rating'],
                   mode='lines+markers', name='Average Rating',
                   line=dict(width=3, color='#1f77b4')),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(x=monthly_data.index, y=monthly_data['Review Text'],
               name='Review Count', opacity=0.3, marker_color='lightblue'),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Average Rating", secondary_y=False)
    fig.update_yaxes(title_text="Review Count", secondary_y=True)
    fig.update_layout(height=400, title="Monthly Performance")
    
    st.plotly_chart(fig, use_container_width=True)


# ==================== DESCRIPTIVE ANALYTICS TAB ====================
def show_descriptive_analytics(df):
    st.header("📈 Descriptive Analytics")
    
    # Service Aspect Scores
    st.subheader("🎯 Service Aspect Performance")
    service_cols = ['Value', 'Rooms', 'Location', 'Cleanliness', 'Service', 'Sleep Quality']
    avg_scores = df[service_cols].mean().sort_values(ascending=False)
    
    fig = px.bar(
        x=avg_scores.values,
        y=avg_scores.index,
        orientation='h',
        text=avg_scores.values.round(2),
        labels={'x': 'Average Score (0-5)', 'y': 'Service Aspect'},
        title='Average Scores by Service Aspect'
    )
    fig.update_traces(textposition='outside', marker_color='lightblue')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display scores as table
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏆 Top Performers")
        for i, (aspect, score) in enumerate(avg_scores.head(3).items(), 1):
            st.success(f"{i}. **{aspect}**: {score:.2f}/5.0")
    
    with col2:
        st.markdown("### ⚠️ Areas for Improvement")
        for i, (aspect, score) in enumerate(avg_scores.tail(3).items(), 1):
            st.warning(f"{i}. **{aspect}**: {score:.2f}/5.0")
    
    st.markdown("---")
    
    # Bivariate Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⭐ Rating vs Trip Type")
        fig = px.box(
            df, x='Trip Type', y='Rating',
            color='Trip Type',
            labels={'Rating': 'Rating (1-5)'},
            title='Rating Distribution by Trip Type'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📅 Rating by Season")
        season_data = df.groupby('Season')['Rating'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=season_data.index,
            y=season_data.values,
            text=season_data.values.round(2),
            labels={'x': 'Season', 'y': 'Average Rating'},
            title='Average Rating by Season'
        )
        fig.update_traces(textposition='outside', marker_color=['#2ca02c', '#ff7f0e'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Review Length vs Rating
    st.subheader("📝 Review Length vs Rating")
    fig = px.scatter(
        df, x='Review_Length_Words', y='Rating',
        color='Sentiment',
        trendline='ols',
        labels={'Review_Length_Words': 'Review Length (words)', 'Rating': 'Rating (1-5)'},
        title='Relationship between Review Length and Rating'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    corr = df[['Review_Length_Words', 'Rating']].corr().iloc[0, 1]
    if corr < 0:
        st.info(f"📊 Correlation: {corr:.3f} - Longer reviews tend to have lower ratings")
    else:
        st.info(f"📊 Correlation: {corr:.3f} - Longer reviews tend to have higher ratings")


# ==================== DIAGNOSTIC ANALYTICS TAB ====================
def show_diagnostic_analytics(df):
    st.header("🔬 Diagnostic Analytics")
    
    # Correlation Analysis
    st.subheader("🔗 Correlation Analysis")
    corr_cols = ['Rating', 'Value', 'Rooms', 'Location', 'Cleanliness', 'Service', 'Sleep Quality']
    corr_matrix = df[corr_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r',
        title='Correlation Heatmap: Rating vs Service Aspects',
        text_auto='.2f'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top correlations
    rating_corr = corr_matrix['Rating'].drop('Rating').sort_values(ascending=False)
    st.markdown("### 🎯 Correlation with Overall Rating")
    for aspect, corr in rating_corr.items():
        emoji = '🔴' if corr > 0.5 else '🟡' if corr > 0.3 else '🟢'
        st.write(f"{emoji} **{aspect}**: {corr:+.3f}")
    
    st.markdown("---")
    
    # Text Mining
    st.subheader("☁️ Word Cloud Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Positive Reviews (5 Stars)")
        positive_reviews = df[df['Rating'] == 5]['Review Text'].str.cat(sep=' ')
        positive_reviews = positive_reviews.lower()
        positive_reviews = re.sub(r'[^a-z\s]', '', positive_reviews)
        
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'is', 'was', 'were', 'are', 'been', 'be', 'have', 'has',
                    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
                    'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'we',
                    'you', 'he', 'she', 'it', 'they', 'them', 'their', 'my', 'your', 'his',
                    'her', 'its', 'our', 'resort', 'hotel', 'stay', 'stayed', 'so', 'more',
                    'not', 'which', 'there', 'out', 'again', 'make', 'made', 'just', 'very',
                    'salalah', 'alway', 'about', 'what', 'up', 'if', 'anantara', 'need', 'u',
                    'go', 'by', 'al baleed'}
        
        wordcloud_positive = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Greens',
            max_words=100,
            stopwords=stopwords
        ).generate(positive_reviews)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud_positive, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### ❌ Negative Reviews (1-2 Stars)")
        if len(df[df['Rating'] <= 2]) > 5:
            negative_reviews = df[df['Rating'] <= 2]['Review Text'].str.cat(sep=' ')
            negative_reviews = negative_reviews.lower()
            negative_reviews = re.sub(r'[^a-z\s]', '', negative_reviews)
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                        'of', 'with', 'is', 'was', 'were', 'are', 'been', 'be', 'have', 'has',
                        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
                        'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'we',
                        'you', 'he', 'she', 'it', 'they', 'them', 'their', 'my', 'your', 'his',
                        'her', 'its', 'our', 'resort', 'hotel', 'stay', 'stayed', 'no', 'not', 'if',
                        'anantara', 'about', 'if', 'than', 'as', 'such', 'so', 'only', 'like',
                        'from', 'us', 'very', 'al baleed', 'salalah', 'until', 'got', 'then', 'up',
                        'just', 'really', 'go', 'out', 'very', 'since', 'there', 'nice', 'by', 'which',
                        'excellence', 'any', 'one', 'what', 'when', 'without', 'however', 'back',
                        }
            
            wordcloud_negative = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap='Reds',
                max_words=100,
                stopwords=stopwords
            ).generate(negative_reviews)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud_negative, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("Not enough negative reviews for word cloud analysis")
    
    # N-Grams Analysis
    st.markdown("---")
    st.subheader("📊 Most Common Phrases (Bi-grams)")

    def get_bigrams(text, n=15):
        words = text.lower().split()
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'were', 'very',
                    'this', 'that', 'our', 'baleed', 'resort', 'you', 'can', 'are', 'like', 'will', 'have', 'been', 'would', 'they'}
        words = [w for w in words if w not in stopwords_list and len(w) > 2]
        bigrams = [' '.join([words[i], words[i+1]]) for i in range(len(words)-1)]
        return Counter(bigrams).most_common(n)

    # Create two columns for positive vs negative bigrams
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ Positive Reviews (5 Stars)")
        positive_bigrams = get_bigrams(positive_reviews, 15)
        bigrams_df_pos = pd.DataFrame(positive_bigrams, columns=['Bigram', 'Count'])

        fig = px.bar(
            bigrams_df_pos, x='Count', y='Bigram', orientation='h',
            title='Top 15 Phrases in Positive Reviews',
            labels={'Count': 'Frequency'},
            color_discrete_sequence=['#2ca02c']
        )
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### ❌ Negative Reviews (1-2 Stars)")
        if len(df[df['Rating'] <= 2]) > 5:
            # Get negative reviews text
            negative_reviews_text = df[df['Rating'] <= 2]['Review Text'].str.cat(sep=' ')
            negative_reviews_text = negative_reviews_text.lower()
            negative_reviews_text = re.sub(r'[^a-z\s]', '', negative_reviews_text)

            negative_bigrams = get_bigrams(negative_reviews_text, 15)
            bigrams_df_neg = pd.DataFrame(negative_bigrams, columns=['Bigram', 'Count'])

            fig = px.bar(
                bigrams_df_neg, x='Count', y='Bigram', orientation='h',
                title='Top 15 Phrases in Negative Reviews',
                labels={'Count': 'Frequency'},
                color_discrete_sequence=['#d62728']
            )
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough negative reviews for bigram analysis (need at least 5 reviews)")

    # Key insights from bigrams
    st.markdown("---")
    st.markdown("### 💡 Bigram Insights")

    col1, col2 = st.columns(2)
    with col1:
        if len(positive_bigrams) > 0:
            st.success(f"**Most Common Positive Phrase**: '{positive_bigrams[0][0]}' ({positive_bigrams[0][1]} times)")
            st.write("**Top 5 Positive Phrases:**")
            for i, (bigram, count) in enumerate(positive_bigrams[:5], 1):
                st.write(f"{i}. {bigram} ({count}x)")

    with col2:
        if len(df[df['Rating'] <= 2]) > 5:
            if len(negative_bigrams) > 0:
                st.error(f"**Most Common Negative Phrase**: '{negative_bigrams[0][0]}' ({negative_bigrams[0][1]} times)")
                st.write("**Top 5 Negative Phrases:**")
                for i, (bigram, count) in enumerate(negative_bigrams[:5], 1):
                    st.write(f"{i}. {bigram} ({count}x)")
        else:
            st.info("Not enough negative reviews for analysis")
    
    # Root Cause Analysis
    st.markdown("---")
    st.subheader("🔴 Root Cause Analysis: Negative Reviews")
    
    negative_df = df[df['Rating'] <= 2]
    
    if len(negative_df) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Negative Reviews", len(negative_df))
        with col2:
            neg_pct = (len(negative_df) / len(df)) * 100
            st.metric("Percentage", f"{neg_pct:.1f}%")
        with col3:
            avg_neg_rating = negative_df['Rating'].mean()
            st.metric("Average Rating", f"{avg_neg_rating:.2f}")
        
        st.markdown("#### 📉 Service Scores in Negative Reviews")
        service_cols = ['Value', 'Rooms', 'Location', 'Cleanliness', 'Service', 'Sleep Quality']
        neg_scores = negative_df[service_cols].mean().sort_values()
        
        fig = px.bar(
            x=neg_scores.values,
            y=neg_scores.index,
            orientation='h',
            text=neg_scores.values.round(2),
            labels={'x': 'Average Score', 'y': 'Aspect'},
            title='Service Aspect Scores in Negative Reviews',
            color=neg_scores.values,
            color_continuous_scale='Reds_r'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.warning(f"⚠️ **Most Problematic Area**: {neg_scores.index[0]} ({neg_scores.iloc[0]:.2f}/5.0)")
    else:
        st.success("✅ Excellent! Very few negative reviews to analyze.")


# ==================== PREDICTIVE ANALYTICS TAB ====================
def show_predictive_analytics(df):
    st.header("🤖 Predictive Analytics")

    # Comprehensive data validation
    if len(df) < 20:
        st.error("❌ Not enough data for machine learning model. Please adjust filters to include at least 20 reviews.")
        st.info("💡 **Tip:** Try expanding your date range or removing some filters.")
        return

    if len(df) < 50:
        st.warning("⚠️ Limited data detected. Model may not be very accurate. Recommended: 50+ reviews for reliable results.")

    # Check rating distribution
    rating_dist = df['Rating'].value_counts()
    if len(rating_dist) < 2:
        st.error("❌ Only one rating class found in filtered data. ML model needs variety in ratings.")
        st.info("💡 **Tip:** Adjust rating filter to include multiple rating levels (e.g., 1-5 stars).")
        return

    # Create sub-tabs for different models
    model_tabs = st.tabs(["🌲 Random Forest", "🎯 SVM", "🧠 LSTM Deep Learning", "📊 Model Comparison"])

    with model_tabs[0]:
        show_random_forest(df)

    with model_tabs[1]:
        show_svm_model(df)

    # with model_tabs[2]:
    #     if TENSORFLOW_AVAILABLE:
    #         show_lstm_models(df)
    #     else:
    #         st.error("❌ TensorFlow is not installed. Please install it to use LSTM models.")
    #         st.code("pip install tensorflow", language="bash")

    # with model_tabs[3]:
    #     show_model_comparison(df)


# ==================== RANDOM FOREST MODEL ====================
def show_random_forest(df):
    st.subheader("🌲 Random Forest Classifier")
    st.markdown("Ensemble learning method for classification based on service scores and features.")

    # Check for classes with very few samples
    rating_dist = df['Rating'].value_counts()
    min_samples = rating_dist.min()
    if min_samples == 1:
        st.warning(f"⚠️ Warning: Some rating classes have only 1 sample. Model may be less reliable.")
        st.info("💡 **Current distribution:**")
        st.dataframe(rating_dist.to_frame(name='Count'))

    with st.spinner("🔄 Training Random Forest model..."):
        # Prepare data
        df_ml = df.copy()

        # Encode categorical variables
        le_trip = LabelEncoder()
        df_ml['Trip_Type_Encoded'] = le_trip.fit_transform(df_ml['Trip Type'])

        le_season = LabelEncoder()
        df_ml['Season_Encoded'] = le_season.fit_transform(df_ml['Season'])

        # Select features
        feature_cols = ['Value', 'Rooms', 'Location', 'Cleanliness', 'Service', 'Sleep Quality',
                        'Trip_Type_Encoded', 'Season_Encoded', 'Review_Length_Words',
                        'Is_Weekend', 'Month']

        X = df_ml[feature_cols]
        y = df_ml['Rating']

        # Check if we have multiple classes
        if len(y.unique()) < 2:
            st.warning("⚠️ Filtered data contains only one rating class. Please adjust filters to include more variety in ratings.")
            return

        # Check class distribution for stratification
        class_counts = y.value_counts()
        min_class_count = class_counts.min()

        # Adaptive test size
        test_size = min(0.2, max(0.1, 20 / len(df)))

        # Use stratify only if all classes have at least 2 samples
        if min_class_count >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            # Disable stratify for small class counts
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            st.info(f"ℹ️ Note: Stratified sampling disabled due to small class sizes (minimum: {min_class_count} samples).")

        # Train model
        start_time = time.time()
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Predictions
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Training Samples", f"{len(X_train):,}")
    with col2:
        st.metric("Test Samples", f"{len(X_test):,}")
    with col3:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col4:
        st.metric("F1-Score", f"{f1:.2%}")
    with col5:
        st.metric("Training Time", f"{training_time:.2f}s")

    st.markdown("---")

    # Feature Importance
    st.subheader("🎯 Feature Importance Analysis")
    st.markdown("**Understanding which factors drive guest satisfaction:**")

    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig = px.bar(
        feature_importance, x='Importance', y='Feature', orientation='h',
        title='Feature Importance: What Drives Guest Ratings?',
        labels={'Importance': 'Importance Score'},
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 Top 3 Most Important Factors:")
    for i, row in feature_importance.head(3).iterrows():
        st.success(f"**{feature_importance.index.get_loc(i)+1}. {row['Feature']}**: {row['Importance']:.3f}")

    # Confusion Matrix
    st.markdown("---")
    st.subheader("📊 Model Performance: Confusion Matrix")

    # Get unique labels from test data
    unique_labels = sorted(y_test.unique())

    # Create confusion matrix with proper labels
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    # Create label strings
    label_strings = [f"{int(label)}" for label in unique_labels]

    fig = px.imshow(
        cm,
        labels=dict(x="Predicted Rating", y="Actual Rating", color="Count"),
        x=label_strings,
        y=label_strings,
        color_continuous_scale='Blues',
        text_auto=True,
        title='Confusion Matrix'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Classification Report
    with st.expander("📈 Detailed Classification Report"):
        try:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0))
        except Exception as e:
            st.warning(f"Could not generate detailed report: {str(e)}")
            st.write("**Basic Metrics:**")
            st.write(f"- Accuracy: {accuracy:.2%}")
            st.write(f"- Precision: {precision:.2%}")
            st.write(f"- Recall: {recall:.2%}")
            st.write(f"- F1-Score: {f1:.2%}")


# ==================== SVM MODEL ====================
def show_svm_model(df):
    st.subheader("🎯 Support Vector Machine (SVM)")
    st.markdown("Support Vector Machine classifier with RBF kernel for non-linear classification.")

    # Check for classes with very few samples
    rating_dist = df['Rating'].value_counts()
    min_samples = rating_dist.min()
    if min_samples == 1:
        st.warning(f"⚠️ Warning: Some rating classes have only 1 sample. Model may be less reliable.")
        st.info("💡 **Current distribution:**")
        st.dataframe(rating_dist.to_frame(name='Count'))

    with st.spinner("🔄 Training SVM model..."):
        # Prepare data
        df_ml = df.copy()

        # Encode categorical variables
        le_trip = LabelEncoder()
        df_ml['Trip_Type_Encoded'] = le_trip.fit_transform(df_ml['Trip Type'])

        le_season = LabelEncoder()
        df_ml['Season_Encoded'] = le_season.fit_transform(df_ml['Season'])

        # Select features
        feature_cols = ['Value', 'Rooms', 'Location', 'Cleanliness', 'Service', 'Sleep Quality',
                        'Trip_Type_Encoded', 'Season_Encoded', 'Review_Length_Words',
                        'Is_Weekend', 'Month']

        X = df_ml[feature_cols]
        y = df_ml['Rating']

        # Check if we have multiple classes
        if len(y.unique()) < 2:
            st.warning("⚠️ Filtered data contains only one rating class. Please adjust filters to include more variety in ratings.")
            return

        # Scale features (important for SVM)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Check class distribution for stratification
        class_counts = y.value_counts()
        min_class_count = class_counts.min()

        # Adaptive test size
        test_size = min(0.2, max(0.1, 20 / len(df)))

        # Use stratify only if all classes have at least 2 samples
        if min_class_count >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )
            st.info(f"ℹ️ Note: Stratified sampling disabled due to small class sizes (minimum: {min_class_count} samples).")

        # Train model with optimized parameters
        start_time = time.time()
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm_model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Predictions
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Training Samples", f"{len(X_train):,}")
    with col2:
        st.metric("Test Samples", f"{len(X_test):,}")
    with col3:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col4:
        st.metric("F1-Score", f"{f1:.2%}")
    with col5:
        st.metric("Training Time", f"{training_time:.2f}s")

    st.markdown("---")

    # SVM Parameters Info
    st.subheader("⚙️ Model Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Kernel**: RBF (Radial Basis Function)")
    with col2:
        st.info(f"**C Parameter**: 1.0")
    with col3:
        st.info(f"**Gamma**: scale")

    st.markdown("""
    **SVM Model Characteristics:**
    - Uses RBF kernel for non-linear decision boundaries
    - Feature scaling applied (StandardScaler)
    - Effective for high-dimensional data
    - Memory efficient for smaller datasets
    """)

    # Confusion Matrix
    st.markdown("---")
    st.subheader("📊 Model Performance: Confusion Matrix")

    # Get unique labels from test data
    unique_labels = sorted(y_test.unique())

    # Create confusion matrix with proper labels
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    # Create label strings
    label_strings = [f"{int(label)}" for label in unique_labels]

    fig = px.imshow(
        cm,
        labels=dict(x="Predicted Rating", y="Actual Rating", color="Count"),
        x=label_strings,
        y=label_strings,
        color_continuous_scale='Greens',
        text_auto=True,
        title='SVM Confusion Matrix'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Classification Report
    with st.expander("📈 Detailed Classification Report"):
        try:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0))
        except Exception as e:
            st.warning(f"Could not generate detailed report: {str(e)}")
            st.write("**Basic Metrics:**")
            st.write(f"- Accuracy: {accuracy:.2%}")
            st.write(f"- Precision: {precision:.2%}")
            st.write(f"- Recall: {recall:.2%}")
            st.write(f"- F1-Score: {f1:.2%}")

    # Performance comparison with baseline
    st.markdown("---")
    st.subheader("📈 Performance Insights")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Strengths:**")
        st.markdown("- Excellent for non-linear patterns")
        st.markdown("- Memory efficient")
        st.markdown("- Robust to outliers")

    with col2:
        st.markdown("**⚠️ Considerations:**")
        st.markdown("- Requires feature scaling")
        st.markdown("- Slower on large datasets")
        st.markdown("- Less interpretable than tree models")


# ==================== LSTM MODELS ====================
def show_lstm_models(df):
    st.subheader("🧠 LSTM Deep Learning Models")
    st.markdown("Long Short-Term Memory networks for text-based sentiment and rating prediction.")

    # Create sub-tabs for sentiment and rating prediction
    lstm_tabs = st.tabs(["💭 Sentiment Prediction", "⭐ Rating Prediction"])

    with lstm_tabs[0]:
        show_lstm_sentiment(df)

    with lstm_tabs[1]:
        show_lstm_rating(df)


def show_lstm_sentiment(df):
    st.markdown("### Sentiment Classification from Review Text")
    st.markdown("Predicts: Positive, Neutral, or Negative sentiment based on review content.")

    # Check data requirements
    if len(df) < 30:
        st.warning("⚠️ LSTM requires at least 30 reviews for reliable training.")
        return

    sentiment_dist = df['Sentiment'].value_counts()
    if len(sentiment_dist) < 2:
        st.error("❌ Need multiple sentiment classes for training.")
        return

    st.info("📊 **Sentiment Distribution:**")
    st.dataframe(sentiment_dist.to_frame(name='Count'))

    # Training configuration
    with st.expander("⚙️ Model Configuration"):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_words = st.number_input("Max Vocabulary Size", 1000, 10000, 5000, 500)
        with col2:
            max_len = st.number_input("Max Sequence Length", 50, 500, 100, 50)
        with col3:
            epochs = st.number_input("Training Epochs", 1, 20, 5, 1)

    if st.button("🚀 Train LSTM Sentiment Model", key="train_lstm_sentiment"):
        with st.spinner("🔄 Training LSTM model... This may take a few minutes."):
            try:
                # Prepare text data
                texts = df['Review Text'].fillna('').values
                labels = df['Sentiment'].values

                # Encode labels
                le = LabelEncoder()
                y_encoded = le.fit_transform(labels)
                num_classes = len(le.classes_)

                # Tokenize texts
                tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
                tokenizer.fit_on_texts(texts)
                sequences = tokenizer.texts_to_sequences(texts)
                X_padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_padded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )

                # Build LSTM model
                model = Sequential([
                    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
                    Bidirectional(LSTM(64, return_sequences=True)),
                    Dropout(0.5),
                    Bidirectional(LSTM(32)),
                    Dropout(0.5),
                    Dense(64, activation='relu'),
                    Dropout(0.3),
                    Dense(num_classes, activation='softmax')
                ])

                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )

                # Train model
                start_time = time.time()
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
                )
                training_time = time.time() - start_time

                # Predictions
                y_pred_probs = model.predict(X_test, verbose=0)
                y_pred = np.argmax(y_pred_probs, axis=1)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                # Display metrics
                st.success("✅ Training completed successfully!")

                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Training Samples", f"{len(X_train):,}")
                with col2:
                    st.metric("Test Samples", f"{len(X_test):,}")
                with col3:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with col4:
                    st.metric("F1-Score", f"{f1:.2%}")
                with col5:
                    st.metric("Training Time", f"{training_time:.1f}s")

                # Training history
                st.markdown("---")
                st.subheader("📈 Training History")

                fig = make_subplots(rows=1, cols=2, subplot_titles=('Accuracy', 'Loss'))

                fig.add_trace(
                    go.Scatter(y=history.history['accuracy'], name='Train Accuracy', mode='lines'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(y=history.history['val_accuracy'], name='Val Accuracy', mode='lines'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(y=history.history['loss'], name='Train Loss', mode='lines'),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(y=history.history['val_loss'], name='Val Loss', mode='lines'),
                    row=1, col=2
                )

                fig.update_xaxes(title_text="Epoch", row=1, col=1)
                fig.update_xaxes(title_text="Epoch", row=1, col=2)
                fig.update_yaxes(title_text="Accuracy", row=1, col=1)
                fig.update_yaxes(title_text="Loss", row=1, col=2)
                fig.update_layout(height=400)

                st.plotly_chart(fig, use_container_width=True)

                # Confusion Matrix
                st.markdown("---")
                st.subheader("📊 Confusion Matrix")

                cm = confusion_matrix(y_test, y_pred)
                label_names = le.classes_

                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted Sentiment", y="Actual Sentiment", color="Count"),
                    x=label_names,
                    y=label_names,
                    color_continuous_scale='Purples',
                    text_auto=True,
                    title='LSTM Sentiment Prediction Confusion Matrix'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Classification Report
                with st.expander("📈 Detailed Classification Report"):
                    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.highlight_max(axis=0))

            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                st.info("💡 Try reducing the number of epochs or adjusting other parameters.")


def show_lstm_rating(df):
    st.markdown("### Rating Prediction from Review Text")
    st.markdown("Predicts: Rating (1-5 stars) based on review content.")

    # Check data requirements
    if len(df) < 30:
        st.warning("⚠️ LSTM requires at least 30 reviews for reliable training.")
        return

    rating_dist = df['Rating'].value_counts()
    if len(rating_dist) < 2:
        st.error("❌ Need multiple rating classes for training.")
        return

    st.info("📊 **Rating Distribution:**")
    st.dataframe(rating_dist.to_frame(name='Count'))

    # Training configuration
    with st.expander("⚙️ Model Configuration"):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_words = st.number_input("Max Vocabulary Size", 1000, 10000, 5000, 500, key="rating_max_words")
        with col2:
            max_len = st.number_input("Max Sequence Length", 50, 500, 150, 50, key="rating_max_len")
        with col3:
            epochs = st.number_input("Training Epochs", 1, 20, 5, 1, key="rating_epochs")

    if st.button("🚀 Train LSTM Rating Model", key="train_lstm_rating"):
        with st.spinner("🔄 Training LSTM model... This may take a few minutes."):
            try:
                # Prepare text data
                texts = df['Review Text'].fillna('').values
                labels = df['Rating'].values

                # Encode labels (ratings are already numeric, but ensure they're 0-indexed for categorical)
                unique_ratings = sorted(df['Rating'].unique())
                rating_to_idx = {rating: idx for idx, rating in enumerate(unique_ratings)}
                idx_to_rating = {idx: rating for rating, idx in rating_to_idx.items()}

                y_encoded = np.array([rating_to_idx[r] for r in labels])
                num_classes = len(unique_ratings)

                # Tokenize texts
                tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
                tokenizer.fit_on_texts(texts)
                sequences = tokenizer.texts_to_sequences(texts)
                X_padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_padded, y_encoded, test_size=0.2, random_state=42,
                    stratify=y_encoded if min(np.bincount(y_encoded)) >= 2 else None
                )

                # Build LSTM model
                model = Sequential([
                    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
                    Bidirectional(LSTM(64, return_sequences=True)),
                    Dropout(0.5),
                    Bidirectional(LSTM(32)),
                    Dropout(0.5),
                    Dense(64, activation='relu'),
                    Dropout(0.3),
                    Dense(num_classes, activation='softmax')
                ])

                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )

                # Train model
                start_time = time.time()
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
                )
                training_time = time.time() - start_time

                # Predictions
                y_pred_probs = model.predict(X_test, verbose=0)
                y_pred = np.argmax(y_pred_probs, axis=1)

                # Convert back to original ratings for display
                y_test_ratings = [idx_to_rating[idx] for idx in y_test]
                y_pred_ratings = [idx_to_rating[idx] for idx in y_pred]

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                # Display metrics
                st.success("✅ Training completed successfully!")

                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Training Samples", f"{len(X_train):,}")
                with col2:
                    st.metric("Test Samples", f"{len(X_test):,}")
                with col3:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with col4:
                    st.metric("F1-Score", f"{f1:.2%}")
                with col5:
                    st.metric("Training Time", f"{training_time:.1f}s")

                # Training history
                st.markdown("---")
                st.subheader("📈 Training History")

                fig = make_subplots(rows=1, cols=2, subplot_titles=('Accuracy', 'Loss'))

                fig.add_trace(
                    go.Scatter(y=history.history['accuracy'], name='Train Accuracy', mode='lines'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(y=history.history['val_accuracy'], name='Val Accuracy', mode='lines'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(y=history.history['loss'], name='Train Loss', mode='lines'),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(y=history.history['val_loss'], name='Val Loss', mode='lines'),
                    row=1, col=2
                )

                fig.update_xaxes(title_text="Epoch", row=1, col=1)
                fig.update_xaxes(title_text="Epoch", row=1, col=2)
                fig.update_yaxes(title_text="Accuracy", row=1, col=1)
                fig.update_yaxes(title_text="Loss", row=1, col=2)
                fig.update_layout(height=400)

                st.plotly_chart(fig, use_container_width=True)

                # Confusion Matrix
                st.markdown("---")
                st.subheader("📊 Confusion Matrix")

                cm = confusion_matrix(y_test, y_pred)
                label_names = [str(int(r)) for r in unique_ratings]

                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted Rating", y="Actual Rating", color="Count"),
                    x=label_names,
                    y=label_names,
                    color_continuous_scale='Purples',
                    text_auto=True,
                    title='LSTM Rating Prediction Confusion Matrix'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Classification Report
                with st.expander("📈 Detailed Classification Report"):
                    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.highlight_max(axis=0))

                # Sample predictions
                st.markdown("---")
                st.subheader("🔍 Sample Predictions")

                sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
                samples_df = pd.DataFrame({
                    'Review Text': [texts[i] for i in sample_indices],
                    'Actual Rating': [y_test_ratings[i] for i in range(len(y_test)) if i in sample_indices],
                    'Predicted Rating': [y_pred_ratings[i] for i in range(len(y_pred)) if i in sample_indices]
                })
                st.dataframe(samples_df)

            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                st.info("💡 Try reducing the number of epochs or adjusting other parameters.")


# ==================== MODEL COMPARISON ====================
def show_model_comparison(df):
    st.subheader("📊 Model Performance Comparison")
    st.markdown("Compare accuracy and performance metrics across all three models.")

    st.info("ℹ️ **Note**: Train each model in their respective tabs to see comprehensive comparison metrics here.")

    # Theoretical comparison table
    st.markdown("### 🎯 Model Characteristics Comparison")

    comparison_data = {
        'Model': ['Random Forest', 'SVM', 'LSTM (Sentiment)', 'LSTM (Rating)'],
        'Type': ['Ensemble Learning', 'Support Vector', 'Deep Learning', 'Deep Learning'],
        'Input Features': ['Numerical + Categorical', 'Numerical + Categorical', 'Text', 'Text'],
        'Interpretability': ['High (Feature Importance)', 'Medium', 'Low (Black Box)', 'Low (Black Box)'],
        'Training Speed': ['Fast', 'Medium', 'Slow', 'Slow'],
        'Best For': ['Structured data', 'Non-linear patterns', 'Sentiment analysis', 'Rating from text'],
        'Pros': ['Easy to interpret', 'Robust to outliers', 'Captures context', 'Understands semantics'],
        'Cons': ['May overfit', 'Requires scaling', 'Needs more data', 'Computationally expensive']
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    st.markdown("---")

    # Visualization of model types
    st.markdown("### 🔍 When to Use Each Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🌲 Random Forest**")
        st.success("✅ Default choice for tabular data")
        st.markdown("""
        - Best accuracy for feature-based prediction
        - Provides feature importance
        - No data scaling required
        - Works well with mixed data types
        """)

    with col2:
        st.markdown("**🎯 SVM**")
        st.info("✅ When you have clear decision boundaries")
        st.markdown("""
        - Excellent for non-linear patterns
        - Memory efficient
        - Good for smaller datasets
        - Requires feature engineering
        """)

    with col3:
        st.markdown("**🧠 LSTM**")
        st.warning("✅ When text content matters most")
        st.markdown("""
        - Understands review semantics
        - Captures sequential patterns
        - No manual feature engineering
        - Requires more training data
        """)

    st.markdown("---")

    # Recommendations
    st.markdown("### 💡 Recommendations")

    st.markdown("""
    **For this Hotel Review Dataset:**

    1. **Use Random Forest** when:
       - You want to predict ratings from service scores (Value, Rooms, Location, etc.)
       - You need to understand which features drive ratings
       - You want fast, reliable predictions

    2. **Use SVM** when:
       - You suspect non-linear relationships between features
       - You want to validate Random Forest results
       - Memory efficiency is important

    3. **Use LSTM** when:
       - You want to predict sentiment/rating directly from review text
       - You have sufficient training data (50+ reviews per class)
       - You care about understanding what guests say, not just numerical scores
       - You want to analyze reviews without manual feature extraction

    **Ensemble Approach:** For best results, combine predictions from multiple models!
    """)


# ==================== PRESCRIPTIVE ANALYTICS TAB ====================
def show_prescriptive_analytics(df):
    st.header("💼 Prescriptive Analytics: Actionable Recommendations")
    
    # Generate insights
    service_cols = ['Value', 'Rooms', 'Location', 'Cleanliness', 'Service', 'Sleep Quality']
    avg_scores = df[service_cols].mean()
    low_performers = avg_scores.sort_values().head(3)
    trip_satisfaction = df.groupby('Trip Type')['Rating'].mean().sort_values()
    
    # Key Insights
    st.subheader("📊 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Satisfaction", f"{(len(df[df['Rating'] >= 4]) / len(df) * 100):.1f}%")
    with col2:
        st.metric("Average Rating", f"{df['Rating'].mean():.2f}/5.0")
    with col3:
        st.metric("Lowest Scoring Aspect", low_performers.index[0])
    
    st.markdown("---")
    
    # Strategic Recommendations
    st.subheader("🎯 Strategic Recommendations")
    
    recommendations = []
    
    # Based on low performers
    if low_performers.iloc[0] < 3.5:
        recommendations.append({
            'Priority': 'HIGH',
            'Department': 'Operations',
            'Action': f'Urgent: Improve {low_performers.index[0]}',
            'Current Score': f'{low_performers.iloc[0]:.2f}/5.0',
            'Target': '4.0/5.0',
            'Timeline': '30 days'
        })
    
    # Based on trip type
    if trip_satisfaction.iloc[0] < df['Rating'].mean():
        recommendations.append({
            'Priority': 'MEDIUM',
            'Department': 'Marketing & Guest Services',
            'Action': f'Develop programs for {trip_satisfaction.index[0]} travelers',
            'Current Score': f'{trip_satisfaction.iloc[0]:.2f}/5.0',
            'Target': f'{df["Rating"].mean():.2f}/5.0',
            'Timeline': '60 days'
        })
    
    # Based on negative reviews
    neg_count = len(df[df['Rating'] <= 2])
    if neg_count > 0:
        recommendations.append({
            'Priority': 'HIGH',
            'Department': 'Quality Assurance',
            'Action': 'Implement proactive complaint resolution',
            'Current Score': f'{neg_count} negative reviews',
            'Target': '< 5 negative reviews/month',
            'Timeline': '45 days'
        })
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        priority_color = 'red' if rec['Priority'] == 'HIGH' else 'orange'
        with st.container():
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; 
                        border-left: 5px solid {priority_color}; margin-bottom: 1rem;'>
                <h4>Recommendation #{i}: {rec['Action']}</h4>
                <p><strong>Priority:</strong> <span style='color: {priority_color};'>{rec['Priority']}</span></p>
                <p><strong>Department:</strong> {rec['Department']}</p>
                <p><strong>Current:</strong> {rec['Current Score']} | <strong>Target:</strong> {rec['Target']}</p>
                <p><strong>Timeline:</strong> {rec['Timeline']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Department-Specific Actions
    st.subheader("📋 Department-Specific Action Plans")
    
    tabs = st.tabs(["👔 GM", "🏨 Front Office", "🧹 Housekeeping", "🔧 Maintenance", "📢 Marketing"])
    
    with tabs[0]:  # GM
        st.markdown("### General Manager Action Items")
        st.markdown(f"""
        - **Overall Performance**: {(len(df[df['Rating'] >= 4]) / len(df) * 100):.1f}% positive reviews (Excellent!)
        - **Strategic Focus**: Invest in top 3 service aspects with highest ROI
        - **Budget Allocation**: Prioritize improvements in {low_performers.index[0]}
        - **Staff Recognition**: Implement incentive programs based on guest satisfaction metrics
        """)
    
    with tabs[1]:  # Front Office
        service_score = df['Service'].mean()
        st.markdown("### Front Office Manager Action Items")
        st.markdown(f"""
        - **Current Service Score**: {service_score:.2f}/5.0
        - **Action 1**: Develop personalized welcome procedures by trip type
        - **Action 2**: Implement 24-hour review response protocol
        - **Action 3**: Cross-train staff on guest service excellence
        """)
    
    with tabs[2]:  # Housekeeping
        clean_score = df['Cleanliness'].mean()
        room_score = df['Rooms'].mean()
        st.markdown("### Housekeeping Manager Action Items")
        st.markdown(f"""
        - **Cleanliness Score**: {clean_score:.2f}/5.0
        - **Room Quality Score**: {room_score:.2f}/5.0
        - **Action 1**: Review and update cleaning protocols
        - **Action 2**: Implement daily room quality audits
        - **Action 3**: Staff training on attention to detail
        """)
    
    with tabs[3]:  # Maintenance
        sleep_score = df['Sleep Quality'].mean()
        st.markdown("### Maintenance Manager Action Items")
        st.markdown(f"""
        - **Sleep Quality Score**: {sleep_score:.2f}/5.0
        - **Action 1**: Conduct comprehensive HVAC system review
        - **Action 2**: Implement preventive maintenance schedule
        - **Action 3**: Address noise issues (soundproofing, door seals)
        """)
    
    with tabs[4]:  # Marketing
        st.markdown("### Marketing Manager Action Items")
        st.markdown(f"""
        - **Leverage Success**: {(len(df[df['Rating'] >= 4]) / len(df) * 100):.0f}% positive reviews in campaigns
        - **Target Segments**: Create packages for {trip_satisfaction.index[0]} travelers
        - **Seasonal Strategy**: Emphasize Khareef season experience
        - **Loyalty Program**: Develop rewards based on guest preferences
        """)


# ==================== RAW DATA TAB ====================
def show_raw_data(df):
    st.header("📝 Raw Data Explorer")
    
    # Display controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_columns = st.multiselect(
            "Select columns to display",
            options=df.columns.tolist(),
            default=['Rating', 'Trip Type', 'Stay Date', 'Review Title', 'Sentiment']
        )
    
    with col2:
        sort_by = st.selectbox("Sort by", options=df.columns.tolist(), index=2)
    
    with col3:
        sort_order = st.radio("Order", options=['Descending', 'Ascending'])
    
    # Apply sorting
    ascending = True if sort_order == 'Ascending' else False
    df_display = df[show_columns].sort_values(by=sort_by, ascending=ascending)
    
    # Display data
    st.dataframe(df_display, use_container_width=True, height=600)
    
    # Download button
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="al_baleed_filtered_data.csv",
        mime="text/csv"
    )
    
    # Statistics
    st.markdown("---")
    st.subheader("📊 Quick Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df_display))
    with col2:
        st.metric("Columns", len(show_columns))
    with col3:
        if 'Rating' in show_columns:
            st.metric("Avg Rating", f"{df_display['Rating'].mean():.2f}")
    with col4:
        if 'Review_Length_Words' in df.columns and 'Review_Length_Words' in show_columns:
            st.metric("Avg Review Length", f"{df_display['Review_Length_Words'].mean():.0f} words")


# ==================== RUN APP ====================
if __name__ == "__main__":
    main()