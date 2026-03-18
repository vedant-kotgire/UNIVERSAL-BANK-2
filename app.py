import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import io
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank — Personal Loan Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1400px;
    }

    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f3a 50%, #0d1525 100%);
        border: 1px solid rgba(0, 102, 255, 0.2);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(0,102,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-header h1 {
        font-size: 1.9rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: #94A3B8;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        flex: 1;
        transition: border-color 0.2s;
    }
    .metric-card:hover {
        border-color: rgba(0,102,255,0.4);
    }
    .metric-label {
        font-size: 0.75rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F9FAFB;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #9CA3AF;
        margin-top: 0.2rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F9FAFB;
        margin: 2rem 0 0.3rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0066FF;
        display: inline-block;
    }
    .section-desc {
        color: #9CA3AF;
        font-size: 0.88rem;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }

    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, #0c1a2e 0%, #111827 100%);
        border-left: 3px solid #0066FF;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.3rem;
        margin: 0.8rem 0 1.2rem 0;
        font-size: 0.88rem;
        color: #D1D5DB;
        line-height: 1.6;
    }
    .insight-box strong {
        color: #60A5FA;
    }

    /* Table styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #1F2937;
    }
    .styled-table thead tr {
        background: #0066FF;
        color: #fff;
        text-align: left;
        font-weight: 600;
    }
    .styled-table th, .styled-table td {
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #1F2937;
        background: #111827;
    }
    .styled-table tbody tr:hover {
        background: #1a2235;
    }
    .best-val {
        color: #34D399;
        font-weight: 700;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1117;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #111827;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        color: #9CA3AF;
        border: 1px solid #1F2937;
    }
    .stTabs [aria-selected="true"] {
        background: #0066FF !important;
        color: #fff !important;
        border-color: #0066FF !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/UniversalBank.csv")
    return df

df = load_data()

# ─────────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────────
@st.cache_data
def clean_data(df):
    df_clean = df.copy()
    df_clean['Experience'] = df_clean['Experience'].apply(lambda x: abs(x))
    df_clean.drop(columns=['ID', 'ZIP Code'], inplace=True)
    return df_clean

df_clean = clean_data(df)

# ─────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────
@st.cache_data
def train_models(df_clean):
    X = df_clean.drop('Personal Loan', axis=1)
    y = df_clean['Personal Loan']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    models = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1
        ),
        'Gradient Boosted Tree': GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        ),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_auc_val = auc(fpr, tpr)

        cm = confusion_matrix(y_test, y_test_pred)

        results[name] = {
            'train_acc': accuracy_score(y_train, y_train_pred),
            'test_acc': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'roc_auc': roc_auc_val,
            'fpr': fpr,
            'tpr': tpr,
            'cm': cm,
            'y_test_proba': y_test_proba,
            'feature_imp': dict(zip(X.columns, model.feature_importances_)),
        }

    return results, trained_models, scaler, X_train, X_test, y_train, y_test, X.columns.tolist()

results, trained_models, scaler, X_train, X_test, y_train, y_test, feature_cols = train_models(df_clean)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size: 2.5rem;'>🏦</div>
        <div style='font-size: 1.1rem; font-weight: 700; color: #F9FAFB; margin-top: 0.3rem;'>Universal Bank</div>
        <div style='font-size: 0.78rem; color: #6B7280;'>Personal Loan Campaign Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Executive Summary",
            "🔍 Customer Deep Dive",
            "🤖 Model Performance",
            "🎯 Campaign Strategy",
            "📤 Predict New Data"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.72rem; color: #4B5563; text-align: center; line-height: 1.6;'>
        Built for Universal Bank<br>
        Marketing Analytics Division<br>
        <span style='color: #374151;'>v2.0 — Classification Suite</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER — Plotly theme
# ─────────────────────────────────────────────
COLORS = {
    'primary': '#0066FF',
    'success': '#10B981',
    'danger': '#EF4444',
    'warning': '#F59E0B',
    'info': '#8B5CF6',
    'cyan': '#06B6D4',
    'pink': '#EC4899',
    'bg': '#0a0e1a',
    'card': '#111827',
    'border': '#1F2937',
    'text': '#E5E7EB',
    'muted': '#6B7280',
    'accent_blue': '#3B82F6',
    'accent_green': '#34D399',
    'accent_orange': '#FB923C',
}

MODEL_COLORS = {
    'Decision Tree': '#F59E0B',
    'Random Forest': '#10B981',
    'Gradient Boosted Tree': '#0066FF',
}

def style_plotly(fig, height=420):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', color='#E5E7EB', size=12),
        height=height,
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(
            bgcolor='rgba(17,24,39,0.8)',
            bordercolor='#1F2937',
            borderwidth=1,
            font=dict(size=11)
        ),
        xaxis=dict(gridcolor='#1F2937', zerolinecolor='#1F2937'),
        yaxis=dict(gridcolor='#1F2937', zerolinecolor='#1F2937'),
    )
    return fig


# ─────────────────────────────────────────────
# PAGE 1: EXECUTIVE SUMMARY
# ─────────────────────────────────────────────
if page == "📊 Executive Summary":
    st.markdown("""
    <div class="hero-header">
        <h1>Personal Loan Campaign Intelligence</h1>
        <p>Descriptive analytics on 5,000 customers from the last campaign — identifying who accepted personal loans and why.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    total = len(df)
    accepted = df['Personal Loan'].sum()
    rejected = total - accepted
    acc_rate = accepted / total * 100
    avg_income_acc = df[df['Personal Loan'] == 1]['Income'].mean()
    avg_income_rej = df[df['Personal Loan'] == 0]['Income'].mean()

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">{total:,}</div>
            <div class="metric-sub">Historical campaign data</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Accepted Loan</div>
            <div class="metric-value" style="color:#34D399">{accepted}</div>
            <div class="metric-sub">{acc_rate:.1f}% acceptance rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Rejected Loan</div>
            <div class="metric-value" style="color:#EF4444">{rejected:,}</div>
            <div class="metric-sub">{100 - acc_rate:.1f}% rejection rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Income (Accepted)</div>
            <div class="metric-value" style="color:#60A5FA">${avg_income_acc:,.0f}K</div>
            <div class="metric-sub">vs ${avg_income_rej:,.0f}K for rejected</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="insight-box"><strong>💡 Key Insight:</strong> Only 9.6% of customers accepted the personal loan in the last campaign. This means 90% of the marketing budget was spent on people who said no. With the budget cut to half, we need to be surgically precise about who we target.</div>', unsafe_allow_html=True)

    # Row 1 — Loan acceptance + Income distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Loan Acceptance Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">The dataset is heavily imbalanced — only 1 in 10 customers accepted. The model must handle this skew.</div>', unsafe_allow_html=True)

        labels = ['Rejected (0)', 'Accepted (1)']
        values = [rejected, accepted]
        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.55,
            marker=dict(colors=[COLORS['danger'], COLORS['success']], line=dict(color='#0a0e1a', width=3)),
            textinfo='label+percent+value',
            textfont=dict(size=13),
            hovertemplate='%{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>'
        ))
        fig.update_layout(
            annotations=[dict(text=f'{acc_rate:.1f}%', x=0.5, y=0.5, font_size=28, font_color='#34D399', showarrow=False, font_family='JetBrains Mono')],
        )
        style_plotly(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Income Distribution by Loan Status</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Loan acceptors clearly skew towards higher incomes. The $100K+ bracket is the sweet spot for targeting.</div>', unsafe_allow_html=True)

        fig = go.Figure()
        for val, name, color in [(0, 'Rejected', COLORS['danger']), (1, 'Accepted', COLORS['success'])]:
            fig.add_trace(go.Histogram(
                x=df[df['Personal Loan'] == val]['Income'],
                name=name, opacity=0.7,
                marker_color=color,
                nbinsx=40,
                hovertemplate='Income: $%{x}K<br>Count: %{y}<extra></extra>'
            ))
        fig.update_layout(barmode='overlay', xaxis_title='Annual Income ($000)', yaxis_title='Count')
        style_plotly(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 — Education + Family
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Acceptance Rate by Education Level</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Customers with Advanced/Professional degrees accept loans at 3x the rate of undergrads. Education is a strong targeting signal.</div>', unsafe_allow_html=True)

        edu_map = {1: 'Undergrad', 2: 'Graduate', 3: 'Advanced/\nProfessional'}
        edu_df = df.copy()
        edu_df['Education_Label'] = edu_df['Education'].map(edu_map)
        edu_stats = edu_df.groupby('Education_Label').agg(
            total=('Personal Loan', 'count'),
            accepted=('Personal Loan', 'sum')
        ).reset_index()
        edu_stats['rate'] = (edu_stats['accepted'] / edu_stats['total'] * 100).round(1)
        edu_stats['order'] = edu_stats['Education_Label'].map({'Undergrad': 1, 'Graduate': 2, 'Advanced/\nProfessional': 3})
        edu_stats = edu_stats.sort_values('order')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=edu_stats['Education_Label'], y=edu_stats['total'],
            name='Total Customers', marker_color=COLORS['border'],
            text=edu_stats['total'], textposition='outside',
            hovertemplate='%{x}<br>Total: %{y}<extra></extra>'
        ))
        fig.add_trace(go.Bar(
            x=edu_stats['Education_Label'], y=edu_stats['accepted'],
            name='Accepted Loan', marker_color=COLORS['primary'],
            text=edu_stats.apply(lambda r: f"{r['accepted']} ({r['rate']}%)", axis=1), textposition='outside',
            hovertemplate='%{x}<br>Accepted: %{y}<extra></extra>'
        ))
        fig.update_layout(barmode='group', yaxis_title='Count', xaxis_title='Education Level')
        style_plotly(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Acceptance Rate by Family Size</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Families of 3-4 members show higher loan acceptance. Larger families likely have more financial needs that personal loans can address.</div>', unsafe_allow_html=True)

        fam_stats = df.groupby('Family').agg(
            total=('Personal Loan', 'count'),
            accepted=('Personal Loan', 'sum')
        ).reset_index()
        fam_stats['rate'] = (fam_stats['accepted'] / fam_stats['total'] * 100).round(1)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fam_stats['Family'].astype(str), y=fam_stats['rate'],
            marker=dict(
                color=fam_stats['rate'],
                colorscale=[[0, '#1E3A5F'], [1, '#0066FF']],
                line=dict(color='#0066FF', width=1)
            ),
            text=fam_stats.apply(lambda r: f"{r['rate']}%<br>({int(r['accepted'])}/{int(r['total'])})", axis=1),
            textposition='outside',
            hovertemplate='Family Size: %{x}<br>Acceptance Rate: %{y:.1f}%<extra></extra>'
        ))
        fig.update_layout(yaxis_title='Acceptance Rate (%)', xaxis_title='Family Size')
        style_plotly(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3 — Credit Card Spend + CD Account
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Credit Card Spending vs Income</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Loan acceptors cluster in the high-income, high-spend quadrant. This combo is the strongest predictor of loan interest.</div>', unsafe_allow_html=True)

        fig = go.Figure()
        for val, name, color in [(0, 'Rejected', 'rgba(239,68,68,0.15)'), (1, 'Accepted', 'rgba(16,185,129,0.7)')]:
            subset = df[df['Personal Loan'] == val]
            fig.add_trace(go.Scatter(
                x=subset['Income'], y=subset['CCAvg'],
                mode='markers', name=name,
                marker=dict(color=color, size=5, line=dict(width=0)),
                hovertemplate='Income: $%{x}K<br>CC Spend: $%{y}K/mo<extra></extra>'
            ))
        fig.update_layout(xaxis_title='Annual Income ($000)', yaxis_title='Avg Monthly CC Spend ($000)')
        style_plotly(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">CD Account — The Hidden Gem</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Customers with a CD account have a massive 44% acceptance rate vs just 7% without. This is the single most powerful binary feature.</div>', unsafe_allow_html=True)

        cd_stats = df.groupby('CD Account').agg(
            total=('Personal Loan', 'count'),
            accepted=('Personal Loan', 'sum')
        ).reset_index()
        cd_stats['rate'] = (cd_stats['accepted'] / cd_stats['total'] * 100).round(1)
        cd_stats['label'] = cd_stats['CD Account'].map({0: 'No CD Account', 1: 'Has CD Account'})

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cd_stats['label'], y=cd_stats['rate'],
            marker_color=[COLORS['muted'], COLORS['success']],
            text=cd_stats.apply(lambda r: f"{r['rate']}%\n({int(r['accepted'])}/{int(r['total'])})", axis=1),
            textposition='outside', textfont=dict(size=14, color='#E5E7EB'),
            hovertemplate='%{x}<br>Acceptance Rate: %{y:.1f}%<extra></extra>'
        ))
        fig.update_layout(yaxis_title='Acceptance Rate (%)', xaxis_title='')
        style_plotly(fig, 400)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 2: CUSTOMER DEEP DIVE
# ─────────────────────────────────────────────
elif page == "🔍 Customer Deep Dive":
    st.markdown("""
    <div class="hero-header">
        <h1>Customer Deep Dive — Diagnostic Analytics</h1>
        <p>Understanding the characteristics that differentiate loan acceptors from non-acceptors to build customer personas.</p>
    </div>
    """, unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Income, CCAvg, and CD Account show the strongest positive correlation with Personal Loan. Age and Experience are almost perfectly correlated (multicollinear) — we can safely use just one of them.</div>', unsafe_allow_html=True)

    corr = df_clean.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[[0, '#0a0e1a'], [0.25, '#1a1f3a'], [0.5, '#1E3A5F'], [0.75, '#0055DD'], [1, '#3B82F6']],
        text=corr.values,
        texttemplate='%{text}',
        textfont=dict(size=10, color='#E5E7EB'),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>',
        colorbar=dict(title='Corr', tickfont=dict(color='#9CA3AF'))
    ))
    fig.update_layout(xaxis=dict(tickangle=45))
    style_plotly(fig, 520)
    st.plotly_chart(fig, use_container_width=True)

    # Comparative box plots
    st.markdown('<div class="section-header">Feature Distributions — Accepted vs Rejected</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Box plots reveal clear separation in Income and CCAvg between the two groups. Mortgage shows a slight edge for acceptors with existing mortgages.</div>', unsafe_allow_html=True)

    numeric_features = ['Income', 'CCAvg', 'Mortgage', 'Age', 'Experience']
    fig = make_subplots(rows=1, cols=5, subplot_titles=numeric_features, horizontal_spacing=0.05)
    for i, feat in enumerate(numeric_features, 1):
        for val, name, color in [(0, 'Rejected', COLORS['danger']), (1, 'Accepted', COLORS['success'])]:
            fig.add_trace(go.Box(
                y=df[df['Personal Loan'] == val][feat],
                name=name, marker_color=color, showlegend=(i == 1),
                boxmean=True,
                hovertemplate=f'{feat}<br>{name}<br>Value: %{{y}}<extra></extra>'
            ), row=1, col=i)
    style_plotly(fig, 400)
    fig.update_layout(margin=dict(t=60))
    st.plotly_chart(fig, use_container_width=True)

    # Binary features comparison
    st.markdown('<div class="section-header">Binary Features — Acceptance Rates</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Having a CD Account is by far the strongest binary indicator. Securities Account and Online banking show modest differences.</div>', unsafe_allow_html=True)

    binary_feats = ['Securities Account', 'CD Account', 'Online', 'CreditCard']
    fig = make_subplots(rows=1, cols=4, subplot_titles=binary_feats, horizontal_spacing=0.08)
    for i, feat in enumerate(binary_feats, 1):
        stats = df.groupby(feat)['Personal Loan'].mean().reset_index()
        stats['rate'] = (stats['Personal Loan'] * 100).round(1)
        stats['label'] = stats[feat].map({0: 'No', 1: 'Yes'})
        fig.add_trace(go.Bar(
            x=stats['label'], y=stats['rate'],
            marker_color=[COLORS['muted'], COLORS['primary']],
            text=stats['rate'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            showlegend=False,
            hovertemplate=f'{feat}: %{{x}}<br>Acceptance Rate: %{{y:.1f}}%<extra></extra>'
        ), row=1, col=i)
        fig.update_yaxes(title_text='Rate %' if i == 1 else '', row=1, col=i)
    style_plotly(fig, 380)
    fig.update_layout(margin=dict(t=60))
    st.plotly_chart(fig, use_container_width=True)

    # Income x Education x Loan heatmap
    st.markdown('<div class="section-header">Income Bracket × Education — Acceptance Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">The combination of high income ($100K+) and Graduate/Advanced education creates the hottest segments. This is your ideal customer persona for hyper-targeting.</div>', unsafe_allow_html=True)

    df_seg = df.copy()
    df_seg['Income_Bracket'] = pd.cut(df_seg['Income'], bins=[0, 50, 100, 150, 225], labels=['<$50K', '$50K-$100K', '$100K-$150K', '$150K+'])
    df_seg['Edu_Label'] = df_seg['Education'].map({1: 'Undergrad', 2: 'Graduate', 3: 'Advanced'})
    heatmap_data = df_seg.pivot_table(values='Personal Loan', index='Edu_Label', columns='Income_Bracket', aggfunc='mean')
    heatmap_data = (heatmap_data * 100).round(1)
    heatmap_data = heatmap_data.reindex(['Undergrad', 'Graduate', 'Advanced'])

    fig = go.Figure(go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns.tolist(),
        y=heatmap_data.index.tolist(),
        colorscale=[[0, '#111827'], [0.3, '#1E3A5F'], [0.6, '#0055DD'], [1, '#34D399']],
        text=heatmap_data.values,
        texttemplate='%{text:.1f}%',
        textfont=dict(size=14, color='#fff'),
        hovertemplate='Education: %{y}<br>Income: %{x}<br>Acceptance: %{z:.1f}%<extra></extra>',
        colorbar=dict(title='Rate %', tickfont=dict(color='#9CA3AF'))
    ))
    fig.update_layout(xaxis_title='Income Bracket', yaxis_title='Education Level')
    style_plotly(fig, 350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box"><strong>🎯 Persona Identified:</strong> The ideal target customer is someone with <strong>$100K+ income</strong>, <strong>Graduate or Advanced education</strong>, a <strong>CD Account</strong>, and <strong>family size 3+</strong>. This segment shows 30-50% acceptance rates — 3-5x the overall average.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 3: MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "🤖 Model Performance":
    st.markdown("""
    <div class="hero-header">
        <h1>Classification Model Performance</h1>
        <p>Comparing Decision Tree, Random Forest, and Gradient Boosted Tree on predicting personal loan acceptance.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Model Comparison Table</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">All three models perform well. Gradient Boosted Tree leads on AUC-ROC and balance between precision/recall. Random Forest is close behind. Accuracy alone is misleading here due to class imbalance — focus on F1-Score and Recall.</div>', unsafe_allow_html=True)

    # Build comparison table
    best = {
        'train_acc': max(r['train_acc'] for r in results.values()),
        'test_acc': max(r['test_acc'] for r in results.values()),
        'precision': max(r['precision'] for r in results.values()),
        'recall': max(r['recall'] for r in results.values()),
        'f1': max(r['f1'] for r in results.values()),
        'roc_auc': max(r['roc_auc'] for r in results.values()),
    }

    def fmt(val, key):
        pct = f"{val*100:.2f}%"
        if val == best[key]:
            return f'<span class="best-val">{pct} ★</span>'
        return pct

    table_html = """<table class="styled-table"><thead><tr>
        <th>Model</th><th>Train Accuracy</th><th>Test Accuracy</th>
        <th>Precision</th><th>Recall</th><th>F1-Score</th><th>AUC-ROC</th>
    </tr></thead><tbody>"""

    for name, r in results.items():
        color = MODEL_COLORS[name]
        table_html += f"""<tr>
            <td style="text-align:left; font-weight:600;"><span style="color:{color}">●</span> {name}</td>
            <td>{fmt(r['train_acc'], 'train_acc')}</td>
            <td>{fmt(r['test_acc'], 'test_acc')}</td>
            <td>{fmt(r['precision'], 'precision')}</td>
            <td>{fmt(r['recall'], 'recall')}</td>
            <td>{fmt(r['f1'], 'f1')}</td>
            <td>{fmt(r['roc_auc'], 'roc_auc')}</td>
        </tr>"""
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown('<div class="insight-box"><strong>📊 Reading the table:</strong> <strong>Precision</strong> = of those we predicted will accept, how many actually did? <strong>Recall</strong> = of all actual acceptors, how many did we catch? For a budget-constrained campaign, <strong>high Precision</strong> saves money (fewer wasted contacts), while <strong>high Recall</strong> ensures we don\'t miss potential customers.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ROC Curve — Single plot, all models
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">ROC Curve — All Models</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">The ROC curve plots True Positive Rate vs False Positive Rate. A curve closer to the top-left corner means better discrimination. All three models significantly outperform random chance (diagonal).</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            line=dict(dash='dash', color='#4B5563', width=1),
            name='Random Chance (AUC = 0.50)',
            hoverinfo='skip'
        ))
        for name, r in results.items():
            fig.add_trace(go.Scatter(
                x=r['fpr'], y=r['tpr'], mode='lines',
                line=dict(color=MODEL_COLORS[name], width=2.5),
                name=f"{name} (AUC = {r['roc_auc']:.4f})",
                fill='tonexty' if name == 'Decision Tree' else None,
                hovertemplate=f'{name}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>'
            ))
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.4, y=0.15, font=dict(size=11))
        )
        style_plotly(fig, 450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">AUC-ROC Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">AUC closer to 1.0 = better model. All models score above 0.95, indicating excellent predictive power.</div>', unsafe_allow_html=True)

        auc_df = pd.DataFrame({
            'Model': list(results.keys()),
            'AUC': [r['roc_auc'] for r in results.values()]
        }).sort_values('AUC', ascending=True)

        fig = go.Figure(go.Bar(
            x=auc_df['AUC'], y=auc_df['Model'],
            orientation='h',
            marker_color=[MODEL_COLORS[m] for m in auc_df['Model']],
            text=auc_df['AUC'].apply(lambda x: f'{x:.4f}'),
            textposition='inside', textfont=dict(size=14, color='#fff', family='JetBrains Mono'),
            hovertemplate='%{y}<br>AUC: %{x:.4f}<extra></extra>'
        ))
        fig.update_layout(xaxis=dict(range=[0.9, 1.0], title='AUC-ROC Score'))
        style_plotly(fig, 300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Confusion Matrices
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Each cell shows the count and percentage. Top-left (TN) and bottom-right (TP) are correct predictions. Top-right (FP) wastes budget on wrong targets. Bottom-left (FN) means missed potential customers.</div>', unsafe_allow_html=True)

    cm_cols = st.columns(3)
    for idx, (name, r) in enumerate(results.items()):
        with cm_cols[idx]:
            cm = r['cm']
            total = cm.sum()
            cm_pct = (cm / total * 100).round(1)

            labels = np.array([
                [f"TN\n{cm[0,0]}\n({cm_pct[0,0]}%)", f"FP\n{cm[0,1]}\n({cm_pct[0,1]}%)"],
                [f"FN\n{cm[1,0]}\n({cm_pct[1,0]}%)", f"TP\n{cm[1,1]}\n({cm_pct[1,1]}%)"]
            ])

            fig = go.Figure(go.Heatmap(
                z=cm, x=['Predicted: No', 'Predicted: Yes'], y=['Actual: No', 'Actual: Yes'],
                colorscale=[[0, '#111827'], [0.5, '#1E3A5F'], [1, MODEL_COLORS[name]]],
                text=labels, texttemplate='%{text}',
                textfont=dict(size=12, color='#fff'),
                hovertemplate='%{y} / %{x}<br>Count: %{z}<extra></extra>',
                showscale=False
            ))
            fig.update_layout(
                title=dict(text=f'<b>{name}</b>', font=dict(size=13, color=MODEL_COLORS[name])),
                xaxis=dict(title='Predicted'),
                yaxis=dict(title='Actual', autorange='reversed'),
            )
            style_plotly(fig, 340)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature Importance
    st.markdown('<div class="section-header">Feature Importance — What Drives Predictions?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Income and CCAvg dominate across all models. CD Account is the most powerful binary feature. This tells us exactly which data points matter most for targeting decisions.</div>', unsafe_allow_html=True)

    fi_cols = st.columns(3)
    for idx, (name, r) in enumerate(results.items()):
        with fi_cols[idx]:
            fi = pd.DataFrame({
                'Feature': list(r['feature_imp'].keys()),
                'Importance': list(r['feature_imp'].values())
            }).sort_values('Importance', ascending=True).tail(8)

            fig = go.Figure(go.Bar(
                x=fi['Importance'], y=fi['Feature'],
                orientation='h',
                marker_color=MODEL_COLORS[name],
                text=fi['Importance'].apply(lambda x: f'{x:.3f}'),
                textposition='inside', textfont=dict(size=11, color='#fff'),
                hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'
            ))
            fig.update_layout(
                title=dict(text=f'<b>{name}</b>', font=dict(size=13, color=MODEL_COLORS[name])),
                xaxis_title='Importance'
            )
            style_plotly(fig, 380)
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 4: CAMPAIGN STRATEGY (PRESCRIPTIVE)
# ─────────────────────────────────────────────
elif page == "🎯 Campaign Strategy":
    st.markdown("""
    <div class="hero-header">
        <h1>Prescriptive Analytics — Campaign Strategy</h1>
        <p>Actionable recommendations to maximize loan acceptance with 50% reduced budget using model predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Use best model to score everyone
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    best_model = trained_models[best_model_name]

    X_all = df_clean.drop('Personal Loan', axis=1)
    X_all_scaled = pd.DataFrame(scaler.transform(X_all), columns=X_all.columns)
    df_scored = df.copy()
    df_scored['Loan_Probability'] = best_model.predict_proba(X_all_scaled)[:, 1]
    df_scored['Risk_Tier'] = pd.cut(
        df_scored['Loan_Probability'],
        bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=['Low (0-20%)', 'Medium (20-50%)', 'High (50-80%)', 'Very High (80-100%)']
    )

    # Tier breakdown
    tier_stats = df_scored.groupby('Risk_Tier', observed=True).agg(
        count=('ID', 'count'),
        actual_accepted=('Personal Loan', 'sum'),
        avg_income=('Income', 'mean'),
        avg_ccavg=('CCAvg', 'mean')
    ).reset_index()
    tier_stats['actual_rate'] = (tier_stats['actual_accepted'] / tier_stats['count'] * 100).round(1)

    st.markdown('<div class="section-header">Customer Segmentation by Predicted Loan Probability</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-desc">Using the <strong>{best_model_name}</strong> model (best F1-Score), every customer is scored with a loan acceptance probability. This allows precision targeting — contact only the high-probability segments.</div>', unsafe_allow_html=True)

    # Tier KPIs
    tier_colors = {'Low (0-20%)': COLORS['muted'], 'Medium (20-50%)': COLORS['warning'], 'High (50-80%)': COLORS['info'], 'Very High (80-100%)': COLORS['success']}

    tier_html = '<div class="metric-row">'
    for _, row in tier_stats.iterrows():
        color = tier_colors.get(row['Risk_Tier'], COLORS['muted'])
        tier_html += f"""
        <div class="metric-card" style="border-left: 3px solid {color};">
            <div class="metric-label">{row['Risk_Tier']}</div>
            <div class="metric-value" style="font-size:1.5rem; color:{color}">{int(row['count']):,}</div>
            <div class="metric-sub">Actual acceptance: {row['actual_rate']}% · Avg Income: ${row['avg_income']:.0f}K</div>
        </div>
        """
    tier_html += '</div>'
    st.markdown(tier_html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Probability Distribution of All Customers</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Most customers cluster at very low probability (<10%). The right tail (>50%) is your gold mine — small group but extremely high conversion potential.</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df_scored[df_scored['Personal Loan'] == 0]['Loan_Probability'],
            name='Actually Rejected', marker_color='rgba(239,68,68,0.4)',
            nbinsx=50, hovertemplate='Probability: %{x:.2f}<br>Count: %{y}<extra></extra>'
        ))
        fig.add_trace(go.Histogram(
            x=df_scored[df_scored['Personal Loan'] == 1]['Loan_Probability'],
            name='Actually Accepted', marker_color='rgba(16,185,129,0.8)',
            nbinsx=50, hovertemplate='Probability: %{x:.2f}<br>Count: %{y}<extra></extra>'
        ))
        fig.update_layout(barmode='overlay', xaxis_title='Predicted Probability', yaxis_title='Count')
        style_plotly(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Budget Optimisation — Cumulative Gain</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">By targeting only the top 20% of customers (ranked by model probability), you can capture ~80% of all potential acceptors. This is how you make half the budget work harder.</div>', unsafe_allow_html=True)

        df_gain = df_scored.sort_values('Loan_Probability', ascending=False).reset_index(drop=True)
        df_gain['cum_accepted'] = df_gain['Personal Loan'].cumsum()
        total_accepted = df_gain['Personal Loan'].sum()
        df_gain['pct_contacted'] = (np.arange(1, len(df_gain) + 1) / len(df_gain) * 100)
        df_gain['pct_captured'] = (df_gain['cum_accepted'] / total_accepted * 100)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_gain['pct_contacted'], y=df_gain['pct_captured'],
            mode='lines', name='Model-Guided',
            line=dict(color=COLORS['primary'], width=3),
            fill='tozeroy', fillcolor='rgba(0,102,255,0.1)',
            hovertemplate='Contact Top %{x:.0f}% → Capture %{y:.0f}% of acceptors<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100], mode='lines', name='Random (No Model)',
            line=dict(dash='dash', color='#4B5563', width=1.5), hoverinfo='skip'
        ))
        # 50% budget line
        fig.add_vline(x=50, line=dict(color=COLORS['warning'], dash='dash', width=1.5))
        fig.add_annotation(x=50, y=95, text="50% Budget Line", showarrow=False, font=dict(color=COLORS['warning'], size=11))

        # Find capture at 20%
        capture_at_20 = df_gain[df_gain['pct_contacted'] <= 20]['pct_captured'].iloc[-1] if len(df_gain[df_gain['pct_contacted'] <= 20]) > 0 else 0
        fig.add_annotation(x=20, y=capture_at_20, text=f"Top 20% → {capture_at_20:.0f}% captured",
                          showarrow=True, arrowhead=2, arrowcolor=COLORS['success'],
                          font=dict(color=COLORS['success'], size=12))

        fig.update_layout(xaxis_title='% Customers Contacted', yaxis_title='% Loan Acceptors Captured')
        style_plotly(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Strategy recommendations
    st.markdown('<div class="section-header">Campaign Playbook — Actionable Recommendations</div>', unsafe_allow_html=True)

    high_value = df_scored[df_scored['Loan_Probability'] >= 0.5]
    high_count = len(high_value)
    high_actual = high_value['Personal Loan'].sum()
    high_rate = (high_actual / high_count * 100) if high_count > 0 else 0

    medium_value = df_scored[(df_scored['Loan_Probability'] >= 0.2) & (df_scored['Loan_Probability'] < 0.5)]
    med_count = len(medium_value)

    st.markdown(f"""
    <div class="insight-box">
        <strong>🎯 TIER 1 — Direct Outreach (High + Very High Probability)</strong><br>
        <strong>{high_count} customers</strong> ({high_count/len(df_scored)*100:.1f}% of base) with predicted acceptance >50%.<br>
        Historical actual acceptance in this group: <strong>{high_rate:.1f}%</strong>.<br>
        <strong>Action:</strong> Personal calls, dedicated relationship manager, pre-approved loan offers with competitive rates.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
        <strong>📧 TIER 2 — Targeted Digital Campaign (Medium Probability)</strong><br>
        <strong>{med_count} customers</strong> ({med_count/len(df_scored)*100:.1f}% of base) with 20-50% predicted acceptance.<br>
        <strong>Action:</strong> Email campaigns, targeted ads, in-app notifications. Personalise messaging based on their profile
        (education level, income bracket, family size). A/B test offers.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
        <strong>🚫 TIER 3 — Do Not Contact (Low Probability)</strong><br>
        <strong>{len(df_scored) - high_count - med_count} customers</strong> with <20% predicted acceptance.<br>
        <strong>Action:</strong> Skip entirely. This is where your budget was being wasted. Redirect savings to Tier 1 and 2.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Profile comparison
    st.markdown('<div class="section-header">Ideal vs Average Customer Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Side-by-side comparison of the ideal target (predicted probability >50%) vs the average customer. Use this to craft messaging that resonates.</div>', unsafe_allow_html=True)

    ideal = high_value[['Income', 'CCAvg', 'Age', 'Family', 'Education', 'Mortgage']].mean()
    average = df_scored[['Income', 'CCAvg', 'Age', 'Family', 'Education', 'Mortgage']].mean()

    profile_data = pd.DataFrame({
        'Feature': ['Income ($K)', 'CC Spend ($K/mo)', 'Age', 'Family Size', 'Education Level', 'Mortgage ($K)'],
        'Ideal Target': [f'${ideal["Income"]:.0f}K', f'${ideal["CCAvg"]:.1f}K', f'{ideal["Age"]:.0f} yrs', f'{ideal["Family"]:.1f}', f'{ideal["Education"]:.1f}', f'${ideal["Mortgage"]:.0f}K'],
        'Average Customer': [f'${average["Income"]:.0f}K', f'${average["CCAvg"]:.1f}K', f'{average["Age"]:.0f} yrs', f'{average["Family"]:.1f}', f'{average["Education"]:.1f}', f'${average["Mortgage"]:.0f}K'],
    })

    categories = ['Income', 'CCAvg', 'Age', 'Family', 'Education', 'Mortgage']
    ideal_vals = [ideal[c] for c in categories]
    avg_vals = [average[c] for c in categories]

    # Normalize for radar
    max_vals = [max(ideal_vals[i], avg_vals[i]) * 1.2 for i in range(len(categories))]
    ideal_norm = [ideal_vals[i] / max_vals[i] * 100 for i in range(len(categories))]
    avg_norm = [avg_vals[i] / max_vals[i] * 100 for i in range(len(categories))]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=ideal_norm + [ideal_norm[0]],
        theta=categories + [categories[0]],
        fill='toself', name='Ideal Target',
        fillcolor='rgba(0,102,255,0.15)',
        line=dict(color=COLORS['primary'], width=2),
    ))
    fig.add_trace(go.Scatterpolar(
        r=avg_norm + [avg_norm[0]],
        theta=categories + [categories[0]],
        fill='toself', name='Average Customer',
        fillcolor='rgba(107,114,128,0.1)',
        line=dict(color=COLORS['muted'], width=2, dash='dash'),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=False),
            angularaxis=dict(gridcolor='#1F2937', linecolor='#1F2937', tickfont=dict(size=12, color='#E5E7EB'))
        ),
    )
    style_plotly(fig, 420)

    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(profile_data, hide_index=True, use_container_width=True)
    with col2:
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 5: PREDICT NEW DATA
# ─────────────────────────────────────────────
elif page == "📤 Predict New Data":
    st.markdown("""
    <div class="hero-header">
        <h1>Predict on New Customer Data</h1>
        <p>Upload a CSV with the same columns (excluding 'Personal Loan') and download predictions with loan probability scores.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <strong>📋 Required Columns:</strong> ID, Age, Experience, Income, ZIP Code, Family, CCAvg, Education, Mortgage, Securities Account, CD Account, Online, CreditCard<br>
        <strong>Note:</strong> 'Personal Loan' column should NOT be present — the model will predict it. A test file <code>Test_UniversalBank.csv</code> is included in the <code>data/</code> folder.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            st.markdown('<div class="section-header">Uploaded Data Preview</div>', unsafe_allow_html=True)
            st.dataframe(new_df.head(10), use_container_width=True)

            # Validate columns
            required = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage',
                       'Securities Account', 'CD Account', 'Online', 'CreditCard']

            missing_cols = [c for c in required if c not in new_df.columns]
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
            else:
                # Prepare features
                new_features = new_df.copy()

                # Drop non-feature columns if present
                drop_cols = ['ID', 'ZIP Code', 'Personal Loan']
                for col in drop_cols:
                    if col in new_features.columns:
                        new_features = new_features.drop(columns=[col])

                # Clean experience
                new_features['Experience'] = new_features['Experience'].apply(lambda x: abs(x))

                # Ensure column order matches training
                new_features = new_features[feature_cols]

                # Scale
                new_features_scaled = pd.DataFrame(
                    scaler.transform(new_features),
                    columns=new_features.columns
                )

                # Predict with best model
                best_name = max(results, key=lambda k: results[k]['f1'])
                best_model = trained_models[best_name]

                proba = best_model.predict_proba(new_features_scaled)[:, 1]
                preds = best_model.predict(new_features_scaled)

                # Build output
                output_df = new_df.copy()
                output_df['Predicted_Personal_Loan'] = preds
                output_df['Loan_Probability'] = (proba * 100).round(2)
                output_df['Risk_Tier'] = pd.cut(
                    proba,
                    bins=[0, 0.2, 0.5, 0.8, 1.0],
                    labels=['Low', 'Medium', 'High', 'Very High'],
                    include_lowest=True
                )

                st.markdown("---")
                st.markdown(f'<div class="section-header">Prediction Results (Model: {best_name})</div>', unsafe_allow_html=True)

                # Summary KPIs
                n_total = len(output_df)
                n_yes = (preds == 1).sum()
                n_no = (preds == 0).sum()
                avg_prob = proba.mean() * 100

                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-label">Total Records</div>
                        <div class="metric-value">{n_total}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Predicted YES</div>
                        <div class="metric-value" style="color:#34D399">{n_yes}</div>
                        <div class="metric-sub">{n_yes/n_total*100:.1f}% of total</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Predicted NO</div>
                        <div class="metric-value" style="color:#EF4444">{n_no}</div>
                        <div class="metric-sub">{n_no/n_total*100:.1f}% of total</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Avg Probability</div>
                        <div class="metric-value" style="color:#60A5FA">{avg_prob:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.dataframe(output_df, use_container_width=True, height=400)

                # Probability distribution
                st.markdown('<div class="section-header">Prediction Probability Distribution</div>', unsafe_allow_html=True)

                fig = go.Figure(go.Histogram(
                    x=proba * 100, nbinsx=20,
                    marker_color=COLORS['primary'],
                    hovertemplate='Probability: %{x:.0f}%<br>Count: %{y}<extra></extra>'
                ))
                fig.update_layout(xaxis_title='Loan Acceptance Probability (%)', yaxis_title='Count')
                style_plotly(fig, 350)
                st.plotly_chart(fig, use_container_width=True)

                # Download
                csv_buffer = io.StringIO()
                output_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="⬇️ Download Predictions as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="UniversalBank_Predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV has the required columns listed above.")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; background: #111827; border-radius: 16px; border: 2px dashed #1F2937; margin-top: 1rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
            <div style="font-size: 1.1rem; color: #9CA3AF; margin-bottom: 0.5rem;">Drag and drop a CSV file here</div>
            <div style="font-size: 0.85rem; color: #4B5563;">or use the upload button above</div>
        </div>
        """, unsafe_allow_html=True)
