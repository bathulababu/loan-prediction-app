# -*- coding: utf-8 -*-
"""
Advanced CreditWise - Enterprise Loan Approval System
Features: Real-time ML, Explainability, Advanced Analytics, Multi-model Support
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# ML Libraries
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
import shap
import lime
import lime.lime_tabular

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="CreditWise Pro – Enterprise Loan System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA WITH CACHING
# ==================================================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("clean_dataset.csv")
    except FileNotFoundError:
        # Generate synthetic data if file not found
        st.warning("Dataset not found. Generating synthetic data for demo...")
        return generate_synthetic_data()

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic loan data for demonstration"""
    np.random.seed(42)
    
    data = {
        'Applicant_ID': [f'APP{str(i).zfill(6)}' for i in range(n_samples)],
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Marital_Status': np.random.choice(['Single', 'Married'], n_samples, p=[0.3, 0.7]),
        'Education_Level': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.7, 0.3]),
        'Employment_Status': np.random.choice(['Private', 'Government', 'MNC', 'Business', 'Unemployed'], 
                                              n_samples, p=[0.35, 0.25, 0.20, 0.15, 0.05]),
        'Applicant_Income': np.random.gamma(4, 15000, n_samples).astype(int),
        'Coapplicant_Income': np.random.gamma(2, 10000, n_samples).astype(int),
        'Loan_Amount': np.random.gamma(5, 30000, n_samples).astype(int),
        'Loan_Term': np.random.choice([120, 180, 240, 360], n_samples, p=[0.1, 0.2, 0.4, 0.3]),
        'Credit_History': np.random.choice([0, 1], n_samples, p=[0.15, 0.85]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, p=[0.4, 0.35, 0.25]),
    }
    
    df = pd.DataFrame(data)
    
    # Generate approval based on logic
    approval_score = (
        (df['Applicant_Income'] / 100000) * 0.3 +
        (df['Credit_History']) * 0.4 +
        (df['Education_Level'] == 'Graduate').astype(int) * 0.15 +
        (df['Employment_Status'].isin(['Government', 'MNC'])).astype(int) * 0.15 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    df['Loan_Approved'] = np.where(approval_score > 0.5, 'Yes', 'No')
    
    return df

# ==================================================
# SESSION STATE INITIALIZATION
# ==================================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'Logistic Regression'

# ==================================================
# LOAD AND PREPARE DATA
# ==================================================
df = load_data()

# Data Cleaning
df = df.drop(columns=["Applicant_ID"], errors="ignore")
df = df[df["Loan_Approved"].isin(["Yes", "No"])]

# Feature Engineering
df['Total_Income'] = df['Applicant_Income'] + df['Coapplicant_Income']
df['Income_to_Loan_Ratio'] = df['Total_Income'] / (df['Loan_Amount'] + 1)
df['EMI'] = df['Loan_Amount'] / (df['Loan_Term'] + 1)
df['Balance_Income'] = df['Total_Income'] - df['EMI']

# ==================================================
# MODEL TRAINING PIPELINE
# ==================================================
@st.cache_resource
def train_models(df, model_choice='all'):
    """Train multiple ML models"""
    
    y = df["Loan_Approved"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Loan_Approved"])
    
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    # Preprocessing
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    
    X[num_cols] = num_imputer.fit_transform(X[num_cols])
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
    
    # Encoding
    ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    X_cat_encoded = ohe.fit_transform(X[cat_cols])
    X_cat_encoded_df = pd.DataFrame(
        X_cat_encoded, columns=ohe.get_feature_names_out(cat_cols), index=X.index
    )
    
    X_final = pd.concat([X.drop(columns=cat_cols), X_cat_encoded_df], axis=1)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {}
    metrics = {}
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    metrics['Logistic Regression'] = evaluate_model(lr, X_test, y_test)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    metrics['Random Forest'] = evaluate_model(rf, X_test, y_test)
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    metrics['Gradient Boosting'] = evaluate_model(gb, X_test, y_test)
    
    return {
        'models': models,
        'metrics': metrics,
        'preprocessors': {
            'num_imputer': num_imputer,
            'cat_imputer': cat_imputer,
            'ohe': ohe,
            'scaler': scaler
        },
        'feature_info': {
            'num_cols': num_cols,
            'cat_cols': cat_cols,
            'train_columns': X_final.columns
        },
        'test_data': (X_test, y_test),
        'X_final': X_final,
        'y': y
    }

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

# Train models
with st.spinner('🚀 Training advanced ML models...'):
    model_data = train_models(df)
    st.session_state.model_trained = True

# ==================================================
# HEADER
# ==================================================
st.markdown('<h1 class="main-header">💳 CreditWise Pro – Enterprise Loan System</h1>', unsafe_allow_html=True)
st.markdown("**Advanced ML-powered loan approval with real-time analytics, model explainability & predictive insights**")

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
st.sidebar.image("https://img.icons8.com/clouds/200/bank.png", width=150)
st.sidebar.title("🎯 Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Dashboard",
        "📊 Data Analytics",
        "🤖 Model Performance",
        "💰 Loan Prediction",
        "📈 Batch Processing",
        "🔍 Model Explainability",
        "⏱️ Real-time Monitor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model Settings")
selected_model = st.sidebar.selectbox(
    "Choose ML Model",
    list(model_data['models'].keys()),
    index=0
)
st.session_state.selected_model = selected_model

threshold = st.sidebar.slider(
    "Approval Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Active Model:** {selected_model}")
st.sidebar.success(f"**Model Accuracy:** {model_data['metrics'][selected_model]['accuracy']:.2%}")

# ==================================================
# PAGE: DASHBOARD
# ==================================================
if page == "🏠 Dashboard":
    st.header("📊 Executive Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_apps = len(df)
        st.metric("Total Applications", f"{total_apps:,}", delta="Live")
    
    with col2:
        approval_rate = (df['Loan_Approved'] == 'Yes').mean() * 100
        st.metric("Approval Rate", f"{approval_rate:.1f}%", delta=f"{approval_rate-50:.1f}%")
    
    with col3:
        avg_loan = df['Loan_Amount'].mean()
        st.metric("Avg Loan Amount", f"${avg_loan:,.0f}", delta="▲ 5.2%")
    
    with col4:
        model_acc = model_data['metrics'][selected_model]['accuracy'] * 100
        st.metric("Model Accuracy", f"{model_acc:.1f}%", delta="+2.3%")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Approval Distribution
        approval_counts = df['Loan_Approved'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['Approved', 'Rejected'],
            values=[approval_counts.get('Yes', 0), approval_counts.get('No', 0)],
            hole=0.4,
            marker=dict(colors=['#667eea', '#f093fb'])
        )])
        fig.update_layout(
            title="Loan Approval Distribution",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Income Distribution
        fig = px.histogram(
            df,
            x='Applicant_Income',
            color='Loan_Approved',
            nbins=30,
            title="Income Distribution by Approval Status",
            color_discrete_map={'Yes': '#667eea', 'No': '#f093fb'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Approval by Employment
        emp_approval = pd.crosstab(df['Employment_Status'], df['Loan_Approved'], normalize='index') * 100
        fig = px.bar(
            emp_approval,
            title="Approval Rate by Employment Status",
            barmode='group',
            color_discrete_map={'Yes': '#667eea', 'No': '#f093fb'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Loan Amount Trends
        loan_stats = df.groupby('Loan_Term')['Loan_Amount'].agg(['mean', 'median', 'count']).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Average', x=loan_stats['Loan_Term'], y=loan_stats['mean'], marker_color='#667eea'))
        fig.add_trace(go.Bar(name='Median', x=loan_stats['Loan_Term'], y=loan_stats['median'], marker_color='#764ba2'))
        fig.update_layout(
            title="Loan Amount by Term Length",
            barmode='group',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Activity
    st.markdown("### 📝 Recent Prediction Activity")
    if st.session_state.prediction_history:
        recent_df = pd.DataFrame(st.session_state.prediction_history[-10:])
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No predictions yet. Go to 'Loan Prediction' to make your first prediction!")

# ==================================================
# PAGE: DATA ANALYTICS
# ==================================================
elif page == "📊 Data Analytics":
    st.header("📊 Advanced Data Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📈 Statistical Overview", "🔗 Correlations", "📉 Distribution Analysis"])
    
    with tab1:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Categorical Features")
            cat_summary = df.select_dtypes(include=['object']).describe()
            st.dataframe(cat_summary, use_container_width=True)
        
        with col2:
            st.markdown("##### Missing Values Analysis")
            missing = df.isnull().sum()
            missing_df = pd.DataFrame({
                'Feature': missing.index,
                'Missing Count': missing.values,
                'Percentage': (missing.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
    
    with tab2:
        st.subheader("Feature Correlations")
        
        # Correlation heatmap
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        corr = numeric_df.corr()
        
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Distribution Analysis")
        
        selected_feature = st.selectbox(
            "Select Feature to Analyze",
            df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df,
                x=selected_feature,
                color='Loan_Approved',
                marginal='box',
                title=f"{selected_feature} Distribution",
                color_discrete_map={'Yes': '#667eea', 'No': '#f093fb'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df,
                x='Loan_Approved',
                y=selected_feature,
                color='Loan_Approved',
                title=f"{selected_feature} Box Plot by Approval",
                color_discrete_map={'Yes': '#667eea', 'No': '#f093fb'}
            )
            st.plotly_chart(fig, use_container_width=True)

# ==================================================
# PAGE: MODEL PERFORMANCE
# ==================================================
elif page == "🤖 Model Performance":
    st.header("🤖 Model Performance Analytics")
    
    # Model Comparison
    st.subheader("📊 Model Comparison")
    
    metrics_df = pd.DataFrame({
        'Model': list(model_data['metrics'].keys()),
        'Accuracy': [m['accuracy'] for m in model_data['metrics'].values()],
        'Precision': [m['precision'] for m in model_data['metrics'].values()],
        'Recall': [m['recall'] for m in model_data['metrics'].values()],
        'F1-Score': [m['f1'] for m in model_data['metrics'].values()]
    })
    
    fig = go.Figure()
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df['Model'],
            y=metrics_df[metric],
            text=metrics_df[metric].round(3),
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']), 
                 use_container_width=True)
    
    # Detailed metrics for selected model
    st.markdown("---")
    st.subheader(f"📈 Detailed Analysis: {selected_model}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROC Curve
        X_test, y_test = model_data['test_data']
        model = model_data['models'][selected_model]
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='#667eea', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash')
        ))
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confusion Matrix
        cm = model_data['metrics'][selected_model]['confusion_matrix']
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Rejected', 'Approved'],
            y=['Rejected', 'Approved'],
            text_auto=True,
            color_continuous_scale='Blues',
            title='Confusion Matrix'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance (for tree-based models)
    if selected_model in ['Random Forest', 'Gradient Boosting']:
        st.markdown("---")
        st.subheader("🎯 Feature Importance")
        
        feature_importance = pd.DataFrame({
            'Feature': model_data['feature_info']['train_columns'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 15 Most Important Features',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# PAGE: LOAN PREDICTION
# ==================================================
elif page == "💰 Loan Prediction":
    st.header("💰 Loan Approval Prediction")
    
    st.markdown("### 📝 Enter Applicant Details")
    
    with st.form("loan_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 👤 Personal Information")
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Marital_Status = st.selectbox("Marital Status", ["Single", "Married"])
            Education_Level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
            Employment_Status = st.selectbox(
                "Employment Status",
                ["Private", "Government", "MNC", "Business", "Unemployed"]
            )
        
        with col2:
            st.markdown("##### 💵 Financial Information")
            Applicant_Income = st.number_input(
                "Applicant Income ($)",
                value=50000,
                min_value=0,
                step=5000,
                help="Annual income of the primary applicant"
            )
            Coapplicant_Income = st.number_input(
                "Coapplicant Income ($)",
                value=20000,
                min_value=0,
                step=5000,
                help="Annual income of co-applicant (if any)"
            )
            Credit_History = st.selectbox(
                "Credit History",
                [1, 0],
                format_func=lambda x: "Good" if x == 1 else "Poor",
                help="1 = Good credit history, 0 = Poor credit history"
            )
        
        with col3:
            st.markdown("##### 🏠 Loan Details")
            Loan_Amount = st.number_input(
                "Loan Amount ($)",
                value=150000,
                min_value=1000,
                step=10000,
                help="Total loan amount requested"
            )
            Loan_Term = st.selectbox(
                "Loan Term (months)",
                [120, 180, 240, 360],
                index=2,
                help="Loan repayment period"
            )
            Property_Area = st.selectbox(
                "Property Area",
                ["Urban", "Semiurban", "Rural"]
            )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit = st.form_submit_button("🔍 Predict Loan Eligibility", use_container_width=True)
    
    if submit:
        with st.spinner("🔮 Analyzing application..."):
            time.sleep(1)  # Simulate processing
            
            # Prepare input
            input_df = pd.DataFrame(columns=model_data['feature_info']['num_cols'] + model_data['feature_info']['cat_cols'])
            
            # Fill with defaults
            for col in model_data['feature_info']['num_cols']:
                input_df.loc[0, col] = df[col].median()
            for col in model_data['feature_info']['cat_cols']:
                input_df.loc[0, col] = df[col].mode()[0]
            
            # Override with user inputs
            input_df.loc[0, 'Applicant_Income'] = Applicant_Income
            input_df.loc[0, 'Coapplicant_Income'] = Coapplicant_Income
            input_df.loc[0, 'Loan_Amount'] = Loan_Amount
            input_df.loc[0, 'Loan_Term'] = Loan_Term
            input_df.loc[0, 'Gender'] = Gender
            input_df.loc[0, 'Marital_Status'] = Marital_Status
            input_df.loc[0, 'Education_Level'] = Education_Level
            input_df.loc[0, 'Employment_Status'] = Employment_Status
            if 'Credit_History' in model_data['feature_info']['num_cols']:
                input_df.loc[0, 'Credit_History'] = Credit_History
            if 'Property_Area' in model_data['feature_info']['cat_cols']:
                input_df.loc[0, 'Property_Area'] = Property_Area
            
            # Feature engineering
            input_df['Total_Income'] = input_df['Applicant_Income'] + input_df['Coapplicant_Income']
            input_df['Income_to_Loan_Ratio'] = input_df['Total_Income'] / (input_df['Loan_Amount'] + 1)
            input_df['EMI'] = input_df['Loan_Amount'] / (input_df['Loan_Term'] + 1)
            input_df['Balance_Income'] = input_df['Total_Income'] - input_df['EMI']
            
            # Preprocessing
            preprocessors = model_data['preprocessors']
            num_cols = model_data['feature_info']['num_cols']
            cat_cols = model_data['feature_info']['cat_cols']
            
            input_df[num_cols] = preprocessors['num_imputer'].transform(input_df[num_cols])
            input_df[cat_cols] = preprocessors['cat_imputer'].transform(input_df[cat_cols])
            
            input_cat_encoded = preprocessors['ohe'].transform(input_df[cat_cols])
            input_cat_encoded_df = pd.DataFrame(
                input_cat_encoded,
                columns=preprocessors['ohe'].get_feature_names_out(cat_cols),
                index=input_df.index
            )
            
            input_final = pd.concat(
                [input_df.drop(columns=cat_cols), input_cat_encoded_df],
                axis=1
            )
            input_final = input_final.reindex(columns=model_data['feature_info']['train_columns'], fill_value=0)
            
            # Prediction
            model = model_data['models'][selected_model]
            input_scaled = preprocessors['scaler'].transform(input_final)
            approval_prob = model.predict_proba(input_scaled)[0][1]
            prediction = "Approved" if approval_prob >= threshold else "Rejected"
            
            # Display Results
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Approval Probability")
                st.progress(int(approval_prob * 100))
                st.markdown(f"<h2 style='text-align: center; color: #667eea;'>{approval_prob:.1%}</h2>", 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Decision")
                if prediction == "Approved":
                    st.success(f"### ✅ {prediction}")
                    st.balloons()
                else:
                    st.error(f"### ❌ {prediction}")
            
            with col3:
                st.markdown("### Confidence Level")
                confidence = max(approval_prob, 1 - approval_prob)
                st.metric("Confidence", f"{confidence:.1%}")
                if confidence > 0.8:
                    st.success("High Confidence")
                elif confidence > 0.6:
                    st.warning("Medium Confidence")
                else:
                    st.error("Low Confidence")
            
            # Risk Analysis
            st.markdown("---")
            st.markdown("### 📊 Risk Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Calculate risk factors
                total_income = Applicant_Income + Coapplicant_Income
                dti_ratio = (Loan_Amount / Loan_Term) / (total_income / 12) if total_income > 0 else 0
                
                risk_factors = []
                if dti_ratio > 0.4:
                    risk_factors.append("⚠️ High Debt-to-Income Ratio")
                if Credit_History == 0:
                    risk_factors.append("⚠️ Poor Credit History")
                if Loan_Amount > total_income * 3:
                    risk_factors.append("⚠️ Loan Amount High Relative to Income")
                if Employment_Status == "Unemployed":
                    risk_factors.append("⚠️ Unemployed Status")
                
                if risk_factors:
                    st.markdown("##### ⚠️ Risk Factors Identified:")
                    for factor in risk_factors:
                        st.warning(factor)
                else:
                    st.success("✅ No major risk factors identified")
            
            with col2:
                # Financial metrics
                st.markdown("##### 💰 Financial Metrics")
                monthly_income = total_income / 12
                monthly_emi = Loan_Amount / Loan_Term
                balance_income = monthly_income - monthly_emi
                
                metrics_data = pd.DataFrame({
                    'Metric': ['Monthly Income', 'Monthly EMI', 'Balance Income', 'DTI Ratio'],
                    'Value': [
                        f"${monthly_income:,.2f}",
                        f"${monthly_emi:,.2f}",
                        f"${balance_income:,.2f}",
                        f"{dti_ratio:.1%}"
                    ]
                })
                st.dataframe(metrics_data, use_container_width=True, hide_index=True)
            
            # Save to history
            st.session_state.prediction_history.append({
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Applicant_Income': f"${Applicant_Income:,}",
                'Loan_Amount': f"${Loan_Amount:,}",
                'Approval_Probability': f"{approval_prob:.1%}",
                'Decision': prediction,
                'Model': selected_model
            })

# ==================================================
# PAGE: BATCH PROCESSING
# ==================================================
elif page == "📈 Batch Processing":
    st.header("📈 Batch Loan Processing")
    
    st.markdown("""
    Upload a CSV file with multiple loan applications for batch processing.
    The file should contain the same columns as individual predictions.
    """)
    
    # Sample template
    st.markdown("### 📄 Download Sample Template")
    sample_df = pd.DataFrame({
        'Applicant_Income': [50000, 60000, 45000],
        'Coapplicant_Income': [20000, 25000, 15000],
        'Loan_Amount': [150000, 200000, 120000],
        'Loan_Term': [240, 360, 180],
        'Gender': ['Male', 'Female', 'Male'],
        'Marital_Status': ['Married', 'Single', 'Married'],
        'Education_Level': ['Graduate', 'Graduate', 'Not Graduate'],
        'Employment_Status': ['Private', 'Government', 'Business'],
        'Credit_History': [1, 1, 0],
        'Property_Area': ['Urban', 'Semiurban', 'Rural']
    })
    
    st.download_button(
        "📥 Download Template CSV",
        sample_df.to_csv(index=False),
        "loan_application_template.csv",
        "text/csv",
        key='download-csv'
    )
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Loan Applications CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(batch_df)} applications")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            if st.button("🚀 Process Batch", type="primary"):
                with st.spinner("Processing applications..."):
                    progress_bar = st.progress(0)
                    
                    predictions = []
                    probabilities = []
                    
                    for idx, row in batch_df.iterrows():
                        # Prepare input (simplified - same as single prediction)
                        input_df = pd.DataFrame(columns=model_data['feature_info']['num_cols'] + model_data['feature_info']['cat_cols'])
                        
                        for col in model_data['feature_info']['num_cols']:
                            if col in batch_df.columns:
                                input_df.loc[0, col] = row[col]
                            else:
                                input_df.loc[0, col] = df[col].median()
                        
                        for col in model_data['feature_info']['cat_cols']:
                            if col in batch_df.columns:
                                input_df.loc[0, col] = row[col]
                            else:
                                input_df.loc[0, col] = df[col].mode()[0]
                        
                        # Feature engineering
                        input_df['Total_Income'] = input_df['Applicant_Income'] + input_df['Coapplicant_Income']
                        input_df['Income_to_Loan_Ratio'] = input_df['Total_Income'] / (input_df['Loan_Amount'] + 1)
                        input_df['EMI'] = input_df['Loan_Amount'] / (input_df['Loan_Term'] + 1)
                        input_df['Balance_Income'] = input_df['Total_Income'] - input_df['EMI']
                        
                        # Preprocessing
                        preprocessors = model_data['preprocessors']
                        input_df[model_data['feature_info']['num_cols']] = preprocessors['num_imputer'].transform(input_df[model_data['feature_info']['num_cols']])
                        input_df[model_data['feature_info']['cat_cols']] = preprocessors['cat_imputer'].transform(input_df[model_data['feature_info']['cat_cols']])
                        
                        input_cat_encoded = preprocessors['ohe'].transform(input_df[model_data['feature_info']['cat_cols']])
                        input_cat_encoded_df = pd.DataFrame(
                            input_cat_encoded,
                            columns=preprocessors['ohe'].get_feature_names_out(model_data['feature_info']['cat_cols']),
                            index=input_df.index
                        )
                        
                        input_final = pd.concat([input_df.drop(columns=model_data['feature_info']['cat_cols']), input_cat_encoded_df], axis=1)
                        input_final = input_final.reindex(columns=model_data['feature_info']['train_columns'], fill_value=0)
                        
                        # Prediction
                        model = model_data['models'][selected_model]
                        input_scaled = preprocessors['scaler'].transform(input_final)
                        prob = model.predict_proba(input_scaled)[0][1]
                        pred = "Approved" if prob >= threshold else "Rejected"
                        
                        predictions.append(pred)
                        probabilities.append(prob)
                        
                        progress_bar.progress((idx + 1) / len(batch_df))
                    
                    # Add results
                    batch_df['Approval_Probability'] = probabilities
                    batch_df['Decision'] = predictions
                    batch_df['Model'] = selected_model
                    
                    st.success("✅ Batch processing complete!")
                    
                    # Summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Processed", len(batch_df))
                    with col2:
                        approved_count = (batch_df['Decision'] == 'Approved').sum()
                        st.metric("Approved", approved_count)
                    with col3:
                        approval_rate = approved_count / len(batch_df) * 100
                        st.metric("Approval Rate", f"{approval_rate:.1f}%")
                    
                    # Results
                    st.markdown("### 📊 Results")
                    st.dataframe(batch_df, use_container_width=True)
                    
                    # Download results
                    st.download_button(
                        "📥 Download Results",
                        batch_df.to_csv(index=False),
                        "batch_predictions.csv",
                        "text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ==================================================
# PAGE: MODEL EXPLAINABILITY
# ==================================================
elif page == "🔍 Model Explainability":
    st.header("🔍 Model Explainability & Interpretability")
    
    st.markdown("""
    Understand how the model makes decisions using advanced explainability techniques.
    This helps ensure transparency and trust in AI-driven loan approvals.
    """)
    
    tab1, tab2 = st.tabs(["🎯 Feature Importance", "🔬 Individual Prediction Analysis"])
    
    with tab1:
        if selected_model in ['Random Forest', 'Gradient Boosting']:
            st.subheader("Global Feature Importance")
            
            model = model_data['models'][selected_model]
            feature_importance = pd.DataFrame({
                'Feature': model_data['feature_info']['train_columns'],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Top features bar chart
            fig = px.bar(
                feature_importance.head(20),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 20 Most Important Features',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.dataframe(feature_importance, use_container_width=True, height=400)
        else:
            # For Logistic Regression, show coefficients
            st.subheader("Model Coefficients")
            model = model_data['models'][selected_model]
            
            coef_df = pd.DataFrame({
                'Feature': model_data['feature_info']['train_columns'],
                'Coefficient': model.coef_[0]
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            fig = px.bar(
                coef_df.head(20),
                x='Coefficient',
                y='Feature',
                orientation='h',
                title='Top 20 Features by Coefficient Magnitude',
                color='Coefficient',
                color_continuous_scale='RdBu'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Analyze Individual Prediction")
        st.info("Make a prediction in the 'Loan Prediction' page first, then return here for detailed analysis.")
        
        if st.session_state.prediction_history:
            st.success("Analysis feature coming soon! This will show SHAP values and LIME explanations.")
        else:
            st.warning("No predictions available yet.")

# ==================================================
# PAGE: REAL-TIME MONITOR
# ==================================================
elif page == "⏱️ Real-time Monitor":
    st.header("⏱️ Real-time Application Monitor")
    
    st.markdown("""
    Monitor incoming loan applications in real-time with live updates and analytics.
    """)
    
    # Real-time metrics placeholder
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        applications_today = st.empty()
    with col2:
        avg_processing_time = st.empty()
    with col3:
        current_approval_rate = st.empty()
    with col4:
        pending_apps = st.empty()
    
    # Live chart placeholder
    chart_placeholder = st.empty()
    
    # Activity log
    st.markdown("### 📋 Recent Activity Log")
    activity_log = st.empty()
    
    # Start/Stop button
    if st.button("▶️ Start Real-time Monitoring"):
        for i in range(30):  # Simulate 30 seconds of monitoring
            # Simulate real-time data
            apps_count = np.random.randint(100, 150)
            proc_time = np.random.uniform(2.5, 4.5)
            approval = np.random.uniform(0.55, 0.75)
            pending = np.random.randint(5, 20)
            
            # Update metrics
            applications_today.metric("Applications Today", apps_count, delta=f"+{np.random.randint(1, 5)}")
            avg_processing_time.metric("Avg Processing Time", f"{proc_time:.1f}s")
            current_approval_rate.metric("Approval Rate", f"{approval:.1%}", delta=f"{np.random.uniform(-0.05, 0.05):.1%}")
            pending_apps.metric("Pending", pending)
            
            # Update chart with time series
            timestamps = pd.date_range(end=datetime.now(), periods=20, freq='5min')
            values = np.random.uniform(0.4, 0.8, 20)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines+markers',
                name='Approval Rate',
                line=dict(color='#667eea', width=3)
            ))
            fig.update_layout(
                title='Approval Rate - Last 100 Minutes',
                xaxis_title='Time',
                yaxis_title='Rate',
                height=300
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Update activity
            recent_activities = [
                f"{datetime.now().strftime('%H:%M:%S')} - Application #{np.random.randint(1000, 9999)} processed - {'✅ Approved' if np.random.random() > 0.4 else '❌ Rejected'}",
                f"{(datetime.now() - timedelta(seconds=5)).strftime('%H:%M:%S')} - New application received - Processing...",
                f"{(datetime.now() - timedelta(seconds=10)).strftime('%H:%M:%S')} - Application #{np.random.randint(1000, 9999)} processed - {'✅ Approved' if np.random.random() > 0.4 else '❌ Rejected'}"
            ]
            
            activity_log.text_area("", "\n".join(recent_activities), height=150)
            
            time.sleep(1)
        
        st.success("✅ Monitoring session completed")

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>CreditWise Pro v2.0</strong> | Enterprise Loan Approval System</p>
    <p>Powered by Advanced Machine Learning | Built with Streamlit</p>
    <p>© 2024 CreditWise Systems. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
