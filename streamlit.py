import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Page configuration
st.set_page_config(
    page_title="Online Shoppers Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

# Title
st.title("🛒 Online Shoppers Purchase Intention Prediction")
st.markdown("---")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("online_shoppers_intention.csv")
    df = df.drop_duplicates().reset_index(drop=True)
    return df

@st.cache_resource
def train_models(df):
    """Train all models and return them with the scaler"""
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    
    # Encode Month
    month_order = {'Feb': 1, 'Mar': 2, 'May': 3, 'June': 4, 'Jul': 5, 
                   'Aug': 6, 'Sep': 7, 'Oct': 8, 'Nov': 9, 'Dec': 10}
    df_encoded['Month'] = df_encoded['Month'].map(month_order)
    
    # VisitorType
    visitor_type_mapping = {vt: idx for idx, vt in enumerate(df['VisitorType'].unique())}
    df_encoded['VisitorType'] = df_encoded['VisitorType'].map(visitor_type_mapping)
    
    # Weekend & Revenue
    df_encoded['Weekend'] = df_encoded['Weekend'].astype(int)
    df_encoded['Revenue'] = df_encoded['Revenue'].astype(int)
    
    X = df_encoded.drop('Revenue', axis=1)
    y = df_encoded['Revenue']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                          random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        trained_models[name] = model
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }
    
    return trained_models, results, scaler, X.columns.tolist(), visitor_type_mapping

# Load data
df = load_data()
trained_models, model_results, scaler, feature_columns, visitor_type_mapping = train_models(df)

# Sidebar - Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Data Overview", "📈 Model Performance", "🎯 Prediction Panel"])

# ==================== PAGE 1: Data Overview ====================
if page == "📊 Data Overview":
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", f"{df.shape[1] - 1}")
    with col3:
        purchase_rate = (df['Revenue'].sum() / len(df)) * 100
        st.metric("Purchase Rate", f"{purchase_rate:.1f}%")
    with col4:
        st.metric("No Purchase Rate", f"{100 - purchase_rate:.1f}%")
    
    st.markdown("---")
    
    # Data sample
    st.subheader("Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # Target distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Target Variable Distribution")
        target_counts = df['Revenue'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(["No Purchase", "Purchase"], target_counts.values, 
                     color=["#FF6B6B", "#4ECDC4"], edgecolor='white', linewidth=2)
        ax.bar_label(bars, labels=[f"{v:,}\n({v/len(df)*100:.1f}%)" for v in target_counts.values])
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Purchase Intention")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Features Information")
        st.markdown("""
        **Numerical Features (10):**
        - Administrative, Administrative_Duration
        - Informational, Informational_Duration
        - ProductRelated, ProductRelated_Duration
        - BounceRates, ExitRates, PageValues, SpecialDay
        
        **Categorical Features (7):**
        - OperatingSystems, Browser, Region, TrafficType
        - VisitorType, Weekend, Month
        """)
    
    st.markdown("---")
    
    # Numerical features distribution
    st.subheader("Numerical Features Distribution")
    numerical_features = ['Administrative', 'Administrative_Duration', 'Informational', 
                          'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                          'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
    
    selected_feature = st.selectbox("Select a feature to visualize", numerical_features)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[selected_feature], bins=50, color='#4ECDC4', edgecolor='white', alpha=0.8)
    ax.set_title(f'Distribution of {selected_feature}')
    ax.set_xlabel(selected_feature)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = df[numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                linewidths=0.5, square=True, ax=ax)
    ax.set_title('Correlation Matrix of Numerical Features')
    st.pyplot(fig)

# ==================== PAGE 2: Model Performance ====================
elif page == "📈 Model Performance":
    st.header("📈 Model Performance Comparison")
    
    # Results DataFrame
    results_df = pd.DataFrame(model_results).T
    results_df = results_df.sort_values('F1-Score', ascending=False)
    
    st.subheader("Performance Metrics Summary")
    st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
    
    st.markdown("---")
    
    # Best model
    best_model = results_df['F1-Score'].idxmax()
    best_f1 = results_df.loc[best_model, 'F1-Score']
    st.success(f"🏆 Best Model: **{best_model}** with F1-Score: **{best_f1:.4f}**")
    
    st.markdown("---")
    
    # Visualization
    st.subheader("Performance Visualization")
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    selected_metric = st.selectbox("Select metric to compare", metrics)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_results = results_df.sort_values(selected_metric, ascending=True)
    bars = ax.barh(sorted_results.index, sorted_results[selected_metric], 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax.bar_label(bars, fmt='%.4f', padding=5)
    ax.set_xlabel(selected_metric)
    ax.set_title(f'{selected_metric} Comparison Across Models')
    ax.set_xlim(0, 1.1)
    st.pyplot(fig)
    
    # All metrics comparison
    st.subheader("All Metrics Comparison")
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    for idx, metric in enumerate(metrics):
        sorted_results = results_df.sort_values(metric, ascending=False)
        bars = axes[idx].bar(range(len(sorted_results)), sorted_results[metric], 
                             color=['#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B'])
        axes[idx].set_title(metric)
        axes[idx].set_xticks(range(len(sorted_results)))
        axes[idx].set_xticklabels(sorted_results.index, rotation=45, ha='right', fontsize=8)
        axes[idx].set_ylim(0, 1)
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)

# ==================== PAGE 3: Prediction Panel ====================
elif page == "🎯 Prediction Panel":
    st.header("🎯 Purchase Prediction Control Panel")
    st.markdown("Enter customer session details to predict purchase intention")
    
    st.markdown("---")
    
    # Model selection
    selected_model_name = st.selectbox("Select Model for Prediction", list(trained_models.keys()))
    selected_model = trained_models[selected_model_name]
    
    st.markdown("---")
    
    # Input features in columns
    st.subheader("📝 Enter Session Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Page Visit Counts**")
        administrative = st.number_input("Administrative Pages", min_value=0, max_value=50, value=0, step=1)
        informational = st.number_input("Informational Pages", min_value=0, max_value=50, value=0, step=1)
        product_related = st.number_input("Product Related Pages", min_value=0, max_value=500, value=1, step=1)
        
        st.markdown("**Time Duration (seconds)**")
        administrative_duration = st.number_input("Administrative Duration", min_value=0.0, max_value=5000.0, value=0.0, step=1.0)
        informational_duration = st.number_input("Informational Duration", min_value=0.0, max_value=5000.0, value=0.0, step=1.0)
        product_related_duration = st.number_input("Product Related Duration", min_value=0.0, max_value=70000.0, value=0.0, step=1.0)
    
    with col2:
        st.markdown("**Behavior Metrics**")
        bounce_rates = st.slider("Bounce Rates", min_value=0.0, max_value=0.2, value=0.02, step=0.001, format="%.3f")
        exit_rates = st.slider("Exit Rates", min_value=0.0, max_value=0.2, value=0.04, step=0.001, format="%.3f")
        page_values = st.number_input("Page Values", min_value=0.0, max_value=400.0, value=0.0, step=0.1)
        special_day = st.slider("Special Day (closeness)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    with col3:
        st.markdown("**Visitor Information**")
        month = st.selectbox("Month", ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        operating_systems = st.selectbox("Operating System", sorted(df['OperatingSystems'].unique()))
        browser = st.selectbox("Browser", sorted(df['Browser'].unique()))
        region = st.selectbox("Region", sorted(df['Region'].unique()))
        traffic_type = st.selectbox("Traffic Type", sorted(df['TrafficType'].unique()))
        visitor_type = st.selectbox("Visitor Type", df['VisitorType'].unique())
        weekend = st.checkbox("Weekend Visit")
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔮 Predict Purchase Intention", type="primary", use_container_width=True):
        # Prepare input data
        month_order = {'Feb': 1, 'Mar': 2, 'May': 3, 'June': 4, 'Jul': 5, 
                       'Aug': 6, 'Sep': 7, 'Oct': 8, 'Nov': 9, 'Dec': 10}
        
        input_data = pd.DataFrame({
            'Administrative': [administrative],
            'Administrative_Duration': [administrative_duration],
            'Informational': [informational],
            'Informational_Duration': [informational_duration],
            'ProductRelated': [product_related],
            'ProductRelated_Duration': [product_related_duration],
            'BounceRates': [bounce_rates],
            'ExitRates': [exit_rates],
            'PageValues': [page_values],
            'SpecialDay': [special_day],
            'Month': [month_order[month]],
            'OperatingSystems': [operating_systems],
            'Browser': [browser],
            'Region': [region],
            'TrafficType': [traffic_type],
            'VisitorType': [visitor_type_mapping.get(visitor_type, 0)],
            'Weekend': [int(weekend)]
        })
        
        # Ensure columns are in correct order
        input_data = input_data[feature_columns]
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = selected_model.predict(input_scaled)[0]
        prediction_proba = selected_model.predict_proba(input_scaled)[0]
        
        st.markdown("---")
        
        # Display results
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if prediction == 1:
                st.success("## ✅ Prediction: WILL PURCHASE")
                st.balloons()
            else:
                st.error("## ❌ Prediction: WILL NOT PURCHASE")
            
            st.markdown("### Probability Breakdown")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.metric("No Purchase Probability", f"{prediction_proba[0]*100:.2f}%")
            with prob_col2:
                st.metric("Purchase Probability", f"{prediction_proba[1]*100:.2f}%")
            
            # Probability bar
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.barh(['Prediction'], [prediction_proba[0]], color='#FF6B6B', label='No Purchase')
            ax.barh(['Prediction'], [prediction_proba[1]], left=[prediction_proba[0]], color='#4ECDC4', label='Purchase')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.legend(loc='upper right')
            ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
            st.pyplot(fig)
    
    st.markdown("---")
    
    # Quick test presets
    st.subheader("🧪 Quick Test Presets")
    st.markdown("Click a button to fill in sample data:")
    
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    
    with preset_col1:
        if st.button("📱 Casual Browser", use_container_width=True):
            st.info("Low engagement user - typically doesn't purchase. Refresh and adjust values manually.")
    
    with preset_col2:
        if st.button("🛍️ Active Shopper", use_container_width=True):
            st.info("High engagement user - higher purchase likelihood. Refresh and adjust values manually.")
    
    with preset_col3:
        if st.button("🎯 Returning Customer", use_container_width=True):
            st.info("Returning visitor - moderate purchase likelihood. Refresh and adjust values manually.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Online Shoppers Purchase Intention Prediction System</p>
    <p>Built with Streamlit | Data Mining Project</p>
</div>
""", unsafe_allow_html=True)
