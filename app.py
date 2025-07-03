import streamlit as st
import pandas as pd
import numpy as np
import os
from utils.classification_utils import evaluate_classifiers, get_confusion
from utils.clustering_utils import run_kmeans, elbow_method
from utils.association_utils import run_apriori
from utils.regression_utils import run_regressors

DATA_PATH = 'data/synthetic_telecom_survey.csv'

st.set_page_config(page_title="Telecom Analytics", layout="wide")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

def main():
    st.title("Telecom Churn & Customer Insights Dashboard")
    df = load_data(DATA_PATH)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Visualization",
        "Classification",
        "Clustering",
        "Association Mining",
        "Regression"
    ])
    
    with tab1:
        st.header("Key Insights & Filters")
        st.write(df.describe())
        st.write("Add your visualizations and insights here!")
        # Example: st.bar_chart(df['Satisfaction_Score'])
        # Add 10+ complex insights here as required

    with tab2:
        st.header("Classification (Churn/No Churn etc.)")
        target = st.selectbox("Select target variable", [col for col in df.columns if df[col].nunique()<10 and df[col].dtype in [np.int64, object]])
        X = pd.get_dummies(df.drop(target, axis=1), drop_first=True)
        y = df[target]
        results, preds, probs = evaluate_classifiers(X, y)
        st.dataframe(results)
        model_choice = st.selectbox("Confusion Matrix for:", results["Model"])
        if st.button("Show Confusion Matrix"):
            y_true, y_pred = preds[model_choice]
            st.write(get_confusion(y_true, y_pred))
        # ROC Curve plotting here

    with tab3:
        st.header("Clustering (Customer Segments)")
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        cluster_vars = st.multiselect("Variables for clustering", df.select_dtypes(include=[np.number]).columns.tolist())
        if cluster_vars:
            clusters, inertia, sil = run_kmeans(df[cluster_vars], n_clusters)
            st.write(f"Silhouette Score: {sil:.2f}")
            st.write(pd.DataFrame({'Cluster': clusters}).value_counts())
            # Persona table, download, elbow chart

    with tab4:
        st.header("Association Rule Mining")
        assoc_cols = st.multiselect("Columns for association", df.columns)
        min_sup = st.slider("Min Support", 0.01, 0.3, 0.05)
        min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.5)
        if assoc_cols:
            rules = run_apriori(df, assoc_cols, min_support=min_sup, min_confidence=min_conf)
            st.dataframe(rules.sort_values("confidence", ascending=False).head(10))

    with tab5:
        st.header("Regression Insights")
        y_var = st.selectbox("Target for regression", df.select_dtypes(include=[np.number]).columns)
        X_vars = st.multiselect("Features", [c for c in df.select_dtypes(include=[np.number]).columns if c != y_var])
        if X_vars:
            reg_results = run_regressors(df[X_vars], df[y_var])
            st.dataframe(reg_results)
            # Add plots and insights

if __name__ == '__main__':
    main()
