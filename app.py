import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation App", layout="centered")

st.title("🧠 Customer Segmentation using K-Means")
st.markdown("Segment customers based on **Annual Income**, **Spending Score**, and other features using unsupervised learning.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your customer CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### Preview of Data:")
    st.dataframe(df.head())

    df.columns = df.columns.str.strip()  # Clean column names

    try:
        # Step 1: Let user select features
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected_columns = st.multiselect("📌 Select features for clustering", numeric_cols, default=['Annual Income (k$)', 'Spending Score (1-100)'])

        if len(selected_columns) >= 2:
            X = df[selected_columns]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Step 2: Select number of clusters
            k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=5)

            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            df['Cluster'] = labels

            # Step 3: 2D Cluster Plot (only if those 2 features exist)
            if 'Annual Income (k$)' in selected_columns and 'Spending Score (1-100)' in selected_columns:
                st.write("### 📊 2D Cluster Plot")
                fig, ax = plt.subplots()
                sns.scatterplot(
                    x='Annual Income (k$)',
                    y='Spending Score (1-100)',
                    hue='Cluster',
                    data=df,
                    palette='Set2',
                    s=100,
                    ax=ax
                )
                plt.title("Customer Segments")
                st.pyplot(fig)

            # Step 4: 3D Plot if Age is available
            if all(col in df.columns for col in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
                st.write("### 📈 3D Cluster Plot (Age, Income, Spending Score)")
                fig3d = px.scatter_3d(
                    df,
                    x='Age',
                    y='Annual Income (k$)',
                    z='Spending Score (1-100)',
                    color='Cluster',
                    title='3D Customer Segments',
                    opacity=0.7
                )
                st.plotly_chart(fig3d)

            # Step 5: Cluster Summary
            st.write("### 📄 Cluster Summary (Mean Values)")
            st.dataframe(df.groupby("Cluster")[selected_columns].mean())

            # Step 6: Feature Insight
            st.markdown("""
            ---
            ### 🧠 Business Insights:
            - Customers with **high income but low spending** might need better engagement or discounts.
            - **Young customers with high spending scores** could represent trend-driven buyers.
            - Outliers in any cluster may represent **high-value or risky segments**.
            """)

        else:
            st.warning("⚠️ Please select at least 2 numerical columns for clustering.")

    except KeyError as e:
        st.error(f"❌ Column not found: {e}")

else:
    st.info("📂 Please upload a CSV file. Expected columns: `Age`, `Annual Income (k$)`, `Spending Score (1-100)`, etc.")
