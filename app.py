import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ======================================================================================
# App Configuration
# ======================================================================================
st.set_page_config(
    page_title="Business Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# ======================================================================================
# Caching Functions for Performance
# ======================================================================================
# Cache the data loading to avoid reloading on every interaction.
@st.cache_data
def load_data():
    """Loads the required CSV files."""
    try:
        transactions_df = pd.read_csv('customer_transactions.csv')
        rules_df = pd.read_csv('association_rules.csv')
        # Convert frozensets from string representation back to frozensets
        rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: frozenset(eval(x)))
        rules_df['consequents'] = rules_df['consequents'].apply(lambda x: frozenset(eval(x)))
        return transactions_df, rules_df
    except FileNotFoundError:
        st.error("Error: Make sure `customer_transactions.csv` and `association_rules.csv` are in the same folder as `app.py`.")
        return None, None

# Cache the segmentation analysis to avoid re-computing.
@st.cache_data
def perform_segmentation(df):
    """Performs RFM and K-Means clustering."""
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    snapshot_date = df['transaction_date'].max() + dt.timedelta(days=1)

    # RFM Calculation
    recency_df = df.groupby('customer_id').agg({'transaction_date': lambda x: (snapshot_date - x.max()).days}).rename(columns={'transaction_date': 'Recency'})
    frequency_df = df.groupby('customer_id').agg({'transaction_id': 'count'}).rename(columns={'transaction_id': 'Frequency'})
    monetary_df = df.groupby('customer_id').agg({'amount': 'sum'}).rename(columns={'amount': 'Monetary'})
    rfm_df = recency_df.join(frequency_df).join(monetary_df)

    # Preprocessing and Clustering
    rfm_log = np.log1p(rfm_df)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    rfm_scaled = pd.DataFrame(rfm_scaled, index=rfm_df.index, columns=rfm_df.columns)

    # K-Means with optimal K=5 based on notebook analysis
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    rfm_df['Cluster'] = kmeans.labels_

    # Define personas based on your analysis
    persona_map = {
        2: 'Champions',
        4: 'Rising Stars',
        0: 'Potential Loyalists',
        3: 'Needs Attention',
        1: 'At Risk'
    }
    rfm_df['Persona'] = rfm_df['Cluster'].map(persona_map)
    return rfm_df

# ======================================================================================
# Main App Logic
# ======================================================================================
# Load data
transactions_df, rules_df = load_data()

if transactions_df is not None and rules_df is not None:
    # Perform segmentation
    rfm_results = perform_segmentation(transactions_df.copy())

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ("Business Summary", "Customer Segmentation", "Market Basket Analysis")
    )

    # ----------------------------------------------------------------------------------
    # Page 1: Business Summary
    # ----------------------------------------------------------------------------------
    if page == "Business Summary":
        st.title("📈 Business Intelligence Summary")
        st.markdown("""
        This dashboard presents actionable insights from an unsupervised learning analysis of customer transaction data.
        We have identified key customer segments and product associations to help drive strategic marketing and sales initiatives.
        """)

        st.header("Key Objectives")
        st.markdown("""
        - **Increase Customer Retention:** By identifying and targeting at-risk customers.
        - **Boost Order Value:** By promoting frequently co-purchased products.
        - **Personalize Marketing:** By tailoring campaigns to specific customer personas.
        """)

        st.header("Actionable Recommendations")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Targeted Marketing Campaigns")
            st.info("""
            - **Champions:** Implement a loyalty program and offer exclusive previews.
            - **Rising Stars:** Encourage repeat purchases with a follow-up discount.
            - **Potential Loyalists & Needs Attention:** Deploy personalized "we miss you" campaigns.
            """)
        with col2:
            st.subheader("Smart Product Bundling")
            st.info("""
            - Create a "Frequently Bought Together" feature on the e-commerce site.
            - Offer small discounts for purchasing associated products as a bundle.
            - In physical stores, place associated products in close proximity.
            """)

    # ----------------------------------------------------------------------------------
    # Page 2: Customer Segmentation
    # ----------------------------------------------------------------------------------
    elif page == "Customer Segmentation":
        st.title("👥 Customer Segmentation Dashboard")
        st.markdown("Customers have been segmented into 5 distinct personas based on their Recency, Frequency, and Monetary (RFM) scores.")

        # Display 3D Scatter Plot
        st.header("Interactive 3D Cluster Visualization")
        fig = px.scatter_3d(
            rfm_results,
            x='Recency', y='Frequency', z='Monetary',
            color='Persona', symbol='Persona',
            size_max=18, opacity=0.8,
            title='3D Scatter Plot of Customer Segments'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display Segment Summary and Customer Lookup
        col1, col2 = st.columns([2, 1])
        with col1:
            st.header("Segment Characteristics")
            summary_df = rfm_results.groupby('Persona').agg({
                'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean', 'Cluster': 'count'
            }).rename(columns={'Cluster': 'Customer Count'}).sort_values(by='Monetary', ascending=False)
            st.dataframe(summary_df.style.format({
                "Recency": "{:.0f}", "Frequency": "{:.1f}", "Monetary": "${:,.2f}"
            }))
        with col2:
            st.header("Customer Lookup")
            customer_id_input = st.text_input("Enter Customer ID (e.g., CUST_001)")
            if customer_id_input:
                if customer_id_input in rfm_results.index:
                    persona = rfm_results.loc[customer_id_input, 'Persona']
                    st.success(f"Customer **{customer_id_input}** belongs to the **{persona}** segment.")
                else:
                    st.warning(f"Customer ID '{customer_id_input}' not found.")

    # ----------------------------------------------------------------------------------
    # Page 3: Market Basket Analysis
    # ----------------------------------------------------------------------------------
    elif page == "Market Basket Analysis":
        st.title("🛒 Market Basket Analysis Interface")
        st.markdown("Discover which products are frequently purchased together to create cross-selling and bundling opportunities.")

        # --- Association Rules Explorer ---
        st.header("Association Rules Explorer")
        col1, col2, col3 = st.columns(3)
        with col1:
            min_lift = st.slider("Minimum Lift", 1.0, 3.0, 1.5, 0.1)
        with col2:
            min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.1, 0.05)
        with col3:
            min_support = st.slider("Minimum Support", 0.0, 0.1, 0.01, 0.005)

        # Filter rules based on slider inputs
        filtered_rules = rules_df[
            (rules_df['lift'] >= min_lift) &
            (rules_df['confidence'] >= min_confidence) &
            (rules_df['support'] >= min_support)
        ].sort_values(by='lift', ascending=False)

        # Display filtered rules
        st.dataframe(filtered_rules)
        st.info(f"Showing **{len(filtered_rules)}** rules out of **{len(rules_df)}** total rules.")

        # --- Product Recommendation Engine ---
        st.header("Product Recommendation Engine")
        st.markdown("Select one or more products to see what customers frequently buy with them.")
        all_products = sorted(list(set([item for sublist in rules_df['antecedents'].tolist() + rules_df['consequents'].tolist() for item in sublist])))
        selected_products = st.multiselect("Select Products", all_products)

        if selected_products:
            recommendations = set()
            for product in selected_products:
                # Find rules where the selected product is in the antecedents
                matches = rules_df[rules_df['antecedents'].apply(lambda x: product in x)]
                for _, row in matches.iterrows():
                    recommendations.update(row['consequents'])
            # Remove already selected items from recommendations
            recommendations -= set(selected_products)

            if recommendations:
                st.success(f"Customers who buy **{', '.join(selected_products)}** also frequently buy:")
                for rec in sorted(list(recommendations)):
                    st.markdown(f"- **{rec}**")
            else:
                st.warning("No strong recommendations found for the selected product(s). Try adjusting the filters above.")
