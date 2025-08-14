# market_basket_analysis
An interactive BI dashboard using Streamlit to analyze customer behavior. It applies K-Means clustering for RFM segmentation and Apriori for market basket analysis, providing actionable insights for marketing and sales strategies to boost retention and revenue.

# Customer Segmentation & Market Basket Analysis Dashboard

**Live Demo:** [Link](https://marketbasketanalysis-aiq4memm74q2msxnt5w444.streamlit.app/)

## Project Overview

This project is an interactive Business Intelligence dashboard built with Streamlit. It analyzes customer transaction data to solve two key business problems:

1.  **Customer Segmentation:** Uses RFM (Recency, Frequency, Monetary) analysis and K-Means clustering to group customers into distinct personas like "Champions," "At Risk," and "Rising Stars."
2.  **Market Basket Analysis:** Implements the Apriori algorithm to discover which products are frequently purchased together, uncovering valuable cross-selling and bundling opportunities.

The goal is to provide a user-friendly tool for non-technical stakeholders to explore customer behavior and make data-driven decisions for marketing and sales strategies.

## Key Features

  * **Interactive Business Summary:** A high-level overview of key findings and actionable recommendations.
  * **Dynamic Customer Segmentation Dashboard:**
      * Interactive 3D scatter plot to visualize customer segments.
      * Detailed summary table of segment characteristics.
      * Customer Lookup tool to find the persona for any customer ID.
  * **Market Basket Analysis Interface:**
      * Association Rules Explorer with dynamic filters for `support`, `confidence`, and `lift`.
      * Product Recommendation Engine to find co-purchased items.

## Technical Stack

  * **Language:** Python
  * **Web Framework:** Streamlit
  * **Data Analysis & ML:** Pandas, NumPy, Scikit-learn, Mlxtend
  * **Data Visualization:** Plotly, Matplotlib, Seaborn
  * **Deployment:** Streamlit Community Cloud

## Setup and Installation

To run this project locally, please follow these steps:

**1. Clone the repository:**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install the required libraries:**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application:**
Make sure you are in the root directory of the project.

```bash
streamlit run app.py
```

The application should now be open in your web browser.

## Repository Structure

```
├── app.py                      # Main Streamlit application script
├── requirements.txt            # Python libraries required to run the app
├── customer_transactions.csv   # Raw transactional dataset
├── association_rules.csv       # Pre-computed association rules
├── business_report.pdf         # Final project report
├── market_basket_analysis.ipynb # Jupyter Notebook with the data analysis
└── README.md                   # This file
```
