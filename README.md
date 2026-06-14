# Customer Segmentation and Behavioral Analysis in E-Commerce Using Unsupervised Machine Learning Techniques

## Overview

This project implements an intelligent customer segmentation framework using **RFM (Recency, Frequency, Monetary)** analysis and **Unsupervised Machine Learning** techniques. The objective is to identify distinct customer groups based on purchasing behavior and transform transactional data into actionable business insights.

The system utilizes the **Online Retail Dataset** from the UCI Machine Learning Repository and applies clustering techniques to segment customers into meaningful business categories such as:

* VIP Customers
* Loyal Customers
* Occasional Buyers
* At-Risk Customers

The project provides:

* Interactive Streamlit Dashboard
* K-Means Customer Segmentation
* Automatic K Selection Support
* Gaussian Mixture Model (GMM) Confidence Analysis
* PCA Visualization
* t-SNE Visualization
* Flask REST API
* Automated CSV and Report Generation

---

# Objectives

* Perform customer segmentation using RFM analysis.
* Apply K-Means clustering to identify customer groups.
* Evaluate clustering quality using Elbow Method and Silhouette Analysis.
* Measure cluster confidence using Gaussian Mixture Models (GMM).
* Visualize customer clusters using PCA and t-SNE dimensionality reduction.
* Generate business-friendly customer segment labels.
* Provide interactive analytics through a web dashboard.
* Export analytical outputs for reporting and business use.

---

# Dataset

## Source

UCI Machine Learning Repository

Dataset: Online Retail

## Description

The dataset contains all transactions occurring between December 2010 and December 2011 for a UK-based online retail company.

## Attributes

| Attribute   | Description                  |
| ----------- | ---------------------------- |
| InvoiceNo   | Invoice Number               |
| StockCode   | Product Code                 |
| Description | Product Description          |
| Quantity    | Quantity Purchased           |
| InvoiceDate | Date and Time of Transaction |
| UnitPrice   | Unit Price                   |
| CustomerID  | Customer Identifier          |
| Country     | Customer Country             |

---

# System Architecture

```text
Transaction Data
        ↓
Data Cleaning & Preprocessing
        ↓
RFM Feature Engineering
        ↓
Feature Scaling
        ↓
K-Means Clustering
        ↓
Cluster Evaluation
        ↓
Business Segment Mapping
        ↓
PCA & t-SNE Visualization
        ↓
Dashboard & Reporting
```

---

# Technologies Used

## Programming Language

* Python 3.11+

## Libraries

* Pandas
* NumPy
* Scikit-Learn
* Plotly
* Streamlit
* Flask
* SQLite
* Kaleido

## Machine Learning Techniques

* K-Means Clustering
* Gaussian Mixture Models (GMM)
* Principal Component Analysis (PCA)
* t-Distributed Stochastic Neighbor Embedding (t-SNE)

---

# Project Structure

```text
project/
│
├── app.py
├── clustering.py
├── dashboard.py
├── requirements.txt
├── README.md
├── ecommerce_retail.db
│
├── outputs/
│   ├── customer_segments.csv
│   ├── cluster_summary.csv
│   ├── elbow_plot.png
│   ├── silhouette_plot.png
│   ├── segment_distribution.png
│   ├── pca_cluster_visualization.png
│   ├── tsne_cluster_visualization.png
│   └── model_metadata.txt
│
└── uci_data/
```

---

# Installation

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Dashboard

```bash
streamlit run dashboard.py
```

Dashboard URL:

```text
http://localhost:8501
```

---

# Running the Flask API

```bash
python app.py
```

API URL:

```text
http://localhost:5000
```

Example Endpoint:

```text
http://localhost:5000/api/v1/metrics
```

---

# Customer Segments

## VIP Customers

### Characteristics

* Recent purchases
* High order frequency
* High monetary contribution

### Business Strategy

* Premium rewards
* Exclusive offers
* Personalized engagement

---

## Loyal Customers

### Characteristics

* Frequent purchasers
* Consistent revenue contribution

### Business Strategy

* Loyalty programs
* Membership benefits

---

## Occasional Buyers

### Characteristics

* Moderate engagement
* Irregular purchasing patterns

### Business Strategy

* Promotional campaigns
* Product recommendations

---

## At-Risk Customers

### Characteristics

* Long inactivity period
* Low recent engagement

### Business Strategy

* Re-engagement campaigns
* Discount incentives

---

# Evaluation Metrics

## Elbow Method

Used to determine the optimal number of clusters by measuring Within-Cluster Sum of Squares (WCSS).

## Silhouette Score

Measures cluster cohesion and separation.

Range:

```text
-1 to +1
```

Higher values indicate better clustering quality.

## Calinski-Harabasz Index

Measures the ratio of between-cluster dispersion to within-cluster dispersion.

Higher values indicate better-defined clusters.

## Gaussian Mixture Model Confidence

Provides probabilistic confidence scores for cluster assignments.

---

# Visualizations

The dashboard generates the following visual outputs:

## Elbow Method Plot

Used for selecting the optimal number of clusters.

## Silhouette Score Plot

Measures clustering quality across different K values.

## Customer Segment Distribution

Pie chart showing customer segment proportions.

## PCA Cluster Visualization

Two-dimensional representation of customer segments using Principal Component Analysis.

## t-SNE Cluster Visualization

Advanced nonlinear visualization showing customer cluster separability.

---

# Generated Outputs

The system automatically generates:

## customer_segments.csv

Contains:

* Customer ID
* Assigned Cluster
* Business Segment
* Recency
* Frequency
* Monetary Value

## cluster_summary.csv

Contains segment-level aggregated statistics.

## model_metadata.txt

Contains:

* Selected K Value
* Silhouette Score
* Calinski-Harabasz Index

## Visualization Files

* elbow_plot.png
* silhouette_plot.png
* segment_distribution.png
* pca_cluster_visualization.png
* tsne_cluster_visualization.png

---

# Future Enhancements

* DBSCAN Clustering
* Hierarchical Clustering
* Customer Lifetime Value Prediction
* Deep Learning-Based Segmentation
* Real-Time Streaming Analytics
* Cloud Deployment on AWS
* PostgreSQL Integration
* Docker Containerization

---

# Academic Contribution

This project demonstrates the practical application of unsupervised machine learning techniques in customer analytics. By combining RFM analysis, clustering algorithms, probabilistic confidence estimation, and advanced visualization techniques, the system provides meaningful business intelligence for customer relationship management and targeted marketing strategies.

---

# Author

**BCA Major Project**

**Customer Segmentation and Behavioral Analysis in E-Commerce Using Unsupervised Machine Learning Techniques**
