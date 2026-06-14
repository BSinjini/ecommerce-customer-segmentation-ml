import os

import streamlit as st
import plotly.express as px

from clustering import (
    load_rfm,
    preprocess,
    evaluate_k,
    run_segmentation
)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide"
)

st.title("Customer Segmentation Dashboard")

SEGMENT_COLORS = {
    "VIP Customers": "#2E86C1",
    "Loyal Customers": "#28B463",
    "Occasional Buyers": "#F39C12",
    "At-Risk Customers": "#E74C3C"
}

# --------------------------------------------------
# K Selection
# --------------------------------------------------

df = load_rfm()
X_scaled = preprocess(df)

k_values, inertia, silhouettes = evaluate_k(
    X_scaled
)

st.header("Model Selection")

col1, col2 = st.columns(2)

with col1:

    elbow_fig = px.line(
        x=k_values,
        y=inertia,
        markers=True,
        title="Elbow Method"
    )

    st.plotly_chart(
        elbow_fig,
        width="stretch"
    )

    elbow_fig.write_image(
        os.path.join(OUTPUT_DIR, "elbow_plot.png"),
        width=1400,
        height=900,
        scale=2
    )

with col2:

    sil_fig = px.line(
        x=k_values,
        y=silhouettes,
        markers=True,
        title="Silhouette Scores"
    )

    st.plotly_chart(
        sil_fig,
        width="stretch"
    )

    sil_fig.write_image(
        os.path.join(OUTPUT_DIR, "silhouette_plot.png"),
        width=1400,
        height=900,
        scale=2
    )

selected_k = st.slider(
    "Select Number of Clusters",
    2,
    10,
    4
)

# --------------------------------------------------
# Run Segmentation
# --------------------------------------------------

results = run_segmentation(
    selected_k,
    OUTPUT_DIR=OUTPUT_DIR
)

seg_df = results["data"]

with open(
    os.path.join(OUTPUT_DIR, "model_metadata.txt"),
    "w"
) as f:

    f.write(f"Selected K: {selected_k}\n")
    f.write(f"Silhouette Score: {results['silhouette']}\n")
    f.write(f"Calinski-Harabasz Score: {results['calinski']}\n")

st.header("Cluster Metrics")

col1, col2 = st.columns(2)

col1.metric(
    "Silhouette Score",
    results["silhouette"]
)

col2.metric(
    "Calinski-Harabasz",
    results["calinski"]
)

# --------------------------------------------------
# Segment Distribution
# --------------------------------------------------

st.header("Customer Segment Distribution")

dist = (
    seg_df["segment"]
    .value_counts()
    .reset_index()
)

dist.columns = [
    "segment",
    "count"
]

fig = px.pie(
    dist,
    names="segment",
    values="count",
    color="segment",
    color_discrete_map=SEGMENT_COLORS,
    title="Customer Segment Distribution"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# --------------------------------------------------
# PCA Visualization
# --------------------------------------------------

st.header(
    "Customer Cluster Visualization (PCA)"
)

pca_fig = px.scatter(
    seg_df,
    x="PCA1",
    y="PCA2",
    color="segment",
    color_discrete_map=SEGMENT_COLORS,
    hover_data=[
        "customer_id",
        "recency",
        "frequency",
        "monetary"
    ],
    title="PCA Projection of Customer Segments"
)

st.plotly_chart(
    pca_fig,
    width="stretch"
)

pca_fig.write_image(
    os.path.join(
        OUTPUT_DIR,
        "pca_cluster_visualization.png"
    ),
    width=1600,
    height=1000,
    scale=2
)

# --------------------------------------------------
# t-SNE Visualization
# --------------------------------------------------

st.header(
    "Customer Cluster Visualization (t-SNE)"
)

tsne_fig = px.scatter(
    seg_df,
    x="TSNE1",
    y="TSNE2",
    color="segment",
    color_discrete_map=SEGMENT_COLORS,
    hover_data=[
        "customer_id",
        "recency",
        "frequency",
        "monetary"
    ],
    title="t-SNE Projection of Customer Segments"
)

st.plotly_chart(
    tsne_fig,
    width="stretch"
)

tsne_fig.write_image(
    os.path.join(
        OUTPUT_DIR,
        "tsne_cluster_visualization.png"
    ),
    width=1600,
    height=1000,
    scale=2
)

# --------------------------------------------------
# Segment Profiles
# --------------------------------------------------

st.header("Business Segment Profiles")

segment_profile = (
    seg_df.groupby("segment")
    [
        [
            "recency",
            "frequency",
            "monetary"
        ]
    ]
    .mean()
)

st.dataframe(segment_profile)

# --------------------------------------------------
# Customer Table
# --------------------------------------------------

st.header("Customer Details")

st.dataframe(seg_df)