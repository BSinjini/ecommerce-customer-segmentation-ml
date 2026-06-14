import sqlite3
import numpy as np
import pandas as pd
import os

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DB_FILE = "ecommerce_retail.db"

# --------------------------------------------------
# Load RFM
# --------------------------------------------------

def load_rfm():

    conn = sqlite3.connect(DB_FILE)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        AND name='online_retail'
    """)

    if cursor.fetchone() is None:
        conn.close()
        raise RuntimeError(
            "Table 'online_retail' does not exist."
        )

    query = """
    WITH cleaned_ledger AS (
        SELECT
            CAST(CustomerID AS TEXT) AS customer_id,
            Quantity,
            UnitPrice,
            InvoiceDate,
            InvoiceNo
        FROM online_retail
        WHERE CustomerID IS NOT NULL
          AND Quantity > 0
          AND UnitPrice > 0
    ),
    aggregation_matrix AS (
        SELECT
            customer_id,
            (julianday('2011-12-10')
             - julianday(MAX(InvoiceDate))) AS recency,
            COUNT(DISTINCT InvoiceNo) AS frequency,
            SUM(Quantity * UnitPrice) AS monetary
        FROM cleaned_ledger
        GROUP BY customer_id
    )
    SELECT *
    FROM aggregation_matrix
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

def preprocess(df):

    X = df[['recency', 'frequency', 'monetary']].values

    # safer than log10
    X = np.log1p(X)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled


# --------------------------------------------------
# Elbow + Silhouette
# --------------------------------------------------

def evaluate_k(X_scaled):

    inertia_values = []
    silhouette_values = []

    k_range = range(2, 11)

    for k in k_range:

        model = KMeans(
            n_clusters=k,
            init='k-means++',
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(X_scaled)

        inertia_values.append(model.inertia_)

        silhouette_values.append(
            silhouette_score(
                X_scaled,
                labels
            )
        )

    return (
        list(k_range),
        inertia_values,
        silhouette_values
    )


# --------------------------------------------------
# Business labels
# --------------------------------------------------

def assign_business_names(cluster_summary):

    labels = {}

    highest_monetary = cluster_summary['monetary'].idxmax()
    highest_frequency = cluster_summary['frequency'].idxmax()
    highest_recency = cluster_summary['recency'].idxmax()

    labels[highest_monetary] = "VIP Customers"

    labels[highest_frequency] = "Loyal Customers"

    labels[highest_recency] = "At-Risk Customers"

    for cluster in cluster_summary.index:

        if cluster not in labels:
            labels[cluster] = "Occasional Buyers"

    return labels


# --------------------------------------------------
# Main clustering
# --------------------------------------------------

def run_segmentation(k, OUTPUT_DIR="outputs"):

    df = load_rfm()

    X_scaled = preprocess(df)

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(X_scaled)

    gmm = GaussianMixture(
        n_components=k,
        random_state=42
    )

    gmm.fit(X_scaled)

    probs = gmm.predict_proba(X_scaled)

    silhouette_vals = silhouette_samples(
        X_scaled,
        labels
    )

    df["cluster"] = labels

    # ==================================================
    # PCA Visualization Components
    # ==================================================

    pca = PCA(
        n_components=2,
        random_state=42
    )

    pca_components = pca.fit_transform(
        X_scaled
    )

    df["PCA1"] = pca_components[:, 0]
    df["PCA2"] = pca_components[:, 1]

    # ==================================================
    # t-SNE Visualization Components
    # ==================================================

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        max_iter=1000
    )

    tsne_components = tsne.fit_transform(
        X_scaled
    )

    df["TSNE1"] = tsne_components[:, 0]
    df["TSNE2"] = tsne_components[:, 1]

    summary = df.groupby(
        "cluster"
    )[
        ["recency",
         "frequency",
         "monetary"]
    ].mean()

    business_names = assign_business_names(
        summary
    )

    df["segment"] = df["cluster"].map(
        business_names
    )

    df["silhouette"] = silhouette_vals

    confidence = []

    for i, cluster in enumerate(labels):
        confidence.append(
            probs[i][cluster]
        )

    df["confidence"] = confidence

    df.to_csv(
        os.path.join(OUTPUT_DIR, "customer_segments.csv"),
        index=False
    )

    summary.to_csv(
        os.path.join(OUTPUT_DIR, "cluster_summary.csv"),
        index=False
    )

    return {
        "data": df,
        "summary": summary,
        "business_names": business_names,
        "silhouette": round(
            silhouette_score(
                X_scaled,
                labels
            ),
            4
        ),
        "calinski": round(
            calinski_harabasz_score(
                X_scaled,
                labels
            ),
            2
        )
    }