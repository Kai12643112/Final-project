#!/usr/bin/env python3
"""
main.py

This script performs K-Means clustering on the provided public (4D) and private (6D) datasets
and outputs submission files in the required format for the Final Project - Big Data.

Usage:
    python main.py --public public_data.csv --private private_data.csv --id r119020XX

Outputs:
    r119020XX_public.csv  # for public dataset evaluation
    r119020XX_private.csv # for private dataset evaluation
"""
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cluster_and_save(input_file: str, output_file: str, n_clusters: int) -> None:
    """
    Load the dataset, perform standardization, run K-Means, and save the labels.
    """
    df = pd.read_csv(input_file)
    # Preserve original sample order via 'id' column or default index
    if 'id' in df.columns:
        ids = df['id']
        X = df.drop(columns=['id'])
    else:
        ids = pd.Series(range(len(df)), name='id')
        X = df

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)

    # Save submission CSV
    submission = pd.DataFrame({'id': ids, 'label': labels})
    submission.to_csv(output_file, index=False)
    print(f"Saved {output_file} with {n_clusters} clusters.")


def main():
    parser = argparse.ArgumentParser(
        description='Cluster public and private Big Data datasets into 4n-1 clusters'
    )
    parser.add_argument(
        '--public', default='public_data.csv',
        help='Path to the public dataset CSV (4 dimensions)'
    )
    parser.add_argument(
        '--private', default='private_data.csv',
        help='Path to the private dataset CSV (6 dimensions)'
    )
    parser.add_argument(
        '--id', required=True,
        help='Your student ID (e.g., r119020XX) to prefix output files'
    )
    args = parser.parse_args()

    public_out = f"{args.id}_public.csv"
    private_out = f"{args.id}_private.csv"

    # Number of clusters: 4n - 1
    cluster_and_save(args.public, public_out, n_clusters=4*4 - 1)
    cluster_and_save(args.private, private_out, n_clusters=4*6 - 1)

if __name__ == '__main__':
    main()
