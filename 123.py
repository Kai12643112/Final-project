#!/usr/bin/env python3
"""
main.py

This script performs clustering on the provided
public (4D) and private (6D) datasets and outputs submission files.
It automatically tries multiple algorithms (KMeans, GMM, Spectral) and selects
the best by Silhouette Score for higher quality.

Usage:
    python main.py \
      [--public public_data.csv] \
      [--private private_data.csv] \
      [--id ID] \
      [--n-init N] [--max-iter M] [--evaluate]

Outputs:
    {id}_public.csv    # placed in same folder as public input
    {id}_private.csv   # placed in same folder as private input
"""
import argparse
import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

# Default student ID if none provided
default_id = 'b12901163'
# Supported algorithms
algorithms = ['kmeans', 'gmm', 'spectral']

def check_file_exists(path: str):
    if not os.path.isfile(path):
        print(f"Error: File not found: '{path}'")
        print(f"Current working directory: {os.getcwd()}")
        print("Files in this directory:", os.listdir(os.getcwd()))
        sys.exit(1)


def fit_and_score(X, n_clusters, algo, n_init, max_iter):
    if algo == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42,
                       n_init=n_init, max_iter=max_iter, algorithm='elkan')
        labels = model.fit_predict(X)
    elif algo == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42,
                                 n_init=n_init, covariance_type='full')
        labels = model.fit_predict(X)
    elif algo == 'spectral':
        model = SpectralClustering(n_clusters=n_clusters, random_state=42,
                                   n_init=n_init, assign_labels='kmeans')
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    score = silhouette_score(X, labels)
    return labels, score


def cluster_and_save(input_path, student_id, n_clusters, n_init, max_iter, evaluate):
    check_file_exists(input_path)
    df = pd.read_csv(input_path)
    # ID column handling
    if 'id' in df.columns:
        ids = df['id']
        features = df.drop(columns=['id'])
    else:
        ids = pd.Series(range(1, len(df)+1), name='id')
        features = df
    # Standardize
    X = StandardScaler().fit_transform(features)

    # Try all algorithms
    best_score, best_algo, best_labels = -1.0, None, None
    for algo in algorithms:
        labels, score = fit_and_score(X, n_clusters, algo, n_init, max_iter)
        print(f"Algorithm {algo} Silhouette: {score:.4f}")
        if score > best_score:
            best_score, best_algo, best_labels = score, algo, labels
    print(f"Selected {best_algo} (Silhouette {best_score:.4f})")

    # Determine output path in same folder as input
    folder = os.path.dirname(os.path.abspath(input_path))
    output_file = f"{student_id}_{os.path.basename(input_path).split('_')[0]}.csv"
    # e.g., public_data.csv -> {id}_public.csv
    # or private_data.csv -> {id}_private.csv
    output_path = os.path.join(folder, output_file)

    # Save
    pd.DataFrame({'id': ids, 'label': best_labels}).to_csv(output_path, index=False)
    print(f"Saved {output_path} using {best_algo}")
    if evaluate:
        print(f"Final Silhouette Score: {best_score:.4f}")


def main():
    p = argparse.ArgumentParser(description='Auto-cluster Big Data datasets')
    p.add_argument('--public', default='public_data.csv', help='Public CSV path')
    p.add_argument('--private', default='private_data.csv', help='Private CSV path')
    p.add_argument('-i', '--id', dest='id', help='Student ID for filenames')
    p.add_argument('--n-init', type=int, default=20, help='n_init')
    p.add_argument('--max-iter', type=int, default=500, help='max_iter')
    p.add_argument('--evaluate', action='store_true', help='print silhouette')
    args = p.parse_args()

    student_id = args.id if args.id else default_id

    # clusters
    pub_c = 4*4 - 1  # 15
    priv_c = 4*6 - 1 # 23

    print("Processing public dataset...")
    cluster_and_save(args.public, student_id, pub_c, args.n_init, args.max_iter, args.evaluate)
    print("Processing private dataset...")
    cluster_and_save(args.private, student_id, priv_c, args.n_init, args.max_iter, args.evaluate)

if __name__ == '__main__':
    main()
