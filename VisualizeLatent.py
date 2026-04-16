"""
Visualize latent cell embeddings with PCA, UMAP, and t-SNE, colored by
Leiden clusters computed on a k-NN graph in the latent space.

Usage:
    python VisualizeLatent.py \
        --z inference_outputs/z_mean_checkpoint_epoch120.pt \
        --out-dir inference_outputs/viz_epoch120 \
        --resolution 0.5
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_embedding(coords, labels, title, path, legend_ncol=2):
    fig, ax = plt.subplots(figsize=(9, 6))
    uniq = sorted(np.unique(labels).tolist())
    cmap = plt.get_cmap("tab20", max(len(uniq), 20))
    for i, c in enumerate(uniq):
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1], s=4, color=cmap(i),
                   label=str(c), alpha=0.7, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel(f"{title}-1")
    ax.set_ylabel(f"{title}-2")
    ax.legend(markerscale=2, fontsize=7, loc="center left",
              bbox_to_anchor=(1.0, 0.5), frameon=False, ncol=legend_ncol)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--z", required=True, help="Path to latent tensor .pt file")
    p.add_argument("--out-dir", default="inference_outputs/viz")
    p.add_argument("--resolution", type=float, default=0.5,
                   help="Leiden resolution (higher = more clusters)")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--n-pcs", type=int, default=50,
                   help="PCs used for neighbor graph (capped at latent_dim)")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--labels", default=None,
                   help="Optional cell_labels.csv from AnnotateCells.py "
                        "(adds a second set of plots colored by cell type)")
    p.add_argument("--label-col", default="majority_voting",
                   help="Column in labels CSV to use (default: majority_voting)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading latent: {args.z}")
    z = torch.load(args.z, map_location="cpu", weights_only=True).numpy().astype(np.float32)
    print(f"  shape={z.shape}  mean={z.mean():.4f}  std={z.std():.4f}")

    adata = ad.AnnData(X=z)
    n_pcs = min(args.n_pcs, z.shape[1] - 1)

    print("Running PCA...")
    sc.pp.pca(adata, n_comps=n_pcs, random_state=args.seed)

    print(f"Building neighbor graph  (k={args.n_neighbors}, n_pcs={n_pcs})...")
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=n_pcs,
                    random_state=args.seed)

    print(f"Leiden clustering  (resolution={args.resolution})...")
    sc.tl.leiden(adata, resolution=args.resolution, random_state=args.seed,
                 flavor="igraph", n_iterations=2, directed=False)
    clusters = adata.obs["leiden"].astype(int).to_numpy()
    n_clusters = clusters.max() + 1
    print(f"  {n_clusters} clusters  |  sizes: "
          f"{np.bincount(clusters).tolist()}")

    print("Running UMAP...")
    sc.tl.umap(adata, random_state=args.seed)
    umap_xy = adata.obsm["X_umap"]

    print("Running t-SNE...")
    sc.tl.tsne(adata, n_pcs=n_pcs, perplexity=args.perplexity,
               random_state=args.seed)
    tsne_xy = adata.obsm["X_tsne"]

    pca_xy = adata.obsm["X_pca"][:, :2]

    # Save coords + labels
    np.savez(
        os.path.join(args.out_dir, "embeddings.npz"),
        pca=adata.obsm["X_pca"], umap=umap_xy, tsne=tsne_xy,
        leiden=clusters,
    )
    print(f"  embeddings.npz saved")

    print("Plotting (colored by Leiden cluster)...")
    plot_embedding(pca_xy,  clusters, "PCA",
                   os.path.join(args.out_dir, "pca_leiden.png"))
    plot_embedding(umap_xy, clusters, "UMAP",
                   os.path.join(args.out_dir, "umap_leiden.png"))
    plot_embedding(tsne_xy, clusters, "tSNE",
                   os.path.join(args.out_dir, "tsne_leiden.png"))

    if args.labels:
        print(f"Loading labels: {args.labels}")
        labels_df = pd.read_csv(args.labels)
        if len(labels_df) != z.shape[0]:
            raise ValueError(
                f"labels rows ({len(labels_df)}) != latent rows ({z.shape[0]})"
            )
        if args.label_col not in labels_df.columns:
            raise ValueError(
                f"column '{args.label_col}' not in {list(labels_df.columns)}"
            )
        cell_types = labels_df[args.label_col].astype(str).to_numpy()
        print(f"  {len(np.unique(cell_types))} cell types")

        print("Plotting (colored by cell type)...")
        plot_embedding(pca_xy,  cell_types, "PCA",
                       os.path.join(args.out_dir, "pca_celltype.png"))
        plot_embedding(umap_xy, cell_types, "UMAP",
                       os.path.join(args.out_dir, "umap_celltype.png"))
        plot_embedding(tsne_xy, cell_types, "tSNE",
                       os.path.join(args.out_dir, "tsne_celltype.png"))

    print("Done.")


if __name__ == "__main__":
    main()
