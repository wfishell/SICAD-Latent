"""
Annotate cells with CellTypist (pre-trained immune atlas model).

Loads the raw RNA counts from a preprocessed directory, builds an AnnData,
log1p-CPM-normalizes it, and runs CellTypist to assign per-cell labels.

Outputs:
    <out_dir>/cell_labels.csv — columns: cell_idx, predicted, majority_voting, conf_score

Usage:
    python AnnotateCells.py \
        --preprocessed-dir data/preprocessed/pbmc10k \
        --model Immune_All_Low.pkl \
        --out-dir inference_outputs/annotation
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import anndata as ad
import scanpy as sc
import celltypist
from celltypist import models


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preprocessed-dir", required=True)
    p.add_argument("--model", default="Immune_All_Low.pkl",
                   help="CellTypist model (e.g. Immune_All_Low.pkl, Immune_All_High.pkl)")
    p.add_argument("--out-dir", default="inference_outputs/annotation")
    p.add_argument("--majority-voting", action="store_true", default=True,
                   help="Refine labels with over-clustering + majority voting")
    p.add_argument("--no-majority-voting", action="store_false", dest="majority_voting")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading RNA tensor and gene names from {args.preprocessed_dir}")
    rna = torch.load(os.path.join(args.preprocessed_dir, "rna_tensor.pt"),
                     weights_only=True).numpy()
    genes = pd.read_csv(os.path.join(args.preprocessed_dir, "gene_names.txt"),
                        header=None)[0].tolist()
    print(f"  {rna.shape[0]} cells  |  {rna.shape[1]} genes")

    adata = ad.AnnData(X=rna)
    adata.var_names = genes
    adata.obs_names = [f"cell_{i}" for i in range(rna.shape[0])]

    print("Normalizing (CPM + log1p)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print(f"Downloading / loading CellTypist model: {args.model}")
    models.download_models(model=[args.model])
    model = models.Model.load(model=args.model)
    print(f"  model trained on {len(model.cell_types)} cell types")

    print(f"Running CellTypist  (majority_voting={args.majority_voting})...")
    pred = celltypist.annotate(adata, model=model,
                               majority_voting=args.majority_voting)
    result = pred.predicted_labels.copy()
    result.insert(0, "cell_idx", np.arange(len(result)))
    if "conf_score" in pred.probability_matrix.columns:
        pass
    # CellTypist stores per-class probs in probability_matrix; grab top prob as confidence
    prob = pred.probability_matrix
    top_prob = prob.max(axis=1).values
    result["conf_score"] = top_prob

    out_path = os.path.join(args.out_dir, "cell_labels.csv")
    result.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

    # Summary
    label_col = "majority_voting" if args.majority_voting else "predicted_labels"
    if label_col not in result.columns:
        label_col = result.columns[1]
    counts = result[label_col].value_counts()
    print(f"\nCell type distribution ({label_col}):")
    for name, n in counts.items():
        print(f"  {n:>5d}  {name}")


if __name__ == "__main__":
    main()
