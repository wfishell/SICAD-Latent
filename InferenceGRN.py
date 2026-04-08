"""
Inference script for SICA_GRN (Phase 2 GRN model).

Loads a grn_checkpoint and runs inference to produce:
  - Latent cell embeddings (z)
  - Per-cell TF activity scores (alpha_TF)
  - GRN-mediated RNA reconstruction (X_hat_GRN)
  - Global regulatory weight matrices (W_act, W_rep, W_reg)
  - Top TF→gene regulatory edges

Usage:
    python InferenceGRN.py \
        --checkpoint checkpoints/grn_checkpoint_epoch80.pt \
        --preprocessed-dir data/preprocessed/pbmc10k \
        [--n-cells 500] \
        [--batch-size 256] \
        [--out-dir inference_outputs] \
        [--device cpu]
"""

import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from SICA_GRN import SICA_GRN


# ── Dimension inference from checkpoint ──────────────────────────────────────

def infer_dims(model_state):
    """Extract model hyperparameters directly from checkpoint weight shapes."""
    # RNA_Dim: Encoder.RNA_encoder.0.weight is [hidden, RNA_Dim + 1]
    rna_enc_w = model_state["Encoder.RNA_encoder.0.weight"]
    RNA_Dim = rna_enc_w.shape[1] - 1

    # ATAC_Dim: Encoder.ATAC_encoder.0.weight is [hidden, ATAC_Dim + 1]
    atac_enc_w = model_state["Encoder.ATAC_encoder.0.weight"]
    ATAC_Dim = atac_enc_w.shape[1] - 1

    # Latent_Dim: Encoder.RNA_mu.weight is [Latent_Dim, hidden]
    Latent_Dim = model_state["Encoder.RNA_mu.weight"].shape[0]

    # TF: Module3.E_TF is [TF, Latent_Dim]
    TF = model_state["Module3.E_TF"].shape[0]

    # n_batches: Module4.B_RNA_GRN is [n_batches, n_genes]
    n_batches = model_state["Module4.B_RNA_GRN"].shape[0]

    # d_proj: Module4.W_act_proj is [Latent_Dim, d_proj]
    d_proj = model_state["Module4.W_act_proj"].shape[1]

    return RNA_Dim, ATAC_Dim, Latent_Dim, TF, n_batches, d_proj


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir):
    rna          = torch.load(os.path.join(data_dir, "rna_tensor.pt"),   weights_only=True)
    atac         = torch.load(os.path.join(data_dir, "atac_tensor.pt"),  weights_only=True)
    log_lib_rna  = torch.load(os.path.join(data_dir, "log_lib_rna.pt"),  weights_only=True).squeeze(-1)
    log_lib_atac = torch.load(os.path.join(data_dir, "log_lib_atac.pt"), weights_only=True).squeeze(-1)
    W_mask       = torch.load(os.path.join(data_dir, "W_mask.pt"),       weights_only=True)
    W_init       = torch.load(os.path.join(data_dir, "W_init.pt"),       weights_only=True)

    n_cells = rna.shape[0]
    # Single-batch one-hot (extend if multi-batch needed)
    batch_ids = torch.zeros(n_cells, 1)
    batch_ids[:, 0] = 1.0

    # Load motif scores and compute per-cell scores: [N, P] @ [P, T] → [N, T]
    motif_path = os.path.join(data_dir, "motif_scores.parquet")
    motif_df = pd.read_parquet(motif_path)
    motif_scores = torch.tensor(motif_df.values, dtype=torch.float32)  # [P, T]
    cell_motif_scores = atac @ motif_scores                             # [N, T]

    # Gene / peak names (optional, used for labelling outputs)
    gene_names, peak_names, tf_names = [], [], []
    gene_path = os.path.join(data_dir, "gene_names.txt")
    peak_path = os.path.join(data_dir, "peak_names.txt")
    if os.path.exists(gene_path):
        gene_names = pd.read_csv(gene_path, header=None)[0].tolist()
    if os.path.exists(peak_path):
        peak_names = pd.read_csv(peak_path, header=None)[0].tolist()
    tf_names = list(motif_df.columns)

    return (rna, atac, log_lib_rna, log_lib_atac, batch_ids,
            cell_motif_scores, W_mask, W_init,
            gene_names, tf_names)


# ── Model construction ────────────────────────────────────────────────────────

def build_model(model_state, M_mask, W_mask, W_init,
                RNA_Dim, ATAC_Dim, Latent_Dim, TF, n_batches, d_proj):
    model = SICA_GRN(
        RNA_Dim      = RNA_Dim,
        ATAC_Dim     = ATAC_Dim,
        TF           = TF,
        Latent_Dim   = Latent_Dim,
        W_mask       = W_mask,
        W_init       = W_init,
        n_batches    = n_batches,
        M_mask       = M_mask,
        dk           = 64,
        d_proj       = d_proj,
        freeze_phase1 = True,
    )
    missing, unexpected = model.load_state_dict(model_state, strict=True)
    if missing:
        print(f"  WARNING: missing keys: {missing}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected}")
    return model


# ── Batched inference ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, rna, atac, log_lib_rna, log_lib_atac,
                  batch_ids, cell_motif_scores, batch_size, device):
    model.eval()
    n_cells = rna.shape[0]

    all_z, all_mu, all_alpha_TF, all_X_hat_GRN = [], [], [], []

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        rna_b          = rna[start:end].to(device)
        atac_b         = atac[start:end].to(device)
        lib_rna_b      = log_lib_rna[start:end].to(device)
        lib_atac_b     = log_lib_atac[start:end].to(device)
        batch_ids_b    = batch_ids[start:end].to(device)
        motif_b        = cell_motif_scores[start:end].to(device)

        outputs = model(rna_b, atac_b, lib_rna_b, lib_atac_b, batch_ids_b, motif_b)
        z, mu, _logvar, _mu_atac, _x_rna, _theta, alpha_TF, X_hat_GRN, *_ = outputs

        all_z.append(mu.cpu())           # use posterior mean (deterministic)
        all_mu.append(mu.cpu())
        all_alpha_TF.append(alpha_TF.cpu())
        all_X_hat_GRN.append(X_hat_GRN.cpu())

        if (start // batch_size) % 5 == 0:
            print(f"  Processed {end}/{n_cells} cells...")

    z         = torch.cat(all_z,         dim=0)
    alpha_TF  = torch.cat(all_alpha_TF,  dim=0)
    X_hat_GRN = torch.cat(all_X_hat_GRN, dim=0)

    # Global regulatory weights (cell-state-independent, derived from embeddings)
    E_TF = model.Module3.E_TF
    W_act, W_rep = model.Module4.get_regulatory_weights(E_TF)
    W_reg = W_act - W_rep

    return z, alpha_TF, X_hat_GRN, W_act.cpu(), W_rep.cpu(), W_reg.cpu()


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(z, alpha_TF, X_hat_GRN, W_act, W_rep, W_reg, M_mask,
           gene_names, tf_names, top_k=20):
    n_cells, n_genes = X_hat_GRN.shape
    n_tfs = alpha_TF.shape[1]

    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    print(f"  Cells:  {n_cells}")
    print(f"  Genes:  {n_genes}")
    print(f"  TFs:    {n_tfs}")

    # ── Latent embeddings ────────────────────────────────────────
    print(f"\n[Latent embeddings z]  shape: {z.shape}")
    print(f"  mean={z.mean():.4f}  std={z.std():.4f}  "
          f"min={z.min():.4f}  max={z.max():.4f}")

    # ── TF activity ──────────────────────────────────────────────
    print(f"\n[TF activity alpha_TF]  shape: {alpha_TF.shape}")
    mean_activity = alpha_TF.mean(0)  # [T]
    top_tf_idx = mean_activity.argsort(descending=True)[:top_k]
    print(f"  Top {top_k} most active TFs (by mean alpha):")
    for rank, i in enumerate(top_tf_idx.tolist()):
        name = tf_names[i] if i < len(tf_names) else f"TF_{i}"
        print(f"    {rank+1:>3}. {name:<30}  mean_alpha={mean_activity[i]:.4f}")

    # ── GRN reconstruction ───────────────────────────────────────
    print(f"\n[GRN RNA reconstruction X_hat_GRN]  shape: {X_hat_GRN.shape}")
    print(f"  mean={X_hat_GRN.mean():.4f}  std={X_hat_GRN.std():.4f}  "
          f"min={X_hat_GRN.min():.4f}  max={X_hat_GRN.max():.4f}")

    # ── Regulatory weights ───────────────────────────────────────
    density = M_mask.float().mean().item()
    n_edges = int(density * n_tfs * n_genes)
    print(f"\n[Regulatory weights W_reg]  shape: {W_reg.shape}")
    print(f"  M_mask density: {density:.4f}  ({n_edges:,} / {n_tfs * n_genes:,} TF-gene edges)")
    print(f"  W_act  — mean={W_act.mean():.4f}  max={W_act.max():.4f}")
    print(f"  W_rep  — mean={W_rep.mean():.4f}  max={W_rep.max():.4f}")
    print(f"  W_reg  — mean={W_reg.mean():.4f}  "
          f"pos={( W_reg > 0).float().mean():.3f}  "
          f"neg={(W_reg < 0).float().mean():.3f}")

    # ── Top regulatory edges ─────────────────────────────────────
    print(f"\n[Top {top_k} activating TF→gene edges  (by W_act)]")
    flat_act = W_act.flatten()
    top_act  = flat_act.argsort(descending=True)[:top_k]
    for rank, idx in enumerate(top_act.tolist()):
        t = idx // n_genes
        g = idx  % n_genes
        tf_name   = tf_names[t]   if t < len(tf_names)   else f"TF_{t}"
        gene_name = gene_names[g] if g < len(gene_names) else f"gene_{g}"
        print(f"    {rank+1:>3}. {tf_name:<30} → {gene_name:<20}  W_act={W_act[t,g]:.4f}")

    print(f"\n[Top {top_k} repressive TF→gene edges  (by W_rep)]")
    flat_rep = W_rep.flatten()
    top_rep  = flat_rep.argsort(descending=True)[:top_k]
    for rank, idx in enumerate(top_rep.tolist()):
        t = idx // n_genes
        g = idx  % n_genes
        tf_name   = tf_names[t]   if t < len(tf_names)   else f"TF_{t}"
        gene_name = gene_names[g] if g < len(gene_names) else f"gene_{g}"
        print(f"    {rank+1:>3}. {tf_name:<30} → {gene_name:<20}  W_rep={W_rep[t,g]:.4f}")

    print("=" * 60)


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_outputs(out_dir, z, alpha_TF, X_hat_GRN, W_act, W_rep, W_reg,
                 gene_names, tf_names):
    os.makedirs(out_dir, exist_ok=True)

    torch.save(z,         os.path.join(out_dir, "z_latent.pt"))
    torch.save(alpha_TF,  os.path.join(out_dir, "alpha_TF.pt"))
    torch.save(X_hat_GRN, os.path.join(out_dir, "X_hat_GRN.pt"))
    torch.save(W_act,     os.path.join(out_dir, "W_act.pt"))
    torch.save(W_rep,     os.path.join(out_dir, "W_rep.pt"))
    torch.save(W_reg,     os.path.join(out_dir, "W_reg.pt"))

    # Save W_reg as a labelled CSV if names are available
    if gene_names and tf_names:
        n_tfs, n_genes = W_reg.shape
        tf_labels   = tf_names[:n_tfs]
        gene_labels = gene_names[:n_genes]
        df = pd.DataFrame(W_reg.numpy(), index=tf_labels, columns=gene_labels)
        df.to_csv(os.path.join(out_dir, "W_reg_labelled.csv"))
        print(f"  W_reg CSV saved → {out_dir}/W_reg_labelled.csv")

    print(f"  Tensors saved → {out_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GRN inference: load checkpoint and extract regulatory outputs"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to grn_checkpoint_epochN.pt")
    parser.add_argument("--preprocessed-dir", required=True,
                        help="Preprocessed data directory (same as used for training)")
    parser.add_argument("--n-cells", type=int, default=None,
                        help="Number of cells to run (default: all)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size for inference (default: 256)")
    parser.add_argument("--out-dir", default="inference_outputs",
                        help="Directory to save output tensors (default: inference_outputs/)")
    parser.add_argument("--device", default=None,
                        help="Device: 'cpu' or 'cuda' (auto-detected if omitted)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top edges/TFs to display (default: 20)")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving outputs to disk")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_state = ckpt["model_state"]
    M_mask      = ckpt["M_mask"]
    print(f"  Epoch: {ckpt['epoch']}  |  Loss: {ckpt['loss']:.4f}")
    print(f"  M_mask: {M_mask.shape}  density={M_mask.float().mean():.4f}")

    # ── Infer dimensions ──────────────────────────────────────────
    RNA_Dim, ATAC_Dim, Latent_Dim, TF, n_batches, d_proj = infer_dims(model_state)
    print(f"\nDimensions inferred from checkpoint:")
    print(f"  RNA_Dim={RNA_Dim}  ATAC_Dim={ATAC_Dim}  Latent_Dim={Latent_Dim}")
    print(f"  TF={TF}  n_batches={n_batches}  d_proj={d_proj}")

    # ── Load data ─────────────────────────────────────────────────
    print(f"\nLoading data from: {args.preprocessed_dir}")
    (rna, atac, log_lib_rna, log_lib_atac, batch_ids,
     cell_motif_scores, W_mask, W_init,
     gene_names, tf_names) = load_data(args.preprocessed_dir)

    n_cells = rna.shape[0]
    if args.n_cells is not None and args.n_cells < n_cells:
        print(f"  Subsetting to {args.n_cells} / {n_cells} cells")
        idx = torch.randperm(n_cells)[:args.n_cells]
        rna              = rna[idx]
        atac             = atac[idx]
        log_lib_rna      = log_lib_rna[idx]
        log_lib_atac     = log_lib_atac[idx]
        batch_ids        = batch_ids[idx]
        cell_motif_scores = cell_motif_scores[idx]
        n_cells = args.n_cells

    print(f"  {n_cells} cells  |  {len(gene_names)} gene names  |  {len(tf_names)} TF names")

    # ── Build model ───────────────────────────────────────────────
    print("\nBuilding SICA_GRN model...")
    model = build_model(model_state, M_mask, W_mask, W_init,
                        RNA_Dim, ATAC_Dim, Latent_Dim, TF, n_batches, d_proj)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Run inference ─────────────────────────────────────────────
    print(f"\nRunning inference on {n_cells} cells (batch_size={args.batch_size})...")
    z, alpha_TF, X_hat_GRN, W_act, W_rep, W_reg = run_inference(
        model, rna, atac, log_lib_rna, log_lib_atac,
        batch_ids, cell_motif_scores, args.batch_size, device
    )

    # ── Report ────────────────────────────────────────────────────
    report(z, alpha_TF, X_hat_GRN, W_act, W_rep, W_reg, M_mask,
           gene_names, tf_names, top_k=args.top_k)

    # ── Save ──────────────────────────────────────────────────────
    if not args.no_save:
        print(f"\nSaving outputs to {args.out_dir}/")
        save_outputs(args.out_dir, z, alpha_TF, X_hat_GRN, W_act, W_rep, W_reg,
                     gene_names, tf_names)


if __name__ == "__main__":
    main()
