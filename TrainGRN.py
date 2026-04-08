import os
import json
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from SICA_PreTrain import SICA_Pretrain
from SICA_GRN import SICA_GRN, compute_structural_prior


# ── Data loading (mirrors Train.py) ──────────────────────────────────────────

def load_preprocessed(dirs):
    rna_list, atac_list = [], []
    log_lib_rna_list, log_lib_atac_list = [], []
    batch_ids_list = []

    ref_genes = pd.read_csv(os.path.join(dirs[0], "gene_names.txt"), header=None)[0].tolist()
    ref_peaks = pd.read_csv(os.path.join(dirs[0], "peak_names.txt"), header=None)[0].tolist()

    for batch_idx, d in enumerate(dirs):
        gene_names = pd.read_csv(os.path.join(d, "gene_names.txt"), header=None)[0].tolist()
        peak_names = pd.read_csv(os.path.join(d, "peak_names.txt"), header=None)[0].tolist()

        if gene_names != ref_genes:
            raise ValueError(f"{d}: gene list doesn't match {dirs[0]}.")
        if peak_names != ref_peaks:
            raise ValueError(f"{d}: peak list doesn't match {dirs[0]}.")

        rna          = torch.load(os.path.join(d, "rna_tensor.pt"),   weights_only=True)
        atac         = torch.load(os.path.join(d, "atac_tensor.pt"),  weights_only=True)
        log_lib_rna  = torch.load(os.path.join(d, "log_lib_rna.pt"),  weights_only=True).squeeze(-1)
        log_lib_atac = torch.load(os.path.join(d, "log_lib_atac.pt"), weights_only=True).squeeze(-1)

        n_cells = rna.shape[0]
        batch_one_hot = torch.zeros(n_cells, len(dirs))
        batch_one_hot[:, batch_idx] = 1.0

        rna_list.append(rna)
        atac_list.append(atac)
        log_lib_rna_list.append(log_lib_rna)
        log_lib_atac_list.append(log_lib_atac)
        batch_ids_list.append(batch_one_hot)

        print(f"Loaded batch {batch_idx} ({os.path.basename(d)}): {n_cells} cells")

    rna          = torch.cat(rna_list,          dim=0)
    atac         = torch.cat(atac_list,         dim=0)
    log_lib_rna  = torch.cat(log_lib_rna_list,  dim=0)
    log_lib_atac = torch.cat(log_lib_atac_list, dim=0)
    batch_ids    = torch.cat(batch_ids_list,    dim=0)

    W_act          = torch.load(os.path.join(dirs[0], "W_act.pt"),  weights_only=True).T  # [P,T] → [T,P]
    W_rep          = torch.load(os.path.join(dirs[0], "W_rep.pt"),  weights_only=True).T  # [P,T] → [T,P]
    W_mask         = torch.load(os.path.join(dirs[0], "W_mask.pt"), weights_only=True)
    W_init         = torch.load(os.path.join(dirs[0], "W_init.pt"), weights_only=True)
    tf_idx         = torch.load(os.path.join(dirs[0], "tf_idx.pt"), weights_only=True)
    motif_scores_df = pd.read_parquet(os.path.join(dirs[0], "motif_scores.parquet"))
    motif_scores   = torch.tensor(motif_scores_df.values, dtype=torch.float32)

    print(f"\nCombined: {rna.shape[0]} cells | {rna.shape[1]} genes | "
          f"{atac.shape[1]} peaks | {len(dirs)} batch(es)")

    return (rna, atac, log_lib_rna, log_lib_atac, batch_ids,
            W_act, W_rep, W_mask, W_init, tf_idx, motif_scores)


# ── Plotting ──────────────────────────────────────────────────────────────────

def save_plots(history, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [h["total_loss"] for h in history], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Phase 2 Total Training Loss")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(plot_dir, "total_loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    loss_keys = [k for k in history[0] if k not in ("epoch", "total_loss")]
    ncols = min(3, len(loss_keys))
    nrows = (len(loss_keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, key in enumerate(loss_keys):
        axes[i].plot(epochs, [h[key] for h in history], linewidth=2)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(key)
        axes[i].set_title(key)
        axes[i].grid(True, alpha=0.3)
    for i in range(len(loss_keys), len(axes)):
        axes[i].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "loss_components.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved to {plot_dir}/")


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    dirs = args.preprocessed_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (rna, atac, log_lib_rna, log_lib_atac, batch_ids,
     W_act_tf_peak, W_rep_tf_peak, W_mask, W_init,
     tf_idx, motif_scores) = load_preprocessed(dirs)

    n_cells   = rna.shape[0]
    RNA_Dim   = rna.shape[1]
    ATAC_Dim  = atac.shape[1]
    n_tfs     = W_act_tf_peak.shape[0]  # [T, P] → T
    n_batches = batch_ids.shape[1]

    cell_motif_scores = atac @ motif_scores

    # ── Step 1: Load Phase 1 checkpoint and extract W_peak-gene ──────────────
    print(f"\nLoading Phase 1 checkpoint: {args.phase1_checkpoint}")
    phase1_ckpt = torch.load(args.phase1_checkpoint, map_location=device, weights_only=True)

    phase1_model = SICA_Pretrain(
        RNA_Dim    = RNA_Dim,
        ATAC_Dim   = ATAC_Dim,
        Batch_Size = args.batch_size,
        TF         = n_tfs,
        Latent_Dim = args.latent_dim,
        W_mask     = W_mask,
        W_init     = W_init,
        n_batches  = n_batches,
    )
    phase1_model.load_state_dict(phase1_ckpt["model_state"])
    phase1_model.eval()

    # ── Step 2: Compute structural prior M_mask ───────────────────────────────
    print("Computing structural prior M_mask from Phase 1 W_peak-gene...")
    W_act_tf_peak = W_act_tf_peak.to(device)
    W_rep_tf_peak = W_rep_tf_peak.to(device)

    with torch.no_grad():
        M_mask, M_act_prior, M_rep_prior = compute_structural_prior(
            W_TF_peak_act       = W_act_tf_peak,
            W_TF_peak_rep       = W_rep_tf_peak,
            W_peak_gene_values  = phase1_model.Module2.W_values,
            mask_indices        = phase1_model.Module2.mask_indices,
            mask_shape          = phase1_model.Module2.mask_shape,
            threshold           = args.mask_threshold,
        )

    density = M_mask.float().mean().item()
    print(f"  M_mask density: {density:.4f} "
          f"({int(density * n_tfs * RNA_Dim):,} / {n_tfs * RNA_Dim:,} TF-gene edges)")

    # ── Step 3: Build Phase 2 model and load Phase 1 weights ─────────────────
    model = SICA_GRN(
        RNA_Dim      = RNA_Dim,
        ATAC_Dim     = ATAC_Dim,
        TF           = n_tfs,
        Latent_Dim   = args.latent_dim,
        W_mask       = W_mask,
        W_init       = W_init,
        n_batches    = n_batches,
        M_mask       = M_mask.cpu(),
        dk           = 64,
        d_proj       = args.d_proj,
        freeze_phase1 = not args.finetune_all,
    )

    # Load Phase 1 weights into Phase 2 model (strict=False: Module4 is new)
    missing, unexpected = model.load_state_dict(phase1_ckpt["model_state"], strict=False)
    expected_missing = {n for n, _ in model.Module4.named_parameters()}
    truly_missing = [k for k in missing if not any(k.startswith(f"Module4.{p}") for p in [""])]
    if truly_missing:
        print(f"  WARNING: Unexpected missing keys: {truly_missing}")
    print(f"  Phase 1 weights loaded. Module 4 parameters initialized fresh.")

    model = model.to(device)
    tf_idx = tf_idx.to(device)

    # Only optimize Module 4 parameters (or all if fine-tuning)
    params = (model.parameters() if args.finetune_all
              else model.Module4.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    dataset = TensorDataset(rna, atac, log_lib_rna, log_lib_atac,
                            batch_ids, cell_motif_scores)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers)

    mode_str = "fine-tune all" if args.finetune_all else "Module 4 only (freeze 1–3)"
    print(f"\nPhase 2 training on {device} — {mode_str}")
    print(f"  {n_cells} cells | {RNA_Dim} genes | {ATAC_Dim} peaks | "
          f"{n_tfs} TFs | {n_batches} batch(es)")
    print(f"  latent_dim={args.latent_dim} | d_proj={args.d_proj} | "
          f"batch_size={args.batch_size} | epochs={args.epochs} | lr={args.lr}\n")

    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(args.out_dir, "training", "module_4_train")
    os.makedirs(log_dir, exist_ok=True)

    history = []
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            rna_b, atac_b, lib_rna_b, lib_atac_b, batch_ids_b, motif_b = [
                x.to(device) for x in batch
            ]

            optimizer.zero_grad()
            outputs = model(rna_b, atac_b, lib_rna_b, lib_atac_b, batch_ids_b, motif_b)
            loss, breakdown = model.compute_loss(
                outputs, rna_b, atac_b, tf_idx, global_step,
                lam_W=args.lam_W, lam_excl=args.lam_excl,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1

        avg_loss = epoch_loss / len(loader)

        record = {"epoch": epoch + 1, "total_loss": avg_loss}
        record.update(breakdown)
        history.append(record)

        print(f"Epoch {epoch+1:>4}/{args.epochs} | loss: {avg_loss:.4f} | {breakdown}")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"grn_checkpoint_epoch{epoch+1}.pt")
            torch.save({
                "epoch":           epoch + 1,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss":            avg_loss,
                "M_mask":          M_mask.cpu(),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Save final GRN outputs
    model.eval()
    with torch.no_grad():
        E_TF = model.Module3.E_TF
        W_act_final, W_rep_final = model.Module4.get_regulatory_weights(E_TF)
        W_reg_final = W_act_final - W_rep_final

    torch.save(W_act_final.cpu(), os.path.join(args.out_dir, "GRN_W_act.pt"))
    torch.save(W_rep_final.cpu(), os.path.join(args.out_dir, "GRN_W_rep.pt"))
    torch.save(W_reg_final.cpu(), os.path.join(args.out_dir, "GRN_W_reg.pt"))
    torch.save(M_mask.cpu(),      os.path.join(args.out_dir, "GRN_M_mask.pt"))
    print(f"\nGRN outputs saved to {args.out_dir}/")

    # Save training history
    history_path = os.path.join(log_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    if args.save_plots:
        save_plots(history, log_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Train GRN inference module (Module 4) on Phase 1 checkpoint"
    )
    parser.add_argument("--preprocessed-dir", nargs="+", required=True,
                        help="Same preprocessed directories used for Phase 1 training")
    parser.add_argument("--phase1-checkpoint", required=True,
                        help="Path to Phase 1 checkpoint (.pt file)")
    parser.add_argument("--out-dir",       default="checkpoints",
                        help="Directory to save GRN checkpoints and outputs")
    parser.add_argument("--latent-dim",    type=int,   default=128)
    parser.add_argument("--d-proj",        type=int,   default=64,
                        help="Projection dimension for bilinear regulatory weights")
    parser.add_argument("--batch-size",    type=int,   default=512)
    parser.add_argument("--epochs",        type=int,   default=80)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--save-every",    type=int,   default=10)
    parser.add_argument("--num-workers",   type=int,   default=4)
    parser.add_argument("--mask-threshold",type=float, default=0.01,
                        help="Binarization threshold for structural prior M_mask")
    parser.add_argument("--lam-W",         type=float, default=0.001,
                        help="L1 sparsity weight on W_act and W_rep")
    parser.add_argument("--lam-excl",      type=float, default=0.01,
                        help="Mutual exclusion penalty weight (W_act ⊙ W_rep)")
    parser.add_argument("--finetune-all",  action="store_true",
                        help="Fine-tune all modules jointly instead of freezing Modules 1–3")
    parser.add_argument("--save-plots",    action="store_true",
                        help="Save loss curve plots to training/grn/")
    args = parser.parse_args()
    train(args)

