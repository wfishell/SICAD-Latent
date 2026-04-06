import os
import json
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from SICA_PreTrain import SICA_Pretrain


# ── Load preprocessed data ────────────────────────────────────────────────────

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
            raise ValueError(f"{d}: gene list doesn't match {dirs[0]}. Preprocess with the same settings.")
        if peak_names != ref_peaks:
            raise ValueError(f"{d}: peak list doesn't match {dirs[0]}. Preprocess with the same settings.")

        rna         = torch.load(os.path.join(d, "rna_tensor.pt"),   weights_only=True)
        atac        = torch.load(os.path.join(d, "atac_tensor.pt"),  weights_only=True)
        log_lib_rna = torch.load(os.path.join(d, "log_lib_rna.pt"),  weights_only=True).squeeze(-1)
        log_lib_atac= torch.load(os.path.join(d, "log_lib_atac.pt"), weights_only=True).squeeze(-1)

        n_cells = rna.shape[0]
        batch_one_hot = torch.zeros(n_cells, len(dirs))
        batch_one_hot[:, batch_idx] = 1.0

        rna_list.append(rna)
        atac_list.append(atac)
        log_lib_rna_list.append(log_lib_rna)
        log_lib_atac_list.append(log_lib_atac)
        batch_ids_list.append(batch_one_hot)

        print(f"Loaded batch {batch_idx} ({os.path.basename(d)}): {n_cells} cells")

    rna         = torch.cat(rna_list,          dim=0)
    atac        = torch.cat(atac_list,         dim=0)
    log_lib_rna = torch.cat(log_lib_rna_list,  dim=0)
    log_lib_atac= torch.cat(log_lib_atac_list, dim=0)
    batch_ids   = torch.cat(batch_ids_list,    dim=0)

    W_act          = torch.load(os.path.join(dirs[0], "W_act.pt"),  weights_only=True)
    W_rep          = torch.load(os.path.join(dirs[0], "W_rep.pt"),  weights_only=True)
    W_mask         = torch.load(os.path.join(dirs[0], "W_mask.pt"), weights_only=True)
    W_init         = torch.load(os.path.join(dirs[0], "W_init.pt"), weights_only=True)
    tf_idx         = torch.load(os.path.join(dirs[0], "tf_idx.pt"), weights_only=True)
    motif_scores_df= pd.read_parquet(os.path.join(dirs[0], "motif_scores.parquet"))
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

    # Total loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [h["total_loss"] for h in history], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Training Loss")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(plot_dir, "total_loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Individual loss components
    loss_keys = [k for k in history[0] if k not in ("epoch", "total_loss")]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, key in enumerate(loss_keys):
        if i >= len(axes):
            break
        axes[i].plot(epochs, [h[key] for h in history], linewidth=2)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(key)
        axes[i].set_title(key)
        axes[i].grid(True, alpha=0.3)
    for i in range(len(loss_keys), len(axes)):
        axes[i].set_visible(False)
    fig.suptitle("Loss Components", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "loss_components.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved to {plot_dir}/")


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    dirs = args.preprocessed_dir

    (rna, atac, log_lib_rna, log_lib_atac, batch_ids,
     W_act, W_rep, W_mask, W_init, tf_idx, motif_scores) = load_preprocessed(dirs)

    n_cells   = rna.shape[0]
    RNA_Dim   = rna.shape[1]
    ATAC_Dim  = atac.shape[1]
    n_tfs     = W_act.shape[1]
    n_batches = batch_ids.shape[1]

    cell_motif_scores = atac @ motif_scores

    dataset = TensorDataset(rna, atac, log_lib_rna, log_lib_atac,
                            batch_ids, cell_motif_scores)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers)

    model = SICA_Pretrain(
        RNA_Dim    = RNA_Dim,
        ATAC_Dim   = ATAC_Dim,
        Batch_Size = args.batch_size,
        TF         = n_tfs,
        Latent_Dim = args.latent_dim,
        W_mask     = W_mask,
        W_init     = W_init,
        n_batches  = n_batches,
    )

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    tf_idx    = tf_idx.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nTraining on {device}")
    print(f"  {n_cells} cells | {RNA_Dim} genes | {ATAC_Dim} peaks | "
          f"{n_tfs} TFs | {n_batches} batch(es)")
    print(f"  latent_dim={args.latent_dim} | batch_size={args.batch_size} | "
          f"epochs={args.epochs} | lr={args.lr}\n")

    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(args.out_dir, "training", "pretrain")
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
            loss, breakdown = model.compute_loss(outputs, rna_b, atac_b, tf_idx, global_step)

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
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss":        avg_loss,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Save training history as JSON
    history_path = os.path.join(log_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    # Save plots if requested
    if args.save_plots:
        save_plots(history, log_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SICAD model on preprocessed multiome data")
    parser.add_argument("--preprocessed-dir", nargs="+", required=True,
                        help="One or more preprocessed directories (one per dataset/batch)")
    parser.add_argument("--out-dir",    default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--latent-dim", type=int,   default=128)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--epochs",     type=int,   default=120)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--save-every", type=int,   default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--num-workers",type=int,   default=4,
                        help="DataLoader worker processes")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save loss curve plots to training/pretrain/")
    args = parser.parse_args()
    train(args)