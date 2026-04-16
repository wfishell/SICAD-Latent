"""
Run the Phase 1 (SICA_Pretrain) encoder on any checkpoint to produce the
posterior mean latent representation Z_mean for all cells.

Usage:
    python InferencePretrain.py \
        --checkpoint checkpoints/checkpoint_epoch120.pt \
        --preprocessed-dir data/preprocessed/pbmc10k \
        --out inference_outputs/z_mean_epoch120.pt
"""

import os
import argparse
import torch

from Encoder import EncoderArchitecture


def load_data(data_dir):
    rna          = torch.load(os.path.join(data_dir, "rna_tensor.pt"),   weights_only=True)
    atac         = torch.load(os.path.join(data_dir, "atac_tensor.pt"),  weights_only=True)
    log_lib_rna  = torch.load(os.path.join(data_dir, "log_lib_rna.pt"),  weights_only=True).squeeze(-1)
    log_lib_atac = torch.load(os.path.join(data_dir, "log_lib_atac.pt"), weights_only=True).squeeze(-1)
    return rna, atac, log_lib_rna, log_lib_atac


def infer_encoder_dims(state):
    RNA_Dim    = state["Encoder.RNA_encoder.0.weight"].shape[1] - 1
    ATAC_Dim   = state["Encoder.ATAC_encoder.0.weight"].shape[1] - 1
    Latent_Dim = state["Encoder.RNA_mu.weight"].shape[0]
    return RNA_Dim, ATAC_Dim, Latent_Dim


def build_encoder(state, RNA_Dim, ATAC_Dim, Latent_Dim):
    encoder = EncoderArchitecture(RNA_Dim, ATAC_Dim, Latent_Dim)
    enc_state = {
        k[len("Encoder."):]: v
        for k, v in state.items() if k.startswith("Encoder.")
    }
    missing, unexpected = encoder.load_state_dict(enc_state, strict=True)
    if missing or unexpected:
        print(f"  missing={missing}  unexpected={unexpected}")
    return encoder


@torch.no_grad()
def run(encoder, rna, atac, log_lib_rna, log_lib_atac, batch_size, device):
    encoder.eval()
    n = rna.shape[0]
    out = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        _, Z_mean, _ = encoder(
            rna[s:e].to(device),
            atac[s:e].to(device),
            log_lib_rna[s:e].to(device),
            log_lib_atac[s:e].to(device),
        )
        out.append(Z_mean.cpu())
        if (s // batch_size) % 10 == 0:
            print(f"  {e}/{n} cells")
    return torch.cat(out, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint_epochN.pt")
    p.add_argument("--preprocessed-dir", required=True)
    p.add_argument("--out", default=None,
                   help="Output tensor path (default: inference_outputs/z_mean_<checkpoint>.pt)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["model_state"]
    print(f"  epoch={ckpt.get('epoch')}  loss={ckpt.get('loss'):.4f}")

    RNA_Dim, ATAC_Dim, Latent_Dim = infer_encoder_dims(state)
    print(f"  RNA_Dim={RNA_Dim}  ATAC_Dim={ATAC_Dim}  Latent_Dim={Latent_Dim}")

    encoder = build_encoder(state, RNA_Dim, ATAC_Dim, Latent_Dim).to(device)

    print(f"Loading data from {args.preprocessed_dir}")
    rna, atac, log_lib_rna, log_lib_atac = load_data(args.preprocessed_dir)
    print(f"  {rna.shape[0]} cells")

    print("Running encoder...")
    z_mean = run(encoder, rna, atac, log_lib_rna, log_lib_atac, args.batch_size, device)
    print(f"  z_mean shape: {tuple(z_mean.shape)}  "
          f"mean={z_mean.mean():.4f}  std={z_mean.std():.4f}")

    if args.out is None:
        ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out_path = os.path.join("inference_outputs", f"z_mean_{ckpt_stem}.pt")
    else:
        out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(z_mean, out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
