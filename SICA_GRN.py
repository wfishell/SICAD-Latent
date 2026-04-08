import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import EncoderArchitecture
from Module1 import ATAC_Seq_Reconstruction
from Module2 import PeakGeneModule
from Module3 import TF_Sparse_Cosine_Attention
from Module4 import GRNModule


def compute_structural_prior(W_TF_peak_act, W_TF_peak_rep, W_peak_gene_values,
                              mask_indices, mask_shape, threshold=0.01):
    """
    Compose Phase 1 learned peak-gene links with fixed TF-peak matrices to
    produce the structural prior M_mask for Module 4.

    M_act[t, g] = Σ_p W_TF-peak^act[t, p] × W_peak-gene[g, p]
    M_rep[t, g] = Σ_p W_TF-peak^rep[t, p] × W_peak-gene[g, p]

    Args:
        W_TF_peak_act:      [T, P] activating TF-peak scores (fixed, from preprocessing)
        W_TF_peak_rep:      [T, P] repressive TF-peak scores (fixed, from preprocessing)
        W_peak_gene_values: 1D tensor of non-zero values in W_peak-gene (from Module2)
        mask_indices:       [2, nnz] indices of non-zero entries in W_peak-gene [G × P]
        mask_shape:         Shape of W_peak-gene [G, P]
        threshold:          Binarization threshold for M_mask (default: 0.01)

    Returns:
        M_mask: Boolean tensor [T, G] — structural prior for regulatory weight learning
        M_act:  Float tensor [T, G] — normalized activation prior
        M_rep:  Float tensor [T, G] — normalized repression prior
    """
    device = W_TF_peak_act.device
    n_genes, n_peaks = mask_shape.tolist()

    # Reconstruct dense W_peak-gene [G, P] from sparse representation
    W_pg_values = torch.clamp(W_peak_gene_values, 0, 1)
    W_pg_sparse = torch.sparse_coo_tensor(
        mask_indices, W_pg_values,
        size=(n_genes, n_peaks), device=device
    ).to_dense()  # [G, P]

    # M_act [T, G] = W_TF-peak^act [T, P] @ W_peak-gene [G, P]^T
    M_act = W_TF_peak_act @ W_pg_sparse.T  # [T, G]
    M_rep = W_TF_peak_rep @ W_pg_sparse.T  # [T, G]

    # Normalize to [0, 1]
    M_act = M_act / (M_act.max() + 1e-8)
    M_rep = M_rep / (M_rep.max() + 1e-8)

    M_combined = torch.max(M_act, M_rep)
    M_mask = (M_combined > threshold)

    return M_mask, M_act, M_rep


class SICA_GRN(nn.Module):
    """
    Phase 2 full model: loads Phase 1 weights (Encoder + Modules 1–3) and
    adds Module 4 (GRN inference).

    By default Modules 1–3 are frozen; only Module 4 is trained.
    Pass freeze_phase1=False to fine-tune all modules jointly.
    """

    def __init__(self, RNA_Dim, ATAC_Dim, TF, Latent_Dim,
                 W_mask, W_init, n_batches,
                 M_mask, dk=64, d_proj=64, freeze_phase1=True):
        super().__init__()

        self.RNA_Dim = RNA_Dim
        self.ATAC_Dim = ATAC_Dim
        self.TF = TF
        self.Latent_Dim = Latent_Dim
        self.freeze_phase1 = freeze_phase1

        # Phase 1 modules (weights loaded externally)
        self.Encoder = EncoderArchitecture(RNA_Dim, ATAC_Dim, Latent_Dim)
        self.Module1 = ATAC_Seq_Reconstruction(Latent_Dim, ATAC_Dim)
        self.Module2 = PeakGeneModule(RNA_Dim, ATAC_Dim, W_mask, W_init, n_batches, Latent_Dim)
        self.Module3 = TF_Sparse_Cosine_Attention(Latent_Dim, dk, TF)

        # Phase 2 module
        self.Module4 = GRNModule(TF, RNA_Dim, Latent_Dim, M_mask, n_batches, d_proj)

        if freeze_phase1:
            for module in [self.Encoder, self.Module1, self.Module2, self.Module3]:
                for p in module.parameters():
                    p.requires_grad_(False)

    def forward(self, RNA_Seq, ATAC_Seq, log_lib_rna, log_lib_atac, batch_ids, motif_scores):
        z, mu, logvar     = self.Encoder(RNA_Seq, ATAC_Seq, log_lib_rna, log_lib_atac)
        mu_ATAC, p_access = self.Module1(z, log_lib_atac)
        X_hat_rna, theta  = self.Module2(p_access, batch_ids, z)
        alpha_TF          = self.Module3(z, motif_scores)

        # E_TF shared from Module 3 into Module 4
        E_TF = self.Module3.E_TF
        X_hat_GRN, theta_GRN, W_act, W_rep, W_reg = self.Module4(
            z, alpha_TF, batch_ids, E_TF
        )

        return (z, mu, logvar,
                mu_ATAC,
                X_hat_rna, theta,
                alpha_TF,
                X_hat_GRN, theta_GRN, W_act, W_rep, W_reg)

    def compute_loss(self, outputs, RNA_Seq, ATAC_Seq, tf_idx, step,
                     beta_RNA=100.0, anneal_steps=10000, beta_max=1.0,
                     lam_alpha=0.01, lam_peak_gene=1e-5, lam_TF_corr=0.1,
                     lam_W=0.001, lam_excl=0.01):

        (_, mu, logvar,
         mu_ATAC,
         X_hat_rna, theta,
         alpha_TF,
         X_hat_GRN, theta_GRN, W_act, W_rep, _) = outputs

        # ── Phase 2 primary loss: GRN-mediated RNA (NB) ──────────────────────
        theta_GRN_pos = F.softplus(theta_GRN)
        p_grn = theta_GRN_pos / (theta_GRN_pos + X_hat_GRN + 1e-8)
        L_RNA_GRN = -torch.distributions.NegativeBinomial(theta_GRN_pos, p_grn).log_prob(RNA_Seq).sum()

        # Sparsity on W_act and W_rep
        L_sparse_W = W_act.abs().mean() + W_rep.abs().mean()

        # Mutual exclusion: penalize entries where both W_act and W_rep are large
        L_excl = (W_act * W_rep).mean()

        total = L_RNA_GRN + lam_W * L_sparse_W + lam_excl * L_excl

        # ── Optionally include Phase 1 losses when fine-tuning ───────────────
        if not self.freeze_phase1:
            L_ATAC = -torch.distributions.Poisson(mu_ATAC + 1e-8).log_prob(ATAC_Seq).sum()

            theta_pos = F.softplus(theta)
            p = theta_pos / (theta_pos + X_hat_rna + 1e-8)
            L_RNA_ATAC = -torch.distributions.NegativeBinomial(theta_pos, p).log_prob(RNA_Seq).sum()

            beta = min(step / anneal_steps, 1.0) * beta_max
            L_KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            X_TF = RNA_Seq[:, tf_idx]
            X_TF_norm  = (X_TF - X_TF.mean(0)) / (X_TF.std(0) + 1e-8)
            alpha_norm = (alpha_TF - alpha_TF.mean(0)) / (alpha_TF.std(0) + 1e-8)
            L_TF_corr  = -(alpha_norm * X_TF_norm).mean()
            L_sparse_tf = alpha_TF.abs().mean()
            L_peak_gene = self.Module2.W_values.pow(2).sum()

            total = (L_ATAC
                     + beta_RNA * L_RNA_ATAC
                     + L_RNA_GRN
                     + beta * L_KL
                     + lam_alpha * L_sparse_tf
                     + lam_peak_gene * L_peak_gene
                     + lam_TF_corr * L_TF_corr
                     + lam_W * L_sparse_W
                     + lam_excl * L_excl)

            breakdown = {
                "L_ATAC":     L_ATAC.item(),
                "L_RNA_ATAC": L_RNA_ATAC.item(),
                "L_RNA_GRN":  L_RNA_GRN.item(),
                "L_KL":       L_KL.item(),
                "L_TF_corr":  L_TF_corr.item(),
                "L_sparse_W": L_sparse_W.item(),
                "L_excl":     L_excl.item(),
            }
        else:
            breakdown = {
                "L_RNA_GRN":  L_RNA_GRN.item(),
                "L_sparse_W": L_sparse_W.item(),
                "L_excl":     L_excl.item(),
            }

        return total, breakdown
