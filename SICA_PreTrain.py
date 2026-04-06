import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import EncoderArchitecture
from Module1 import ATAC_Seq_Reconstruction
from Module2 import PeakGeneModule
from Module3 import TF_Sparse_Cosine_Attention
class SICA_Pretrain(nn.Module):
    def __init__(self, RNA_Dim, ATAC_Dim, Batch_Size, TF, Latent_Dim, W_mask, W_init, n_batches, dk=64):
        super().__init__()
        self.RNA_Dim=RNA_Dim
        self.ATAC_Dim=ATAC_Dim
        self.Batch_Size=Batch_Size
        self.Latent_Dim=Latent_Dim
        self.TF=TF
        self.Encoder=EncoderArchitecture(self.RNA_Dim,self.ATAC_Dim,self.Latent_Dim)
        self.Module1=ATAC_Seq_Reconstruction(self.Latent_Dim,self.ATAC_Dim)
        self.Module2=PeakGeneModule(self.RNA_Dim, self.ATAC_Dim, W_mask, W_init, n_batches, self.Latent_Dim)
        self.Module3=TF_Sparse_Cosine_Attention(self.Latent_Dim, dk, self.TF)

    def forward(self, RNA_Seq, ATAC_Seq, log_lib_rna, log_lib_atac, batch_ids, motif_scores):
        z, mu, logvar          = self.Encoder(RNA_Seq, ATAC_Seq, log_lib_rna, log_lib_atac)
        mu_ATAC, p_access      = self.Module1(z, log_lib_atac)
        X_hat_rna, theta       = self.Module2(p_access, batch_ids, z)
        alpha_TF               = self.Module3(z, motif_scores)
        return z, mu, logvar, mu_ATAC, X_hat_rna, theta, alpha_TF

    def compute_loss(self, outputs, RNA_Seq, ATAC_Seq, tf_idx, step,
                     beta_RNA=100.0, anneal_steps=10000, beta_max=1.0,
                     lam_alpha=0.01, lam_peak_gene=1e-5, lam_TF_corr=0.1):

        _, mu, logvar, mu_ATAC, X_hat_rna, theta, alpha_TF = outputs

        # Module 1 — Poisson NLL
        L_ATAC = -torch.distributions.Poisson(mu_ATAC + 1e-8).log_prob(ATAC_Seq).sum()

        # Module 2 — Negative Binomial NLL
        theta_pos = F.softplus(theta)
        p = theta / (theta + X_hat_rna + 1e-8)
        L_RNA = -torch.distributions.NegativeBinomial(theta, p).log_prob(RNA_Seq).sum()

        # Encoder — KL divergence with annealing
        beta = min(step / anneal_steps, 1.0) * beta_max
        L_KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Module 3 — TF correlation loss
        X_TF = RNA_Seq[:, tf_idx]
        X_TF_norm   = (X_TF - X_TF.mean(0)) / (X_TF.std(0) + 1e-8)
        alpha_norm  = (alpha_TF - alpha_TF.mean(0)) / (alpha_TF.std(0) + 1e-8)
        L_TF_corr   = -(alpha_norm * X_TF_norm).mean()

        # sparsity + peak-gene regularization
        L_sparse_tf = alpha_TF.abs().mean()
        L_peak_gene = self.Module2.W_values.pow(2).sum()

        total = (L_ATAC
                 + beta_RNA * L_RNA
                 + beta * L_KL
                 + lam_alpha * L_sparse_tf
                 + lam_peak_gene * L_peak_gene
                 + lam_TF_corr * L_TF_corr)

        breakdown = {
            "L_ATAC":     L_ATAC.item(),
            "L_RNA":      L_RNA.item(),
            "L_KL":       L_KL.item(),
            "L_TF_corr":  L_TF_corr.item(),
            "L_sparse":   L_sparse_tf.item(),
            "L_peak_gene":L_peak_gene.item(),
        }
        return total, breakdown