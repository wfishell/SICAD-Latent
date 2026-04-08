import torch
import torch.nn as nn


class GRNModule(nn.Module):
    """
    Module 4: GRN Inference (Stage 2)

    Learns cell-state-specific TF→gene regulatory weights constrained by the
    mechanistic TF→peak→gene structural prior (M_mask) derived from Phase 1.

    Regulatory weights use the embedding-based parameterization (Option A):
        W_act = σ(E_TF · W_act_proj · (E_G · W_G_proj)^T) ⊙ M_mask  [T × G]
        W_rep = σ(E_TF · W_rep_proj · (E_G · W_G_proj)^T) ⊙ M_mask  [T × G]
        W_reg = W_act - W_rep  ∈ [-1, 1]^{T × G}

    GRN-mediated RNA reconstruction:
        log(μ_g) = baseline_g(z) + Σ_t α_TF[i,t] · W_reg[t,g] · scale_t
        X̂_RNA_GRN = exp(log_μ)
    """

    def __init__(self, n_tfs, n_genes, latent_dim, M_mask, n_batches, d_proj=64):
        """
        Args:
            n_tfs:      Number of TFs (T)
            n_genes:    Number of genes (G)
            latent_dim: Latent dimension (d), must match Encoder/Module3
            M_mask:     Boolean structural prior [T × G], pre-computed from
                        Phase 1 W_peak-gene and W_TF-peak^act/rep
            n_batches:  Number of batches for batch correction
            d_proj:     Projection dimension for bilinear regulatory weights
        """
        super().__init__()

        self.n_tfs = n_tfs
        self.n_genes = n_genes

        # Structural prior mask (fixed after Phase 1)
        self.register_buffer('M_mask', M_mask.float())  # [T, G]

        # Gene embeddings (learned in Phase 2)
        self.E_G = nn.Parameter(torch.randn(n_genes, latent_dim) * (0.02 ** 0.5))

        # Separate projection matrices for activation and repression
        self.W_act_proj = nn.Parameter(torch.randn(latent_dim, d_proj) * (0.02 ** 0.5))
        self.W_rep_proj = nn.Parameter(torch.randn(latent_dim, d_proj) * (0.02 ** 0.5))
        self.W_G_proj   = nn.Parameter(torch.randn(latent_dim, d_proj) * (0.02 ** 0.5))

        # GRN-mediated RNA reconstruction
        self.baseline_mlp = nn.Sequential(
            nn.Linear(latent_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, n_genes),
        )

        # Learnable per-TF scale factor
        self.scale_t = nn.Parameter(torch.ones(n_tfs))

        # Batch correction
        self.B_RNA_GRN = nn.Parameter(torch.zeros(n_batches, n_genes))

        # Gene-specific dispersion for NB loss
        self.theta_GRN = nn.Parameter(torch.ones(n_genes))

    def get_regulatory_weights(self, E_TF):
        """
        Compute W_act and W_rep from TF and gene embeddings.

        Args:
            E_TF: TF embeddings [T, d], shared from Module 3

        Returns:
            W_act: [T, G] activation weights in [0, 1], masked
            W_rep: [T, G] repression weights in [0, 1], masked
        """
        TF_act = E_TF @ self.W_act_proj    # [T, d_proj]
        TF_rep = E_TF @ self.W_rep_proj    # [T, d_proj]
        G_proj = self.E_G @ self.W_G_proj  # [G, d_proj]

        W_act = torch.sigmoid(TF_act @ G_proj.T) * self.M_mask  # [T, G]
        W_rep = torch.sigmoid(TF_rep @ G_proj.T) * self.M_mask  # [T, G]

        return W_act, W_rep

    def forward(self, z, alpha_TF, batch_ids, E_TF):
        """
        Args:
            z:         Latent cell state [N, d]
            alpha_TF:  Sparse TF activity from Module 3 [N, T]
            batch_ids: One-hot batch encoding [N, K]
            E_TF:      TF embeddings from Module 3 [T, d]

        Returns:
            X_hat_GRN: GRN-mediated RNA reconstruction [N, G]
            theta_GRN: Gene-specific dispersion [G]
            W_act:     Activation regulatory weights [T, G]
            W_rep:     Repression regulatory weights [T, G]
            W_reg:     Signed regulatory matrix [T, G]
        """
        W_act, W_rep = self.get_regulatory_weights(E_TF)
        W_reg = W_act - W_rep  # [T, G]

        # Cell-state-specific GRN contribution
        # α_TF [N,T] * scale_t [T] → scaled_alpha [N,T]
        # scaled_alpha @ W_reg [T,G] → grn_contribution [N,G]
        scaled_alpha = alpha_TF * self.scale_t.unsqueeze(0)  # [N, T]
        grn_contribution = scaled_alpha @ W_reg              # [N, G]

        # Gene-specific baseline from latent state
        baseline = self.baseline_mlp(z)                      # [N, G]

        # Batch correction
        batch_correction = batch_ids @ self.B_RNA_GRN        # [N, G]

        log_mu = baseline + grn_contribution + batch_correction
        X_hat_GRN = torch.exp(log_mu)

        return X_hat_GRN, self.theta_GRN, W_act, W_rep, W_reg
