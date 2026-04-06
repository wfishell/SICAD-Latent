import torch
import torch.nn as nn

class PeakGeneModule(nn.Module):
    def __init__(self, n_genes, n_peaks, W_mask, W_init, n_batches, latent_dim):
        super().__init__()

        mask_sparse = W_mask.to_sparse()
        self.register_buffer('mask_indices', mask_sparse.indices())
        self.register_buffer('mask_shape', torch.tensor(W_mask.shape))

        n_nonzero = mask_sparse._nnz()
        self.W_values = nn.Parameter(mask_sparse.values().clone())

        self.B_RNA_ATAC = nn.Parameter(torch.zeros(n_batches, n_genes))
        self.theta = nn.Parameter(torch.ones(n_genes))
        self.library_mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.batch_norm = nn.BatchNorm1d(n_genes)
        self.n_genes = n_genes
        self.n_peaks = n_peaks

    def forward(self, p_access, batch_ids, z):
        values = torch.clamp(self.W_values, 0, 1)
        W_sparse = torch.sparse_coo_tensor(
            self.mask_indices, values,
            size=(self.n_genes, self.n_peaks),
            device=p_access.device
        )

        p_min = p_access.min(dim=1, keepdim=True).values
        p_max = p_access.max(dim=1, keepdim=True).values
        p_norm = (p_access - p_min) / (p_max - p_min + 1e-8)

        R = torch.sparse.mm(W_sparse, p_norm.T).T

        R = R + batch_ids @ self.B_RNA_ATAC
        R = torch.softmax(self.batch_norm(R), dim=1)

        L_lib = torch.exp(self.library_mlp(z))
        X_hat = L_lib * R

        return X_hat, self.theta