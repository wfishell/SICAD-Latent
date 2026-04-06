import torch
import torch.nn as nn

class ATAC_Seq_Reconstruction(nn.Module):
    def __init__(self, latent_dim, n_peaks):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, n_peaks)
        )

    def forward(self, z, log_library_atac):
        logits_ATAC = self.mlp(z)
        mu_ATAC     = torch.exp(logits_ATAC) * torch.exp(log_library_atac.unsqueeze(1))
        p_access    = torch.softmax(logits_ATAC, dim=1)

        return mu_ATAC, p_access
