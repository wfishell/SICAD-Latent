import torch
import torch.nn as nn

class EncoderArchitecture(nn.Module):
    def __init__(self, RNA_Seq_Input, ATAC_Seq_Input, Latent_Dim):
        super().__init__()
        self.Latent_Dim = Latent_Dim

        # RNA encoder branch
        self.RNA_encoder = nn.Sequential(
            nn.Linear(RNA_Seq_Input + 1, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.RNA_mu     = nn.Linear(256, Latent_Dim)
        self.RNA_logvar = nn.Linear(256, Latent_Dim)

        # ATAC encoder branch
        self.ATAC_encoder = nn.Sequential(
            nn.Linear(ATAC_Seq_Input + 1, 512, bias=False), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.ATAC_mu     = nn.Linear(256, Latent_Dim)
        self.ATAC_logvar = nn.Linear(256, Latent_Dim)

    def forward(self, RNA_Seq, ATAC_Seq, log_lib_rna, log_lib_atac):
        # concatenate log library size to inputs
        RNA_Seq  = torch.cat([RNA_Seq,  log_lib_rna.unsqueeze(1)],  dim=1)  # [N, G+1]
        ATAC_Seq = torch.cat([ATAC_Seq, log_lib_atac.unsqueeze(1)], dim=1)  # [N, P+1]

        # RNA branch
        RNA_x      = self.RNA_encoder(RNA_Seq)
        RNA_mu     = self.RNA_mu(RNA_x)
        RNA_logvar = self.RNA_logvar(RNA_x)

        # ATAC branch
        ATAC_x      = self.ATAC_encoder(ATAC_Seq)
        ATAC_mu     = self.ATAC_mu(ATAC_x)
        ATAC_logvar = self.ATAC_logvar(ATAC_x)

        # Product of Experts fusion
        RNA_prec  = 1 / torch.exp(RNA_logvar)
        ATAC_prec = 1 / torch.exp(ATAC_logvar)
        precision = RNA_prec + ATAC_prec + 1.0       # +1 = unit Gaussian prior

        Z_mean   = (RNA_mu * RNA_prec + ATAC_mu * ATAC_prec) / precision
        Z_logvar = -torch.log(precision)
        Z_logvar = torch.clamp(Z_logvar, -10, 10)    # numerical stability

        # reparameterization
        epsilon  = torch.randn_like(Z_mean)
        Z_latent = Z_mean + torch.exp(0.5 * Z_logvar) * epsilon

        return Z_latent, Z_mean, Z_logvar
