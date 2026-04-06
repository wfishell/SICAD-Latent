import torch
import torch.nn as nn
class TF_Sparse_Cosine_Attention(nn.Module):
    def __init__(self, Latent_Dim, dk, TF):                                                                                                                                                     
        super().__init__()       
        self.Latent_Dim = Latent_Dim
        self.dk = dk                                                                                                                                                                              
        self.TF = TF
                                                                                                                                                                                                
        # TF embedding matrix
        self.E_TF = nn.Parameter(torch.randn(TF, Latent_Dim) * (0.02 ** 0.5))                                                                                                          
                                                                                                                                                                                                
        self.W_Q = nn.Parameter(torch.randn(Latent_Dim, dk))                                                                                                                         
        self.W_K = nn.Parameter(torch.randn(Latent_Dim, dk))                                                                                                                        
                                                                                                                                                                                                
        self.tau = nn.Parameter(torch.tensor(0.1))                                                                                                                                                
                
        self.motif_mlp = nn.Sequential(
            nn.Linear(TF, 128),                                                                                                                                                                   
            nn.ReLU(),
            nn.Linear(128, TF)                      
        )

    def sparsemax(self, z):                                                                                                                                                                       
          sorted_z, _ = torch.sort(z, dim=-1, descending=True)
          cumsum = torch.cumsum(sorted_z, dim=-1)                                                                                                                                                   
          k = torch.arange(1, z.shape[-1] + 1, device=z.device).float()
          support = (1 + k * sorted_z) > cumsum                                                                                                                                                     
          k_z = support.sum(dim=-1, keepdim=True).float()                                                                                                                                           
          tau_z = (cumsum.gather(-1, (k_z - 1).long()) - 1) / k_z
          return torch.clamp(z - tau_z, min=0)

    def forward(self,z,motif_scores):
        Q=z @ self.W_Q
        K=self.E_TF@self.W_K
        Q_norm = Q / (Q.norm(dim=1, keepdim=True) + 1e-8)                                                                                                                                         
        K_norm = K / (K.norm(dim=1, keepdim=True) + 1e-8)
        scores = (Q_norm @ K_norm.T) / self.tau                                                                                                                                                                                                                                                                                                                   
        beta_motif = self.motif_mlp(motif_scores)                                                                                                                                                                                                                                                                                                                 
        alpha_TF = self.sparsemax(scores + beta_motif)                                                                                                                    
        return alpha_TF
