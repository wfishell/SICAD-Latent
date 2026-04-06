# SICA-Latent v2: Technical Report (Revised)

## Title
Inference of Continuous Gene Regulatory Networks via Dual-Stage Sparse Attention with Mechanistic Peak-Mediated Constraints

**Project Code:** SICA-Latent v2 (Sparse Interpretable Cosine Attention on Latent Manifolds)

**Data Modality:** snMultiome (scRNA-seq + scATAC-seq) + HiChIP/PLAC-seq (validation only) + TF Knockout scRNA-seq (validation only)

**Target Journal:** Nature Biotechnology / Nature Methods

---

## 1. Executive Summary

Gene Regulatory Network (GRN) inference from single-cell data is limited by: (1) discrete cluster-based methods that cannot capture continuous regulatory dynamics, (2) models that conflate reconstruction with causal regulation, (3) lack of mechanistic path decomposition (TF→peak→gene), and (4) absence of principled uncertainty quantification.

**SICA-Latent v2** is a multi-modal Variational Autoencoder framework that addresses all four limitations through:

- **Variational latent manifold** with Product-of-Experts fusion for uncertainty-aware, continuous cell-state representation
- **Dual-stage sparse cosine attention** (Stage 1: cell-specific TF activity; Stage 2: regulatory weight learning)
- **Mechanistic peak-mediated GRN path**: Explicit TF→peak, peak→gene, and TF→gene inference via in-silico ChIP preprocessing and learned peak-gene linkages
- **Two-phase training**: Core reconstruction first (ATAC + RNA-from-ATAC + TF activity), then GRN module
- **Gradient-based in silico perturbation** through fully differentiable architecture
- **Separate activation/repression decomposition** for biologically accurate TF→target modeling

Causality is validated by holding out TF knockout scRNA-seq data entirely. 3D chromatin (HiChIP/PLAC-seq) is reserved exclusively for independent structural validation—not used in training.

---

## 2. Relationship to scDoRI and Key Innovations

### 2.1. scDoRI Overview (Saraswat et al., 2025)

scDoRI is a concurrent deep-learning framework for eGRN inference from snMultiome data. It employs a deterministic encoder–decoder architecture where:
- The encoder maps RNA+ATAC → softmax topic mixture θ (analogous to LDA)
- Four decoder modules reconstruct ATAC (Module 1), RNA from ATAC via peak-gene links (Module 2), TF expression (Module 3), and GRN-mediated RNA (Module 4)
- TF–peak interactions are pre-computed as signed in-silico ChIP matrices
- GRNs are topic-specific (discrete), with cells represented as topic mixtures
- Two-phase training: reconstruction first, GRN second

### 2.2. What We Adopt from scDoRI (with attribution)

| Adopted Element | scDoRI Source | Our Adaptation |
|----------------|--------------|----------------|
| **In-silico ChIP preprocessing** | §1.3: Motif × TF-accessibility correlation → signed W_TF-peak | Same pipeline; produces TF→peak scores as fixed input |
| **Peak-gene linkage learning** | §1.4.4: W_gene-peak with distance mask/decay, clamped [0,1] | Module 2: learns peak→gene links from ATAC→RNA reconstruction |
| **TF→peak→gene decomposition** | §1.4.6: Compose TF-peak × peak-gene for mechanistic GRN | Peak-derived structural prior masks regulatory weight learning |
| **Two-phase training** | §1.6: Phase 1 (reconstruction), Phase 2 (GRN) | Same strategy; stabilizes latent space before GRN learning |
| **Poisson loss for ATAC** | §1.5.1: Poisson likelihood for fragment counts | Replace BCE with Poisson; better for count data |
| **Empirical GRN thresholding** | §1.9.5: Permutation-based significance testing | Post-hoc filtering of TF-gene edges |
| **Activator/repressor via signed TF-peak** | §1.3.5: Positive/negative TF-accessibility correlation | Preprocessing step producing W_TF-peak^act and W_TF-peak^rep |

### 2.3. Our Innovations Beyond scDoRI

| Innovation | SICA-Latent v2 | scDoRI | Significance |
|-----------|---------------|--------|-------------|
| **Variational latent space (VAE + PoE)** | Continuous probabilistic manifold with uncertainty quantification | Deterministic softmax topic mixture | Enables sampling, interpolation, proper uncertainty; continuous gradient field |
| **Cosine attention + sparsemax** | Cell-specific TF activity via cos(z·W_Q, E_TF·W_K)/τ with exact sparsity | Linear topic–TF decoder (Wtopic-TF) aggregated per topic | Magnitude-invariant, automatically sparse, per-cell not per-topic |
| **Continuous cell-state GRNs** | A(z) = diag(α_TF(z)) · W_reg at any z on manifold | G_net[t] is discrete per-topic; cell GRN = Σ_t θ[t]·G_net[t] | Query GRN at any trajectory point; no topic discretization |
| **Gradient-based perturbation** | Differentiable KO simulation, inverse design via ∂/∂z | Post-hoc downstream analysis only | Compute exact gradients ∂gene/∂TF; find minimal perturbations |
| **Learnable motif prior in attention** | β_motif(z) = MLP(motif_scores) as conditional bias in attention | Motif used only in preprocessing (in-silico ChIP) | Motif information modulates attention dynamically, not just as static prior |
| **HiChIP-free training + HiChIP validation** | 3D chromatin reserved as independent validation metric | No 3D chromatin data used at all | Stronger validation: structural consistency tested without training leakage |
| **Held-out KO causal validation** | Single-TF knockout scRNA-seq held out entirely | Benchmarked on semi-synthetic ENCODE + Neftel expression | Gold-standard causal validation without circularity |
| **Per-cell TF activity (not per-topic)** | α_TF(z) is computed for each cell's latent state | TF activity aggregated at topic level, then distributed to cells via θ | Captures fine-grained, within-topic variation in TF activity |

### 2.4. Architectural Differences Summary

```
scDoRI:                                    SICA-Latent v2:
──────                                     ──────────────
Encoder: concat MLPs → softmax θ           Encoder: parallel MLPs → PoE → z ~ N(μ,σ²)
Latent: θ ∈ Δ^T (simplex, T topics)       Latent: z ∈ ℝ^d (continuous manifold, d=128)
GRN: G_net[t] per topic (discrete)         GRN: A(z) continuous at any z
TF activity: Wtopic-TF (linear, per-topic) TF activity: cosine attention (nonlinear, per-cell)
Loss: Poisson + NB + regularization        Loss: Poisson + NB + KL + sparsity + regularization
Training: 2-phase                          Training: 2-phase
Perturbation: post-hoc analysis            Perturbation: gradient-based, differentiable
```

---

## 3. Key Design Decisions (Updated)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **ChIP in loss?** | **No.** Use motif scanning + in-silico ChIP as preprocessing | Many cell types lack ChIP data; avoid circularity |
| **TF list source** | Filter AnimalTFDB TFs by expression (mean > threshold) | Expected ~500–800 TFs |
| **Repression modeling** | **Separate W_act and W_rep matrices** | Biologically realistic; signed regulation |
| **Knockout data** | **Hold out entirely for validation** | Clean causal claims |
| **HiChIP/PLAC-seq** | **Validation only** (NOT used in training) | Independent structural validation; avoids data leakage |
| **TF→peak→gene** | In-silico ChIP + learned peak-gene links | Mechanistic decomposition; produces 3 output types |
| **Training strategy** | **Two-phase** (reconstruction → GRN) | Stabilizes latent space before GRN learning |
| **ATAC loss** | **Poisson** (not BCE) | More appropriate for count data |
| **Compute budget** | 2× NVIDIA A5500 GPUs (24GB VRAM each) | DataParallel, mixed precision |

---

## 4. Data Preprocessing

### 4.1. Input Data

| Input | Notation | Shape | Description |
|-------|----------|-------|-------------|
| RNA counts | X | [N × G] | Raw scRNA-seq counts (N cells, G genes) |
| ATAC counts | Y | [N × P] | Raw scATAC-seq fragment counts (N cells, P peaks) |
| Library sizes | L_RNA, L_ATAC | [N] | Log-transformed total counts per modality |
| Batch IDs | B | [N × K] | One-hot batch encoding (K batches) |
| TF indices | tf_idx | [T] | Column indices in X corresponding to TFs |
| Peak coordinates | peak_coords | [P × 3] | chr, start, end for each peak |
| Gene coordinates | gene_coords | [G × 4] | chr, start, end, strand for each gene |

### 4.2. In-Silico ChIP-seq: Signed TF–Peak Matrices (Adopted from scDoRI §1.3)

**Goal:** Pre-compute TF–peak binding scores with regulatory sign (activator/repressor).

**Step 1: Pseudobulk aggregation for correlation estimation**
- Perform Leiden clustering on batch-corrected (Harmony) RNA PCA embeddings
- Aggregate cells into pseudobulks (≥50 metacells recommended)
- This aggregation is ONLY for preprocessing, not for model training

**Step 2: Correlation estimation**
For each pseudobulk:
- Library-size normalize ATAC peak counts, min-max scale
- Library-size normalize TF expression, log-transform
- Compute Pearson correlation: `R_peak-TF ∈ ℝ^{P × T}`

**Step 3: Motif scanning**
- Run FIMO against reference genome with cisBP/JASPAR/HOCOMOCO motifs
- Produce motif match matrix: `M ∈ ℝ^{P × T}`

**Step 4: TF-specific filtering**
For each TF:
- Define background: bottom 20% peaks by motif score
- Compute background correlation distribution
- Apply TF-specific threshold: 95th percentile (activators), 5th percentile (repressors)
- Zero out correlations below threshold

**Step 5: Split by sign**
```
R_act[p,t] = max(R_filtered[p,t], 0)
R_rep[p,t] = min(R_filtered[p,t], 0)
```

**Step 6: Integrate motif and correlation**
```
W_TF-peak^act = M ⊙ R_act  ∈ ℝ^{P × T}  (or equivalently [T × P] transposed)
W_TF-peak^rep = M ⊙ |R_rep|  ∈ ℝ^{P × T}
```

**Outputs (fixed, not learned):**
- `W_TF-peak^act ∈ ℝ^{T × P}`: Activating TF–peak binding scores
- `W_TF-peak^rep ∈ ℝ^{T × P}`: Repressive TF–peak binding scores

### 4.3. Peak–Gene Distance Mask

```
W_mask[g, p] = 𝟙(distance(peak_p, gene_g) < window)
```
- Default window: 150 kb around gene body (following scDoRI and ArchR)
- Species-adjustable parameter
- Expected density: ~0.5–1% of G × P entries

### 4.4. Peak–Gene Distance Initialization

Initialize learned peak-gene weights with exponential distance decay:
```
W_init[g, p] = exp(-distance(peak_p, TSS_g) / decay_rate) × W_mask[g, p]
```
- `decay_rate`: 10,000 bp (following ArchR gene score logic)
- Promotes proximal peaks; distal peaks start with low weight

### 4.5. Motif Enrichment Scores for TF Activity Prior

Per-cell motif enrichment (pre-computed):
```
motif_scores[i, t] = Σ_p Y[i, p] · 𝟙[motif_t ∈ p]
```
Or use chromVAR deviation z-scores for more robust estimation.

---

## 5. Mathematical Framework & Architecture

### 5.1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              SICA-Latent v2 (Revised)                               │
│                                                                                     │
│   ┌──────────────┐       ┌──────────────┐                                           │
│   │   RNA (X)    │       │   ATAC (Y)   │                                           │
│   │  [N × G]     │       │   [N × P]    │                                           │
│   └──────┬───────┘       └──────┬───────┘                                           │
│          │                      │                                                   │
│          ▼                      ▼                                                   │
│   ┌──────────────┐       ┌──────────────┐                                           │
│   │  Encoder_RNA │       │ Encoder_ATAC │                                           │
│   │    (MLP)     │       │    (MLP)     │                                           │
│   └──────┬───────┘       └──────┬───────┘                                           │
│          └──────────┬───────────┘                                                   │
│                     ▼                                                               │
│             ┌───────────────┐                                                       │
│             │ Product of    │                                                       │
│             │ Experts (PoE) │                                                       │
│             └───────┬───────┘                                                       │
│                     ▼                                                               │
│             ┌───────────────┐                                                       │
│             │   z ~ q(z|X,Y)│   Latent Cell State [N × d]                           │
│             └───────┬───────┘                                                       │
│                     │                                                               │
│    ┌────────────────┼────────────────┬──────────────────┐                            │
│    │                │                │                  │                            │
│    ▼                ▼                ▼                  ▼                            │
│ ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────────────┐               │
│ │ MODULE 1 │  │ MODULE 2 │  │  MODULE 3  │  │      MODULE 4        │               │
│ │   ATAC   │  │ RNA from │  │ TF Activity│  │   GRN Inference      │               │
│ │  Recon   │  │   ATAC   │  │  (Stage 1) │  │    (Stage 2)         │               │
│ │          │  │          │  │            │  │                      │               │
│ │ z → Ŷ   │  │ Ŷ·W_pg  │  │ α_TF(z)   │──▶│ W_act, W_rep         │               │
│ │(Poisson) │  │ → X̂_atac│  │ cosine     │  │ → A(z) = GRN         │               │
│ │          │  │ (NB loss)│  │ +sparsemax │  │ → X̂_grn (NB loss)   │               │
│ └──────────┘  └──────────┘  └────────────┘  └──────────────────────┘               │
│                                                                                     │
│  PHASE 1: Modules 1-3          PHASE 2: Module 4 (freeze/fine-tune 1-3)            │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2. Encoder: Dual-Encoder VAE with Product-of-Experts

**Inputs (with library size, following scDoRI):**
- `x_rna = [X[i,:]; log(L_RNA[i])]` ∈ ℝ^{G+1}
- `x_atac = [Y[i,:]; log(L_ATAC[i])]` ∈ ℝ^{P+1}

**Encoders (MLPs with BatchNorm + ReLU + Dropout):**
```
μ_RNA, logvar_RNA = Enc_RNA(x_rna)     # → [N, d]
μ_ATAC, logvar_ATAC = Enc_ATAC(x_atac)  # → [N, d]
```

Architecture: `input → 512 → BN → ReLU → Drop(0.1) → 256 → BN → ReLU → Drop(0.1) → (μ, logvar)`

**Product-of-Experts Fusion:**
```
precision_z = 1/exp(logvar_RNA) + 1/exp(logvar_ATAC) + 1  (include unit prior)
μ_z = (μ_RNA/exp(logvar_RNA) + μ_ATAC/exp(logvar_ATAC)) / precision_z
logvar_z = -log(precision_z)
```

**Reparameterization:**
```
z = μ_z + exp(0.5 · logvar_z) · ε,   ε ~ N(0, I)
```

Clamping: logvar ∈ [-10, 10]

### 5.3. Module 1: ATAC Reconstruction

**Goal:** Reconstruct ATAC fragment counts from latent state.

```
logits_ATAC = MLP_ATAC(z)                    # z → 256 → BN → ReLU → 512 → BN → ReLU → P
μ_ATAC = exp(logits_ATAC) · exp(L_ATAC)     # Scale by library size
```

**Peak accessibility profile (used by Modules 2 and 4):**
```
p_access(z) = softmax(logits_ATAC)          # Normalized peak accessibility ∈ [0,1]^P
```

**Loss:** Poisson negative log-likelihood
```
L_ATAC = -Σ_{i,p} [Y[i,p] · log(μ_ATAC[i,p]) - μ_ATAC[i,p]]
```

### 5.4. Module 2: Peak–Gene Linkage (RNA from ATAC)

**Goal:** Reconstruct RNA expression from ATAC peak accessibility via learned peak-gene links.

**Learnable Parameters:**
- `W_peak-gene ∈ [0,1]^{G × P}`: Peak-gene linkage weights
  - Masked by W_mask (genomic distance constraint)
  - Initialized with distance decay (§4.4)
  - Clamped to [0, 1] after each update
  - Stored as sparse matrix (only entries within W_mask)

- `B_RNA_ATAC ∈ ℝ^{K × G}`: Batch correction weights

**Reconstruction:**
```
# Get normalized peak accessibility per cell
p̃(z) = MinMaxNorm(softmax(logits_ATAC))     # [N, P]

# Map peaks to genes via learned linkage
R_RNA_ATAC = p̃(z) @ W_peak-gene^T + B @ B_RNA_ATAC    # [N, G]

# Normalize and scale
R_RNA_ATAC = softmax(BatchNorm(R_RNA_ATAC))
X̂_RNA_ATAC = L_lib_RNA · R_RNA_ATAC
```

Where `L_lib_RNA ∈ ℝ^{N×1}` is a learned library size factor from a 2-layer MLP on X (following scVI/scDoRI).

**Loss:** Negative Binomial
```
L_RNA_ATAC = -Σ_{i,g} log NB(X[i,g] | μ=X̂_RNA_ATAC[i,g], θ=θ_g^ATAC)
```

Gene-specific dispersion `θ_g^ATAC` is learned.

**Outputs:** W_peak-gene provides the **peak→gene** inference product.

### 5.5. Module 3: TF Activity Inference (Stage 1 — Our Key Innovation)

**Goal:** Infer cell-specific TF activity scores via sparse cosine attention on the latent manifold.

**Learnable TF Embeddings:**
```
E_TF ∈ ℝ^{T × d}   (initialized ~ N(0, 0.02))
```

**Cosine Attention with Motif Prior:**
```
Q = z · W_Q              # [N, d_k],  W_Q ∈ ℝ^{d × d_k}, d_k=64
K = E_TF · W_K           # [T, d_k],  W_K ∈ ℝ^{d × d_k}

scores = cos(Q, K) / τ   # [N, T], cosine similarity scaled by temperature
                          # cos(a, b) = (a / ||a||₂) · (b / ||b||₂)^T

α_TF(z) = sparsemax(scores + β_motif(z))   # [N, T], sparse TF activity
```

Where:
- `τ`: Learnable temperature (nn.Parameter, init=0.1)
- `sparsemax`: Sparse alternative to softmax; produces exact zeros
- `β_motif(z) = MLP_β(motif_scores[i])`: Conditional motif prior bias
  - MLP: `T → 128 → ReLU → T`

**Interpretation:** `α_TF[i, t]` = activity score of TF t in cell i. Exactly sparse (most entries zero).

**Auxiliary TF expression correlation loss (soft constraint):**
```
L_TF_corr = -corr(α_TF, X_TF_normalized)
```

Encourages α_TF to correlate with actual TF expression as a regularizer (weight: λ_TF_corr = 0.1).

### 5.6. Module 4: GRN Inference (Stage 2 — Peak-Mediated + Embedding Hybrid)

**Goal:** Learn TF→gene regulatory weights constrained by mechanistic TF→peak→gene path.

#### 5.6.1. Peak-Derived Structural Prior (Replaces HiChIP-based M_prior)

Compose pre-computed TF-peak matrices with learned peak-gene links:

```
M_act[t, g] = Σ_p W_TF-peak^act[t, p] × W_peak-gene[g, p]    # [T × G]
M_rep[t, g] = Σ_p W_TF-peak^rep[t, p] × W_peak-gene[g, p]    # [T × G]
```

- Computed from Phase 1 learned W_peak-gene (frozen when entering Phase 2)
- Provides the structural mask for TF→gene learning
- **This is the mechanistic link**: TF can only regulate gene g if TF binds a peak that is linked to gene g

Normalize to [0, 1]:
```
M_act = M_act / (max(M_act) + ε)
M_rep = M_rep / (max(M_rep) + ε)
M_combined = max(M_act, M_rep)   # Union mask for W_act/W_rep
```

Binarize or soft-threshold:
```
M_mask[t, g] = 𝟙(M_combined[t, g] > threshold)   # Default threshold: 0.01
```

Expected density: ~5–10% of T × G entries.

#### 5.6.2. Regulatory Weight Learning

**Option A: Embedding-based (default):**
```
W_act = σ(E_TF · W_act_proj · (E_G · W_G_proj)^T) ⊙ M_mask   # [T × G]
W_rep = σ(E_TF · W_rep_proj · (E_G · W_G_proj)^T) ⊙ M_mask   # [T × G]
```
Where:
- `E_G ∈ ℝ^{G × d}` (initialized ~ N(0, 0.02)): Gene embeddings
- `W_act_proj, W_rep_proj ∈ ℝ^{d × d_proj}`: Separate projection matrices
- `W_G_proj ∈ ℝ^{d × d_proj}`: Shared gene projection
- `σ`: Sigmoid (bounds to [0, 1])

**Option B: Direct learnable tensors (scDoRI-style, alternative):**
```
W_act = ReLU(G_act_learn) ⊙ M_act   # [T × G], non-negative
W_rep = ReLU(G_rep_learn) ⊙ M_rep   # [T × G], non-negative
```
Where G_act_learn, G_rep_learn are learnable parameter tensors.

**Signed Regulatory Matrix:**
```
W_reg = W_act - W_rep  ∈ [-1, 1]^{T × G}
```

**Cell-State-Specific GRN:**
```
A(z) = diag(α_TF(z)) · W_reg  ∈ ℝ^{T × G}
```

Implemented as: `A = einsum('bt,tg->btg', alpha_TF, W_reg)`

**Interpretation:** `A(z)[t, g]` = signed regulatory effect of TF t on gene g in cell state z.

#### 5.6.3. GRN-Mediated RNA Reconstruction

```
grn_contribution_g = Σ_t α_TF[i, t] · W_reg[t, g] · scale_t
log(μ_g) = baseline_g(z) + grn_contribution_g
X̂_RNA_GRN = exp(log_mu)
```

Where:
- `baseline_g(z)`: MLP `z → 256 → BN → ReLU → G` (gene-specific baseline)
- `scale_t`: Learnable per-TF scale factor (nn.Parameter(torch.ones(T)))
- `B_RNA_GRN ∈ ℝ^{K × G}`: Batch correction for GRN decoder

Output distribution: Negative Binomial with learned gene-specific dispersion θ_g^GRN.

**Loss:**
```
L_RNA_GRN = -Σ_{i,g} log NB(X[i,g] | μ=X̂_RNA_GRN[i,g], θ=θ_g^GRN)
```

### 5.7. Loss Functions

#### Phase 1 Loss (Modules 1–3):
```
L_phase1 = L_ATAC + β_RNA · L_RNA_ATAC + β · D_KL
           + λ_α · ||α_TF||₁
           + λ_peak_gene · ||W_peak-gene||₂²
           + λ_TF_corr · L_TF_corr
```

| Component | Formula | Default Weight |
|-----------|---------|---------------|
| L_ATAC | Poisson NLL | 1.0 |
| L_RNA_ATAC | NB NLL for RNA from ATAC path | β_RNA = 100 |
| D_KL | -0.5 · mean(1 + logvar - μ² - exp(logvar)) | β (annealed) |
| Sparse α | mean(\|α_TF\|₁) | λ_α = 0.01 |
| Peak-gene L2 | \|\|W_peak-gene\|\|₂² | λ_peak_gene = 1e-5 |
| TF corr | -corr(α_TF, X_TF_norm) | λ_TF_corr = 0.1 |

**β-annealing:** β = min(step / anneal_steps, 1.0) × β_max. Linear from 0 to β_max=1.0 over 10,000 steps.

#### Phase 2 Loss (Module 4, optionally Modules 1–3):

If freezing Modules 1–3:
```
L_phase2 = L_RNA_GRN + λ_W · (||W_act||₁ + ||W_rep||₁) + λ_excl · mean(W_act ⊙ W_rep)
```

If fine-tuning all modules:
```
L_phase2 = L_ATAC + β_RNA · L_RNA_ATAC + L_RNA_GRN + β · D_KL + regularization
```

| Component | Formula | Default Weight |
|-----------|---------|---------------|
| L_RNA_GRN | NB NLL for GRN-mediated RNA | 1.0 |
| Sparse W | \|\|W_act\|\|₁ + \|\|W_rep\|\|₁ | λ_W = 0.001 |
| Mutual exclusion | mean(W_act ⊙ W_rep) | λ_excl = 0.01 |

### 5.8. Two-Phase Training

#### Phase 1: Core Reconstruction (Modules 1–3)

**Objective:** Learn stable latent space, peak accessibility, peak-gene links, and TF activity.
- Train encoder + ATAC decoder + peak-gene module + TF activity attention
- All parameters optimized jointly
- β-annealing for KL stability

**Duration:** ~60% of total training budget (e.g., 120 epochs for 200 total)

**Transition criterion:** Validation loss plateaus (early stopping patience = 20 epochs)

#### Phase 2: GRN Inference (Module 4)

**Prerequisite:** Compute M_act, M_rep from Phase 1 learned W_peak-gene + fixed W_TF-peak^act/rep.

**Options:**
1. **Freeze Modules 1–3** (default, recommended): Only optimize Module 4 parameters. Faster, preserves latent space.
2. **Fine-tune all**: Allows latent space to adapt to regulatory signals. May improve GRN accuracy but risks destabilizing.

**Duration:** ~40% of total training budget

### 5.9. Gradient-Based In Silico Perturbation

**Type 1: TF Knockout Simulation**
```
α_TF^KO_t = α_TF ⊙ (1 - e_t)         # Zero out TF t
X̂_KO = Decoder(z, α_TF^KO_t)
ΔX_KO_t = X̂_KO - X̂_orig
log2FC = log2((μ_KO + ε) / (μ_orig + ε))
```

**Type 2: State Transition Drivers**
```
Δα = α_TF(z_target) - α_TF(z_source)
gene_change = |Δα · W_reg|
```

**Type 3: Gradient-based target identification**
```
∂X̂_g / ∂α_TF[t] = W_reg[t, g] · scale_t   (analytic)
```

---

## 6. Model Outputs (Three-Level GRN Products)

### 6.1. TF → Peak (from preprocessing)
- `W_TF-peak^act ∈ ℝ^{T × P}`: Activating TF–peak binding scores
- `W_TF-peak^rep ∈ ℝ^{T × P}`: Repressive TF–peak binding scores
- Pre-computed, fixed, interpretable

### 6.2. Peak → Gene (from Module 2)
- `W_peak-gene ∈ [0, 1]^{G × P}`: Learned peak-gene linkage weights
- Distance-masked, interpretable enhancer–gene links
- Comparable to scDoRI Module 2 output or SCENIC+ peak-gene links

### 6.3. TF → Gene (from Module 4)
- `W_act ∈ [0, 1]^{T × G}`: Activation regulatory weights
- `W_rep ∈ [0, 1]^{T × G}`: Repression regulatory weights
- `W_reg = W_act - W_rep ∈ [-1, 1]^{T × G}`: Signed regulatory matrix
- Constrained by peak-derived structural prior (TF→peak→gene composition)

### 6.4. Cell-State-Specific GRN
- `A(z) = diag(α_TF(z)) · W_reg ∈ ℝ^{T × G}`: GRN at any latent state z
- Continuous, queryable at any trajectory point

### 6.5. Composed TF → Peak → Gene (post-hoc)
For any TF t and gene g:
```
TF_peak_gene_score[t, g] = Σ_p W_TF-peak[t,p] × p_access(z)[p] × W_peak-gene[g,p]
```
Cell-state-specific via peak accessibility p_access(z).

---

## 7. Validation Framework (4-Level Pyramid)

### Level 4 (Gold Standard): TF Knockout Concordance
- **Data:** Single-TF knockout scRNA-seq, held out entirely
- **Protocol:**
  1. Predicted activated targets: {g : W_act[t,g] > θ}
  2. Predicted repressed targets: {g : W_rep[t,g] > θ}
  3. Compute DE in KO vs WT
  4. Test: activated targets → downregulated in KO; repressed targets → upregulated
- **Metrics:** Signed AUROC, direction accuracy, Pearson r of predicted vs observed log2FC

### Level 3: ChIP-seq / CUT&RUN Target Recovery
- **Data:** ChIP-seq/CUT&RUN for held-out TFs (e.g., MYT1L CUT&RUN from scDoRI paper)
- **Protocol:** Compare predicted targets (|W_reg[t,g]| > θ) vs ChIP-defined targets
- **Metric:** AUPRC

### Level 2: HiChIP / PLAC-seq Structural Consistency (Independent)
- **Data:** HiChIP/PLAC-seq from matched cell types (NOT used in training)
- **Metric:** Fraction of high-weight edges in W_reg supported by chromatin loops
- **This is a unique validation**: scDoRI uses no 3D chromatin; we validate against it independently

### Level 1: Reconstruction Quality
| Modality | Metric |
|----------|--------|
| ATAC | Poisson log-likelihood |
| RNA (from ATAC) | NB log-likelihood, Pearson r per gene |
| RNA (from GRN) | NB log-likelihood, Pearson r per gene |

### Level 0: Latent Space Quality
- UMAP visualization colored by cell type
- Silhouette score in latent space
- Batch mixing (kBET)

### Baseline Comparisons
- SCENIC+ (pySCENIC)
- scDoRI
- CellOracle
- Optionally: GRaNIE, FigR

### Empirical GRN Thresholding (adopted from scDoRI §1.9.5)
Post-hoc permutation test:
1. Permute W_TF-peak matrices 1000 times
2. Recompute TF→gene scores for each permutation
3. Retain edges exceeding 95th percentile (activators) or 5th percentile (repressors) of null distribution

---

## 8. Computational Specifications

### Hardware
- 2× NVIDIA A5500 (24GB VRAM each)
- DataParallel, mixed precision (FP16)

### Memory Estimation (N=100k, T=800, G=4000, P=80000)

| Component | Shape | Memory |
|-----------|-------|--------|
| z, μ, logvar | N × 128 | ~150 MB |
| E_TF, E_G | 800×128 + 4000×128 | ~2.5 MB |
| W_peak-gene (sparse, ~0.6%) | 4000 × 80000 | ~8 MB (sparse) |
| W_TF-peak^act/rep (sparse) | 800 × 80000 (each) | ~25 MB each (sparse) |
| M_mask | 800 × 4000 | ~12 MB |
| W_act, W_rep | 800 × 4000 each | ~25 MB each |
| Encoder params | 2 MLPs | ~100 MB |
| ATAC decoder params | MLP | ~80 MB |
| **Batch computation** | batch×P + batch×G | ~12 GB at batch 2048 |

**Verdict:** Fits on single A5500; use DataParallel with batch 4096.

### Training Time Estimates

| Phase | Dataset | Epochs | Time |
|-------|---------|--------|------|
| Phase 1 | 100k cells | 120 | ~12 hours |
| Phase 2 (frozen) | 100k cells | 80 | ~6 hours |
| Phase 2 (fine-tune) | 100k cells | 80 | ~10 hours |
| Total | 100k cells | 200 | ~18–22 hours |

---

## 9. Hyperparameter Defaults & Sweep Ranges

### Model Architecture

| Parameter | Default | Notes |
|-----------|---------|-------|
| latent_dim | 128 | Manifold dimension |
| encoder_hidden_rna | [512, 256] | RNA encoder MLP |
| encoder_hidden_atac | [512, 256] | ATAC encoder MLP |
| decoder_hidden_atac | [256, 512] | ATAC decoder MLP |
| decoder_hidden_rna_baseline | [256] | RNA baseline MLP |
| attention_key_dim | 64 | Q/K projection |
| attention_proj_dim | 64 | TF→Target projection |
| tau_init | 0.1 | Attention temperature |
| dropout | 0.1 | Dropout rate |
| use_sparsemax | True | vs softmax |
| peak_gene_window | 150000 | 150kb |
| peak_gene_decay_rate | 10000 | Distance decay rate (bp) |
| n_topics | None | Not applicable (continuous) |
| phase2_freeze | True | Freeze modules 1-3 in phase 2 |

### Loss Weights

| Parameter | Default | Sweep |
|-----------|---------|-------|
| β_RNA (RNA-ATAC weight) | 100 | [50, 100, 200] |
| β_max (KL weight) | 1.0 | [0.5, 1.0, 2.0] |
| λ_α (sparsity on α_TF) | 0.01 | [0.001, 0.01, 0.1] |
| λ_W (sparsity on W_act/rep) | 0.001 | [0.0001, 0.001, 0.01] |
| λ_excl (mutual exclusion) | 0.01 | [0.001, 0.01, 0.1] |
| λ_peak_gene (L2 on W_pg) | 1e-5 | [1e-6, 1e-5, 1e-4] |
| λ_TF_corr (TF expression) | 0.1 | [0.01, 0.1, 0.5] |
| β_anneal_steps | 10000 | [5000, 10000, 20000] |

### Training

| Parameter | Default |
|-----------|---------|
| batch_size | 4096 (2 GPUs) |
| learning_rate_phase1 | 1e-3 |
| learning_rate_phase2 | 5e-4 |
| optimizer | Adam (β₁=0.9, β₂=0.999) |
| scheduler | CosineAnnealingWarmRestarts(T_0=50) |
| max_epochs_phase1 | 120 |
| max_epochs_phase2 | 80 |
| early_stopping_patience | 20 |
| gradient_clip | max_norm=5.0 |

### Sparsity Targets

| Metric | Target |
|--------|--------|
| Active TFs per cell | 5–20 non-zero α_TF |
| Targets per TF | 10–50 genes with \|W_reg[t,g]\| > 0.1 |
| W_reg density | 1%–5% |
| Peak-gene density | ~0.5–1% of G×P |

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Structural prior too sparse | Lower threshold on M_combined; increase motif sensitivity |
| Latent collapse | β-annealing; monitor KL; increase reconstruction weight |
| TF activity not interpretable | Auxiliary TF expression correlation loss (λ_TF_corr) |
| KO validation fails | Check TF has sufficient targets in M_mask; relax prior |
| Memory overflow | Sparse peak-gene/TF-peak storage; reduce batch; gradient checkpointing |
| Gradient instability | Clamp logvar [-10,10]; gradient clipping 5.0 |
| W_act ≈ W_rep everywhere | Mutual exclusion loss λ_excl · mean(W_act ⊙ W_rep) |
| Phase 2 destabilizes latent | Default: freeze modules 1-3; monitor reconstruction loss |
| Peak-gene learns trivially | L2 regularization; clamp to [0,1]; monitor W_peak-gene sparsity |
| In-silico ChIP noisy | Require ≥50 metacells; strict percentile filtering; validate with external ChIP |
