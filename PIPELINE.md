# SICA-Latent Data Pipeline

## Step 1: Load Raw snMultiome Data
- Read the 10x h5 file (single combined file containing both modalities)
- Split into RNA and ATAC by feature type (`Gene Expression` vs `Peaks`)
- Make variable names unique
- Verify shapes look correct (cells × genes, cells × peaks)

## Step 2: Basic Quality Control
- Remove cells with zero counts
- Remove mitochondrial genes
- Align cells so RNA and ATAC have the same cells in the same order

## Step 3: Convert to Tensors
- Convert RNA count matrix → torch tensor [N, G]
- Convert ATAC count matrix → torch tensor [N, P]
- Compute log library sizes: log(sum of counts per cell) → [N]

## Step 4: Extract Transcription Factors
- Load CIS-BP motif database (.meme file, bundled in scDoRI)
- Parse TF names from motif database
- Match TF names to gene names in RNA data (case-insensitive for mouse)
- Record TF indices in RNA matrix → tf_idx [T]

## Step 5: Compute W_mask and W_init (Peak-Gene Distance Matrix)
- Use scDoRI's `compute_gene_peak_distance_matrix()`
- Requires: peak coordinates (from ATAC var names) + gene coordinates (from GTF)
- Output W_init [G, P]: exponential distance decay weights
- Output W_mask [G, P]: binary mask, 1 if peak within 150kb of gene

## Step 6: Compute Motif Scores
- Use scDoRI's `compute_motif_scores()`
- Requires: peak BED file + genome FASTA + CIS-BP .meme file
- Output: motif score matrix [P, T] — how strongly each TF motif appears in each peak
- Multiply by ATAC counts to get per-cell motif scores [N, T]

## Step 7: Compute In-Silico ChIP-seq Matrices
- Create pseudobulk metacells using scDoRI's `create_metacells()`
- Use scDoRI's `compute_in_silico_chipseq()`
- Requires: pseudobulk ATAC + pseudobulk TF expression + motif scores
- Output W_TF_peak_act [P, T]: activating TF-peak binding scores
- Output W_TF_peak_rep [P, T]: repressive TF-peak binding scores

## Step 8: Feed into Model
- Pass RNA tensor, ATAC tensor, log library sizes into Encoder
- Pass W_mask, W_init into Module 2
- Pass motif scores into Module 3
- Pass W_TF_peak_act, W_TF_peak_rep into Module 4 (Phase 2)
