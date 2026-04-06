import os
import argparse
import urllib.request
import gzip
import shutil
import ssl
import numpy as np
import scanpy as sc
import pandas as pd
import torch
from scipy.sparse import issparse
from scdori.pp.motif_scanning import load_motif_database, compute_motif_scores
from scdori.pp.correlation import compute_in_silico_chipseq
from scdori.pp.metacells import create_metacells
from scdori.pp.gene_selection import load_gtf, compute_hvgs_and_tfs

GENOME_URLS = {
    "mm10": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz",
    "mm39": "https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.fa.gz",
    "hg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
    "hg19": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz",
}
MOUSE_CHROMS = {f'chr{i}' for i in list(range(1, 20)) + ['X', 'Y', 'M']}
HUMAN_CHROMS = {f'chr{i}' for i in list(range(1, 23)) + ['X', 'Y', 'M']}
GENOME_CHROMS = {
    "mm10": MOUSE_CHROMS, "mm39": MOUSE_CHROMS,
    "hg38": HUMAN_CHROMS, "hg19": HUMAN_CHROMS,
}


# ── Genome FASTA ──────────────────────────────────────────────────────────────

def _download_fasta(genome, fasta_dir):
    if genome not in GENOME_URLS:
        raise ValueError(f"Unknown genome '{genome}'. Available: {list(GENOME_URLS.keys())}")
    fasta_path = os.path.abspath(os.path.join(fasta_dir, f'{genome}.fa'))
    if os.path.exists(fasta_path):
        print(f"FASTA already present: {fasta_path}")
        return fasta_path
    os.makedirs(fasta_dir, exist_ok=True)
    gz_path = fasta_path + '.gz'
    print(f"Downloading {genome} genome (~900 MB compressed) ...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(GENOME_URLS[genome], context=ctx) as response, \
         open(gz_path, 'wb') as f_out:
        shutil.copyfileobj(response, f_out)
    print("Decompressing ...")
    with gzip.open(gz_path, 'rb') as f_in, open(fasta_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    print(f"Done: {fasta_path}")
    return fasta_path


# ── Data loading ──────────────────────────────────────────────────────────────

def LoadMultiomeData(h5_path, genome, fasta_dir):
    data = sc.read_10x_h5(h5_path, gex_only=False)
    data.var_names_make_unique()
    print(data.var['feature_types'].value_counts())
    rna  = data[:, data.var['feature_types'] == 'Gene Expression']
    atac = data[:, data.var['feature_types'] == 'Peaks']
    sample_name = os.path.basename(h5_path).replace('.h5', '')
    rna.obs['sample']  = sample_name
    atac.obs['sample'] = sample_name
    print(f"RNA:  {rna.shape}")
    print(f"ATAC: {atac.shape}")
    fasta_path = _download_fasta(genome, fasta_dir)
    return rna, atac, fasta_path


# ── Peak filtering ────────────────────────────────────────────────────────────

def ExtractPeakCoordinates(atac, bed_path, genome):
    chroms = GENOME_CHROMS[genome]
    peak_coords = atac.var['interval'].str.extract(r'(.+):(\d+)-(\d+)')
    peak_coords.columns = ['chr', 'start', 'end']
    peak_coords[['start', 'end']] = peak_coords[['start', 'end']].astype(int)
    peak_coords['mid'] = (peak_coords['start'] + peak_coords['end']) // 2

    mask = peak_coords['chr'].isin(chroms)
    peak_coords = peak_coords[mask]
    print(f"Kept {len(peak_coords)}/{len(mask)} peaks on canonical chromosomes")

    bed = peak_coords[['chr', 'start', 'end']].copy()
    bed['name'] = peak_coords.index
    bed.to_csv(bed_path, sep='\t', header=False, index=False)
    print(f"Wrote {len(bed)} peaks to {bed_path}")

    atac_filtered = atac[:, mask.values].copy()
    # Store coordinates in var for downstream use
    atac_filtered.var['chr']       = peak_coords['chr'].values
    atac_filtered.var['start']     = peak_coords['start'].values
    atac_filtered.var['end']       = peak_coords['end'].values
    atac_filtered.var['mid']       = peak_coords['mid'].values
    atac_filtered.var['peak_name'] = atac_filtered.var_names

    return peak_coords, atac_filtered


# ── Gene filtering ────────────────────────────────────────────────────────────

def FilterGenes(rna, tf_names_all, gtf_path, num_genes, num_tfs):
    """Filter RNA to HVGs + TFs using protein-coding genes from GTF."""
    from scdori.pp.gene_selection import filter_protein_coding_genes
    print("Loading GTF ...")
    gtf_df = load_gtf(gtf_path)
    rna_pc = filter_protein_coding_genes(rna, gtf_df)
    print(f"Protein-coding genes: {rna_pc.shape[1]}")
    rna_filtered, final_genes, final_tfs = compute_hvgs_and_tfs(
        rna_pc, tf_names_all, num_genes=num_genes, num_tfs=num_tfs
    )
    print(f"Selected {len(final_genes)} HVGs + {len(final_tfs)} TFs = {rna_filtered.shape[1]} genes")
    return rna_filtered, final_genes, final_tfs, gtf_df


# ── Peak-gene mask ────────────────────────────────────────────────────────────

def _prepare_gene_coords(gtf_df, gene_names):
    """Extract gene coordinates from GTF for the given gene list."""
    gtf_genes = gtf_df[gtf_df['feature'] == 'gene'].drop_duplicates('gene_name')
    gtf_genes = gtf_genes.set_index('gene_name')
    available = [g for g in gene_names if g in gtf_genes.index]
    gtf_genes = gtf_genes.loc[available]
    coords = pd.DataFrame({
        'chr_gene': gtf_genes['seqname'].values,
        'start':    gtf_genes['start'].values,
        'end':      gtf_genes['end'].values,
        'strand':   gtf_genes['strand'].values,
    }, index=gtf_genes.index)
    return coords


def ComputePeakGeneMask(atac, gene_names, gtf_df, window=250_000):
    """
    Build W_mask [P, G] — 1 where peak is within `window` bp of gene TSS/body.
    Build W_init [P, G] — same as W_mask (learned from this starting point).
    Both are float32 tensors.
    """
    print(f"Computing peak-gene mask ({atac.shape[1]} peaks x {len(gene_names)} genes, window={window//1000}kb) ...")
    gene_coords = _prepare_gene_coords(gtf_df, gene_names)

    peak_chr = atac.var['chr'].values
    peak_mid = atac.var['mid'].values.astype(float)
    n_peaks  = len(peak_chr)
    n_genes  = len(gene_names)

    mask = np.zeros((n_peaks, n_genes), dtype=np.float32)

    for g_idx, gene in enumerate(gene_names):
        if gene not in gene_coords.index:
            continue
        row      = gene_coords.loc[gene]
        g_chr    = row['chr_gene']
        g_start  = float(row['start'])
        g_end    = float(row['end'])
        g_strand = row['strand']

        same_chr = peak_chr == g_chr

        if g_strand == '+':
            within = (peak_mid >= (g_start - 5_000)) & (peak_mid <= g_end)
        else:
            within = (peak_mid >= g_start) & (peak_mid <= (g_end + 5_000))

        dist = np.where(within, 0,
                        np.minimum(np.abs(peak_mid - g_start),
                                   np.abs(peak_mid - g_end)))
        in_window = same_chr & (dist <= window)
        mask[:, g_idx] = in_window.astype(np.float32)

        if (g_idx + 1) % 500 == 0:
            print(f"  ... {g_idx + 1}/{n_genes} genes")

    n_links = int(mask.sum())
    print(f"Peak-gene mask: {n_links} peak-gene links ({n_links / (n_peaks * n_genes) * 100:.2f}% density)")
    # Module2 expects [G, P] so it can do p_norm @ W.T = [N,P] @ [P,G] → [N,G]
    W_mask = torch.tensor(mask.T, dtype=torch.float32)
    W_init = W_mask.clone()
    return W_mask, W_init


# ── Library sizes ─────────────────────────────────────────────────────────────

def ComputeLogLibrarySizes(rna_tensor, atac_tensor):
    log_lib_rna  = torch.log(rna_tensor.sum(dim=1,  keepdim=True) + 1)
    log_lib_atac = torch.log(atac_tensor.sum(dim=1, keepdim=True) + 1)
    return log_lib_rna, log_lib_atac


# ── TF index ──────────────────────────────────────────────────────────────────

def ComputeTFIndex(gene_names, tf_names):
    """Indices of TF genes within the gene list."""
    gene_list = list(gene_names)
    tf_set    = set(tf_names)
    tf_idx    = torch.tensor([i for i, g in enumerate(gene_list) if g in tf_set], dtype=torch.long)
    print(f"tf_idx: {len(tf_idx)} TFs found in gene list")
    return tf_idx


# ── Save / load ───────────────────────────────────────────────────────────────

def save_outputs(out_dir, rna_tensor, atac_tensor,
                 log_lib_rna, log_lib_atac,
                 W_act, W_rep, df_motif_scores,
                 W_mask, W_init, tf_idx,
                 rna, atac):
    os.makedirs(out_dir, exist_ok=True)

    torch.save(rna_tensor,   os.path.join(out_dir, "rna_tensor.pt"))
    torch.save(atac_tensor,  os.path.join(out_dir, "atac_tensor.pt"))
    torch.save(log_lib_rna,  os.path.join(out_dir, "log_lib_rna.pt"))
    torch.save(log_lib_atac, os.path.join(out_dir, "log_lib_atac.pt"))
    torch.save(torch.tensor(W_act, dtype=torch.float32), os.path.join(out_dir, "W_act.pt"))
    torch.save(torch.tensor(W_rep, dtype=torch.float32), os.path.join(out_dir, "W_rep.pt"))
    torch.save(W_mask,  os.path.join(out_dir, "W_mask.pt"))
    torch.save(W_init,  os.path.join(out_dir, "W_init.pt"))
    torch.save(tf_idx,  os.path.join(out_dir, "tf_idx.pt"))

    df_motif_scores.to_parquet(os.path.join(out_dir, "motif_scores.parquet"))

    pd.Series(list(rna.var_names)).to_csv(
        os.path.join(out_dir, "gene_names.txt"), index=False, header=False)
    pd.Series(list(atac.var_names)).to_csv(
        os.path.join(out_dir, "peak_names.txt"), index=False, header=False)
    pd.Series(list(df_motif_scores.columns)).to_csv(
        os.path.join(out_dir, "tf_names.txt"), index=False, header=False)

    print(f"\nSaved to {out_dir}/")
    print(f"  rna_tensor.pt        {tuple(rna_tensor.shape)}")
    print(f"  atac_tensor.pt       {tuple(atac_tensor.shape)}")
    print(f"  log_lib_rna.pt       {tuple(log_lib_rna.shape)}")
    print(f"  log_lib_atac.pt      {tuple(log_lib_atac.shape)}")
    print(f"  W_act.pt             {W_act.shape}")
    print(f"  W_rep.pt             {W_rep.shape}")
    print(f"  W_mask.pt            {tuple(W_mask.shape)}")
    print(f"  W_init.pt            {tuple(W_init.shape)}")
    print(f"  tf_idx.pt            {tuple(tf_idx.shape)}")
    print(f"  motif_scores.parquet {df_motif_scores.shape}")
    print(f"  gene_names.txt       {len(rna.var_names)} genes")
    print(f"  peak_names.txt       {len(atac.var_names)} peaks")
    print(f"  tf_names.txt         {len(df_motif_scores.columns)} TFs")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main(args):
    out_dir  = args.out_dir
    bed_path = os.path.join(out_dir, "peaks.bed")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load raw data + FASTA
    rna, atac, fasta_path = LoadMultiomeData(args.input, args.genome, args.fasta_dir)

    # 2. Filter ATAC to canonical chromosomes
    _, atac = ExtractPeakCoordinates(atac, bed_path, args.genome)

    # 3. Load motif database (against all RNA genes to find all possible TFs)
    pwms_all, key_to_tf_all = load_motif_database(args.motif_db, list(rna.var_names))
    tf_names_all = list(set(key_to_tf_all.values()))
    print(f"Motif DB: {len(pwms_all)} TF motifs")

    # 4. Filter RNA to HVGs + TFs (requires GTF)
    if args.gtf:
        rna, final_genes, final_tfs, gtf_df = FilterGenes(
            rna, tf_names_all, args.gtf, args.num_genes, args.num_tfs
        )
        # Re-filter motif DB to TFs present in our gene list
        pwms, key_to_tf = load_motif_database(args.motif_db, list(rna.var_names))
    else:
        print("WARNING: --gtf not provided. Skipping HVG filtering and W_mask/W_init computation.")
        final_tfs = tf_names_all
        pwms, key_to_tf = pwms_all, key_to_tf_all
        gtf_df = None

    # 5. Raw tensors (on filtered genes)
    rna_dense  = rna.X.toarray()  if issparse(rna.X)  else rna.X
    atac_dense = atac.X.toarray() if issparse(atac.X) else atac.X
    rna_tensor  = torch.tensor(rna_dense,  dtype=torch.float32)
    atac_tensor = torch.tensor(atac_dense, dtype=torch.float32)
    print(f"RNA tensor:  {rna_tensor.shape}")
    print(f"ATAC tensor: {atac_tensor.shape}")

    # 6. Log library sizes
    log_lib_rna, log_lib_atac = ComputeLogLibrarySizes(rna_tensor, atac_tensor)

    # 7. Motif scores [P, T]
    print("Computing motif scores ...")
    df_motif_scores = compute_motif_scores(
        bed_file   = bed_path,
        fasta_file = fasta_path,
        pwms_sub   = pwms,
        key_to_tf  = key_to_tf,
        n_peaks    = atac.shape[1],
        window     = 500,
        threshold  = 1e-3,
    )
    print(f"Motif scores: {df_motif_scores.shape}")

    # 8. Metacells
    rna_meta, atac_meta = create_metacells(rna, atac)
    print(f"Metacells — RNA: {rna_meta.shape}  ATAC: {atac_meta.shape}")

    # 9. In-silico ChIP-seq [P, T]
    tf_names    = list(df_motif_scores.columns)
    rna_genes   = list(rna.var_names)
    tf_mask     = [g in set(tf_names) for g in rna_genes]
    rna_tf_mat  = rna_meta.X[:, tf_mask]
    W_act, W_rep = compute_in_silico_chipseq(atac_meta.X, rna_tf_mat, df_motif_scores)
    print(f"W_act: {W_act.shape}  W_rep: {W_rep.shape}")

    # 10. TF index
    tf_idx = ComputeTFIndex(rna.var_names, set(tf_names))

    # 11. Peak-gene mask [P, G]
    if gtf_df is not None:
        W_mask, W_init = ComputePeakGeneMask(
            atac, list(rna.var_names), gtf_df, window=args.peak_gene_window
        )
    else:
        print("WARNING: W_mask and W_init not computed (no GTF). Saving all-ones placeholders.")
        P, G = atac.shape[1], rna.shape[1]
        W_mask = torch.ones(G, P, dtype=torch.float32)
        W_init = torch.ones(G, P, dtype=torch.float32) * 0.01

    # 12. Save
    save_outputs(out_dir, rna_tensor, atac_tensor,
                 log_lib_rna, log_lib_atac,
                 W_act, W_rep, df_motif_scores,
                 W_mask, W_init, tf_idx,
                 rna, atac)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess multiome data for SICAD training")
    parser.add_argument("--input",     required=True,
                        help="Path to 10x .h5 file")
    parser.add_argument("--out-dir",   required=True,
                        help="Directory to write all preprocessed outputs")
    parser.add_argument("--genome",    default="mm10",
                        choices=list(GENOME_URLS.keys()),
                        help="Reference genome (default: mm10)")
    parser.add_argument("--motif-db",  default="scDoRI/assets/motif_database/cisbp_mouse.meme",
                        help="Path to CIS-BP .meme motif database")
    parser.add_argument("--fasta-dir", default="data",
                        help="Directory to store/find genome FASTA (default: data/)")
    parser.add_argument("--gtf",       default=None,
                        help="Path to genome GTF — required for HVG filtering and W_mask/W_init")
    parser.add_argument("--num-genes", type=int, default=4000,
                        help="Number of HVGs to select (default: 4000)")
    parser.add_argument("--num-tfs",   type=int, default=500,
                        help="Number of TFs to select (default: 500)")
    parser.add_argument("--peak-gene-window", type=int, default=250_000,
                        help="Max bp distance for peak-gene links (default: 250000)")
    args = parser.parse_args()
    main(args)
