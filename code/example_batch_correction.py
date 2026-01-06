"""
Example script demonstrating how to use batch_correction_integration.py

This script shows how to:
1. Load single-cell RNA-seq data
2. Perform batch correction
3. Visualize the results with UMAP
"""

import scanpy as sc
import matplotlib.pyplot as plt
from batch_correction_integration import batch_correct_integrate

# Set scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white', fontsize=12)

# ============================================================================
# Load your data
# ============================================================================
print("Loading data...")
# Replace with your actual data path
data_path = './data/integratedskindata.h5ad'
adata = sc.read_h5ad(data_path)

print(f"Loaded data: {adata.n_obs} cells, {adata.n_vars} genes")
print(f"Sample IDs: {adata.obs['sample'].unique() if 'sample' in adata.obs.columns else 'N/A'}")

# ============================================================================
# Perform batch correction
# ============================================================================
print("\n" + "="*60)
print("Performing batch correction...")
print("="*60)

# Specify the column name in adata.obs that contains sample/batch IDs
sample_id_col = 'sub_sample'  # Change this to match your data

# Perform batch correction using Harmony (default, fast and effective)
adata = batch_correct_integrate(
    adata,
    sample_id_col=sample_id_col,
    method='harmony',  # Options: 'harmony', 'scanorama', 'scvi', 'combat'
    n_comps=50
)

# Alternative methods (uncomment to use):
# 
# # Scanorama
# adata = batch_correct_integrate(
#     adata,
#     sample_id_col=sample_id_col,
#     method='scanorama',
#     n_comps=50
# )
#
# # scVI (requires more time but often better for complex cases)
# adata = batch_correct_integrate(
#     adata,
#     sample_id_col=sample_id_col,
#     method='scvi',
#     n_comps=30,
#     max_epochs=400
# )
#
# # Combat
# adata = batch_correct_integrate(
#     adata,
#     sample_id_col=sample_id_col,
#     method='combat',
#     n_comps=50
# )

# ============================================================================
# Compute neighbors and UMAP using the corrected embedding
# ============================================================================
print("\n" + "="*60)
print("Computing neighbors and UMAP...")
print("="*60)

# Determine which embedding to use based on the method
# Harmony stores in 'X_harmony', Scanorama in 'X_scanorama', etc.
embedding_key = 'X_harmony'  # Change based on method used above

# Compute neighborhood graph using corrected embedding
sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=15, n_pcs=50)

# Compute UMAP
sc.tl.umap(adata)

# ============================================================================
# Visualize results
# ============================================================================
print("\n" + "="*60)
print("Visualizing results...")
print("="*60)

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot UMAP colored by sample (before correction would show batch effects)
sc.pl.umap(
    adata,
    color=sample_id_col,
    ax=axes[0],
    title='UMAP colored by sample',
    show=False,
    frameon=False
)

# Plot UMAP colored by cell type (leiden clusters)
if 'leiden' in adata.obs.columns:
    sc.pl.umap(
        adata,
        color='leiden',
        ax=axes[1],
        title='UMAP colored by leiden clusters',
        show=False,
        frameon=False
    )
else:
    # If no leiden, just show sample again or another available column
    sc.pl.umap(
        adata,
        color=sample_id_col,
        ax=axes[1],
        title='UMAP (alternative view)',
        show=False,
        frameon=False
    )

plt.tight_layout()
plt.savefig('../output/batch_correction_umap.png', dpi=300, bbox_inches='tight')
print("Saved UMAP plot to ../output/batch_correction_umap.png")

# ============================================================================
# Optional: Save the corrected data
# ============================================================================
output_path = '../data/integratedskindata_corrected.h5ad'
print(f"\nSaving corrected data to {output_path}...")
adata.write_h5ad(output_path)
print("Done!")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Corrected embedding stored in: adata.obsm['{embedding_key}']")
print(f"Shape of corrected embedding: {adata.obsm[embedding_key].shape}")
print(f"Number of samples: {adata.obs[sample_id_col].nunique()}")
print(f"Total cells: {adata.n_obs}")
