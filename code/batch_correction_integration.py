"""
Batch Effect Correction and Data Integration for Single-Cell RNA-seq Data

This script performs batch effect correction on single-cell RNA-seq data using
various integration methods. It takes an AnnData object and a sample_id column
as input, and returns an AnnData object with batch-corrected embeddings stored
in obsm for downstream visualization (e.g., UMAP).

Supported methods:
- Harmony: Fast, linear correction method
- Scanorama: Mutual nearest neighbors-based integration
- scVI: Deep learning-based variational inference
- Combat: Empirical Bayes batch correction
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Optional, Literal
import warnings
warnings.filterwarnings('ignore')

# Set scanpy settings
sc.settings.verbosity = 2


def batch_correct_harmony(
    adata: ad.AnnData,
    sample_id_col: str,
    basis: str = 'X_pca',
    n_comps: int = 50,
    **harmony_kwargs
) -> ad.AnnData:
    """
    Perform batch correction using Harmony.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    sample_id_col : str
        Column name in adata.obs containing sample/batch IDs
    basis : str, default 'X_pca'
        Key in adata.obsm containing the embedding to correct
    n_comps : int, default 50
        Number of principal components to use if PCA needs to be computed
    **harmony_kwargs
        Additional arguments passed to harmony.run_harmony
    
    Returns
    -------
    AnnData
        AnnData object with corrected embedding in adata.obsm['X_harmony']
    """
    try:
        import harmonypy as hm
    except ImportError:
        raise ImportError(
            "harmonypy is required for Harmony integration. "
            "Install with: pip install harmonypy"
        )
    
    # Ensure PCA exists
    if basis not in adata.obsm.keys():
        if 'X_pca' not in adata.obsm.keys():
            print(f"Computing PCA with {n_comps} components...")
            sc.tl.pca(adata, n_comps=n_comps)
            basis = 'X_pca'
        else:
            basis = 'X_pca'
    
    # Run Harmony
    print("Running Harmony batch correction...")
    ho = hm.run_harmony(
        adata.obsm[basis],
        adata.obs,
        sample_id_col,
        **harmony_kwargs
    )
    
    # Store corrected embedding
    adata.obsm['X_harmony'] = ho.Z_corr.T
    
    print(f"Harmony correction complete. Embedding stored in adata.obsm['X_harmony']")
    return adata


def batch_correct_scanorama(
    adata: ad.AnnData,
    sample_id_col: str,
    n_comps: int = 50,
    **scanorama_kwargs
) -> ad.AnnData:
    """
    Perform batch correction using Scanorama.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    sample_id_col : str
        Column name in adata.obs containing sample/batch IDs
    n_comps : int, default 50
        Number of components for the corrected embedding
    **scanorama_kwargs
        Additional arguments passed to scanorama.correct_scanpy
    
    Returns
    -------
    AnnData
        AnnData object with corrected embedding in adata.obsm['X_scanorama']
    """
    try:
        import scanorama
    except ImportError:
        raise ImportError(
            "scanorama is required for Scanorama integration. "
            "Install with: pip install scanorama"
        )
    
    print("Running Scanorama batch correction...")
    
    # Split data by batch
    batches = adata.obs[sample_id_col].unique()
    adatas = [adata[adata.obs[sample_id_col] == batch].copy() for batch in batches]
    
    # Run Scanorama
    integrated = scanorama.correct_scanpy(
        adatas,
        return_dimred=True,
        dimred=n_comps,
        **scanorama_kwargs
    )
    
    # Concatenate results
    adata_corrected = ad.concat(integrated, join='outer', index_unique=None)
    
    # Ensure same cell order as input
    adata_corrected = adata_corrected[adata.obs_names]
    
    # Store corrected embedding
    if 'X_scanorama' in adata_corrected.obsm:
        adata.obsm['X_scanorama'] = adata_corrected.obsm['X_scanorama']
    else:
        # If scanorama doesn't store in obsm, compute PCA on corrected data
        sc.tl.pca(adata_corrected, n_comps=n_comps)
        adata.obsm['X_scanorama'] = adata_corrected.obsm['X_pca']
    
    print(f"Scanorama correction complete. Embedding stored in adata.obsm['X_scanorama']")
    return adata


def batch_correct_scvi(
    adata: ad.AnnData,
    sample_id_col: str,
    n_latent: int = 30,
    n_layers: int = 2,
    n_hidden: int = 128,
    max_epochs: int = 400,
    use_gpu: bool = False,
    **scvi_kwargs
) -> ad.AnnData:
    """
    Perform batch correction using scVI (scvi-tools).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    sample_id_col : str
        Column name in adata.obs containing sample/batch IDs
    n_latent : int, default 30
        Dimensionality of the latent space
    n_layers : int, default 2
        Number of layers in the encoder/decoder
    n_hidden : int, default 128
        Number of hidden units per layer
    max_epochs : int, default 400
        Maximum number of training epochs
    use_gpu : bool, default False
        Whether to use GPU for training
    **scvi_kwargs
        Additional arguments passed to scvi.model.SCVI
    
    Returns
    -------
    AnnData
        AnnData object with corrected embedding in adata.obsm['X_scvi']
    """
    try:
        import scvi
    except ImportError:
        raise ImportError(
            "scvi-tools is required for scVI integration. "
            "Install with: pip install scvi-tools"
        )
    
    print("Preparing data for scVI...")
    
    # Prepare data for scVI
    adata_scvi = adata.copy()
    
    # Ensure raw counts are available
    if adata_scvi.raw is None:
        print("Warning: No raw counts found. Using .X as counts.")
        adata_scvi.layers['counts'] = adata_scvi.X.copy()
    else:
        adata_scvi.layers['counts'] = adata_scvi.raw.X.copy()
    
    # Setup scVI
    scvi.model.SCVI.setup_anndata(
        adata_scvi,
        layer='counts',
        batch_key=sample_id_col,
        **scvi_kwargs
    )
    
    print("Training scVI model...")
    model = scvi.model.SCVI(
        adata_scvi,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        **scvi_kwargs
    )
    
    model.train(
        max_epochs=max_epochs,
        use_gpu=use_gpu,
        plan_kwargs={'lr': 1e-3}
    )
    
    # Get latent representation
    print("Extracting batch-corrected embedding...")
    adata.obsm['X_scvi'] = model.get_latent_representation(adata_scvi)
    
    print(f"scVI correction complete. Embedding stored in adata.obsm['X_scvi']")
    return adata


def batch_correct_combat(
    adata: ad.AnnData,
    sample_id_col: str,
    n_comps: int = 50,
    **combat_kwargs
) -> ad.AnnData:
    """
    Perform batch correction using Combat (empirical Bayes).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    sample_id_col : str
        Column name in adata.obs containing sample/batch IDs
    n_comps : int, default 50
        Number of principal components after correction
    **combat_kwargs
        Additional arguments passed to scanpy.pp.combat
    
    Returns
    -------
    AnnData
        AnnData object with corrected embedding in adata.obsm['X_combat']
    """
    print("Running Combat batch correction...")
    
    # Make a copy for Combat
    adata_combat = adata.copy()
    
    # Ensure we have normalized, log-transformed data
    if 'log1p' not in adata_combat.uns.keys():
        print("Normalizing and log-transforming data...")
        sc.pp.normalize_total(adata_combat, target_sum=1e4)
        sc.pp.log1p(adata_combat)
    
    # Run Combat
    sc.pp.combat(adata_combat, key=sample_id_col, **combat_kwargs)
    
    # Compute PCA on corrected data
    sc.tl.pca(adata_combat, n_comps=n_comps)
    
    # Store corrected embedding
    adata.obsm['X_combat'] = adata_combat.obsm['X_pca']
    
    print(f"Combat correction complete. Embedding stored in adata.obsm['X_combat']")
    return adata


def batch_correct_integrate(
    adata: ad.AnnData,
    sample_id_col: str,
    method: Literal['harmony', 'scanorama', 'scvi', 'combat'] = 'harmony',
    basis: Optional[str] = None,
    n_comps: int = 50,
    **method_kwargs
) -> ad.AnnData:
    """
    Main function for batch effect correction and data integration.
    
    This function performs batch correction on single-cell RNA-seq data using
    the specified method. The corrected embedding is stored in adata.obsm for
    downstream visualization and analysis.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell RNA-seq data
    sample_id_col : str
        Column name in adata.obs containing sample/batch IDs to correct for
    method : {'harmony', 'scanorama', 'scvi', 'combat'}, default 'harmony'
        Batch correction method to use:
        - 'harmony': Fast linear correction (recommended for most cases)
        - 'scanorama': Mutual nearest neighbors-based integration
        - 'scvi': Deep learning-based variational inference (best for complex cases)
        - 'combat': Empirical Bayes correction (classical method)
    basis : str, optional
        Key in adata.obsm containing the embedding to correct (for Harmony).
        If None, will use 'X_pca' or compute PCA if needed.
    n_comps : int, default 50
        Number of components for the corrected embedding
    **method_kwargs
        Additional arguments passed to the specific batch correction method
    
    Returns
    -------
    AnnData
        AnnData object with batch-corrected embedding stored in:
        - adata.obsm['X_harmony'] for method='harmony'
        - adata.obsm['X_scanorama'] for method='scanorama'
        - adata.obsm['X_scvi'] for method='scvi'
        - adata.obsm['X_combat'] for method='combat'
    
    Examples
    --------
    >>> import scanpy as sc
    >>> import anndata as ad
    >>> 
    >>> # Load your data
    >>> adata = sc.read_h5ad('data.h5ad')
    >>> 
    >>> # Perform batch correction
    >>> adata_corrected = batch_correct_integrate(
    ...     adata,
    ...     sample_id_col='sample',
    ...     method='harmony',
    ...     n_comps=50
    ... )
    >>> 
    >>> # Compute UMAP on corrected embedding
    >>> sc.pp.neighbors(adata_corrected, use_rep='X_harmony')
    >>> sc.tl.umap(adata_corrected)
    >>> sc.pl.umap(adata_corrected, color='sample')
    """
    
    # Validate inputs
    if sample_id_col not in adata.obs.columns:
        raise ValueError(
            f"Column '{sample_id_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    
    if method not in ['harmony', 'scanorama', 'scvi', 'combat']:
        raise ValueError(
            f"Method must be one of ['harmony', 'scanorama', 'scvi', 'combat'], "
            f"got '{method}'"
        )
    
    # Check for multiple batches
    n_batches = adata.obs[sample_id_col].nunique()
    if n_batches < 2:
        print(f"Warning: Only {n_batches} batch(es) found. Batch correction may not be necessary.")
    
    print(f"Running batch correction with method: {method}")
    print(f"Number of batches: {n_batches}")
    print(f"Number of cells: {adata.n_obs}")
    print(f"Number of genes: {adata.n_vars}")
    
    # Run the specified method
    if method == 'harmony':
        if basis is None:
            basis = 'X_pca'
        return batch_correct_harmony(
            adata,
            sample_id_col,
            basis=basis,
            n_comps=n_comps,
            **method_kwargs
        )
    elif method == 'scanorama':
        return batch_correct_scanorama(
            adata,
            sample_id_col,
            n_comps=n_comps,
            **method_kwargs
        )
    elif method == 'scvi':
        return batch_correct_scvi(
            adata,
            sample_id_col,
            n_latent=n_comps,
            **method_kwargs
        )
    elif method == 'combat':
        return batch_correct_combat(
            adata,
            sample_id_col,
            n_comps=n_comps,
            **method_kwargs
        )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch correction and integration for single-cell RNA-seq data'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input h5ad file'
    )
    parser.add_argument(
        '--sample-id-col',
        type=str,
        required=True,
        help='Column name in obs containing sample/batch IDs'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['harmony', 'scanorama', 'scvi', 'combat'],
        default='harmony',
        help='Batch correction method to use (default: harmony)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Path to output h5ad file (default: input_file with _corrected suffix)'
    )
    parser.add_argument(
        '--n-comps',
        type=int,
        default=50,
        help='Number of components for corrected embedding (default: 50)'
    )
    parser.add_argument(
        '--basis',
        type=str,
        help='Basis embedding to correct (for Harmony, default: X_pca)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    adata = sc.read_h5ad(args.input_file)
    
    # Run batch correction
    adata_corrected = batch_correct_integrate(
        adata,
        sample_id_col=args.sample_id_col,
        method=args.method,
        basis=args.basis,
        n_comps=args.n_comps
    )
    
    # Save output
    if args.output_file is None:
        output_file = args.input_file.replace('.h5ad', '_corrected.h5ad')
    else:
        output_file = args.output_file
    
    print(f"Saving corrected data to {output_file}...")
    adata_corrected.write_h5ad(output_file)
    print("Done!")
