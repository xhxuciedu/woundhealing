# Skin Wound Healing Analysis

Single-cell RNA-seq analysis pipeline for wound healing studies, including data integration, batch correction, cell cycle analysis, RNA velocity, and cell-cell communication analysis.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd woundhealing
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Additional Dependencies

Some packages may require additional system dependencies:

- **scvi-tools**: Requires PyTorch (automatically installed as a dependency)
- **sccoda**: May require JAX/TensorFlow (automatically installed as a dependency)
- **pyscenic**: Requires additional dependencies for motif analysis (see [pyscenic documentation](https://pyscenic.readthedocs.io/))

For optimal performance with GPU-accelerated methods (e.g., scVI), ensure you have:
- CUDA-compatible GPU (optional, for GPU acceleration)
- Appropriate CUDA drivers installed

## Usage

The analysis pipeline consists of several Jupyter notebooks and Python scripts:

### Notebooks
- **Data Integration**: `code/notebooks/integrate_skin_data.ipynb`
- **Cell Cycle Analysis**: `code/notebooks/cell_cycle_analysis_fibroblasts.ipynb`
- **Fibroblast Heterogeneity**: `code/notebooks/fibroblast_heterogeneity_drivers.ipynb`
- **Functionality Analysis**: `code/notebooks/functionality_analysis_fibroblasts.ipynb`
- **RNA Velocity**: `code/notebooks/rna_velocity_and_paga.ipynb`
- **SCENIC Analysis**: `code/notebooks/scenic_analysis_fibroblasts.ipynb`
- **Subclustering Analysis**: `code/notebooks/subclustering_analysis.ipynb`
- **Compositional Analysis**: `code/notebooks/unwounded_compositional_analysis.ipynb`

### Python Scripts
- **Batch Correction**: `code/batch_correction_integration.py` and `code/example_batch_correction.py`

### R Scripts
- **CellChat Analysis**: `code/cellchat_analysis_fibroblasts_and_immune_cells.R`
- **Differential Expression**: `code/de_analysis_fibroblast_functionality_genes_mast.R`

## Project Structure

```
woundhealing/
├── code/
│   ├── notebooks/     # Jupyter notebooks for analysis
│   ├── *.py           # Python scripts
│   └── *.R            # R scripts
├── data/              # Input data files (.h5ad, .csv)
├── output/            # Analysis results and figures
└── requirements.txt
```