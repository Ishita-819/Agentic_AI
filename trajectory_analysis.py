# ===============================================================
# Trajectory Inference and Pseudotime Ordering (Astrocytes)
# ===============================================================

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
DATA_PATH = "resAA/agentic_bio/adata_agentic_bio.h5ad"  # Input from clustering step
OUTDIR = "resAA/trajectory_inference"
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
logging.info("📦 Loading clustered and annotated AnnData object...")
adata = sc.read_h5ad(DATA_PATH)
logging.info(f"✅ Loaded AnnData: {adata.shape}")

# ---------------------------------------------------------------
# FILTER FOR ASTROCYTES (Clusters 13 & 19)
# ---------------------------------------------------------------
astro = adata[adata.obs["cluster"].isin(["13", "19"])].copy()
logging.info(f"🧠 Astrocyte subset shape: {astro.shape}")

# Optional: filter for AD-positive cells (A+T+)
if "atscore" in astro.obs.columns:
    astro = astro[astro.obs["atscore"] == "A+T+"].copy()
    logging.info(f"📊 Filtered A+T+ astrocytes: {astro.shape}")

# ---------------------------------------------------------------
# NORMALIZATION & HVG SELECTION
# ---------------------------------------------------------------
if np.max(astro.X) > 50:
    logging.info("🧮 Detected raw counts — normalizing and log-transforming.")
    sc.pp.normalize_total(astro, target_sum=1e4)
    sc.pp.log1p(astro)

sc.pp.highly_variable_genes(astro, n_top_genes=2000, subset=True)
sc.pp.scale(astro, max_value=10)

# ---------------------------------------------------------------
# PCA & NEIGHBOR GRAPH
# ---------------------------------------------------------------
logging.info("🔍 Computing PCA and neighborhood graph...")
sc.tl.pca(astro, svd_solver='arpack')
sc.pp.neighbors(astro, n_neighbors=15, n_pcs=30)

# ---------------------------------------------------------------
# UMAP AND DPT TRAJECTORY
# ---------------------------------------------------------------
logging.info("📈 Computing UMAP and pseudotime using DPT...")
sc.tl.umap(astro)
sc.tl.diffmap(astro)

# Set root cell
root_cell = np.random.choice(astro.obs_names)
astro.uns['iroot'] = astro.obs_names.get_loc(root_cell)
logging.info(f"🪴 Using {root_cell} as root cell for pseudotime ordering.")

# Compute pseudotime
sc.tl.dpt(astro, n_dcs=10)

# ---------------------------------------------------------------
# VISUALIZE PSEUDOTIME
# ---------------------------------------------------------------
sc.pl.umap(
    astro,
    color=["dpt_pseudotime", "cluster"],
    cmap="viridis",
    save="_astrocyte_pseudotime.png",
    show=False
)
logging.info("🖼️ UMAP with pseudotime saved.")

# ---------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------
astro.obs["pseudotime"] = astro.obs["dpt_pseudotime"]
astro.write(f"{OUTDIR}/astrocyte_pseudotime.h5ad")
astro.obs.to_csv(f"{OUTDIR}/astrocyte_pseudotime_metadata.tsv", sep="\t")
logging.info("💾 Saved pseudotime trajectory results.")

# ---------------------------------------------------------------
# FIND GENES CORRELATED WITH PSEUDOTIME
# ---------------------------------------------------------------
logging.info("🔬 Identifying genes correlated with pseudotime progression...")

expr = pd.DataFrame(
    astro.X.toarray() if hasattr(astro.X, "toarray") else astro.X,
    columns=astro.var_names,
    index=astro.obs_names
)

pseudotime = astro.obs["pseudotime"]
corrs = expr.corrwith(pseudotime, axis=0)
corr_df = pd.DataFrame({"gene": corrs.index, "correlation": corrs.values})
corr_df["abs_corr"] = np.abs(corr_df["correlation"])

top_genes = corr_df.sort_values("abs_corr", ascending=False).head(50)
top_genes.to_csv(f"{OUTDIR}/pseudotime_correlated_genes.tsv", sep="\t", index=False)
logging.info("✅ Saved top pseudotime-correlated genes.")

# ---------------------------------------------------------------
# PSEUDOTIME HEATMAP VISUALIZATION
# ---------------------------------------------------------------
logging.info("🎨 Generating heatmap of top pseudotime-correlated genes...")

# Order cells by pseudotime
astro_ordered = astro[astro.obs.sort_values("pseudotime").index, :]

# Extract expression of top genes
heatmap_genes = top_genes["gene"].tolist()
heatmap_data = pd.DataFrame(
    astro_ordered[:, heatmap_genes].X.toarray() if hasattr(astro.X, "toarray") else astro_ordered[:, heatmap_genes].X,
    columns=heatmap_genes,
    index=astro_ordered.obs_names
)

# Z-score normalize each gene for better contrast
heatmap_data = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()

# Plot
plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_data.T,
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Z-score"},
    yticklabels=True
)
plt.title("Top 50 Pseudotime-Correlated Genes (Astrocytes)")
plt.xlabel("Cells Ordered by Pseudotime")
plt.ylabel("Genes")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/pseudotime_top_genes_heatmap.png", dpi=300)
plt.close()

logging.info("🔥 Heatmap saved: pseudotime_top_genes_heatmap.png")
