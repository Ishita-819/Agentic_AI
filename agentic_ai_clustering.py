# ===============================================================
# Stage 4️⃣: Agentic AI Biological Interpretation (Groq LLM + Enrichment)
# ===============================================================

import os
import json
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from dotenv import load_dotenv
from groq import Groq  # ✅ Groq client
from scipy.stats import fisher_exact

# ---------- CONFIG ----------
DATA_PATH = "GSE243292_ADsnRNAseq_GEO.h5ad"
OUTDIR = "resAA/agentic_bio"
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

# ---------- API Setup ----------
load_dotenv()
client = Groq(api_key=os.getenv("groq_api_key"))  # ✅ Use Groq key from .env
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===============================================================
# 1️⃣ Known cell-type markers
# ===============================================================
MARKER_GENES = {
    "Excitatory Neurons": ["SLC17A7", "CAMK2A", "SATB2", "VGLUT1"],
    "Inhibitory Neurons": ["GAD1", "GAD2", "SST", "PVALB", "VIP"],
    "Astrocytes": ["GFAP", "AQP4", "SLC1A3", "ALDH1L1"],
    "Oligodendrocytes": ["MBP", "MOG", "PLP1", "MOBP"],

    "OPCs": ["PDGFRA", "CSPG4"],
    "Microglia": ["TREM2", "CX3CR1", "C1QA", "P2RY12"],
    "Endothelial": ["CLDN5", "FLT1", "VWF", "PECAM1"],
    "Pericytes": ["PDGFRB", "RGS5"]
}

# ===============================================================
# 2️⃣ Load AnnData
# ===============================================================
logging.info("🔍 Loading clustered data...")
adata = sc.read_h5ad(DATA_PATH)
print("Available columns in adata.obs:")
print(adata.obs.columns.tolist())
print(adata.obs.head())

logging.info(f"✅ Loaded AnnData object with shape: {adata.shape}")
"""
# Check cluster info
if 'leiden' not in adata.obs.columns and 'cluster' not in adata.obs.columns:
    logging.warning("⚠️ No clustering found — computing PCA, neighbors, Leiden, and UMAP.")
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)
    adata.obs['cluster'] = adata.obs['leiden'].astype(str)

    clustered_path = f"{OUTDIR}/adata_clustered.h5ad"
    adata.write(clustered_path)
    logging.info(f"✅ Clustering completed and saved to: {clustered_path}")
else:
    if 'cluster' not in adata.obs.columns:
        adata.obs['cluster'] = adata.obs['leiden'].astype(str)
    logging.info("✅ Clustering already present in AnnData object.")

"""
if 'cluster' not in adata.obs.columns:
    logging.warning("⚠️ No 'cluster' column found — using 'leiden' instead.")
    if 'leiden' not in adata.obs.columns:
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution=0.5)
    adata.obs['cluster'] = adata.obs['leiden'].astype(str)
    clustered_path = f"{OUTDIR}/adata_clustered.h5ad"
    adata.write(clustered_path)
    logging.info(f"✅ Clustering completed and saved to: {clustered_path}")

# ===============================================================
# 3️⃣ Log-normalize if raw counts detected
# ===============================================================
if np.max(adata.X) > 50:
    logging.info("🧮 Detected raw count data — applying log1p normalization.")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

# ===============================================================
# 4️⃣ Identify marker genes per cluster
# ===============================================================
logging.info("🔬 Identifying marker genes per cluster...")

if adata.obs['cluster'].nunique() < 2:
    logging.error("❌ Only one cluster detected — cannot compute differential expression.")
    marker_df = pd.DataFrame()
else:
    sc.tl.rank_genes_groups(adata, groupby='cluster', method='wilcoxon', n_genes=50)

    try:
        marker_df = sc.get.rank_genes_groups_df(adata, None)
    except Exception as e:
        logging.warning(f"⚠️ Using backup extraction for marker genes: {e}")
        marker_df = pd.DataFrame({
            "names": adata.uns['rank_genes_groups']['names'].flatten(),
            "scores": adata.uns['rank_genes_groups']['scores'].flatten(),
            "group": np.repeat(list(adata.uns['rank_genes_groups']['names'].dtype.names),
                               adata.uns['rank_genes_groups']['names'].shape[0])
        })

# Fix column naming if needed
if 'group' not in marker_df.columns:
    marker_df['group'] = None

# Summarize
cluster_summary = []
for cluster in sorted(adata.obs['cluster'].unique()):
    subset = marker_df[marker_df['group'] == str(cluster)]
    if subset.empty:
        continue
    top_genes = subset.sort_values("scores", ascending=False).head(10)["names"].tolist()
    cluster_summary.append(f"Cluster {cluster}: {', '.join(top_genes)}")

# ===============================================================
# 5️⃣ Agentic AI LLM-driven biological interpretation (Groq)
# ===============================================================
prompt = f"""
You are an expert genomics AI analyzing single-nucleus RNA-seq clusters from human brain (AD and Down Syndrome).
Each cluster represents a potential brain cell population.
Use the known marker gene dictionary below to predict the most likely cell type for each cluster.

Known markers: {json.dumps(MARKER_GENES)}
Top marker genes per cluster:
{chr(10).join(cluster_summary)}

Respond ONLY in valid JSON format as:
{{
  "0": "Excitatory Neuron",
  "1": "Microglia",
  "2": "Astrocyte",
  ...
}}
"""

logging.info("🤖 Sending cluster marker summary to Groq LLM for biological interpretation...")
try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # ✅ Use Groq’s Mixtral model
        messages=[
            {"role": "system", "content": "You are a bioinformatics AI assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    interpretation_text = response.choices[0].message.content
    celltype_map = json.loads(interpretation_text)
    logging.info("✅ Groq LLM successfully classified cell types for clusters.")
except Exception as e:
    logging.error(f"❌ Groq LLM request or parsing failed: {e}")
    celltype_map = {}

adata.obs['cell_type_pred'] = adata.obs['cluster'].map(celltype_map).fillna("Unknown")

# ===============================================================
# 6️⃣ Compute UMAP for visualization
# ===============================================================
if 'X_umap' not in adata.obsm.keys():
    logging.info("📈 Computing UMAP embedding for visualization...")
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

sc.pl.umap(
    adata,
    color=['cell_type_pred'],
    legend_loc='on data',
    title='Agentic AI Predicted Cell Types (Groq)',
    show=False
)
plt.savefig(f"{OUTDIR}/umap_predicted_celltypes.png", dpi=300)
plt.close()

# ===============================================================
# 7️⃣ Save report
# ===============================================================
report_path = f"{OUTDIR}/agentic_biological_report.txt"
with open(report_path, "w") as f:
    f.write("=== Agentic AI Biological Interpretation ===\n\n")
    f.write(f"Dataset: GSE243292 (AD + Down Syndrome snRNA-seq)\n")
    f.write(f"Clusters identified: {adata.obs['cluster'].nunique()}\n\n")
    f.write("Predicted Cell Type per Cluster:\n")
    f.write(json.dumps(celltype_map, indent=2))
    f.write("\n\nKnown Marker References:\n")
    for k, v in MARKER_GENES.items():
        f.write(f"- {k}: {', '.join(v)}\n")

logging.info(f"🧠 Final biological interpretation report saved to: {report_path}")

# ===============================================================
# 8️⃣ Save updated AnnData
# ===============================================================
adata.write(f"{OUTDIR}/adata_agentic_bio.h5ad")
logging.info("✅ Agentic AI biological interpretation pipeline completed (Groq version).")




"""
# ===============================================================
# Stage 4️⃣: Agentic AI Biological Interpretation (LLM + Enrichment)
# ===============================================================



import os
import json
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import fisher_exact

# ---------- CONFIG ----------
DATA_PATH = "GSE243292_ADsnRNAseq_GEO.h5ad"
OUTDIR = "resAA/agentic_bio"
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

# ---------- API Setup ----------
load_dotenv()
client = OpenAI(api_key=os.getenv("openai_key_ai"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===============================================================
# 1️⃣ Known cell-type markers
# ===============================================================
MARKER_GENES = {
    "Excitatory Neurons": ["SLC17A7", "CAMK2A", "SATB2", "VGLUT1"],
    "Inhibitory Neurons": ["GAD1", "GAD2", "SST", "PVALB", "VIP"],
    "Astrocytes": ["GFAP", "AQP4", "SLC1A3", "ALDH1L1"],
    "Oligodendrocytes": ["MBP", "MOG", "PLP1", "MOBP"],
    "OPCs": ["PDGFRA", "CSPG4"],
    "Microglia": ["TREM2", "CX3CR1", "C1QA", "P2RY12"],
    "Endothelial": ["CLDN5", "FLT1", "VWF", "PECAM1"],
    "Pericytes": ["PDGFRB", "RGS5"]
}

# ===============================================================
# 2️⃣ Load AnnData
# ===============================================================
logging.info("🔍 Loading clustered data...")
adata = sc.read_h5ad(DATA_PATH)
logging.info(f"✅ Loaded AnnData object with shape: {adata.shape}")

# Check cluster info
if 'cluster' not in adata.obs.columns:
    logging.warning("⚠️ No 'cluster' column found — using 'leiden' instead.")
    if 'leiden' not in adata.obs.columns:
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution=0.5)
    adata.obs['cluster'] = adata.obs['leiden'].astype(str)

# ===============================================================
# 3️⃣ Log-normalize if raw counts detected
# ===============================================================
if np.max(adata.X) > 50:
    logging.info("🧮 Detected raw count data — applying log1p normalization.")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

# ===============================================================
# 4️⃣ Identify marker genes per cluster
# ===============================================================
logging.info("🔬 Identifying marker genes per cluster...")

if adata.obs['cluster'].nunique() < 2:
    logging.error("❌ Only one cluster detected — cannot compute differential expression.")
    marker_df = pd.DataFrame()
else:
    sc.tl.rank_genes_groups(adata, groupby='cluster', method='wilcoxon', n_genes=50)

    try:
        marker_df = sc.get.rank_genes_groups_df(adata, None)
    except Exception as e:
        logging.warning(f"⚠️ Using backup extraction for marker genes: {e}")
        marker_df = pd.DataFrame({
            "names": adata.uns['rank_genes_groups']['names'].flatten(),
            "scores": adata.uns['rank_genes_groups']['scores'].flatten(),
            "group": np.repeat(list(adata.uns['rank_genes_groups']['names'].dtype.names),
                               adata.uns['rank_genes_groups']['names'].shape[0])
        })

# Fix column naming if needed
if 'group' not in marker_df.columns:
    marker_df['group'] = None

# Summarize
cluster_summary = []
for cluster in sorted(adata.obs['cluster'].unique()):
    subset = marker_df[marker_df['group'] == str(cluster)]
    if subset.empty:
        continue
    top_genes = subset.sort_values("scores", ascending=False).head(10)["names"].tolist()
    cluster_summary.append(f"Cluster {cluster}: {', '.join(top_genes)}")

# ===============================================================
# 5️⃣ Agentic AI LLM-driven biological interpretation
#===============================================================

"""
#prompt = f"""
"""You are an expert genomics AI analyzing single-nucleus RNA-seq clusters from human brain (AD and Down Syndrome).
Each cluster represents a potential brain cell population.
Use the known marker gene dictionary below to predict the most likely cell type for each cluster.

Known markers: {json.dumps(MARKER_GENES)}
Top marker genes per cluster:
{chr(10).join(cluster_summary)}
"""
"""
Respond ONLY in valid JSON format as:
{{
  "0": "Excitatory Neuron",
  "1": "Microglia",
  "2": "Astrocyte",
  ...
}}
"""
"""
logging.info("🤖 Sending cluster marker summary to LLM for biological interpretation...")
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a bioinformatics AI assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    interpretation_text = response.choices[0].message.content
    celltype_map = json.loads(interpretation_text)
    logging.info("✅ LLM successfully classified cell types for clusters.")
except Exception as e:
    logging.error(f"❌ LLM request or parsing failed: {e}")
    celltype_map = {}

adata.obs['cell_type_pred'] = adata.obs['cluster'].map(celltype_map).fillna("Unknown")

# ===============================================================
# 6️⃣ Compute UMAP for visualization
# ===============================================================
if 'X_umap' not in adata.obsm.keys():
    logging.info("📈 Computing UMAP embedding for visualization...")
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

sc.pl.umap(
    adata,
    color=['cell_type_pred'],
    legend_loc='on data',
    title='Agentic AI Predicted Cell Types',
    show=False
)
plt.savefig(f"{OUTDIR}/umap_predicted_celltypes.png", dpi=300)
plt.close()

# ===============================================================
# 7️⃣ Save report
# ===============================================================
report_path = f"{OUTDIR}/agentic_biological_report.txt"
with open(report_path, "w") as f:
    f.write("=== Agentic AI Biological Interpretation ===\n\n")
    f.write(f"Dataset: GSE243292 (AD + Down Syndrome snRNA-seq)\n")
    f.write(f"Clusters identified: {adata.obs['cluster'].nunique()}\n\n")
    f.write("Predicted Cell Type per Cluster:\n")
    f.write(json.dumps(celltype_map, indent=2))
    f.write("\n\nKnown Marker References:\n")
    for k, v in MARKER_GENES.items():
        f.write(f"- {k}: {', '.join(v)}\n")

logging.info(f"🧠 Final biological interpretation report saved to: {report_path}")

# ===============================================================
# 8️⃣ Save updated AnnData
# ===============================================================
adata.write(f"{OUTDIR}/adata_agentic_bio.h5ad")
logging.info("✅ Agentic AI biological interpretation pipeline completed.")



"""