# ===============================================================
# Differential Expression Analysis for Astrocytes (Python version)
# Dataset: GSE243292 (AD + Down Syndrome snRNA-seq)
# ===============================================================

import scanpy as sc
import pandas as pd
import numpy as np
import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
import logging

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
DATA_PATH = "resAA/agentic_bio/adata_clustered.h5ad"   # normalized AnnData
OUTDIR = "resAA/DEG_astrocytes"
Path(OUTDIR).mkdir(parents=True, exist_ok=True)
load_dotenv()
client = Groq(api_key=os.getenv("groq_api_key"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
logging.info("📦 Loading AnnData object...")
adata = sc.read_h5ad(DATA_PATH)
# Print available metadata columns (similar to a pandas DataFrame)
print("📋 Columns available in adata.obs:")
print(adata.obs.columns.tolist())

# If you want to preview the data in the first few rows
print("\n🔍 Preview of adata.obs:")
print(adata.obs.head())
logging.info(f"✅ Loaded AnnData: {adata.shape}")

# Verify required metadata
required_cols = ["cluster", "atscore"]
for col in required_cols:
    if col not in adata.obs.columns:
        raise ValueError(f"❌ Missing column in adata.obs: {col}")

# ---------------------------------------------------------------
# ASTROCYTE CLUSTERS
# ---------------------------------------------------------------
ASTRO_CLUSTERS = [13, 19]

# ---------------------------------------------------------------
# Helper Function: DEG analysis
# ---------------------------------------------------------------
def run_deg_analysis(adata, cluster, groupby, ident_1, ident_2, method, contrast_label):
    logging.info(f"🔬 Running DEG ({method}) for cluster {cluster} | {ident_1} vs {ident_2}")
    subset = adata[adata.obs["cluster"] == str(cluster)].copy()

    groups = subset.obs[groupby].unique().tolist()
    if ident_1 not in groups or ident_2 not in groups:
        logging.warning(f"⚠️ Skipping cluster {cluster} — missing groups {ident_1}, {ident_2}")
        return pd.DataFrame()

    try:
        sc.tl.rank_genes_groups(subset, groupby=groupby, groups=[ident_1],
                                reference=ident_2, method=method, n_genes=300)
        result = sc.get.rank_genes_groups_df(subset, None)
        result["Cluster"] = cluster
        result["Contrast"] = contrast_label
        result["Method"] = method
        return result
    except Exception as e:
        logging.error(f"❌ DEG failed for cluster {cluster} ({method}): {e}")
        return pd.DataFrame()

# ---------------------------------------------------------------
# 1️⃣ Run DEG using MAST (approximated via logistic regression)
# ---------------------------------------------------------------
mast_results = []
for c in ASTRO_CLUSTERS:
    mast_results.append(run_deg_analysis(adata, c, "atscore",
                                         "A+T+", "A+T-",
                                         method="logreg",
                                         contrast_label="A+T+-vs-A+T-"))

# Filter out empty frames before concat
mast_results = [r for r in mast_results if not r.empty]
if not mast_results:
    raise ValueError("❌ No DEG results were generated for any astrocyte cluster.")

mast_df = pd.concat(mast_results, ignore_index=True)

# --- Handle column name differences and direction assignment (Seurat-style) ---
lfc_candidates = ["logfoldchanges", "logfoldchange", "log2foldchange", "log2fc", "avg_logFC"]
lfc_col = next((c for c in lfc_candidates if c in mast_df.columns), None)

# If no log fold change column found, fallback to 'scores'
if lfc_col is None:
    logging.warning(f"⚠️ No log fold change column found. Using 'scores' as proxy. "
                    f"Available columns: {mast_df.columns.tolist()}")
    mast_df["Direction"] = np.where(mast_df["scores"] > 0, "up", "down")
else:
    mast_df["Direction"] = np.where(mast_df[lfc_col] > 0, "up", "down")



"""
# --- Handle column name differences robustly ---
lfc_candidates = ["logfoldchanges", "logfoldchange", "log2foldchange", "log2fc"]
lfc_col = next((c for c in lfc_candidates if c in mast_df.columns), None)
if lfc_col is None:
    raise KeyError(f"❌ No log fold change column found in DEG results. Available columns: {mast_df.columns.tolist()}")

mast_df["Direction"] = np.where(mast_df[lfc_col] > 0, "up", "down")


# --- Handle column name differences ---
lfc_col = "logfoldchanges" if "logfoldchanges" in mast_df.columns else "logfoldchange"
mast_df["Direction"] = np.where(mast_df[lfc_col] > 0, "up", "down")
"""
mast_path = f"{OUTDIR}/cluster_astrocytes_disease_DEGs_MAST.tsv"
mast_df.to_csv(mast_path, sep="\t", index=False)
logging.info(f"💾 Saved MAST-like results: {mast_path}")

# ---------------------------------------------------------------
# 2️⃣ Run DEG using Wilcoxon test
# ---------------------------------------------------------------
wilcox_results = []
for c in ASTRO_CLUSTERS:
    wilcox_results.append(run_deg_analysis(adata, c, "atscore",
                                           "A+T+", "A+T-",
                                           method="wilcoxon",
                                           contrast_label="A+T+-vs-A+T-"))

wilcox_results = [r for r in wilcox_results if not r.empty]
if not wilcox_results:
    raise ValueError("❌ No Wilcoxon DEG results were generated for any astrocyte cluster.")

wilcox_df = pd.concat(wilcox_results, ignore_index=True)

# --- Handle column name differences and direction assignment (Seurat-style) ---
lfc_candidates_w = ["logfoldchanges", "logfoldchange", "log2foldchange", "log2fc", "avg_logFC"]
lfc_col_w = next((c for c in lfc_candidates_w if c in wilcox_df.columns), None)

if lfc_col_w is None:
    logging.warning(f"⚠️ No log fold change column found in Wilcoxon results. "
                    f"Using 'scores' as proxy. Available columns: {wilcox_df.columns.tolist()}")
    wilcox_df["Direction"] = np.where(wilcox_df["scores"] > 0, "up", "down")
else:
    wilcox_df["Direction"] = np.where(wilcox_df[lfc_col_w] > 0, "up", "down")



"""

lfc_candidates_w = ["logfoldchanges", "logfoldchange", "log2foldchange", "log2fc"]
lfc_col_w = next((c for c in lfc_candidates_w if c in wilcox_df.columns), None)
if lfc_col_w is None:
    raise KeyError(f"❌ No log fold change column found in Wilcoxon results. Available columns: {wilcox_df.columns.tolist()}")

wilcox_df["Direction"] = np.where(wilcox_df[lfc_col_w] > 0, "up", "down")



lfc_col_w = "logfoldchanges" if "logfoldchanges" in wilcox_df.columns else "logfoldchange"
wilcox_df["Direction"] = np.where(wilcox_df[lfc_col_w] > 0, "up", "down")
"""

wilcox_path = f"{OUTDIR}/cluster_astrocytes_disease_DEGs_wilcox.tsv"
wilcox_df.to_csv(wilcox_path, sep="\t", index=False)
logging.info(f"💾 Saved Wilcoxon results: {wilcox_path}")

# ---------------------------------------------------------------
# 3️⃣ Compare Results (MAST vs Wilcoxon)
# ---------------------------------------------------------------
mast_summary = mast_df.groupby(["Cluster", "Direction"]).size().reset_index(name="MAST")
wilcox_summary = wilcox_df.groupby(["Cluster", "Direction"]).size().reset_index(name="Wilcox")

common_genes = pd.merge(
    mast_df[["names", "Cluster", "Direction"]],
    wilcox_df[["names", "Cluster", "Direction"]],
    on=["names", "Cluster", "Direction"]
)
common_summary = common_genes.groupby(["Cluster", "Direction"]).size().reset_index(name="Common")

compare_df = mast_summary.merge(wilcox_summary, on=["Cluster", "Direction"], how="outer")
compare_df = compare_df.merge(common_summary, on=["Cluster", "Direction"], how="outer")
compare_df.fillna(0, inplace=True)

compare_path = f"{OUTDIR}/cluster_astrocytes_disease_DEGs_MAST_Wilcox_compare.tsv"
compare_df.to_csv(compare_path, sep="\t", index=False)
logging.info(f"📊 Saved comparison results: {compare_path}")

# ---------------------------------------------------------------
# 4️⃣ Optional: LLM summary via Groq
# ---------------------------------------------------------------
prompt = f"""
You are a genomics AI assistant. Interpret the following differential expression summary table.
Each row shows how many genes were up/down-regulated in astrocyte clusters (13,19) in A+T+ versus A+T- Alzheimer's pathology.
Provide a 3-sentence biological summary of what the results might indicate.

Summary Table:
{compare_df.head(10).to_string(index=False)}
"""

try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a bioinformatics expert."},
            {"role": "user", "content": prompt},
        ],
    )
    summary_text = response.choices[0].message.content
    with open(f"{OUTDIR}/groq_summary.txt", "w") as f:
        f.write(summary_text)
    logging.info("🧠 Groq AI biological interpretation saved.")
except Exception as e:
    logging.warning(f"⚠️ Groq LLM summary failed: {e}")

logging.info("✅ DEG analysis pipeline completed successfully.")
