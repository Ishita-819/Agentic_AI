import os
import json
import re
import subprocess
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
from groq import Groq
import pandas as pd


load_dotenv()
GROQ_KEY = os.getenv("groq_api_key")
if not GROQ_KEY:
    raise RuntimeError("GROQ API key not found in environment variable 'groq_api_key'. Add it to .env")

client = Groq(api_key=GROQ_KEY)

BASE_DIR = Path("resAA")
OUTDIR = BASE_DIR / "agentic_orchestration"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Path to uploaded local file
UPLOADED_FILE_PATH = "GSE243292_ADsnRNAseq_GEO.h5ad"
UPLOADED_FILE_URL = f"file://{Path(UPLOADED_FILE_PATH).resolve()}"

# Stage scripts
STAGE_SCRIPTS = {
    "clustering": "agentic_ai_clustering.py",
    "differential_expression": "differential_expression.py",
    "trajectory": "trajectory_analysis.py",
}

# Expected outputs
OUTPUT_PATHS = {
    "bio_report": BASE_DIR / "agentic_bio" / "agentic_biological_report.txt",
    "deg_compare": BASE_DIR / "DEG_astrocytes" / "cluster_astrocytes_disease_DEGs_MAST_Wilcox_compare.tsv",
    "traj_genes": BASE_DIR / "trajectory_inference" / "pseudotime_correlated_genes.tsv",
    "umap": BASE_DIR / "agentic_bio" / "umap_predicted_celltypes.png",
}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------- Core Agent Functions ----------------
def run_script(script_name: str) -> Tuple[bool, str]:
    """Execute a Python script as a subprocess."""
    logging.info(f"🚀 Running: {script_name}")
    try:
        subprocess.run(["python", script_name], check=True)
        logging.info(f"✅ Completed: {script_name}")
        return True, f"{script_name} completed"
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Script failed: {script_name} | {e}")
        return False, f"{script_name} failed: {e}"
    except FileNotFoundError:
        logging.error(f"❌ Script not found: {script_name}")
        return False, f"{script_name} not found"


def safe_read_text(p: Path, max_chars: int = 6000) -> str:
    """Safely read text file with truncation."""
    try:
        t = Path(p).read_text(encoding="utf-8")
        return t[:max_chars] if len(t) > max_chars else t
    except Exception as e:
        logging.warning(f"Could not read {p}: {e}")
        return f"[Could not read {p}: {e}]"


def safe_read_all_table(p: Path) -> Optional[pd.DataFrame]:
    """Safely read entire tabular data file."""
    try:
        df = pd.read_csv(p, sep="\t")
        return df
    except Exception as e:
        logging.warning(f"Could not read entire table {p}: {e}")
        return None


def safe_read_table_preview(p: Path, nrows: int = 100) -> Optional[pd.DataFrame]:
    """Safely read tabular data preview."""
    try:
        df = pd.read_csv(p, sep="\t", nrows=nrows)
        return df
    except Exception as e:
        logging.warning(f"Could not read table {p}: {e}")
        return None


def groq_chat(messages: List[Dict[str, str]], temperature: float = 0.25, max_tokens: int = 2000) -> str:
    """Wrapper to call Groq chat completion."""
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        logging.error(f"Groq call failed: {e}")
        return f"[Groq error: {e}]"

"""
def execute_mandatory_pipeline() -> Dict[str, Any]:
    
    
    print("\n" + "="*60)
    print("PHASE 1: MANDATORY SEQUENTIAL PIPELINE")
    print("="*60)
    
    results = {}
    
    # Step 1: Clustering (foundation for everything)
    print("\n📊 Step 1/3: Clustering Analysis")
    ok, msg = run_script(STAGE_SCRIPTS["clustering"])
    results["clustering"] = {"success": ok, "message": msg}
    # if not ok:
    #     logging.error("Clustering failed. Pipeline cannot continue.")
    #     return results
    
    # time.sleep(1)  # Allow file system to settle
    
    # Step 2: Differential Expression (requires clusters)
    print("\n📈 Step 2/3: Differential Expression Analysis")
    ok, msg = run_script(STAGE_SCRIPTS["differential_expression"])
    results["de"] = {"success": ok, "message": msg}
    
    # time.sleep(1)
    
    # Step 3: Trajectory Inference (uses clusters/embeddings)
    print("\n🔄 Step 3/3: Trajectory Inference")
    ok, msg = run_script(STAGE_SCRIPTS["trajectory"])
    results["trajectory"] = {"success": ok, "message": msg}
    
    # time.sleep(1)  # Final wait for all files
    
    return results
"""

def execute_mandatory_pipeline() -> Dict[str, Any]:
    """
    Execute the required sequential pipeline:
    Clustering → Differential Expression → Trajectory Inference
    Returns a dict with per-step success + messages.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: MANDATORY SEQUENTIAL PIPELINE")
    print("=" * 60)

    results = {}

    # Step 1: Clustering
    print("\n📊 Step 1/3: Clustering Analysis")
    ok, msg = run_script(STAGE_SCRIPTS["clustering"])
    results["clustering"] = {"success": ok, "message": msg}

    # Step 2: Differential Expression
    print("\n📈 Step 2/3: Differential Expression Analysis")
    ok, msg = run_script(STAGE_SCRIPTS["differential_expression"])
    results["de"] = {"success": ok, "message": msg}

    # Step 3: Trajectory Inference
    print("\n🔄 Step 3/3: Trajectory Inference")
    ok, msg = run_script(STAGE_SCRIPTS["trajectory"])
    results["trajectory"] = {"success": ok, "message": msg}

    return results


def load_complete_results() -> Tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load ALL results for comprehensive analysis."""
    print("\n📊 Loading COMPLETE analysis results...")
    
    # Read clustering report (truncated for prompt, but substantial)
    bio_text = safe_read_text(OUTPUT_PATHS["bio_report"], max_chars=6000)
    
    # Read ENTIRE tables for comprehensive analysis
    deg_df = safe_read_all_table(OUTPUT_PATHS["deg_compare"])
    traj_df = safe_read_all_table(OUTPUT_PATHS["traj_genes"])
    
    # Statistics output
    print("\n📈 DATA STATISTICS:")
    print("-" * 40)
    print(f"Clustering report: {len(bio_text)} characters")
    
    if deg_df is not None:
        deg_count = deg_df.shape[0]
        print(f"Differential Expression: {deg_count} genes, {deg_df.shape[1]} columns")
        # Try to find p-value column for significance stats
        pval_cols = [col for col in deg_df.columns if 'pval' in col.lower() or 'adj' in col.lower()]
        if pval_cols:
            pval_col = pval_cols[0]
            sig_genes = deg_df[deg_df[pval_col] < 0.05]
            sig_count = len(sig_genes)
            sig_percent = (sig_count/deg_count*100) if deg_count > 0 else 0
            print(f"  Significant genes (p<0.05): {sig_count} ({sig_percent:.1f}%)")
    else:
        print("Differential Expression: No data found")
    
    if traj_df is not None:
        traj_count = traj_df.shape[0]
        print(f"Trajectory Genes: {traj_count} genes, {traj_df.shape[1]} columns")
        # Try to find correlation column
        corr_cols = [col for col in traj_df.columns if 'cor' in col.lower() or 'rho' in col.lower()]
        if corr_cols:
            corr_col = corr_cols[0]
            strong_pos = traj_df[traj_df[corr_col] > 0.5]
            strong_neg = traj_df[traj_df[corr_col] < -0.5]
            print(f"  Strong positive correlation (>0.5): {len(strong_pos)} genes")
            print(f"  Strong negative correlation (<-0.5): {len(strong_neg)} genes")
    else:
        print("Trajectory Genes: No data found")
    
    print("-" * 40)
    
    return bio_text, deg_df, traj_df



def extract_comprehensive_insights(bio_text: str, deg_df: pd.DataFrame, 
                                  traj_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract comprehensive insights from ALL data for hypothesis generation.
    Analyzes statistical significance, patterns, and key findings.
    """
    insights = {
        "clustering_summary": bio_text[:4000],
        "deg_comprehensive": {},
        "trajectory_comprehensive": {},
        "key_integrated_findings": [],
        "data_summary": {}
    }
    
    # Comprehensive DEG Analysis
    if deg_df is not None and not deg_df.empty:
        insights["data_summary"]["deg_total_genes"] = len(deg_df)
        
        # Find relevant columns
        pval_cols = [col for col in deg_df.columns if 'pval' in col.lower() or 'adj' in col.lower()]
        logfc_cols = [col for col in deg_df.columns if 'logfc' in col.lower() or 'log2fc' in col.lower()]
        gene_cols = [col for col in deg_df.columns if 'gene' in col.lower() or 'id' in col.lower()]
        
        gene_col = gene_cols[0] if gene_cols else deg_df.columns[0]
        pval_col = pval_cols[0] if pval_cols else None
        logfc_col = logfc_cols[0] if logfc_cols else None
        
        # Significant genes
        if pval_col:
            sig_genes = deg_df[deg_df[pval_col] < 0.05]
            insights["deg_comprehensive"]["significant_genes"] = len(sig_genes)
            insights["deg_comprehensive"]["significance_percentage"] = (len(sig_genes) / len(deg_df) * 100) if len(deg_df) > 0 else 0
        
        # Top genes by fold change
        if logfc_col:
            top_up = deg_df.nlargest(15, logfc_col)
            top_down = deg_df.nsmallest(15, logfc_col)
            
            insights["deg_comprehensive"]["top_upregulated"] = [
                {gene_col: row[gene_col], logfc_col: row[logfc_col], 
                 pval_col: row[pval_col] if pval_col else None}
                for _, row in top_up.iterrows()
            ][:10]  # Keep top 10
            
            insights["deg_comprehensive"]["top_downregulated"] = [
                {gene_col: row[gene_col], logfc_col: row[logfc_col], 
                 pval_col: row[pval_col] if pval_col else None}
                for _, row in top_down.iterrows()
            ][:10]
        
        # Detailed preview for prompt
        insights["deg_detailed_preview"] = deg_df.head(150).to_string(index=False)
    
    # Comprehensive Trajectory Analysis
    if traj_df is not None and not traj_df.empty:
        insights["data_summary"]["trajectory_total_genes"] = len(traj_df)
        
        # Find correlation column
        corr_cols = [col for col in traj_df.columns if 'cor' in col.lower() or 'rho' in col.lower()]
        gene_cols = [col for col in traj_df.columns if 'gene' in col.lower() or 'id' in col.lower()]
        
        gene_col = gene_cols[0] if gene_cols else traj_df.columns[0]
        corr_col = corr_cols[0] if corr_cols else None
        
        if corr_col:
            strong_pos = traj_df[traj_df[corr_col] > 0.5]
            strong_neg = traj_df[traj_df[corr_col] < -0.5]
            
            insights["trajectory_comprehensive"]["strong_positive"] = len(strong_pos)
            insights["trajectory_comprehensive"]["strong_negative"] = len(strong_neg)
            
            # Top correlated genes
            top_pos_corr = traj_df.nlargest(15, corr_col)
            top_neg_corr = traj_df.nsmallest(15, corr_col)
            
            insights["trajectory_comprehensive"]["top_positive_correlated"] = [
                {gene_col: row[gene_col], corr_col: row[corr_col]}
                for _, row in top_pos_corr.iterrows()
            ][:10]
            
            insights["trajectory_comprehensive"]["top_negative_correlated"] = [
                {gene_col: row[gene_col], corr_col: row[corr_col]}
                for _, row in top_neg_corr.iterrows()
            ][:10]
        
        # Detailed preview for prompt
        insights["trajectory_detailed_preview"] = traj_df.head(150).to_string(index=False)
    
    # Extract integrated findings
    integrated_findings = []
    
    # From clustering
    clustering_keywords = ["cluster", "subtype", "population", "annotation", "umap", "t-sne"]
    if any(keyword in bio_text.lower() for keyword in clustering_keywords):
        integrated_findings.append("Identified distinct cellular subpopulations through clustering analysis")
    
    # From DEGs
    if insights.get("deg_comprehensive", {}).get("significant_genes", 0) > 0:
        sig_count = insights["deg_comprehensive"]["significant_genes"]
        integrated_findings.append(f"Found {sig_count} statistically significant differentially expressed genes (p < 0.05)")
        
        if insights["deg_comprehensive"].get("top_upregulated"):
            top_gene = insights["deg_comprehensive"]["top_upregulated"][0].get(list(insights["deg_comprehensive"]["top_upregulated"][0].keys())[0], "")
            if top_gene:
                integrated_findings.append(f"Top upregulated gene: {top_gene}")
    
    # From trajectory
    if insights.get("trajectory_comprehensive", {}).get("strong_positive", 0) > 0:
        pos_count = insights["trajectory_comprehensive"]["strong_positive"]
        integrated_findings.append(f"Identified {pos_count} genes strongly positively correlated with pseudotime (corr > 0.5)")
    
    insights["key_integrated_findings"] = integrated_findings
    
    return insights

# global_store.py
hypotheses = []

def set_value(x):
    global hypotheses
    hypotheses = x

def get_value():
    return hypotheses

# reasoning_store.py
reasoning_trace = []

def add_reasoning(step, summary, result=None):
    reasoning_trace.append({
        "step": step,
        "summary": summary,
        "result": result
    })

def get_reasoning():
    return reasoning_trace




def generate_comprehensive_hypotheses(bio_text: str, deg_df: pd.DataFrame, 
                                     traj_df: pd.DataFrame, n: int = 5) -> List[str]:
    """Generate hypotheses using COMPREHENSIVE analysis of ALL data."""
    print(f"\n💡 Generating {n} integrated hypotheses from COMPLETE dataset...")
    
    # Extract comprehensive insights from ALL data
    insights = extract_comprehensive_insights(bio_text, deg_df, traj_df)
    
    # Format numbers without thousands separators to avoid formatting conflicts
    deg_total = insights['data_summary'].get('deg_total_genes', 'N/A')
    deg_sig = insights['deg_comprehensive'].get('significant_genes', 'N/A')
    deg_percent = insights['deg_comprehensive'].get('significance_percentage', 0)
    
    traj_total = insights['data_summary'].get('trajectory_total_genes', 'N/A')
    traj_pos = insights['trajectory_comprehensive'].get('strong_positive', 'N/A')
    traj_neg = insights['trajectory_comprehensive'].get('strong_negative', 'N/A')
    
    # Format upregulated genes safely
    upregulated_text = ""
    if insights['deg_comprehensive'].get('top_upregulated'):
        for gene in insights['deg_comprehensive']['top_upregulated'][:7]:
            gene_values = list(gene.values())
            if len(gene_values) >= 3:
                upregulated_text += f"  • {gene_values[0]}: logFC={gene_values[1]:.2f}, p={gene_values[2]:.3e}\n"
            else:
                upregulated_text += f"  • {gene_values[0]}\n"
    else:
        upregulated_text = "  • No upregulated gene data available\n"
    
    # Format downregulated genes safely
    downregulated_text = ""
    if insights['deg_comprehensive'].get('top_downregulated'):
        for gene in insights['deg_comprehensive']['top_downregulated'][:7]:
            gene_values = list(gene.values())
            if len(gene_values) >= 3:
                downregulated_text += f"  • {gene_values[0]}: logFC={gene_values[1]:.2f}, p={gene_values[2]:.3e}\n"
            else:
                downregulated_text += f"  • {gene_values[0]}\n"
    else:
        downregulated_text = "  • No downregulated gene data available\n"
    
    # Format positively correlated genes safely
    pos_corr_text = ""
    if insights['trajectory_comprehensive'].get('top_positive_correlated'):
        for gene in insights['trajectory_comprehensive']['top_positive_correlated'][:7]:
            gene_values = list(gene.values())
            if len(gene_values) >= 2:
                pos_corr_text += f"  • {gene_values[0]}: corr={gene_values[1]:.3f}\n"
            else:
                pos_corr_text += f"  • {gene_values[0]}\n"
    else:
        pos_corr_text = "  • No positively correlated gene data available\n"
    
    # Format negatively correlated genes safely
    neg_corr_text = ""
    if insights['trajectory_comprehensive'].get('top_negative_correlated'):
        for gene in insights['trajectory_comprehensive']['top_negative_correlated'][:7]:
            gene_values = list(gene.values())
            if len(gene_values) >= 2:
                neg_corr_text += f"  • {gene_values[0]}: corr={gene_values[1]:.3f}\n"
            else:
                neg_corr_text += f"  • {gene_values[0]}\n"
    else:
        neg_corr_text = "  • No negatively correlated gene data available\n"
    
    # Build the prompt safely without formatting conflicts
    prompt_text = f"""
You are an expert computational biologist analyzing single-cell RNA-seq data from Alzheimer's disease astrocytes.

COMPREHENSIVE ANALYSIS RESULTS (USING ALL DATA):

1. CLUSTERING/CELL TYPE ANNOTATION:
{insights['clustering_summary']}

2. DIFFERENTIAL EXPRESSION ANALYSIS (COMPLETE DATASET):
- Total genes analyzed: {deg_total}
- Statistically significant DEGs (p < 0.05): {deg_sig} 
  ({deg_percent:.1f}% of all genes)

TOP UPREGULATED GENES IN DISEASE:
{upregulated_text}
TOP DOWNREGULATED GENES IN DISEASE:
{downregulated_text}
3. TRAJECTORY/PSEUDOTIME ANALYSIS (COMPLETE DATASET):
- Total trajectory-correlated genes: {traj_total}
- Genes strongly positively correlated (corr > 0.5): {traj_pos}
- Genes strongly negatively correlated (corr < -0.5): {traj_neg}

TOP POSITIVELY CORRELATED GENES WITH PSEUDOTIME:
{pos_corr_text}
TOP NEGATIVELY CORRELATED GENES WITH PSEUDOTIME:
{neg_corr_text}
KEY INTEGRATED FINDINGS:
{chr(10).join(['• ' + finding for finding in insights['key_integrated_findings']])}

DETAILED DEG DATA (first 150 genes - representative sample):
{insights.get('deg_detailed_preview', '[No DEG data available]')[:3000]}

DETAILED TRAJECTORY DATA (first 150 genes - representative sample):
{insights.get('trajectory_detailed_preview', '[No trajectory data available]')[:3000]}

TASK: Generate exactly {n} MECHANISTIC biological hypotheses that INTEGRATE ALL findings above.

Each hypothesis MUST:
1. Be a single, concise sentence (20-50 words)
2. Propose a specific biological mechanism or causal relationship
3. Connect at least TWO different analysis dimensions (clustering, DEG, trajectory)
4. Be specific to Alzheimer's disease pathophysiology in astrocytes
5. Be testable/falsifiable with follow-up experiments
6. Reference specific genes or pathways when possible
7. Hypothesis must motivate new research direction helpful for drug discovery or Alzheimer's treatment

Return ONLY a valid JSON array of exactly {n} hypothesis strings.
DO NOT include any explanations, numbering, or additional text.
"""
    
    messages = [
        {"role": "system", "content": "You are an expert computational biologist specialized in Alzheimer's disease and single-cell genomics. Generate specific, integrated, testable hypotheses based on COMPLETE analysis of all data dimensions. Return ONLY a JSON array."},
        {"role": "user", "content": prompt_text}
    ]
    
    try:
        resp = groq_chat(messages, temperature=0.3, max_tokens=2000)
        
        # Clean response and parse JSON
        resp_clean = resp.strip()
        # Try to extract JSON if there's extra text
        json_match = re.search(r'\[\s*\{.*\}\s*\]|\["[^"]*"(?:\s*,\s*"[^"]*")*\s*\]', resp_clean, re.DOTALL)
        if json_match:
            resp_clean = json_match.group(0)
        
        # Parse JSON response
        parsed = json.loads(resp_clean)
        if isinstance(parsed, list):
            hypotheses = [h.strip() for h in parsed[:n]]
            print(f"✅ Generated {len(hypotheses)} comprehensive hypotheses")
            set_value(hypotheses)
            return hypotheses
    except (json.JSONDecodeError, AttributeError) as e:
        logging.warning(f"Failed to parse JSON from LLM: {e}")
        print("⚠️  JSON parsing failed, attempting to extract hypotheses from text...")
    
    # Fallback: Try to extract from text response
    try:
        # Look for numbered or bulleted hypotheses
        lines = resp.split('\n')
        hypotheses = []
        for line in lines:
            line = line.strip()
            # Look for patterns like "1. ", "- ", "* ", "• ", "Hypothesis X:"
            if re.match(r'^(\d+\.\s+|[-*•]\s+|\d+\)\s+|Hypothesis\s+\d+:\s*)', line, re.IGNORECASE):
                hypothesis = re.sub(r'^(\d+\.\s+|[-*•]\s+|\d+\)\s+|Hypothesis\s+\d+:\s*)', '', line, flags=re.IGNORECASE)
                if 20 < len(hypothesis) < 300 and any(keyword in hypothesis.lower() for keyword in ['astrocyte', 'alzheimer', 'disease', 'gene', 'cluster', 'trajectory', 'expression']):
                    hypotheses.append(hypothesis)
        
        if len(hypotheses) >= n:
            print(f"✅ Extracted {len(hypotheses[:n])} hypotheses from text")
            return hypotheses[:n]
        
        # If not enough, split by sentences that look like hypotheses
        sentences = re.split(r'[.!?]+', resp)
        for sentence in sentences:
            sentence = sentence.strip()
            if (30 < len(sentence) < 250 and 
                any(keyword in sentence.lower() for keyword in ['astrocyte', 'alzheimer', 'disease', 'cluster', 'gene', 'expression', 'trajectory', 'hypothesis', 'suggest', 'propose']) and
                not sentence.startswith('Example') and not sentence.startswith('Return')):
                hypotheses.append(sentence + '.')
        
        if hypotheses:
            print(f"✅ Extracted {len(hypotheses[:n])} hypotheses from sentences")
            return hypotheses[:n]
        
        # Last resort: return insights as hypotheses
        print("⚠️  Could not extract structured hypotheses, returning key findings")
        return insights["key_integrated_findings"][:n]
    
    except Exception as e:
        logging.error(f"Failed to generate hypotheses: {e}")
        return ["Error in comprehensive hypothesis generation from complete dataset"]


# ---------------- Phase 2: Interactive Orchestrator with REAL Analysis ----------------
def get_llm_suggested_next_steps(hypotheses: List[str],reasoning_context: List[str], bio_text: str, 
                                deg_df: pd.DataFrame, traj_df: pd.DataFrame, k: int = 5) -> List[str]:
    """
    LLM suggests simple, clear next steps (1 line each).
    """
    print("\n🤖 LLM Orchestrator: Suggesting clear next steps...")
    
    # Get simple previews
    deg_preview = deg_df.head(10).to_string(index=False) if deg_df is not None else "[No DEG data]"
    traj_preview = traj_df.head(10).to_string(index=False) if traj_df is not None else "[No trajectory data]"
    
    prompt = f"""  You are an expert computational biologist.
Based on these analysis results, suggest {k} SIMPLE, CLEAR next steps for further investigation.
Return as a JSON array of strings.

RULES:
- Each suggestion must be exactly ONE LINE
- Start with a verb (e.g., "Analyze", "Visualize", "Compare", "Check", "Validate")
- Be specific but concise
- Focus on actionable analyses

ANALYSIS RESULTS:
- Clustering : {bio_text}
- Differential Expression: {deg_df.shape[0] if deg_df is not None else 0} genes analyzed
- Trajectory: {traj_df.shape[0] if traj_df is not None else 0} genes correlated with pseudotime

KEY HYPOTHESES:
{chr(10).join([f"{i+1}. {h}" for i, h in enumerate(hypotheses[:3])])}

Analyses already performed: {reasoning_context}
"""
    
    messages = [
        {"role": "system", "content": "You are a bioinformatics assistant. Suggest simple, clear, one-line next steps. Return ONLY a JSON array."},
        {"role": "user", "content": prompt}
    ]
    
    response = groq_chat(messages, temperature=0.3)
    
    try:
        parsed = json.loads(response)
        if isinstance(parsed, list):
            suggestions = [s.strip() for s in parsed[:k]]
            print(f"✅ Generated {len(suggestions)} clear next steps")
            return suggestions
    except json.JSONDecodeError:
        logging.warning(f"Failed to parse JSON from LLM, extracting suggestions from text")
    
    # If JSON parsing fails, extract suggestions from text
    suggestions = []
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        # Look for numbered or bulleted suggestions
        if re.match(r'^(\d+[\.\)]\s+|\-\s+|\*\s+|•\s+)', line):
            suggestion = re.sub(r'^(\d+[\.\)]\s+|\-\s+|\*\s+|•\s+)', '', line)
            if suggestion and len(suggestion) < 100:  # Reasonable length
                suggestions.append(suggestion)
    
    # If not enough, use default simple suggestions
    if len(suggestions) < k:
        default_suggestions = get_default_next_steps()
        suggestions.extend(default_suggestions)
    
    return suggestions[:k]

def get_default_next_steps() -> List[str]:
    """Default simple next steps if LLM fails."""
    return [
        "Analyze inflammatory gene expression in trajectory",
        "Visualize top DEGs in disease clusters",
        "Check pathway enrichment for trajectory-correlated genes",
        "Compare results with Alzheimer's disease databases",
        "Generate detailed clustering report"
    ]


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
ALL_KNOWN_GENES = {
    g for genes in MARKER_GENES.values() for g in genes
}


def execute_detailed_analysis(
    suggestion: str,
    execution_result: dict

) -> str:
    """
    Interpret ONLY what was actually executed.
    """

    report = [
        f"📊 DETAILED ANALYSIS REPORT: {suggestion}",
        "=" * 60
    ]

    if not execution_result.get("success"):
        report.append("❌ Analysis could not be executed.")
        return "\n".join(report)

    action = execution_result["action_type"]
    result = execution_result["result"]

    if action == "deg_analysis":
        report.append("🔬 DIFFERENTIAL EXPRESSION RESULTS:")
        report.append(f"• Total genes analyzed: {result['total_genes']}")
        report.append(f"• Significant genes (p<0.05): {result['significant_genes']}")

    elif action == "trajectory_analysis":
        report.append("📈 TRAJECTORY ANALYSIS RESULTS:")
        report.append(f"• Total genes analyzed: {result['total_genes']}")
        report.append(f"• Strong correlations (>0.5): {result['strongly_correlated']}")

    elif action == "marker_validation":
        report.append("🧬 MARKER VALIDATION:")
        report.append(f"• Known markers checked: {len(result['identified_markers'])}")

    report.append("=" * 60)
    report.append(f"📅 Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(report)



"""
def execute_detailed_analysis(
    suggestion: str,
    deg_df: pd.DataFrame,
    traj_df: pd.DataFrame
) -> str:
   

    suggestion_lower = suggestion.lower()
    report_lines = [f"📊 DETAILED ANALYSIS REPORT: {suggestion}", "=" * 60]

    # ---------------- 1. Gene extraction ----------------
    #def extract_valid_genes(text: str, gene_universe: set) -> List[str]:
    candidates = re.findall(r'\b[A-Z][A-Z0-9]{1,6}\b', suggestion.upper())
    genes = sorted(set(g for g in candidates if g in ALL_KNOWN_GENES))
    
    # genes = extract_valid_genes(suggestion, ALL_KNOWN_GENES)

    # gene_pattern = r'\b([A-Z][A-Z0-9]{1,5})\b'
    # genes = re.findall(gene_pattern, suggestion.upper())

    if genes:
        report_lines.append(f"\n🎯 TARGET GENES IDENTIFIED: {', '.join(genes)}")

        # ---- Trajectory analysis ----
        if traj_df is not None and not traj_df.empty:
            report_lines.append("\n📈 TRAJECTORY CORRELATION ANALYSIS:")
            gene_col = next((c for c in traj_df.columns if 'gene' in c.lower() or 'id' in c.lower()), None)
            corr_col = next((c for c in traj_df.columns if 'cor' in c.lower() or 'rho' in c.lower()), None)

            if gene_col and corr_col:
                found = False
                for gene in genes:
                    rows = traj_df[traj_df[gene_col] == gene]
                    if not rows.empty:
                        corr = rows.iloc[0][corr_col]
                        strength = "STRONG" if abs(corr) > 0.5 else "Moderate"
                        direction = "Positive" if corr > 0 else "Negative"
                        report_lines.append(f"  • {gene}: {direction} correlation (r={corr:.3f}) — {strength}")
                        found = True
                if not found:
                    report_lines.append("  • No specified genes found in trajectory data")

        # ---- DEG analysis ----
        if deg_df is not None and not deg_df.empty:
            report_lines.append("\n🔬 DIFFERENTIAL EXPRESSION ANALYSIS:")
            gene_col = next((c for c in deg_df.columns if 'gene' in c.lower() or 'id' in c.lower()), None)
            logfc_col = next((c for c in deg_df.columns if 'logfc' in c.lower()), None)
            pval_col = next((c for c in deg_df.columns if 'pval' in c.lower() or 'adj' in c.lower()), None)

            if gene_col:
                found = False
                for gene in genes:
                    rows = deg_df[deg_df[gene_col] == gene]
                    if not rows.empty:
                        row = rows.iloc[0]
                        logfc = row[logfc_col] if logfc_col else None
                        pval = row[pval_col] if pval_col else None
                        if logfc is not None and pval is not None:
                            direction = "Upregulated" if logfc > 0 else "Downregulated"
                            sig = "Significant" if pval < 0.05 else "Not significant"
                            report_lines.append(
                                f"  • {gene}: {direction} (logFC={logfc:.2f}, p={pval:.2e}) — {sig}"
                            )
                        else:
                            report_lines.append(f"  • {gene}: Found in DEG data")
                        found = True
                if not found:
                    report_lines.append("  • Selected genes are not among top pseudotime-correlated genes, "
        "though global trajectory trends exist")

    # ---------------- 2. Pathway inference ----------------
    report_lines.append("\n🧬 PATHWAY ANALYSIS:")
    pathway_map = {
        "wnt": "Wnt/β-catenin signaling",
        "inflammatory": "Inflammatory response",
        "synaptic": "Synaptic transmission",
        "autophagy": "Autophagy",
        "gaba": "GABAergic signaling",
        "calcium": "Calcium signaling"
    }

    matched = [v for k, v in pathway_map.items() if k in suggestion_lower]
    if matched:
        for p in matched:
            report_lines.append(f"  • {p} pathway implicated")
    else:
        report_lines.append("  • No pathway keywords detected")

    # ---------------- 3. Statistical summary ----------------
    report_lines.append("\n📊 STATISTICAL SUMMARY:")

    if traj_df is not None:
        report_lines.append(f"  • Trajectory genes analyzed: {len(traj_df)}")
        corr_col = next((c for c in traj_df.columns if 'cor' in c.lower() or 'rho' in c.lower()), None)
        if corr_col:
            report_lines.append(f"  • Strong correlations (>0.5): {len(traj_df[traj_df[corr_col].abs() > 0.5])}")

    if deg_df is not None:
        report_lines.append(f"  • DEG genes analyzed: {len(deg_df)}")
        pval_col = next((c for c in deg_df.columns if 'pval' in c.lower() or 'adj' in c.lower()), None)
        if pval_col:
            report_lines.append(f"  • Significant DEGs (p<0.05): {len(deg_df[deg_df[pval_col] < 0.05])}")

    # ---------------- 4. Interpretation ----------------
    report_lines.append("\n💡 BIOLOGICAL INTERPRETATION:")
    if "alzheimer" in suggestion_lower or "disease" in suggestion_lower:
        report_lines.append("  • Findings relate to Alzheimer's astrocyte dysfunction")

    # ---------------- 5. Recommendations ----------------
    # report_lines.append("\n🎯 RECOMMENDED NEXT STEPS:")
    # report_lines.extend([
    #     "  1. Experimentally validate key genes",
    #     "  2. Perform pathway enrichment analysis",
    #     "  3. Compare with external AD datasets",
    #     "  4. Assess druggability of targets"
    # ])

    report_lines.append("\n" + "=" * 60)
    report_lines.append(f"📅 Analysis completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(report_lines)
"""


def parse_suggestion_action(suggestion: str) -> dict:
    s = suggestion.lower()

    if "trajectory" in s or "pseudotime" in s:
        return {"type": "trajectory_analysis"}

    if "deg" in s or "expression" in s:
        return {"type": "deg_analysis"}

    if "marker" in s or "cell type" in s:
        return {"type": "marker_validation"}

    if "pathway" in s:
        return {"type": "pathway_stub"}

    return {"type": "unsupported"}

def map_suggestion_to_action(
    suggestion: str,
    bio_text: str,
    deg_df: pd.DataFrame,
    traj_df: pd.DataFrame
) -> dict:
    """
    Execute the selected next step deterministically.
    NO interpretation, NO hallucination.
    """

    action = parse_suggestion_action(suggestion)
    result = {}

    if action["type"] == "deg_analysis" and deg_df is not None:
        pval_col = next((c for c in deg_df.columns if 'pval' in c.lower() or 'adj' in c.lower()), None)
        sig = deg_df[deg_df[pval_col] < 0.05] if pval_col else pd.DataFrame()

        result = {
            "total_genes": len(deg_df),
            "significant_genes": len(sig),
            "top_genes": sig.head(10).to_dict("records")
        }

    elif action["type"] == "trajectory_analysis" and traj_df is not None:
        corr_col = next((c for c in traj_df.columns if 'cor' in c.lower() or 'rho' in c.lower()), None)
        strong = traj_df[traj_df[corr_col].abs() > 0.5] if corr_col else pd.DataFrame()

        result = {
            "total_genes": len(traj_df),
            "strongly_correlated": len(strong),
            "top_genes": strong.head(10).to_dict("records")
        }

    elif action["type"] == "marker_validation":
        result = {
            "identified_markers": list(ALL_KNOWN_GENES)
        }

    else:
        return {
            "success": False,
            "summary": "Unsupported or unexecutable analysis step"
        }

    return {
        "success": True,
        "action_type": action["type"],
        "result": result
    }



"""
def map_suggestion_to_action(
    suggestion: str,
    bio_text: str,
    deg_df: pd.DataFrame,
    traj_df: pd.DataFrame
) -> dict:
    

    suggestion_lower = suggestion.lower()
    analysis = []

    candidates = re.findall(r'\b[A-Z][A-Z0-9]{1,6}\b', suggestion.upper())
    genes = sorted(set(g for g in candidates if g in ALL_KNOWN_GENES))
    

    # gene_pattern = r'\b([A-Z][A-Z0-9]{1,5})\b'
    # genes = re.findall(gene_pattern, suggestion.upper())

    if genes:
        analysis.append(f"Identified target genes: {', '.join(genes)}")

    # ---- Trajectory check ----
    if traj_df is not None and not traj_df.empty:
        gene_col = next((c for c in traj_df.columns if 'gene' in c.lower() or 'id' in c.lower()), None)
        corr_col = next((c for c in traj_df.columns if 'cor' in c.lower() or 'rho' in c.lower()), None)

        if gene_col and corr_col and genes:
            for gene in genes:
                rows = traj_df[traj_df[gene_col] == gene]
                if not rows.empty:
                    corr = rows.iloc[0][corr_col]
                    analysis.append(f"{gene}: pseudotime correlation r={corr:.3f}")

    # ---- DEG check ----
    if deg_df is not None and not deg_df.empty:
        gene_col = next((c for c in deg_df.columns if 'gene' in c.lower() or 'id' in c.lower()), None)
        logfc_col = next((c for c in deg_df.columns if 'logfc' in c.lower()), None)
        pval_col = next((c for c in deg_df.columns if 'pval' in c.lower() or 'adj' in c.lower()), None)

        if gene_col and genes:
            for gene in genes:
                rows = deg_df[deg_df[gene_col] == gene]
                if not rows.empty:
                    row = rows.iloc[0]
                    if logfc_col and pval_col:
                        analysis.append(
                            f"{gene}: logFC={row[logfc_col]:.2f}, p={row[pval_col]:.2e}"
                        )

    # ---- Interpretation ----
    if "validate" in suggestion_lower:
        analysis.append("Validation-focused analysis completed")
    elif "compare" in suggestion_lower:
        analysis.append("Comparative analysis completed")
    elif "analyze" in suggestion_lower:
        analysis.append("Targeted analysis completed")

    if not analysis:
        analysis.append("Analysis executed successfully")

    return {
        "success": True,
        "summary": "\n".join(analysis)
    }



def run_analysis_for_web():
    pipeline_status = {
        "clustering": False,
        "de": False,
        "trajectory": False
    }

    def update_status(step, success):
        pipeline_status[step] = success

    pipeline_results = execute_mandatory_pipeline()

    # if not pipeline_results.get("clustering"):
    #     return {"error": "Clustering failed"}

    bio_text, deg_df, traj_df = load_complete_results()

    hypotheses = generate_comprehensive_hypotheses(
        bio_text, deg_df, traj_df, n=5
    )

    next_steps = get_llm_suggested_next_steps(
        hypotheses, bio_text, deg_df, traj_df, k=5
    )

    return {
        "pipeline": pipeline_status,
        "hypotheses": hypotheses,
        "next_steps": next_steps
    }
"""


def llm_biological_interpretation(
    suggestion: str,
    execution_result: dict,
    bio_text: str
) -> str:
    """
    LLM-based biological interpretation.
    STRICTLY grounded in execution_result.
    """

    system_prompt = """
You are a biomedical domain expert.
You MUST:
- Interpret only the provided results in context to given action.
- NOT invent new genes, pathways, or statistics
- Clearly state uncertainty if data is insufficient
"""

    user_prompt = f"""
USER REQUESTED ACTION:
{suggestion}

EXECUTED RESULTS (GROUND TRUTH):
{json.dumps(execution_result, indent=2)}

BIOLOGICAL CONTEXT:
{bio_text}

TASK:
Provide biological interpretation:
- What do these results suggest biologically?
- Are these patterns expected?
"""

    messages = [
        {"role": "system", "content":  system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = groq_chat(messages, temperature=0.3)
    print("✅ Generated biological interpretation", response)
    return response




def run_analysis_for_web() -> Dict[str, Any]:
    """
    Orchestrate the full analysis for the web UI:
    - Run mandatory pipeline
    - Load results
    - Generate hypotheses
    - Suggest next steps
    Returns a JSON‑serializable dict.
    """
    # 1) Run mandatory pipeline
    pipeline_results = execute_mandatory_pipeline()

    # Normalize to simple status strings for the UI
    pipeline_status = {
        "clustering": (
            "done" if pipeline_results.get("clustering", {}).get("success") else "failed"
        ),
        "differential_expression": (
            "done" if pipeline_results.get("de", {}).get("success") else "failed"
        ),
        "trajectory": (
            "done" if pipeline_results.get("trajectory", {}).get("success") else "failed"
        ),
    }

    # 2) Load all results
    bio_text, deg_df, traj_df = load_complete_results()

    # 3) Generate hypotheses
    hypotheses = generate_comprehensive_hypotheses(
        bio_text, deg_df, traj_df, n=5
    )
    reasoning_context=[]

    # 4) Suggest next steps
    next_steps = get_llm_suggested_next_steps(
        hypotheses, reasoning_context,bio_text, deg_df, traj_df, k=5
    )

    # 5) Return full payload for the UI
    return {
        "pipeline": pipeline_status,
        "hypotheses": hypotheses,
        "next_steps": next_steps,
    }



def run_selected_next_step(
    suggestion: str,
    context: Dict[str, Any]
):
    bio_text = context["bio_text"]
    deg_df = context["deg_df"]
    traj_df = context["traj_df"]

    execution  = map_suggestion_to_action(
        suggestion, bio_text, deg_df, traj_df
    )

    detailed_report = execute_detailed_analysis(
        suggestion, execution 
    )

    biological_interpretation = None
    if execution.get("success"):
        biological_interpretation = llm_biological_interpretation(
            suggestion=suggestion,
            execution_result=execution,
            bio_text=bio_text
        )
        
    hypotheses = get_value()
    if execution.get("success"):
        add_reasoning(
            step=suggestion,
            summary=biological_interpretation,
            result=detailed_report
        )

    # add_reasoning(
    #     step=suggestion,
    #     summary=quick["summary"],
    #     result=detailed_report
    # )
    # reasoning_context = get_reasoning()
    reasoning_context = [
    f"{r['step']}: {r['summary']}"
    for r in get_reasoning()
]



    next_steps = get_llm_suggested_next_steps(
        hypotheses, reasoning_context,bio_text, deg_df, traj_df, k=5
    )

    return {
        "suggestion": suggestion,
        "execution_result": execution,
        "biological_interpretation": biological_interpretation,
        "detailed_report": detailed_report,
        "next_steps": next_steps
    }

