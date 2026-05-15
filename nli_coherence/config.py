"""
config.py — Configuration for NLI-based Coherence Measure.

This analysis uses DeBERTa-v3-large (via a fine-tuned NLI variant) to score
entailment between each model's stated justification and its endorsed action,
entirely independent of the Kohlberg framework.
"""

from pathlib import Path
import matplotlib as mpl

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "evaluation_data"
OUT_DIR  = ROOT / "nli_coherence" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# McNemar p-values from analysis5 (decoupling scores)
MCNEMAR_CSV = ROOT / "analysis5" / "results" / "mcnemar_per_model.csv"

# ── HuggingFace Model ─────────────────────────────────────────────────────
# Fine-tuned DeBERTa-v3-large for NLI / zero-shot classification
NLI_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"

# API settings
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY    = 5   # seconds between retries
API_BATCH_DELAY    = 0.5 # seconds between API calls to respect rate limits

# ── Model registry (mirrored from analysis5 for consistency) ───────────────
MODEL_META: dict[str, tuple[str, float, str]] = {
    "ministral_8b_instruct":     ("Ministral 8B",            8,    "Mistral AI"),
    "claude_haiku_3_5":          ("Claude 3.5 Haiku",       20,    "Anthropic"),
    "qwen3_30B_Coder_Instruct":  ("Qwen3-30B Coder",        30,    "Alibaba"),
    "qwen3_32b":                 ("Qwen3-32B",               32,    "Alibaba"),
    "llama3_70B":                ("Llama 3.3 70B",           70,    "Meta"),
    "qwen3_80b_instruct":        ("Qwen3-80B",               80,    "Alibaba"),
    "llama4_scout":              ("Llama 4 Scout 109B",     109,    "Meta"),
    "gpt-oss-120B":              ("GPT-OSS 120B",           120,    "OpenAI"),
    "claude_sonnet_4_5":         ("Claude Sonnet 4.5",      175,    "Anthropic"),
    "gpt-4o":                    ("GPT-4o",                 200,    "OpenAI"),
    "qwen3_235b_thinking":       ("Qwen3-235B (Think)",     235,    "Alibaba"),
    "deepseek_r1":               ("DeepSeek-R1 671B",       671,    "DeepSeek"),
    "deepseek_v3_1":             ("DeepSeek-V3.1 671B",     671,    "DeepSeek"),
}

# ── Provider color palette ─────────────────────────────────────────────────
PROVIDER_COLORS: dict[str, str] = {
    "Anthropic":  "#E69F00",
    "OpenAI":     "#56B4E9",
    "Meta":       "#009E73",
    "Mistral AI": "#CC79A7",
    "Alibaba":    "#0072B2",
    "DeepSeek":   "#D55E00",
}

# ── Publication rcParams ───────────────────────────────────────────────────
PUBLICATION_STYLE: dict = {
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "DejaVu Serif", "Palatino"],
    "font.size":             10,
    "axes.titlesize":        12,
    "axes.labelsize":        11,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "legend.fontsize":       9,
    "legend.title_fontsize": 9,
    "axes.linewidth":        0.8,
    "xtick.major.width":     0.8,
    "ytick.major.width":     0.8,
    "lines.linewidth":       1.4,
    "axes.grid":             True,
    "grid.linestyle":        "--",
    "grid.linewidth":        0.5,
    "grid.alpha":            0.45,
    "axes.axisbelow":        True,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.format":        "png",
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.05,
    "patch.linewidth":       0.6,
    "legend.framealpha":     0.9,
    "legend.edgecolor":      "#cccccc",
    "legend.borderpad":      0.4,
}

def apply_publication_style() -> None:
    mpl.rcParams.update(PUBLICATION_STYLE)
