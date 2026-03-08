"""
config.py — Configuration for Analysis 9: Capability Correlation Analysis.

Research question: Is there a capability threshold for post-conventional
reasoning (Stage 5+), and which capabilities predict moral reasoning stage?

Capability metrics computed from response text (always available):
  - response_length        : token count (whitespace split)
  - sentence_count         : number of sentences
  - avg_sentence_length    : tokens per sentence
  - lexical_diversity      : type-token ratio (unique / total tokens)
  - syntactic_complexity   : avg_sentence_length × lexical_diversity
  - semantic_density       : abstract/academic vocab coverage
  - coherence              : kohlberg_confidence (1–5) as evaluator proxy
  - post_conv_pct          : % responses with stage >= 5 per model  (model-level)
  - mean_stage             : mean Kohlberg stage per model           (model-level)
"""

from pathlib import Path
import matplotlib as mpl

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluation_data"
OUT_DIR  = ROOT / "analysis9" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model registry ─────────────────────────────────────────────────────────────
# key → (display_name, params_B, provider, scale_group, training_type)
MODEL_META: dict[str, tuple] = {
    "ministral_8b_instruct":     ("Ministral 8B",          8,   "Mistral AI", "Small",  "Base-RLHF"),
    "claude_haiku_3_5":          ("Claude 3.5 Haiku",      20,  "Anthropic",  "Small",  "Base-RLHF"),
    "qwen3_30B_Coder_Instruct":  ("Qwen3-30B Coder",       30,  "Alibaba",    "Small",  "Coding-Tuned"),
    "qwen3_32b":                 ("Qwen3-32B",              32,  "Alibaba",    "Small",  "Base-RLHF"),
    "llama3_70B":                ("Llama 3.3 70B",          70,  "Meta",       "Mid",    "Base-RLHF"),
    "qwen3_80b_instruct":        ("Qwen3-80B",              80,  "Alibaba",    "Mid",    "Base-RLHF"),
    "llama4_scout":              ("Llama 4 Scout 109B",    109,  "Meta",       "Mid",    "Base-RLHF"),
    "gpt-oss-120B":              ("GPT-OSS 120B",          120,  "OpenAI",     "Mid",    "Base-RLHF"),
    "claude_sonnet_4_5":         ("Claude Sonnet 4.5",     175,  "Anthropic",  "Large",  "Base-RLHF"),
    "gpt-4o":                    ("GPT-4o",                200,  "OpenAI",     "Large",  "Base-RLHF"),
    "qwen3_235b_thinking":       ("Qwen3-235B (Think)",    235,  "Alibaba",    "Large",  "Reasoning-Tuned"),
    "deepseek_r1":               ("DeepSeek-R1 671B",      671,  "DeepSeek",   "Large",  "Reasoning-Tuned"),
    "deepseek_v3_1":             ("DeepSeek-V3.1 671B",    671,  "DeepSeek",   "Large",  "Base-RLHF"),
}

# ── Analysis constants ─────────────────────────────────────────────────────────
STAGES            = list(range(1, 7))
CI_LEVEL          = 0.95
ALPHA             = 0.05
POST_CONV_STAGE   = 5          # Stage 5+ is "post-conventional"
POST_CONV_THRESH  = 0.20       # model is "capable" if ≥20% responses at Stage 5+
N_BOOTSTRAP       = 2000       # bootstrap iterations for CIs

# Ordered factor levels
SCALE_ORDER    = ["Small", "Mid", "Large"]
TRAINING_ORDER = ["Base-RLHF", "Coding-Tuned", "Reasoning-Tuned"]

# Model-level capability columns used in all analyses
CAPABILITY_COLS = [
    "coherence",
    "response_length",
    "sentence_count",
    "avg_sentence_length",
    "lexical_diversity",
    "syntactic_complexity",
    "semantic_density",
]

# ── Color palettes ─────────────────────────────────────────────────────────────
SCALE_COLORS: dict[str, str] = {
    "Small": "#6a994e",
    "Mid":   "#e76f51",
    "Large": "#457b9d",
}

SCALE_MARKERS: dict[str, str] = {
    "Small": "o",
    "Mid":   "s",
    "Large": "^",
}

TRAINING_COLORS: dict[str, str] = {
    "Base-RLHF":       "#4e79a7",
    "Coding-Tuned":    "#f28e2b",
    "Reasoning-Tuned": "#e15759",
}

PROVIDER_COLORS: dict[str, str] = {
    "Anthropic":  "#E69F00",
    "OpenAI":     "#56B4E9",
    "Meta":       "#009E73",
    "Mistral AI": "#CC79A7",
    "Alibaba":    "#0072B2",
    "DeepSeek":   "#D55E00",
}

# ── Publication rcParams ───────────────────────────────────────────────────────
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
    "lines.linewidth":       1.8,
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
}


def apply_publication_style() -> None:
    mpl.rcParams.update(PUBLICATION_STYLE)
