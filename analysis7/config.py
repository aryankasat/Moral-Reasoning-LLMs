"""
config.py — Configuration for Analysis 7: Emergence Threshold Detection.

Adaptation note:
  The originally prescribed methodology uses Pythia training checkpoints along a
  training-tokens axis. This project contains commercial / open-weight LLMs evaluated
  on fixed moral-dilemma prompts. We therefore use **parameter count (log scale)** as
  the primary scale axis, which captures the same emergence questions in a valid and
  scientifically meaningful way.
"""

from pathlib import Path
import matplotlib as mpl

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluation_data"
OUT_DIR  = ROOT / "analysis7" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model registry ─────────────────────────────────────────────────────────────
# key → (display_name, params_B, provider)
MODEL_META: dict[str, tuple[str, float, str]] = {
    "ministral_8b_instruct":     ("Ministral 8B",           8,    "Mistral AI"),
    "claude_haiku_3_5":          ("Claude 3.5 Haiku",       20,   "Anthropic"),
    "qwen3_30B_Coder_Instruct":  ("Qwen3-30B Coder",        30,   "Alibaba"),
    "qwen3_32b":                 ("Qwen3-32B",               32,   "Alibaba"),
    "llama3_70B":                ("Llama 3.3 70B",           70,   "Meta"),
    "qwen3_80b_instruct":        ("Qwen3-80B",               80,   "Alibaba"),
    "llama4_scout":              ("Llama 4 Scout 109B",     109,   "Meta"),
    "gpt-oss-120B":              ("GPT-OSS 120B",           120,   "OpenAI"),
    "claude_sonnet_4_5":         ("Claude Sonnet 4.5",      175,   "Anthropic"),
    "gpt-4o":                    ("GPT-4o",                 200,   "OpenAI"),
    "qwen3_235b_thinking":       ("Qwen3-235B (Think)",     235,   "Alibaba"),
    "deepseek_r1":               ("DeepSeek-R1 671B",       671,   "DeepSeek"),
    "deepseek_v3_1":             ("DeepSeek-V3.1 671B",     671,   "DeepSeek"),
}

# ── Scale groups for three-panel figure ──────────────────────────────────────
SCALE_GROUPS = {
    "Small (8B–32B)":   [8,  20, 30, 32],
    "Mid   (70B–120B)": [70, 80, 109, 120],
    "Large (175B–671B)":[175, 200, 235, 671],
}

# ── Analysis constants ─────────────────────────────────────────────────────────
STAGES              = list(range(1, 7))
POST_CONV_THRESHOLD = 0.20   # ≥20 % responses at Stage 5+ = "post-conventional emerged"
BOOTSTRAP_ITERS     = 2000   # bootstrap iterations for changepoint CI
CI_LEVEL            = 0.95   # confidence interval level

# Minimum penalty for PELT changepoint detection (lower = more changepoints)
PELT_PENALTY        = "bic"  # use BIC criterion

# ── Color palettes ─────────────────────────────────────────────────────────────
PROVIDER_COLORS: dict[str, str] = {
    "Anthropic":  "#E69F00",
    "OpenAI":     "#56B4E9",
    "Meta":       "#009E73",
    "Mistral AI": "#CC79A7",
    "Alibaba":    "#0072B2",
    "DeepSeek":   "#D55E00",
}

STAGE_COLORS: dict[int, str] = {
    1: "#d73027",
    2: "#fc8d59",
    3: "#fee090",
    4: "#91bfdb",
    5: "#4575b4",
    6: "#1a237e",
}

SCALE_GROUP_COLORS = {
    "Small (8B–32B)":   "#6a994e",
    "Mid   (70B–120B)": "#e76f51",
    "Large (175B–671B)":"#457b9d",
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
    "lines.linewidth":       1.6,
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
