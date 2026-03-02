"""
config.py — Configuration for Analysis 5: Action-Reasoning Consistency.
"""

from pathlib import Path
import matplotlib as mpl

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "evaluation_data"
OUT_DIR  = ROOT / "analysis5" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model registry ─────────────────────────────────────────────────────────
MODEL_META: dict[str, tuple[str, float, str]] = {
    "mistral_tiny":              ("Mistral Tiny 7B",         7,    "Mistral AI"),
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

STAGES = [1, 2, 3, 4, 5, 6]

# ── Action Categories ──────────────────────────────────────────────────────
# Two main bins for analyzing consistency
ACTION_CATEGORIES = ["Rule-Following", "Rule-Breaking"]

# Expected mapping based on moral stage:
# S1-2: Often Rule-Following (avoid punishment) but can be Breaking if self-interest outweighs risk.
# S3-4: Rule-Following (laws, rules, duties).
# S5-6: Principled Rule-Breaking when life/dignity > property (for these specific dilemmas).
EXPECTED_ACTION_BY_STAGE = {
    1: "Rule-Following",  # Assuming basic fear of punishment
    2: "Rule-Following",  # Unless explicit exchange, usually compliant
    3: "Rule-Following",  # Good boy/girl, conform to expectations
    4: "Rule-Following",  # Law and order
    5: "Rule-Breaking",   # Social contract (life > property in Heinz/StolenFood)
    6: "Rule-Breaking",   # Universal ethical principles (highest value on life)
}

# ── Color palettes ─────────────────────────────────────────────────────────
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

ACTION_COLORS: dict[str, str] = {
    "Rule-Following": "#009E73",  # Green
    "Rule-Breaking":  "#D55E00",  # Vermilion
    "Ambiguous/Other": "#999999", # Grey
}

CONSISTENCY_COLORS: dict[str, str] = {
    "Consistent": "#0072B2",   # Blue
    "Inconsistent": "#D55E00", # Vermilion
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
