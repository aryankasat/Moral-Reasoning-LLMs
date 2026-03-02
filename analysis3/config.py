"""
config.py — Central configuration for the Consistency / Stability Analysis (Analysis 3).

Contains:
  - Directory paths
  - MODEL_META registry
  - Provider and prompt-type colour palettes (colorblind-friendly)
  - Kohlberg stage labels
  - Matplotlib rcParams for publication-quality figures
"""

from pathlib import Path
import matplotlib as mpl

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "evaluation_data"
OUT_DIR  = ROOT / "analysis3" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model registry ────────────────────────────────────────────────────────
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

# ── Kohlberg stage labels ─────────────────────────────────────────────────
STAGE_LABELS = {
    1: "Stage 1\n(Obedience)",
    2: "Stage 2\n(Self-Interest)",
    3: "Stage 3\n(Conformity)",
    4: "Stage 4\n(Law & Order)",
    5: "Stage 5\n(Social Contract)",
    6: "Stage 6\n(Universal Ethics)",
}

STAGE_LABELS_SHORT = {i: f"S{i}" for i in range(1, 7)}

# ── Dilemma display names ─────────────────────────────────────────────────
DILEMMA_LABELS = {
    "HEINZ_DILEMMA":        "Heinz",
    "LIFEBOAT_DILEMMA":     "Lifeboat",
    "TROLLEY_DILLEMA":      "Trolley",
    "DOCTOR_DILLEMA":       "Doctor",
    "STOLEN_FOOD_DILEMMA":  "Stolen Food",
    "PROMISE_DILEMMA":      "Promise",
}

# ── Prompt type display names & order ─────────────────────────────────────
PROMPT_ORDER    = ["ZERO_SHOT", "COT", "ROLEPLAY"]
PROMPT_LABELS   = {"ZERO_SHOT": "Zero-Shot", "COT": "CoT", "ROLEPLAY": "Roleplay"}

# ── Color palettes ─────────────────────────────────────────────────────────
# Okabe-Ito colorblind-safe palette
PROVIDER_COLORS: dict[str, str] = {
    "Anthropic":  "#E69F00",
    "OpenAI":     "#56B4E9",
    "Meta":       "#009E73",
    "Mistral AI": "#CC79A7",
    "Alibaba":    "#0072B2",
    "DeepSeek":   "#D55E00",
}

PROMPT_COLORS: dict[str, str] = {
    "ZERO_SHOT": "#56B4E9",   # sky blue
    "COT":       "#E69F00",   # orange
    "ROLEPLAY":  "#009E73",   # bluish green
}

# ── Publication-quality matplotlib rcParams ───────────────────────────────
PUBLICATION_STYLE: dict = {
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif", "Palatino"],
    "font.size":            10,
    "axes.titlesize":       12,
    "axes.labelsize":       11,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      9,
    "legend.title_fontsize": 9,

    "axes.linewidth":       0.8,
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,
    "xtick.minor.width":    0.5,
    "ytick.minor.width":    0.5,
    "lines.linewidth":      1.4,

    "axes.grid":            True,
    "grid.linestyle":       "--",
    "grid.linewidth":       0.5,
    "grid.alpha":           0.45,
    "axes.axisbelow":       True,

    "axes.spines.top":      False,
    "axes.spines.right":    False,

    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.format":       "png",
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,

    "patch.linewidth":      0.6,

    "legend.framealpha":    0.9,
    "legend.edgecolor":     "#cccccc",
    "legend.borderpad":     0.4,
}


def apply_publication_style() -> None:
    """Apply PUBLICATION_STYLE to global matplotlib rcParams."""
    mpl.rcParams.update(PUBLICATION_STYLE)
