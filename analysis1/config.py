"""
config.py — Central configuration for the Scale vs. Moral Reasoning analysis.

Contains:
  - Directory paths
  - MODEL_META registry (file stem → display name, parameter count, provider)
  - Provider colour palette (colorblind-friendly)
  - Kohlberg stage labels
  - Matplotlib rcParams for publication-quality figures
"""

from pathlib import Path
import matplotlib as mpl

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "evaluation_data"
OUT_DIR  = ROOT / "analysis1" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model registry ────────────────────────────────────────────────────────
# key  : Excel file stem (without _evaluation)
# value: (display_name, approx_params_B, provider)
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

# ── Color palettes ─────────────────────────────────────────────────────────
# Okabe-Ito colorblind-safe palette (extended)
PROVIDER_COLORS: dict[str, str] = {
    "Anthropic":  "#E69F00",   # orange
    "OpenAI":     "#56B4E9",   # sky blue
    "Meta":       "#009E73",   # bluish green
    "Mistral AI": "#CC79A7",   # reddish purple
    "Alibaba":    "#0072B2",   # blue
    "DeepSeek":   "#D55E00",   # vermillion
}

# ── Publication-quality matplotlib rcParams ───────────────────────────────
PUBLICATION_STYLE: dict = {
    # Font
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "Palatino"],
    "font.size":         12,
    "axes.titlesize":    14,
    "axes.labelsize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "legend.title_fontsize": 11,

    # Lines
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "lines.linewidth":   1.4,

    # Grid
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "grid.alpha":        0.45,
    "axes.axisbelow":    True,

    # Spines
    "axes.spines.top":   False,
    "axes.spines.right": False,

    # Figure
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.format":    "png",
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,

    # Patches / fill
    "patch.linewidth":   0.6,

    # Legend
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cccccc",
    "legend.borderpad":  0.4,
}


def apply_publication_style() -> None:
    """Apply PUBLICATION_STYLE to global matplotlib rcParams."""
    mpl.rcParams.update(PUBLICATION_STYLE)
