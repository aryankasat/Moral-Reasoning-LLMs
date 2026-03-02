"""
config.py — Central configuration for Analysis 4: Stage Distribution Patterns.
"""

from pathlib import Path
import matplotlib as mpl

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "evaluation_data"
OUT_DIR  = ROOT / "analysis4" / "results"
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

# ── Kohlberg stages ────────────────────────────────────────────────────────
STAGES = [1, 2, 3, 4, 5, 6]

STAGE_LABELS = {
    1: "Stage 1\n(Obedience)",
    2: "Stage 2\n(Self-Interest)",
    3: "Stage 3\n(Conformity)",
    4: "Stage 4\n(Law & Order)",
    5: "Stage 5\n(Social Contract)",
    6: "Stage 6\n(Universal Ethics)",
}
STAGE_LABELS_SHORT = {i: f"S{i}" for i in STAGES}

# ── Human developmental baselines (Colby & Kohlberg, 1987) ────────────────
# Proportions must sum to 1.0 for each group.
HUMAN_DIST: dict[str, dict[int, float]] = {
    "Adult": {
        1: 0.00, 2: 0.00, 3: 0.15, 4: 0.40, 5: 0.35, 6: 0.10,
    },
    "Adolescent": {
        1: 0.05, 2: 0.20, 3: 0.40, 4: 0.30, 5: 0.05, 6: 0.00,
    },
    "Children": {
        1: 0.45, 2: 0.40, 3: 0.15, 4: 0.00, 5: 0.00, 6: 0.00,
    },
}

# Primary adult baseline used for all inferential stats
HUMAN_ADULT = HUMAN_DIST["Adult"]

# ── Dilemma display names ──────────────────────────────────────────────────
DILEMMA_LABELS = {
    "HEINZ_DILEMMA":        "Heinz",
    "LIFEBOAT_DILEMMA":     "Lifeboat",
    "TROLLEY_DILLEMA":      "Trolley",
    "DOCTOR_DILLEMA":       "Doctor",
    "STOLEN_FOOD_DILEMMA":  "Stolen Food",
    "PROMISE_DILEMMA":      "Promise",
}

# ── Prompt types ───────────────────────────────────────────────────────────
PROMPT_ORDER  = ["ZERO_SHOT", "COT", "ROLEPLAY"]
PROMPT_LABELS = {"ZERO_SHOT": "Zero-Shot", "COT": "CoT", "ROLEPLAY": "Roleplay"}

# ── Color palettes (Okabe-Ito colorblind-safe) ─────────────────────────────
PROVIDER_COLORS: dict[str, str] = {
    "Anthropic":  "#E69F00",
    "OpenAI":     "#56B4E9",
    "Meta":       "#009E73",
    "Mistral AI": "#CC79A7",
    "Alibaba":    "#0072B2",
    "DeepSeek":   "#D55E00",
}

# Stage fill colours — gradient red→green across 6 stages
STAGE_COLORS: dict[int, str] = {
    1: "#d73027",   # deep red    (Obedience)
    2: "#fc8d59",   # orange-red  (Self-Interest)
    3: "#fee090",   # yellow      (Conformity)
    4: "#91bfdb",   # light blue  (Law & Order)
    5: "#4575b4",   # blue        (Social Contract)
    6: "#1a237e",   # deep indigo (Universal Ethics)
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
