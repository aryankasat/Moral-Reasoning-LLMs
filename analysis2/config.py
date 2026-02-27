"""
config.py — Central configuration for the Alignment Training vs Moral Reasoning
analysis (analysis2).

Alignment taxonomy
------------------
  Instruction-Tuned (IT) : Supervised fine-tuning on instructions; no RL signal.
  RLHF / RL-Aligned      : Reinforcement Learning from Human Feedback, Constitutional
                            AI, or RL-based reasoning training.

Within-family comparison pairs
-------------------------------
  Mistral     : Mistral Tiny 7B  (IT)  vs Ministral 8B   (IT)  — same family, gen step
  DeepSeek    : DeepSeek-V3.1    (IT)  vs DeepSeek-R1    (RL)  — direct IT vs RL
  Claude      : Claude 3.5 Haiku (RLHF) vs Claude Sonnet 4.5 (RLHF) — capability tiers
  OpenAI      : GPT-OSS 120B     (IT)  vs GPT-4o         (RLHF) — training method
  Llama       : Llama 3.3 70B    (IT)  vs Llama 4 Scout  (IT)  — generation step
  Qwen3       : Qwen3-32B (IT) vs Qwen3-80B (IT) vs Qwen3-235B-Think (RL) — scale+RL
"""

from pathlib import Path
import matplotlib as mpl

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "evaluation_data"
OUT_DIR  = ROOT / "analysis2" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Alignment type constants ──────────────────────────────────────────────
IT   = "Instruction-Tuned"
RLHF = "RLHF / RL-Aligned"

# ── Model registry ────────────────────────────────────────────────────────
# key  : Excel file stem (without _evaluation)
# value: (display_name, params_B, family, alignment_type)
MODEL_META: dict[str, tuple[str, float, str, str]] = {
    "mistral_tiny":             ("Mistral Tiny 7B",       7,   "Mistral",  IT),
    "ministral_8b_instruct":    ("Ministral 8B",          8,   "Mistral",  IT),
    "llama3_70B":               ("Llama 3.3 70B",        70,   "Llama",    IT),
    "llama4_scout":             ("Llama 4 Scout 109B",  109,   "Llama",    IT),
    "qwen3_30B_Coder_Instruct": ("Qwen3-30B Coder",      30,   "Qwen3",    IT),
    "qwen3_32b":                ("Qwen3-32B",            32,   "Qwen3",    IT),
    "qwen3_80b_instruct":       ("Qwen3-80B",            80,   "Qwen3",    IT),
    "qwen3_235b_thinking":      ("Qwen3-235B (Think)",  235,   "Qwen3",    RLHF),
    "deepseek_v3_1":            ("DeepSeek-V3.1",       671,   "DeepSeek", IT),
    "deepseek_r1":              ("DeepSeek-R1",         671,   "DeepSeek", RLHF),
    "gpt-oss-120B":             ("GPT-OSS 120B",        120,   "OpenAI",   IT),
    "gpt-4o":                   ("GPT-4o",              200,   "OpenAI",   RLHF),
    "claude_haiku_3_5":         ("Claude 3.5 Haiku",     20,   "Claude",   RLHF),
    "claude_sonnet_4_5":        ("Claude Sonnet 4.5",   175,   "Claude",   RLHF),
}

# Within-family comparison pairs: (stem_A, stem_B, label)
# A = less-aligned / smaller, B = more-aligned / larger
FAMILY_PAIRS: list[tuple[str, str, str]] = [
    ("deepseek_v3_1",    "deepseek_r1",       "DeepSeek  IT → RL"),
    ("mistral_tiny",     "ministral_8b_instruct", "Mistral  7B → 8B (IT)"),
    ("llama3_70B",       "llama4_scout",       "Llama  Gen3 → Gen4 (IT)"),
    ("gpt-oss-120B",     "gpt-4o",             "OpenAI  IT → RLHF"),
    ("claude_haiku_3_5", "claude_sonnet_4_5",  "Claude  Haiku → Sonnet (RLHF)"),
    ("qwen3_32b",        "qwen3_235b_thinking","Qwen3  IT → Think-RL"),
]

# ── Colour palette ────────────────────────────────────────────────────────
# Alignment-type colours (colorblind-safe)
ALIGN_COLORS: dict[str, str] = {
    IT:   "#56B4E9",   # sky blue
    RLHF: "#E69F00",   # orange
}

# Family colours
FAMILY_COLORS: dict[str, str] = {
    "Mistral":  "#CC79A7",
    "Llama":    "#009E73",
    "Qwen3":    "#0072B2",
    "DeepSeek": "#D55E00",
    "OpenAI":   "#56B4E9",
    "Claude":   "#E69F00",
}

# ── Kohlberg helpers ──────────────────────────────────────────────────────
ALL_STAGES = list(range(1, 7))
POST_CONV_THRESHOLD = 5   # Stage 5+ = post-conventional

# ── Publication rcParams ──────────────────────────────────────────────────
PUBLICATION_STYLE: dict = {
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "DejaVu Serif"],
    "font.size":           10,
    "axes.titlesize":      12,
    "axes.labelsize":      11,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.fontsize":     9,
    "legend.title_fontsize": 9,
    "axes.linewidth":      0.8,
    "lines.linewidth":     1.4,
    "axes.grid":           True,
    "grid.linestyle":      "--",
    "grid.linewidth":      0.5,
    "grid.alpha":          0.45,
    "axes.axisbelow":      True,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.05,
    "patch.linewidth":     0.6,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#cccccc",
}


def apply_publication_style() -> None:
    mpl.rcParams.update(PUBLICATION_STYLE)
