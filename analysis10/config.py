"""
config.py — Configuration for Analysis 10: Stage Transition Dynamics.

Research question: How do models transition between stages as scale increases—
gradually or suddenly? Do they consolidate at stages before progressing?

Adaptation note: Since evaluation_data contains modern LLMs (not Pythia
checkpoints), models are ordered by parameter count (params_B) as the
"scale progression" axis — analogous to training steps.
"""

from pathlib import Path
import matplotlib as mpl

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluation_data"
OUT_DIR  = ROOT / "analysis10" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model registry ─────────────────────────────────────────────────────────────
# key → (display_name, params_B, provider, scale_group, training_type)
# Ordered by params_B ascending — this is our "progression axis"
MODEL_META: dict[str, tuple] = {
    "ministral_8b_instruct":     ("Ministral 8B",        8,   "Mistral AI", "Small",  "Base-RLHF"),
    "claude_haiku_3_5":          ("Claude Haiku 3.5",    20,  "Anthropic",  "Small",  "Base-RLHF"),
    "qwen3_30B_Coder_Instruct":  ("Qwen3-30B Coder",     30,  "Alibaba",    "Small",  "Coding-Tuned"),
    "qwen3_32b":                 ("Qwen3-32B",            32,  "Alibaba",    "Small",  "Base-RLHF"),
    "llama3_70B":                ("Llama 3.3 70B",        70,  "Meta",       "Mid",    "Base-RLHF"),
    "qwen3_80b_instruct":        ("Qwen3-80B",            80,  "Alibaba",    "Mid",    "Base-RLHF"),
    "llama4_scout":              ("Llama 4 Scout 109B",  109,  "Meta",       "Mid",    "Base-RLHF"),
    "gpt-oss-120B":              ("GPT-OSS 120B",        120,  "OpenAI",     "Mid",    "Base-RLHF"),
    "claude_sonnet_4_5":         ("Claude Sonnet 4.5",   175,  "Anthropic",  "Large",  "Base-RLHF"),
    "gpt-4o":                    ("GPT-4o",              200,  "OpenAI",     "Large",  "Base-RLHF"),
    "qwen3_235b_thinking":       ("Qwen3-235B (Think)",  235,  "Alibaba",    "Large",  "Reasoning-Tuned"),
    "deepseek_r1":               ("DeepSeek-R1 671B",    671,  "DeepSeek",   "Large",  "Reasoning-Tuned"),
    "deepseek_v3_1":             ("DeepSeek-V3.1 671B",  671,  "DeepSeek",   "Large",  "Base-RLHF"),
}

# Models in ascending param order (the "progression" sequence)
MODEL_ORDER: list[str] = [
    "ministral_8b_instruct",
    "claude_haiku_3_5",
    "qwen3_30B_Coder_Instruct",
    "qwen3_32b",
    "llama3_70B",
    "qwen3_80b_instruct",
    "llama4_scout",
    "gpt-oss-120B",
    "claude_sonnet_4_5",
    "gpt-4o",
    "qwen3_235b_thinking",
    "deepseek_r1",
    "deepseek_v3_1",
]

# Short display labels (for crowded axis ticks)
SHORT_NAMES: dict[str, str] = {
    "ministral_8b_instruct":    "Mistral\n8B",
    "claude_haiku_3_5":         "Haiku\n3.5",
    "qwen3_30B_Coder_Instruct": "Qwen3\n30B-Cdr",
    "qwen3_32b":                "Qwen3\n32B",
    "llama3_70B":               "Llama3\n70B",
    "qwen3_80b_instruct":       "Qwen3\n80B",
    "llama4_scout":             "Llama4\nScout",
    "gpt-oss-120B":             "GPT-OSS\n120B",
    "claude_sonnet_4_5":        "Sonnet\n4.5",
    "gpt-4o":                   "GPT-4o\n200B",
    "qwen3_235b_thinking":      "Qwen3\n235B",
    "deepseek_r1":              "DS-R1\n671B",
    "deepseek_v3_1":            "DS-V3.1\n671B",
}

# ── Analysis constants ─────────────────────────────────────────────────────────
STAGES            = list(range(1, 7))   # Kohlberg stages 1-6
ACTIVE_STAGES     = [4, 5, 6]           # Stages actually present in data
POST_CONV_STAGE   = 5
ALPHA             = 0.05
N_BOOTSTRAP       = 2000
MAX_ENTROPY       = 2.585               # log₂(6)

# Transition window thresholds (from problem statement)
NEXT_STAGE_APPEAR_THRESH = 0.10        # >10% next stage → start of transition
CURR_STAGE_EXIT_THRESH   = 0.30        # <30% current stage → end of transition

# ── Okabe-Ito palette (colour-blind safe) ─────────────────────────────────────
OI = {
    "orange":      "#E69F00",
    "sky_blue":    "#56B4E9",
    "green":       "#009E73",
    "yellow":      "#F0E442",
    "blue":        "#0072B2",
    "vermillion":  "#D55E00",
    "pink":        "#CC79A7",
    "black":       "#000000",
}

STAGE_COLORS: dict[int, str] = {
    1: OI["pink"],
    2: OI["orange"],
    3: OI["yellow"],
    4: OI["green"],         # Stage 4
    5: OI["sky_blue"],      # Stage 5
    6: OI["vermillion"],    # Stage 6
}

SCALE_COLORS: dict[str, str] = {
    "Small": OI["blue"],
    "Mid":   OI["vermillion"],
    "Large": OI["green"],
}

TRAINING_COLORS: dict[str, str] = {
    "Base-RLHF":       OI["blue"],
    "Coding-Tuned":    OI["orange"],
    "Reasoning-Tuned": OI["vermillion"],
}

# Figure geometry constants
SINGLE_COL = (3.5, 2.8)   # inches — single journal column
DOUBLE_COL = (7.2, 3.6)   # inches — double journal column
TALL_SINGLE = (3.5, 4.0)
TALL_DOUBLE = (7.2, 4.8)

# ── Publication rcParams ───────────────────────────────────────────────────────
PUBLICATION_STYLE: dict = {
    "font.family":          "DejaVu Sans",
    "font.size":            9,
    "axes.titlesize":       10,
    "axes.labelsize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8,
    "legend.title_fontsize":8,
    "axes.linewidth":       0.7,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.linestyle":       ":",
    "grid.linewidth":       0.4,
    "grid.alpha":           0.55,
    "grid.color":           "#bbbbbb",
    "axes.axisbelow":       True,
    "lines.linewidth":      1.6,
    "lines.markersize":     6,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.02,
}


def apply_publication_style() -> None:
    mpl.rcParams.update(PUBLICATION_STYLE)
