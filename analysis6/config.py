"""
config.py — Configuration for Analysis 6: Reasoning Pattern Analysis.
"""

from pathlib import Path
import matplotlib as mpl

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "evaluation_data"
OUT_DIR  = ROOT / "analysis6" / "results"
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

# ── Qualitative Target Keywords ────────────────────────────────────────────
# Pre-defined theoretical terminology associated with Kohlberg stages
TARGET_KEYWORDS = {
    1: ["punish", "trouble", "caught", "authority", "obey", "jail", "prison", "scolded", "consequence"],
    2: ["fair", "exchange", "right", "interest", "benefit", "transaction", "deserve", "reciprocity"],
    3: ["good", "trust", "relationship", "social", "approval", "society", "expectations", "care"],
    4: ["law", "duty", "order", "rules", "societal", "institution", "obligation", "system"],
    5: ["rights", "contract", "balance", "greater good", "utility", "democratic", "unjust", "welfare"],
    6: ["principle", "justice", "dignity", "universal", "categorical", "imperative", "value of life", "intrinsic"]
}

# English stop words broadly (scikit-learn uses a similar list, custom terms added for LLM context)
CUSTOM_STOP_WORDS = [
    "heinz", "trolley", "doctor", "drug", "lever", "steal", "stealing", "wife", "patient", "die", "life",
    "children", "father", "friend", "store", "owner", "moral", "dilemma", "stage", "kohlberg", "pull", 
    "divert", "train", "five", "one", "people", "would", "could", "should", "must", "might", "therefore",
    "thus", "however", "although", "because", "primary", "secondary", "response", "indicators", "reasoning",
    "action", "endorsed", "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "how", "what",
    "where", "why", "who", "which", "with", "from", "for", "to", "in", "on", "at", "by", "about", "as", "into"
]

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
}

def apply_publication_style() -> None:
    mpl.rcParams.update(PUBLICATION_STYLE)
