"""
data_collector.py — Collects moral dilemma responses for Analysis 11.

Routing:
  BASE models    → HuggingFace Serverless Inference API  (text completion)
  INSTRUCT models → Groq API (Llama, Qwen) or Mistral AI API (Mistral)

Run modes:
  python rlhf_causal_analysis/data_collector.py                # collect all pairs
  python rlhf_causal_analysis/data_collector.py --pair llama31_8b
  python rlhf_causal_analysis/data_collector.py --dry-run      # API connectivity test only
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "LLM calls"))
sys.path.insert(0, str(ROOT / "rlhf_causal_analysis"))

load_dotenv(ROOT / ".env")

from prompt_hub import Dilemma_types, Prompt_Types
from config import (
    MODEL_PAIRS, PAIR_ORDER, DATA_DIR,
    BASE_MODEL_WRAPPER, BASE_MODEL_PARAMS,
    INSTRUCT_SYSTEM_PROMPT,
)


# ── HuggingFace Inference API client ──────────────────────────────────────────

def _hf_text_generation(
    hf_model_id: str,
    prompt: str,
    hf_token: str,
    params: dict | None = None,
    max_retries: int = 4,
    retry_delay: float = 20.0,
) -> str:
    """
    Call HuggingFace serverless Inference API for text generation.
    Handles model cold-start (503) with retries.
    """
    import requests

    url = f"https://api-inference.huggingface.co/models/{hf_model_id}"
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload: dict = {"inputs": prompt}
    if params:
        payload["parameters"] = params

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                # returns list of dicts with "generated_text"
                if isinstance(data, list) and data:
                    return data[0].get("generated_text", "")
                return str(data)
            elif resp.status_code == 503:
                # Model loading — wait and retry
                est = resp.json().get("estimated_time", retry_delay)
                wait = max(float(est), retry_delay)
                print(f"      [HF] Model loading (~{wait:.0f}s). Attempt {attempt}/{max_retries}…")
                time.sleep(wait)
            elif resp.status_code == 429:
                print(f"      [HF] Rate limited. Waiting {retry_delay*2}s…")
                time.sleep(retry_delay * 2)
            else:
                print(f"      [HF] HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(retry_delay)
        except Exception as exc:
            print(f"      [HF] Exception: {exc}. Retrying in {retry_delay}s…")
            time.sleep(retry_delay)

    return "[HF_ERROR: max retries exceeded]"


def _hf_chat_completion(
    hf_model_id: str,
    prompt: str,
    hf_token: str,
    max_retries: int = 4,
    retry_delay: float = 20.0,
) -> str:
    """
    Call HuggingFace serverless Inference API for chat (instruct) models.
    Uses the /v1/chat/completions endpoint.
    """
    import requests

    url = f"https://api-inference.huggingface.co/models/{hf_model_id}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": INSTRUCT_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": 600,
        "temperature": 0.7,
        "stream": False,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            elif resp.status_code == 503:
                est = resp.json().get("estimated_time", retry_delay)
                wait = max(float(est), retry_delay)
                print(f"      [HF-Chat] Model loading (~{wait:.0f}s). Attempt {attempt}/{max_retries}…")
                time.sleep(wait)
            elif resp.status_code == 429:
                print(f"      [HF-Chat] Rate limited. Waiting {retry_delay*2}s…")
                time.sleep(retry_delay * 2)
            else:
                print(f"      [HF-Chat] HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(retry_delay)
        except Exception as exc:
            print(f"      [HF-Chat] Exception: {exc}. Retrying in {retry_delay}s…")
            time.sleep(retry_delay)

    return "[HF_CHAT_ERROR: max retries exceeded]"


# ── Groq chat completion ───────────────────────────────────────────────────────

def _groq_chat_completion(model_id: str, prompt: str, groq_api_key: str) -> str:
    """Call Groq API for RLHF-instruct models (Llama, Qwen)."""
    from groq import Groq

    client = Groq(api_key=groq_api_key)
    full_response = ""
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": INSTRUCT_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.7,
        max_tokens=600,
        stream=True,
    )
    for chunk in completion:
        full_response += chunk.choices[0].delta.content or ""
    return full_response


# ── Mistral AI chat completion ─────────────────────────────────────────────────

def _mistral_chat_completion(model_id: str, prompt: str, mistral_api_key: str) -> str:
    """Call Mistral AI API for instruct (RLHF) counterpart."""
    from mistralai import Mistral

    client = Mistral(api_key=mistral_api_key)
    resp = client.chat.complete(
        model=model_id,
        messages=[
            {"role": "system", "content": INSTRUCT_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.7,
        max_tokens=600,
    )
    return resp.choices[0].message.content


# ── Collect one pair ───────────────────────────────────────────────────────────

def collect_pair(
    pair_id: str,
    pair_cfg: dict,
    hf_token: str,
    groq_api_key: str,
    mistral_api_key: str,
    dry_run: bool = False,
) -> None:
    """
    Collect responses for both BASE and INSTRUCT variants of one pair.
    Saves to DATA_DIR/<pair_id>_base.xlsx and DATA_DIR/<pair_id>_instruct.xlsx.
    """
    print(f"\n{'='*60}")
    print(f"  Pair: {pair_id}  [{pair_cfg['architecture']} {pair_cfg['params_B']}B]")
    print(f"{'='*60}")

    for variant in ("base", "instruct"):
        out_file = DATA_DIR / f"{pair_id}_{variant}.xlsx"
        if out_file.exists():
            print(f"  [{variant.upper()}] ✅ Already collected → {out_file.name} (skip)")
            continue

        label = pair_cfg.get(f"{variant}_label", f"{pair_id}_{variant}")
        print(f"\n  [{variant.upper()}] Collecting: {label}")

        if dry_run:
            print(f"    [DRY RUN] Would collect from {'HF' if variant == 'base' else pair_cfg.get('instruct_api_src','?').upper()}")
            continue

        rows = []
        for prompt_item in Prompt_Types:
            for dilemma in Dilemma_types:
                prompt_text = f"{dilemma.value}\n\n{prompt_item.value}"

                if variant == "base":
                    # Wrap dilemma for text-completion style
                    full_prompt = BASE_MODEL_WRAPPER.format(dilemma=prompt_text)
                    hf_id = pair_cfg["base_hf_id"]
                    print(f"    → {dilemma.name} / {prompt_item.name} [HF text-gen: {hf_id}]")
                    start = time.time()
                    response = _hf_text_generation(hf_id, full_prompt, hf_token, BASE_MODEL_PARAMS)
                    elapsed = round(time.time() - start, 2)
                    api_src = "huggingface"

                else:
                    # Instruct variant — route to correct API
                    api_src  = pair_cfg.get("instruct_api_src", "groq")
                    print(f"    → {dilemma.name} / {prompt_item.name} [{api_src.upper()}]")
                    start = time.time()

                    if api_src == "groq":
                        response = _groq_chat_completion(
                            pair_cfg["instruct_api_id"], prompt_text, groq_api_key
                        )
                    elif api_src == "mistral":
                        response = _mistral_chat_completion(
                            pair_cfg["instruct_api_id"], prompt_text, mistral_api_key
                        )
                    elif api_src == "hf":
                        hf_id = pair_cfg.get("instruct_hf_id", "")
                        response = _hf_chat_completion(hf_id, prompt_text, hf_token)
                    else:
                        response = f"[UNKNOWN API SOURCE: {api_src}]"

                    elapsed = round(time.time() - start, 2)

                rows.append({
                    "pair_id":          pair_id,
                    "architecture":     pair_cfg["architecture"],
                    "params_B":         pair_cfg["params_B"],
                    "variant":          variant,
                    "model_label":      label,
                    "dilemma_type":     dilemma.name,
                    "prompt_type":      prompt_item.name,
                    "prompt":           prompt_text,
                    "response":         response,
                    "response_length":  len(response),
                    "inference_time":   elapsed,
                    "api_source":       api_src if variant == "instruct" else "huggingface",
                    "timestamp":        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

                print(f"       ✓ {elapsed}s  len={len(response)}")
                # Brief pause to respect rate limits
                time.sleep(1.5)

        df = pd.DataFrame(rows)
        df.to_excel(out_file, index=False)
        print(f"\n  [{variant.upper()}] ✅ Saved {len(df)} rows → {out_file.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect base vs. instruct model responses (Analysis 11)")
    parser.add_argument("--pair",    default=None, help="Restrict to one pair_id (e.g. llama31_8b)")
    parser.add_argument("--dry-run", action="store_true", help="Test API connectivity without LLM calls")
    args = parser.parse_args()

    hf_token       = os.getenv("HF_TOKEN", "")
    groq_api_key   = os.getenv("GROQ_API_KEY", "")
    mistral_api_key = os.getenv("MISTRAL_API_KEY", "")

    if not args.dry_run:
        if not hf_token:
            print("❌ HF_TOKEN is not set in .env — required for base model inference.")
            print("   Get your token at: https://huggingface.co/settings/tokens")
            sys.exit(1)
        if not groq_api_key:
            print("⚠️  GROQ_API_KEY not set — Groq-routed instruct models will fail.")
        if not mistral_api_key:
            print("⚠️  MISTRAL_API_KEY not set — Mistral instruct model will fail.")
    else:
        print("🔍 DRY RUN — checking imports and configuration only (no API calls).")

    pairs_to_run = PAIR_ORDER if args.pair is None else [args.pair]
    if args.pair and args.pair not in MODEL_PAIRS:
        print(f"❌ Unknown pair: '{args.pair}'. Valid options: {list(MODEL_PAIRS.keys())}")
        sys.exit(1)

    for pair_id in pairs_to_run:
        collect_pair(
            pair_id       = pair_id,
            pair_cfg      = MODEL_PAIRS[pair_id],
            hf_token      = hf_token,
            groq_api_key  = groq_api_key,
            mistral_api_key = mistral_api_key,
            dry_run       = args.dry_run,
        )

    if args.dry_run:
        print("\n✅ Dry run complete — all imports and config are valid.")
    else:
        print("\n✅ Data collection complete. Proceed with: python rlhf_causal_analysis/evaluator.py")


if __name__ == "__main__":
    main()
