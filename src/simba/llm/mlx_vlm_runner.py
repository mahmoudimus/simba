"""Text-only generation on an MLX-VLM model, printing ONLY the generated text.

Gemma 3/4 (>=4B) ship as multimodal checkpoints, so the text-only
``mlx_lm.generate`` can't load them. ``mlx_vlm`` runs them text-only, but its CLI
echoes the templated prompt + timing stats to stdout. This thin wrapper calls the
``mlx_vlm`` Python API instead and writes only the generation to stdout, so
``simba.llm.client``'s ``mlx-vlm`` provider gets clean output.

Invoked as ``python -m simba.llm.mlx_vlm_runner --model <repo> --max-tokens N
--temperature T`` with the prompt on **stdin**. Fail-open: any error → exit 1
with no stdout (the LlmClient treats a nonzero exit as "").
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args, _ = ap.parse_known_args()

    prompt = sys.stdin.read()
    if not prompt.strip():
        return 1

    try:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        model, processor = load(args.model)
        config = load_config(args.model)
        formatted = apply_chat_template(processor, config, prompt, num_images=0)
        result = generate(
            model,
            processor,
            formatted,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verbose=False,
        )
        text = getattr(result, "text", None)
        if text is None:
            text = str(result)
        sys.stdout.write(text)
        return 0
    except Exception as exc:
        print(f"mlx_vlm_runner: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
