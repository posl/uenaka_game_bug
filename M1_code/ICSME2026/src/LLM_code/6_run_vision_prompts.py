#!/usr/bin/env python3
"""Read pre-built prompt JSONL and call the OpenAI Responses API for each record."""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from estimate_image_costs import calculate_cost_usd

load_dotenv()


def load_prompts(jsonl_path: Path) -> list[dict]:
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_existing_results(output_path: Path) -> set[str]:
    """Return set of already-processed frame names for resumability."""
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                done.add(rec["frame"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send pre-built vision prompts to OpenAI Responses API."
    )
    parser.add_argument(
        "--scenario",
        choices=["all", "oob", "fg"],
        default="all",
        help="Label scenario (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: results/<scenario>_vision_results.jsonl)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Sleep between requests to avoid rate limits (seconds)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max output tokens per request (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries on API error (default: 3)",
    )
    parser.add_argument(
        "--pricing-tier",
        choices=["standard", "batch"],
        default="standard",
        help="Pricing tier for cost calculation (default: standard)",
    )
    args = parser.parse_args()

    input_path = Path("prompts") / f"{args.scenario}_vision.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = (
        Path(args.output)
        if args.output
        else Path("results") / f"{args.scenario}_vision_results.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_prompts(input_path)
    print(f"Loaded {len(records)} prompts from {input_path}")

    done = load_existing_results(output_path)
    remaining = [r for r in records if r["frame"] not in done]
    print(f"Already completed: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        print("All prompts already processed.")
        return

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with output_path.open("a", encoding="utf-8") as out_f:
        for rec in tqdm(remaining, desc="Processing prompts"):
            req = rec["request"]

            for attempt in range(1, args.max_retries + 1):
                try:
                    response = client.responses.create(
                        model=req["model"],
                        input=req["input"],
                        temperature=args.temperature,
                        max_output_tokens=args.max_tokens,
                    )
                    break
                except Exception as e:
                    if attempt == args.max_retries:
                        print(f"\nFailed after {args.max_retries} retries for {rec['frame']}: {e}")
                        raise
                    wait = 2 ** attempt
                    print(f"\nRetry {attempt}/{args.max_retries} for {rec['frame']} "
                          f"(waiting {wait}s): {e}")
                    time.sleep(wait)

            usage = None
            cost = None
            if response.usage:
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                cost = calculate_cost_usd(
                    model=response.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    tier=args.pricing_tier,
                )["total_cost_usd"]

            raw_text = response.output_text.strip()
            try:
                parsed = json.loads(raw_text)
                prediction = parsed.get("prediction")
                reason = parsed.get("reason", "")
            except json.JSONDecodeError:
                prediction = raw_text
                reason = ""

            result = {
                "frame": rec["frame"],
                "label": rec["label"],
                "scenario": rec["scenario"],
                "prediction": prediction,
                "reason": reason,
                "response_id": response.id,
                "model": response.model,
                "usage": usage,
                "cost": cost,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    main()
