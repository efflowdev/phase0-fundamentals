"""The same model, the same prompt, on Metal and in a container. Twice.

This is the measurement behind the compose file's most consequential decision.
Docker on macOS runs a Linux VM, and the VM has no path to the Metal GPU — so an
Ollama in that VM is running on CPU cores no matter what the host hardware is.
Everything else about the two daemons is identical: same image of the same model,
same weights, same sampler settings, same seed.

Which makes it the cleanest available demonstration of a thing that is otherwise
abstract: a container is a packaging boundary, not a portability guarantee, and
for inference workloads the accelerator sits on the far side of it. That is the
same reason a GPU container on Linux needs nvidia-container-toolkit and a device
passthrough flag rather than just working.

  host      : ollama on macOS, Metal          → OLLAMA_HOST=0.0.0.0 ollama serve
  container : compose service, CPU in the VM  → docker compose --profile full up -d ollama
              then                              docker compose exec ollama ollama pull llama3.2

Reported per run: tokens/second from Ollama's own eval_count / eval_duration,
which excludes prompt processing and model load, and wall-clock, which does not.
Both, because the gap between them is where a cold model load hides.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

RUNS = Path(__file__).resolve().parent / "runs"

PROMPT = (
    "Write one paragraph explaining what a product SKU is and why an "
    "e-commerce catalogue needs one. Do not use bullet points."
)

ENDPOINTS = {
    "host_metal": "http://localhost:11434",
    "container_cpu": "http://localhost:11435",
}


@dataclass
class Sample:
    wall_s: float
    eval_tokens: int
    eval_s: float
    load_s: float

    @property
    def tokens_per_s(self) -> float:
        return self.eval_tokens / self.eval_s if self.eval_s else 0.0


def one_call(client: httpx.Client, model: str, num_predict: int) -> Sample:
    started = time.perf_counter()
    response = client.post(
        "/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": False,
            # Fixed seed and temperature 0. Note what this does NOT buy: the two
            # daemons still produce different token counts from identical
            # settings, because the CPU and Metal backends do their arithmetic
            # in a different order and the divergence compounds. Day 3's finding,
            # met again one layer down. Throughput is per-token, so the
            # comparison survives it — the exact-output comparison would not.
            "options": {
                "temperature": 0.0,
                "seed": 42,
                "num_predict": num_predict,
            },
        },
        timeout=httpx.Timeout(connect=5.0, read=600.0, write=10.0, pool=10.0),
    )
    response.raise_for_status()
    wall = time.perf_counter() - started
    body = response.json()
    return Sample(
        wall_s=wall,
        eval_tokens=body.get("eval_count", 0),
        eval_s=body.get("eval_duration", 0) / 1e9,
        load_s=body.get("load_duration", 0) / 1e9,
    )


def bench(name: str, url: str, model: str, runs: int, num_predict: int):
    try:
        with httpx.Client(base_url=url, timeout=10.0) as client:
            tags = client.get("/api/tags").raise_for_status().json()
            available = [m["name"] for m in tags.get("models", [])]
            if not any(m.split(":")[0] == model.split(":")[0] for m in available):
                return None, f"{model} not pulled here (has: {available or 'nothing'})"

            # First call discarded: it pays the model load, and including it in a
            # throughput mean makes the slower endpoint look better than it is,
            # since load is a smaller fraction of a long run.
            warmup = one_call(client, model, num_predict)
            samples = [one_call(client, model, num_predict) for _ in range(runs)]
    except httpx.ConnectError:
        return None, f"nothing listening on {url}"

    return {
        "endpoint": url,
        "runs": runs,
        "warmup_load_s": round(warmup.load_s, 2),
        "eval_tokens": samples[0].eval_tokens,
        # Median, not mean. The CPU arm shares the VM's vCPUs with whatever else
        # the machine is doing and throws occasional 4x-slow outliers; the Metal
        # arm holds a 3% band. A mean lets one stalled run set the headline
        # number, so the summary is the median and the spread is printed beside
        # it rather than hidden inside it.
        "tokens_per_s_median": round(
            statistics.median(s.tokens_per_s for s in samples), 2
        ),
        "tokens_per_s_min": round(min(s.tokens_per_s for s in samples), 2),
        "tokens_per_s_max": round(max(s.tokens_per_s for s in samples), 2),
        "wall_s_median": round(statistics.median(s.wall_s for s in samples), 2),
    }, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--num-predict", type=int, default=160)
    args = parser.parse_args()

    report: dict[str, object] = {"model": args.model, "num_predict": args.num_predict}

    for name, url in ENDPOINTS.items():
        print(f"{name:<14} {url} ...", flush=True)
        result, problem = bench(name, url, args.model, args.runs, args.num_predict)
        if problem:
            print(f"  skipped: {problem}")
            report[name] = {"skipped": problem}
            continue
        print(
            f"  {result['tokens_per_s_median']:.2f} tok/s median  "
            f"({result['tokens_per_s_min']:.2f}–{result['tokens_per_s_max']:.2f}), "
            f"{result['eval_tokens']} tokens, wall {result['wall_s_median']:.2f}s, "
            f"load {result['warmup_load_s']:.2f}s"
        )
        report[name] = result

    both = [report.get(k) for k in ENDPOINTS]
    if all(isinstance(r, dict) and "tokens_per_s_median" in r for r in both):
        host, container = both  # type: ignore[misc]
        ratio = host["tokens_per_s_median"] / container["tokens_per_s_median"]
        report["metal_speedup"] = round(ratio, 2)
        print(f"\nMetal is {ratio:.2f}x the containerised CPU daemon")

    RUNS.mkdir(exist_ok=True)
    (RUNS / "ollama_bench.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {RUNS / 'ollama_bench.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
