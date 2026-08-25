#!/usr/bin/env python3
"""Send each paper PDF in this folder to Claude Opus 5 with a shared prompt.

Each paper is categorized for how it uses "persona". Later papers receive the
running category list and latest taxonomy table so the scheme can accumulate.

Outputs one .tex file per paper under outputs/.

    python -u run.py
    python -u run.py --prompt prompt.txt --skip-existing
    python -u run.py --only 1,5,14
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import sys
import time
from pathlib import Path

from anthropic import Anthropic, APIStatusError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TIMEOUT = 1800.0
MAX_RETRIES = 4

SYSTEM = (
    "You are coding workshop papers for how they use the concept of persona. "
    "Follow the user prompt. Categorize this paper against the running taxonomy "
    "when one is provided; reuse existing labels when they fit. "
    "Write the entire response as LaTeX. Do not wrap it in markdown fences. "
    "Do not add commentary outside the LaTeX."
)

CATEGORY_RE = re.compile(r"^%\s*PERSONA_CATEGORY:\s*(.+)$", re.M)
SUBCATEGORY_RE = re.compile(r"^%\s*PERSONA_SUBCATEGORY:\s*(.+)$", re.M)
TABLE_RE = re.compile(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", re.S)
TABULAR_RE = re.compile(r"\\begin\{tabularx?\}.*?\\end\{tabularx?\}", re.S)

PER_PAPER_INSTRUCTIONS = """
This request is for ONE attached paper: {name}

Produce LaTeX that includes:
1. A short snippet on how THIS paper uses the concept of persona (its conceptual understanding and how persona is operationalized).
2. A category assignment for THIS paper as LaTeX comments, using this exact form:
   % PERSONA_CATEGORY: <main category>
   % PERSONA_SUBCATEGORY: <subcategory or omit this line>
   Prefer an existing category label when it fits. Add or refine a label only when this paper's use of persona is genuinely different.
3. An updated LaTeX table of ALL papers categorized so far (previous papers plus this one). Keep prior rows. The table should show main categories and, if needed, subcategories that describe varying uses of the persona concept.

{previous}
""".strip()


def log(msg: str) -> None:
    print(msg, flush=True)


def paper_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.match(r"Paper\s+(\d+)", path.name, re.I)
    if match:
        return (0, int(match.group(1)), path.name.lower())
    return (1, 0, path.name.lower())


def paper_number(path: Path) -> int | None:
    match = re.match(r"Paper\s+(\d+)", path.name, re.I)
    return int(match.group(1)) if match else None


def list_papers(folder: Path, only: set[int] | None) -> list[Path]:
    papers = sorted(folder.glob("*.pdf"), key=paper_sort_key)
    if only:
        papers = [p for p in papers if paper_number(p) in only]
    return papers


def output_path(out_dir: Path, pdf: Path) -> Path:
    return out_dir / f"{pdf.stem}.tex"


def strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:latex|tex)?[ \t]*\n", "", cleaned, count=1, flags=re.I)
        cleaned = re.sub(r"\n```[ \t]*$", "", cleaned)
    return cleaned.strip() + "\n"


def load_prompt(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"Prompt file not found: {path}")
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt or prompt.startswith("[Paste your prompt"):
        raise SystemExit(f"Put your prompt text in {path} before running.")
    return prompt


def extract_tables(latex: str) -> str:
    parts = TABLE_RE.findall(latex)
    if not parts:
        parts = TABULAR_RE.findall(latex)
    return "\n\n".join(parts).strip()


def parse_assignment(latex: str, pdf: Path) -> dict:
    cat = CATEGORY_RE.search(latex)
    sub = SUBCATEGORY_RE.search(latex)
    return {
        "paper": pdf.name,
        "stem": pdf.stem,
        "number": paper_number(pdf),
        "category": cat.group(1).strip() if cat else "",
        "subcategory": sub.group(1).strip() if sub else "",
        "table": extract_tables(latex),
        "latex": latex,
    }


def format_previous(items: list[dict]) -> str:
    if not items:
        return (
            "This is the first paper. Start the category scheme from how THIS paper "
            "uses persona. Create the first version of the category table with this "
            "paper as the first row."
        )

    lines = ["Existing category assignments from previous papers:"]
    seen: list[str] = []
    for item in items:
        number = item["number"] if item["number"] is not None else "?"
        category = item["category"] or "(see table)"
        extra = f" / {item['subcategory']}" if item["subcategory"] else ""
        lines.append(f"- Paper {number}: {category}{extra}")
        if item["category"] and item["category"] not in seen:
            seen.append(item["category"])

    if seen:
        lines.append("")
        lines.append("Category labels in use: " + "; ".join(seen))

    latest_table = next((item["table"] for item in reversed(items) if item["table"]), "")
    if latest_table:
        lines.append("")
        lines.append("Latest taxonomy table (update this; do not start over):")
        lines.append(latest_table)
    else:
        lines.append("")
        lines.append("Previous paper LaTeX (use its table if present):")
        lines.append(items[-1]["latex"])
    return "\n".join(lines)


def compose_prompt(base_prompt: str, pdf: Path, previous: list[dict]) -> str:
    return (
        base_prompt.strip()
        + "\n\n"
        + PER_PAPER_INSTRUCTIONS.format(name=pdf.name, previous=format_previous(previous))
    )


def write_running_files(out_dir: Path, previous: list[dict]) -> None:
    if not previous:
        return
    cat_path = out_dir / "_categories.txt"
    rows = []
    for item in previous:
        number = item["number"] if item["number"] is not None else "?"
        category = item["category"] or "(unparsed)"
        extra = f" / {item['subcategory']}" if item["subcategory"] else ""
        rows.append(f"Paper {number}: {category}{extra}")
    cat_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    latest_table = next((item["table"] for item in reversed(previous) if item["table"]), "")
    if latest_table:
        (out_dir / "_taxonomy.tex").write_text(latest_table + "\n", encoding="utf-8")


def document_block(client: Anthropic, pdf: Path) -> dict:
    if hasattr(client, "files") and hasattr(client.files, "upload"):
        with pdf.open("rb") as handle:
            uploaded = client.files.upload(file=(pdf.name, handle, "application/pdf"))
        return {
            "type": "document",
            "source": {"type": "file", "file_id": uploaded.id},
            "title": pdf.stem,
        }

    data = base64.standard_b64encode(pdf.read_bytes()).decode("ascii")
    return {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": data},
        "title": pdf.stem,
    }


def extract_text(message) -> str:
    parts: list[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def run_one(client: Anthropic, pdf: Path, prompt: str, model: str, max_tokens: int) -> str:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            log(f"  uploading {pdf.name} ({pdf.stat().st_size / 1_048_576:.1f} MB)")
            doc = document_block(client, pdf)
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            doc,
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            ) as stream:
                for _chunk in stream.text_stream:
                    pass
                message = stream.get_final_message()
            text = extract_text(message).strip()
            if not text:
                raise RuntimeError("model returned no text")
            return strip_fences(text)
        except APIStatusError as exc:
            last_error = exc
            wait = min(120, 15 * (2**attempt))
            retryable = exc.status_code in {408, 409, 429, 500, 502, 503, 529}
            log(f"  attempt {attempt + 1} failed ({exc.status_code}): {exc}. retry in {wait}s")
            if not retryable or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(wait)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            wait = min(120, 15 * (2**attempt))
            log(f"  attempt {attempt + 1} failed: {exc}. retry in {wait}s")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(wait)
    raise RuntimeError(f"failed after {MAX_RETRIES} retries: {last_error}")


def parse_only(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    numbers: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        numbers.add(int(part))
    return numbers or None


def main() -> int:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run a prompt on each paper PDF with Claude Opus 5.")
    parser.add_argument("--prompt", type=Path, default=ROOT / "prompt.txt", help="Prompt .txt file")
    parser.add_argument("--papers", type=Path, default=ROOT, help="Folder of PDFs")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs", help="Folder for .tex files")
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("ANTHROPIC_MAX_TOKENS", DEFAULT_MAX_TOKENS)))
    parser.add_argument("--only", help="Comma-separated paper numbers, e.g. 1,5,14")
    parser.add_argument("--skip-existing", action="store_true", help="Skip papers that already have a .tex file")
    parser.add_argument("--dry-run", action="store_true", help="List papers without calling the API")
    args = parser.parse_args()

    prompt_path = args.prompt if args.prompt.is_absolute() else ROOT / args.prompt
    papers_dir = args.papers if args.papers.is_absolute() else ROOT / args.papers
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out

    papers = list_papers(papers_dir, parse_only(args.only))
    if not papers:
        raise SystemExit(f"No PDFs found in {papers_dir}")

    if args.dry_run:
        prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.is_file() else ""
    else:
        prompt = load_prompt(prompt_path)
        out_dir.mkdir(parents=True, exist_ok=True)

    log(f"model: {args.model}")
    log(f"prompt: {prompt_path}" + (f" ({len(prompt)} chars)" if prompt else " (not ready)"))
    log(f"papers: {len(papers)}")
    prior = 0
    for pdf in papers:
        dest = output_path(out_dir, pdf)
        if args.skip_existing and dest.exists():
            log(f"  [skip] {pdf.name} -> {dest.name} (will still feed its categories forward)")
            prior += 1
            continue
        log(f"  [run]  {pdf.name} -> {dest.name} (prior categories: {prior})")
        prior += 1

    if args.dry_run:
        return 0

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")

    client = Anthropic(api_key=api_key, timeout=DEFAULT_TIMEOUT)
    failed: list[str] = []
    previous: list[dict] = []

    for i, pdf in enumerate(papers, start=1):
        dest = output_path(out_dir, pdf)
        log(f"\n[{i}/{len(papers)}] {pdf.name}")
        if args.skip_existing and dest.exists():
            assignment = parse_assignment(dest.read_text(encoding="utf-8"), pdf)
            previous.append(assignment)
            write_running_files(out_dir, previous)
            log(f"  already exists: {dest}")
            if assignment["category"]:
                log(f"  loaded category: {assignment['category']}")
            continue
        try:
            paper_prompt = compose_prompt(prompt, pdf, previous)
            latex = run_one(client, pdf, paper_prompt, args.model, args.max_tokens)
            dest.write_text(latex, encoding="utf-8")
            assignment = parse_assignment(latex, pdf)
            previous.append(assignment)
            write_running_files(out_dir, previous)
            log(f"  wrote {dest} ({len(latex)} chars)")
            if assignment["category"]:
                log(f"  category: {assignment['category']}")
            else:
                log("  category comment not parsed; next paper will still get the latest table")
        except Exception as exc:  # noqa: BLE001
            failed.append(pdf.name)
            log(f"  FAILED: {exc}")

    if failed:
        log(f"\nFinished with {len(failed)} failure(s):")
        for name in failed:
            log(f"  - {name}")
        return 2

    log("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
