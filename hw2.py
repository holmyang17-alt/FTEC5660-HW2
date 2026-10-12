#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()


def build_agent(tools: list[Any]) -> Any:
    """Create and return your agent once."""
    from langchain_deepseek import ChatDeepSeek
    from langchain.agents import create_agent

    model = ChatDeepSeek(model=MODEL_NAME, temperature=0)

    system_prompt = (
        "You are a strict CV verification agent. Given a candidate's CV, verify "
        "whether its claims match the candidate's LinkedIn and Facebook profiles.\n\n"
        "STEP 1 - Extract from the CV: full name, city, every job (company, title, "
        "start year, end year), every education entry (school, degree, field, "
        "graduation year), and every listed skill.\n\n"
        "STEP 2 - Find the candidate on LinkedIn (PRIMARY source) and Facebook. "
        "Many people share the same name: narrow the search with the CV's city "
        "and industry. Facebook display names may differ from legal names; match "
        "on original_name if needed.\n\n"
        "STEP 3 - Fetch the full LinkedIn profile and the Facebook profile.\n\n"
        "STEP 4 - Compare FIELD BY FIELD and write a short table in your reply:\n"
        "  - city: CV vs LinkedIn (match / MISMATCH)\n"
        "  - each job: company, title, start_year, end_year (match / MISMATCH)\n"
        "  - each education: school, degree, field, end_year (match / MISMATCH)\n"
        "  - each skill in the CV: is it on LinkedIn? (match / MISMATCH)\n\n"
        "A MISMATCH exists when the CV states a value that contradicts the "
        "profile: different start or end year (even by one year), different "
        "company, different school, different degree level, different city, or a "
        "skill that is NOT on the LinkedIn profile.\n"
        "The following are NOT mismatches: wording differences such as "
        "'Bachelor of Science' vs 'BSc', 'UI/UX Design' vs 'UI/UX', 'Senior "
        "Engineer' for a role whose seniority is senior, or listing FEWER skills "
        "than the profile. Job description, headline, and hometown are NEVER "
        "compared.\n\n"
        "STEP 5 - Decide the score strictly:\n"
        "  - If ANY field is a MISMATCH: score MUST be <= 0.3.\n"
        "  - If every field matches: score MUST be >= 0.9.\n"
        "  - Only if you are genuinely unable to find the person, use 0.5.\n"
        "Do not give 0.7 or 0.8 when there is a mismatch: that is wrong.\n\n"
        "End your reply with exactly one line:\n"
        "SCORE: <number between 0 and 1>"
    )

    agent = create_agent(model, tools, system_prompt=system_prompt)
    return agent


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Run your agent and return one reliability score per CV."""
    sem = asyncio.Semaphore(3)

    async def score_one(name: str, text: str) -> tuple[str, float]:
        async with sem:
            try:
                result = await agent.ainvoke({
                    "messages": [("user",
                        "Verify this CV. The filename is " + name + ".\n\n"
                        "CV content:\n" + text + "\n\n"
                        "Output your final answer with a line: SCORE: <number>"
                    )]
                })
                messages = result.get("messages", [])
                last = messages[-1] if messages else None
                content = getattr(last, "content", "")
                if isinstance(content, list):
                    parts = []
                    for b in content:
                        if isinstance(b, dict):
                            parts.append(b.get("text", ""))
                        else:
                            parts.append(str(b))
                    content = " ".join(parts)
                content = str(content)
                m = re.search(r"SCORE:\s*(-?\d+(?:\.\d+)?)", content, re.IGNORECASE)
                if m:
                    score = float(m.group(1))
                    if 0.0 <= score <= 1.0:
                        return name, score
                return name, 0.5
            except Exception:
                return name, 0.5

    tasks = [score_one(name, text) for name, text in cvs.items()]
    results = await asyncio.gather(*tasks)
    return dict(results)


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
