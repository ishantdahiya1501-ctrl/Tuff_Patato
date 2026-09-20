"""Download TinyStoriesInstruct (instruction -> story pairs) from HuggingFace.

The upstream repo (roneneldan/TinyStoriesInstruct) stores plain-text blocks:

    Features: Dialogue            (optional)
    Words: quit, oak, gloomy      (optional)
    Summary: Sara and Ben ...
    Story:
    <story paragraphs>
    <|endoftext|>

which arrive as one streamed row per line, so we parse them into
{"user": <instruction block>, "assistant": <story>} pairs and save them as a
JSON list. If the primary dataset name fails, the closest alternative is tried
instead, and whichever dataset is actually used is printed.
"""

import argparse
import json
import os

from tqdm import tqdm

PRIMARY = "roneneldan/TinyStoriesInstruct"
FALLBACKS = [PRIMARY, "roneneldan/TinyStories"]

INSTRUCTION_PREFIXES = ("Features:", "Words:", "Summary:")
STORY_PREFIX = "Story:"
END_OF_EXAMPLE = "<|endoftext|>"


def iter_instruction_stories(lines):
    """Yield {"user", "assistant"} pairs from an iterable of text lines."""
    instr_lines, story_lines, in_story = [], [], False
    for line in lines:
        line = (line or "").strip()
        if line == END_OF_EXAMPLE:
            if instr_lines and story_lines:
                yield {
                    "user": " ".join(" ".join(instr_lines).split()),
                    "assistant": " ".join(" ".join(story_lines).split()),
                }
            instr_lines, story_lines, in_story = [], [], False
            continue
        if line.startswith(STORY_PREFIX):
            in_story = True
            rest = line[len(STORY_PREFIX):].strip()
            if rest:
                story_lines.append(rest)
            continue
        if not in_story and line.startswith(INSTRUCTION_PREFIXES):
            instr_lines.append(line)
        elif in_story and line:
            story_lines.append(line)


def row_text(row):
    if isinstance(row, dict):
        return row.get("text") or ""
    return row or ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/chat/raw_chat.json")
    parser.add_argument("--max_samples", type=int, default=50000,
                        help="Stop after this many conversations (default: 50000)")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("The 'datasets' package is required: pip install datasets")

    dataset, used_name = None, None
    for name in FALLBACKS:
        try:
            print(f"Trying HuggingFace dataset: {name} ...")
            dataset = load_dataset(name, split="train", streaming=True)
            used_name = name
            break
        except Exception as exc:
            print(f"  failed ({exc.__class__.__name__}: {exc})")
    if dataset is None:
        raise SystemExit("Could not download any TinyStories dataset from HuggingFace")
    print(f"Using dataset: {used_name}")

    rows = []
    if used_name == PRIMARY:
        examples = iter_instruction_stories(row_text(row) for row in dataset)
        for example in tqdm(examples, total=args.max_samples, desc="downloading",
                            unit="conv"):
            rows.append(example)
            if len(rows) >= args.max_samples:
                break
    else:
        # Fallback: plain TinyStories has no instructions; synthesize the user turn.
        print("Fallback dataset has no instruction column; synthesizing "
              "'write a story' user turns.")
        for row in tqdm(dataset, total=args.max_samples, desc="downloading", unit="story"):
            story = " ".join(row_text(row).split())
            if story:
                rows.append({"user": "Write a short story for me.", "assistant": story})
            if len(rows) >= args.max_samples:
                break

    if not rows:
        raise SystemExit("No usable conversations found in the dataset")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)

    print(f"\nSaved {len(rows):,} conversations to {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")
    for i, row in enumerate(rows[:3], 1):
        print(f"\n--- sample {i} ---")
        print("user:      " + row["user"][:220])
        print("assistant: " + row["assistant"][:220]
              + (" ..." if len(row["assistant"]) > 220 else ""))


if __name__ == "__main__":
    main()
