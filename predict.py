#!/usr/bin/env python3
"""Suggest complete next words from a trained, local language model.

Examples:
    python predict.py --text "I am sure" --top-k 5
    python predict.py --text "I am" --prefix s --json
"""
import argparse
import json
from pathlib import Path

from autocomplete.data import context_tokens, normalize_text
from autocomplete.storage import load_model

ROOT = Path(__file__).resolve().parent


def suggest(model, vocabulary, text, top_k=5, prefix=""):
    """Return lexical candidates and their original model probabilities.

    The supplied text consists of completed words. The optional prefix belongs
    to the *next* word and filters candidates without changing the context.
    Unknown context words are mapped to UNK; special tokens are never offered.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    words = context_tokens(text)
    lookup = {word: i for i, word in enumerate(vocabulary)}
    context = [lookup.get(word, 0) for word in words]
    probabilities = model.next_distribution(context)
    prefix = normalize_text(prefix).lower()
    eligible = [i for i in range(2, len(vocabulary))
                if vocabulary[i].startswith(prefix)]
    # Stable tie handling matches the experiment's deterministic lexical rank.
    ranked = sorted(eligible, key=lambda i: (-float(probabilities[i]), i))[:top_k]
    return {
        "context_words": words,
        "unknown_context_words": [word for word in words if word not in lookup],
        "prefix": prefix,
        "suggestions": [{"word": vocabulary[i], "probability": float(probabilities[i])}
                        for i in ranked],
        "probability_note": "Original model probabilities over the full vocabulary; displayed suggestions need not sum to one.",
    }


def resolve_model(path=None):
    if path:
        return Path(path).expanduser()
    pointer = ROOT / "output" / "selected_model.json"
    if not pointer.exists():
        raise FileNotFoundError("No trained model found. Run: python experiment.py")
    selected = json.loads(pointer.read_text(encoding="utf-8"))
    model_path = Path(selected["path"])
    return model_path if model_path.is_absolute() else ROOT / model_path


def print_result(result, model_path, json_output=False):
    """Render one prediction without changing model probabilities."""
    result["model_file"] = str(model_path)
    if json_output:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print(f"Model: {model_path.stem}")
    print(f"Context: {' '.join(result['context_words']) or '(start of sentence)'}")
    if result["unknown_context_words"]:
        print("Unknown context words: " + ", ".join(result["unknown_context_words"]))
    if result["suggestions"]:
        for rank, item in enumerate(result["suggestions"], 1):
            print(f"{rank}. {item['word']:<18} {item['probability']:.4%}")
    else:
        print("No vocabulary words match that prefix.")
    print(result["probability_note"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_options = parser.add_mutually_exclusive_group(required=True)
    input_options.add_argument("--text", help="Text so far (completed words); quote it.")
    input_options.add_argument("--interactive", action="store_true", help="Keep accepting text in a terminal session.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of suggestions (default: 5).")
    parser.add_argument("--prefix", default="", help="Optional beginning of the next word.")
    parser.add_argument("--model", help="A saved model JSON; default: winner selected on validation NLL.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    try:
        path = resolve_model(args.model)
        model, vocabulary, metadata = load_model(path)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    if not args.interactive:
        result = suggest(model, vocabulary, args.text, args.top_k, args.prefix)
        print_result(result, path, args.json)
        return
    print("Enter completed text and press Enter. Type :quit to exit.")
    while True:
        try:
            text = input("Text> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() == ":quit":
            break
        result = suggest(model, vocabulary, text, args.top_k, args.prefix)
        print_result(result, path, args.json)


if __name__ == "__main__":
    main()
