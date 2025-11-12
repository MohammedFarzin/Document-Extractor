"""
Structured document extraction using Hugging Face Donut.

Loads a Donut model and extracts JSON from images (invoices / receipts).
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required. Install via `pip install torch --index-url https://download.pytorch.org/whl/cpu`"
    ) from exc

try:
    import sentencepiece  # noqa: F401  # pragma: no cover
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("sentencepiece is required. Install it via `pip install sentencepiece`.") from exc


def _glob_inputs(patterns: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            paths.extend(glob(pattern))
        else:
            paths.append(pattern)
    return sorted({p for p in paths if os.path.isfile(p)})


def _has_mps() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def resolve_device(requested: str | None) -> str:
    choice = (requested or "auto").lower()

    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _has_mps():
            return "mps"
        return "cpu"
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return "cuda"
    if choice == "mps":
        if not _has_mps():
            raise SystemExit("MPS requested but unavailable.")
        return "mps"
    if choice == "cpu":
        return "cpu"
    raise SystemExit("--device must be one of: auto, cpu, cuda, mps")


def load_model(model_id: str, device: str):
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return processor, model


def donut_infer(
    processor,
    model,
    image_path: str,
    device: str,
    task_prompt: str | None = None,
) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")

    if task_prompt is None:
        task_prompt = "<s_cord-v2>"

    pixel_inputs = processor(image, return_tensors="pt")
    pixel_values = pixel_inputs["pixel_values"].to(device)
    prompt_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    outputs = model.generate(
        pixel_values=pixel_values,
        decoder_input_ids=prompt_ids,
        max_length=model.config.decoder.max_position_embeddings,
        num_beams=1,
        no_repeat_ngram_size=3,
        do_sample=False,
        use_cache=True,
    )

    sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
    sequence = processor.batch_decode(sequences, skip_special_tokens=False)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")

    try:
        result = processor.token2json(sequence)
    except Exception:  # pragma: no cover
        result = {"raw": sequence}

    return result


def save_json(obj: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured OCR via Donut (HF)")
    parser.add_argument("--model-id", default="naver-clova-ix/donut-base-finetuned-cord-v2", help="Hugging Face model id")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["samples/invoice/*.png"],
        help="Image files or globs",
    )
    parser.add_argument("--output-dir", default="output", help="Directory to write JSON results")
    parser.add_argument("--task-prompt", default=None, help="Override Donut task prompt (e.g., <s_cord-v2>)")
    parser.add_argument("--device", default="auto", help="Device selection: auto, cpu, cuda, or mps")

    args = parser.parse_args()
    paths = _glob_inputs(args.inputs)
    if not paths:
        raise SystemExit("No input images found. Check --inputs path/glob.")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    try:
        processor, model = load_model(args.model_id, device)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            f"Failed to load model '{args.model_id}'. Ensure transformers/torch are installed.\n{exc}"
        ) from exc

    for img_path in paths:
        print(f"Processing: {img_path}")
        pred = donut_infer(processor, model, img_path, device=device, task_prompt=args.task_prompt)
        base = Path(img_path).stem
        out_path = os.path.join(args.output_dir, f"{base}_donut.json")
        save_json(pred, out_path)
        print(f"  -> Saved: {out_path}")


if __name__ == "__main__":
    main()
