"""Deterministic PaddleOCR-based extractor for ERP archive documents."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from paddleocr import PaddleOCR


@dataclass
class ExtractionResult:
    file_path: str
    doc_type: str
    confidence: float
    fields: Dict[str, Any]
    validations: List[str]


class DocumentClassifier:
    KEYWORDS = {
        "invoice": ["invoice", "amount due", "bill to", "po number"],
        "receipt": ["receipt", "change", "thank you", "pos"],
        "contract": ["agreement", "contract", "between", "terms"],
        "slip": ["delivery", "packing", "waybill"],
    }

    def classify(self, text: str) -> str:
        scores = {k: 0 for k in self.KEYWORDS}
        lower = text.lower()
        for doc_type, keywords in self.KEYWORDS.items():
            for keyword in keywords:
                if keyword in lower:
                    scores[doc_type] += 1
        best = max(scores, key=scores.get)
        return best if scores[best] else "unknown"


class ArchiveExtractor:
    def __init__(self) -> None:
        self.ocr = PaddleOCR(lang="en")
        self.classifier = DocumentClassifier()

    def process(self, file_path: str) -> ExtractionResult:
        ocr_result = self.ocr.ocr(file_path, cls=False)
        lines = [segment[1][0] for page in ocr_result for segment in page if segment]
        text = "\n".join(lines)
        doc_type = self.classifier.classify(text)

        fields = self._extract_fields(text)
        validations = self._run_validations(fields)
        confidence = 0.6 + 0.1 * len(validations)

        return ExtractionResult(
            file_path=file_path,
            doc_type=doc_type,
            confidence=min(confidence, 0.99),
            fields=fields,
            validations=validations,
        )

    def _extract_fields(self, text: str) -> Dict[str, Any]:
        fields: Dict[str, Any] = {
            "vendor_name": None,
            "invoice_number": None,
            "document_date": None,
            "total_amount": None,
        }
        for line in text.splitlines():
            lower = line.lower()
            if "invoice" in lower and "#" in line:
                fields["invoice_number"] = line.split("#", 1)[-1].strip()
            if "total" in lower:
                fields["total_amount"] = line.split()[-1]
            if "date" in lower and not fields["document_date"]:
                fields["document_date"] = line.split(":")[-1].strip()
        fields = {k: v for k, v in fields.items() if v}
        return fields

    def _run_validations(self, fields: Dict[str, Any]) -> List[str]:
        validations: List[str] = []
        if "invoice_number" in fields:
            validations.append("invoice_number_detected")
        if "total_amount" in fields:
            validations.append("total_amount_detected")
        return validations


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process documents with PaddleOCR")
    parser.add_argument("inputs", nargs="+", help="Files or globs to process")
    parser.add_argument("--output", default="output", help="Directory to store JSON results")
    args = parser.parse_args()

    extractor = ArchiveExtractor()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for pattern in args.inputs:
        paths.extend(Path().glob(pattern))
    if not paths:
        raise SystemExit("No input files found")

    for file_path in paths:
        if not file_path.is_file():
            continue
        print(f"Processing {file_path}")
        result = extractor.process(str(file_path))
        out_path = output_dir / f"{file_path.stem}_archive.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "file_path": result.file_path,
                    "doc_type": result.doc_type,
                    "confidence": result.confidence,
                    "fields": result.fields,
                    "validations": result.validations,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        print(f"  -> Saved {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
