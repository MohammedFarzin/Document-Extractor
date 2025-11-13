"""
ERP Archive Document Processor - Open Source Edition
Vision-language (VL) pipeline powered by Hugging Face models.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
from datetime import datetime

import httpx
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pdf2image import convert_from_path
from PIL import Image
## Transformers not used in Ollama-only mode

DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:7b")


class DocumentType(Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    PURCHASE_ORDER = "purchase_order"
    DELIVERY_NOTE = "delivery_note"
    TAX_DOCUMENT = "tax_document"
    UNKNOWN = "unknown"


class ProcessingStatus(Enum):
    AUTO_APPROVED = "auto_approved"
    PENDING_REVIEW = "pending_review"
    MANUAL_ENTRY_REQUIRED = "manual_entry_required"
    FAILED = "failed"


class OpenSourceDocumentProcessor:
    """End-to-end VL document processor."""

    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        quantized_model_name: Optional[str] = None,
        prefer_quantized: bool = False,
        categories: Optional[List[str]] = None,
        departments: Optional[List[str]] = None,
        device: str = "auto",
        use_gpu: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_gpu_memory_gb: Optional[int] = None,
        max_cpu_memory_gb: Optional[int] = None,
        # Ollama backend
        use_ollama: bool = True,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        ollama_base_url: str = "http://localhost:11434",
        ollama_timeout_s: int = 600,
        ollama_keep_alive: str = "5m",
        ollama_options: Optional[Dict[str, Any]] = None,
        confidence_threshold_auto: float = 0.95,
        confidence_threshold_review: float = 0.80,
        max_pages: int = 1,
        pdf_dpi: int = 280,
    ) -> None:
        self.base_model_name = model_name
        self.model_name = f"ollama:{ollama_model}"
        self.quantized_model_name = quantized_model_name
        self.prefer_quantized = prefer_quantized
        self.categories = categories or []
        self.departments = departments or []
        self.confidence_threshold_auto = confidence_threshold_auto
        self.confidence_threshold_review = confidence_threshold_review
        self.max_pages = max_pages
        self.pdf_dpi = pdf_dpi
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.max_gpu_memory_gb = max_gpu_memory_gb
        self.max_cpu_memory_gb = max_cpu_memory_gb
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_timeout_s = int(ollama_timeout_s)
        self.ollama_keep_alive = ollama_keep_alive
        self.ollama_options = ollama_options or {"num_predict": 512, "temperature": 0.0}

        # Device selection not required for Ollama; set to CPU for logs
        self.device = "cpu"

        print("🔧 Initializing Open Source Processor")
        print(f"   Base model: {self.base_model_name}")
        if self.quantized_model_name:
            print(f"   Quantized checkpoint: {self.quantized_model_name}")
            print(f"   Prefer quantized: {self.prefer_quantized}")
        print(f"   Device: {self.device}")
        print(f"   8-bit quantization: {load_in_8bit}")
        print(f"   4-bit quantization: {load_in_4bit}")
        if self.max_gpu_memory_gb:
            print(f"   Max GPU memory per device: {self.max_gpu_memory_gb} GiB")
        if self.max_cpu_memory_gb:
            print(f"   Max CPU offload memory: {self.max_cpu_memory_gb} GiB")
        print(f"   Backend: Ollama -> model={self.ollama_model} url={self.ollama_base_url}")
        print(f"   Ollama timeout: {self.ollama_timeout_s}s, keep_alive: {self.ollama_keep_alive}")

        # Initialize Ollama backend only
        self._init_vl_model()
        print("✅ Processor initialized successfully")

    def _init_vl_model(self) -> None:
        # Ollama-only path: verify server and model availability
        self._verify_ollama_backend()
        print("📥 Using Ollama backend; no Transformers model to load")

    def _quantization_available(self) -> bool:
        # Not applicable in Ollama-only mode
        return False

    def _build_max_memory(self) -> Dict[object, str]:
        # Not used in Ollama-only mode
        return {}

    def _cuda_supports_bfloat16(self) -> bool:
        # Not applicable in Ollama-only mode
        return False

    async def process_document(
        self,
        file_path: str,
        user_id: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        document_id = self._generate_document_id()
        print(f"[{document_id}] Processing: {file_path}")

        try:
            structured_data, doc_type, language, vision_confidence = await self._analyze_with_vl_model(file_path)

            confidence_score = self._calculate_confidence(structured_data, vision_confidence)
            status = self._determine_status(confidence_score)

            result = self._format_output(
                document_id=document_id,
                file_path=file_path,
                structured_data=structured_data,
                confidence_score=confidence_score,
                status=status,
                doc_type=doc_type,
                language=language,
                user_id=user_id,
                vision_confidence=vision_confidence,
            )
            print(
                f"[{document_id}] ✅ Complete - Status: {status.value}, Confidence: {confidence_score:.2%}"
            )
            return result
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"[{document_id}] ❌ Error: {exc}")
            return {
                "document_id": document_id,
                "status": ProcessingStatus.FAILED.value,
                "error": str(exc),
            }

    async def _analyze_with_vl_model(
        self,
        file_path: str,
    ) -> Tuple[Dict, DocumentType, str, float]:
        images = self._load_document_images(file_path)
        if not images:
            raise ValueError("No pages produced from document")

        prompt = self._build_vl_prompt()
        response_text = self._run_vl_inference(prompt, images)

        data = self._parse_vl_json(response_text)
        doc_type, language = self._extract_doc_attributes(data)
        data = self._validate_extracted_data(data)
        data = self._clean_description(data, doc_type)

        tags = data.get("tags", [])
        for tag in filter(None, [doc_type.value, language, "vision_extraction"]):
            if tag not in tags:
                tags.append(tag)
        data["tags"] = tags[:10]

        vision_confidence = self._estimate_vision_confidence(data)
        return data, doc_type, language, vision_confidence

    def _run_vl_inference(self, prompt: str, images: List[Image.Image]) -> str:
        # Always route to Ollama
        return self._run_ollama_inference(prompt, images)

    def _img_to_base64(self, image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
        buf = io.BytesIO()
        # Ensure RGB for JPEG; for PNG switch format and remove quality
        img = image.convert("RGB") if format.upper() == "JPEG" else image
        img.save(buf, format=format, quality=quality)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _run_ollama_inference(self, prompt: str, images: List[Image.Image]) -> str:
        """Call Ollama with multimodal input via /api/generate only."""
        img_b64 = [self._img_to_base64(img) for img in images]

        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "images": img_b64,
            "stream": False,
            "keep_alive": self.ollama_keep_alive,
            "options": self.ollama_options,
        }
        timeout = httpx.Timeout(connect=30.0, read=self.ollama_timeout_s, write=30.0, pool=30.0)
        attempts = 3
        last_exc: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                break
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt < attempts:
                    sleep_s = 1.5 ** (attempt - 1)
                    print(f"   ⚠️ Ollama request timeout/connection issue. Retrying in {sleep_s:.1f}s (attempt {attempt}/{attempts})")
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(f"Ollama /api/generate timed out after {attempts} attempts") from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                body = exc.response.text if exc.response is not None else ""
                raise RuntimeError(f"Ollama /api/generate failed: {status} {body}") from exc

        text = data.get("response") or data.get("output") or ""
        if not text:
            raise RuntimeError("Empty response from Ollama /api/generate")
        return text

    def _verify_ollama_backend(self) -> None:
        """Ensure the Ollama server is reachable and the requested model exists."""
        version_url = f"{self.ollama_base_url}/api/version"
        tags_url = f"{self.ollama_base_url}/api/tags"
        try:
            with httpx.Client(timeout=10) as client:
                version_resp = client.get(version_url)
                version_resp.raise_for_status()
                tags_resp = client.get(tags_url)
                tags_resp.raise_for_status()
                models = tags_resp.json().get("models", [])
        except Exception as exc:  # pragma: no cover - network guard
            raise RuntimeError(
                f"Unable to reach Ollama server at {self.ollama_base_url}; "
                f"ensure 'ollama serve' is running. Details: {exc}"
            ) from exc

        available = {model.get("name") for model in models if model.get("name")}
        if self.ollama_model not in available:
            hint = ", ".join(sorted(available)) if available else "none"
            print(
                f"⚠️  Ollama model '{self.ollama_model}' not found on the server. "
                f"Available models: {hint or '[]'}. "
                f"Run `ollama pull {self.ollama_model}` on the host if needed."
            )

    def _load_document_images(self, file_path: str) -> List[Image.Image]:
        suffix = Path(file_path).suffix.lower()
        images: List[Image.Image] = []

        if suffix == ".pdf":
            pdf_images = convert_from_path(file_path, dpi=self.pdf_dpi)
            images = [self._prepare_image(img) for img in pdf_images[: self.max_pages]]
        elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
            with Image.open(file_path) as img:
                images = [self._prepare_image(img)]
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return images

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        max_side = 1920
        width, height = rgb.size
        if max(width, height) <= max_side:
            return rgb

        scale = max_side / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        resample = getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        return rgb.resize(new_size, resample)

    def _build_vl_prompt(self) -> str:
        categories = ", ".join(self.categories) if self.categories else "Financial, Legal, HR, Operations"
        departments = ", ".join(self.departments) if self.departments else "Finance, Legal, HR, Operations"
        doc_types = ", ".join([dt.value for dt in DocumentType if dt != DocumentType.UNKNOWN])

        return f"""
You are an expert accountant seeing document images directly. Extract structured data.
Return ONLY valid JSON with this schema:
{{
  "basic_information": {{
    "document_name": "DocType_Entity_Date_Number",
    "description": "Short semantic summary"
  }},
  "categorization": {{
    "category": "one of: {categories}",
    "department": "one of: {departments}",
    "category_confidence": 0.0-1.0,
    "department_confidence": 0.0-1.0
  }},
  "tags": ["tag1", "tag2"],
  "dates": {{
    "document_date": "YYYY-MM-DD or null",
    "due_date": "YYYY-MM-DD or null",
    "expiry_date": "YYYY-MM-DD or null"
  }},
  "extracted_metadata": {{
    "vendor_name": "string or null",
    "customer_name": "string or null",
    "document_number": "string or null",
    "total_amount": "number or null",
    "currency": "ISO code or symbol"
  }},
  "document_metadata": {{
    "document_type": "one of: {doc_types}",
    "language": "en, ar, or mixed",
    "vision_confidence": 0.0-1.0
  }}
}}
Rules:
- Copy numbers/dates exactly as printed.
- Preserve Arabic text when present.
- Use null when a field is absent.
- Do not add explanations.
"""

    def _parse_vl_json(self, response_text: str) -> Dict:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            print("   ⚠️  VL output missing JSON; using default structure")
            return self._get_default_structure()
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            print(f"   ⚠️  JSON parse error: {exc}; falling back to default structure")
            return self._get_default_structure()

    def _extract_doc_attributes(self, data: Dict) -> Tuple[DocumentType, str]:
        metadata = data.get("document_metadata") or {}
        doc_type_value = metadata.get("document_type") or data.get("document_type")
        language = (metadata.get("language") or data.get("language") or "en").lower()
        doc_type = self._map_doc_type(doc_type_value)
        return doc_type, language

    def _map_doc_type(self, value: Optional[str]) -> DocumentType:
        if not value:
            return DocumentType.UNKNOWN
        normalized = value.strip().lower().replace("-", "_")
        for doc_type in DocumentType:
            if doc_type.value == normalized:
                return doc_type
        return DocumentType.UNKNOWN

    def _estimate_vision_confidence(self, data: Dict) -> float:
        metadata = data.get("extracted_metadata", {})
        coverage_fields = ["vendor_name", "customer_name", "document_number", "total_amount"]
        filled = sum(1 for field in coverage_fields if metadata.get(field))
        coverage = filled / len(coverage_fields)

        categorization = data.get("categorization", {})
        cat_conf = categorization.get("category_confidence", 0.5)
        dept_conf = categorization.get("department_confidence", 0.5)
        tag_score = min(len(data.get("tags", [])) / 5, 1.0)

        score = 0.4 * coverage + 0.4 * ((cat_conf + dept_conf) / 2) + 0.2 * tag_score
        return round(min(1.0, max(0.1, score)), 3)

    def _validate_extracted_data(self, data: Dict) -> Dict:
        data.setdefault("basic_information", {})
        data.setdefault("categorization", {})
        data.setdefault("tags", [])
        data.setdefault("dates", {})
        data.setdefault("extracted_metadata", {})
        data.setdefault("document_metadata", {})

        if self.categories and data["categorization"].get("category") not in self.categories:
            data["categorization"]["category"] = self.categories[0]
            data["categorization"]["category_confidence"] = 0.5

        if self.departments and data["categorization"].get("department") not in self.departments:
            data["categorization"]["department"] = self.departments[0]
            data["categorization"]["department_confidence"] = 0.5

        for key in ("document_date", "due_date", "expiry_date"):
            value = data["dates"].get(key)
            if value:
                try:
                    datetime.strptime(value, "%Y-%m-%d")
                except (ValueError, TypeError):
                    data["dates"][key] = None

        if not isinstance(data["tags"], list):
            data["tags"] = []

        data["tags"] = data["tags"][:10]
        return data

    def _clean_description(self, data: Dict, doc_type: DocumentType) -> Dict:
        description = data.get("basic_information", {}).get("description", "")
        ocr_error_patterns = [r"[A-Z]{2,}\s+[A-Z]{2,}", r"Unils|Unil|Tolal", r"^[^a-zA-Z]*$"]
        has_issues = any(re.search(pattern, description) for pattern in ocr_error_patterns)
        if has_issues or len(description) < 15:
            metadata = data.get("extracted_metadata", {})
            vendor = metadata.get("vendor_name")
            customer = metadata.get("customer_name")
            amount = metadata.get("total_amount")
            parts = [doc_type.value.title()]
            if vendor:
                parts.append(f"from {vendor}")
            if customer:
                parts.append(f"to {customer}")
            if amount:
                parts.append(f"for {amount}")
            data["basic_information"]["description"] = " ".join(parts)
        return data

    def _calculate_confidence(self, structured_data: Dict, vision_confidence: float) -> float:
        scores = {
            "vision_quality": vision_confidence,
            "field_completeness": 0.0,
            "category_confidence": 0.0,
            "data_quality": 0.0,
        }

        required_fields = [
            "basic_information.document_name",
            "categorization.category",
            "categorization.department",
            "tags",
        ]
        filled = sum(1 for field in required_fields if self._get_nested_field(structured_data, field))
        scores["field_completeness"] = filled / len(required_fields)

        cat_conf = structured_data.get("categorization", {}).get("category_confidence", 0.5)
        dept_conf = structured_data.get("categorization", {}).get("department_confidence", 0.5)
        scores["category_confidence"] = (cat_conf + dept_conf) / 2

        scores["data_quality"] = self._assess_data_quality(structured_data)

        weights = [0.25, 0.35, 0.25, 0.15]
        return round(sum(value * weight for value, weight in zip(scores.values(), weights)), 3)

    def _assess_data_quality(self, data: Dict) -> float:
        score = 1.0
        doc_name = data.get("basic_information", {}).get("document_name", "")
        if not doc_name or "_" not in doc_name:
            score -= 0.2
        description = data.get("basic_information", {}).get("description", "")
        if not description or len(description) < 10:
            score -= 0.2
        if len(data.get("tags", [])) < 2:
            score -= 0.2
        metadata = data.get("extracted_metadata", {})
        if not (metadata.get("vendor_name") or metadata.get("customer_name")):
            score -= 0.2
        return max(0.0, score)

    def _get_nested_field(self, data: Dict, path: str):
        current = data
        for part in path.split('.'):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
            if current is None:
                return None
        return current

    def _determine_status(self, confidence_score: float) -> ProcessingStatus:
        if confidence_score >= self.confidence_threshold_auto:
            return ProcessingStatus.AUTO_APPROVED
        if confidence_score >= self.confidence_threshold_review:
            return ProcessingStatus.PENDING_REVIEW
        return ProcessingStatus.MANUAL_ENTRY_REQUIRED

    def _format_output(
        self,
        document_id: str,
        file_path: str,
        structured_data: Dict,
        confidence_score: float,
        status: ProcessingStatus,
        doc_type: DocumentType,
        language: str,
        user_id: str,
        vision_confidence: float,
    ) -> Dict:
        return {
            "document_id": document_id,
            "status": status.value,
            "confidence_score": confidence_score,
            "processing_timestamp": datetime.now().isoformat(),
            "document_submission": {
                "basic_information": structured_data.get("basic_information", {}),
                "categorization": {
                    "category": structured_data.get("categorization", {}).get("category"),
                    "department": structured_data.get("categorization", {}).get("department"),
                    "confidence_scores": {
                        "category": structured_data.get("categorization", {}).get("category_confidence", 0),
                        "department": structured_data.get("categorization", {}).get("department_confidence", 0),
                    },
                },
                "tags": structured_data.get("tags", []),
                "dates": structured_data.get("dates", {}),
                "additional_options": {
                    "physical_document_available": False,
                    "link_to_related_documents": [],
                },
                "file_upload": {
                    "original_file": file_path,
                    "processed_file": file_path,
                },
                "extraction_metadata": {
                    "processing_timestamp": datetime.now().isoformat(),
                    "extraction_confidence": confidence_score,
                    "vision_confidence": vision_confidence,
                    "document_type": doc_type.value,
                    "language": language,
                    "vision_model": self.model_name,
                    "requires_review": status == ProcessingStatus.PENDING_REVIEW,
                    "user_id": user_id,
                    "extracted_fields": structured_data.get("extracted_metadata", {}),
                },
            },
        }

    def _get_default_structure(self) -> Dict:
        return {
            "basic_information": {
                "document_name": "Unknown_Document",
                "description": "Document processing in progress",
            },
            "categorization": {
                "category": self.categories[0] if self.categories else "Uncategorized",
                "department": self.departments[0] if self.departments else "General",
                "category_confidence": 0.5,
                "department_confidence": 0.5,
            },
            "tags": ["processing"],
            "dates": {
                "document_date": None,
                "due_date": None,
                "expiry_date": None,
            },
            "extracted_metadata": {},
            "document_metadata": {},
        }

    def _generate_document_id(self) -> str:
        return f"DOC_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(self) % 10000}"


async def process_folder(
    processor: OpenSourceDocumentProcessor,
    input_dir: str,
    output_dir: str,
    user_id: str,
    limit: Optional[int] = None,
) -> None:
    """Batch process supported documents in a folder tree."""
    exts = {".pdf", ".png", ".jpg", ".jpeg"}
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Batch input directory not found: {input_dir}")

    files: List[Path] = []
    for candidate in sorted(input_path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in exts:
            files.append(candidate)
            if limit is not None and len(files) >= limit:
                break

    if not files:
        raise FileNotFoundError(f"No supported documents found under {input_dir}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        print(f"Processing {file_path} ...")
        result = await processor.process_document(str(file_path), user_id=user_id)
        out_path = out_dir / f"{file_path.stem}_result.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> saved {out_path}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="ERP Archive document processor")
    parser.add_argument("--file", "-f", default="samples/invoice/invoice_1.png", help="Single document to process")
    parser.add_argument("--batch-dir", "-b", help="Process all supported documents under this directory")
    parser.add_argument("--output-dir", "-o", default="output", help="Directory for JSON outputs")
    parser.add_argument("--user-id", default="demo_user", help="User identifier for metadata")
    parser.add_argument("--limit", type=int, help="Optional max number of files when batching")
    parser.add_argument("--ollama-timeout-s", type=int, default=int(os.environ.get("OLLAMA_TIMEOUT_S", "600")), help="Read timeout seconds for Ollama requests")
    parser.add_argument("--ollama-keep-alive", default=os.environ.get("OLLAMA_KEEP_ALIVE", "5m"), help="How long to keep model loaded in Ollama (e.g., 5m, 1h)")
    parser.add_argument("--ollama-num-predict", type=int, default=int(os.environ.get("OLLAMA_NUM_PREDICT", "512")), help="Max tokens to predict (testing)")
    parser.add_argument("--max-pages", type=int, default=1, help="Max pages to OCR per document")
    parser.add_argument("--pdf-dpi", type=int, default=280, help="DPI when rasterizing PDFs")
    args = parser.parse_args()

    categories = [
        "Financial Documents",
        "Legal Documents",
        "HR Documents",
        "Operational Documents",
    ]
    departments = ["Finance", "Legal", "HR", "Operations"]

    # Toggle Ollama via env to enable Qwen2-VL 7B (4-bit) served locally
    use_ollama_env = os.environ.get("USE_OLLAMA", "true").lower() in {"1", "true", "yes"}
    ollama_model_env = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    ollama_base_env = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    processor = OpenSourceDocumentProcessor(
        model_name=DEFAULT_OLLAMA_MODEL,
        categories=categories,
        departments=departments,
        use_gpu=True,
        load_in_8bit=False,
        load_in_4bit=False,
        max_pages=args.max_pages,
        pdf_dpi=args.pdf_dpi,
        use_ollama=use_ollama_env,
        ollama_model=ollama_model_env,
        ollama_base_url=ollama_base_env,
        ollama_timeout_s=args.ollama_timeout_s,
        ollama_keep_alive=args.ollama_keep_alive,
        ollama_options={"num_predict": args.ollama_num_predict, "temperature": 0.0},
    )

    if args.batch_dir:
        await process_folder(
            processor=processor,
            input_dir=args.batch_dir,
            output_dir=args.output_dir,
            user_id=args.user_id,
            limit=args.limit,
        )
    else:
        file_path = Path(args.file)
        if not file_path.is_file():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        result = await processor.process_document(file_path=str(file_path), user_id=args.user_id)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{file_path.stem}_result.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Saved result to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
