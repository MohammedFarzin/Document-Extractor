"""
ERP Archive Document Processor - Open Source Edition
Vision-language (VL) pipeline powered by Hugging Face models.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

try:  # Optional model classes (available in transformers >= 4.46)
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:  # pragma: no cover
    Qwen2VLForConditionalGeneration = None

try:  # Qwen2.5 VL
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover
    Qwen2_5_VLForConditionalGeneration = None

try:  # Optional quantization
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None


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
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        categories: Optional[List[str]] = None,
        departments: Optional[List[str]] = None,
        device: str = "auto",
        use_gpu: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = True,
        confidence_threshold_auto: float = 0.95,
        confidence_threshold_review: float = 0.80,
        max_pages: int = 3,
        pdf_dpi: int = 280,
    ) -> None:
        self.model_name = model_name
        self.categories = categories or []
        self.departments = departments or []
        self.confidence_threshold_auto = confidence_threshold_auto
        self.confidence_threshold_review = confidence_threshold_review
        self.max_pages = max_pages
        self.pdf_dpi = pdf_dpi
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit

        if device == "auto":
            if torch.cuda.is_available() and use_gpu:
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and use_gpu:
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print("🔧 Initializing Open Source Processor")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   8-bit quantization: {load_in_8bit}")
        print(f"   4-bit quantization: {load_in_4bit}")

        self._init_vl_model()
        print("✅ Processor initialized successfully")

    def _init_vl_model(self) -> None:
        print(f"📥 Loading model: {self.model_name} ...")

        model_kwargs: Dict[str, object] = {"trust_remote_code": True}
        load_in_8bit = self.load_in_8bit
        load_in_4bit = self.load_in_4bit
        is_cuda = self.device == "cuda"
        wants_quant = (load_in_8bit or load_in_4bit) and is_cuda

        if wants_quant and BitsAndBytesConfig is None:
            raise ImportError(
                "bitsandbytes is required for 4/8-bit quantization on CUDA. Install bitsandbytes or disable quantization."
            )

        if wants_quant:
            model_kwargs["device_map"] = "auto"
            if load_in_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
        else:
            if self.device == "cuda":
                model_kwargs["dtype"] = torch.bfloat16
            elif self.device == "mps":
                model_kwargs["dtype"] = torch.float16
            else:
                model_kwargs["dtype"] = torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

        lower_name = self.model_name.lower()
        model_cls = AutoModelForCausalLM
        if any(key in lower_name for key in ("qwen2.5-vl", "qwen2_5-vl", "qwen2_5_vl")):
            if Qwen2_5_VLForConditionalGeneration is None:
                raise ImportError(
                    "Qwen2_5_VLForConditionalGeneration unavailable. Upgrade transformers to >= 4.46."
                )
            model_cls = Qwen2_5_VLForConditionalGeneration
        elif "qwen2-vl" in lower_name:
            if Qwen2VLForConditionalGeneration is None:
                raise ImportError(
                    "Qwen2VLForConditionalGeneration unavailable. Upgrade transformers to >= 4.45."
                )
            model_cls = Qwen2VLForConditionalGeneration

        self.model = model_cls.from_pretrained(self.model_name, **model_kwargs)

        if not wants_quant and self.device in {"cpu", "mps"}:
            self.model.to(self.device)

        print("✅ Vision-language model loaded successfully")

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
        message = {"role": "user", "content": [{"type": "text", "text": prompt}]}
        for image in images:
            message["content"].append({"type": "image", "image": image})

        chat_template = self.processor.apply_chat_template(
            [message], add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(text=[chat_template], images=images, return_tensors="pt")

        target_device = torch.device("cpu")
        if self.device == "cuda":
            target_device = torch.device("cuda")
        elif self.device == "mps":
            target_device = torch.device("mps")

        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(target_device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1200,
                temperature=0.0,
                do_sample=False,
            )

        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response

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


async def main() -> None:
    categories = [
        "Financial Documents",
        "Legal Documents",
        "HR Documents",
        "Operational Documents",
    ]
    departments = ["Finance", "Legal", "HR", "Operations"]

    processor = OpenSourceDocumentProcessor(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        categories=categories,
        departments=departments,
        use_gpu=True,
        load_in_8bit=True,
        load_in_4bit=False,
    )

    file_path = "samples/invoice/example.png"
    result = await processor.process_document(file_path=file_path, user_id="demo_user")

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(file_path).stem}_result.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"Saved result to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
