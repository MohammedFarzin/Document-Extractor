"""Async client for the Lyncs Archive API."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import aiohttp


class LyncsAPIError(RuntimeError):
    """Raised when the archive API returns an error."""


class LyncsArchiveClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "LyncsArchiveClient":
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.session:
            await self.session.close()

    async def submit_document(self, document_data: Dict, file_path: str) -> Dict:
        file_result = await self._upload_file(file_path)
        file_id = file_result.get("file_id")
        if not file_id:
            raise LyncsAPIError("File upload succeeded but no file_id returned")

        payload = self._prepare_submission_data(document_data, file_id)
        archive_result = await self._submit_metadata(payload)

        return {
            "success": True,
            "archive_id": archive_result.get("archive_id"),
            "file_id": file_id,
            "submitted_at": datetime.now().isoformat(),
            "document_name": payload.get("document_name"),
        }

    async def _upload_file(self, file_path: str) -> Dict:
        url = f"{self.base_url}/api/files/upload"
        data = aiohttp.FormData()
        data.add_field(
            "file",
            open(file_path, "rb"),
            filename=Path(file_path).name,
            content_type="application/pdf",
        )

        async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as session:
            async with session.post(url, data=data) as response:
                if response.status != 200:
                    raise LyncsAPIError(
                        f"File upload failed: {response.status} - {await response.text()}"
                    )
                return await response.json()

    async def _submit_metadata(self, payload: Dict) -> Dict:
        if not self.session:
            raise RuntimeError("Client session not initialized. Use 'async with'.")

        url = f"{self.base_url}/api/archive/upload"
        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                raise LyncsAPIError(
                    f"Archive submission failed: {response.status} - {await response.text()}"
                )
            return await response.json()

    def _prepare_submission_data(self, document_data: Dict, file_id: str) -> Dict:
        submission = document_data.get("document_submission", {})
        basic = submission.get("basic_information", {})
        categorization = submission.get("categorization", {})
        dates = submission.get("dates", {})
        tags = submission.get("tags", [])
        metadata = submission.get("extraction_metadata", {})

        return {
            "document_name": basic.get("document_name", "Untitled Document"),
            "description": basic.get("description", ""),
            "category": categorization.get("category"),
            "department": categorization.get("department"),
            "tags": tags,
            "document_date": dates.get("document_date"),
            "due_date": dates.get("due_date"),
            "expiry_date": dates.get("expiry_date"),
            "physical_document_available": submission.get("additional_options", {}).get(
                "physical_document_available", False
            ),
            "related_documents": submission.get("additional_options", {}).get(
                "link_to_related_documents", []
            ),
            "file_id": file_id,
            "metadata": {
                "auto_extracted": True,
                "extraction_confidence": document_data.get("confidence_score", 0),
                "extraction_timestamp": document_data.get("processing_timestamp"),
                "document_type": metadata.get("document_type"),
                "language": metadata.get("language"),
            },
        }


async def _demo() -> None:  # pragma: no cover
    async with LyncsArchiveClient("https://qa-lyncs.example", "demo-key") as client:
        fake_doc = {
            "document_submission": {
                "basic_information": {
                    "document_name": "Invoice_Demo",
                    "description": "Demo invoice",
                },
                "categorization": {
                    "category": "Financial Documents",
                    "department": "Finance",
                },
                "tags": ["invoice"],
                "dates": {"document_date": "2024-01-15"},
                "extraction_metadata": {"document_type": "invoice", "language": "en"},
            },
            "confidence_score": 0.9,
            "processing_timestamp": datetime.now().isoformat(),
        }
        # Replace with a real file and endpoint before running
        try:
            result = await client.submit_document(fake_doc, "samples/invoice/example.pdf")
            print(result)
        except Exception as exc:
            print(f"Demo submission failed: {exc}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_demo())
