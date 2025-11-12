"""Quick PaddleOCR smoke test."""

from paddleocr import PaddleOCR

if __name__ == "__main__":  # pragma: no cover
    reader = PaddleOCR(use_angle_cls=False, lang="en")
    result = reader.ocr("samples/invoice/example.png", cls=False)
    for line in result:
        print(line)
