"""Extract plain text from PDF files using pdfminer.six.

Includes light post-processing for CMA weather bulletins:
- Remove page footer markers like ``— 12 —`` followed by form feed
- Collapse double newlines that are not sentence endings.
"""
from __future__ import annotations

import argparse
import logging
import re
from io import StringIO
from pathlib import Path

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

logger = logging.getLogger(__name__)

_PAGE_FOOTER_RE = re.compile(r"\n\n— \d+ —\n\n\f", flags=re.MULTILINE)
_NON_SENTENCE_DOUBLE_NL = re.compile(r"(?<!\.)\n\n", flags=re.MULTILINE)


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract and lightly clean text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Cleaned plain text.
    """
    output = StringIO()
    try:
        with open(pdf_path, "rb") as fp:
            parser = PDFParser(fp)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, output, laparams=LAParams())
            try:
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                for page in PDFPage.create_pages(doc):
                    interpreter.process_page(page)
            finally:
                device.close()
        text = output.getvalue()
    finally:
        output.close()

    text = _PAGE_FOOTER_RE.sub("", text)
    text = _NON_SENTENCE_DOUBLE_NL.sub("\n", text)
    return text


def process_pdf(pdf_path: str | Path, output_path: str | Path | None = None) -> str:
    """Extract text from a single PDF and optionally write to *output_path*.

    Returns:
        Extracted text.
    """
    text = extract_text_from_pdf(pdf_path)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        logger.info("Saved: %s", out)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from PDF files")
    parser.add_argument("--input", required=True, help="PDF file path")
    parser.add_argument("--output", default=None, help="Output text file path (optional)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    text = process_pdf(args.input, args.output)
    if args.output is None:
        print(text)


if __name__ == "__main__":
    main()
