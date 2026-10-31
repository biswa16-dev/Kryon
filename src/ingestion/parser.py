import os
import re
import csv
import zlib
import zipfile
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from PIL import Image


class DocumentSection(BaseModel):
    section_id: str
    heading: str
    level: int
    content: str
    page_estimate: int = 1


class ParsedDocument(BaseModel):
    doc_id: str
    title: str
    filename: str
    file_path: str
    file_type: str
    sections: List[DocumentSection] = []
    full_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sha256: str


class DocumentParser:
    """Multi-format document parser supporting PDF, Word (DOCX/DOC), Excel (XLSX/XLS/CSV), Images (PNG/JPG/WEBP), and Markdown/Text."""

    @staticmethod
    def compute_sha256(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()

    @classmethod
    def parse_file(cls, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found at {path}")

        ext = path.suffix.lower()
        if ext in [".md", ".markdown", ".txt", ".json"]:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return cls.parse_markdown_text(content, filename=path.name, file_path=str(path), file_type="markdown")
        elif ext == ".pdf":
            return cls.parse_pdf(path)
        elif ext in [".docx", ".doc"]:
            return cls.parse_docx(path)
        elif ext in [".xlsx", ".xls"]:
            return cls.parse_xlsx(path)
        elif ext in [".csv", ".tsv"]:
            return cls.parse_csv(path)
        elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
            return cls.parse_image(path)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return cls.parse_markdown_text(content, filename=path.name, file_path=str(path), file_type="text")

    @classmethod
    def parse_markdown_text(cls, text: str, filename: str = "document.md", file_path: str = "", file_type: str = "markdown") -> ParsedDocument:
        title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
        stem = Path(filename).stem.replace("_", " ").title()
        title = title_match.group(1).strip() if title_match else stem

        header_pattern = re.compile(r"^(#{1,4})\s+(.+)$", flags=re.MULTILINE)
        matches = list(header_pattern.finditer(text))

        sections: List[DocumentSection] = []
        if not matches:
            sections.append(
                DocumentSection(
                    section_id="sec-0",
                    heading=title,
                    level=1,
                    content=text.strip(),
                    page_estimate=1,
                )
            )
        else:
            for idx, match in enumerate(matches):
                level = len(match.group(1))
                heading = match.group(2).strip()
                start_pos = match.end()
                end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                section_body = text[start_pos:end_pos].strip()
                page_estimate = max(1, (match.start() // 1800) + 1)
                sections.append(
                    DocumentSection(
                        section_id=f"sec-{idx+1}",
                        heading=heading,
                        level=level,
                        content=f"{heading}\n\n{section_body}",
                        page_estimate=page_estimate,
                    )
                )

        sha = cls.compute_sha256(text)
        return ParsedDocument(
            doc_id=f"doc_{sha[:10]}",
            title=title,
            filename=filename,
            file_path=file_path,
            file_type=file_type,
            sections=sections,
            full_text=text,
            metadata={
                "char_count": len(text),
                "section_count": len(sections),
                "estimated_pages": max(1, len(text) // 1800),
            },
            sha256=sha,
        )

    @classmethod
    def parse_docx(cls, path: Path) -> ParsedDocument:
        import xml.etree.ElementTree as ET
        paragraphs = []
        try:
            with zipfile.ZipFile(path) as z:
                if "word/document.xml" in z.namelist():
                    xml_content = z.read("word/document.xml")
                    root = ET.fromstring(xml_content)
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    for p in root.iterfind(".//w:p", ns):
                        texts = [node.text for node in p.iterfind(".//w:t", ns) if node.text]
                        if texts:
                            paragraphs.append("".join(texts))
        except Exception:
            try:
                with open(path, "rb") as f:
                    raw = f.read()
                words = re.findall(rb"[\x20-\x7E]{4,}", raw)
                paragraphs = [w.decode("latin1", errors="replace") for w in words]
            except Exception as e:
                paragraphs = [f"Word Document: {path.name} (Error: {e})"]

        md_text = f"# {path.stem.replace('_', ' ').title()}\n\n" + "\n\n".join(paragraphs)
        return cls.parse_markdown_text(md_text, filename=path.name, file_path=str(path), file_type="docx")

    @classmethod
    def parse_xlsx(cls, path: Path) -> ParsedDocument:
        import xml.etree.ElementTree as ET
        rows = []
        try:
            with zipfile.ZipFile(path) as z:
                shared_strings = []
                if "xl/sharedStrings.xml" in z.namelist():
                    ss_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
                    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                    for si in ss_root.iterfind(".//m:si", ns):
                        t = "".join([node.text for node in si.iterfind(".//m:t", ns) if node.text])
                        shared_strings.append(t)

                sheet_files = sorted([n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")])
                for sfile in sheet_files:
                    sname = Path(sfile).stem.title()
                    rows.append(f"## Sheet: {sname}")
                    sroot = ET.fromstring(z.read(sfile))
                    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                    for r in sroot.iterfind(".//m:row", ns):
                        cells = []
                        for c in r.iterfind(".//m:c", ns):
                            v = c.find("m:v", ns)
                            t_type = c.get("t")
                            if v is not None and v.text:
                                if t_type == "s" and v.text.isdigit() and int(v.text) < len(shared_strings):
                                    cells.append(shared_strings[int(v.text)])
                                else:
                                    cells.append(v.text)
                        if cells:
                            rows.append(" | ".join(cells))
                    rows.append("")
        except Exception:
            try:
                with open(path, "rb") as f:
                    raw = f.read()
                words = re.findall(rb"[\x20-\x7E]{4,}", raw)
                rows = [w.decode("latin1", errors="replace") for w in words]
            except Exception as e:
                rows = [f"Spreadsheet: {path.name} (Error: {e})"]

        md_text = f"# {path.stem.replace('_', ' ').title()}\n\n" + "\n".join(rows)
        return cls.parse_markdown_text(md_text, filename=path.name, file_path=str(path), file_type="xlsx")

    @classmethod
    def parse_csv(cls, path: Path) -> ParsedDocument:
        lines = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    lines.append(f"## Table: {path.stem}")
                    lines.append(" | ".join(header))
                    lines.append(" | ".join(["---"] * len(header)))
                for row in reader:
                    if any(cell.strip() for cell in row):
                        lines.append(" | ".join(row))
        except Exception as e:
            lines = [f"CSV Table: {path.name} (Error: {e})"]

        md_text = f"# {path.stem.replace('_', ' ').title()}\n\n" + "\n".join(lines)
        return cls.parse_markdown_text(md_text, filename=path.name, file_path=str(path), file_type="csv")

    @classmethod
    def parse_image(cls, path: Path) -> ParsedDocument:
        text_lines = [f"# Image: {path.stem.replace('_', ' ').title()}"]
        try:
            with Image.open(path) as img:
                fmt = img.format or path.suffix.replace(".", "").upper()
                w, h = img.size
                mode = img.mode
                text_lines.append(f"**Image File**: `{path.name}` ({fmt}, {w}x{h} px, Mode: {mode})")
                text_lines.append(f"\n## Visual Document Properties")
                text_lines.append(f"- **Aspect Ratio**: {w/max(1, h):.2f}")
                text_lines.append(f"- **Resolution**: {w * h:,} total pixels")

                exif_data = getattr(img, "_getexif", lambda: None)()
                if exif_data:
                    text_lines.append("\n## Embedded Metadata")
                    for tag_id, val in exif_data.items():
                        text_lines.append(f"- Tag `{tag_id}`: {str(val)[:80]}")

                text_lines.append(f"\n## Document Context")
                text_lines.append(f"This is an uploaded visual academic record (`{path.name}`). It represents an academic syllabus, timetable chart, regulation flowchart, or exam grade slip submitted by a university student.")
        except Exception as e:
            text_lines.append(f"Visual Asset: {path.name} (Processing note: {e})")

        md_text = "\n".join(text_lines)
        return cls.parse_markdown_text(md_text, filename=path.name, file_path=str(path), file_type="image")

    @classmethod
    def parse_pdf(cls, path: Path) -> ParsedDocument:
        text_blocks = []
        try:
            with open(path, "rb") as f:
                raw_bytes = f.read()

            stream_pattern = re.compile(b"stream[\r\n]+(.*?)[\r\n]+endstream", re.DOTALL)
            for match in stream_pattern.finditer(raw_bytes):
                stream_data = match.group(1)
                decompressed = None
                try:
                    decompressed = zlib.decompress(stream_data)
                except Exception:
                    try:
                        decompressed = zlib.decompress(stream_data, -15)
                    except Exception:
                        pass
                
                target = decompressed if decompressed else stream_data
                try:
                    text_str = target.decode("latin1", errors="replace")
                    tjs = re.findall(r"\(((?:[^()\\]|\\.)*)\)\s*(?:Tj|\'|\")", text_str)
                    if tjs:
                        clean_tjs = [t.replace("\\(", "(").replace("\\)", ")").replace("\\n", " ").strip() for t in tjs if len(t.strip()) > 1]
                        if clean_tjs:
                            text_blocks.append(" ".join(clean_tjs))
                except Exception:
                    pass

            if not text_blocks:
                content = raw_bytes.decode("latin1", errors="replace")
                tokens = re.findall(r"\((.*?)\)[\s]*Tj", content)
                if tokens:
                    text_blocks.append(" ".join(tokens))

            extracted = "\n\n".join(text_blocks) if text_blocks else f"PDF Content of {path.name}"
            return cls.parse_markdown_text(extracted, filename=path.name, file_path=str(path), file_type="pdf")
        except Exception as e:
            fallback = f"Document content for {path.name} (Error parsing PDF: {e})"
            return cls.parse_markdown_text(fallback, filename=path.name, file_path=str(path), file_type="pdf")
