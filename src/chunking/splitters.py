import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.ingestion.parser import ParsedDocument, DocumentSection


class TextChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    filename: str
    section_heading: str
    page_number: int
    text: str
    token_count: int
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkingEngine:
    """Configurable multi-strategy chunking engine for RAG pipelines."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # Approximate 1 token ~= 4 characters or word count * 1.3
        words = text.split()
        return max(1, int(len(words) * 1.25))

    @classmethod
    def recursive_split(
        cls,
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ) -> List[str]:
        if separators is None:
            separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

        target_char_size = chunk_size * 4
        overlap_char_size = chunk_overlap * 4

        if len(text) <= target_char_size:
            return [text.strip()] if text.strip() else []

        # Find best separator
        sep = separators[-1]
        for s in separators:
            if s in text:
                sep = s
                break

        splits = text.split(sep)
        chunks: List[str] = []
        current_chunk = ""

        for part in splits:
            candidate = f"{current_chunk}{sep}{part}" if current_chunk else part
            if len(candidate) <= target_char_size:
                current_chunk = candidate
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # Keep overlap from previous chunk
                overlap_text = current_chunk[-overlap_char_size:] if overlap_char_size > 0 else ""
                current_chunk = f"{overlap_text}{sep}{part}" if overlap_text else part

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    @classmethod
    def chunk_document(
        cls,
        doc: ParsedDocument,
        strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> List[TextChunk]:
        """Chunks a parsed document preserving section headings and citations."""
        all_chunks: List[TextChunk] = []
        chunk_counter = 0

        if strategy == "section":
            # Section-based chunking: preserve whole sections unless they exceed chunk_size
            for sec in doc.sections:
                sec_tokens = cls.estimate_tokens(sec.content)
                if sec_tokens <= chunk_size * 1.5:
                    chunk_counter += 1
                    all_chunks.append(
                        TextChunk(
                            chunk_id=f"{doc.doc_id}_c{chunk_counter}",
                            doc_id=doc.doc_id,
                            doc_title=doc.title,
                            filename=doc.filename,
                            section_heading=sec.heading,
                            page_number=sec.page_estimate,
                            text=sec.content,
                            token_count=sec_tokens,
                            chunk_index=chunk_counter,
                            metadata={"strategy": "section", "heading": sec.heading},
                        )
                    )
                else:
                    sub_pieces = cls.recursive_split(sec.content, chunk_size, chunk_overlap)
                    for piece in sub_pieces:
                        chunk_counter += 1
                        all_chunks.append(
                            TextChunk(
                                chunk_id=f"{doc.doc_id}_c{chunk_counter}",
                                doc_id=doc.doc_id,
                                doc_title=doc.title,
                                filename=doc.filename,
                                section_heading=sec.heading,
                                page_number=sec.page_estimate,
                                text=piece,
                                token_count=cls.estimate_tokens(piece),
                                chunk_index=chunk_counter,
                                metadata={"strategy": "section_subsplit", "heading": sec.heading},
                            )
                        )
        else:
            # Recursive strategy across sections to preserve logical context
            for sec in doc.sections:
                pieces = cls.recursive_split(sec.content, chunk_size, chunk_overlap)
                for piece in pieces:
                    chunk_counter += 1
                    all_chunks.append(
                        TextChunk(
                            chunk_id=f"{doc.doc_id}_c{chunk_counter}",
                            doc_id=doc.doc_id,
                            doc_title=doc.title,
                            filename=doc.filename,
                            section_heading=sec.heading,
                            page_number=sec.page_estimate,
                            text=piece,
                            token_count=cls.estimate_tokens(piece),
                            chunk_index=chunk_counter,
                            metadata={"strategy": strategy, "heading": sec.heading},
                        )
                    )

        return all_chunks

