from rag_chunking.chunking.base import Chunk, Chunker
from rag_chunking.chunking.fixed import FixedSizeChunker
from rag_chunking.chunking.structure import StructureChunker
from rag_chunking.config import FixedParams, StructureParams


def make_chunker(params: FixedParams | StructureParams) -> Chunker:
    if isinstance(params, FixedParams):
        return FixedSizeChunker(params.size_chars, params.overlap_chars)
    return StructureChunker(
        params.min_tokens, params.max_tokens, params.overlap_tokens, params.prefix
    )


__all__ = ["Chunk", "Chunker", "FixedSizeChunker", "StructureChunker", "make_chunker"]
