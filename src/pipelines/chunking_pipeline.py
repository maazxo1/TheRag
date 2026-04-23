"""
Small-to-Big chunking strategy.

Retrieval precision uses small child chunks (128 tokens).
LLM generation uses their large parent chunks (1024 tokens).
This decouples retrieval quality from generation context richness.
"""

import warnings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, Document

import config


def build_small_to_big_nodes(documents: list) -> tuple[list, list, dict]:
    """
    Returns:
        parent_nodes    — large chunks (1024 tok), stored for LLM context
        child_nodes     — small chunks (128 tok), embedded + indexed for retrieval
        parent_node_map — {parent_node_id: parent_node} for fast lookup
    """
    parent_splitter = SentenceSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP,
    )
    child_splitter = SentenceSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
    )

    parent_nodes = parent_splitter.get_nodes_from_documents(documents)

    child_nodes = []
    for parent in parent_nodes:
        # Strip metadata to just the file name to avoid "metadata too long" warnings
        # when files with long names are split into small child chunks.
        slim_meta = {k: v for k, v in parent.metadata.items() if k == "file_name"}
        parent_doc = Document(text=parent.text, metadata=slim_meta)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Metadata length.*is close to chunk size")
            raw_children = child_splitter.get_nodes_from_documents([parent_doc])
        for child in raw_children:
            child_node = TextNode(
                text=child.text,
                metadata={**child.metadata, "parent_id": parent.node_id},
            )
            child_nodes.append(child_node)

    parent_node_map = {p.node_id: p for p in parent_nodes}
    return parent_nodes, child_nodes, parent_node_map


def expand_to_parents(child_results: list, parent_node_map: dict) -> list:
    """
    Given reranked child nodes, return their unique parent nodes.
    Deduplicates so each parent appears at most once.
    """
    seen, parents = set(), []
    for node in child_results:
        pid = node.metadata.get("parent_id", node.node_id)
        if pid not in seen and pid in parent_node_map:
            seen.add(pid)
            parents.append(parent_node_map[pid])
    return parents
