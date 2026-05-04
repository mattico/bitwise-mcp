"""Tests for pdfplumber table bbox compatibility logic."""

from __future__ import annotations

from mcp_embedded_docs.ingestion.table_detector import TableDetector


class _TableWithTupleCells:
    def __init__(self):
        self.cells = [
            (10.0, 20.0, 40.0, 30.0),
            (42.0, 20.0, 70.0, 30.0),
            (10.0, 31.0, 70.0, 45.0),
        ]

    @property
    def bbox(self):
        # Simulate the failing pdfplumber path seen in production.
        raise AttributeError("'tuple' object has no attribute 'bbox'")


class _TableWithBBox:
    def __init__(self):
        self.bbox = (1.0, 2.0, 3.0, 4.0)


def test_extract_pdfplumber_table_bbox_prefers_bbox_property():
    table = _TableWithBBox()
    assert TableDetector._extract_pdfplumber_table_bbox(table) == (1.0, 2.0, 3.0, 4.0)


def test_extract_pdfplumber_table_bbox_falls_back_to_tuple_cells():
    table = _TableWithTupleCells()
    assert TableDetector._extract_pdfplumber_table_bbox(table) == (10.0, 20.0, 70.0, 45.0)
