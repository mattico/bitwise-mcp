---
description: Search embedded systems documentation for registers, memory maps, and peripheral details
---

Search the indexed embedded systems documentation for the user's query. Use the `search_docs` MCP tool to find relevant sections, register definitions, and memory map information from ingested PDF datasheets.

If the user is looking for a specific register by name, prefer `find_register` (case-insensitive; the peripheral prefix is optional).

Search in the manual's own vocabulary (peripheral, register and bit names, "errata", "initialization sequence"); if a search misses, rephrase with synonyms rather than repeating it. Each hit carries a chunk id and 1-based pages: use `read_section(chunk_id)` to read the whole section, or `read_pages(doc, pages)` for exact pages, instead of searching again.

If no documents are indexed yet, suggest using `ingest_docs` first with the path to a PDF datasheet.
