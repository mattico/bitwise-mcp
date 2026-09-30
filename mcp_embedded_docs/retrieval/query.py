"""Keyword query planning, relaxation and snippet extraction.

Adapted from Beck-Enterprises/knowledgebase scripts/search/search_engine.py.

Query semantics:
  * A query with no FTS5 operators is natural language: its content terms are
    AND-ed first; if that finds too few chunks the same terms are re-run as a
    ranked OR, then as a ranked prefix-OR. Strict hits always keep their rank.
  * A query containing `"` `*` `(` `)` `^` or a bare-uppercase OR/AND/NOT/NEAR is
    passed to FTS5 verbatim (after quoting terms FTS5 cannot read bare, like
    'sha-256' or '0x5802_4400'). If FTS5 still rejects it, the query is re-run
    as natural language and the note says so -- agents rarely mean FTS5 syntax
    when they type a stray parenthesis.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

# Bare-uppercase FTS5 keywords, quotes, parens, prefix/boost operators. ':' is
# left out on purpose: 'Errata: USB' is prose far more often than a column filter.
_EXPLICIT = re.compile(r'["()*^]|\b(?:OR|AND|NOT|NEAR)\b')
# A word as FTS5's unicode61 tokenizer sees it: letters and digits of any
# script ('µs', 'kΩ', 'I²C' are single tokens), split on everything else.
_WORD = re.compile(r"[^\W_]+")
# Punctuation trimmed from the ends of a whitespace-separated query token.
_EDGE_PUNCT = "\"'`()[]{}<>,;:!?.…"
# Dropped from generated queries only (an explicit query is never touched).
# A stopword typed in capitals is kept: 'CAN' is a bus, not a verb.
STOPWORDS = frozenset("""a an and are as at be but by can did do does for from had has
have how i if in into is it its of on or should that the their there these this to
was were what when where which who why will with""".split())

_FTS_KEYWORDS = frozenset(("OR", "AND", "NOT", "NEAR"))
# Characters FTS5 reads as structure. Everything between two of them is a term,
# and a term is only legal bare if it is alphanumeric (or non-ASCII).
_FTS_STRUCT = frozenset(' \t\r\n"()*:^,+{}')
# sqlite messages that mean "FTS5 could not parse this MATCH expression".
_SYNTAX_MESSAGES = ("fts5: syntax error", "unterminated string", "no such column",
                    "unknown special query", "malformed match")


class SearchSyntaxError(Exception):
    """Raised when FTS5 rejects a MATCH expression."""


def is_syntax_error(message: str) -> bool:
    """True for sqlite errors caused by the query text, not by the database."""
    low = message.lower()
    return any(m in low for m in _SYNTAX_MESSAGES)


def looks_explicit(query: str) -> bool:
    """True if the query already uses FTS5 syntax and should be passed through."""
    return bool(_EXPLICIT.search(query))


@dataclass(frozen=True)
class Unit:
    """One query term: `fts` is its FTS5 phrase, `text` the form to highlight."""
    fts: str
    text: str
    stop: bool = False


def units_of(query: str) -> List[Unit]:
    """Query terms, in order, deduped.

    A token with inner punctuation stays one unit, searched as a phrase:
    '2.25.2' -> "2 25 2" and 'OTG_GUSBCFG' -> "otg gusbcfg", i.e. those words
    adjacent, which is how the tokenizer indexed them. Stopwords and lone
    letters are marked `stop`.
    """
    seen, out = set(), []
    for raw in query.split():
        tok = raw.strip(_EDGE_PUNCT)
        words = _WORD.findall(tok)
        if not words:
            continue
        low_words = [w.lower() for w in words]
        if len(words) > 1:
            # 'e.g.', 'i.e.': punctuated runs of single letters carry no meaning
            stop = all(len(w) == 1 and not w.isdigit() for w in words)
            unit = Unit('"' + " ".join(low_words) + '"', tok.lower(), stop)
        else:
            w, low = words[0], low_words[0]
            stop = (low in STOPWORDS and not (w.isupper() and len(w) > 1)) or (
                len(low) == 1 and not low.isdigit())
            unit = Unit(f'"{low}"', low, stop)
        if unit.fts not in seen:
            seen.add(unit.fts)
            out.append(unit)
    content = [u for u in out if not u.stop]
    return content or out


def terms_of(query: str) -> List[str]:
    """Highlightable forms of the query's content terms, e.g. 'how do I configure
    the MPU' -> configure, mpu; 'OTG_GUSBCFG reset' -> otg_gusbcfg, reset."""
    return [u.text for u in units_of(query)]


def substring_terms(query: str) -> List[str]:
    """Terms for the substring fallback, or [] when the query has no literal.

    The fallback exists for literals FTS cannot see as indexed tokens (odd
    punctuation inside addresses and part numbers). It requires every content
    term, so a literal like '1.8' cannot match on its own.
    """
    units = [u for u in units_of(query) if not u.stop]
    has_literal = any(
        len("".join(_WORD.findall(u.text))) >= 2
        and (u.text.startswith("0x") or len(_WORD.findall(u.text)) > 1)
        for u in units)
    return [u.text for u in units] if has_literal else []


def _bareword(text: str) -> bool:
    return all(ch.isalnum() or ord(ch) > 127 for ch in text)


def sanitize_fts(query: str) -> str:
    """`query` with each term FTS5 cannot parse bare wrapped into a phrase.

    Existing phrases and the OR/AND/NOT/NEAR operators pass through untouched,
    so an explicit query keeps its meaning. ':' is quoted too, so a stray colon
    never turns into a column filter.
    """
    out: List[str] = []
    i, n = 0, len(query)
    while i < n:
        ch = query[i]
        if ch == '"':  # a phrase: copy verbatim, "" escapes a quote inside it
            j = i + 1
            while j < n:
                if query[j] == '"':
                    if j + 1 < n and query[j + 1] == '"':
                        j += 2
                        continue
                    break
                j += 1
            out.append(query[i:min(j + 1, n)])
            i = j + 1
        elif ch == ":":
            out.append(" ")
            i += 1
        elif ch in _FTS_STRUCT:
            out.append(ch)
            i += 1
        else:
            j = i
            while j < n and query[j] not in _FTS_STRUCT:
                j += 1
            word = query[i:j]
            out.append(word if word in _FTS_KEYWORDS or _bareword(word)
                       else '"' + word.replace('"', '""') + '"')
            i = j
    return "".join(out)


def build_plans(query: str) -> List[Tuple[str, str]]:
    """[(mode, fts_query)] to try in order."""
    if looks_explicit(query):
        return [("as-given", sanitize_fts(query))]
    units = units_of(query)
    if not units:
        return []
    phrases = [u.fts for u in units]
    plans = [("strict", " ".join(phrases))]  # implicit AND
    if len(phrases) > 1:
        plans.append(("relaxed-or", " OR ".join(phrases)))
    plans.append(("prefix-or", " OR ".join(f"{p}*" for p in phrases)))
    return plans


@dataclass
class PlanResult:
    hits: List[Tuple[str, float]] = field(default_factory=list)
    mode: str = ""
    note: str = ""
    terms: List[str] = field(default_factory=list)


def highlight_terms(query: str) -> List[str]:
    """Terms worth highlighting in snippets for `query`."""
    if looks_explicit(query):
        return terms_of(_EXPLICIT.sub(" ", query))
    return terms_of(query)


def run_plans(query: str, n: int, min_hits: int,
              run: Callable[[str, int], List[Tuple[str, float]]]) -> PlanResult:
    """Execute the plan ladder via `run(fts_query, limit) -> [(id, score)]`.

    Later (wider) plans only add ids the stricter ones did not find, and rank
    after them. Stops as soon as `min(min_hits, n)` distinct ids are in hand.
    """
    plans = build_plans(query)
    notes: List[str] = []
    used: List[str] = []
    if plans and plans[0][0] == "as-given":
        try:
            rows = run(plans[0][1], n)
            if rows:
                return PlanResult(hits=rows, mode="as-given", terms=highlight_terms(query))
            notes.append("nothing matched the query as FTS5 syntax (quotes, OR/AND/NOT/NEAR, "
                         "*, parentheses); searched its plain words instead.")
        except SearchSyntaxError as exc:
            notes.append(f"query is not valid FTS5 syntax ({exc}); searched its words instead.")
        used.append("as-given")
        query = _EXPLICIT.sub(" ", query)
        plans = build_plans(query)

    seen: Dict[str, Tuple[str, float]] = {}
    for mode, fts in plans:
        try:
            rows = run(fts, n)
        except SearchSyntaxError:
            continue
        used.append(mode)
        for row in rows:
            seen.setdefault(row[0], row)
        if len(seen) >= min(min_hits, n):
            break
    if len([u for u in used if u != "as-given"]) > 1 and seen:
        notes.append(f"all terms together matched too few chunks; widened to {used[-1]} "
                     "(terms optional, ranked by relevance), strict matches listed first.")
    return PlanResult(hits=list(seen.values())[:n], mode="+".join(used),
                      note=" ".join(notes), terms=terms_of(query))


# ---------------------------------------------------------------- snippets

def _stem_prefix(term: str) -> str:
    """Crude stem so 'configure' highlights 'configuration' and 'timing' 'timings'
    -- the index is porter-stemmed, so hits are often inflected forms."""
    if len(term) <= 5 or not term.isalpha():
        return term
    return term[:max(5, len(term) - 3)]


def snippets(body: str, terms: List[str], k: int = 2, width: int = 240) -> List[str]:
    """Up to k highlighted windows around term hits, preferring windows that cover
    the most distinct terms. Whitespace is collapsed. Empty if nothing matched."""
    if not body or not terms:
        return []
    stems = sorted({_stem_prefix(t) for t in terms}, key=len, reverse=True)
    pat = re.compile(r"(?<![^\W_])(" + "|".join(re.escape(s) for s in stems) + r")[^\W_]*",
                     re.I)
    spans = [(m.start(), m.end(), m.group(1).lower()) for m in pat.finditer(body)][:600]
    if not spans:
        return []
    wins: List[Dict[str, Any]] = []
    for s, e, t in spans:
        prev = wins[-1] if wins else None
        # extend the open window while hits keep coming, but cap its length so a
        # dense chunk still yields several windows instead of one wall of text
        if prev and s < prev["end"] and prev["end"] - prev["start"] < 2 * width:
            prev["end"] = max(prev["end"], e + width // 2)
            prev["terms"].add(t)
            prev["marks"].append((s, e))
            continue
        start = max(0, s - width // 3)
        if prev:
            start = max(start, prev["end"])
        wins.append({"start": start, "end": e + width, "terms": {t}, "marks": [(s, e)]})
    wins.sort(key=lambda w: (-len(w["terms"]), w["start"]))
    chosen = sorted(wins[:k], key=lambda w: w["start"])
    out = []
    for w in chosen:
        s, e = w["start"], min(len(body), w["end"])
        text = body[s:e]
        for ms, me in sorted(w["marks"], reverse=True):
            if s <= ms and me <= e:
                text = text[:ms - s] + "**" + text[ms - s:me - s] + "**" + text[me - s:]
        out.append(("… " if s else "") + " ".join(text.split()) + (" …" if e < len(body) else ""))
    return out


def matched_terms(body: str, terms: List[str]) -> List[str]:
    """The query terms that occur (as a word prefix) in `body`."""
    low = body.lower()
    return [t for t in terms
            if re.search(r"(?<![^\W_])" + re.escape(_stem_prefix(t)), low)]
