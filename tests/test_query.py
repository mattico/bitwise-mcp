"""Tests for keyword query planning, relaxation and snippets."""

from mcp_embedded_docs.retrieval import query as q


def test_terms_drop_stopwords_and_split_identifiers():
    assert q.terms_of("how do I configure the MPU") == ["configure", "mpu"]
    assert q.terms_of("OTG_GUSBCFG reset") == ["otg_gusbcfg", "reset"]


def test_literals_stay_phrases_and_unicode_terms_survive():
    # a dotted section number must not dissolve into its parts
    assert q.build_plans("2.25.2")[0][1] == '"2 25 2"'
    assert q.build_plans("OTG_GUSBCFG FDMOD")[0][1] == '"otg gusbcfg" "fdmod"'
    assert q.terms_of("setup time 10 µs") == ["setup", "time", "10", "µs"]
    assert q.terms_of("I²C pull-up kΩ") == ["i²c", "pull-up", "kω"]
    # capitalized stopwords are kept ('CAN' is a bus); 'e.g.' is noise
    assert q.terms_of("CAN bit timing e.g. FDCAN") == ["can", "bit", "timing", "fdcan"]


def test_plain_query_builds_strict_then_relaxed_ladder():
    plans = q.build_plans("HSE bypass startup")
    assert [m for m, _ in plans] == ["strict", "relaxed-or", "prefix-or"]
    assert plans[0][1] == '"hse" "bypass" "startup"'
    assert plans[1][1] == '"hse" OR "bypass" OR "startup"'


def test_explicit_query_is_passed_through_with_bad_barewords_quoted():
    assert q.looks_explicit('"clock security" OR CSS')
    assert q.sanitize_fts('sha-256 OR "a b"') == '"sha-256" OR "a b"'
    # a colon never becomes a column filter
    assert q.sanitize_fts('errata: USB') == 'errata  USB'


def test_run_plans_widens_only_when_strict_is_too_narrow():
    calls = []

    def run(fts, limit):
        calls.append(fts)
        if " OR " in fts:
            return [("strict1", 3.0), ("wide1", 2.0), ("wide2", 1.0)]
        return [("strict1", 5.0)]

    res = q.run_plans("alpha beta", 10, 3, run)
    assert res.mode == "strict+relaxed-or"
    assert [h[0] for h in res.hits] == ["strict1", "wide1", "wide2"]
    assert "widened" in res.note

    calls.clear()
    res = q.run_plans("alpha beta", 10, 1, run)
    assert res.mode == "strict" and len(calls) == 1


def test_run_plans_recovers_from_fts_syntax_error():
    def run(fts, limit):
        if "(" in fts:
            raise q.SearchSyntaxError("fts5: syntax error")
        return [("c1", 1.0)]

    res = q.run_plans("RCC_BDCR (backup domain", 5, 1, run)
    assert res.hits == [("c1", 1.0)]
    assert "not valid FTS5 syntax" in res.note


def test_explicit_query_with_no_hits_falls_back_to_words():
    def run(fts, limit):
        # the verbatim query (with NOT) matches nothing; plain words do
        return [] if "NOT" in fts else [("c1", 1.0)]

    res = q.run_plans("why does USB NOT enumerate", 5, 1, run)
    assert res.hits == [("c1", 1.0)]
    assert res.mode.startswith("as-given+strict")
    assert "plain words" in res.note


def test_no_widen_note_when_nothing_matches():
    res = q.run_plans("zzqx", 5, 3, lambda fts, limit: [])
    assert res.hits == [] and res.note == ""


def test_no_widen_note_when_widening_adds_nothing():
    # one strict hit; prefix-or runs but finds the same chunk
    res = q.run_plans("GUSBCFG", 5, 3, lambda fts, limit: [("c1", 1.0)])
    assert res.mode == "strict+prefix-or"
    assert res.note == ""


def test_single_term_widen_note_does_not_mention_all_terms():
    def run(fts, limit):
        return [("c1", 1.0), ("c2", 0.5)] if "*" in fts else [("c1", 1.0)]

    res = q.run_plans("GUSBCFG", 5, 3, run)
    assert [h[0] for h in res.hits] == ["c1", "c2"]
    assert "prefixes" in res.note and "all terms" not in res.note


def test_fallback_says_not_was_not_applied():
    def run(fts, limit):
        return [] if "NOT" in fts else [("c1", 1.0)]

    res = q.run_plans("USB NOT host", 5, 1, run)
    assert res.hits == [("c1", 1.0)]
    assert "NOT was not applied" in res.note

    res = q.run_plans('"USB core" OR host', 5, 1, lambda fts, limit: [] if "OR" in fts else [("c1", 1.0)])
    assert "NOT was not applied" not in res.note


def test_snippets_highlight_inflected_forms_and_collapse_whitespace():
    body = "Intro.\n\n   The IWDG   counter stops when the core is halted in debug mode.   Timings vary."
    out = q.snippets(body, ["debug", "timing"], k=2)
    assert out
    joined = " ".join(out)
    assert "**debug**" in joined
    assert "**Timings**" in joined
    assert "  " not in joined


def test_matched_terms_uses_word_prefixes():
    assert q.matched_terms("Configuration of the watchdog", ["configure", "watchdog", "dma"]) == [
        "configure", "watchdog"]


def test_substring_terms_require_a_real_literal_and_keep_all_terms():
    assert q.substring_terms("0x58024400 RCC_BDCR plain words") == [
        "0x58024400", "rcc_bdcr", "plain", "words"]
    assert q.substring_terms("plain words only") == []
    assert q.substring_terms("- --") == []
    assert q.substring_terms("e.g. zzqqxx") == []


def test_syntax_error_classification():
    assert q.is_syntax_error('fts5: syntax error near ""')
    assert q.is_syntax_error("unterminated string")
    assert not q.is_syntax_error("database is locked")
    assert not q.is_syntax_error("fts5: missing row 12 from content table 'main'.'chunks'")
