from backend.services.text_normalize import decode_bytes, normalize_text, decode_and_normalize


def test_decode_bytes_utf8_and_bom():
    assert decode_bytes("café résumé".encode("utf-8")) == "café résumé"
    assert decode_bytes("\ufeffhello".encode("utf-8")) == "hello"  # BOM stripped
    assert decode_bytes("already a str") == "already a str"


def test_decode_bytes_cp1252_recovers_smart_punctuation():
    # Bytes that are invalid UTF-8 but valid Windows-1252 (mis-saved Word export).
    raw = b"don" + bytes([0x92]) + b"t " + bytes([0x93]) + b"hi" + bytes([0x94]) + b" 50" + bytes([0x96]) + b"60"
    decoded = decode_bytes(raw)
    assert decoded == "don’t “hi” 50–60"  # recovered, not dropped


def test_normalize_maps_typographic_punctuation_to_ascii():
    assert normalize_text("“quote” — it’s 50–60…") == '"quote" - it\'s 50-60...'


def test_normalize_folds_ligatures():
    assert normalize_text("eﬃcient ﬁnding") == "efficient finding"


def test_normalize_preserves_real_letters_and_notation():
    # Accents, Greek, micro sign, superscript, fraction must survive.
    assert normalize_text("Müller café η² µg ½") == "Müller café η² µg ½"


def test_normalize_strips_control_replacement_and_normalises_newlines():
    assert normalize_text("a\x00b�c­d e\r\nf") == "abcd e\nf"  # NUL, U+FFFD, soft hyphen gone; CRLF->LF
    assert normalize_text("a\u00a0b") == "a b"  # nbsp -> space


def test_normalize_is_idempotent_and_empty_safe():
    sample = "“ﬁx’ — it… café"
    assert normalize_text(normalize_text(sample)) == normalize_text(sample)
    assert normalize_text("") == ""
    assert decode_bytes(b"") == ""
    assert decode_and_normalize(b"") == ""


def test_evidence_builder_normalizes_display_but_keeps_raw_cache_text():
    from backend.services.evidence import build_text_evidence_source
    raw = "The “smart” quote — it’s eﬃcient…"
    payload = build_text_evidence_source(source_id="P", label="Paper", text=raw, chunk_prefix="PAPER")
    seg = " ".join(payload["segments"])
    # displayed segments + manifest text are normalised to ASCII
    assert all(ord(c) < 128 for c in seg), seg
    assert '"smart"' in seg and "it's" in seg and "efficient" in seg and "..." in seg and " - " in seg
    assert all(ord(c) < 128 for c in next(iter(payload["chunks"].values()))["text"])
    # render docText is normalised too, so the text-mode locator matches the normalised quote
    assert all(ord(c) < 128 for c in payload["render_data"]["text"])
    # the cache-key source text stays RAW so corpus/manifest IDs stay aligned (keys unchanged)
    assert payload["text"] == raw
