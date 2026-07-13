/* RegCheck quote locator — robust text matching for evidence tracing.
 *
 * Pure functions, no DOM. Loaded before report.js in the browser
 * (window.RegCheckLocator) and imported directly by the node test suite.
 *
 * The core problem: a quote (RAG chunk text) must be found inside a rendered
 * document text even when the two differ in whitespace, quote/dash glyphs,
 * ligatures, soft hyphens, or hyphenated line wraps. Matching is tiered:
 *   1. exact indexOf
 *   2. normalized match (glyph folding + whitespace collapse, index-mapped)
 *   3. dehyphenated normalized match (handles "exam- ple" line wraps)
 *   4. seed-and-extend: anchor a distinctive fragment, extend to quote length
 * All results are spans in ORIGINAL document indices.
 */
(function (root, factory) {
    const api = factory();
    if (typeof module !== "undefined" && module.exports) module.exports = api;
    if (root) root.RegCheckLocator = api;
})(typeof window !== "undefined" ? window : null, function () {
    "use strict";

    const LIGATURES = {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
    };
    const SINGLE_QUOTES = /[\u2018\u2019\u201A\u201B\u2032]/;
    const DOUBLE_QUOTES = /[\u201C\u201D\u201E\u201F\u2033]/;
    const DASHES = /[\u2010\u2011\u2012\u2013\u2014\u2212]/;
    const SPACES = /\s/; // \s matches Unicode whitespace (incl. NBSP) in JS
    const DROPPED = /[\u00AD\u200B\u200C\u200D\uFEFF]/;

    // Normalize text for matching while keeping a map from every normalized
    // character back to its original index.
    function normalizeWithMap(text) {
        const chars = [];
        const map = [];
        const push = (ch, origIndex) => {
            chars.push(ch);
            map.push(origIndex);
        };
        for (let i = 0; i < text.length; i += 1) {
            const c = text[i];
            if (DROPPED.test(c)) continue;
            if (LIGATURES[c]) {
                const expansion = LIGATURES[c];
                for (let k = 0; k < expansion.length; k += 1) push(expansion[k], i);
                continue;
            }
            if (SPACES.test(c)) {
                if (chars.length && chars[chars.length - 1] !== " ") push(" ", i);
                continue;
            }
            let out = c;
            if (SINGLE_QUOTES.test(c)) out = "'";
            else if (DOUBLE_QUOTES.test(c)) out = '"';
            else if (DASHES.test(c)) out = "-";
            else out = c.toLowerCase();
            for (let k = 0; k < out.length; k += 1) push(out[k], i);
        }
        while (chars.length && chars[chars.length - 1] === " ") {
            chars.pop();
            map.pop();
        }
        return { norm: chars.join(""), map };
    }

    const ALNUM = /[a-z0-9]/;

    // Remove "- " sequences produced by hyphenated line wraps ("exam-\nple"
    // normalizes to "exam- ple"; dehyphenation yields "example").
    function dehyphenate(result) {
        const chars = [];
        const map = [];
        const norm = result.norm;
        for (let i = 0; i < norm.length; i += 1) {
            if (
                norm[i] === "-" &&
                norm[i + 1] === " " &&
                i > 0 && ALNUM.test(norm[i - 1]) &&
                i + 2 < norm.length && ALNUM.test(norm[i + 2])
            ) {
                i += 1; // skip the hyphen and the following space
                continue;
            }
            chars.push(norm[i]);
            map.push(result.map[i]);
        }
        return { norm: chars.join(""), map };
    }

    function spanFromNormHit(docResult, normStart, normLength) {
        if (normStart < 0) return null;
        const lastIndex = Math.min(normStart + normLength - 1, docResult.map.length - 1);
        if (lastIndex < normStart) return null;
        return {
            start: docResult.map[normStart],
            end: docResult.map[lastIndex] + 1,
        };
    }

    // Locate quoteText inside docText. Returns {start, end, approximate} in
    // original docText indices, or null.
    function locateQuote(docText, quoteText) {
        if (!docText || !quoteText) return null;

        const exact = docText.indexOf(quoteText);
        if (exact >= 0) return { start: exact, end: exact + quoteText.length, approximate: false };

        const doc = normalizeWithMap(docText);
        const quote = normalizeWithMap(quoteText);
        if (!doc.norm || !quote.norm) return null;

        let hit = doc.norm.indexOf(quote.norm);
        if (hit >= 0) {
            const span = spanFromNormHit(doc, hit, quote.norm.length);
            if (span) return { start: span.start, end: span.end, approximate: false };
        }

        const docDehyph = dehyphenate(doc);
        const quoteDehyph = dehyphenate(quote);
        const variants = [
            [docDehyph, quote],
            [doc, quoteDehyph],
            [docDehyph, quoteDehyph],
        ];
        for (const [docVariant, quoteVariant] of variants) {
            hit = docVariant.norm.indexOf(quoteVariant.norm);
            if (hit >= 0) {
                const span = spanFromNormHit(docVariant, hit, quoteVariant.norm.length);
                if (span) return { start: span.start, end: span.end, approximate: false };
            }
        }

        // Seed-and-extend: anchor on a distinctive fragment of the quote.
        const qn = quoteDehyph.norm;
        if (qn.length >= 32) {
            const seedLength = Math.min(48, Math.floor(qn.length / 2));
            const middleStart = Math.max(0, Math.floor(qn.length / 2 - seedLength / 2));
            const seeds = [
                { offset: 0, text: qn.slice(0, seedLength) },
                { offset: middleStart, text: qn.slice(middleStart, middleStart + seedLength) },
                { offset: qn.length - seedLength, text: qn.slice(qn.length - seedLength) },
            ];
            for (const seed of seeds) {
                if (seed.text.length < 24) continue;
                const seedHit = docDehyph.norm.indexOf(seed.text);
                if (seedHit < 0) continue;
                const normStart = Math.max(0, seedHit - seed.offset);
                const span = spanFromNormHit(docDehyph, normStart, qn.length);
                if (span) return { start: span.start, end: span.end, approximate: true };
            }
        }
        return null;
    }

    // Validate that a stored [start, end) span still corresponds to the
    // expected text (guards against stale or shifted offsets).
    function spanMatches(docText, start, end, expected) {
        if (!docText || !expected) return false;
        if (!Number.isFinite(start) || !Number.isFinite(end)) return false;
        if (start < 0 || end <= start || end > docText.length) return false;
        const a = normalizeWithMap(docText.slice(start, end)).norm;
        const b = normalizeWithMap(expected).norm;
        if (!a || !b) return false;
        if (a === b) return true;
        const shorter = a.length <= b.length ? a : b;
        const longer = a.length <= b.length ? b : a;
        return shorter.length / longer.length >= 0.8 && longer.indexOf(shorter) >= 0;
    }

    // Map a char offset to its page. Pages may carry exact {start, end}
    // offsets (new manifests); otherwise fall back to a proportional estimate
    // (old manifests), flagged approximate.
    function pageForOffset(pages, offset, textLength) {
        if (!Array.isArray(pages) || !pages.length || !Number.isFinite(offset)) return null;
        const withOffsets = pages.filter(
            (p) => Number.isFinite(p.start) && Number.isFinite(p.end) && p.end >= p.start
        );
        if (withOffsets.length === pages.length) {
            for (const page of withOffsets) {
                if (offset >= page.start && offset <= page.end) {
                    return { page: page.page_number, approximate: false };
                }
            }
            const last = withOffsets[withOffsets.length - 1];
            if (offset > last.end) return { page: last.page_number, approximate: true };
            return { page: withOffsets[0].page_number, approximate: true };
        }
        if (Number.isFinite(textLength) && textLength > 0) {
            const index = Math.min(
                pages.length - 1,
                Math.max(0, Math.floor((offset / textLength) * pages.length))
            );
            return { page: pages[index].page_number || index + 1, approximate: true };
        }
        return null;
    }

    return { normalizeWithMap, dehyphenate, locateQuote, spanMatches, pageForOffset };
});
