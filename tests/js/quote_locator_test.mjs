// Node test suite for static/js/quote-locator.js (run: node tests/js/quote_locator_test.mjs)
import { createRequire } from "module";
import assert from "assert";

const require = createRequire(import.meta.url);
const L = require("../../static/js/quote-locator.js");

const checks = [];
function check(name, fn) {
    checks.push([name, fn]);
}

check("exact match", () => {
    const doc = "alpha beta gamma delta";
    const hit = L.locateQuote(doc, "beta gamma");
    assert.deepStrictEqual(hit, { start: 6, end: 16, approximate: false });
});

check("curly quotes and em dash fold to ascii", () => {
    const doc = "He said “time–pressure” affects ‘moral’ responding.";
    const quote = 'He said "time-pressure" affects \'moral\' responding.';
    const hit = L.locateQuote(doc, quote);
    assert.ok(hit && !hit.approximate, "should match exactly after normalization");
    assert.ok(doc.slice(hit.start, hit.end).includes("time"));
});

check("whitespace runs and newlines collapse", () => {
    const doc = "The planned   sample\n size is\n\n 120 participants.";
    const quote = "The planned sample size is 120 participants.";
    const hit = L.locateQuote(doc, quote);
    assert.ok(hit && !hit.approximate);
    assert.strictEqual(hit.start, 0);
    assert.strictEqual(hit.end, doc.length);
});

check("ligatures expand", () => {
    const doc = "These scientiﬁc ﬁndings conﬂate measures.";
    const quote = "scientific findings conflate";
    const hit = L.locateQuote(doc, quote);
    assert.ok(hit && !hit.approximate);
});

check("soft hyphen is ignored", () => {
    const doc = "The regis­tration was filed early.";
    const hit = L.locateQuote(doc, "The registration was filed");
    assert.ok(hit && !hit.approximate);
});

check("hyphenated line wrap dehyphenates", () => {
    const doc = "We measured the regis-\ntration quality with three raters.";
    const hit = L.locateQuote(doc, "the registration quality");
    assert.ok(hit && !hit.approximate);
    assert.ok(doc.slice(hit.start, hit.end).startsWith("the regis-"));
});

check("seed-and-extend recovers from a corrupted middle", () => {
    const prefix = "In the confirmatory analysis we compared the deontology index across conditions ";
    const middle = "using an independent samples t-test with alpha at five percent ";
    const suffix = "and report the standardized effect size alongside the raw difference.";
    const doc = "Intro text. " + prefix + middle + suffix + " Closing text.";
    const corrupted = prefix + "USING SOMETHING ENTIRELY DIFFERENT HERE INSTEAD " + suffix;
    const hit = L.locateQuote(doc, corrupted);
    assert.ok(hit, "seed-and-extend should anchor on prefix");
    assert.ok(hit.approximate, "must be flagged approximate");
    assert.ok(Math.abs(hit.start - doc.indexOf(prefix)) <= 2);
});

check("no match returns null", () => {
    assert.strictEqual(L.locateQuote("alpha beta", "entirely absent text"), null);
    assert.strictEqual(L.locateQuote("", "x"), null);
    assert.strictEqual(L.locateQuote("doc", ""), null);
});

check("spanMatches validates good offsets and rejects stale ones", () => {
    const doc = "Participants who failed two attention checks were excluded.";
    const expected = "failed two attention checks";
    const start = doc.indexOf(expected);
    assert.ok(L.spanMatches(doc, start, start + expected.length, expected));
    // Normalization tolerance: curly vs straight inside the span.
    const doc2 = "He said “yes” loudly.";
    assert.ok(L.spanMatches(doc2, 8, 13, '"yes"'));
    // Shifted span pointing at the wrong text must fail.
    assert.ok(!L.spanMatches(doc, 0, 12, expected));
    // Out-of-range must fail.
    assert.ok(!L.spanMatches(doc, -1, 5, expected));
    assert.ok(!L.spanMatches(doc, 0, doc.length + 10, expected));
});

check("pageForOffset uses exact page offsets", () => {
    const pages = [
        { page_number: 1, start: 0, end: 100 },
        { page_number: 2, start: 102, end: 220 },
    ];
    assert.deepStrictEqual(L.pageForOffset(pages, 50), { page: 1, approximate: false });
    assert.deepStrictEqual(L.pageForOffset(pages, 150), { page: 2, approximate: false });
    assert.deepStrictEqual(L.pageForOffset(pages, 999), { page: 2, approximate: true });
});

check("pageForOffset falls back to proportional estimate", () => {
    const pages = [{ page_number: 1 }, { page_number: 2 }, { page_number: 3 }];
    const hit = L.pageForOffset(pages, 500, 900);
    assert.ok(hit && hit.approximate);
    assert.strictEqual(hit.page, 2);
    assert.strictEqual(L.pageForOffset(pages, 500, undefined), null);
});

check("normalizeWithMap maps back to original indices", () => {
    const doc = "A  B—C";
    const { norm, map } = L.normalizeWithMap(doc);
    assert.strictEqual(norm, "a b-c");
    assert.strictEqual(doc[map[norm.indexOf("c")]], "C");
});

let failed = 0;
for (const [name, fn] of checks) {
    try {
        fn();
        console.log(`ok    ${name}`);
    } catch (error) {
        failed += 1;
        console.error(`FAIL  ${name}\n      ${error.message}`);
    }
}
console.log(`${checks.length - failed}/${checks.length} passed`);
process.exit(failed ? 1 : 0);
