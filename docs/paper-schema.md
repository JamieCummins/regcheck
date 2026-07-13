# Paper schema — RegCheck proposal

A proposed schema for an ingested document (paper, preregistration, or registry
record), for the shared import/ingest effort. It is framed to slot into a
papercheck / scienceverse‑style tabular model (an `info`/`authors`/`text`/
`references` set of tables; see <https://www.scienceverse.org/schema/>) and adds
the few things RegCheck depends on that a plain GROBID→text pipeline tends to drop.

Status: draft for discussion (Jamie / Lisa / Jakub).

## What RegCheck needs the schema to honour

These are the non‑negotiables from RegCheck's side; everything else is detail.

1. **One canonical text string + character offsets as the universal coordinate
   system.** Every section, segment, box, and downstream quote references
   `char_start`/`char_end` into a single `full_text`. RegCheck's quote‑tracing and
   highlighting work by locating a quote in the original text and mapping its
   offset → page → rectangle. Offsets are parser‑agnostic, so this survives a
   GROBID→bibr swap.
2. **Keep the verbatim original, not just the parsed version.** The parsed/cleaned
   `text` is a *derived convenience*; the original must be recoverable, because
   RegCheck fuzzy‑matches quotes against the original and shows it in the viewer.
3. **A coordinate layer (pages + boxes) that is optional but standardised.** PDF
   parsers (PyMuPDF, GROBID) populate it; born‑digital/structured sources leave it
   empty. This is the part a text‑only paper schema usually lacks and RegCheck most
   needs.
4. **Source‑type‑agnostic.** RegCheck ingests papers *and* ClinicalTrials.gov
   records *and* OSF/Word preregistrations through the same machinery, so the
   schema should hold a "born‑structured" record (sections = API fields, no boxes)
   in the same shape as a parsed PDF.
5. **Provenance on every document** (`parser`, `parser_version`) so consumers know
   how the text was produced — directly relevant to the grobid→bibr migration.

## Tables

### `paper` — one row per document (the `info` table)

| column | type | notes |
|---|---|---|
| `paper_id` | text PK | stable id |
| `title`, `abstract`, `doi`, `journal`, `year` | text | bibliographic metadata |
| `source_format` | enum | `pdf` · `docx` · `xml` · `html` · `ctgov_json` · `osf` |
| `original_filename` | text | as uploaded |
| `raw_file_uri` | text | pointer to the **stored original bytes** (the actual file) |
| `full_text` | text | **the canonical verbatim extracted text** that all offsets index into |
| `n_pages` | int? | PDF only |
| `parser`, `parser_version`, `parsed_at` | text / ts | provenance (`grobid` / `bibr` / `pymupdf` / `dpt2` / `ctgov_api`) |

### `sections` — canonical structure + hierarchy

| column | type | notes |
|---|---|---|
| `section_id` | text PK | |
| `paper_id` | FK | |
| `parent_id` | FK? | self‑reference for nesting |
| `position` | int | order |
| `level` | int | heading depth |
| `header` | text | the heading text (canonical home) |
| `section_type` | enum? | `abstract`/`intro`/`method`/`results`/`discussion`/… (controlled, nullable) |
| `char_start`, `char_end` | int | span in `paper.full_text` |

### `text` — segment table (sentence‑ or paragraph‑level; mirrors papercheck's `div`/`p`/`s`)

| column | type | notes |
|---|---|---|
| `text_id` | text PK | stable; what a retrieval/annotation layer keys onto |
| `paper_id`, `section_id` | FK | |
| `header` | text? | **denormalised cache** of the section header (see Open questions) |
| `div`, `p`, `s` | int | structural indices |
| `type` | enum | `sentence`/`paragraph`/`heading`/`caption`/`table`/`footnote` |
| `text` | text | parsed/cleaned segment |
| `text_raw` | text | **verbatim original span** (see Open questions) |
| `char_start`, `char_end` | int | span in `paper.full_text` |

### `references`

`ref_id`, `paper_id`, `position`, `raw_text` (verbatim), `doi`, `title`, `year`,
`authors`, `container_title`.

### `authors`

`paper_id`, `position`, `name`, `given`, `family`, `orcid`, `affiliation`,
`is_corresponding`, `email`.

### `pages` — *RegCheck extension; PDF only*

`paper_id`, `page_number`, `width`, `height`, `char_start`, `char_end`.
RegCheck already stores exactly this: each page's character range + point
dimensions.

### `boxes` — *RegCheck extension; PDF only*

`text_id` (FK), `page_number`, `x0`, `y0`, `x1`, `y1`, in PDF‑point space matching
`pages.width`/`pages.height`. Per‑segment bounding boxes for rendering + highlight
overlays.

## Open questions (recommendations)

**Section headers — only in `sections`?** Keep `sections` as the **canonical** home
(header + hierarchy + offsets), put a `section_id` FK on every `text` row, *and*
keep a **denormalised `header` (and maybe `section_type`) cached on the `text`
row**. The `text` table is the one actually consumed — flattened into a dataframe
or pasted into an LLM prompt — and "which section is this sentence in?" is needed
constantly; forcing a join every time is friction. Normalised truth + a cheap
cache. (RegCheck specifically uses the section header as a fallback *locator* for
non‑PDF sources where page numbers are meaningless.)

**Where does the original live?** Both places, deliberately: `text_raw` on each
segment (the verbatim span) **and** `paper.full_text` as the whole canonical
original that every `char_start`/`char_end` indexes into, **plus** `raw_file_uri`
to the actual file. Treat parsed `text` as a lossy convenience over `full_text`;
never let it be the only copy. RegCheck can't function without this — it locates
quotes by matching against the original and maps offsets → pages → rects, so a
"parsed‑only" representation breaks highlighting and is exactly where GROBID's
mangling bites.

## How RegCheck maps onto this

- RegCheck's evidence **manifest is already this schema in miniature**: per source
  it stores the canonical `full_text`, a `pages` array (page → char range +
  width/height), and per‑chunk PDF rectangles — i.e. `paper.full_text` + `pages` +
  `boxes`.
- A RegCheck **"chunk"** is just a contiguous run of `text` segments (a
  `char_start`/`char_end` span); its retrieval IDs (`PREREG_xxxx` / `PAPER_xxxx`)
  and relevance scores live in a **retrieval layer above** this schema, not in it.
- A **ClinicalTrials.gov** record drops in cleanly: `sections` = the API fields
  (Eligibility, Outcome Measures, Interventions…), `text` = the field values,
  `pages`/`boxes` empty, `parser = ctgov_api`. Same for an OSF/Word prereg
  (sections from headings, no boxes).

## Decision to settle

The **coordinate layer (`pages`/`boxes`)** is the main place RegCheck's
requirements exceed a text‑only paper schema. Decide whether it lives in the shared
schema as optional tables (recommended) or stays a RegCheck‑side extension keyed by
`text_id` — either works as long as the `text` ids and `char` offsets are shared
and stable.
