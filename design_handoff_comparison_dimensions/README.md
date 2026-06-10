# Handoff: Comparison Dimensions Setup Screen (RegCheck)

## Overview
Step 4 of 8 in the RegCheck wizard, where a researcher specifies the **dimensions** (name + definition) on which a preregistration and its published paper will be compared. Users can edit dimensions, reorder them by dragging, delete them, add new ones, and load discipline-specific default sets (Psychology, Clinical/Medical, Economics, General social science).

The chosen design is a **two-pane layout**: a compact ordered list on the left (with the defaults dropdown as its header) and a quiet, borderless editor for the selected dimension on the right.

## About the Design Files
The files in this bundle are **design references created in HTML/React** — prototypes showing intended look and behavior, **not production code to copy directly**. The task is to **recreate this design in the target codebase's existing environment** (its framework, component library, and conventions). If the RegCheck site already has a stack (e.g., React/Next.js), implement there using its established patterns; only the visual and behavioral spec below is binding. The prototype intentionally mimics RegCheck's existing dark-navy visual identity — reuse the production site's actual nav, progress bar, and footer rather than the placeholder chrome in these files.

## Fidelity
**High-fidelity.** Colors, typography, spacing, radii, and interaction states are final intent. Recreate pixel-perfectly, substituting equivalent tokens/components where the codebase already defines them.

## Screen / View

### Comparison dimensions (wizard step 4 of 8)
- **Purpose**: Review and edit the set of comparison dimensions before running the comparison.
- **Page scaffold** (existing site chrome, shown for context only): top nav (RegCheck brand + links, TOOLS active), progress bar at 50% with `STEP 4 OF 8` label, and a Back / Next footer below the panel (Back left, ghost style; Next right, primary gradient).
- **Main panel**: max-width 1280px page container (40px side padding); panel has 20px radius, 1px border `#222B40`, background = subtle vertical gradient `rgba(22,29,46,.55) → rgba(17,23,38,.55)` over page bg, padding `38px 40px 34px`.
- **Panel header**: H1 "Comparison dimensions" (31px / 800 / -0.025em), one-line sub: "RegCheck compares the registration and paper on each of these dimensions, in order." (15.5px, `#9BA7BD`). No other copy — show, don't tell.

#### Two-pane editor (`.tp2`)
- CSS grid: `grid-template-columns: 320px 1fr; gap: 20px; align-items: stretch;` **fixed height 520px** — both panes are always exactly equal height.
- Below 920px viewport width: single column, height auto; the list area caps at `max-height: 340px` with internal scroll.

**Left pane — ordered list** (`.tp2-left`)
- Container: radius 16px, border 1px `#222B40`, background `rgba(12,17,28,.45)`, padding 12px, `display:flex; flex-direction:column; min-height:0; overflow:hidden`.
- **Defaults dropdown as list header** (full width, margin-bottom 8px). Button: radius 12px, bg `#161D2E`, border 1px `#2C3753`; contents: mono uppercase caption "DEFAULTS" (10.5px, `#6B7689`), cyan dot (8px, `#38BDF8` with 3px soft halo), current discipline label (14px/600), chevron pushed right (rotates 180° when open). Menu: anchored below, full pane width, radius 14px, bg `#111726`, border `#2C3753`, large drop shadow; caption row "LOAD DEFAULTS FOR…"; each option = 30px icon tile + name (14px/600) + meta ("7 dimensions", 12px `#6B7689`) + check mark on the active item. Selecting an option **replaces the entire list** with that discipline's defaults (see data.js).
- **Scrollable rows area** (`.tp2-rows`): `flex:1; min-height:0; overflow-y:auto;` 6px gap. The dropdown header and Add button never scroll — only the rows.
- **Row** (`.tp-row`): flex, gap 11px, padding `11px 12px`, radius 11px, transparent bg; hover bg `#161D2E`; selected bg `rgba(59,130,246,.14)` + 1px border `rgba(59,130,246,.45)`. Contents:
  - Drag handle: 2×3 dot grip, only visible on hover/selected (opacity 0 → 1), `cursor:grab`.
  - Number badge: 26px rounded square (radius 8), gradient `135deg #2563EB → #38BDF8`, white mono 12px digit. Numbers always reflect current order.
  - Name: 14.5px/600, single line, ellipsis. Empty name renders placeholder "Untitled" in `#515C72`.
  - Delete: trash icon button (28px), only visible on row hover; hover state = red tint (`rgba(242,96,126,.14)` bg, `#F2607E` icon).
- **Add dimension** button: full-width, dashed 1.5px border `#2C3753`, radius 13px, plus icon + "Add dimension" (14.5px/700, no wrap); hover = accent border + `rgba(59,130,246,.14)` bg.

**Right pane — quiet editor** (`.tp2-detail`)
- Container: radius 16px, bg `#161D2E`, border 1px `#222B40`, padding `26px 30px 28px`, `overflow-y:auto; min-height:0`.
- Top row: 32px number badge (same gradient) + mono uppercase meta "OF {N}" (11px, letter-spacing .14em, `#6B7689`, no-wrap). **No delete button here** — deletion lives only on list rows.
- **Name field**: borderless transparent input, 23px/800/-0.02em, padding `5px 9px`, radius 9, negative left margin -9px so text aligns with the meta row. Hover/focus: bg `#1B2335`; focus also adds 1px ring `#2C3753`. Placeholder "Untitled dimension" in `#515C72`. Autofocus when the dimension has no name yet (i.e., just added).
- **Definition field**: borderless auto-growing textarea, 15.5px, line-height 1.65, color `#9BA7BD` (brightens to `#EDF1F8` on focus), same hover/focus treatment. Placeholder: "Add a definition — what should RegCheck look for on this dimension?" No visible box until interaction — reads as plain text.
- **Empty state** (zero dimensions): centered ghost icon + "Add a dimension to get started."

## Interactions & Behavior
- **Select**: clicking a list row selects it and shows it in the editor. Edits apply live to the list (name updates as you type).
- **Drag-to-reorder**: drag is initiated **only from the grip handle** (so text in the row stays selectable), but the **whole row** is the drag image. While dragging: dragged row at 40% opacity; the row under the cursor shows a 2px cyan top inset line as the drop indicator. Drop moves the item to that index; numbering re-derives from order. (Prototype uses HTML5 DnD with a "armed by handle mousedown" pattern; in production use the codebase's DnD library if one exists, e.g. dnd-kit.)
- **Add**: appends a blank dimension, selects it, auto-scrolls the list to the bottom so the new row is visible (effect keyed on list length, after render commit), and focuses the name field.
- **Delete**: trash on row hover removes immediately. If the selected row is deleted, selection falls back to the first remaining dimension; empty list shows the editor empty state.
- **Defaults preset**: replaces the whole list and keeps the chosen discipline highlighted in the menu. *Open decision: consider a confirm dialog when the user has unsaved edits.*
- **Pane height**: fixed 520px; ≥ ~8 rows triggers internal scroll of the rows area only. Both panes always end on the same line.
- Transitions: backgrounds/borders 140–150ms ease; dropdown opens with a 140ms fade/slide (`translateY(-6px) → 0`).
- **Next/Back**: out of scope here — Next should persist the ordered dimension array `[{name, definition}, …]` to the wizard state.

## State Management
- `dimensions: Array<{ id, name, definition }>` — single source of truth; order in array = comparison order.
- `selectedId` — currently edited dimension; effect keeps it valid when items are deleted (fallback to first).
- `discipline` — active preset key; selecting a preset replaces `dimensions` with a deep copy of that set.
- Drag state (local): `fromIndex`, `overIndex`, `armedIndex` (handle pressed).
- Validation suggestion for production: disable Next or warn when a dimension has an empty name.

## Design Tokens
**Colors**
| Token | Value | Use |
|---|---|---|
| bg | `#0A0E17` | page background (plus faint blue radial glows) |
| surface | `#111726` | panel, menus |
| surface-2 | `#161D2E` | cards, inputs, row hover |
| surface-3 | `#1B2335` | quiet-field hover/focus bg |
| border | `#222B40` | default borders |
| border-2 | `#2C3753` | input borders, focus rings |
| text | `#EDF1F8` | primary text |
| text-dim | `#9BA7BD` | secondary text |
| text-mute | `#6B7689` | labels, meta |
| text-faint | `#515C72` | placeholders |
| accent | `#3B82F6` / `#38BDF8` / `#2563EB` | gradient `135deg #2563EB → #38BDF8` |
| accent-soft | `rgba(59,130,246,.14)` | selected/hover tints |
| danger | `#F2607E` (soft: `rgba(242,96,126,.14)`) | delete states |

**Typography**: UI = Hanken Grotesk (400–800); meta/labels/numbers = JetBrains Mono (400–600, uppercase, letter-spacing .12–.18em). Scale: 31 (H1) / 23 (editor name) / 15.5 (body) / 14.5 (rows) / 11 (mono meta).

**Radii**: rows 11 · inputs/buttons 12–13 · panes/cards 16 · panel 20. **Pane height**: 520px desktop.

**Shadows**: card `0 1px 0 rgba(255,255,255,.02) inset, 0 8px 24px -12px rgba(0,0,0,.6)`; primary-button glow `0 8px 30px -8px rgba(56,140,246,.55)`; menu `0 18px 50px -16px rgba(0,0,0,.7)`.

## Assets
None required. All icons are simple inline SVG line icons (grip dots, chevron, plus, trash, check, info) — substitute the codebase's icon set (e.g., Lucide: `grip-vertical`, `chevron-down`, `plus`, `trash-2`, `check`). Fonts via Google Fonts: Hanken Grotesk, JetBrains Mono.

## Files
- `Comparison Dimensions v2.html` — entry point; open in a browser to see the live prototype.
- `styles.css` — all design tokens (`:root`) and component styles. The final two-pane styles are under the `Two-pane v2` section; earlier sections (`.acc-*`, `.card-*`, `.inl-*`, `.tp` v1, `.proto-bar`) belong to discarded explorations and can be ignored.
- `data.js` — discipline default sets (Psychology, Clinical/Medical, Economics, General social science) with full definition copy. **Reuse this content verbatim.**
- `chrome.jsx` — shared primitives: `SortableList` (handle-armed drag reorder), `AutoTextarea` (auto-growing), `PresetDropdown`, icons, plus placeholder site chrome (`RegNav`, `Progress`, `Footer`).
- `twopane_final.jsx` — the chosen two-pane component (`TwoPaneFinal`). Primary implementation reference.
- `app_v2.jsx` — state container wiring handlers (`onChange`, `onDelete`, `onAdd`, `onReorder`, `onPreset`) to `TwoPaneFinal`.
