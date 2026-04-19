# Pipeline Tab — Tri-Table Redesign

**Status:** Approved 2026-04-19
**Target file:** `interface/static/index.html` — `renderPipeline()` at line 1765

## Goal

Replace the current flex-row + colored-circle rendering of the Pipeline tab with three side-by-side tables: Layers, Quality Components, Session Rules.

## Layout

CSS grid, 3 equal columns on wide screens (`grid-template-columns: repeat(3, 1fr); gap: 14px`), stacks to one column below ~1100px.

### Table 1 — 9-Layer Confirmation Pipeline

Columns: `#`, `Layer`, `Status`, `Description`, `Detail`

- Static order by layer ID 1→9 (matches signal-pipeline semantics).
- Status rendered as a colored pill chip in the Status column only:
  - `PASS` → `var(--green)`
  - `WEAK` → `var(--yellow)`
  - `FAIL` / `BLOCKED` → `var(--red)`
  - `PENDING` → `var(--cyan)`
  - `UNKNOWN` → `var(--text3)`
- Fields pulled from `/api/layers` response: `id`, `name`, `status`, `description`, `detail`.
- Description in smaller secondary text; detail (may be empty) in monospace.

### Table 2 — Quality Components

Columns: `Component`, `Score`, `Bar`

- One row per key in `/api/quality` `components` (7 keys: `htf_trend`, `technical_confluence`, `smc_confluence`, `volume_flow`, `regime`, `ml_confidence`, `liquidity_depth`).
- `Score` column: numeric 0-100, monospace, right-aligned.
- `Bar` column: 6px horizontal bar, width = score%, blue fill.
- Header row above the table shows the big total (existing 36px number) + `/100` + partial/preview flag text.
- `tvQualityChip` top-bar update logic preserved.

### Table 3 — Session Rules

Columns: `Field`, `Value`

Rows:
- `Session` → uppercase `active_session.name` (or "None" fallback)
- `UTC Window` → `start:00 — end:00`
- `Size Mult` → `Nx`
- `Allowed Types` → comma list (or "NONE (no-trade)")
- Conditional rows: `ICT Killzone` (if `in_killzone`) → "⚡ ACTIVE — 1.2x"; `Weekend` (if `is_weekend`) → "🚫 Trading halted"

## Behavior

- All three tables populated by the existing `loadPipeline()` `Promise.all` fetch (no endpoint changes).
- Empty/error state: "Pipeline data unavailable" unchanged.
- No sorting, no filtering, no row interaction — static tables.
- Server restart not required (static file served fresh on reload).

## Backend / wiring notes (verified)

- `/api/layers` (`dashboard_api.py:2395`) — 9-layer payload intact; no changes.
- `/api/quality` (`dashboard_api.py:2592`) — 7-component breakdown; no changes.
- `/api/session` (`dashboard_api.py:2462`) — session + killzone; no changes.

## Non-goals

- Column sorting, filtering, pagination.
- Per-row color tinting (chip-only).
- Changing API schema or signal_generator internals.
