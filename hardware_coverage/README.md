# UI hardware coverage

VoltForge keeps UI selectability separate from AI electrical verification.
`service.py` reports exact board-variant coverage. `component_service.py`
reports every non-board component type in the canvas catalog.

Board electrical coverage and canvas geometry are independent. The eight
electrically verified board variants have manufacturer-bound connector-order
fixtures in `boardGeometryFixtures.v1.json`. Those fixtures drive selectable
canvas pins by board type, so Leonardo and Micro no longer inherit UNO and
Nano pin labels merely because their generated bodies share dimensions.

Every exact fixture records a geometry revision and its primary pinout source.
The generated SVG bodies remain `documented-limitation`: board outline,
connector pitch, mounting holes, and unrepresented auxiliary connectors are
not claimed to be scale-verified. Existing wire pin IDs remain valid, and a
saved node containing an obsolete/nonphysical pin ID retains its legacy pin
array rather than losing its wire anchor.

The component report deliberately verifies zero generic UI labels. Fifty-one
physical labels require manufacturer, orderable part number, package or module
revision, terminal or connector pinout, and electrical ratings before the AI
may make part-specific claims. Nine sources, references, ideal models, and
instruments are classified as `simulation-only` rather than physical parts.

The two existing exact Adafruit SSD1306 records are exposed as candidates for
the generic OLED types, but `candidateSelectionRequired=true` and
`genericLabelsMaySelectCandidate=false` prevent either record from becoming a
silent default.

Verify both UI catalogs against the independent AI inventories:

```powershell
python tools/verify_hardware_coverage.py `
  --ui-catalog ../Voltforge_UI/src/features/canvas/boardCatalog.ts `
  --ui-component-catalog ../Voltforge_UI/src/features/canvas/componentCatalog.ts `
  --ui-board-geometry-fixtures ../Voltforge_UI/src/features/canvas/boardGeometryFixtures.v1.json
```

Generate or verify the source-bound FU012 readiness receipt with
`tools/evaluate_board_geometry_readiness.py`. The item stays gated until a
browser review covers pin dots and the separate electrical/pinout badges in
the required themes and responsive widths.

New exact component records require owner-prioritized candidates, primary
manufacturer or maintained module evidence, and an immutable corpus release.
The current v1.1.0 corpus is not relabeled or overwritten for this report.
