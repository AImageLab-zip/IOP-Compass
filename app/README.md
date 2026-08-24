# The IOP-Compass viewer

Upload a patient's intraoral photographs, get FDI-numbered tooth instances, correct
them, save.

```bash
make install     # frontend dependencies, once
make check       # validate every weight before starting
make dev         # API on :5000, UI on http://localhost:5173
```

Full documentation — the pipeline it serves, the API, the editing tools, the
configuration variables and the honest limits — is in
[`../docs/viewer.md`](../docs/viewer.md).

## Layout

```
backend/
  server.py      app factory, CLI, start-up weight validation
  api.py         the HTTP surface
  pipeline.py    classify -> segment (full image) -> post-process
  cases.py       bounded in-memory case store
  contours.py    masks <-> polygons, in the uploaded image's coordinates
  config.py      every path and threshold, all environment-overridable
  smoke_test.py  drives the real API with a held-out patient
frontend/
  src/App.jsx           layout shell and case orchestration
  src/state/useCase.js  case state and the undo stack
  src/components/       Header, ViewStrip, Canvas, Toolbar, InstancePanel, StatusBar
  src/lib/              FDI vocabulary and colours, polygon geometry
  src/theme.css         the dark clinical theme
```

The backend implements no model logic of its own: it calls the same
`iop_compass` modules the benchmark called, so the viewer cannot drift from the
measured pipeline without `make smoke` failing.
