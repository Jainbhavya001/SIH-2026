# Frontend — Sentinel Screening Console

React 18 + TypeScript + Vite + Tailwind CSS + Framer Motion. A dark
"command-center" UI for border-security officers to run and review AI
document screenings.

## Setup & run

```bash
cd frontend
npm install
npm run dev
```

Opens on http://localhost:5173. The dev server proxies `/api/*` to the
backend at `http://127.0.0.1:8000` (see `vite.config.ts`) — start the
backend first (see [../backend/README.md](../backend/README.md)).

## Build

```bash
npm run build   # type-checks with tsc, then bundles with vite
npm run preview # serve the production build locally
```

## Design system

- **Palette**: deep navy/charcoal base with cyan/teal accents for primary
  actions, amber for "review," rose for "reject," emerald for "clear" —
  defined once in `tailwind.config.ts` (`base.*`, `accent.*`).
- **Typography**: Inter for UI text, JetBrains Mono for document
  numbers/MRZ data (evokes a terminal/security feel and makes long
  alphanumeric strings easier to scan).
- **Motion**: Framer Motion for entrance transitions and the animated risk
  gauge; kept subtle and fast (200–1100ms) so it reads as responsive, not
  showy.

## Folder map

```
frontend/src/
├── pages/            One file per route (Overview, New Scan, Report, Audit Log)
├── components/
│   ├── layout/        AppShell, Sidebar, Topbar (live system-status badges)
│   ├── upload/         Document type selector + drag-and-drop dropzone
│   ├── risk/            RiskGauge, RiskBadge, ScoreBar — the report's visual language
│   └── ui/               Generic primitives: Button, Card, Badge, Spinner
├── lib/               api.ts (typed fetch client), utils.ts (formatting/color helpers)
├── types/              Mirrors the backend's Pydantic response models
└── App.tsx            Route table
```
