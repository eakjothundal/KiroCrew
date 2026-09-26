/**
 * Standalone dev harness for the layout editor (RFC §7, PR 2 — core pass).
 *
 * This is NOT a shipped surface — it is a route developers open to exercise the
 * editor in isolation: drag palette tiles onto the grid, move/remove panes,
 * resize the grid via the steppers. It holds a `GridSpec` in local state and
 * renders the editor over it, plus a live read-out of the serialized
 * `LayoutTree` (`toTree` → `serializeLayout`) so you can see the model the
 * editor produces round-trips. It depends ONLY on the merged PR 1 model and the
 * editor; it touches nothing on the render path or the Crew Members page.
 */
import { useMemo, useState } from 'react'
import LayoutEditor from '../components/crew/layout/LayoutEditor'
import { toTree } from '../components/crew/layout/editModel'
import { serializeLayout } from '../components/crew/layout/layoutTree'
import type { GridSpec } from '../components/crew/layout/grid'
import { i18nT } from '../i18n/t'

/** A small starter arrangement so the harness opens with something to grab:
 *  a 2×2 grid with chat spanning the left column and a side panel top-right. */
const SEED_SPEC: GridSpec = {
  cols: 2,
  rows: 2,
  colSizes: [3, 2],
  items: [
    { id: 'seed-chat', element: 'chat', x: 0, y: 0, w: 1, h: 2 },
    { id: 'seed-side', element: 'sidePanel', x: 1, y: 0, w: 1, h: 1 },
  ],
}

export default function LayoutEditorHarnessPage() {
  const [spec, setSpec] = useState<GridSpec>(SEED_SPEC)

  const serialized = useMemo(() => serializeLayout(toTree(spec)), [spec])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, background: 'var(--bg)' }}>
      <header
        style={{
          flexShrink: 0,
          display: 'flex',
          alignItems: 'baseline',
          gap: 12,
          padding: '10px 16px',
          borderBottom: '1px solid var(--border)',
        }}
      >
        <h1 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: 'var(--text)' }}>
          {i18nT('pages.layoutEditorHarness.title')}
        </h1>
        <span style={{ fontSize: 11.5, color: 'var(--muted)' }}>{i18nT('pages.layoutEditorHarness.subtitle')}</span>
        <button
          type="button"
          onClick={() => setSpec(SEED_SPEC)}
          style={{
            marginLeft: 'auto',
            fontSize: 11.5,
            padding: '4px 10px',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border)',
            background: 'var(--bg-elevated)',
            color: 'var(--text)',
            cursor: 'pointer',
          }}
        >
          {i18nT('pages.layoutEditorHarness.reset')}
        </button>
      </header>

      <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>
        <div style={{ flex: 1, minWidth: 0, minHeight: 0 }}>
          <LayoutEditor spec={spec} onChange={setSpec} />
        </div>
        <aside
          style={{
            flexShrink: 0,
            width: 320,
            minHeight: 0,
            display: 'flex',
            flexDirection: 'column',
            borderLeft: '1px solid var(--border)',
            background: 'var(--bg)',
          }}
        >
          <div
            style={{
              flexShrink: 0,
              padding: '8px 12px',
              fontSize: 10,
              fontWeight: 700,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              color: 'var(--muted)',
              borderBottom: '1px solid var(--border)',
            }}
          >
            {i18nT('pages.layoutEditorHarness.serialized')}
          </div>
          <pre
            data-testid="harness-serialized"
            style={{
              flex: 1,
              minHeight: 0,
              margin: 0,
              padding: 12,
              overflow: 'auto',
              fontSize: 11,
              lineHeight: 1.5,
              color: 'var(--text)',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {serialized}
          </pre>
        </aside>
      </div>
    </div>
  )
}
