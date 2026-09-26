/**
 * The layout editor — the Neo-Frame "Bench" build mode, ported (core pass).
 *
 * This PR ships the CORE editing experience: draggable palette tiles that spawn
 * a cursor-following ghost, drop-onto-cell with a live valid/invalid preview,
 * drag-a-pane's-title-bar to move it (and off the grid to remove it), and
 * grid-dimension steppers. The interaction model is ported from spark-neoframe's
 * `App.tsx`; the LOOK is ported from its `App.css` onto KiroCrew's theme tokens
 * (`layoutEditor.css`), with a per-element tint like Neo-Frame's chipStyle.
 *
 * Three further interaction layers land as their own small follow-up PRs and are
 * deliberately NOT here: corner-grip resize, draggable fr track dividers, and the
 * tabs container. Each is additive and independently revertible (RFC §7's
 * one-thing-at-a-time rollout).
 *
 * It edits an EDIT MODEL (`GridSpec`, from the merged PR 1 model) and never
 * renders a live pane — no subject is selected here, so a cell only shows its
 * element label. That is deliberate (RFC §7): the editor produces and edits a
 * spec, the renderer (a later PR) draws live panes. This file therefore imports
 * only the pure model (`grid.ts` geometry + `editModel.ts` mint) and the shared
 * element vocabulary, nothing from the render path.
 */
import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  MessageSquare,
  PanelRight,
  LayoutGrid,
  Rows,
  X,
  FolderTree,
  GitBranch,
  GitPullRequest,
  Users,
  TerminalSquare,
  NotebookPen,
  ListChecks,
  type LucideIcon,
} from 'lucide-react'
import {
  canPlace,
  clamp,
  findItem,
  firstFreeCell,
  removeItemById,
  type GridItem,
  type GridSpec,
} from './grid'
import { newItem } from './editModel'
import type { ElementKind } from './layoutTree'
import { i18nT } from '../../../i18n/t'
import './layoutEditor.css'

/** Per-element glyph + tint + label catalog key, the KiroCrew analogue of
 *  Neo-Frame's def.icon/def.tint. The label is a catalog key resolved with
 *  `i18nT` at render (a module constant cannot hold a translated literal). */
const ELEMENT_META: Record<GridItem['element'], { icon: LucideIcon; labelKey: string; tint: string }> = {
  chat: { icon: MessageSquare, labelKey: 'components.crewLayout.element.chat', tint: '#5b8cf5' },
  sidePanel: { icon: PanelRight, labelKey: 'components.crewLayout.element.sidePanel', tint: '#c48ee0' },
  files: { icon: FolderTree, labelKey: 'components.crewLayout.element.files', tint: '#3ddc84' },
  git: { icon: GitBranch, labelKey: 'components.crewLayout.element.git', tint: '#e0846e' },
  changes: { icon: GitPullRequest, labelKey: 'components.crewLayout.element.changes', tint: '#f0b054' },
  subagents: { icon: Users, labelKey: 'components.crewLayout.element.subagents', tint: '#56c6d6' },
  terminal: { icon: TerminalSquare, labelKey: 'components.crewLayout.element.terminal', tint: '#9b8ec4' },
  notes: { icon: NotebookPen, labelKey: 'components.crewLayout.element.notes', tint: '#d6a656' },
  workLog: { icon: ListChecks, labelKey: 'components.crewLayout.element.workLog', tint: '#7ac77a' },
  group: { icon: LayoutGrid, labelKey: 'components.crewLayout.element.group', tint: '#f0b054' },
  tabs: { icon: Rows, labelKey: 'components.crewLayout.element.tabs', tint: '#e0846e' },
}

// The tabs container is added in a follow-up PR; the core palette places content
// elements only.
const CONTENT_PALETTE: ElementKind[] = [
  'chat',
  'sidePanel',
  'files',
  'git',
  'changes',
  'subagents',
  'terminal',
  'notes',
  'workLog',
]
const MAX_DIM = 6

type Drag =
  | { kind: 'palette'; element: GridItem['element']; x: number; y: number; startX: number; startY: number }
  | {
      kind: 'move'
      itemId: string
      element: GridItem['element']
      w: number
      h: number
      grabX: number
      grabY: number
      x: number
      y: number
    }

interface Preview {
  rect: { x: number; y: number; w: number; h: number }
  valid: boolean
  replaceId?: string
}

/** fr track weights sized to a dimension, defaulting to equal tracks. A spec's
 *  optional `colSizes`/`rowSizes` carry a non-equal split through save→reopen;
 *  a length mismatch (dimension changed) falls back to equal so the editor never
 *  renders against a stale-length track array. */
function trackSizes(sizes: number[] | undefined, dim: number): number[] {
  return sizes && sizes.length === dim ? sizes : Array(dim).fill(1)
}

export default function LayoutEditor({ spec, onChange }: { spec: GridSpec; onChange: (next: GridSpec) => void }) {
  const canvasRef = useRef<HTMLDivElement | null>(null)
  const [drag, setDrag] = useState<Drag | null>(null)

  // Track weights live ON the spec (colSizes/rowSizes) so a non-equal split
  // (e.g. the floor seed's [3, 2]) renders; the divider-drag that EDITS them
  // arrives in a follow-up PR.
  const cols = trackSizes(spec.colSizes, spec.cols)
  const rows = trackSizes(spec.rowSizes, spec.rows)

  /* ----------------------- pointer → cell targeting ---------------------- */

  const cellAt = (px: number, py: number): { cx: number; cy: number } | null => {
    const el = canvasRef.current
    if (!el) return null
    const r = el.getBoundingClientRect()
    if (px < r.left || px > r.right || py < r.top || py > r.bottom) return null
    const cx = clamp(Math.floor(((px - r.left) / r.width) * spec.cols), 0, spec.cols - 1)
    const cy = clamp(Math.floor(((py - r.top) / r.height) * spec.rows), 0, spec.rows - 1)
    return { cx, cy }
  }

  const itemAt = (cx: number, cy: number, ignoreId?: string) =>
    spec.items.find((i) => i.id !== ignoreId && cx >= i.x && cx < i.x + i.w && cy >= i.y && cy < i.y + i.h)

  /* --------------------------- live preview ------------------------------ */

  const preview: Preview | null = (() => {
    if (!drag) return null
    const loc = cellAt(drag.x, drag.y)
    if (!loc) return null
    if (drag.kind === 'palette') {
      const rect = { x: loc.cx, y: loc.cy, w: 1, h: 1 }
      if (canPlace(spec.items, rect, spec.cols, spec.rows)) return { rect, valid: true }
      // Dropping onto an occupied cell replaces the item under it.
      const target = itemAt(loc.cx, loc.cy)
      if (target) return { rect: { x: target.x, y: target.y, w: target.w, h: target.h }, valid: true, replaceId: target.id }
      return { rect, valid: false }
    }
    const w = Math.min(drag.w, spec.cols)
    const h = Math.min(drag.h, spec.rows)
    const rect = {
      x: clamp(loc.cx - drag.grabX, 0, spec.cols - w),
      y: clamp(loc.cy - drag.grabY, 0, spec.rows - h),
      w,
      h,
    }
    if (canPlace(spec.items, rect, spec.cols, spec.rows, drag.itemId)) return { rect, valid: true }
    const target = itemAt(loc.cx, loc.cy, drag.itemId)
    if (target) return { rect: { x: target.x, y: target.y, w: target.w, h: target.h }, valid: true, replaceId: target.id }
    return { rect, valid: false }
  })()

  /* ----------------------------- drops ----------------------------------- */

  useEffect(() => {
    if (!drag) return
    const onMove = (e: PointerEvent) => setDrag({ ...drag, x: e.clientX, y: e.clientY })
    const onUp = (e: PointerEvent) => {
      const p = preview
      if (drag.kind === 'palette') {
        const moved = Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) > 5
        if (!moved) {
          // A click (no drag) drops into the first free cell.
          const free = firstFreeCell(spec.items, spec.cols, spec.rows)
          if (free) onChange({ ...spec, items: [...spec.items, newItem(drag.element, { ...free, w: 1, h: 1 })] })
        } else if (p?.valid) {
          const target = p.replaceId ? spec.items.filter((i) => i.id !== p.replaceId) : spec.items
          onChange({ ...spec, items: [...target, newItem(drag.element, p.rect)] })
        }
      } else if (drag.kind === 'move') {
        const dragged = findItem(spec.items, drag.itemId)
        const loc = cellAt(e.clientX, e.clientY)
        if (dragged && !loc) {
          // Dragged off the grid → remove.
          onChange({ ...spec, items: removeItemById(spec.items, drag.itemId) })
        } else if (dragged && p?.valid) {
          let items = spec.items
          if (p.replaceId) items = items.filter((i) => i.id !== p.replaceId)
          onChange({ ...spec, items: items.map((i) => (i.id === drag.itemId ? { ...i, ...p.rect } : i)) })
        }
      }
      setDrag(null)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
  })

  /* ------------------------------ grid dims ------------------------------ */

  const minCols = Math.max(1, ...spec.items.map((i) => i.x + i.w))
  const minRows = Math.max(1, ...spec.items.map((i) => i.y + i.h))

  const setDims = (nextCols: number, nextRows: number) => {
    // Never shrink through a placed item — clamp to the max occupied edge.
    const c = clamp(nextCols, minCols, MAX_DIM)
    const r = clamp(nextRows, minRows, MAX_DIM)
    // Reset track weights to equal for the new dimension count — a resized grid
    // has a different number of tracks, so the old fr array no longer applies.
    onChange({ ...spec, cols: c, rows: r, colSizes: Array(c).fill(1), rowSizes: Array(r).fill(1) })
  }

  /* ------------------------------ item move ------------------------------ */

  const startMove = (e: React.PointerEvent, item: GridItem) => {
    if (e.button !== 0) return
    e.preventDefault()
    const loc = cellAt(e.clientX, e.clientY)
    setDrag({
      kind: 'move',
      itemId: item.id,
      element: item.element,
      w: item.w,
      h: item.h,
      grabX: loc ? clamp(loc.cx - item.x, 0, item.w - 1) : 0,
      grabY: loc ? clamp(loc.cy - item.y, 0, item.h - 1) : 0,
      x: e.clientX,
      y: e.clientY,
    })
  }

  /* ------------------------------- render -------------------------------- */

  const renderBody = (item: GridItem): ReactNode => {
    // The builder never renders a live pane — no subject is selected here, so a
    // registry cell would only return its "Select a member" empty state anyway.
    // Draw that empty state directly from the shared element meta so the editor
    // depends on the model + element meta alone, not on the renderer / any real
    // component. A cell's live rendering is entirely the renderer's job.
    const meta = ELEMENT_META[item.element]
    if (!meta)
      return (
        <div className="p-2 text-[11px]" style={{ color: 'var(--danger)' }}>
          {i18nT('components.crewLayout.unknownElement', { element: item.element })}
        </div>
      )
    return <div className="le-cell-label">{i18nT(meta.labelKey)}</div>
  }

  const ghostMeta = drag ? ELEMENT_META[drag.element] : null
  const ghostOffGrid = drag?.kind === 'move' && cellAt(drag.x, drag.y) === null
  const GhostIcon = ghostMeta?.icon

  const Tile = ({ element }: { element: GridItem['element'] }) => {
    const meta = ELEMENT_META[element]
    const Icon = meta.icon
    return (
      <button
        type="button"
        className="le-tile"
        title={i18nT('components.crewLayout.dragTile', { label: i18nT(meta.labelKey) })}
        onPointerDown={(e) => {
          if (e.button !== 0) return
          e.preventDefault()
          setDrag({ kind: 'palette', element, x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY })
        }}
      >
        <span className="le-tile-icon" style={{ ['--le-tint' as string]: meta.tint }}>
          <Icon size={15} strokeWidth={2} />
        </span>
        <span className="le-tile-label">{i18nT(meta.labelKey)}</span>
      </button>
    )
  }

  return (
    <div className="le-root" data-testid="layout-editor">
      {/* Palette shelf */}
      <div className="le-palette">
        <div className="le-palette-group">
          <span className="le-palette-label">{i18nT('components.crewLayout.paletteContent')}</span>
          <div className="le-tiles">
            {CONTENT_PALETTE.map((el) => (
              <Tile key={el} element={el} />
            ))}
          </div>
        </div>
        <span className="le-hint">{i18nT('components.crewLayout.hint')}</span>
        <div className="le-dims">
          <Stepper axis="cols" value={spec.cols} min={minCols} max={MAX_DIM} onChange={(c) => setDims(c, spec.rows)} />
          <span className="le-dims-x">×</span>
          <Stepper axis="rows" value={spec.rows} min={minRows} max={MAX_DIM} onChange={(rw) => setDims(spec.cols, rw)} />
        </div>
      </div>

      {/* Canvas */}
      <div className="le-canvas-wrap">
        <div
          ref={canvasRef}
          className="le-canvas"
          style={{
            gridTemplateColumns: cols.map((n) => `${n}fr`).join(' '),
            gridTemplateRows: rows.map((n) => `${n}fr`).join(' '),
          }}
        >
          {/* drop-target cells */}
          {Array.from({ length: spec.cols * spec.rows }, (_, i) => {
            const x = i % spec.cols
            const y = Math.floor(i / spec.cols)
            const inPrev =
              preview &&
              x >= preview.rect.x &&
              x < preview.rect.x + preview.rect.w &&
              y >= preview.rect.y &&
              y < preview.rect.y + preview.rect.h
            return (
              <div
                key={`c${x}-${y}`}
                className={`le-cell ${inPrev ? (preview!.valid ? 'is-target' : 'is-bad') : ''}`}
                style={{ gridColumn: `${x + 1}`, gridRow: `${y + 1}` }}
              />
            )
          })}

          {/* placed items */}
          {spec.items.map((item) => {
            const isMoving = drag?.kind === 'move' && drag.itemId === item.id
            const isTarget = preview?.replaceId === item.id
            const meta = ELEMENT_META[item.element]
            const BarIcon = meta.icon
            return (
              <div
                key={item.id}
                className={`le-item ${isMoving ? 'is-moving' : ''} ${isTarget ? 'is-target' : ''}`}
                style={{
                  ['--le-tint' as string]: meta.tint,
                  gridColumn: `${item.x + 1} / span ${item.w}`,
                  gridRow: `${item.y + 1} / span ${item.h}`,
                  zIndex: isMoving ? 2 : 1,
                }}
              >
                <div className="le-bar" onPointerDown={(e) => startMove(e, item)}>
                  <span className="le-bar-icon">
                    <BarIcon size={11} strokeWidth={2.25} />
                  </span>
                  <span className="le-bar-title">{i18nT(meta.labelKey)}</span>
                  <button
                    type="button"
                    className="le-close"
                    title={i18nT('components.crewLayout.remove')}
                    aria-label={i18nT('components.crewLayout.removeElement', { label: i18nT(meta.labelKey) })}
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={() => onChange({ ...spec, items: removeItemById(spec.items, item.id) })}
                  >
                    <X size={13} />
                  </button>
                </div>
                <div className="le-body">{renderBody(item)}</div>
              </div>
            )
          })}
        </div>
      </div>

      {/* drag ghost */}
      {ghostMeta && GhostIcon && drag && (
        <div
          className={`le-ghost ${ghostOffGrid ? 'is-remove' : ''}`}
          style={{ ['--le-tint' as string]: ghostMeta.tint, left: drag.x, top: drag.y }}
        >
          <span className="le-ghost-icon">
            <GhostIcon size={18} strokeWidth={2} />
          </span>
          <span>{ghostOffGrid ? i18nT('components.crewLayout.remove') : i18nT(ghostMeta.labelKey)}</span>
        </div>
      )}
    </div>
  )
}

function Stepper({
  axis,
  value,
  min,
  max,
  onChange,
}: {
  axis: 'cols' | 'rows'
  value: number
  min: number
  max: number
  onChange: (v: number) => void
}) {
  const noun = i18nT(axis === 'cols' ? 'components.crewLayout.cols' : 'components.crewLayout.rows')
  const fewer = i18nT(axis === 'cols' ? 'components.crewLayout.fewerCols' : 'components.crewLayout.fewerRows')
  const more = i18nT(axis === 'cols' ? 'components.crewLayout.moreCols' : 'components.crewLayout.moreRows')
  return (
    <span className="le-stepper" title={`${noun} (${min}–${max})`}>
      <button type="button" disabled={value <= min} onClick={() => onChange(value - 1)} aria-label={fewer}>
        −
      </button>
      <span className="le-stepper-val">{value}</span>
      <button type="button" disabled={value >= max} onClick={() => onChange(value + 1)} aria-label={more}>
        +
      </button>
    </span>
  )
}
