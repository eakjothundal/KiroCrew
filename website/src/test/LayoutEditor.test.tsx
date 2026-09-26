/**
 * Layout editor (core pass) component behavior — the interactions that do NOT
 * depend on real pixel layout (jsdom has none): the palette renders every
 * content element, a placed pane's close button removes it through `onChange`,
 * and the dimension steppers respect the occupied-edge minimum and reset tracks
 * on grow. The pure drag/drop GEOMETRY is covered by grid.test.ts on the model;
 * here we prove the editor wires the model ops to the DOM.
 *
 * Plus the harness round-trip: the standalone dev page's serialized read-out is
 * the `toTree → serializeLayout` of the live spec.
 *
 * Resize, track dividers, and the tabs container land in follow-up PRs and are
 * tested there.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent, within } from '@testing-library/react'
import LayoutEditor from '../components/crew/layout/LayoutEditor'
import LayoutEditorHarnessPage from '../pages/LayoutEditorHarnessPage'
import type { GridSpec } from '../components/crew/layout/grid'
import { toTree } from '../components/crew/layout/editModel'
import { serializeLayout } from '../components/crew/layout/layoutTree'

function baseSpec(): GridSpec {
  return {
    cols: 2,
    rows: 2,
    colSizes: [3, 2],
    items: [
      { id: 'a', element: 'chat', x: 0, y: 0, w: 1, h: 2 },
      { id: 'b', element: 'sidePanel', x: 1, y: 0, w: 1, h: 1 },
    ],
  }
}

describe('LayoutEditor (core)', () => {
  it('renders the palette with every content element', () => {
    render(<LayoutEditor spec={baseSpec()} onChange={() => {}} />)
    for (const label of ['Chat', 'Side panel', 'Files', 'Git', 'Changes', 'Subagents', 'Terminal', 'Notes', 'Work log']) {
      expect(screen.getByTitle(`Drag ${label} onto the grid`)).toBeTruthy()
    }
  })

  it('renders each placed item as a card with its label', () => {
    render(<LayoutEditor spec={baseSpec()} onChange={() => {}} />)
    const editor = screen.getByTestId('layout-editor')
    expect(within(editor).getAllByText('Chat').length).toBeGreaterThan(0)
    expect(within(editor).getAllByText('Side panel').length).toBeGreaterThan(0)
  })

  it('removes a pane through onChange when its close button is clicked', () => {
    const onChange = vi.fn()
    render(<LayoutEditor spec={baseSpec()} onChange={onChange} />)
    fireEvent.click(screen.getByLabelText('Remove Side panel'))
    expect(onChange).toHaveBeenCalledTimes(1)
    const next: GridSpec = onChange.mock.calls[0][0]
    expect(next.items.map((i) => i.id)).toEqual(['a'])
  })

  it('does not shrink columns below the occupied edge (stepper min)', () => {
    // An item spanning both columns forces min cols = 2, so the "fewer cols"
    // stepper button is disabled and cannot drop a track through the pane.
    const spec: GridSpec = {
      cols: 2,
      rows: 1,
      items: [{ id: 'wide', element: 'chat', x: 0, y: 0, w: 2, h: 1 }],
    }
    const onChange = vi.fn()
    render(<LayoutEditor spec={spec} onChange={onChange} />)
    const fewer = screen.getByLabelText('fewer cols') as HTMLButtonElement
    expect(fewer.disabled).toBe(true)
    fireEvent.click(fewer)
    expect(onChange).not.toHaveBeenCalled()
  })

  it('grows the grid through the stepper and resets tracks to equal', () => {
    const onChange = vi.fn()
    render(<LayoutEditor spec={baseSpec()} onChange={onChange} />)
    fireEvent.click(screen.getByLabelText('more cols'))
    expect(onChange).toHaveBeenCalledTimes(1)
    const next: GridSpec = onChange.mock.calls[0][0]
    expect(next.cols).toBe(3)
    // New track count → equal weights, since the old [3,2] no longer fits 3 cols.
    expect(next.colSizes).toEqual([1, 1, 1])
  })
})

describe('LayoutEditorHarnessPage', () => {
  it('renders the editor and a serialized read-out of the seed spec', () => {
    render(<LayoutEditorHarnessPage />)
    expect(screen.getByTestId('layout-editor')).toBeTruthy()
    const readout = screen.getByTestId('harness-serialized')
    const seed: GridSpec = {
      cols: 2,
      rows: 2,
      colSizes: [3, 2],
      items: [
        { id: 'seed-chat', element: 'chat', x: 0, y: 0, w: 1, h: 2 },
        { id: 'seed-side', element: 'sidePanel', x: 1, y: 0, w: 1, h: 1 },
      ],
    }
    expect(readout.textContent).toBe(serializeLayout(toTree(seed)))
  })
})
