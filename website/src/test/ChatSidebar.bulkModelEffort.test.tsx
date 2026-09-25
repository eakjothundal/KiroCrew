/**
 * Switch All Sessions panel: the effort picker that rides along with the model.
 *
 * The pick is sent only when the target model can use effort ("Keep" and a
 * non-effort model both omit it, so each session keeps its own), the Switch
 * count includes sessions whose only difference is their effort and is split
 * beside the button into the ones that reset and the ones that keep their
 * conversation (mid-turn and remote-bound effort-only sessions excluded, as
 * the backend skips them), the Default row names where its default comes
 * from, and a partial outcome keeps the panel open with a status line.
 * Harness shared with ChatSidebar.bulkModelRosterError.test.tsx.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Provider } from 'react-redux'
import { MemoryRouter } from 'react-router-dom'
import { createTestStore } from './helpers'
import { ThemeProvider } from '../hooks/useTheme'
import type { RootState } from '../store'

// Render framer-motion elements as plain DOM because jsdom cannot run projection.
vi.mock('framer-motion', async () => {
  const React = await import('react')
  const FRAMER_PROPS = new Set([
    'layout', 'layoutId', 'layoutScroll', 'initial', 'animate', 'exit',
    'transition', 'variants', 'whileHover', 'whileTap', 'whileInView',
    'drag', 'dragConstraints', 'dragElastic', 'onAnimationComplete',
  ])
  const make = (tag: string) =>
    React.forwardRef((props: Record<string, unknown>, ref: React.Ref<unknown>) => {
      const clean: Record<string, unknown> = {}
      for (const k of Object.keys(props)) {
        if (k === 'children') continue
        if (k === 'layoutId') { clean['data-layout-id'] = props[k]; continue }
        if (FRAMER_PROPS.has(k)) continue
        clean[k] = props[k]
      }
      return React.createElement(tag, { ...clean, ref }, props.children as React.ReactNode)
    })
  const motion = new Proxy({}, { get: (_t, tag: string) => make(tag) })
  return {
    motion,
    AnimatePresence: ({ children }: { children?: React.ReactNode }) => React.createElement(React.Fragment, null, children),
    LayoutGroup: ({ children }: { children?: React.ReactNode }) => React.createElement(React.Fragment, null, children),
  }
})

vi.mock('../components/ProjectPicker', () => ({ default: () => null }))

vi.mock('../pages/chat/ChatSettings', () => ({
  loadChatConfig: () => ({ tagColumnsEnabled: false, confirmCloseSession: false, defaultAutopilot: false }),
  saveChatConfig: vi.fn(),
}))

const mocks = vi.hoisted(() => ({
  models: vi.fn(),
  chatSlotsModel: vi.fn(),
  chatFolders: vi.fn(),
  chatTags: vi.fn(),
  tagColumns: vi.fn(),
  kirocrewConfig: vi.fn(),
  sessions: vi.fn(),
}))
vi.mock('../api/client', () => ({
  SEARCH_MIN_CHARS: 2,
  api: new Proxy(mocks as unknown as Record<string, unknown>, {
    get: (target, prop: string) => (prop in target ? target[prop] : vi.fn().mockResolvedValue([])),
  }),
}))

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((q: string) => ({
    matches: false, media: q, onchange: null,
    addListener: vi.fn(), removeListener: vi.fn(),
    addEventListener: vi.fn(), removeEventListener: vi.fn(), dispatchEvent: vi.fn(),
  })),
})

import ChatSidebar from '../pages/ChatSidebar'

type TestSlot = { key: string; title: string; running: boolean; messages: number; model?: string; reasoning_effort?: string; executor?: 'local' | 'remote' }

const LIVE_ROSTER = [
  { model_name: 'auto', description: 'Default' },
  { model_name: 'opus-4.8', description: 'Opus' },
  { model_name: 'sonnet-4.7', description: 'Sonnet' },
]

function renderSidebar(slots: TestSlot[]) {
  const defaults = createTestStore().getState()
  const store = createTestStore({
    dashboard: {
      ...defaults.dashboard,
      status: {}, connected: true, slots, approvalMode: 'normal',
      channelTrusted: false, refreshTrigger: 0, unreadSlots: [], updateProgress: null,
      slotsLoaded: true,
      subagentRunning: {}, subagentDetails: {}, subagentText: {},
      sessionDefaultColor: null, sessionColorsMode: 'tint', sessionColorsPalette: 'horizon', sessionColorsIntensity: 'clear',
    } as unknown as RootState['dashboard'],
    chat: {
      ...defaults.chat,
      activeSlot: null, slotStatusDetail: {}, subagents: {}, slotActivity: {},
      automations: {}, workflowRuns: {}, subagentQueued: {}, slotHistory: [],
      revealRequest: null, revealNonce: 0,
    } as unknown as RootState['chat'],
  })
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  qc.setQueryData(['chat-folders'], [])
  qc.setQueryData(['tag-columns'], [])
  render(
    <QueryClientProvider client={qc}>
      <Provider store={store}>
        <ThemeProvider>
          <MemoryRouter>
            <ChatSidebar
              slots={slots} activeSlot={null} unreadSlots={[]}
              history={[]} historyHasMore={false}
              defaultAgent="" installedAgents={[{ name: 'builder', source: 'builtin' }]}
            />
          </MemoryRouter>
        </ThemeProvider>
      </Provider>
    </QueryClientProvider>,
  )
}

async function openSwitchAllPanel() {
  fireEvent.keyDown(screen.getAllByLabelText('More options')[0], { key: 'Enter' })
  fireEvent.click(await screen.findByText('Switch all to model…'))
  expect(screen.getByText('Switch All Sessions')).toBeTruthy()
}

const effortTrigger = () => screen.getByRole('combobox', { name: 'Effort' })
const switchBtn = () => screen.getByRole('button', { name: /^Switch \d+ sessions?$/ })

/** Pick a row in the model listbox by the model id it sends. */
async function pickModel(id: string) {
  const listbox = screen.getByRole('listbox', { name: 'Model list' })
  await waitFor(() => expect(within(listbox).getAllByRole('option').length).toBe(LIVE_ROSTER.length))
  const row = within(listbox)
    .getAllByRole('option')
    .find(o => o.querySelector('[data-model-id]')?.getAttribute('data-model-id') === id)
  expect(row).toBeTruthy()
  fireEvent.click(row!)
}

async function pickEffort(label: string) {
  fireEvent.click(effortTrigger())
  fireEvent.click(await screen.findByRole('option', { name: label }))
}

const IDLE_ON_OPUS: TestSlot[] = [
  { key: 'k-a', title: 'Idle A', running: false, messages: 1, model: 'opus-4.8', reasoning_effort: 'low' },
  { key: 'k-b', title: 'Idle B', running: false, messages: 1, model: 'opus-4.8', reasoning_effort: 'high' },
]
const IDLE_ON_SONNET: TestSlot[] = [
  { key: 'k-a', title: 'Idle A', running: false, messages: 1, model: 'sonnet-4.7' },
]

beforeEach(() => {
  localStorage.clear()
  mocks.models.mockResolvedValue(LIVE_ROSTER)
  mocks.chatSlotsModel.mockResolvedValue({ ok: true, switched: [], skipped_running: [], unchanged: [], failed: [], effort_not_applied: [] })
  mocks.chatFolders.mockResolvedValue([])
  mocks.chatTags.mockResolvedValue([])
  mocks.tagColumns.mockResolvedValue([])
  mocks.sessions.mockResolvedValue({ sessions: [], has_more: false })
  mocks.kirocrewConfig.mockResolvedValue({ agent: { reasoning_effort: 'medium' } })
})
afterEach(() => {
  vi.clearAllMocks()
})

describe('ChatSidebar — Switch All Sessions effort', () => {
  it('is disabled until a model that takes effort is picked', async () => {
    renderSidebar(IDLE_ON_OPUS)
    await openSwitchAllPanel()
    expect(effortTrigger()).toBeDisabled()
    const hint = screen.getByTestId('bulk-effort-unsupported')
    expect(hint).toHaveTextContent('Pick a model to choose its effort level.')
    expect(effortTrigger().getAttribute('aria-describedby')).toBe(hint.id)
    await pickModel('opus-4.8')
    expect(effortTrigger()).not.toBeDisabled()
    expect(effortTrigger()).toHaveTextContent('Keep current effort')
  })

  it('sends the picked level with the model', async () => {
    renderSidebar(IDLE_ON_SONNET)
    await openSwitchAllPanel()
    await pickModel('opus-4.8')
    await pickEffort('Max')
    fireEvent.click(switchBtn())
    await waitFor(() => expect(mocks.chatSlotsModel).toHaveBeenCalledWith('opus-4.8', true, 'max'))
  })

  it('omits the level under Keep, so each session keeps its own', async () => {
    renderSidebar(IDLE_ON_SONNET)
    await openSwitchAllPanel()
    await pickModel('opus-4.8')
    fireEvent.click(switchBtn())
    await waitFor(() => expect(mocks.chatSlotsModel).toHaveBeenCalledTimes(1))
    expect(mocks.chatSlotsModel.mock.calls[0][2]).toBeUndefined()
  })

  it('counts sessions whose only difference is their effort, as ones that keep their conversation', async () => {
    // Both sessions already run opus-4.8, so under Keep there is nothing to do;
    // picking High makes the one at Low a switch, and the one at High stays put.
    renderSidebar(IDLE_ON_OPUS)
    await openSwitchAllPanel()
    await pickModel('opus-4.8')
    expect(switchBtn()).toBeDisabled()
    expect(screen.queryByTestId('bulk-switch-split')).toBeNull()  // nothing to split at 0
    await pickEffort('High')
    expect(switchBtn()).toHaveTextContent('Switch 1 session')
    expect(switchBtn()).not.toBeDisabled()
    expect(screen.getByTestId('bulk-switch-split')).toHaveTextContent('1 keeps its conversation')
    expect(screen.getByTestId('bulk-switch-split')).not.toHaveTextContent('resets')
    fireEvent.click(switchBtn())
    await waitFor(() => expect(mocks.chatSlotsModel).toHaveBeenCalledWith('opus-4.8', true, 'high'))
  })

  it('splits the button count into sessions that reset and sessions that keep their conversation', async () => {
    // Two on sonnet change model (they reset); one on opus at Low changes only
    // its effort (it keeps); one on opus at High is unchanged. The button says
    // 3, the line beside it says which are which, and the reset part carries
    // the same warning colour as the description's "resets its conversation".
    renderSidebar([
      { key: 'k-a', title: 'A', running: false, messages: 1, model: 'sonnet-4.7' },
      { key: 'k-b', title: 'B', running: false, messages: 1, model: 'sonnet-4.7', reasoning_effort: 'high' },
      { key: 'k-c', title: 'C', running: false, messages: 1, model: 'opus-4.8', reasoning_effort: 'low' },
      { key: 'k-d', title: 'D', running: false, messages: 1, model: 'opus-4.8', reasoning_effort: 'high' },
    ])
    await openSwitchAllPanel()
    await pickModel('opus-4.8')
    // Under Keep only the model changes count, and all of them reset.
    expect(switchBtn()).toHaveTextContent('Switch 2 sessions')
    expect(screen.getByTestId('bulk-switch-split')).toHaveTextContent('2 reset their conversations')
    expect(screen.getByTestId('bulk-switch-split')).not.toHaveTextContent('keep')
    await pickEffort('High')
    expect(switchBtn()).toHaveTextContent('Switch 3 sessions')
    const split = screen.getByTestId('bulk-switch-split')
    expect(split).toHaveTextContent('2 reset their conversations · 1 keeps its conversation')
    expect(screen.getByText('2 reset their conversations')).toHaveClass('text-danger')
    expect(screen.getByText('1 keeps its conversation')).not.toHaveClass('text-danger')
  })

  it('does not count a mid-turn effort-only session, whatever the running checkbox says', async () => {
    // The backend leaves an effort-only session alone while it is mid-turn (no
    // turn to tear down, so it waits for a retry) even under skip_running=false,
    // while a model change on a running session IS forced through by that box.
    renderSidebar([
      { key: 'k-a', title: 'Busy effort-only', running: true, messages: 1, model: 'opus-4.8', reasoning_effort: 'low' },
      { key: 'k-b', title: 'Busy model change', running: true, messages: 1, model: 'sonnet-4.7' },
      { key: 'k-c', title: 'Idle effort-only', running: false, messages: 1, model: 'opus-4.8', reasoning_effort: 'low' },
    ])
    await openSwitchAllPanel()
    await pickModel('opus-4.8')
    await pickEffort('High')
    // Skipping running sessions (the default): only the idle effort-only one.
    expect(switchBtn()).toHaveTextContent('Switch 1 session')
    expect(screen.getByTestId('bulk-switch-split')).toHaveTextContent('1 keeps its conversation')
    // Untick "skip running": the running model change joins; the running
    // effort-only session still does not.
    fireEvent.click(screen.getByRole('checkbox'))
    expect(switchBtn()).toHaveTextContent('Switch 2 sessions')
    expect(screen.getByTestId('bulk-switch-split')).toHaveTextContent('1 resets its conversation · 1 keeps its conversation')
  })

  it('does not count a remote-bound effort-only session, but still counts its model change', async () => {
    // A remote-bound session's effort is its peer's to run, so the backend
    // never applies an effort-only pick to it; its model still moves.
    renderSidebar([
      { key: 'k-a', title: 'Remote effort-only', running: false, messages: 1, model: 'opus-4.8', reasoning_effort: 'low', executor: 'remote' },
      { key: 'k-b', title: 'Remote model change', running: false, messages: 1, model: 'sonnet-4.7', reasoning_effort: 'low', executor: 'remote' },
      { key: 'k-c', title: 'Local effort-only', running: false, messages: 1, model: 'opus-4.8', reasoning_effort: 'low', executor: 'local' },
    ])
    await openSwitchAllPanel()
    await pickModel('opus-4.8')
    await pickEffort('High')
    expect(switchBtn()).toHaveTextContent('Switch 2 sessions')
    expect(screen.getByTestId('bulk-switch-split')).toHaveTextContent('1 resets its conversation · 1 keeps its conversation')
  })

  it('keeps the panel open with a status line when sessions were skipped or kept their effort', async () => {
    // A 200 with skipped or effort-not-applied entries and nothing failed is a
    // partial outcome, not a success: the panel stays open, its picks intact,
    // and says what did not happen and whether a retry can help.
    mocks.chatSlotsModel.mockResolvedValue({
      ok: true, switched: ['k-a'], skipped_running: ['k-b', 'k-c'], unchanged: [], failed: [], effort_not_applied: ['k-a'],
    })
    renderSidebar(IDLE_ON_SONNET)
    await openSwitchAllPanel()
    await pickModel('opus-4.8')
    await pickEffort('High')
    fireEvent.click(switchBtn())
    const notice = await screen.findByTestId('bulk-model-notice')
    expect(notice).toHaveTextContent(
      '2 sessions were busy and kept their settings. Try again when their replies finish. 1 remote session keeps its own effort level.',
    )
    expect(notice).toHaveAttribute('role', 'status')
    expect(screen.getByText('Switch All Sessions')).toBeTruthy()  // still open
    expect(effortTrigger()).toHaveTextContent('High')              // pick kept for the retry
    expect(screen.queryByTestId('bulk-model-error')).toBeNull()   // not an error
  })

  it('says only what applies: a lone busy skip, singular', async () => {
    mocks.chatSlotsModel.mockResolvedValue({
      ok: true, switched: [], skipped_running: ['k-a'], unchanged: [], failed: [], effort_not_applied: [],
    })
    renderSidebar(IDLE_ON_SONNET)
    await openSwitchAllPanel()
    await pickModel('opus-4.8')
    fireEvent.click(switchBtn())
    const notice = await screen.findByTestId('bulk-model-notice')
    expect(notice).toHaveTextContent('1 session was busy and kept its settings. Try again when its reply finishes.')
    expect(notice).not.toHaveTextContent('remote')
  })

  it('closes the panel on a clean success and clears the picks', async () => {
    renderSidebar(IDLE_ON_SONNET)
    await openSwitchAllPanel()
    await pickModel('opus-4.8')
    await pickEffort('High')
    fireEvent.click(switchBtn())
    await waitFor(() => expect(screen.queryByText('Switch All Sessions')).toBeNull())
    expect(screen.queryByTestId('bulk-model-notice')).toBeNull()
  })

  it('names where the Default row\'s default comes from, apart from the model list\'s own Default', async () => {
    renderSidebar(IDLE_ON_OPUS)
    await openSwitchAllPanel()
    await pickModel('opus-4.8')
    fireEvent.click(effortTrigger())
    expect(await screen.findByRole('option', { name: 'Default from Settings · Medium' })).toBeTruthy()
    expect(screen.queryByRole('option', { name: 'Default · Medium' })).toBeNull()
  })

  it('names the model default on the Default row when no default is configured', async () => {
    mocks.kirocrewConfig.mockResolvedValue({ agent: { reasoning_effort: '' } })
    renderSidebar(IDLE_ON_OPUS)
    await openSwitchAllPanel()
    await pickModel('opus-4.8')
    fireEvent.click(effortTrigger())
    expect(await screen.findByRole('option', { name: 'Model default' })).toBeTruthy()
  })

  it('says which change resets a conversation and which keeps it', async () => {
    renderSidebar(IDLE_ON_OPUS)
    await openSwitchAllPanel()
    const intro = screen.getByText(/Pick a model for every session, and an effort level/)
    expect(intro).toHaveTextContent(
      "Pick a model for every session, and an effort level if that model uses one. Changing a session's model resets its conversation. Changing only its effort keeps the conversation.",
    )
    // The red emphasis stays on the reset wording, and only on it.
    expect(screen.getByText('resets its conversation')).toHaveClass('text-danger')
  })

  it('cues the direction of the scale while the control is live', async () => {
    renderSidebar(IDLE_ON_OPUS)
    await openSwitchAllPanel()
    expect(screen.queryByTestId('bulk-effort-scale')).toBeNull()  // nothing before a model is picked
    await pickModel('opus-4.8')
    const hint = screen.getByTestId('bulk-effort-scale')
    expect(hint).toHaveTextContent('Lower levels are faster, higher levels are smarter.')
    expect(effortTrigger().getAttribute('aria-describedby')).toBe(hint.id)
  })

  it('drops the pick and says why when the model does not take effort', async () => {
    renderSidebar(IDLE_ON_SONNET)
    await openSwitchAllPanel()
    await pickModel('opus-4.8')
    await pickEffort('High')
    await pickModel('auto')
    expect(effortTrigger()).toBeDisabled()
    const hint = screen.getByTestId('bulk-effort-unsupported')
    expect(hint).toHaveTextContent("This model doesn't use effort levels, so each session keeps its own.")
    expect(screen.queryByTestId('bulk-effort-scale')).toBeNull()  // one hint at a time
    expect(effortTrigger().getAttribute('aria-describedby')).toBe(hint.id)
    fireEvent.click(switchBtn())
    await waitFor(() => expect(mocks.chatSlotsModel).toHaveBeenCalledTimes(1))
    expect(mocks.chatSlotsModel.mock.calls[0][0]).toBe('auto')
    expect(mocks.chatSlotsModel.mock.calls[0][2]).toBeUndefined()
  })
})
