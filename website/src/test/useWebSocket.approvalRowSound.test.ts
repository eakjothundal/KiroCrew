/**
 * useWebSocket `chat_message` (role `permission`) -> MC_NOTIFICATION_EVENT for
 * an interactive tool approval.
 *
 * A chat parked on a tool prompt announces itself through the `permission`
 * row the chat runner appends: it is delivered once to every dashboard window
 * as a `chat_message` frame (carrying `resolved` on the batch-rejection
 * re-append), then the future is registered and the slots pushed. No
 * `approval` frame is emitted for it — that frame belongs to the coordinator
 * registry — so the websocket layer must synthesize the `approval` sound from
 * this row, the way it synthesizes `turn` from `chat_done`.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'
import { createElement } from 'react'
import { Provider } from 'react-redux'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createTestStore } from './helpers'
import { store as globalStore } from '../store'
import { api } from '../api/client'
import { useWebSocket } from '../hooks/useWebSocket'
import { MC_NOTIFICATION_EVENT, type McNotificationDetail } from '../hooks/notificationEvent'

vi.mock('../api/client', () => ({
  api: {
    chatSlots: vi.fn().mockResolvedValue([]),
    voiceConfig: vi.fn().mockResolvedValue({ autoSpeak: false }),
    approvals: vi.fn().mockResolvedValue([]),
    notifications: vi.fn().mockResolvedValue({ notifications: [], unread: 0 }),
    chatSlotDetail: vi.fn().mockResolvedValue({ messages: [], running: false, has_more: false, total: 0, queue: [] }),
    autonudgeList: vi.fn().mockResolvedValue({ enabled: true, loops: [] }),
    monitorsList: vi.fn().mockResolvedValue({ enabled: true, monitors: [] }),
    workflowRuns: vi.fn().mockResolvedValue({ runs: [] }),
  },
}))

const sockets: MockWebSocket[] = []
class MockWebSocket {
  static OPEN = 1
  static CONNECTING = 0
  readyState = MockWebSocket.CONNECTING
  onopen: ((ev: Event) => void) | null = null
  onmessage: ((ev: MessageEvent) => void) | null = null
  onclose: ((ev: CloseEvent) => void) | null = null
  onerror: ((ev: Event) => void) | null = null
  send = vi.fn()
  close = vi.fn()
  constructor() { sockets.push(this) }
  open() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.(new Event('open'))
  }
  frame(type: string, data: unknown) {
    act(() => this.onmessage?.(new MessageEvent('message', { data: JSON.stringify({ type, data }) })))
  }
}

const SLOT = 'chat-parked'
const OTHER = 'chat-other'

/** The runner's `permission` row as `_broadcast_chat_message` ships it: the
 *  `cls` JSON kept for older clients, its keys lifted into `meta` with
 *  `request_id` normalized to `approval_id`, plus the row's minted `mid`. */
function permissionRow(slot: string, requestId: string, mid: string, extra: Record<string, unknown> = {}) {
  const cls = { request_id: requestId, tool_input: 'ls', tool_call_id: `tc-${requestId}`, ...extra }
  return {
    slot,
    role: 'permission',
    content: 'shell',
    ts: '2026-01-01T00:00:00Z',
    cls: JSON.stringify(cls),
    meta: { approval_id: requestId, tool_input: 'ls', tool_call_id: `tc-${requestId}`, mid, ...extra },
  }
}

describe('interactive approval sound from the permission row', () => {
  let testStore: ReturnType<typeof createTestStore>
  let queryClient: QueryClient
  let kinds: (string | undefined)[]
  const onSound = (event: Event) => kinds.push((event as CustomEvent<McNotificationDetail>).detail.kind)

  beforeEach(() => {
    vi.clearAllMocks()
    sockets.length = 0
    kinds = []
    testStore = createTestStore()
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    vi.spyOn(globalStore, 'getState').mockImplementation(testStore.getState)
    vi.spyOn(globalStore, 'subscribe').mockImplementation(testStore.subscribe)
    vi.stubGlobal('WebSocket', MockWebSocket)
    window.addEventListener(MC_NOTIFICATION_EVENT, onSound)
  })

  afterEach(async () => {
    await act(async () => cleanup())
    queryClient.clear()
    window.removeEventListener(MC_NOTIFICATION_EVENT, onSound)
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  async function connect() {
    const wrapper = ({ children }: { children: React.ReactNode }) => createElement(
      Provider, { store: testStore },
      createElement(QueryClientProvider, { client: queryClient }, children),
    )
    renderHook(() => useWebSocket(), { wrapper })
    await act(async () => sockets[0].open())
    return sockets[0]
  }

  it('a new unresolved permission row sounds once', async () => {
    const ws = await connect()
    ws.frame('chat_message', permissionRow(SLOT, 'req-1', 'mid-1'))
    expect(kinds).toEqual(['approval'])
  })

  it('a row re-appended as resolved never sounds', async () => {
    const ws = await connect()
    // The batch-rejection path appends a fresh row carrying `resolved`, a
    // prompt nobody will answer.
    ws.frame('chat_message', permissionRow(SLOT, 'req-1', 'mid-1', { resolved: 'rejected' }))
    expect(kinds).toEqual([])
  })

  it('two new rows sound once each', async () => {
    const ws = await connect()
    // Two dispatches: the 300 ms listener throttle decides how many are heard,
    // the transport owes one event per row.
    ws.frame('chat_message', permissionRow(SLOT, 'req-1', 'mid-1'))
    ws.frame('chat_message', permissionRow(OTHER, 'req-1', 'mid-2'))
    expect(kinds).toEqual(['approval', 'approval'])
  })

  it('the slots frame announcing a parked session is not a trigger', async () => {
    // A prompt already parked when the tab opened is history, not news — it
    // reaches this window through the connect snapshot and the transcript
    // rehydration, neither of which is the live row.
    const ws = await connect()
    ws.frame('slots', [{
      key: SLOT, messages: 3, running: true, pending_approval: true,
      pending_approval_info: { tool: 'shell', tool_input: 'ls', tool_kind: 'shell', request_id: 'req-1' },
    }])
    expect(kinds).toEqual([])
  })

  it('a coordinator approval sounds on its own frame only', async () => {
    const ws = await connect()
    ws.frame('approval', { id: 'coord-1', tool: 'shell', source: 'subagent', slot: SLOT })
    expect(kinds).toEqual(['approval'])
    // The card it injects is client-synthesized and never arrives as a
    // server permission row; the runner's own prompt in the same session,
    // even under the same id string, is a distinct row.
    ws.frame('chat_message', permissionRow(SLOT, 'coord-1', 'mid-1'))
    expect(kinds).toEqual(['approval', 'approval'])
  })

  it('stays silent through reconnect catch-up and resumes after it settles', async () => {
    const ws = await connect()
    ws.frame('chat_message', permissionRow(SLOT, 'req-1', 'mid-1'))
    expect(kinds).toEqual(['approval'])
    let finishSync!: () => void
    vi.mocked(api.chatSlots).mockImplementationOnce(() => new Promise(resolve => {
      finishSync = () => resolve([])
    }))
    act(() => ws.open())
    // A row replayed before the catch-up fetch settles is silent.
    ws.frame('chat_message', permissionRow(SLOT, 'req-2', 'mid-2'))
    expect(kinds).toEqual(['approval'])
    await act(async () => finishSync())
    ws.frame('chat_message', permissionRow(SLOT, 'req-3', 'mid-3'))
    expect(kinds).toEqual(['approval', 'approval'])
  })
})
