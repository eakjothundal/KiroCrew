import React from 'react'
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ChatInput from '../components/ChatInput'
import { ComposerVoiceSliceOverride } from '../chat-core/composer/Composer'
import { parseOptions } from '../app-sdk/protocol'
import { appendFollowUpOption } from '../lib/followUpToggle'
import { SlotProvider } from '../providers/SlotContext'
import { TerminalHostContext, terminalCommand } from '../hooks/useTerminalCommand'
import { KILLING_ESCAPE_MS } from '../hooks/useStopEscapeHatch'
import { parseDirTokens } from '../utils/fileTokens'
import { formatToken } from '../utils/pasteTokens'
import { renderWithProviders as renderWithAppProviders } from './helpers'

const terminal = vi.hoisted(() => ({
  enabled: true,
  addTab: vi.fn<(cwd?: string) => string | null>(),
  send: vi.fn(),
  bringBack: vi.fn(),
  unsubscribe: vi.fn(),
  ready: null as (() => void) | null,
  failure: null as ((code: 'terminal_invalid_cwd') => void) | null,
  callbacks: new Map<string, () => void>(),
}))
vi.mock('../hooks/useBottomTerminal', () => ({
  addTab: (cwd?: string) => terminal.addTab(cwd),
}))
vi.mock('../utils/terminalPopout', () => ({
  bringBack: () => terminal.bringBack(),
}))
vi.mock('../utils/terminalRegistry', () => ({
  isTerminalEnabled: () => terminal.enabled,
  useTerminalEnabled: () => terminal.enabled,
  onTerminalReady: (id: string, callback: () => void, onFailure?: (code: 'terminal_invalid_cwd') => void) => {
    terminal.ready = callback
    terminal.failure = onFailure ?? null
    terminal.callbacks.set(id, callback)
    return terminal.unsubscribe
  },
  sendToTerminalSession: (...args: unknown[]) => terminal.send(...args),
}))

beforeEach(() => {
  localStorage.clear()
  vi.clearAllMocks()
  terminal.enabled = true
  terminal.addTab.mockReset().mockReturnValue('terminal-1')
  terminal.send.mockReturnValue(true)
  terminal.ready = null
  terminal.failure = null
  terminal.callbacks.clear()
})
afterEach(() => { vi.useRealTimers() })

function renderWithProviders(ui: React.ReactElement) {
  const view = renderWithAppProviders(<TerminalHostContext.Provider value="docked">{ui}</TerminalHostContext.Provider>)
  return {
    ...view,
    rerender: (next: React.ReactElement) => view.rerender(
      <TerminalHostContext.Provider value="docked">{next}</TerminalHostContext.Provider>,
    ),
  }
}

function setup(overrides: Partial<React.ComponentProps<typeof ChatInput>> = {}) {
  const props = {
    value: '! pwd',
    onChange: vi.fn(),
    onSend: vi.fn(),
    onSteer: vi.fn(),
    terminalCommands: 'local' as const,
    project: '/work/selected project',
    ...overrides,
  }
  const view = renderWithProviders(<ChatInput {...props} />)
  return { ...view, props }
}

function confirmRun() {
  fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: /^Run(?: anyway)?$/ }))
}

describe('direct terminal commands', () => {
  it('advertises the syntax only while a local terminal is enabled and docked', () => {
    const { props, rerender } = setup({ value: '' })
    const input = screen.getByLabelText('Message input')
    const advertised = /^Terminal: ! command \(space after !\)/
    expect(input).toHaveAttribute('placeholder', expect.stringMatching(advertised))

    for (const target of ['remote', undefined] as const) {
      rerender(<ChatInput {...props} terminalCommands={target} />)
      expect(input).not.toHaveAttribute('placeholder', expect.stringMatching(advertised))
    }
    for (const host of ['unavailable', 'detached'] as const) {
      rerender(<TerminalHostContext.Provider value={host}><ChatInput {...props} /></TerminalHostContext.Provider>)
      expect(screen.getByLabelText('Message input')).not.toHaveAttribute('placeholder', expect.stringMatching(advertised))
    }

    terminal.enabled = false
    rerender(<ChatInput {...props} />)
    expect(screen.getByLabelText('Message input')).not.toHaveAttribute('placeholder', expect.stringMatching(advertised))
    expect(terminal.addTab).not.toHaveBeenCalled()
  })

  it.each([
    [{ placeholder: 'Ask about this selection' }, /Ask about this selection/],
    [{ connected: false }, /offline/i],
    [{ disabled: true }, /stopping/i],
    [{ continuable: true, continueIsRecovery: true, onContinue: vi.fn() }, /interrupted/i],
  ] satisfies [Partial<React.ComponentProps<typeof ChatInput>>, RegExp][])('keeps an existing placeholder ahead of terminal discovery: %j', (overrides, expected) => {
    setup({ value: '', ...overrides })
    expect(screen.getByLabelText('Message input')).toHaveAttribute('placeholder', expect.stringMatching(expected))
  })

  it('keeps recording and transcription guidance ahead of terminal discovery', () => {
    const props = { value: '', onChange: vi.fn(), onSend: vi.fn(), terminalCommands: 'local' as const }
    const { rerender } = renderWithProviders(
      <ComposerVoiceSliceOverride inputProps={{ voiceRecording: true }}><ChatInput {...props} /></ComposerVoiceSliceOverride>,
    )
    expect(screen.getByLabelText('Message input')).toHaveAttribute('placeholder', expect.stringMatching(/recording/i))
    rerender(
      <ComposerVoiceSliceOverride inputProps={{ voiceTranscribing: true }}><ChatInput {...props} /></ComposerVoiceSliceOverride>,
    )
    expect(screen.getByLabelText('Message input')).toHaveAttribute('placeholder', expect.stringMatching(/transcribing/i))
  })

  it.each([
    ['! pwd', 'pwd'],
    ['  ! printf "$HOME\\n" | cat', 'printf "$HOME\\n" | cat'],
    ['!\npwd\nls', 'pwd\nls'],
    ['!', ''],
    ['!!', null],
    ['!important', null],
    ['![image](photo.png)', null],
    ['say ! pwd', null],
  ])('parses %j as %j', (input, expected) => {
    expect(terminalCommand(input)).toBe(expected)
  })

  it.each(['idle', 'busy', 'stopping', 'killing'] as const)('runs without an agent when %s', state => {
    const onStop = vi.fn()
    const { props } = setup({
      isRunning: state !== 'idle',
      canSteer: true,
      stopState: state === 'stopping' ? 'soft_pending' : state === 'killing' ? 'killing' : 'idle',
      onStop,
    })
    const run = screen.getByRole('button', { name: 'Run in terminal' })
    expect(run).toHaveTextContent(/^Run$/)
    fireEvent.click(run)
    expect(screen.getByRole('dialog')).toHaveTextContent('pwd')
    expect(terminal.addTab).not.toHaveBeenCalled()
    expect(terminal.bringBack).not.toHaveBeenCalled()
    expect(terminal.ready).toBeNull()
    confirmRun()
    expect(terminal.addTab).toHaveBeenCalledWith('/work/selected project')
    expect(terminal.send).not.toHaveBeenCalled()
    expect(props.onChange).not.toHaveBeenCalled()
    act(() => terminal.ready?.())
    expect(terminal.send).toHaveBeenCalledExactlyOnceWith('terminal-1', 'pwd')
    expect(props.onChange).toHaveBeenCalledWith('')
    expect(props.onSend).not.toHaveBeenCalled()
    expect(props.onSteer).not.toHaveBeenCalled()
    expect(onStop).not.toHaveBeenCalled()
  })

  it.each([
    ['idle', 'Stop generation'],
    ['soft_pending', 'Force kill session (discards in-progress work and queued messages)'],
    ['killing', 'Force reset session (taking longer than expected)'],
  ] as const)('keeps %s cancellation separate from terminal commands', (stopState, stopLabel) => {
    const onStop = vi.fn()
    const { props, rerender } = setup({ isRunning: true, canSteer: true, onStop })
    if (stopState === 'killing') vi.useFakeTimers()
    rerender(<ChatInput {...props} isRunning={stopState === 'idle'} stopState={stopState} />)
    const run = screen.getByRole('button', { name: 'Run in terminal' })
    expect(run).toBeEnabled()
    if (stopState === 'killing') {
      expect(screen.getByRole('button', { name: 'Killing session' })).toBeDisabled()
      act(() => { vi.advanceTimersByTime(KILLING_ESCAPE_MS) })
    }

    fireEvent.click(screen.getByRole('button', { name: stopLabel }))
    expect(onStop).toHaveBeenCalledOnce()
    expect(screen.getByLabelText('Message input')).toHaveValue('! pwd')
    expect(props.onChange).not.toHaveBeenCalled()
    expect(terminal.addTab).not.toHaveBeenCalled()
    expect(terminal.send).not.toHaveBeenCalled()
    expect(props.onSend).not.toHaveBeenCalled()
    expect(props.onSteer).not.toHaveBeenCalled()

    fireEvent.click(run)
    confirmRun()
    act(() => terminal.ready?.())
    expect(terminal.send).toHaveBeenCalledExactlyOnceWith('terminal-1', 'pwd')
    expect(props.onChange).toHaveBeenCalledExactlyOnceWith('')
    expect(onStop).toHaveBeenCalledOnce()
    expect(props.onSend).not.toHaveBeenCalled()
    expect(props.onSteer).not.toHaveBeenCalled()
  })

  it.each(['soft_pending', 'killing'] as const)('offers only Run when %s has no stop callback', stopState => {
    setup({ isRunning: true, canSteer: true, stopState })
    expect(screen.getByRole('button', { name: 'Run in terminal' })).toBeEnabled()
    expect(screen.queryByTestId('stop-button-pulsing')).not.toBeInTheDocument()
    expect(screen.queryByTestId('stop-button-killing')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Send' })).not.toBeInTheDocument()
  })

  it.each(['click', 'enter'] as const)('refuses a staged folder in a terminal draft submitted by %s', submit => {
    const onSend = vi.fn()
    const onSteer = vi.fn()
    const onChange = vi.fn()
    function Host() {
      const [value, setValue] = React.useState('@docs/ ')
      return (
        <ChatInput
          value={value}
          onChange={next => { onChange(next); setValue(next) }}
          onSend={onSend}
          onSteer={onSteer}
          pendingDirs={parseDirTokens(value).map(token => token.rel)}
          project="/work/proj"
          terminalCommands="local"
        />
      )
    }
    renderWithProviders(<Host />)
    expect(screen.getByTitle('docs/')).toBeVisible()
    const input = screen.getByLabelText('Message input')
    fireEvent.change(input, { target: { value: '! ls @docs/' } })
    onChange.mockClear()
    if (submit === 'click') fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    else fireEvent.keyDown(input, { key: 'Enter' })

    expect(screen.getByText('Remove attachments before running a terminal command.')).toBeVisible()
    expect(input).toHaveValue('! ls @docs/')
    expect(screen.getByTitle('docs/')).toBeVisible()
    expect(onChange).not.toHaveBeenCalled()
    expect(terminal.addTab).not.toHaveBeenCalled()
    expect(terminal.send).not.toHaveBeenCalled()
    expect(onSend).not.toHaveBeenCalled()
    expect(onSteer).not.toHaveBeenCalled()
  })

  it('uses Enter for a command even when the busy default is steer', () => {
    const { props } = setup({ isRunning: true, canSteer: true })
    fireEvent.keyDown(screen.getByLabelText('Message input'), { key: 'Enter' })
    expect(terminal.addTab).not.toHaveBeenCalled()
    confirmRun()
    act(() => terminal.ready?.())
    expect(terminal.send).toHaveBeenCalledWith('terminal-1', 'pwd')
    expect(props.onSteer).not.toHaveBeenCalled()
  })

  it.each([false, true])('requires confirmation for an actual OPTIONS chip selection (Lexical: %s)', async lexicalComposer => {
    const option = '! cat ~/.aws/credentials'
    const { options } = parseOptions(`Suggested next step [OPTIONS: ${option} | Skip]`)
    const onSend = vi.fn()
    const onSteer = vi.fn()
    function Host() {
      const [value, setValue] = React.useState('')
      return (
        <ChatInput
          value={value}
          onChange={setValue}
          onSend={onSend}
          onSteer={onSteer}
          terminalCommands="local"
          lexicalComposer={lexicalComposer}
          followUpOptions={options}
          quickSend={false}
          onFollowUpSelect={text => setValue(previous => appendFollowUpOption(previous, null, text).value)}
        />
      )
    }
    renderWithProviders(<Host />)
    const input = await screen.findByRole('textbox')
    fireEvent.click(screen.getByRole('button', { name: option }))
    const expectDraft = () => {
      if (lexicalComposer) expect(input).toHaveTextContent(option)
      else expect(input).toHaveValue(option)
    }
    await waitFor(expectDraft)
    fireEvent.focus(input)
    fireEvent.keyDown(input, { key: 'Enter' })
    const dialog = await screen.findByRole('dialog')
    expect(dialog).toHaveTextContent('cat ~/.aws/credentials')
    expect(dialog).toHaveTextContent('Reads credential files')
    expect(within(dialog).getByRole('button', { name: 'Run anyway' })).toBeInTheDocument()
    expect(terminal.addTab).not.toHaveBeenCalled()
    expect(terminal.bringBack).not.toHaveBeenCalled()
    expect(terminal.send).not.toHaveBeenCalled()

    fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
    expectDraft()
    expect(terminal.addTab).not.toHaveBeenCalled()
    expect(terminal.send).not.toHaveBeenCalled()
    expect(onSend).not.toHaveBeenCalled()
    expect(onSteer).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    confirmRun()
    act(() => terminal.ready?.())
    expect(terminal.send).toHaveBeenCalledExactlyOnceWith('terminal-1', 'cat ~/.aws/credentials')
    expect(onSend).not.toHaveBeenCalled()
    expect(onSteer).not.toHaveBeenCalled()
  })

  it.each(['!ls', '![photo](photo.png)', '!!', '!important'])('keeps %j on its existing chat send path without a terminal warning', value => {
    const { props } = setup({ value })
    expect(screen.queryByRole('button', { name: 'Run in terminal' })).not.toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.queryByText(/space after !/)).not.toBeInTheDocument()
    fireEvent.keyDown(screen.getByLabelText('Message input'), { key: 'Enter' })
    expect(props.onSend).toHaveBeenCalledOnce()
    expect(terminal.addTab).not.toHaveBeenCalled()
  })

  it('closes chat path completion before sending a literal shell path', () => {
    const onSend = vi.fn()
    function Host() {
      const [value, setValue] = React.useState('')
      return <ChatInput value={value} onChange={setValue} onSend={onSend} project="/work/proj" terminalCommands="local" />
    }
    renderWithProviders(<Host />)
    const input = screen.getByLabelText('Message input')
    fireEvent.change(input, { target: { value: 'read ./' } })
    expect(screen.getByRole('listbox')).toBeInTheDocument()
    fireEvent.change(input, { target: { value: '! cat ./' } })
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument()
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(terminal.addTab).not.toHaveBeenCalled()
    confirmRun()
    act(() => terminal.ready?.())
    expect(terminal.send).toHaveBeenCalledExactlyOnceWith('terminal-1', 'cat ./')
    expect(onSend).not.toHaveBeenCalled()
  })

  it('does not treat the prompt-optimizer shortcut as permission to run a command', () => {
    const { props } = setup({ sendOnEnter: 'ctrl-enter' })
    fireEvent.keyDown(screen.getByLabelText('Message input'), {
      key: 'Enter', ctrlKey: true, shiftKey: true,
    })
    expect(terminal.addTab).not.toHaveBeenCalled()
    expect(props.onSend).not.toHaveBeenCalled()
  })

  it.each([
    [{ terminalCommands: 'remote' as const }, 'Terminal commands are unavailable for remote crew sessions.'],
    [{ pendingFiles: ['/work/image.png'] }, 'Remove attachments before running a terminal command.'],
    [{ value: '!' }, 'Type a command after !.'],
  ] satisfies [Partial<React.ComponentProps<typeof ChatInput>>, string][])('explains and disables a known refusal before submission: %j', (overrides, message) => {
    const { props } = setup(overrides)
    const run = screen.getByRole('button', { name: 'Run in terminal' })
    expect(run).toBeDisabled()
    expect(screen.getByText(message)).toBeVisible()
    fireEvent.click(run)
    fireEvent.keyDown(screen.getByLabelText('Message input'), { key: 'Enter' })
    expect(terminal.addTab).not.toHaveBeenCalled()
    expect(props.onSend).not.toHaveBeenCalled()
    expect(props.onChange).not.toHaveBeenCalled()
  })

  it('keeps an offline command disabled and preserves its draft', () => {
    const { props } = setup({ connected: false })
    expect(screen.getByRole('button', { name: 'Run in terminal' })).toBeDisabled()
    fireEvent.keyDown(screen.getByLabelText('Message input'), { key: 'Enter' })
    expect(terminal.addTab).not.toHaveBeenCalled()
    expect(props.onSend).not.toHaveBeenCalled()
    expect(props.onChange).not.toHaveBeenCalled()
  })

  it('honors the terminal feature flag', () => {
    terminal.enabled = false
    const { props } = setup()
    expect(screen.getByRole('button', { name: 'Run in terminal' })).toBeDisabled()
    expect(screen.getByText('Terminal is disabled for this gateway.')).toBeVisible()
    expect(terminal.addTab).not.toHaveBeenCalled()
    expect(props.onSend).not.toHaveBeenCalled()
  })

  it('asks for the main window when the host has no terminal dock', () => {
    const onSend = vi.fn()
    const onChange = vi.fn()
    renderWithAppProviders(<ChatInput value="! pwd" onChange={onChange} onSend={onSend} terminalCommands="local" />)
    expect(screen.getByRole('button', { name: 'Run in terminal' })).toBeDisabled()
    expect(screen.getByText('Open this session in the main window to run terminal commands.')).toBeVisible()
    expect(terminal.addTab).not.toHaveBeenCalled()
    expect(terminal.bringBack).not.toHaveBeenCalled()
    expect(onSend).not.toHaveBeenCalled()
    expect(onChange).not.toHaveBeenCalled()
  })

  it('returns a detached terminal before creating the command tab', () => {
    vi.useFakeTimers()
    const props = { value: '! pwd', onChange: vi.fn(), onSend: vi.fn(), terminalCommands: 'local' as const }
    const { rerender } = renderWithAppProviders(
      <TerminalHostContext.Provider value="detached"><ChatInput {...props} /></TerminalHostContext.Provider>,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    act(() => { vi.advanceTimersByTime(30_000) })
    expect(screen.getByRole('dialog')).toHaveTextContent('pwd')
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(terminal.addTab).not.toHaveBeenCalled()
    expect(terminal.bringBack).not.toHaveBeenCalled()
    expect(terminal.ready).toBeNull()
    confirmRun()
    expect(terminal.bringBack).toHaveBeenCalledOnce()
    expect(terminal.addTab).not.toHaveBeenCalled()
    rerender(<TerminalHostContext.Provider value="docked"><ChatInput {...props} /></TerminalHostContext.Provider>)
    act(() => terminal.ready?.())
    expect(terminal.send).toHaveBeenCalledExactlyOnceWith('terminal-1', 'pwd')
    expect(props.onChange).toHaveBeenCalledWith('')
  })

  it('does not create a terminal when a timed-out popout returns late', () => {
    vi.useFakeTimers()
    const props = { value: '! pwd', onChange: vi.fn(), onSend: vi.fn(), terminalCommands: 'local' as const }
    const { rerender } = renderWithAppProviders(
      <TerminalHostContext.Provider value="detached"><ChatInput {...props} /></TerminalHostContext.Provider>,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    confirmRun()
    act(() => { vi.advanceTimersByTime(6000) })
    rerender(<TerminalHostContext.Provider value="docked"><ChatInput {...props} /></TerminalHostContext.Provider>)
    expect(terminal.addTab).not.toHaveBeenCalled()
    expect(terminal.send).not.toHaveBeenCalled()
    expect(props.onChange).not.toHaveBeenCalled()
  })

  it('keeps the draft and reports failure at the terminal cap', () => {
    terminal.addTab.mockReturnValue(null)
    const { props } = setup()
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    confirmRun()
    expect(screen.getByText(/Could not send the command/)).toBeVisible()
    expect(props.onChange).not.toHaveBeenCalled()
  })

  it.each(['invalid_cwd', 'failed'] as const)('dismisses %s without changing the draft or retrying the command', failure => {
    if (failure === 'failed') terminal.addTab.mockReturnValueOnce(null)
    const onStop = vi.fn()
    const { props } = setup({ onStop })
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    confirmRun()
    const oldReady = terminal.ready
    if (failure === 'invalid_cwd') act(() => terminal.failure?.('terminal_invalid_cwd'))

    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent(failure === 'invalid_cwd' ? /workspace no longer exists/ : /Could not send the command/)
    fireEvent.click(within(alert).getByRole('button', { name: 'Dismiss' }))
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Message input')).toHaveValue('! pwd')
    expect(props.onChange).not.toHaveBeenCalled()
    act(() => oldReady?.())
    expect(terminal.addTab).toHaveBeenCalledOnce()
    expect(terminal.send).not.toHaveBeenCalled()
    expect(props.onSend).not.toHaveBeenCalled()
    expect(props.onSteer).not.toHaveBeenCalled()
    expect(onStop).not.toHaveBeenCalled()

    // Dismissal does not hide a later failed attempt, or consume a later Run.
    terminal.addTab.mockReturnValueOnce(null)
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    confirmRun()
    expect(screen.getByRole('alert')).toHaveTextContent(/Could not send the command/)
    fireEvent.click(within(screen.getByRole('alert')).getByRole('button', { name: 'Dismiss' }))
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    confirmRun()
    act(() => terminal.ready?.())
    expect(terminal.send).toHaveBeenCalledExactlyOnceWith('terminal-1', 'pwd')
    expect(props.onChange).toHaveBeenCalledExactlyOnceWith('')
    expect(props.onSend).not.toHaveBeenCalled()
    expect(props.onSteer).not.toHaveBeenCalled()
    expect(onStop).not.toHaveBeenCalled()
  })

  it('reports a rejected workspace immediately and retries only on a new submission', () => {
    const { props, rerender } = setup()
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    confirmRun()
    const oldReady = terminal.ready
    act(() => terminal.failure?.('terminal_invalid_cwd'))
    expect(screen.getByText('That workspace no longer exists. Select an existing workspace, then run the command again.')).toBeVisible()
    expect(screen.getByLabelText('Message input')).toHaveValue('! pwd')
    expect(props.onChange).not.toHaveBeenCalled()
    act(() => oldReady?.())
    expect(terminal.send).not.toHaveBeenCalled()

    rerender(<ChatInput {...props} project="/work/recovered" />)
    expect(screen.queryByText(/That workspace no longer exists/)).not.toBeInTheDocument()
    expect(terminal.addTab).toHaveBeenCalledOnce()
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    confirmRun()
    act(() => terminal.ready?.())
    expect(terminal.addTab).toHaveBeenLastCalledWith('/work/recovered')
    expect(terminal.send).toHaveBeenCalledExactlyOnceWith('terminal-1', 'pwd')
    expect(props.onChange).toHaveBeenCalledExactlyOnceWith('')
    expect(props.onSend).not.toHaveBeenCalled()
  })

  it('never delivers a timed-out command after a late ready event', () => {
    vi.useFakeTimers()
    const { props } = setup()
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    confirmRun()
    act(() => { vi.advanceTimersByTime(6000) })
    act(() => terminal.ready?.())
    expect(terminal.unsubscribe).toHaveBeenCalled()
    expect(terminal.send).not.toHaveBeenCalled()
    expect(props.onChange).not.toHaveBeenCalled()
    expect(screen.getByText(/Could not send the command/)).toBeVisible()
  })

  it.each(['dialog', 'handoff'] as const)('does not clear a draft replaced during the %s', phase => {
    const { props, rerender } = setup()
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    if (phase === 'handoff') confirmRun()
    rerender(<ChatInput {...props} value="my next question" />)
    if (phase === 'dialog') {
      expect(screen.getByRole('dialog')).toHaveTextContent('pwd')
      expect(screen.getByRole('dialog')).not.toHaveTextContent('my next question')
      confirmRun()
    }
    act(() => terminal.ready?.())
    expect(terminal.send).toHaveBeenCalledWith('terminal-1', 'pwd')
    expect(props.onChange).not.toHaveBeenCalled()
  })

  it.each(['dialog', 'handoff'] as const)('keeps a replacement paste with the same chip label during the %s', phase => {
    const original = { id: 'original', seq: 1, content: 'pwd\nls\nwhoami', lines: 3 }
    const replacement = { ...original, id: 'replacement', content: 'echo one\necho two\necho three' }
    const { props, rerender } = setup({
      value: `! ${formatToken(original)}`,
      pasteBlocks: [original],
      onPasteBlocksChange: vi.fn(),
    })
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    if (phase === 'handoff') confirmRun()
    expect(formatToken(replacement)).toBe(formatToken(original))
    rerender(<ChatInput {...props} pasteBlocks={[replacement]} />)
    if (phase === 'dialog') {
      expect(screen.getByRole('dialog')).toHaveTextContent('whoami')
      expect(screen.getByRole('dialog')).not.toHaveTextContent('echo three')
      expect(terminal.addTab).not.toHaveBeenCalled()
      confirmRun()
    }
    act(() => terminal.ready?.())
    expect(terminal.send).toHaveBeenCalledExactlyOnceWith('terminal-1', original.content)
    expect(props.onChange).not.toHaveBeenCalled()
    expect(props.onPasteBlocksChange).not.toHaveBeenCalled()
  })

  it.each(['slot', 'project', 'target'] as const)('cancels an open confirmation when its %s changes', async context => {
    const props = { value: '! pwd', onChange: vi.fn(), onSend: vi.fn(), terminalCommands: 'local' as const, project: '/work/a' }
    const { rerender } = renderWithProviders(
      <SlotProvider slotId="first"><ChatInput {...props} /></SlotProvider>,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    expect(screen.getByRole('dialog')).toHaveTextContent('pwd')
    rerender(
      <SlotProvider slotId={context === 'slot' ? 'second' : 'first'}>
        <ChatInput {...props} project={context === 'project' ? '/work/b' : props.project} terminalCommands={context === 'target' ? 'remote' : 'local'} />
      </SlotProvider>,
    )
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
    rerender(<SlotProvider slotId="first"><ChatInput {...props} /></SlotProvider>)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(terminal.addTab).not.toHaveBeenCalled()
    expect(terminal.bringBack).not.toHaveBeenCalled()
    expect(terminal.send).not.toHaveBeenCalled()
    expect(props.onChange).not.toHaveBeenCalled()
    expect(props.onSend).not.toHaveBeenCalled()
  })

  it('does not revive a cancelled request from the dialog exit animation', () => {
    const { props } = setup()
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    const dialog = screen.getByRole('dialog')
    const confirm = within(dialog).getByRole('button', { name: 'Run', exact: true })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }))
    // AnimatePresence keeps this mounted briefly after cancellation.
    fireEvent.click(confirm)
    expect(terminal.addTab).not.toHaveBeenCalled()
    expect(terminal.bringBack).not.toHaveBeenCalled()
    expect(terminal.send).not.toHaveBeenCalled()
    expect(props.onChange).not.toHaveBeenCalled()
    expect(props.onSend).not.toHaveBeenCalled()
  })

  it.each([
    ['dialog', 'flag'], ['dialog', 'attachments'],
    ['handoff', 'flag'], ['handoff', 'attachments'],
  ] as const)('rechecks %s-time %s changes before terminal execution', (phase, changed) => {
    const { props, rerender } = setup()
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    if (phase === 'handoff') confirmRun()
    if (changed === 'flag') terminal.enabled = false
    else rerender(<ChatInput {...props} pendingDirs={['docs/']} />)
    if (phase === 'dialog') {
      confirmRun()
      expect(terminal.addTab).not.toHaveBeenCalled()
      expect(terminal.bringBack).not.toHaveBeenCalled()
    } else act(() => terminal.ready?.())
    expect(terminal.send).not.toHaveBeenCalled()
    expect(props.onChange).not.toHaveBeenCalled()
    expect(props.onSend).not.toHaveBeenCalled()
    expect(props.onSteer).not.toHaveBeenCalled()
  })

  it('cancels the handoff when its session changes', () => {
    const props = { value: '! pwd', onChange: vi.fn(), onSend: vi.fn(), terminalCommands: 'local' as const }
    const { rerender } = renderWithProviders(
      <SlotProvider slotId="first"><ChatInput {...props} /></SlotProvider>,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    confirmRun()
    rerender(<SlotProvider slotId="second"><ChatInput {...props} /></SlotProvider>)
    act(() => terminal.ready?.())
    expect(terminal.unsubscribe).toHaveBeenCalled()
    expect(terminal.send).not.toHaveBeenCalled()
    expect(props.onChange).not.toHaveBeenCalled()
  })

  it('keeps pane workspaces and pending commands independent of another pane changing sessions', async () => {
    const clearA = vi.fn()
    const clearB = vi.fn()
    const onSend = vi.fn()
    terminal.addTab.mockReturnValueOnce('terminal-a').mockReturnValueOnce('terminal-b')
    function Panes({ slotA }: { slotA: string }) {
      return (
        <>
          <div data-testid="pane-a">
            <SlotProvider slotId={slotA}>
              <ChatInput value="! pwd" onChange={clearA} onSend={onSend} project="/work/a" terminalCommands="local" />
            </SlotProvider>
          </div>
          <div data-testid="pane-b">
            <SlotProvider slotId="session-b">
              <ChatInput value="! ls" onChange={clearB} onSend={onSend} project="/work/b" terminalCommands="local" />
            </SlotProvider>
          </div>
        </>
      )
    }
    const { rerender } = renderWithProviders(<Panes slotA="session-a" />)
    fireEvent.click(within(screen.getByTestId('pane-a')).getByRole('button', { name: 'Run in terminal' }))
    confirmRun()
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
    fireEvent.click(within(screen.getByTestId('pane-b')).getByRole('button', { name: 'Run in terminal' }))
    confirmRun()
    expect(terminal.addTab.mock.calls).toEqual([['/work/a'], ['/work/b']])

    rerender(<Panes slotA="session-c" />)
    act(() => {
      terminal.callbacks.get('terminal-a')?.()
      terminal.callbacks.get('terminal-b')?.()
    })
    expect(terminal.send).toHaveBeenCalledExactlyOnceWith('terminal-b', 'ls')
    expect(clearA).not.toHaveBeenCalled()
    expect(clearB).toHaveBeenCalledExactlyOnceWith('')
    expect(onSend).not.toHaveBeenCalled()
  })

  it('expands pasted shell text without rewriting shell variables or file sigils', () => {
    const block = { id: 'paste', seq: 1, content: 'echo "$HOME"\ncat @data', lines: 2 }
    const { props } = setup({
      value: `! ${formatToken(block)}`,
      pasteBlocks: [block],
      onPasteBlocksChange: vi.fn(),
    })
    fireEvent.click(screen.getByRole('button', { name: 'Run in terminal' }))
    expect(screen.getByRole('dialog')).toHaveTextContent('echo "$HOME"')
    expect(screen.getByRole('dialog')).toHaveTextContent('cat @data')
    confirmRun()
    act(() => terminal.ready?.())
    expect(terminal.send).toHaveBeenCalledWith('terminal-1', block.content)
    expect(props.onPasteBlocksChange).toHaveBeenCalledWith([])
  })
})
