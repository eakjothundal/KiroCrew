/**
 * Screenshot harness for the effort picker in the Switch All Sessions panel.
 *
 * Runs the REAL built SPA (website/dist) behind the shared `serveDist` server
 * and answers every /api/** call from fixtures through the shared
 * `stubDashboardApi` helper. No gateway, no dashboard auth, no kiro-cli.
 *
 * Scene-specific stubs: `/api/models` (a roster with one model that takes
 * effort and Auto, which does not), `/api/config/kirocrew` (a configured
 * default effort, so the Default row names it), and `POST /api/chat/slots/model`
 * (records the body so the harness can assert what the panel sent).
 *
 * Scenes, each in dark and light:
 *   00 the panel just opened: no model picked, the picker disabled with its
 *      reason beneath it;
 *   01 an effort-capable model picked, the effort list open;
 *   02 High picked, the scale cue beneath the picker, the Switch count
 *      including the effort-only difference and, above the buttons, its split
 *      into the session that resets and the one that keeps its conversation;
 *   03 Auto picked: the picker disabled with its reason beneath it.
 *
 * Usage: node scripts/capture-bulk-model-effort.mjs [outDir]
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { join } from 'node:path'

import { json } from './lib/boot-api.mjs'
import { serveDist } from './lib/serve-dist.mjs'
import { KIROCREW_CONFIG_FIXTURE, stubDashboardApi } from './lib/stub-dashboard-api.mjs'

const OUT = process.argv[2] || '/tmp/bulk-model-effort-shots'
mkdirSync(OUT, { recursive: true })

const SIDEBAR_WIDTH = 300

// Two sessions already on the effort-capable model at different levels, and
// one on another model: picking claude-opus-4.8 at High switches two of them
// (the sonnet one for its model, the Low one for its effort).
const SLOTS = [
  { key: 'chat-1', title: 'Refactor the auth middleware', messages: 12, running: false, agent: 'kirocrew', mode: '', model: 'claude-opus-4.8', reasoning_effort: 'low' },
  { key: 'chat-2', title: 'Weekly report draft', messages: 4, running: false, agent: 'kirocrew', mode: '', model: 'claude-opus-4.8', reasoning_effort: 'high' },
  { key: 'chat-3', title: 'Debug the flaky e2e suite', messages: 27, running: false, agent: 'kirocrew', mode: '', model: 'claude-sonnet-4.7' },
]

const MODELS = [
  { model_name: 'auto', description: 'Models chosen by task' },
  { model_name: 'claude-opus-4.8', description: 'Claude Opus 4.8' },
  { model_name: 'claude-sonnet-4.7', description: 'Claude Sonnet 4.7' },
]

const CONFIG = {
  ...KIROCREW_CONFIG_FIXTURE,
  agent: { ...KIROCREW_CONFIG_FIXTURE.agent, reasoning_effort: 'medium' },
}

const sent = []

const { srv, base } = await serveDist()
const browser = await chromium.launch()

for (const theme of ['dark', 'light']) {
  const context = await browser.newContext({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2 })
  const page = await context.newPage()
  const extra = async (path, route) => {
    if (path === '/api/models') { await json(route, MODELS); return true }
    if (path === '/api/config/kirocrew') { await json(route, CONFIG); return true }
    if (path === '/api/chat/slots/model' && route.request().method() === 'POST') {
      sent.push(route.request().postDataJSON())
      await json(route, { ok: true, model: 'claude-opus-4.8', switched: ['chat-1', 'chat-3'], skipped_running: [], unchanged: ['chat-2'], failed: [], effort_not_applied: [] })
      return true
    }
    return false
  }
  await stubDashboardApi(page, {
    slots: SLOTS,
    theme,
    extra,
    localStorageEntries: { 'mc-active-slot': 'chat-1', 'mc-lang': 'en', 'mc-sidebar-width': String(SIDEBAR_WIDTH) },
  })
  await page.goto(base + '/chat', { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(2500)

  const panel = () => page.locator('div.animate-rise').filter({ hasText: 'Switch All Sessions' }).first()
  const effort = () => panel().getByRole('combobox', { name: 'Effort' })
  const switchBtn = () => panel().getByRole('button', { name: /^Switch \d+ session/ })
  const shot = async (name, { list = false } = {}) => {
    await page.waitForTimeout(300)
    const out = join(OUT, `${theme}-${name}.png`)
    if (list) {
      // The open list is portalled outside the panel, so frame the region that
      // holds both instead of the panel element alone.
      const p = await panel().boundingBox()
      const l = await page.getByRole('listbox').last().boundingBox()
      if (!p || !l) throw new Error('panel or effort list has no bounding box')
      const x = Math.min(p.x, l.x) - 8
      const y = Math.min(p.y, l.y) - 8
      const w = Math.max(p.x + p.width, l.x + l.width) - x + 8
      const h = Math.max(p.y + p.height, l.y + l.height) - y + 8
      await page.screenshot({ path: out, clip: { x, y, width: w, height: h } })
    } else {
      await panel().screenshot({ path: out })
    }
    console.log('wrote', out)
  }

  await page.getByLabel('More options').first().click()
  await page.getByRole('menuitem', { name: /Switch all to model/ }).click()
  await panel().waitFor({ state: 'visible', timeout: 5000 })
  if (!(await effort().isDisabled())) throw new Error('effort picker should be disabled before a model is picked')
  await panel().getByTestId('bulk-effort-unsupported').waitFor({ state: 'visible', timeout: 5000 })
  await shot('00-no-model')

  // 01: pick the effort-capable model, open the effort list.
  await panel().getByRole('option', { name: /claude-opus-4\.8/ }).first().click()
  if (await effort().isDisabled()) throw new Error('effort picker still disabled after picking claude-opus-4.8')
  await effort().click()
  await page.getByRole('option', { name: 'Default from Settings · Medium' }).waitFor({ state: 'visible', timeout: 5000 })
  await shot('01-effort-list-open', { list: true })

  // 02: High picked; two sessions differ (one by model, one by effort).
  await page.getByRole('option', { name: 'High', exact: true }).click()
  const label = (await switchBtn().textContent())?.trim()
  if (label !== 'Switch 2 sessions') throw new Error(`expected "Switch 2 sessions", saw ${JSON.stringify(label)}`)
  await panel().getByTestId('bulk-effort-scale').waitFor({ state: 'visible', timeout: 5000 })
  const split = (await panel().getByTestId('bulk-switch-split').textContent())?.trim()
  if (split !== '1 resets its conversation · 1 keeps its conversation') {
    throw new Error(`expected the reset/keep split beside the button, saw ${JSON.stringify(split)}`)
  }
  await shot('02-high-picked')

  // 03: Auto takes no effort: disabled, with the reason beneath.
  await panel().getByRole('option', { name: /auto/i }).first().click()
  if (!(await effort().isDisabled())) throw new Error('effort picker should be disabled for auto')
  await panel().getByTestId('bulk-effort-unsupported').waitFor({ state: 'visible', timeout: 5000 })
  await shot('03-auto-disabled')

  // Back to the capable model at High, then Switch: the body carries the level.
  await panel().getByRole('option', { name: /claude-opus-4\.8/ }).first().click()
  await effort().click()
  await page.getByRole('option', { name: 'High', exact: true }).click()
  await switchBtn().click()
  await panel().waitFor({ state: 'hidden', timeout: 5000 })
  const body = sent.at(-1)
  if (body?.model !== 'claude-opus-4.8' || body?.reasoning_effort !== 'high') {
    throw new Error(`unexpected request body ${JSON.stringify(body)}`)
  }
  await context.close()
}

await browser.close()
srv.close()
console.log('request bodies:', JSON.stringify(sent))
