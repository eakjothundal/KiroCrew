import { describe, expect, it, vi } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'

const H = vi.hoisted(() => ({
  members: vi.fn(),
  autonudgeList: vi.fn(),
}))

vi.mock('../../api/client', () => ({
  api: {
    members: H.members,
    autonudgeList: H.autonudgeList,
  },
}))

import { MEMBERS_ROSTER_QUERY_KEY } from '../../api/membersQuery'
import { useCrewPerpetual } from './useCrewPerpetual'

describe('useCrewPerpetual', () => {
  it('refreshes the roster after a registry update when floor polling is off', async () => {
    H.members.mockResolvedValue({
      members: [{ name: 'Radar', slug: 'radar', slot_key: 'member-radar', perpetual: 'on' }],
    })
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [] })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries')
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    )

    renderHook(() => useCrewPerpetual('Radar', { poll: false }), { wrapper })

    await waitFor(() => {
      expect(invalidate).toHaveBeenCalledWith({ queryKey: MEMBERS_ROSTER_QUERY_KEY })
    })
    expect(H.autonudgeList).toHaveBeenCalledTimes(1)
  })
})
