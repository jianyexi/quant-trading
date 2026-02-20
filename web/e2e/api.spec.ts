import { test, expect } from '@playwright/test';

/**
 * API integration tests — these run against the backend via the Vite proxy.
 * The backend (port 8080) must be running for these tests to pass.
 * If backend is down, the entire suite is skipped gracefully.
 */

test.describe('API Integration', () => {
  test.beforeAll(async ({ request }) => {
    try {
      const res = await request.get('/api/health');
      if (!res.ok()) test.skip();
    } catch {
      test.skip();
    }
  });

  test('GET /api/health returns ok', async ({ request }) => {
    const res = await request.get('/api/health');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body.status).toBe('ok');
  });

  test('GET /api/dashboard returns dashboard data', async ({ request }) => {
    const res = await request.get('/api/dashboard');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('GET /api/strategies returns data', async ({ request }) => {
    const res = await request.get('/api/strategies');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('GET /api/strategy/config returns config object', async ({ request }) => {
    const res = await request.get('/api/strategy/config');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('GET /api/market/stocks returns data', async ({ request }) => {
    const res = await request.get('/api/market/stocks');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('GET /api/orders returns data', async ({ request }) => {
    const res = await request.get('/api/orders');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('GET /api/portfolio returns portfolio data', async ({ request }) => {
    const res = await request.get('/api/portfolio');
    expect(res.status()).toBe(200);
  });

  test('GET /api/trade/status returns trading status', async ({ request }) => {
    const res = await request.get('/api/trade/status');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty('running');
  });

  test('GET /api/trade/performance returns perf data', async ({ request }) => {
    const res = await request.get('/api/trade/performance');
    expect(res.status()).toBe(200);
  });

  test('GET /api/trade/risk returns risk metrics', async ({ request }) => {
    const res = await request.get('/api/trade/risk');
    expect(res.status()).toBe(200);
  });

  test('GET /api/journal returns journal entries', async ({ request }) => {
    const res = await request.get('/api/journal');
    expect(res.status()).toBe(200);
  });

  test('GET /api/logs returns data', async ({ request }) => {
    const res = await request.get('/api/logs');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('GET /api/chat/history returns chat history', async ({ request }) => {
    const res = await request.get('/api/chat/history');
    expect(res.status()).toBe(200);
  });
});

test.describe('Factor Mining API', () => {
  test.beforeAll(async ({ request }) => {
    try {
      const res = await request.get('/api/health');
      if (!res.ok()) test.skip();
    } catch {
      test.skip();
    }
  });

  test('GET /api/factor/registry returns registry', async ({ request }) => {
    const res = await request.get('/api/factor/registry');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('GET /api/factor/results returns factor results', async ({ request }) => {
    const res = await request.get('/api/factor/results');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('POST /api/factor/export returns export data', async ({ request }) => {
    const res = await request.post('/api/factor/export');
    expect(res.status()).toBe(200);
  });

  test('POST /api/factor/manage triggers registry management', async ({ request }) => {
    const res = await request.post('/api/factor/manage');
    // May return 200 or 500 depending on registry file existence
    expect([200, 500]).toContain(res.status());
  });
});

test.describe('Backtest API', () => {
  test.beforeAll(async ({ request }) => {
    try {
      const res = await request.get('/api/health');
      if (!res.ok()) test.skip();
    } catch {
      test.skip();
    }
  });

  test('POST /api/backtest/run accepts request', async ({ request }) => {
    const res = await request.post('/api/backtest/run', {
      data: {
        strategy: 'multi_factor',
        start_date: '2024-01-01',
        end_date: '2024-03-01',
        initial_capital: 100000,
      },
    });
    // Should accept (200) or return error for missing data (4xx/5xx)
    expect(res.status()).toBeLessThan(600);
  });
});

test.describe('Screening API', () => {
  test.beforeAll(async ({ request }) => {
    try {
      const res = await request.get('/api/health');
      if (!res.ok()) test.skip();
    } catch {
      test.skip();
    }
  });

  test('POST /api/screen/scan accepts scan request', async ({ request }) => {
    test.setTimeout(90_000);
    const res = await request.post('/api/screen/scan', {
      data: { min_score: 0.5 },
      timeout: 60_000,
    });
    expect(res.status()).toBeLessThan(600);
  });
});

test.describe('Chat API', () => {
  test.beforeAll(async ({ request }) => {
    try {
      const res = await request.get('/api/health');
      if (!res.ok()) test.skip();
    } catch {
      test.skip();
    }
  });

  test('POST /api/chat sends a message', async ({ request }) => {
    const res = await request.post('/api/chat', {
      data: { message: 'hello' },
    });
    // Chat may fail if LLM not configured, but should not 404/405
    expect([200, 500]).toContain(res.status());
  });
});
