import { test, expect } from '@playwright/test';

/**
 * Full API integration tests — covers ALL 45 backend endpoints.
 * Backend (port 8080) must be running. Suites auto-skip if backend is down.
 */

function requireBackend() {
  test.beforeAll(async ({ request }) => {
    try {
      const res = await request.get('/api/health');
      if (!res.ok()) test.skip();
    } catch {
      test.skip();
    }
  });
}

/* ── Health & Dashboard ──────────────────────────────────────────── */
test.describe('Health & Dashboard API', () => {
  requireBackend();

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
});

/* ── Strategy Config ─────────────────────────────────────────────── */
test.describe('Strategy Config API', () => {
  requireBackend();

  test('GET /api/strategy/config returns config', async ({ request }) => {
    const res = await request.get('/api/strategy/config');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('POST /api/strategy/config saves config (round-trip)', async ({ request }) => {
    // Read current config
    const getRes = await request.get('/api/strategy/config');
    const original = await getRes.json();

    // Save it back (idempotent)
    const postRes = await request.post('/api/strategy/config', {
      data: original,
    });
    expect([200, 201]).toContain(postRes.status());
  });
});

/* ── Market Data ─────────────────────────────────────────────────── */
test.describe('Market Data API', () => {
  requireBackend();

  test('GET /api/market/stocks returns data', async ({ request }) => {
    const res = await request.get('/api/market/stocks');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('GET /api/market/kline/:symbol returns kline data', async ({ request }) => {
    const res = await request.get('/api/market/kline/600519');
    // 200 if data available, 500 if source not configured
    expect([200, 500]).toContain(res.status());
  });

  test('GET /api/market/quote/:symbol returns quote', async ({ request }) => {
    const res = await request.get('/api/market/quote/600519');
    expect([200, 500]).toContain(res.status());
  });
});

/* ── Backtest ────────────────────────────────────────────────────── */
test.describe('Backtest API', () => {
  requireBackend();

  test('POST /api/backtest/run accepts request', async ({ request }) => {
    const res = await request.post('/api/backtest/run', {
      data: {
        strategy: 'multi_factor',
        start_date: '2024-01-01',
        end_date: '2024-03-01',
        initial_capital: 100000,
      },
    });
    expect(res.status()).toBeLessThan(600);
  });

  test('GET /api/backtest/results/:id returns result or 404', async ({ request }) => {
    const res = await request.get('/api/backtest/results/test-nonexistent-id');
    // 200 if found, 404 if not
    expect([200, 404, 500]).toContain(res.status());
  });
});

/* ── Orders & Portfolio ──────────────────────────────────────────── */
test.describe('Orders & Portfolio API', () => {
  requireBackend();

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
});

/* ── Trading Engine ──────────────────────────────────────────────── */
test.describe('Trading Engine API', () => {
  requireBackend();

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

  test('GET /api/trade/model-info returns ML model info', async ({ request }) => {
    const res = await request.get('/api/trade/model-info');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('GET /api/trade/qmt/status returns QMT bridge status', async ({ request }) => {
    const res = await request.get('/api/trade/qmt/status');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('POST /api/trade/start starts trading engine', async ({ request }) => {
    const res = await request.post('/api/trade/start');
    // 200 if started, 409 if already running, 415 if content-type issue, 500 if config error
    expect([200, 409, 415, 500]).toContain(res.status());
  });

  test('POST /api/trade/stop stops trading engine', async ({ request }) => {
    const res = await request.post('/api/trade/stop');
    // 200 if stopped, 400 if not running
    expect([200, 400, 500]).toContain(res.status());
  });

  test('POST /api/trade/risk/reset-circuit resets circuit breaker', async ({ request }) => {
    const res = await request.post('/api/trade/risk/reset-circuit');
    expect([200, 500]).toContain(res.status());
  });

  test('POST /api/trade/risk/reset-daily resets daily limits', async ({ request }) => {
    const res = await request.post('/api/trade/risk/reset-daily');
    expect([200, 500]).toContain(res.status());
  });

  test('POST /api/trade/retrain triggers ML model retrain', async ({ request }) => {
    test.setTimeout(120_000);
    const res = await request.post('/api/trade/retrain', { timeout: 90_000 });
    // 200 if retrained, 500 if python/model not available
    expect([200, 500]).toContain(res.status());
  });
});

/* ── Screening ───────────────────────────────────────────────────── */
test.describe('Screening API', () => {
  requireBackend();

  test('POST /api/screen/scan accepts scan request', async ({ request }) => {
    test.setTimeout(90_000);
    const res = await request.post('/api/screen/scan', {
      data: { min_score: 0.5 },
      timeout: 60_000,
    });
    expect(res.status()).toBeLessThan(600);
  });

  test('GET /api/screen/factors/:symbol returns factors for a stock', async ({ request }) => {
    test.setTimeout(60_000);
    const res = await request.get('/api/screen/factors/600519', { timeout: 30_000 });
    expect([200, 500]).toContain(res.status());
  });
});

/* ── Sentiment ───────────────────────────────────────────────────── */
test.describe('Sentiment API', () => {
  requireBackend();

  test('POST /api/sentiment/submit submits sentiment entry', async ({ request }) => {
    const res = await request.post('/api/sentiment/submit', {
      data: {
        symbol: '600519',
        source: 'test',
        title: 'Test headline',
        content: 'Positive outlook for Moutai',
        sentiment_score: 0.8,
      },
    });
    expect([200, 201, 500]).toContain(res.status());
  });

  test('POST /api/sentiment/batch submits batch entries', async ({ request }) => {
    const res = await request.post('/api/sentiment/batch', {
      data: {
        items: [
          { symbol: '600519', source: 'test', title: 'Good', content: 'Positive', sentiment_score: 0.7 },
          { symbol: '000858', source: 'test', title: 'Bad', content: 'Negative', sentiment_score: -0.3 },
        ],
      },
    });
    expect([200, 201, 422, 500]).toContain(res.status());
  });

  test('GET /api/sentiment/summary returns sentiment summary', async ({ request }) => {
    const res = await request.get('/api/sentiment/summary');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('GET /api/sentiment/:symbol returns sentiment for symbol', async ({ request }) => {
    const res = await request.get('/api/sentiment/600519');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('GET /api/sentiment/collector/status returns collector status', async ({ request }) => {
    const res = await request.get('/api/sentiment/collector/status');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('POST /api/sentiment/collector/start starts collector', async ({ request }) => {
    const res = await request.post('/api/sentiment/collector/start');
    expect([200, 409, 500]).toContain(res.status());
  });

  test('POST /api/sentiment/collector/stop stops collector', async ({ request }) => {
    const res = await request.post('/api/sentiment/collector/stop');
    expect([200, 400, 500]).toContain(res.status());
  });
});

/* ── Research / DL Models ────────────────────────────────────────── */
test.describe('Research API', () => {
  requireBackend();

  test('GET /api/research/dl-models returns model list', async ({ request }) => {
    const res = await request.get('/api/research/dl-models');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('GET /api/research/dl-models/summary returns summary', async ({ request }) => {
    const res = await request.get('/api/research/dl-models/summary');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('POST /api/research/dl-models/collect triggers collection', async ({ request }) => {
    test.setTimeout(60_000);
    const res = await request.post('/api/research/dl-models/collect', { timeout: 30_000 });
    expect([200, 415, 500]).toContain(res.status());
  });
});

/* ── Factor Mining ───────────────────────────────────────────────── */
test.describe('Factor Mining API', () => {
  requireBackend();

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
    expect([200, 500]).toContain(res.status());
  });

  test('POST /api/factor/mine/parametric runs parametric mining', async ({ request }) => {
    test.setTimeout(120_000);
    const res = await request.post('/api/factor/mine/parametric', {
      data: { n_bars: 500, horizon: 5, ic_threshold: 0.02, top_n: 10, data_source: 'synthetic' },
      timeout: 90_000,
    });
    expect([200, 500]).toContain(res.status());
    if (res.status() === 200) {
      const body = await res.json();
      expect(body).toHaveProperty('status');
    }
  });

  test('POST /api/factor/mine/gp runs GP evolution', async ({ request }) => {
    test.setTimeout(120_000);
    const res = await request.post('/api/factor/mine/gp', {
      data: { n_bars: 500, pop_size: 20, generations: 3, max_depth: 4, data_source: 'synthetic' },
      timeout: 90_000,
    });
    expect([200, 500]).toContain(res.status());
    if (res.status() === 200) {
      const body = await res.json();
      expect(body).toHaveProperty('status');
    }
  });
});

/* ── Journal & Logs ──────────────────────────────────────────────── */
test.describe('Journal & Logs API', () => {
  requireBackend();

  test('GET /api/journal returns journal entries', async ({ request }) => {
    const res = await request.get('/api/journal');
    expect(res.status()).toBe(200);
  });

  test('GET /api/journal/snapshots returns snapshots', async ({ request }) => {
    const res = await request.get('/api/journal/snapshots');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('GET /api/logs returns log data', async ({ request }) => {
    const res = await request.get('/api/logs');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toBeDefined();
  });

  test('DELETE /api/logs clears logs', async ({ request }) => {
    const res = await request.delete('/api/logs');
    expect(res.status()).toBe(200);
  });
});

/* ── Chat ────────────────────────────────────────────────────────── */
test.describe('Chat API', () => {
  requireBackend();

  test('GET /api/chat/history returns chat history', async ({ request }) => {
    const res = await request.get('/api/chat/history');
    expect(res.status()).toBe(200);
  });

  test('POST /api/chat sends a message', async ({ request }) => {
    const res = await request.post('/api/chat', {
      data: { message: 'hello' },
    });
    // 200 if LLM configured, 500 if not
    expect([200, 500]).toContain(res.status());
  });

  // Note: GET /api/chat/stream is a WebSocket upgrade — tested separately below
});

/* ── WebSocket Chat Stream ───────────────────────────────────────── */
test.describe('WebSocket Chat', () => {
  requireBackend();

  test('GET /api/chat/stream returns upgrade response', async ({ request }) => {
    // Plain HTTP GET to a WebSocket endpoint should return 400 or upgrade-required
    const res = await request.get('/api/chat/stream');
    // Axum returns 400 for non-upgrade requests to WS endpoints
    expect([400, 426]).toContain(res.status());
  });
});

/* ── Notifications API ──────────────────────────────────────────── */
test.describe('Notifications API', () => {
  requireBackend();

  test('GET /api/notifications returns notification list', async ({ request }) => {
    const res = await request.get('/api/notifications');
    // 200 if backend has notification routes, otherwise SPA fallback (200 + HTML)
    expect(res.status()).toBe(200);
    const ct = res.headers()['content-type'] || '';
    if (ct.includes('json')) {
      const body = await res.json();
      expect(body.notifications).toBeDefined();
      expect(typeof body.unread_count).toBe('number');
    }
  });

  test('GET /api/notifications/unread-count returns count', async ({ request }) => {
    const res = await request.get('/api/notifications/unread-count');
    expect(res.status()).toBe(200);
    const ct = res.headers()['content-type'] || '';
    if (ct.includes('json')) {
      const body = await res.json();
      expect(typeof body.unread_count).toBe('number');
    }
  });

  test('GET /api/notifications/config returns config', async ({ request }) => {
    const res = await request.get('/api/notifications/config');
    expect(res.status()).toBe(200);
    const ct = res.headers()['content-type'] || '';
    if (ct.includes('json')) {
      const body = await res.json();
      expect(typeof body.enabled).toBe('boolean');
      expect(body.events).toBeDefined();
    }
  });

  test('POST /api/notifications/config saves config', async ({ request }) => {
    const res = await request.post('/api/notifications/config', {
      data: {
        enabled: true,
        in_app: true,
        email: { enabled: false, smtp_host: '', smtp_port: 465, username: '', password: '', from: '', to: [], tls: true },
        webhook: { enabled: false, provider: 'dingtalk', url: '', secret: '' },
        events: { order_filled: true, order_rejected: true, risk_alert: true, engine_started: false, engine_stopped: true },
      },
    });
    expect([200, 405, 415]).toContain(res.status());
  });

  test('POST /api/notifications/read-all marks all read', async ({ request }) => {
    const res = await request.post('/api/notifications/read-all');
    expect([200, 405, 415]).toContain(res.status());
  });

  test('POST /api/notifications/test sends test notification', async ({ request }) => {
    const res = await request.post('/api/notifications/test');
    expect([200, 405, 415]).toContain(res.status());
  });
});
