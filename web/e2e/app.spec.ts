import { test, expect, type Page } from '@playwright/test';

/*
 * Helper: wait for the backend API to be reachable via the Vite proxy.
 * If the backend is down the UI still loads but API calls fail.
 */
async function apiAvailable(page: Page): Promise<boolean> {
  try {
    const res = await page.request.get('/api/health');
    return res.ok();
  } catch {
    return false;
  }
}

test.describe('Navigation & Layout', () => {
  test('should load the app and show sidebar', async ({ page }) => {
    await page.goto('/');
    // Sidebar should be visible with nav items
    await expect(page.locator('nav')).toBeVisible();
    // Title or brand should appear
    await expect(page.locator('text=Dashboard').first()).toBeVisible();
  });

  test('sidebar contains all navigation links', async ({ page }) => {
    await page.goto('/');
    const navTexts = [
      'Dashboard',
      'Market',
      'Strategy',
      'Backtest',
      '选股',
      '交易',
      '风控',
      '舆情',
      'DL',
      '因子',
      'Portfolio',
      '日志',
      'Chat',
    ];
    for (const text of navTexts) {
      await expect(
        page.locator(`nav >> text=/${text}/i`).first()
      ).toBeVisible();
    }
  });

  test('can navigate to each page without crash', async ({ page }) => {
    const routes = [
      { path: '/', heading: /dashboard/i },
      { path: '/market', heading: /market|行情/i },
      { path: '/strategy', heading: /strateg|策略/i },
      { path: '/backtest', heading: /backtest|回测/i },
      { path: '/screener', heading: /screen|选股/i },
      { path: '/autotrade', heading: /trade|交易/i },
      { path: '/risk', heading: /risk|风控/i },
      { path: '/sentiment', heading: /sentiment|舆情/i },
      { path: '/dl-models', heading: /dl|model|研究/i },
      { path: '/factor-mining', heading: /factor|因子/i },
      { path: '/portfolio', heading: /portfolio|持仓/i },
      { path: '/logs', heading: /log|日志/i },
      { path: '/chat', heading: /chat|ai/i },
    ];

    for (const { path } of routes) {
      await page.goto(path);
      // Page should render without crash — no uncaught errors
      await expect(page.locator('body')).toBeVisible();
      // No white-screen-of-death: at least some content
      const html = await page.content();
      expect(html.length).toBeGreaterThan(500);
    }
  });
});

test.describe('Dashboard', () => {
  test('renders dashboard cards and sections', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(1000);

    // Should have some card-like elements (overview panels)
    const body = await page.textContent('body');
    expect(body).toBeTruthy();
    // No React error boundary text
    expect(body).not.toContain('Something went wrong');
  });
});

test.describe('Factor Mining Page', () => {
  test('renders with all tabs', async ({ page }) => {
    await page.goto('/factor-mining');
    await page.waitForTimeout(500);

    // Tab labels
    await expect(page.locator('text=总览').first()).toBeVisible();
    await expect(page.locator('text=参数化搜索').first()).toBeVisible();
    await expect(page.locator('text=GP进化').first()).toBeVisible();
    await expect(page.locator('text=因子注册表').first()).toBeVisible();
    await expect(page.locator('text=导出集成').first()).toBeVisible();
  });

  test('parametric tab has data source config', async ({ page }) => {
    await page.goto('/factor-mining');
    // Click parametric tab button (not description text)
    await page.locator('button', { hasText: '参数化搜索' }).click();
    await page.waitForTimeout(500);

    // Data source buttons (contain emoji prefix)
    await expect(page.getByText('模拟数据').first()).toBeVisible();
    await expect(page.getByText('真实行情').first()).toBeVisible();

    // Parameter inputs
    await expect(page.getByText('IC阈值').first()).toBeVisible();
    await expect(page.getByText('Top N').first()).toBeVisible();
  });

  test('GP tab has data source config and GP params', async ({ page }) => {
    await page.goto('/factor-mining');
    await page.locator('button', { hasText: 'GP进化' }).click();
    await page.waitForTimeout(500);

    await expect(page.getByText('模拟数据').first()).toBeVisible();
    await expect(page.getByText('真实行情').first()).toBeVisible();
    await expect(page.getByText('种群大小').first()).toBeVisible();
    await expect(page.getByText('迭代代数').first()).toBeVisible();
    await expect(page.getByText('最大树深').first()).toBeVisible();
  });

  test('switching to akshare shows symbols and date inputs', async ({ page }) => {
    await page.goto('/factor-mining');
    await page.locator('button', { hasText: '参数化搜索' }).click();
    await page.waitForTimeout(500);

    // Click akshare button
    await page.locator('button', { hasText: '真实行情' }).click();
    await page.waitForTimeout(500);

    // Should show symbols input and date inputs
    await expect(page.getByText('股票代码').first()).toBeVisible();
    await expect(page.getByText('开始日期').first()).toBeVisible();
    await expect(page.getByText('结束日期').first()).toBeVisible();
  });

  test('switching to synthetic shows n_bars input', async ({ page }) => {
    await page.goto('/factor-mining');
    await page.locator('button', { hasText: '参数化搜索' }).click();
    await page.waitForTimeout(500);

    // Default is synthetic — should show bars input
    await expect(page.getByText(/数据量/).first()).toBeVisible();
  });
});

test.describe('Strategy Config Page', () => {
  test('renders strategy configuration form', async ({ page }) => {
    await page.goto('/strategy');
    await page.waitForTimeout(500);
    const body = await page.textContent('body');
    expect(body).toBeTruthy();
    expect(body).not.toContain('Something went wrong');
  });
});

test.describe('Chat Page', () => {
  test('renders chat interface with input', async ({ page }) => {
    await page.goto('/chat');
    await page.waitForTimeout(500);

    // Should have a text input or textarea for chat
    const inputOrTextarea = page.locator('input[type="text"], textarea').first();
    await expect(inputOrTextarea).toBeVisible();
  });
});
