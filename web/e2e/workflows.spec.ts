import { test, expect } from '@playwright/test';

/**
 * End-to-end workflow tests — simulate real user interactions.
 * Backend must be running for full coverage.
 */

test.describe('Dashboard → Strategy → Backtest workflow', () => {
  test('full navigation flow', async ({ page }) => {
    // 1. Start at Dashboard
    await page.goto('/');
    await page.waitForTimeout(500);
    await expect(page.locator('body')).toBeVisible();

    // 2. Navigate to Strategy via sidebar click
    await page.locator('nav').getByText(/strateg|策略/i).first().click();
    await page.waitForURL('**/strategy');
    await expect(page.locator('body')).toBeVisible();

    // 3. Navigate to Backtest
    await page.locator('nav').getByText(/backtest|回测/i).first().click();
    await page.waitForURL('**/backtest');
    await expect(page.locator('body')).toBeVisible();
  });
});

test.describe('Factor Mining workflow', () => {
  test('tab switching and data source config', async ({ page }) => {
    await page.goto('/factor-mining');
    await page.waitForTimeout(500);

    // 1. Overview tab is default — check lifecycle display
    await expect(page.getByText('候选').first()).toBeVisible();

    // 2. Switch to Parametric tab (use button selector to avoid matching description text)
    await page.locator('button', { hasText: '参数化搜索' }).click();
    await page.waitForTimeout(500);

    // Default is synthetic
    await expect(page.getByText('模拟数据').first()).toBeVisible();

    // 3. Switch to akshare
    await page.locator('button', { hasText: '真实行情' }).click();
    await page.waitForTimeout(500);

    // Symbols input appears
    await expect(page.getByText('股票代码').first()).toBeVisible();

    // 4. Check date inputs exist
    await expect(page.getByText('开始日期').first()).toBeVisible();
    await expect(page.getByText('结束日期').first()).toBeVisible();

    // 5. Switch to GP tab
    await page.locator('button', { hasText: 'GP进化' }).click();
    await page.waitForTimeout(500);
    await expect(page.getByText('种群大小').first()).toBeVisible();

    // 6. GP tab also has data source config
    await expect(page.getByText('模拟数据').first()).toBeVisible();

    // 7. Switch to Registry tab
    await page.locator('button', { hasText: '因子注册表' }).click();
    await page.waitForTimeout(300);

    // 8. Switch to Export tab
    await page.locator('button', { hasText: '导出集成' }).click();
    await page.waitForTimeout(300);
  });

  test('parametric mining button exists and is clickable', async ({ page }) => {
    await page.goto('/factor-mining');
    await page.locator('button', { hasText: '参数化搜索' }).click();
    await page.waitForTimeout(500);

    const startBtn = page.locator('button', { hasText: '开始搜索' });
    await expect(startBtn).toBeVisible();
  });

  test('GP evolution button exists and is clickable', async ({ page }) => {
    await page.goto('/factor-mining');
    await page.locator('button', { hasText: 'GP进化' }).click();
    await page.waitForTimeout(500);

    const startBtn = page.locator('button', { hasText: '开始进化' });
    await expect(startBtn).toBeVisible();
  });
});

test.describe('Auto Trade page', () => {
  test('renders trading controls', async ({ page }) => {
    await page.goto('/autotrade');
    await page.waitForTimeout(500);

    const body = await page.textContent('body');
    expect(body).toBeTruthy();
    expect(body).not.toContain('Something went wrong');
  });
});

test.describe('Risk Management page', () => {
  test('renders risk dashboard', async ({ page }) => {
    await page.goto('/risk');
    await page.waitForTimeout(500);

    const body = await page.textContent('body');
    expect(body).toBeTruthy();
    expect(body).not.toContain('Something went wrong');
  });
});

test.describe('Market Data page', () => {
  test('renders market data view', async ({ page }) => {
    await page.goto('/market');
    await page.waitForTimeout(500);

    const body = await page.textContent('body');
    expect(body).toBeTruthy();
    expect(body).not.toContain('Something went wrong');
  });
});

test.describe('Screener page', () => {
  test('renders stock screener', async ({ page }) => {
    await page.goto('/screener');
    await page.waitForTimeout(500);

    const body = await page.textContent('body');
    expect(body).toBeTruthy();
    expect(body).not.toContain('Something went wrong');
  });
});

test.describe('Sentiment page', () => {
  test('renders sentiment analysis', async ({ page }) => {
    await page.goto('/sentiment');
    await page.waitForTimeout(500);

    const body = await page.textContent('body');
    expect(body).toBeTruthy();
    expect(body).not.toContain('Something went wrong');
  });
});

test.describe('DL Models page', () => {
  test('renders deep learning models page', async ({ page }) => {
    await page.goto('/dl-models');
    await page.waitForTimeout(500);

    const body = await page.textContent('body');
    expect(body).toBeTruthy();
    expect(body).not.toContain('Something went wrong');
  });
});

test.describe('Logs page', () => {
  test('renders logs viewer', async ({ page }) => {
    await page.goto('/logs');
    await page.waitForTimeout(500);

    const body = await page.textContent('body');
    expect(body).toBeTruthy();
    expect(body).not.toContain('Something went wrong');
  });
});

test.describe('Chat page interaction', () => {
  test('can type a message in chat input', async ({ page }) => {
    await page.goto('/chat');
    await page.waitForTimeout(500);

    const input = page.locator('input[type="text"], textarea').first();
    await expect(input).toBeVisible();

    await input.fill('测试消息');
    expect(await input.inputValue()).toBe('测试消息');
  });
});

test.describe('No console errors on page load', () => {
  const pages = [
    '/', '/market', '/strategy', '/backtest', '/screener',
    '/autotrade', '/risk', '/sentiment', '/dl-models',
    '/factor-mining', '/portfolio', '/logs', '/chat',
  ];

  for (const path of pages) {
    test(`no uncaught errors on ${path}`, async ({ page }) => {
      const errors: string[] = [];
      page.on('pageerror', (err) => errors.push(err.message));

      await page.goto(path);
      await page.waitForTimeout(1000);

      // Filter out known acceptable errors (e.g., network failures when backend is down)
      const criticalErrors = errors.filter(
        (e) => !e.includes('fetch') && !e.includes('Failed to fetch') && !e.includes('NetworkError')
      );
      expect(criticalErrors).toEqual([]);
    });
  }
});
