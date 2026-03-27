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

    // 2. Expand 交易执行 group, then click 策略管理
    await page.locator('nav button', { hasText: '交易执行' }).click();
    await page.locator('nav a', { hasText: '策略管理' }).click();
    await page.waitForURL('**/strategy');
    await expect(page.locator('body')).toBeVisible();

    // 3. Navigate to Backtest via direct URL (no sidebar link for /backtest)
    await page.goto('/backtest');
    await page.waitForTimeout(500);
    await expect(page.locator('body')).toBeVisible();
  });
});

test.describe('Factor Mining workflow', () => {
  test('tab switching and data source config', async ({ page }) => {
    await page.goto('/factor-mining');
    // Wait for loading to complete (API calls fail quickly when backend is down)
    await page.waitForTimeout(2000);

    // 1. Overview tab is default — check header
    await expect(page.getByText('因子挖掘').first()).toBeVisible();

    // 2. Switch to Parametric tab
    await page.locator('button', { hasText: '参数化搜索' }).click();
    await page.waitForTimeout(500);

    // Scope to parametric section (unique heading avoids matching hidden tabs)
    await expect(page.locator('h3:visible', { hasText: '参数化因子搜索' })).toBeVisible();
    await expect(page.locator('span:visible', { hasText: '数据来源' })).toBeVisible();
    await expect(page.locator('label:visible', { hasText: '股票代码' })).toBeVisible();
    await expect(page.locator('label:visible', { hasText: '开始日期' })).toBeVisible();
    await expect(page.locator('label:visible', { hasText: '结束日期' })).toBeVisible();

    // 3. Switch to GP tab
    await page.locator('button', { hasText: 'GP进化' }).click();
    await page.waitForTimeout(500);

    // Scope to GP section
    await expect(page.locator('h3:visible', { hasText: '遗传编程因子进化' })).toBeVisible();
    await expect(page.locator(':visible:text("种群大小")')).toBeVisible();
    await expect(page.locator('span:visible', { hasText: '数据来源' })).toBeVisible();

    // 4. Switch to Registry tab
    await page.locator('button', { hasText: '因子注册表' }).click();
    await page.waitForTimeout(300);

    // 5. Switch to Export tab
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

    const input = page.locator('textarea[placeholder*="Ask about"]');
    await expect(input).toBeVisible();

    await input.fill('测试消息');
    expect(await input.inputValue()).toBe('测试消息');
  });
});

test.describe('No console errors on page load', () => {
  const pages = [
    '/', '/market', '/strategy', '/backtest', '/screener',
    '/autotrade', '/risk', '/sentiment', '/dl-models',
    '/factor-mining', '/portfolio', '/notifications', '/logs', '/chat',
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
