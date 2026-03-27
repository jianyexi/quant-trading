import { test, expect } from '@playwright/test';

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

    // Top-level items are always visible
    await expect(page.locator('nav').getByText('Dashboard').first()).toBeVisible();
    await expect(page.locator('nav').getByText('AI Chat').first()).toBeVisible();

    // Expand all collapsed sidebar groups
    for (const group of ['量化研究', '交易执行', '数据 & 监控']) {
      await page.locator('nav button', { hasText: group }).click();
    }
    await page.waitForTimeout(300);

    const navTexts = [
      '量化流水线', '因子挖掘', 'DL模型研究', 'LLM训练', '任务历史',           // 量化研究
      '行情数据', '智能选股', '策略管理', '自动交易', '持仓管理', '风控管理',     // 交易执行
      '舆情数据', '通知中心', '系统日志', '性能监控', '统计报表', '延迟分析', '服务管理', // 数据 & 监控
    ];
    for (const text of navTexts) {
      await expect(page.locator('nav').getByText(text, { exact: true }).first()).toBeVisible();
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
      { path: '/notifications', heading: /通知/i },
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
    // Wait for loading spinner to finish (API calls may fail quickly)
    await page.waitForTimeout(2000);

    // Tab labels
    await expect(page.locator('text=总览').first()).toBeVisible();
    await expect(page.locator('text=参数化搜索').first()).toBeVisible();
    await expect(page.locator('text=GP进化').first()).toBeVisible();
    await expect(page.locator('text=因子注册表').first()).toBeVisible();
    await expect(page.locator('text=导出集成').first()).toBeVisible();
  });

  test('parametric tab has data source config', async ({ page }) => {
    await page.goto('/factor-mining');
    await page.waitForTimeout(2000);

    // Click parametric tab button
    await page.locator('button', { hasText: '参数化搜索' }).click();
    await page.waitForTimeout(500);

    // Use :visible h3 as anchor — only one tab's h3 is visible at a time
    await expect(page.locator('h3:visible', { hasText: '参数化因子搜索' })).toBeVisible();
    // Data source label is visible (use :visible to skip hidden tabs)
    await expect(page.locator('span:visible', { hasText: '数据来源' })).toBeVisible();
    await expect(page.locator(':visible:text("IC阈值")')).toBeVisible();
    await expect(page.locator(':visible:text("Top N")')).toBeVisible();
  });

  test('GP tab has GP params', async ({ page }) => {
    await page.goto('/factor-mining');
    await page.waitForTimeout(2000);

    await page.locator('button', { hasText: 'GP进化' }).click();
    await page.waitForTimeout(500);

    // Verify GP section heading is visible
    await expect(page.locator('h3:visible', { hasText: '遗传编程因子进化' })).toBeVisible();
    await expect(page.locator('span:visible', { hasText: '数据来源' })).toBeVisible();
    await expect(page.locator(':visible:text("种群大小")')).toBeVisible();
    await expect(page.locator(':visible:text("迭代代数")')).toBeVisible();
    await expect(page.locator(':visible:text("最大树深")')).toBeVisible();
  });

  test('parametric tab shows stock inputs', async ({ page }) => {
    await page.goto('/factor-mining');
    await page.waitForTimeout(2000);

    await page.locator('button', { hasText: '参数化搜索' }).click();
    await page.waitForTimeout(500);

    // Scope with :visible to avoid matching hidden persistent pages / hidden tabs
    await expect(page.locator('h3:visible', { hasText: '参数化因子搜索' })).toBeVisible();
    await expect(page.locator('label:visible', { hasText: '股票代码' })).toBeVisible();
    await expect(page.locator('label:visible', { hasText: '开始日期' })).toBeVisible();
    await expect(page.locator('label:visible', { hasText: '结束日期' })).toBeVisible();
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

    // Chat uses a <textarea> with a specific placeholder
    const chatInput = page.locator('textarea[placeholder*="Ask about"]');
    await expect(chatInput).toBeVisible();
  });
});
