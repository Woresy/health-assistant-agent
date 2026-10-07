const { test } = require('node:test');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const net = require('node:net');
const { chromium } = require('playwright');

async function reservePort() {
  const server = net.createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return port;
}

test('photo meal: desktop and mobile confirmation without weight input', { timeout: 120000 }, async () => {
  const root = path.resolve(__dirname, '../..');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'multi-meal-browser-'));
  const port = await reservePort();
  const modelPort = await reservePort();
  const provider = http.createServer((req, res) => {
    let body = '';
    req.on('data', data => { body += data; });
    req.on('end', () => {
      const request = JSON.parse(body);
      const vision = request.messages.some(m => Array.isArray(m.content) && m.content.some(c => c.type === 'image_url'));
      const content = vision ? JSON.stringify({ foods: [
        { name: '三明治餐盒', group_id: 'left', confidence: .95, estimated_grams: 350, portion_description: '左侧约一盒', estimated_nutrition: { calories_per_100g: 200, protein_per_100g: 12, fat_per_100g: 8, carbs_per_100g: 20 } },
        { name: '意面餐盒', group_id: 'middle', confidence: .9, estimated_grams: 250, portion_description: '中间约一盒', estimated_nutrition: { calories_per_100g: 160, protein_per_100g: 6, fat_per_100g: 4, carbs_per_100g: 25 } },
      ] }) : '你好，这是进度测试回复。';
      setTimeout(() => {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ choices: [{ message: { content } }] }));
      }, 2500);
    });
  });
  await new Promise(resolve => provider.listen(modelPort, '127.0.0.1', resolve));
  const child = spawn(process.env.HEALTHOS_PYTHON || path.join(root, '.venv/bin/python'), ['app.py'], {
    cwd: root, env: { ...process.env,
      APP_HOST: '127.0.0.1', APP_PORT: String(port), RAG_MODE: 'lexical',
      STORAGE_BACKEND: 'sqlite', SQLITE_DATABASE_PATH: path.join(dir, 'events.db'),
      AGENT_TRACE_PATH: path.join(dir, 'agent.jsonl'),
      AGENT_API_KEY: 'test-only', AGENT_MODEL: 'test-chat', AGENT_PROVIDER_MODE: 'openai_compatible',
      AGENT_BASE_URL: `http://127.0.0.1:${modelPort}/v1`, GRADIO_ANALYTICS_ENABLED: 'false',
      MEAL_DETECTION_MODE: 'vlm', MEAL_DETECTION_VLM_MODEL: 'test-vlm',
      MEAL_DETECTION_VLM_API_KEY: 'test-only', MEAL_DETECTION_VLM_BASE_URL: `http://127.0.0.1:${modelPort}/v1`,
      FEISHU_REMINDER_ENABLED: 'false', LANGSMITH_TRACING: 'false',
    }, stdio: ['ignore', 'pipe', 'pipe'],
  });
  let logs = '';
  for (const stream of [child.stdout, child.stderr]) stream.on('data', data => { logs = (logs + data).slice(-10000); });
  let browser;
  try {
    const url = `http://127.0.0.1:${port}`;
    let ready = false;
    for (let i = 0; i < 150; i++) {
      try { ready = (await fetch(url)).ok; } catch {}
      if (ready) break;
      await new Promise(resolve => setTimeout(resolve, 400));
    }
    assert.ok(ready, logs);
    browser = await chromium.launch({ headless: true });
    for (const [name, viewport] of [['desktop', { width: 1440, height: 1000 }], ['mobile', { width: 390, height: 844 }]]) {
      const context = await browser.newContext({ viewport });
      const page = await context.newPage();
      await page.goto(url);
      await page.getByRole('heading', { name: '今日观察', exact: true }).waitFor();
      const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.getByRole('button', { name: '添加图片', exact: true }).click()]);
      await chooser.setFiles((process.env.HEALTHOS_MEAL_TEST_IMAGE || path.join(root, 'tests/fixtures/meal.png')));
      await page.getByText('正在识别图片中的食物并估算份量', { exact: false }).waitFor();
      await page.getByText('这一餐约 1100 千卡', { exact: false }).waitFor({ timeout: 30000 });
      assert.equal(await page.getByLabel('吃了多少（克）').isVisible(), false);
      assert.ok((await page.locator('.meal-preview').first().innerText()).includes('按照片估计'));
      assert.equal(await page.getByText('未找到同名食物', { exact: false }).count(), 0);
      assert.equal(await page.getByText('约 350 克', { exact: false }).isVisible(), true);
      assert.equal(await page.getByText('约 250 克', { exact: false }).isVisible(), true);
      const panel = page.locator('.chat-meal-workflow');
      await panel.scrollIntoViewIfNeeded();
      if (process.env.HEALTHOS_SCREENSHOT_DIR) {
        fs.mkdirSync(process.env.HEALTHOS_SCREENSHOT_DIR, { recursive: true });
        await page.screenshot({ path: path.join(process.env.HEALTHOS_SCREENSHOT_DIR, `multi-meal-${name}.png`), fullPage: true });
      }
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
      if (name === 'desktop') {
        await panel.getByRole('button', { name: '保存这一餐', exact: true }).click();
        await page.getByText('饮食记录已保存，共 2 项。', { exact: false }).waitFor();
      } else {
        await panel.getByRole('button', { name: '取消', exact: true }).click();
        await panel.waitFor({ state: 'hidden' });
      }
      await page.getByPlaceholder('直接说：我刚喝了水').fill('你好');
      await page.getByRole('button', { name: '发送', exact: true }).click();
      await page.locator('.agent-process.active').getByText('正在分析请求并整理下一步', { exact: true }).waitFor();
      assert.equal(await page.getByText('你好，这是进度测试回复。', { exact: true }).count(), 0);
      if (name === 'mobile') {
        assert.equal(await page.evaluate(() => {
          const progress = document.querySelector('.agent-process.active').getBoundingClientRect();
          const composer = document.querySelector('.composer-shell').getBoundingClientRect();
          return progress.bottom <= composer.top + 1;
        }), true, 'mobile composer must not cover live progress');
      }
      if (process.env.HEALTHOS_SCREENSHOT_DIR) {
        await page.locator('.agent-process.active').scrollIntoViewIfNeeded();
        await page.screenshot({ path: path.join(process.env.HEALTHOS_SCREENSHOT_DIR, `progress-${name}.png`), fullPage: true });
      }
      await page.getByText('你好，这是进度测试回复。', { exact: true }).waitFor();
      await page.locator('.agent-process.complete').waitFor();
      await context.close();
    }
  } catch (error) {
    error.message += '\n' + logs;
    throw error;
  } finally {
    if (browser) await browser.close();
    child.kill('SIGTERM');
    await new Promise(resolve => provider.close(resolve));
  }
});
