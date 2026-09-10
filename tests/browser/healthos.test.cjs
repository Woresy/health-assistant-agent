const assert = require("node:assert/strict");
const { spawn, execFileSync } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const { after, before, test } = require("node:test");

const projectRoot = path.resolve(__dirname, "../..");
const screenshotDirectory = process.env.HEALTHOS_SCREENSHOT_DIR || "";
const python = process.env.HEALTHOS_PYTHON || path.join(projectRoot, ".venv/bin/python");
const playwright = process.env.PLAYWRIGHT_MODULE_PATH
  ? require(process.env.PLAYWRIGHT_MODULE_PATH)
  : require("playwright");

let appProcess;
let baseUrl;
let browser;
let modelServer;
let temporaryDirectory;
let serverOutput = "";

function reservePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close(() => resolve(address.port));
    });
  });
}

function startModelServer(port) {
  return new Promise((resolve) => {
    modelServer = http.createServer((request, response) => {
      if (request.method !== "POST" || !request.url.endsWith("/chat/completions")) {
        response.writeHead(404).end();
        return;
      }
      request.resume();
      request.on("end", () => {
        response.writeHead(200, { "content-type": "application/json" });
        response.end(JSON.stringify({
          id: "browser-e2e-response",
          object: "chat.completion",
          created: Math.floor(Date.now() / 1000),
          model: "browser-e2e-model",
          choices: [{
            index: 0,
            message: { role: "assistant", content: "浏览器测试回复：消息已收到。" },
            finish_reason: "stop",
          }],
        }));
      });
    });
    modelServer.listen(port, "127.0.0.1", resolve);
  });
}

async function waitForApplication(url, timeoutMs = 90000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (appProcess.exitCode !== null) {
      throw new Error(`Gradio exited early (${appProcess.exitCode}).\n${serverOutput}`);
    }
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch (_) {
      // The server is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  throw new Error(`Gradio did not become ready.\n${serverOutput}`);
}

async function openApp(viewport) {
  const context = await browser.newContext({ viewport });
  const page = await context.newPage();
  await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
  await page.getByRole("heading", { name: "个人健康工作台" }).waitFor();
  return { context, page };
}

async function openNavigation(page, name) {
  const tab = page.getByRole("tab", { name, exact: true });
  if (await tab.isVisible()) {
    await tab.click();
    return;
  }
  await page.getByRole("button", { name: "More tabs" }).click();
  await page
    .locator("button:visible")
    .filter({ hasText: new RegExp(`^${name}$`) })
    .last()
    .click();
}

before(async () => {
  temporaryDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "healthos-browser-"));
  const databasePath = path.join(temporaryDirectory, "healthos.db");
  execFileSync(python, ["tests/browser/seed_browser_state.py", databasePath], {
    cwd: projectRoot,
    stdio: "inherit",
  });
  const port = await reservePort();
  const modelPort = await reservePort();
  await startModelServer(modelPort);
  baseUrl = `http://127.0.0.1:${port}`;
  appProcess = spawn(python, ["app.py"], {
    cwd: projectRoot,
    env: {
      ...process.env,
      AGENT_API_KEY: "browser-e2e-only",
      AGENT_BASE_URL: `http://127.0.0.1:${modelPort}/v1`,
      AGENT_MODEL: "browser-e2e-model",
      AGENT_PROVIDER_MODE: "openai_compatible",
      AGENT_TARGET_RESPONSE_SECONDS: "2",
      AGENT_REQUEST_TIMEOUT: "5",
      AGENT_TRACE_PATH: path.join(temporaryDirectory, "agent-traces.jsonl"),
      APP_HOST: "127.0.0.1",
      APP_PORT: String(port),
      FEISHU_DESTINATION_LABEL: "飞书群「浏览器验收」",
      FEISHU_REMINDER_ENABLED: "true",
      FEISHU_WEBHOOK_URL: "https://open.feishu.cn/open-apis/bot/v2/hook/browser-e2e",
      LANGSMITH_TRACING: "false",
      RAG_MODE: "lexical",
      SQLITE_DATABASE_PATH: databasePath,
      STORAGE_BACKEND: "sqlite",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  for (const stream of [appProcess.stdout, appProcess.stderr]) {
    stream.on("data", (chunk) => {
      serverOutput = `${serverOutput}${chunk}`.slice(-16000);
    });
  }
  await waitForApplication(baseUrl);
  browser = await playwright.chromium.launch({ headless: true });
});

after(async () => {
  if (browser) await browser.close();
  if (appProcess && appProcess.exitCode === null) {
    appProcess.kill("SIGINT");
    await new Promise((resolve) => {
      appProcess.once("exit", resolve);
      setTimeout(resolve, 3000);
    });
    if (appProcess.exitCode === null) appProcess.kill("SIGKILL");
  }
  if (modelServer) await new Promise((resolve) => modelServer.close(resolve));
  if (temporaryDirectory) {
    fs.rmSync(temporaryDirectory, { recursive: true, force: true });
  }
});

test("desktop: history restores and a daily record opens its edit conversation", async () => {
  const { context, page } = await openApp({ width: 1440, height: 1000 });
  const input = page.getByPlaceholder("直接说：我刚喝了水");
  await input.fill("请回复这条浏览器测试消息");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  try {
    await page.getByText("浏览器测试回复：消息已收到。").waitFor();
  } catch (error) {
    console.error((await page.locator("body").innerText()).slice(-5000));
    console.error(serverOutput);
    throw error;
  }

  await page.reload({ waitUntil: "domcontentloaded" });
  await page.getByText("请回复这条浏览器测试消息").waitFor();
  await page.getByText("浏览器测试回复：消息已收到。").waitFor();

  await openNavigation(page, "今天");
  await page.getByRole("button", { name: "刷新今日", exact: true }).click();
  const waterRecord = page.getByRole("button", { name: "水 350 ml", exact: true });
  await waterRecord.waitFor();
  await waterRecord.click();
  await page.getByText("正在修改").waitFor();
  assert.equal(await page.locator("#healthos-record").isVisible(), true);
  assert.equal(await page.getByPlaceholder("直接说：我刚喝了水").inputValue(), "请把这条记录修改为：");
  await context.close();
});

test("desktop: memory controls expose human-readable data and require confirmation", async () => {
  const { context, page } = await openApp({ width: 1440, height: 1000 });
  await openNavigation(page, "数据与隐私");
  await page.getByText("饮食偏好").waitFor();
  await page.getByText("少油").waitFor();
  assert.equal(await page.getByText(/22222222-|11111111-/).count(), 0);

  await page.getByRole("button", { name: "清除全部记忆" }).click();
  await page.getByText("确认清除全部长期记忆？").waitFor();
  await page.getByRole("button", { name: "取消" }).click();
  await page.getByText("少油").waitFor();
  await context.close();
});

test("desktop: active check-in stays off until its in-chat draft is confirmed", async () => {
  const { context, page } = await openApp({ width: 1440, height: 1000 });
  await openNavigation(page, "提醒");
  await page.getByText("主动 check-in 默认关闭").waitFor();
  await page.getByRole("button", { name: "设置主动 check-in", exact: true }).click();

  const input = page.getByPlaceholder("直接说：我刚喝了水");
  await page.waitForFunction(
    () => document.querySelector('textarea[placeholder="直接说：我刚喝了水"]')?.value.includes("主动 check-in"),
  );
  assert.equal(
    await input.inputValue(),
    "我想每天晚上 9 点通过飞书主动 check-in，关注饮食、饮水和运动",
  );
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await page.getByText("确认启用主动 check-in").waitFor();
  await page.getByText(/不会自动写入/).waitFor();
  await page.getByRole("button", { name: "确认启用", exact: true }).click();

  await openNavigation(page, "提醒");
  await page.getByRole("button", { name: "主动健康 check-in", exact: true }).waitFor();
  await page.getByRole("button", { name: "每天", exact: true }).waitFor();
  await page.getByRole("button", { name: "飞书群「浏览器验收」", exact: true }).waitFor();
  if (screenshotDirectory) {
    fs.mkdirSync(screenshotDirectory, { recursive: true });
    await page.screenshot({ path: path.join(screenshotDirectory, "active-check-in-desktop.png"), fullPage: true });
  }
  await context.close();
});

test("mobile: primary chat controls stay inside the viewport and remain keyboard reachable", async () => {
  const { context, page } = await openApp({ width: 390, height: 844 });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  assert.ok(overflow <= 1, `page has ${overflow}px horizontal overflow`);

  const input = page.getByPlaceholder("直接说：我刚喝了水");
  const send = page.getByRole("button", { name: "发送", exact: true });
  for (const locator of [input, send]) {
    const box = await locator.boundingBox();
    assert.ok(box, "control must be visible");
    assert.ok(box.x >= 0 && box.x + box.width <= 390, "control must fit the viewport");
    assert.ok(box.height >= 42, "primary control must meet the documented touch target");
  }
  await input.focus();
  await page.keyboard.press("Tab");
  assert.equal(await send.evaluate((element) => element === document.activeElement), true);
  if (screenshotDirectory) {
    fs.mkdirSync(screenshotDirectory, { recursive: true });
    await page.screenshot({ path: path.join(screenshotDirectory, "chat-mobile.png"), fullPage: true });
  }
  await context.close();
});
