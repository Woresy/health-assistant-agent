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
        setTimeout(() => {
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
        }, 450);
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
  await page.getByRole("heading", { name: "今日观察", exact: true }).waitFor();
  return { context, page };
}

async function openNavigation(page, name) {
  const tab = page.getByRole("tab", { name, exact: true });
  if (await tab.isVisible()) {
    await tab.click();
    return;
  }
  const mobileMore = page.getByRole("button", { name: "更多", exact: true });
  if (await mobileMore.isVisible()) {
    await mobileMore.click();
    await page
      .locator("#mobileToolsMenu button:visible")
      .filter({ hasText: new RegExp(`^${name}$`) })
      .click();
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
  await page.getByText("小满正在处理", { exact: true }).waitFor();
  await page.getByText("正在选择合适的健康工具", { exact: true }).waitFor();
  if (screenshotDirectory) {
    fs.mkdirSync(screenshotDirectory, { recursive: true });
    await page.screenshot({ path: path.join(screenshotDirectory, "thinking-desktop.png"), fullPage: false });
  }
  try {
    await page.getByText("浏览器测试回复：消息已收到。").waitFor();
  } catch (error) {
    console.error((await page.locator("body").innerText()).slice(-5000));
    console.error(serverOutput);
    throw error;
  }
  const processSummary = page.locator("details.agent-process summary").filter({ hasText: "本次处理" });
  await processSummary.waitFor();
  await processSummary.click();
  await page.getByText("理解你的请求与当前对话", { exact: true }).waitFor();
  if (screenshotDirectory) {
    await page.screenshot({ path: path.join(screenshotDirectory, "process-desktop.png"), fullPage: false });
  }

  await page.reload({ waitUntil: "domcontentloaded" });
  await page.locator("#health-chat").getByText("请回复这条浏览器测试消息", { exact: true }).waitFor();
  await page.getByText("浏览器测试回复：消息已收到。").waitFor();

  await page.locator("#healthos-record").waitFor();
  await page.getByText(/未设置目标/).first().waitFor();
  assert.equal(await page.getByText(/1800 ml/).count(), 0);
  if (screenshotDirectory) {
    fs.mkdirSync(screenshotDirectory, { recursive: true });
    await page.screenshot({ path: path.join(screenshotDirectory, "desktop.png"), fullPage: false });
  }
  await openNavigation(page, "今日完整汇总");
  const waterRecord = page.getByRole("button", { name: "水 350 ml", exact: true });
  await waterRecord.waitFor();
  await waterRecord.click();
  await page.getByText("正在修改").waitFor();
  assert.equal(await page.locator("#healthos-record").isVisible(), true);
  assert.equal(await page.getByPlaceholder("直接说：我刚喝了水").inputValue(), "请把这条记录修改为：");
  await context.close();
});

test("desktop: a short user message stays on one line in a comfortably wide bubble", async () => {
  const { context, page } = await openApp({ width: 1440, height: 1000 });
  const input = page.getByPlaceholder("直接说：我刚喝了水");
  const messageText = "每天晚上9点提醒我拉伸";
  await input.fill(messageText);
  await page.getByRole("button", { name: "发送", exact: true }).click();
  const message = page.locator("#health-chat").getByText(messageText, { exact: true });
  await message.waitFor();
  const metrics = await message.evaluate((element) => {
    const bubble = element.closest(".message") || element;
    const textStyle = getComputedStyle(element);
    return {
      bubbleWidth: bubble.getBoundingClientRect().width,
      lineHeight: Number.parseFloat(textStyle.lineHeight),
      textHeight: element.getBoundingClientRect().height,
    };
  });
  assert.ok(metrics.bubbleWidth >= 240, JSON.stringify(metrics));
  assert.ok(metrics.textHeight <= metrics.lineHeight * 1.25, JSON.stringify(metrics));
  if (screenshotDirectory) {
    fs.mkdirSync(screenshotDirectory, { recursive: true });
    await page.screenshot({ path: path.join(screenshotDirectory, "chat-bubble-desktop.png"), fullPage: false });
  }
  await context.close();
});

test("wide desktop: the full-width shell keeps sidebar controls inside the navigation rail", async () => {
  const { context, page } = await openApp({ width: 1920, height: 1000 });
  const shell = await page.locator(".gradio-container").boundingBox();
  const createConversation = await page
    .getByRole("button", { name: /创建新对话/ })
    .first()
    .boundingBox();
  const conversationListLocator = page.locator("#sidebar-conversations");
  const conversationList = await conversationListLocator.boundingBox();

  assert.ok(shell, "application shell must be rendered");
  assert.ok(shell.x <= 1 && shell.width >= 1919, "application shell must span a wide viewport");
  for (const box of [createConversation, conversationList]) {
    assert.ok(box, "sidebar control must be rendered");
    assert.ok(box.x >= 0 && box.x + box.width <= 252, "sidebar control must stay inside the 252px rail");
  }
  const itemBoxes = await page
    .locator("#sidebar-conversations label:has(input)")
    .evaluateAll((elements) =>
      elements.slice(0, 8).map((element) => element.getBoundingClientRect().toJSON()),
    );
  assert.ok(itemBoxes.length >= 8, "wide layout fixture must exercise a long history list");
  assert.ok(
    itemBoxes.every((box) => Math.abs(box.x - itemBoxes[0].x) < 1),
    "history entries must remain in one vertical column",
  );
  assert.ok(
    conversationList && conversationList.y + conversationList.height <= 1000,
    "history scroller must stay inside the viewport",
  );
  assert.equal(await page.getByText("本地真实数据", { exact: true }).count(), 0);
  if (screenshotDirectory) {
    fs.mkdirSync(screenshotDirectory, { recursive: true });
    await page.screenshot({ path: path.join(screenshotDirectory, "desktop-wide.png"), fullPage: false });
  }
  await context.close();
});

test("desktop: new conversations preserve and reopen real history", async () => {
  const { context, page } = await openApp({ width: 1440, height: 1000 });
  const input = page.getByPlaceholder("直接说：我刚喝了水");
  await page
    .locator("#health-chat")
    .getByText(/你好，我在这里/)
    .waitFor();
  assert.equal(
    await page.getByText("记录会去哪里", { exact: true }).count(),
    0,
    "the removed conversation status card must not be rendered",
  );

  await input.fill("第一段会话：请回复一声你好");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  try {
    await page.getByText("浏览器测试回复：消息已收到。").waitFor();
  } catch (error) {
    console.error((await page.locator("body").innerText()).slice(-5000));
    console.error(serverOutput);
    throw error;
  }
  const firstConversation = page
    .locator("#sidebar-conversations label")
    .filter({ hasText: "第一段会话：请回复一声你好" });
  await firstConversation.waitFor();

  await page.getByRole("button", { name: /创建新对话/ }).first().click();
  await input.fill("第二段会话：请回复一声晚上好");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await page
    .locator("#health-chat")
    .getByText("第二段会话：请回复一声晚上好", { exact: true })
    .waitFor();
  await page
    .locator("#sidebar-conversations label")
    .filter({ hasText: "第二段会话：请回复一声晚上好" })
    .waitFor();
  await page.locator("#sidebar-conversations input:checked").waitFor();

  await firstConversation.click();
  try {
    await page
      .locator("#health-chat")
      .getByText("第一段会话：请回复一声你好", { exact: true })
      .waitFor();
  } catch (error) {
    console.error((await page.locator("body").innerText()).slice(-5000));
    console.error(serverOutput);
    throw error;
  }
  assert.equal(
    await page
      .locator("#health-chat")
      .getByText("第二段会话：请回复一声晚上好", { exact: true })
      .count(),
    0,
  );
  await context.close();
});

test("desktop: current recent conversation returns from reminders to chat", async () => {
  const { context, page } = await openApp({ width: 1440, height: 1000 });
  const input = page.getByPlaceholder("直接说：我刚喝了水");
  await page.getByRole("button", { name: /创建新对话/ }).first().click();
  await input.fill("导航回对话验收");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await page.getByText("浏览器测试回复：消息已收到。").waitFor();
  const currentConversation = page.locator(
    "#sidebar-conversations label:has(input:checked)",
  );
  await currentConversation.waitFor();

  await openNavigation(page, "提醒");
  assert.equal(await page.locator("#healthos-reminders").isVisible(), true);
  await currentConversation.click();

  await page.locator("#healthos-record:visible").waitFor();
  assert.equal(await page.locator("#healthos-reminders").isVisible(), false);
  await context.close();
});

test("desktop: confirmed records live in the right rail and reminders do not use a wide table", async () => {
  const { context, page } = await openApp({ width: 1440, height: 1000 });

  assert.equal(await page.locator("#healthos-record .today-rhythm-panel").count(), 0);
  await page.locator("#healthos-margin").getByText("今日已确认记录", { exact: true }).waitFor();
  await page.locator("#healthos-margin").getByText("水 350 ml", { exact: true }).waitFor();

  await openNavigation(page, "提醒");
  assert.equal(await page.locator("#healthos-reminders table").count(), 0);
  assert.equal(await page.getByText("本地提醒中心", { exact: true }).count(), 0);
  if (screenshotDirectory) {
    fs.mkdirSync(screenshotDirectory, { recursive: true });
    await page.screenshot({ path: path.join(screenshotDirectory, "reminders-list-desktop.png"), fullPage: false });
  }
  await context.close();
});

test("desktop: meal images open a real review flow inside the conversation", async () => {
  const { context, page } = await openApp({ width: 1440, height: 1000 });
  assert.equal(
    await page.getByRole("tab", { name: "餐食图片", exact: true }).count(),
    0,
    "meal images must not remain as a separate page",
  );

  const [fileChooser] = await Promise.all([
    page.waitForEvent("filechooser"),
    page.getByRole("button", { name: "添加图片", exact: true }).click(),
  ]);
  await fileChooser.setFiles(path.join(projectRoot, "tests/fixtures/meal.png"));
  await page.getByText("确认餐食信息", { exact: true }).waitFor();
  await page.getByLabel("食物名称").fill("米饭");
  await page.getByLabel("估计份量（g）").fill("150");
  await page.getByRole("button", { name: "匹配食物", exact: true }).click();
  await page.getByText(/最接近的食物|几种相近的食物/).waitFor();
  const candidate = page.getByLabel("选择最接近的食物");
  if (!(await candidate.inputValue()).trim()) {
    await candidate.click();
    await page.getByRole("option").filter({ hasText: /米饭/ }).first().click();
  }
  assert.equal(await page.locator(".candidate-table").count(), 0);
  assert.equal(await page.getByText("查看候选检索证据", { exact: true }).count(), 0);
  assert.doesNotMatch(await page.locator("body").innerText(), /stage\s+\d|food_id|match_type/);
  await page.getByRole("button", { name: "查看营养估算", exact: true }).click();
  await page.getByText("待确认饮食记录", { exact: true }).waitFor();
  await page.getByText(/尚未保存/).waitFor();
  assert.equal(await page.getByText("查看确定性计算公式", { exact: true }).count(), 0);
  if (screenshotDirectory) {
    fs.mkdirSync(screenshotDirectory, { recursive: true });
    await page.screenshot({ path: path.join(screenshotDirectory, "meal-chat-desktop.png"), fullPage: true });
  }
  await context.close();
});

test("desktop: trends use the existing daily summaries without fabricated scores", async () => {
  const { context, page } = await openApp({ width: 1440, height: 1000 });
  await openNavigation(page, "趋势与报告");
  await page.getByRole("button", { name: "刷新趋势", exact: true }).click();
  await page.locator('#healthos-trends .cell-wrap:visible').filter({ hasText: "350 ml" }).waitFor();
  await page.getByText(/1 天有数据/).waitFor();
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
  await page.getByText("主动问候默认关闭").waitFor();
  await page.getByRole("button", { name: "设置主动问候", exact: true }).click();

  const input = page.getByPlaceholder("直接说：我刚喝了水");
  await page.waitForFunction(
    () => document.querySelector('textarea[placeholder="直接说：我刚喝了水"]')?.value.includes("主动问我"),
  );
  assert.equal(
    await input.inputValue(),
    "我想每天晚上 9 点通过飞书主动问我，关注饮食、饮水和运动",
  );
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await page.getByText("确认启用主动问候").waitFor();
  await page.getByText(/不会自动写入/).waitFor();
  await page.getByRole("button", { name: "确认启用", exact: true }).click();

  await openNavigation(page, "提醒");
  await page.locator("#healthos-reminders").getByText("主动健康问候", { exact: true }).waitFor();
  await page.locator("#healthos-reminders").getByText(/每天/).waitFor();
  await page
    .locator("#healthos-reminders .reminder-item-main small")
    .getByText("发送到 飞书群「浏览器验收」", { exact: true })
    .waitFor();
  if (screenshotDirectory) {
    fs.mkdirSync(screenshotDirectory, { recursive: true });
    await page.screenshot({ path: path.join(screenshotDirectory, "active-check-in-desktop.png"), fullPage: true });
  }
  await context.close();
});

test("desktop: remind-me prefix keeps a relative reminder in the deterministic flow", async () => {
  const { context, page } = await openApp({ width: 1440, height: 1000 });
  const input = page.getByPlaceholder("直接说：我刚喝了水");

  await input.fill("我想创建一个提醒：两分钟后在飞书上提醒我去喝水");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await page.getByRole("button", { name: "确认安排提醒", exact: true }).waitFor();
  assert.equal(await page.getByText("提醒时间必须晚于当前时间", { exact: true }).count(), 0);
  await page.getByRole("button", { name: "取消", exact: true }).click();
  await context.close();
});

test("mobile: primary chat controls stay inside the viewport and remain keyboard reachable", async () => {
  const { context, page } = await openApp({ width: 390, height: 844 });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  const overflowSources = overflow > 1
    ? await page.evaluate(() => Array.from(document.querySelectorAll("body *"))
      .map((element) => ({
        tag: element.tagName,
        id: element.id,
        className: typeof element.className === "string" ? element.className : "",
        rect: element.getBoundingClientRect().toJSON(),
      }))
      .filter(({ rect }) => rect.width > 0 && (rect.right > window.innerWidth + 1 || rect.left < -1))
      .slice(0, 12))
    : [];
  assert.ok(overflow <= 1, `page has ${overflow}px horizontal overflow: ${JSON.stringify(overflowSources)}`);
  for (const name of ["今日观察", "对话", "健康时间线", "趋势与报告", "目标与教练"]) {
    assert.equal(await page.getByRole("tab", { name, exact: true }).isVisible(), true);
  }
  assert.equal(await page.getByRole("button", { name: "更多", exact: true }).isVisible(), true);
  const moreBox = await page.getByRole("button", { name: "更多", exact: true }).boundingBox();
  assert.ok(moreBox && moreBox.x >= 0 && moreBox.x + moreBox.width <= 390, "mobile more menu must fit the viewport");
  await page.getByRole("button", { name: "更多", exact: true }).click();
  await page.locator("#mobileToolsMenu").getByRole("button", { name: "数据与隐私", exact: true }).waitFor();
  await page.getByRole("button", { name: "更多", exact: true }).click();
  const topbarBox = await page.locator(".app-topbar").boundingBox();
  assert.ok(topbarBox, "mobile top bar must be visible");
  assert.ok(topbarBox.x <= 1 && topbarBox.y <= 1, "mobile top bar must start at the viewport origin");
  assert.ok(topbarBox.width >= 389, "mobile top bar must span the viewport");
  const topbarTitleBox = await page.locator(".app-topbar h1").boundingBox();
  assert.ok(topbarTitleBox, "mobile top-bar title must be visible");
  assert.ok(
    topbarTitleBox.x >= 0 && topbarTitleBox.x + topbarTitleBox.width <= 390,
    "mobile top-bar title must fit the viewport",
  );

  const input = page.getByPlaceholder("直接说：我刚喝了水");
  const send = page.getByRole("button", { name: "发送", exact: true });
  const addImage = page.getByRole("button", { name: "添加图片", exact: true });
  for (const locator of [input, send, addImage]) {
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
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: path.join(screenshotDirectory, "mobile.png"), fullPage: false });
  }
  await context.close();
});
