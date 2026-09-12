# 提醒自动化与飞书机器人

## 行为边界

飞书提醒沿用现有两阶段确认协议：

```text
用户提出提醒
→ create_reminder_draft 默认使用飞书渠道
→ 页面展示发送群聊、消息内容、时间和时区
→ 用户确认
→ SQLite 保存 scheduled 提醒
→ 本地调度器领取到期任务
→ 飞书 Webhook 发送
→ 保存 completed、failed 或 unknown 状态与转换历史
```

生成草稿不会发送，用户取消不会创建任务。Webhook 地址与签名密钥只从本机环境
读取，不进入模型上下文、SQLite、确认卡或 Trace。提醒保存 Webhook 的不可逆目标
指纹；如果确认后更换 Webhook，旧任务会失败并要求重新创建、重新确认，不能静默
改发到另一个群聊。

主动 check-in 复用同一条确认链路，但使用 `reminder_type=check_in` 和
`recurrence=daily/weekdays`。默认没有任何主动任务；用户在提醒页选择“设置主动
check-in”、调整自然语言中的时间和频率并发送后，页面先展示草稿，只有确认才保存。
到期时调度器读取当天已确认的饮食、饮水、运动或体重记录，区分“已记录”和“暂未
看到记录”，再通过飞书询问是否需要补记。它不推断用户没有完成，也不会自动写入。

## 飞书机器人配置

在目标飞书群中添加自定义机器人，复制 Webhook；建议同时开启签名校验。然后在
`.env` 中配置：

```dotenv
FEISHU_REMINDER_ENABLED=true
FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/REPLACE_ME
FEISHU_SIGNING_SECRET=REPLACE_ME
FEISHU_DESTINATION_LABEL=飞书群「我的健康提醒」
FEISHU_REQUEST_TIMEOUT=8
REMINDER_POLL_SECONDS=30
```

不要把 `.env`、Webhook 或签名密钥提交到 GitHub。Provider 只允许 HTTPS 且域名为
`open.feishu.cn` 或 `open.larksuite.com` 的机器人 Hook 路径。

## 使用和验证

启动应用：

```bash
.venv/bin/python app.py
```

对话示例：

```text
明天上午 10 点通过飞书提醒我喝水
我想每天晚上 9 点通过飞书主动 check-in，关注饮食、饮水和运动
```

确认卡必须显示：

- 发送到哪个飞书群；
- 发送什么内容；
- 计划时间和时区；
- 确认前尚未安排。

专项测试：

```bash
.venv/bin/python -m pytest tests/unit/test_reminder_automation.py -q
```

## 可靠性语义

- 本地调度器随 `app.py` 启动，每次轮询只领取已确认且到期的任务；
- 未确认草稿、历史遗留的非飞书记录、已暂停/取消/完成提醒不会发送；
- 用户关闭提醒或当前处于免打扰时段时暂不发送，离开免打扰后再处理；
- 发送前先把状态原子更新为 `fired`，避免同一进程并发重复发送；
- 飞书成功后写入 `completed`，失败写入 `failed`，用户可以恢复后重试；
- 重复 check-in 成功发送后重新进入 `scheduled` 并计算下次本地时间；工作日任务会
  跳过周六和周日；
- 网络超时等无法判断飞书是否已收到的情况写入 `unknown`（页面显示“发送结果待核实”），
  不自动重发，避免重复打扰；用户核实群聊后可再决定是否恢复；
- 应用关闭时没有后台发送；重新启动后处理仍为 `scheduled/snoozed` 的到期任务；
- Webhook 没有业务幂等键，因此网络请求结果不确定或进程在发送中崩溃时采用
  “不自动重发”的保守策略，避免同一健康提醒重复打扰。

如果未来要求机器关机后仍发送，应把相同 Provider 和确认协议迁移到长期运行的
worker/cron，而不是让浏览器承担调度责任。
