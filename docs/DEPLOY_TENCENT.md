# 腾讯云香港轻量应用服务器部署

本方案面向当前的单用户 HealthOS：一台腾讯云香港轻量应用服务器运行一个 Gradio
实例和一个 Caddy 反向代理。SQLite、对话、记忆、提醒和 Agent Trace 保存在 SSD
系统盘；Caddy 负责 HTTPS 和访问密码。

## 1. 购买时这样选择

在腾讯云控制台购买轻量应用服务器（Lighthouse）：

| 购买项 | 选择 |
| --- | --- |
| 地域 | 中国香港 |
| 套餐 | Linux 2 核 4GB、70GB SSD、30Mbps、2TB/月 |
| 镜像 | Docker CE，Ubuntu Server 24.04 LTS |
| 数量 | 1 台 |
| 自动续费 | 初期关闭，稳定运行后再决定 |

中国香港实例无需 ICP 备案。购买页价格和活动优惠会变化，下单前以控制台最终金额为准。
当前项目只支持单实例，不要购买负载均衡或额外数据库。

## 2. 配置防火墙与登录

在轻量应用服务器控制台的防火墙中配置：

| 端口 | 来源 | 用途 |
| --- | --- | --- |
| 22/TCP | 你自己的公网 IP | SSH 管理 |
| 80/TCP | 全部 IPv4/IPv6 | HTTPS 证书签发与跳转 |
| 443/TCP | 全部 IPv4/IPv6 | HTTPS |
| 443/UDP | 全部 IPv4/IPv6 | HTTP/3，可选 |

不要开放 7860。Gradio 只能由 Docker 私有网络中的 Caddy 访问。使用控制台提供的密钥
或重置后的服务器密码登录，首次登录后及时更新系统：

```bash
sudo apt update
sudo apt upgrade -y
docker version
docker compose version
```

Docker CE 应用镜像已经配置 Docker。如果 `docker compose` 不存在，按腾讯云 Docker
CE 指引更新应用镜像或安装 Compose plugin。

## 3. 准备域名

将域名的 A 记录解析到轻量服务器公网 IP。等待解析生效后，可用下面的命令确认：

```bash
getent hosts 你的域名
```

Caddy 需要公网可解析域名才能自动签发可信 HTTPS 证书。不要使用裸 IP 公开健康数据，
因为 Basic Auth 在纯 HTTP 下无法保护传输内容。

## 4. 拉取项目并创建持久目录

```bash
git clone <你的 GitHub 仓库地址>
cd health-assistant-agent
cp deploy/tencent/production.env.example deploy/tencent/production.env
mkdir -p deploy/tencent/runtime-data
sudo chown -R 10001:10001 deploy/tencent/runtime-data
chmod 700 deploy/tencent/runtime-data
```

生成确认签名密钥：

```bash
openssl rand -hex 32
```

生成网页登录密码哈希。该命令会交互式询问密码，明文不会写入 shell 历史：

```bash
docker run --rm -it caddy:2.10.2-alpine caddy hash-password
```

编辑 `deploy/tencent/production.env`，至少替换：

- `DOMAIN`：已解析到服务器的域名；
- `BASIC_AUTH_USER`：网页登录用户名；
- `BASIC_AUTH_HASH`：上一步的完整哈希，保留单引号；
- `HEALTH_CONFIRMATION_SECRET`：`openssl` 输出；
- `AGENT_API_KEY`：模型密钥。

飞书和 LangSmith 按需开启。密钥只放在 `production.env`，该文件已被 Git 忽略。
首版保持 `RAG_MODE=lexical`，先验证部署和提醒稳定性。

## 5. 构建与启动

```bash
docker compose \
  --env-file deploy/tencent/production.env \
  -f deploy/tencent/compose.yaml \
  up -d --build
```

检查容器健康与证书日志：

```bash
docker compose \
  --env-file deploy/tencent/production.env \
  -f deploy/tencent/compose.yaml \
  ps

docker compose \
  --env-file deploy/tencent/production.env \
  -f deploy/tencent/compose.yaml \
  logs --tail=100 healthos caddy
```

访问 `https://你的域名`，输入 Basic Auth 用户名和密码。

## 6. 上线验收

1. 新建一条喝水记录并确认；
2. 刷新页面，确认对话和记录仍存在；
3. 执行下面的命令重启应用，确认数据仍存在；

```bash
docker compose \
  --env-file deploy/tencent/production.env \
  -f deploy/tencent/compose.yaml \
  restart healthos
```

然后继续验收：

4. 创建两分钟后的飞书提醒，核对发送目标和内容后确认；
5. 到时检查飞书消息和提醒状态；
6. 在运行证据页确认没有 Webhook、API Key、确认令牌和健康原文；
7. 使用手机网络访问，检查对话输入、确认按钮和今日记录布局。

## 7. 备份

运行期间可以创建事务一致的 SQLite 备份：

```bash
docker compose \
  --env-file deploy/tencent/production.env \
  -f deploy/tencent/compose.yaml \
  exec healthos python scripts/backup_sqlite.py \
  --database /app/runtime/healthos.db \
  --output-dir /app/runtime/backups
```

备份位于 `deploy/tencent/runtime-data/backups/`。建议同时使用腾讯云快照，并定期把
SQLite 备份复制到另一处受控存储。服务器磁盘和快照都不是多用户权限系统。

## 8. 更新和回滚

先备份，再更新：

```bash
git pull --ff-only
docker compose \
  --env-file deploy/tencent/production.env \
  -f deploy/tencent/compose.yaml \
  up -d --build
```

代码容器重建不会删除 `runtime-data`。回滚时切回已知可用 Git commit，再重新构建。
恢复数据库属于覆盖操作：必须先停止应用、额外保存当前数据库，再替换
`runtime-data/healthos.db`。

## 8.5 可选：开启图片食物识别

镜像里没有检测权重——本项目是 MIT，而 Ultralytics YOLOv8 的权重是 AGPL-3.0，
不随镜像分发。构建阶段也不做导出：那会给每次构建多下载约 2GB 的 torch。

在**宿主机**导出一次，放进已经挂载的持久目录即可：

```bash
# 在一个临时虚拟环境里导出，不污染服务器运行环境
python3 -m venv /tmp/yolo-export
/tmp/yolo-export/bin/pip install ultralytics
/tmp/yolo-export/bin/python scripts/prepare_food_detection_model.py

# runtime/ 已经通过 ./runtime-data:/app/runtime 挂载进容器
mkdir -p deploy/tencent/runtime-data/models
cp runtime/models/yolov8n.onnx deploy/tencent/runtime-data/models/
cp runtime/models/detection_manifest.json deploy/tencent/runtime-data/models/
rm -rf /tmp/yolo-export
```

然后在 `production.env` 里设置 `MEAL_DETECTION_MODE=onnx` 并重启：

```bash
docker compose --env-file production.env -f compose.yaml up -d
```

验收：上传一张苹果或香蕉的照片，"食物名称"应当被自动填好，"估计份量"仍然空白。
权重没放对时页面会说明缺什么、怎么补，手填链路照常可用。

## 9. 当前边界

- Basic Auth 保护整个站点，但所有访问者仍共享同一个 `local-demo-user`；
- SQLite 适合当前单用户、单实例作品集，不适合多实例 SaaS；
- 服务器停机期间提醒无法准时发送，恢复后才会继续处理待发送任务；
- 2 核 4GB 适合 Lexical RAG 和 YOLO nano 级 CPU 推理，不适合大型视觉模型或高并发；
- Hybrid RAG 和 YOLO 上线前应重新测量内存、冷启动与图片响应时间；
- 图片识别只认 COCO 的 10 个食物类，中餐菜品识别不到，不能对外宣称通用识别。
