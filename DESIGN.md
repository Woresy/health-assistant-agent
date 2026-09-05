---
name: "HealthOS · 小满健康助手"
description: "平静、决策优先、由用户掌控的个人健康工作台"
colors:
  workbench-ink: "#17382f"
  ink-soft: "#31564b"
  action-green: "#2f765e"
  action-green-deep: "#225846"
  confirmation-mint: "#dff0df"
  next-step-lime: "#dceea8"
  workbench-canvas: "#e9efe9"
  workbench-surface: "#f8faf8"
  warm-surface: "#fffdf7"
  card-white: "#ffffff"
  secondary-surface: "#f4f7f1"
  field-surface: "#f7f9f5"
  divider-green: "#d6e1d8"
  muted-text: "#53675d"
  danger-surface: "#fff3ef"
  danger-text: "#9d5547"
typography:
  display:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
    fontSize: "30px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.025em"
  headline:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
    fontSize: "20px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.025em"
  title:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
    fontSize: "16px"
    fontWeight: 700
    lineHeight: 1.35
    letterSpacing: "-0.015em"
  body:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.65
    letterSpacing: "normal"
  label:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
    fontSize: "11px"
    fontWeight: 750
    lineHeight: 1.5
    letterSpacing: "0.08em"
rounded:
  compact: "8px"
  control: "10px"
  inset: "12px"
  card: "14px"
  shell: "16px"
  pill: "999px"
spacing:
  micro: "4px"
  xs: "8px"
  sm: "10px"
  md: "14px"
  lg: "18px"
  xl: "22px"
  page: "34px"
components:
  button-primary:
    backgroundColor: "{colors.action-green}"
    textColor: "{colors.card-white}"
    rounded: "{rounded.control}"
    padding: "9px 14px"
    height: "42px"
  button-primary-hover:
    backgroundColor: "{colors.action-green-deep}"
    textColor: "{colors.card-white}"
    rounded: "{rounded.control}"
  button-secondary:
    backgroundColor: "{colors.secondary-surface}"
    textColor: "{colors.ink-soft}"
    rounded: "{rounded.control}"
    padding: "9px 14px"
    height: "42px"
  button-danger:
    backgroundColor: "{colors.danger-surface}"
    textColor: "{colors.danger-text}"
    rounded: "{rounded.control}"
    padding: "9px 14px"
    height: "42px"
  card:
    backgroundColor: "{colors.card-white}"
    textColor: "{colors.workbench-ink}"
    rounded: "{rounded.card}"
    padding: "18px"
  input:
    backgroundColor: "{colors.field-surface}"
    textColor: "{colors.workbench-ink}"
    rounded: "{rounded.control}"
  nav-item-active:
    backgroundColor: "{colors.next-step-lime}"
    textColor: "{colors.workbench-ink}"
    rounded: "{rounded.control}"
    padding: "9px 12px"
    height: "40px"
  status-chip:
    backgroundColor: "{colors.confirmation-mint}"
    textColor: "{colors.action-green-deep}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "5px 8px"
---

# Design System: HealthOS · 小满健康助手

## Overview

**Creative North Star: "个人健康工作台"**

HealthOS 的视觉系统像一张安静、可靠的个人工作台：冷白工作面承载事实，深墨绿结构提供方向，青绿操作色让可执行动作清楚可见。它把注意力放在“现在应判断什么、下一步做什么”，而不是把健康数据堆成令人紧张的专业仪表盘。

系统的气质平静、具体且有边界。信息密度可以高，但通过稳定分组、短标签、轻分隔与清楚的确认状态维持可扫描性；本地保存、写入前确认和可纠正性应在关键状态中持续可见。主要用户希望在很短时间内完成记录，因此控件保持熟悉、目标明确，不用装饰制造额外认知负担。

**Key Characteristics:**

- 决策优先：先突出判断、下一步与待确认动作，再提供完整证据。
- 冷静可信：低饱和绿系、冷白表面和短促反馈避免医疗化与焦虑感。
- 本地可控：确认、撤销、状态和数据边界使用明确而非戏剧化的视觉语言。
- 工作台结构：稳定的系统导航、工作面和工具组件形成连续操作环境。
- 固定浅色：当前实现只提供浅色视觉，以避免宿主暗色继承造成局部黑底与不可读内容。

## Colors

配色以深墨绿建立系统骨架，以克制青绿表达操作，以荧光感很低的黄绿色只标记当前选择或明确下一步；大面积区域保持冷白和淡灰绿。

### Primary

- **工作台墨绿**：用于导航轨、最高层级文字和深色图标底，承担产品的结构感与可信度。
- **行动青绿**：用于主要按钮、进度和焦点反馈；更深的行动青绿只用于悬停与强调状态。

### Secondary

- **确认薄荷绿**：用于已选择、已确认、记忆或温和状态面，不承担主操作。

### Tertiary

- **下一步青柠**：用于活动导航项和少量“下一步”提示。它的稀缺性让用户立刻找到当前任务。

### Neutral

- **工作台画布**：浏览器边缘和应用外层底色，将浅色应用壳与页面背景区分开。
- **冷白工作面**：页面主工作区和壳体表面，避免医院式纯白的生硬感。
- **卡片白**：可操作卡片与数据容器的清晰底面。
- **字段浅绿白**：输入、聊天工作面与轻量次级区域。
- **次级控制面**：次要按钮、面板和交替表格行的安静底色。
- **柔和墨绿**：次级标题、表头和弱化但仍重要的文本。
- **静音绿灰**：说明、元数据和占位文字；不要用于关键结论。
- **绿灰分隔线**：界定容器、表格与输入，不以高对比描边包围所有内容。

**The Lime Means Next Rule.** 下一步青柠只表示当前选择、直接下一步或关键方向，不用于普通装饰和大面积品牌铺色。

**The Fixed-Light Rule.** 当前实现必须把浅色与暗色宿主 token 都映射到同一套浅色值；这是为修复浏览器或 Gradio 暗色继承导致的黑色不可读区域，不代表已实现暗色主题。

## Typography

**Display Font:** 中文系统 UI 字体栈（`-apple-system`、`BlinkMacSystemFont`、`Segoe UI`、`PingFang SC`、`Microsoft YaHei`、`sans-serif`）  
**Body Font:** 同一中文系统 UI 字体栈  
**Label/Mono Font:** 无独立字体；数字通过等宽数字特性改善对齐

**Character:** 这是一个诚实依赖操作系统的中文界面字体系，没有捆绑或下载品牌字体。字重、紧凑字距和清晰字号层级负责营造稳健的工作台感，同时保持跨平台可用性。

### Hierarchy

- **Display**（700，30px，1.2）：页面级判断或工作区标题；移动端降为 23px，并限制行宽以保持动作导向。
- **Headline**（700，20px，1.2）：应用品牌信息与较高层级标题。
- **Title**（700，16px，1.35）：卡片标题、确认摘要和局部结论。
- **Body**（400，15px，1.65）：输入和主要正文；解释性段落通常收窄到约 70ch。
- **Label**（750，11px，1.5，必要时 0.08em）：导航分组、状态与元数据。10px 仅用于辅助性的非核心微标签，不用于主要交互文案。

**The System-Font Honesty Rule.** 不指定并未随产品交付的定制字体；新增界面继续使用现有中文系统 UI 栈。

## Layout

系统采用“导航轨 + 工作面”的工作台壳体。桌面端应用最大宽度为 1580px，左侧导航轨固定为 226px，右侧工作面可伸缩；内容区最大宽度为 1260px，常规页面内边距为 34px。内部布局以 8–22px 的紧凑节奏组织关联元素，以 34px 区分页面级边界。

默认入口是“对话”，并在日常导航中排在“今天”之前。用户可直接用一句自然语言开始记录或查询；草稿留在对话中确认，确认后的事实、目标差距与汇总由“今天”承接。快捷语句必须触发真实 Agent 链路，不能只是填充输入框或作为装饰示例。

980px 以下，侧边导航转为可横向滚动的顶部导航，工作面取消视口高度约束；多列指标和判断区收拢，筛选控件允许按可用宽度换行。680px 以下，外层留白归零、页面内边距缩为 14px、卡片标题与按钮上下重排、操作标签必要时换行、双栏卡片与指标改为单列；宽表只在自身容器内横向滚动。响应式变化保持任务顺序，不隐藏记录、确认或今天状态。

**The Workbench Map Rule.** 系统级导航必须稳定呈现信息地图；页面内容可以变化，但不能把所有能力重新铺成无层级入口。

## Elevation & Depth

系统以色调分层和细分隔线为主，默认卡片保持平坦。只有承载核心判断的摘要容器使用一层低饱和环境阴影（`0 14px 34px rgba(23, 56, 47, .08)`），用于把决策从工作面轻轻抬起；按钮、导航和普通卡片不靠阴影表达可点击性。

### Shadow Vocabulary

- **决策浮层**（`0 14px 34px rgba(23, 56, 47, .08)`）：只用于高优先级判断摘要，不扩散到普通容器。
- **选中内描边**（`inset 0 0 0 1px rgba(47, 118, 94, .08)`）：在浅色选中状态需要额外边界时使用。

**The Flat-by-Default Rule.** 普通表面以背景、边框和间距分层；阴影只服务于决策优先级，不作为通用卡片装饰。

## Shapes

形状语言是克制的柔和矩形。操作控件以 10px 圆角保持触感与精确度，信息内嵌面多用 12px，主要卡片统一为 14px，应用壳体使用 16px；药丸形只用于状态、当前选择和极短标签。圆形只承担状态点、进度提示或单字符标识，不发展为装饰图案。

## Components

### Buttons

- **Shape:** 紧凑柔和矩形（10px），最小高度 42px，字重约 680。
- **Primary:** 行动青绿底、白字和同色边框；用于当前流程唯一的主动作。
- **Hover / Focus:** 悬停加深为行动深绿；键盘焦点使用半透明青绿 3px 外轮廓并偏移 2px；按下缩放至 0.98，状态过渡为 160ms。
- **Secondary / Danger:** 次级按钮使用字段浅绿白底、柔和墨绿字和绿灰边框；取消或破坏性动作使用浅暖红底与暗红字，不用饱和红制造恐慌。

### Chips

- **Style:** 状态与当前选择使用薄荷绿或淡灰绿底、深青绿文字和全圆药丸形。
- **State:** 青柠药丸只用于直接下一步；短状态文本不得仅依靠颜色表达含义。

### Cards / Containers

- **Corner Style:** 主要卡片为 14px，内嵌信息面为 11–12px。
- **Background:** 普通卡片为白色，次级工作面为冷白或浅绿白。
- **Shadow Strategy:** 默认无阴影；只有核心决策摘要使用“决策浮层”。
- **Border:** 1px 绿灰线用于需要明确归属的卡片、表格和字段。
- **Internal Padding:** 紧凑卡片为 14px，标准卡片为 18px，重点任务面可扩至 22px。

### Inputs / Fields

- **Style:** 字段浅绿白底、绿灰描边和 10px 圆角，正文为工作台墨绿，占位文字为静音绿灰。
- **Focus:** 与按钮共享青绿 3px 外轮廓和 2px 偏移，保证键盘路径明确。
- **Error / Disabled:** 错误应同时写明是否已写入与下一步；禁用态降至 48% 不透明度并移除阴影。

### Navigation

桌面导航使用深墨绿轨道，默认项为浅灰绿文字，悬停以半透明白提亮，当前项以青柠底和墨绿字明确定位。导航分组使用紧凑、稍宽字距的小标签；980px 以下转成浅色横向导航，未选中项改用柔和墨绿保证对比度，溢出入口使用 30px 可见按钮置于 44px 点击区域内。

### Confirmation Card

确认卡是系统的签名状态组件：14px 圆角、极浅绿底和清楚的标题—摘要—后果结构。待确认、变更前后与删除风险必须同时通过文本和布局表达；确认完成可使用 200ms 的轻微上移淡入，但减少动态效果偏好下应禁用。

## Do's and Don'ts

### Do:

- **Do** 先用一个清晰结论或下一步建立视觉层级，再展开指标、证据和历史。
- **Do** 在写入、修改、删除和提醒动作附近持续显示确认状态、影响范围与本地控制信息。
- **Do** 使用 14px 卡片、10px 控件和 160–220ms 状态过渡维持一致的工作台触感。
- **Do** 在桌面到移动端的转换中保持记录、确认和今日状态可达。
- **Do** 让浅色与暗色宿主 token 继续映射到同一浅色系统，直至真正设计并验证暗色主题。

### Don't:

- **Don't** 把界面做成充满仪表、图表和告警色的医疗或专业运动仪表盘。
- **Don't** 用青柠色装饰普通卡片；它只服务于当前选择和下一步。
- **Don't** 依赖颜色单独传达成功、风险、删除或是否已写入。
- **Don't** 引入未经交付的网络字体，或声称系统已有品牌字体资产。
- **Don't** 让宿主的暗色模式自动覆盖当前组件；已知结果是局部黑底和不可读内容。
