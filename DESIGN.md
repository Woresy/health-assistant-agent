---
name: "HealthOS · 小满健康助手"
description: "平静、决策优先、由用户掌控的个人健康工作台"
colors:
  workbench-ink: "#17253b"
  ink-soft: "#465267"
  muted-text: "#707988"
  action-blue: "#173e68"
  action-blue-hover: "#24567e"
  signal-coral: "#d96555"
  signal-coral-soft: "#f9e8e3"
  workbench-canvas: "#e7e9e7"
  workbench-surface: "#fbfcfa"
  secondary-surface: "#f2f4f1"
  card-white: "#ffffff"
  divider: "#d7dcd8"
  divider-strong: "#b9c2be"
  danger-surface: "#fff3ef"
  danger-text: "#9d5547"
typography:
  display:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
    fontSize: "38px"
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
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.65
    letterSpacing: "normal"
  label:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
    fontSize: "10px"
    fontWeight: 750
    lineHeight: 1.5
    letterSpacing: "0.08em"
rounded:
  compact: "8px"
  control: "9px"
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
    backgroundColor: "{colors.action-blue}"
    textColor: "{colors.card-white}"
    rounded: "{rounded.control}"
    padding: "9px 14px"
    height: "42px"
  button-primary-hover:
    backgroundColor: "{colors.action-blue-hover}"
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
    backgroundColor: "{colors.secondary-surface}"
    textColor: "{colors.workbench-ink}"
    rounded: "{rounded.control}"
  nav-item-active:
    backgroundColor: "#31425a"
    textColor: "{colors.card-white}"
    rounded: "{rounded.control}"
    padding: "9px 12px"
    height: "40px"
  status-chip:
    backgroundColor: "{colors.signal-coral-soft}"
    textColor: "{colors.signal-coral}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "5px 8px"
---

# Design System: HealthOS · 小满健康助手

## Overview

**Creative North Star: "个人健康工作台"**

HealthOS 的视觉系统像一份安静、可靠的身体观察手册：冷白纸张承载事实，午夜蓝建立结构与主操作，珊瑚色只标出需要注意的状态。它把注意力放在“现在应判断什么、下一步做什么”，而不是把健康数据堆成令人紧张的专业仪表盘。

系统的气质平静、具体且有边界。信息密度可以高，但通过稳定分组、短标签、轻分隔与清楚的确认状态维持可扫描性；本地保存、写入前确认和可纠正性应在关键状态中持续可见。主要用户希望在很短时间内完成记录，因此控件保持熟悉、目标明确，不用装饰制造额外认知负担。

**Key Characteristics:**

- 决策优先：先突出判断、下一步与待确认动作，再提供完整证据。
- 冷静可信：午夜蓝、冷白表面和短促反馈避免医疗化与焦虑感。
- 本地可控：确认、撤销、状态和数据边界使用明确而非戏剧化的视觉语言。
- 工作台结构：稳定的系统导航、工作面和工具组件形成连续操作环境。
- 固定浅色：当前实现只提供浅色视觉，以避免宿主暗色继承造成局部黑底与不可读内容。

## Colors

配色以午夜蓝建立系统骨架和主操作，以克制珊瑚色表达注意、焦点和关键状态；大面积区域保持冷白与中性灰。

### Primary

- **工作台午夜蓝**：用于导航轨、最高层级文字和主按钮，承担产品的结构感与可信度。
- **行动蓝**：用于主要按钮和选中状态；较亮蓝只用于悬停与强调状态。

### Secondary

- **信号珊瑚**：只用于注意、焦点和关键状态，不承担大面积背景。

### Tertiary

- **珊瑚柔面**：用于“下一步”和需要关注的轻量状态面，其稀缺性帮助用户快速定位。

### Neutral

- **工作台画布**：浏览器边缘和应用外层底色，将浅色应用壳与页面背景区分开。
- **冷白工作面**：页面主工作区和壳体表面，避免医院式纯白的生硬感。
- **卡片白**：可操作卡片与数据容器的清晰底面。
- **字段冷白**：输入、聊天工作面与轻量次级区域。
- **次级控制面**：次要按钮、面板和交替表格行的安静底色。
- **柔和蓝灰**：次级标题、表头和弱化但仍重要的文本。
- **静音中灰**：说明、元数据和占位文字；不要用于关键结论。
- **中性分隔线**：界定容器、表格与输入，不以高对比描边包围所有内容。

**The Coral Means Attention Rule.** 珊瑚色只表示注意、焦点或关键状态，不用于普通装饰和大面积品牌铺色。

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

系统采用“导航轨 + 工作面 + 页边笔记”的三栏壳体。桌面端左侧导航固定 252px，中部工作面弹性伸缩，右侧今日摘要固定 308px，顶栏高 68px；常规页面内边距为 30–34px。内部布局以 8–22px 的紧凑节奏组织关联元素，以 34px 区分页面级边界。

默认入口是“今日观察”，同一工作面先呈现当天事实，再承接自然语言记录与查询；“对话”负责创建、切换和继续历史上下文。草稿留在对话中确认，确认后的事实、目标差距与汇总由今日工作面承接。快捷语句必须触发真实 Agent 链路，不能只是填充输入框或作为装饰示例。

1180px 以下隐藏右侧页边笔记，并将导航收窄为 224px。720px 以下隐藏桌面侧栏，启用固定底部五个主导航和顶栏“更多”入口；页面内边距缩为 16px，多列指标改为单列，宽表只在自身容器内横向滚动。响应式变化保持任务顺序，不隐藏记录、确认或今天状态。

**The Workbench Map Rule.** 系统级导航必须稳定呈现信息地图；页面内容可以变化，但不能把所有能力重新铺成无层级入口。

## Elevation & Depth

系统以色调分层和细分隔线为主，默认卡片保持平坦。只有承载核心判断的摘要容器使用一层低饱和环境阴影（`0 14px 34px rgba(23, 37, 59, .08)`），用于把决策从工作面轻轻抬起；按钮、导航和普通卡片不靠阴影表达可点击性。

### Shadow Vocabulary

- **决策浮层**（`0 14px 34px rgba(23, 37, 59, .08)`）：只用于高优先级判断摘要，不扩散到普通容器。
- **选中内描边**（`inset 0 0 0 1px rgba(23, 62, 104, .10)`）：在浅色选中状态需要额外边界时使用。

**The Flat-by-Default Rule.** 普通表面以背景、边框和间距分层；阴影只服务于决策优先级，不作为通用卡片装饰。

## Shapes

形状语言是克制的柔和矩形。操作控件以 10px 圆角保持触感与精确度，信息内嵌面多用 12px，主要卡片统一为 14px，应用壳体使用 16px；药丸形只用于状态、当前选择和极短标签。圆形只承担状态点、进度提示或单字符标识，不发展为装饰图案。

## Components

### Buttons

- **Shape:** 紧凑柔和矩形（10px），最小高度 42px，字重约 680。
- **Primary:** 行动蓝底、白字和同色边框；用于当前流程唯一的主动作。
- **Hover / Focus:** 悬停提亮为行动蓝悬停色；键盘焦点使用半透明行动蓝 3px 外轮廓并偏移 2px；按下缩放至 0.98，状态过渡为 160ms。
- **Secondary / Danger:** 次级按钮使用次级冷白底、蓝灰文字和中性边框；取消或破坏性动作使用浅暖红底与暗红字，不用饱和红制造恐慌。

### Chips

- **Style:** 状态与当前选择使用珊瑚柔面或次级冷白底、对应深色文字和全圆药丸形。
- **State:** 珊瑚药丸只用于需要注意的状态；短状态文本不得仅依靠颜色表达含义。

### Cards / Containers

- **Corner Style:** 主要卡片为 14px，内嵌信息面为 11–12px。
- **Background:** 普通卡片为白色，次级工作面为冷白或浅中性灰。
- **Shadow Strategy:** 默认无阴影；只有核心决策摘要使用“决策浮层”。
- **Border:** 1px 中性线用于需要明确归属的卡片、表格和字段。
- **Internal Padding:** 紧凑卡片为 14px，标准卡片为 18px，重点任务面可扩至 22px。

### Inputs / Fields

- **Style:** 字段冷白底、中性描边和 10px 圆角，正文为工作台午夜蓝，占位文字为静音中灰。
- **Focus:** 与按钮共享行动蓝 3px 外轮廓和 2px 偏移，保证键盘路径明确。
- **Error / Disabled:** 错误应同时写明是否已写入与下一步；禁用态降至 48% 不透明度并移除阴影。

### Navigation

桌面导航使用午夜蓝轨道，默认项为浅蓝灰文字，悬停以半透明白提亮，当前项以较亮蓝面和白字明确定位。导航分组使用紧凑、稍宽字距的小标签；1180px 以下收窄导航，720px 以下转为固定底部导航，溢出入口使用 30px 可见按钮置于 44px 点击区域内。

### Confirmation Card

确认卡是系统的签名状态组件：14px 圆角、冷白底与珊瑚注意面，以及清楚的标题—摘要—后果结构。待确认、变更前后与删除风险必须同时通过文本和布局表达；确认完成可使用 200ms 的轻微上移淡入，但减少动态效果偏好下应禁用。

## Do's and Don'ts

### Do:

- **Do** 先用一个清晰结论或下一步建立视觉层级，再展开指标、证据和历史。
- **Do** 在写入、修改、删除和提醒动作附近持续显示确认状态、影响范围与本地控制信息。
- **Do** 使用 14px 卡片、10px 控件和 160–220ms 状态过渡维持一致的工作台触感。
- **Do** 在桌面到移动端的转换中保持记录、确认和今日状态可达。
- **Do** 让浅色与暗色宿主 token 继续映射到同一浅色系统，直至真正设计并验证暗色主题。

### Don't:

- **Don't** 把界面做成充满仪表、图表和告警色的医疗或专业运动仪表盘。
- **Don't** 用珊瑚色装饰普通卡片；它只服务于注意、焦点和关键状态。
- **Don't** 依赖颜色单独传达成功、风险、删除或是否已写入。
- **Don't** 引入未经交付的网络字体，或声称系统已有品牌字体资产。
- **Don't** 让宿主的暗色模式自动覆盖当前组件；已知结果是局部黑底和不可读内容。
