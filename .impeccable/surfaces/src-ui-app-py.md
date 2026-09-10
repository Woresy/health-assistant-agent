---
version: 1
slug: "src-ui-app-py"
primary_target: "src/ui/app.py"
related_targets: ["src/ui/theme.css"]
---

# HealthOS Gradio App

Mode: Operate

Audience: 想改善习惯但不愿维护复杂表单的学生与年轻职场人。

Job: 在 30 秒内完成记录，判断今天还差什么，并掌控目标、提醒和数据。

Constraints: 保留现有功能、回调、确认机制、本地持久化与医疗边界。

## Direction contract

THESIS: 健康工作台把每天的健康事项变成连续任务流，拒绝八个功能入口平铺的原型结构。

OWN-WORLD: 冷白工作面、墨绿导航轨、青绿主操作、荧光黄绿只提示下一步；卡片 14px，控件 10px。

STORY: 用户先看今天的判断，再记录或处理确认，随后回看趋势并安排提醒。

FIRST VIEWPORT: 左侧窄导航固定系统地图；右侧顶部显示日期与主记录按钮，主体为今日判断、下一步和最近记录。

FORM: 健康工作台，排序第 7，seed 3a3628a0。签名交互是从今日记录直接进入对话编辑并保留上下文；动效只用 160-220ms 状态过渡表达按下、选中和确认完成。

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
