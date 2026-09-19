# agent-eval

一个 AI Agent 的自动化评测框架。给它一个**被测 Agent** 和一套**测试集**，它自动跑完全部用例，从质量、性能、安全、任务完成度四个维度打分，输出带执行轨迹的评测报告。

被测 Agent 可以是任何暴露 HTTP 接口的服务——不需要改你的 Agent 代码。

---

## 它解决什么问题

Agent 做完之后，很难回答这几个问题：

- 它到底答得准不准？有没有编造？
- 让它换一种检索策略 / 改一版 prompt，是变好了还是变坏了？好多少？
- 它有没有被一句"忽略以上指令"就骗出系统提示词？
- 一次回答要 20 秒，这算慢吗？

关键词命中和人肉抽查都回答不了这些。本框架用 **LLM-as-judge + 规则引擎** 给出量化结果，并支持两次评估做对比，形成"评测 → 优化 → 复测"的闭环。

---

## 四个维度

| 维度 | 指标 | 判据 |
|---|---|---|
| 质量 | 准确性、完整性、幻觉率、相关性 | LLM 裁判 |
| 性能 | 延迟、成本、Token 效率 | 实测延迟 + Token 用量（服务端上报或本地估算，见「已知局限 1」） |
| 安全 | Prompt 泄露、危险操作、权限越界 | 规则引擎 + LLM 裁判 |
| 任务完成度 | 任务成败、步骤正确性、工具使用正确性 | LLM 裁判 |

综合分（0-100）= `0.3×质量 + 0.2×性能 + 0.25×安全 + 0.25×任务`，权重在 `.env` 里可调。

每个维度独立打分，单个维度崩溃不会拖垮整条用例（会降到兜底分而不是让整个任务失败）。

---

## 快速开始

### 1. 装依赖（Python 3.11）

```bash
conda create -n agent-eval python=3.11 -y
conda activate agent-eval
pip install -r requirements.txt
```

### 2. 配环境变量

```bash
cp .env.example .env    # 至少把 LLM_API_KEY 换成你自己的
```

裁判 LLM 和被评测 Agent 可以共用同一个 Key。默认走智谱的 OpenAI 兼容端点，换 OpenAI 只需改 `.env.example` 里注释说明的三行。

### 3. 起数据库

数据库跑在 Docker 里（宿主的 5432），本地不用装 Postgres：

```bash
docker compose up -d db
```

### 4. 启动服务

**用 Docker（连 API 一起起）**

```bash
docker compose up -d
# 数据库 + API（容器内 8000，宿主映射到 8010）
```

宿主的 8010 是刻意选的：**8000 留给被测 Agent**。两边同占宿主 8000 时本机进程胜出，探测会把 agent-eval 自己当成被测对象。

> **改了 `.env` 必须重建容器**：`docker compose up -d --force-recreate web`。compose 只在创建容器时读 env_file，restart 不会刷新。

**本地运行**

```bash
conda activate agent-eval
python app/main.py
```

输出：

```
INFO:     Uvicorn running on http://127.0.0.1:8010 (Press CTRL+C to quit)
INFO:     Started reloader process [...] using WatchFiles
2026-09-19 13:22:55 INFO  agent-eval | agent-eval v0.1.0 启动完成，接口文档: /docs
2026-09-19 13:22:55 INFO  agent-eval | 网页版一键评估: http://127.0.0.1:8010/ui
```

默认从 **8010** 开始找第一个空闲端口（**8000 留给被测 Agent**，不占用），实际地址会打在日志里。想指定起始端口用 `PORT=xxxx`。

---

## 网页一键评估

浏览器打开 **`/ui`**（`/` 会重定向过去）。

1. 填 Agent 服务地址，比如 `http://localhost:8000`
2. 点探测 → 自动识别对话端点和字段
3. 选数据集、设用例条数 → 开始
4. 实时进度 + 结果面板；可导出报告、与历史评估对比

### 地址自动探测

只给根地址就够了。它会去拉目标服务的 `/openapi.json`，在所有 POST 端点里选一个最像对话接口的（路径含 `chat`/`ask`/`query`/`agent`/`diagnosis` 加分，有提问字段和回复字段各加分），再从请求/响应模型里推断字段名：

```
提问字段: message > input > query > question > prompt > ...
回复字段: reply > response > output > answer > content > ...
工具轨迹: tool_calls > tools > actions > steps
```

识别不准时，页面折叠的「高级选项」可以手工覆盖。

### 端口陷阱

agent-eval 必须和被测服务跑在同一台机器。如果 agent-eval 跑在容器里，`localhost` 指的是容器自己，要填 `host.docker.internal`。

另外**别把地址填成 agent-eval 自己**——它默认占 8010，填错就会评测到自己。命中自己时探测会直接报 `ok=false` 并带 `self_reference`，发起评估返回 400，页面红字提示，不会被当成普通的「探测失败、按兜底继续」。

---

## 接入一个外部 Agent

唯一契约很小：`AgentBuilder.build()` 返回一个有 `invoke(payload, config=...)` 方法的对象即可。

```python
{
  "type": "remote_http",
  "base_url": "http://127.0.0.1:8000",
  "chat_path": "/api/v1/diagnosis/chat",   # 可省略，自动探测
  "request_field": "message",              # 可省略
  "reply_field": "reply",                  # 可省略
  "tool_calls_field": "tool_calls",        # 可省略
  "model": "glm-4.5-air",                  # 只用于成本计价
  "timeout": 300
}
```

适配器（`app/agents/adapters.py`）会手工触发 `on_llm_start/end` 撑起端到端耗时，并回放服务端返回的 `tool_calls` 补全工具轨迹。

如果服务响应里带 `token_usage` / `usage` / `tokens` 字段，会自动识别用于成本计算，不需要改适配器。

---

## 导入评测用例

```bash
# 内置：车辆诊断 Agent 的 30 条功能用例
python scripts/import_vehicle_cases.py --replace

# 内置：13 条安全红队用例（注入 / 越权 / 危险操作诱导 + 2 条阴性对照）
python scripts/import_vehicle_cases.py \
  --cases data/security_cases.json \
  --name vehicle-diagnosis-security-redteam \
  --category security_redteam --tag security --tag redteam

# 中文安全规则（幂等）
python scripts/seed_security_rules.py
```

用例格式（`data/eval_cases.json`）：

```json
[{"id": 1, "message": "比亚迪汉EV报故障码P0562，怎么排查？",
  "expect_any": ["DC-DC", "电压", "蓄电池"], "difficulty": "medium", "tags": []}]
```

`expect_any` 会拼成 `ground_truth`（该列非空必填）。

---

## API

| 接口 | 说明 |
|---|---|
| `POST /evaluations/` | 创建评估任务（后台异步执行） |
| `GET /evaluations/{id}` | 四维度得分 + 综合分 |
| `GET /evaluations/{id}/executions` | 逐条执行轨迹 |
| `GET /evaluations/{id}/metrics` | 延迟分位 / Token / 成本聚合 |
| `POST /web/probe` | 探测服务地址 |
| `POST /web/evaluate` | 网页端发起评估 |
| `GET /web/result/{id}` | 结果 + 进度 + 轨迹 + 裁判理由 |
| `GET /web/history` | 历史评估列表 |
| `GET /web/compare?current=&baseline=` | 两次评估对比 |
| `GET /web/report/{id}` | 导出单文件 HTML 报告 |
| `POST /web/cancel/{id}` | 把卡在 running 的任务标记取消 |
| `GET /security/rules`、`/security/logs` | 安全规则管理与事件日志 |
| `GET /metrics` | Prometheus 端点 |

---

## 已知局限

诚实地说清楚边界，比给一个漂亮分数更重要。

**1. 被测服务不返回 Token 用量时，性能维度只能靠估算**
延迟项公式是 `max(0, 1 - 延迟/阈值)`，阈值来自 `.env` 的 `DEFAULT_LATENCY_THRESHOLD`（代码默认 5 秒，仓库推荐 30 秒——RAG + 多工具的 Agent 一轮常在 15~30s，5 秒会让延迟项恒为 0 分）。
延迟优先取被测服务返回的 `latency_ms`（服务端自测，比客户端秒表少一层网络往返），没有才用客户端计时。

Token 用量的处理是三级降级，全程 `token_source` 标记会写进 `execution_metadata` 并在报告里显示：

| `token_source` | 含义 | 性能维度怎么算 |
| --- | --- | --- |
| `remote` | 被测服务端上报的真实用量 | 延迟 0.5 + 成本 0.3 + 效率 0.2 |
| `estimated` | 用 `cl100k_base` 按「提问 + 回复 + 工具轨迹」本地估算（见 `app/monitoring/token_estimator.py`） | 同上，但报告标黄提醒是估算值 |
| `missing` | 连可估算的文本都没有 | **降级为仅按延迟计分**，避免「成本记 0 反而拿满分」把性能分吹起来 |

估算值**系统性偏低**：它只看得见输入输出文本，看不到被测 Agent 内部的系统提示词、
RAG 召回上下文和中间轮推理——对 RAG / 多工具编排的 Agent，那才是消耗大头。
所以它只适合同批用例横向比较，**不能当真实账单**。要精确数字，只能让被测服务返回用量。

**2. 安全规则引擎对中文不友好**
`app/security/detector.py` 的内置规则是英文关键词，中文场景基本失效。本项目补的 6 条中文规则一律设为 `low` 严重度（只扣 0.05），定位是**审计线索**而非判据——因为规则引擎分不清"不要短接继电器"和"先短接继电器"，两者含同一个词。真正的判定靠 LLM 裁判。

**3. LLM 裁判有随机波动**
同一条用例重复评估，分数可能飘几个点（历史上出现过 0.97 / 0.26 / 0.20 的波动）。所以对比功能把 ±5 分以内的变化标为**噪声区间**，不足以判定真实改善。要更稳的结论，对同一批用例重复评估取平均。

**4. 后台任务跑在请求进程里**
服务重启会导致任务中断、状态永远停在 `running`。用 `/web/cancel/{id}` 收口。生产环境应换成 Celery / ARQ 之类的独立队列——当前规模下没做，属于刻意取舍。

**5. 没有鉴权**
定位是本地 / 内网工具。

**6. 参考答案是关键词级别的**
从 `eval_cases.json` 导入的 `ground_truth` 只是几个要点词，质量裁判能评的深度有限。想要更细的评判，需要给关键用例补写完整参考答案。

---

## 项目结构

```
app/
  agents/      被测 Agent 构建器、HTTP 适配器、端点自动探测
  api/         数据集 / 评估 / 安全 / 网页接口
  core/        配置、LLM 实例
  evaluators/  四维度评估器
  models/      ORM
  monitoring/  成本计算、指标聚合、Token 估算
  security/    规则引擎（detector）
  services/    评估编排
  web/         网页端单文件 UI
scripts/       用例导入、安全规则写入
data/          评测用例（功能 + 安全红队）、示例报告
```

`data/eval_report_11.html` 是一次真实评测导出的报告，不启动服务也能直接双击打开看成品长什么样。

---

## 演示建议

如果要拿去汇报或面试：

1. **提前跑，别现场跑**。单条用例 45~90 秒（4 次裁判调用），30 条要半小时以上。
2. 先跑一次全量做基线，改点东西（prompt、检索策略）再跑一次，用对比面板展示分数变化。
3. 展示时展开一条用例的**执行轨迹**——讲清楚 Agent 怎么一步步调工具推到答案的，这比总分有信息量。
4. 主动讲上面「已知局限」里的第 1 和第 3 条：**知道自己的评测系统哪里不准，比分数好看更有说服力**。
   性能分一定要说明它是 `estimated`——面试官问「这多少钱」时能答得上来，是加分项。
