# 实验 B 执行方案：让多个 LLM agent 做 choose-N（给另一个 Claude）

把下面分隔线之间的内容整段粘贴给负责实验 B 的 Claude。论文写作在另一个会话里进行，两边通过 git 分支交接。

---

你负责一篇 AAMAS 2027 投稿论文的一个补充实验（实验 B）。论文作者已经同意这项实验并批准了预算。截稿时间是北京时间 2026-10-09 20:00，**实验结果最晚在 10-05 22:00（北京时间）前推上来**，写作那边要用。你的任务是：实现、在 GPU 上运行、检查关卡、汇报结果。不要改动实验设计；如果设计上有问题，先停下来问。

## 1. 背景（只需要知道这些）

仓库：`https://github.com/cunicle/aamas2`。从分支 `claude/elegant-goldberg-jjk1e0` 新建分支 `exp-b-agents`，你的所有提交都推到 `exp-b-agents`。

开工前先读：
- `HANDOFF.md`：项目现状、环境、RunPod 操作注意事项（第 9 节必读）；
- `ptcdiag/data/choose.py`：choose-N 探针的数据生成；
- `scripts/choose_analysis.py`：choose-N 的分析；
- `ptcdiag/pipeline.py`、`ptcdiag/decoding/ar.py`：dLLM 和 AR 跑单条样例的入口；
- `scripts/run_minimal.sh` 中的 `choose_dream`、`choose_ar` 两个阶段。

论文研究的是：一个以扩散语言模型（dLLM）为决策核心的工具调用 agent，在一轮里同时发出多个工具调用时，各调用之间会不会协调失败。dLLM 用骨架约束解码，每个调用的参数值是被 mask 的槽。同一步提交的槽彼此看不见，就像同时行动、互不通信的多个决策者。

已有结果（`results/summary/choose.md`）：choose-N 的请求不规定 N 个调用各选哪个城市。
- 一张画布写全部 N 个调用的情况：
  - Dream 在同一步提交多个槽时（k≥2）出现重复：open 变体 100%，list 变体 10–15%；
  - 每步只提交一个 token 时（k=1），重复率为 0%；
  - LLaDA2.0（块因果，各个调用实际上轮流解码）和 Qwen（自回归）基本不重复。
- 论文的解释：同一步提交的槽 ≈ 同时行动、不通信的 agent；k=1 和块因果 ≈ 轮流行动。在 list 变体里，槽的位置（第 i 个调用）加上列表顺序可以充当约定，open 变体里则没有可以对应的东西。

**实验 B 的目的**：用真正的多个 LLM agent 做同一个任务，检验上面的类比是否成立。为此系统地操控两个因素：
1. agent 能否看到别人已经做出的调用（同时行动 vs 轮流行动）；
2. agent 是否有角色编号（匿名 vs 有编号）。

然后把结果和“一张画布写全部调用”的已有结果并排比较。

## 2. 设计

### 2.1 任务和数据
- 直接用 `data/choose.jsonl` 的 105 条（由 `ptcdiag.data.choose.generate()` 确定性生成，`run_minimal.sh` 里的 `choose_data` 会生成它）。
  - list 变体 60 条：n=2/3/4 各 20 条，每条列出 n+3 个单 token 城市；
  - open 变体 45 条：n=2/3/4 各 15 条。
- 一条题对应一个由 n 个 agent 组成的团队。每个 agent 只发出 1 个 `get_weather` 调用（选 1 个城市）。

### 2.2 四种协议（2 × 2）

| 协议 | 能看到之前的调用吗 | 有角色编号吗 |
|---|---|---|
| `sim-anon` | 否（同时行动） | 否 |
| `sim-label` | 否（同时行动） | 是（“assistant i of n”） |
| `turn-anon` | 是（轮流行动：agent i 能看到 agent 1..i−1 的调用） | 否 |
| `turn-label` | 是 | 是 |

### 2.3 每个 agent 的输入（文本必须完全照抄，便于在论文里复现）

- system 消息不变，即 `ptcdiag/prompting.py` 里的统一系统提示。
- user 消息 = 原请求文本 + 一个空行 + 下列协议说明之一，其中 `{n}`、`{i}`、`{calls}` 按实际值替换：

```text
sim-anon:   You are one of {n} assistants answering this request at the same time. Each assistant makes exactly one of the {n} calls, and the assistants cannot see each other's calls. Make your one call.
sim-label:  You are assistant {i} of {n} answering this request at the same time. Each assistant makes exactly one of the {n} calls, and the assistants cannot see each other's calls. Make your one call.
turn-anon:  You are one of {n} assistants answering this request one after another. Each assistant makes exactly one of the {n} calls. Calls made so far: {calls}. Make your one call.
turn-label: You are assistant {i} of {n} answering this request one after another. Each assistant makes exactly one of the {n} calls. Calls made so far: {calls}. Make your one call.
```

- `{calls}` 的格式：
  - 没有之前的调用时，写 `none`；
  - 否则是之前各 agent 解析出的调用，用紧凑 JSON 数组表示，例如 `[{"name": "get_weather", "arguments": {"city": "Chicago"}}]`；
  - 如果之前某个 agent 的输出解析失败，就放它的原始输出文本。
- 骨架：每个 agent 的样例只含 1 个调用，`ground_truth` 取原题的第 1 项，所以骨架里只有一个 city 槽，槽长与原实验相同（第一个允许城市的 token 数，即 1）。这样与已有的 choose-N 结果可以直接比较。open 变体同样受 1 token 槽长的限制，所以 open 只看重复率。

### 2.4 模型和解码

- **Qwen2.5-7B-Instruct**（AR agent）：走 `run_example_ar(..., mode="skeleton")`。
- **Dream-v0-Instruct-7B**（dLLM agent）：走 `run_example(..., mode="skeleton")`，k=1，confidence 顺序，不分块。槽只有 1 个 token，k 不影响结果。
- **主实验全部贪心解码**，与论文其余部分一致。
- LLaDA2.0 agent 不做。

### 2.5 可选的 B2：采样（主实验完成并汇报后再做）

1. agent：`sim-anon` 和 `sim-label`，temperature 0.7，seed 0–4，两个模型都跑。
   - Qwen 的 AR 骨架解码目前只支持贪心：给 `generate_skeleton` / `run_example_ar` 加可选的 `temperature`、`seed` 参数，默认值保持贪心。
   - **加完必须验证**：用默认参数重跑 5 条 BFCL 题，输出要与 `release/results_2026-10-03.tar.gz` 里 `results/qwen/bfcl_skeleton.jsonl` 的 `text` 逐字相同。
2. 单画布对照：Dream 在 choose-N 上以 temperature 0.7 跑 k ∈ {1,16}、seed 0–4：
   `run_dllm.py --data probe:data/choose.jsonl --mode skeleton --k 1,16 --temperature 0.7 --seeds 0,1,2,3,4`
   输出写到新文件 `results/dream/choose_sample.jsonl`，不要追加到原来的 `choose.jsonl`。

### 2.6 预先写下的预测（结果相反时原样报告，不要为了符合预测去调设置）

| 协议 | list 重复率 | open 重复率 | 对应的单画布条件 |
|---|---|---|---|
| sim-anon | 100%：贪心解码下输入相同、输出就相同，这是**理智检查** | 100%（同上） | — |
| sim-label | 低：agent i 选列表里第 i 个城市 | 高：编号没有可以对应的顺序 | Dream 同步提交（k≥2）：list 10–15%，open 100% |
| turn-anon / turn-label | 约 0% | 约 0% | Dream k=1、LLaDA2.0 块因果、Qwen 单次生成：约 0% |

如果有编号的 agent 在 open 变体里也能错开（例如约定按字母顺序选），要原样报告。那说明 agent 有一种 dLLM 的槽没有的约定，论文里的类比就只能部分成立。

## 3. 实现规格

新增文件（只加文件，不改现有文件的行为）：

1. **`ptcdiag/data/agents.py`**
   - `PROTOCOLS = {"sim-anon": (False, False), "sim-label": (False, True), "turn-anon": (True, False), "turn-label": (True, True)}`，值为 (observe, label)。
   - 2.3 节的四段说明文字，作为常量。
   - `agent_example(ex, i, n, protocol, previous)` 返回一个 `Example`：
     - `id = f"{ex.id}_a{i}"`；
     - `messages` = 一条 user 消息：原文 + `"\n\n"` + 说明；
     - `functions` 不变；`ground_truth = ex.ground_truth[:1]`；
     - `meta` 在 `ex.meta` 基础上加 `agent=i`、`protocol=protocol`；
     - `category` 不变。如果单调用的诊断出错，改用合适的类别并在提交说明里写明。
   - i 从 1 开始；`previous` 是之前各 agent 的调用列表（解析失败时为原始文本）。

2. **`scripts/run_agents.py`**
   - 参数：`--model`、`--backend {dllm,ar}`、`--data probe:data/choose.jsonl`、`--protocols`（逗号列表）、`--temperature`（默认 0）、`--seeds`（默认 0）、`--out`、`--limit`、`--dry-run`。
   - `--dry-run` 不加载模型，只打印每个 agent 的 user 文本；轮流协议里之前的调用用占位城市代替。
   - 对每条题 × 协议 × seed，按 i=1..n 依次构造并解码每个 agent。轮流协议要把前面 agent 的结果放进 `previous`；同时协议的 `previous` 永远为空。
   - 每个团队写一条 JSONL 记录：
     ```json
     {"id", "variant", "n", "listed", "protocol", "temperature", "seed", "model_id", "backend",
      "agents": [{"i", "user_text", "text", "syntax_ok", "city"}], "cities": [...]}
     ```
     `user_text` 必须原样保存，供审阅。
   - 断点续跑：跳过已经存在的 (id, protocol, temperature, seed)。

3. **`scripts/agents_analysis.py`**
   - 按 模型 × 协议 × 变体 × temperature 统计团队层面的指标，口径与 `choose_analysis.py` 一致（城市都先用 `standardize_string` 规范化）：
     - `ok`：所有城市都合法，且两两不同；
     - `duplicate`：至少两个 agent 选了同一个城市；
     - `invalid`：有城市不在允许集合里，或者解析失败；
     - `in_order`（只算 list 变体）：agent i 选的正是列表第 i 个城市（对所有 i 成立）。
   - 再按 n 拆一张表。
   - 输出 `results/summary/agents.md` 和 `.csv`。
   - 另外从每种协议各挑 2 个团队，写入 `results/summary/agents_examples.md`：完整的各 agent 输入和输出，给论文挑样例用。

4. **`tests/test_agents.py`**（CPU）：
   - `sim-anon` 下同一团队各 agent 的 user 文本完全相同；
   - `sim-label` 下只有编号不同；
   - 轮流协议里 agent i 的文本包含之前 i−1 个调用；
   - agent 样例的骨架只有一个槽，且槽长与原题相同。

5. **`scripts/run_minimal.sh`**：在 case 语句**末尾**（`summary)` 之后、`*)` 之前）新增以下阶段，不要改动已有阶段：
   - `agents_ar`：Qwen，四种协议；
   - `agents_dream`：Dream，四种协议；
   - `agents_sample`：B2 的两个 agent 模型；
   - `choose_sample_dream`：B2 的单画布对照；
   - `summary_agents`：运行 `agents_analysis.py`。

## 4. 执行步骤与关卡（每个关卡不过就停下汇报）

1. **CPU**：`python -m pytest tests -q` 全部通过，包括新测试；再用 `run_agents.py --dry-run --limit 3` 检查四种协议的提示词拼接。
2. **冒烟**（GPU）：两个模型、四种协议，各跑 3 条（list 和 open 都要有）。打印每个 agent 的 `user_text` 和输出，人工确认三点：
   - 提示拼接正确；
   - 轮流协议确实带上了之前的调用；
   - 输出是合法的单个调用。
3. **理智检查**：`sim-anon` 贪心下，同一团队各 agent 的输出必须完全相同，重复率 100%。如果不是，说明提示没有做到完全相同，是 bug。
4. **主实验**：两个模型 × 四种协议 × 105 条，贪心。然后运行 `summary_agents`。
5. **汇报**：先汇报主实验，再决定是否做 B2。B2 要在 10-05 22:00 前能完成才做。

## 5. 资源和预算

- 一张 40GB 以上的卡就够了；Dream 和 Qwen 依次加载，不需要 80GB。
- 主实验约 2,500 次很短的单调用解码（105 个团队 × 平均 3 个 agent × 4 种协议 × 2 个模型），预计几十分钟。加上安装和下载模型，**估计 1–1.5 GPU 小时**。这是估计值，冒烟时测一下速度再更新。
- 累计 GPU 时间超过 **4 小时**时，先停下来问用户。
- 跑完立即把结果拉回本地，然后停机（HANDOFF 第 9 节）。不要让 pod 空转：上一次约 7 小时的空转就是这么来的。

## 6. 交付

推到 `exp-b-agents`：
- 新代码和测试；
- `results/summary/agents.md`、`agents.csv`、`agents_examples.md`；如果做了 B2，还有 B2 的汇总。都用 `git add -f`，因为 `.gitignore` 忽略了 `/results/`；
- 原始 JSONL：打包成 `release/agents_<日期>.tar.gz` 一并提交（体积很小）。

**不要改动 `paper/` 下的任何文件，也不要改已有的结果文件**，写作那边会同时在改。

向用户汇报：
1. 主实验表（模型 × 协议 × 变体：ok / duplicate / invalid / in_order）；
2. 与预测（2.6 节）逐条对照，哪里相反；
3. 每种协议一个代表性团队的输入和输出；
4. 实际花费的 GPU 时间；
5. 任何偏离本方案的地方及原因。

## 7. 规则

- 只跑本方案里的配置；需要改设计时先问。
- 没跑出来的数字写 TBD，不要估计或外推。
- 结果和预测相反时原样报告。
- 每处代码修改单独提交，commit message 写清原因。

---
