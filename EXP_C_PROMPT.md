# 实验 C 执行方案：agent 团队做有序请求，以及收尾符对照（给另一个 Claude）

把下面分隔线之间的内容整段粘贴给负责实验 C 的 Claude。论文写作在另一个会话里进行，两边通过 git 分支交接。实验 C 沿用实验 B 的代码和约定（`EXP_B_PROMPT.md`）。

---

你负责一篇 AAMAS 2027 投稿论文的一组补充实验（实验 C）。论文作者已经批准了这组实验和它的 GPU 时间。截稿时间是北京时间 2026-10-09 20:00，**结果最晚在 10-05 22:00（北京时间）前推上来，越早越好**，写作那边 10-06 要用。你的任务是：实现、在 GPU 上运行、检查关卡、汇报结果。不要改动实验设计；如果设计上有问题，先停下来问。

## 1. 背景（只需要知道这些）

仓库：`https://github.com/cunicle/aamas2`。从分支 `claude/elegant-goldberg-jjk1e0` 的最新提交新建分支 `exp-c-agents`（它已经包含实验 B 的代码），你的所有提交都推到 `exp-c-agents`。

开工前先读：
- `HANDOFF.md`：项目现状、环境、RunPod 操作注意事项（第 9 节必读）；
- `EXP_B_PROMPT.md`：实验 B 的方案。实验 C 的协议文本、输出格式、关卡和规则都与它一致；
- `ptcdiag/data/agents.py`、`scripts/run_agents.py`、`scripts/agents_analysis.py`：实验 B 的实现，尽量复用；
- `ptcdiag/decoding/constraints.py`：开头的说明，以及 `build_skeleton`、`SkeletonConstraint`（`_end_mask`、`after_commit`、`cuts`）、`oracle_lengths`、`swapped_lengths`、`sibling_groups`；
- `ptcdiag/pipeline.py` 的 `run_example`（参数 `lengths`、`length_mode`），`ptcdiag/decoding/ar.py` 的 `run_example_ar`；
- `scripts/length_prior.py`：teacher-forced 探针；
- `ptcdiag/eval/taxonomy.py`：`diagnose`、`value_ok`、`is_cross_call`，以及标签 `duplicate_call`；
- `ptcdiag/eval/bfcl_checker.py`：`simple_function_checker`。

**论文研究什么**：一个以扩散语言模型（dLLM）为决策核心的工具调用 agent，在一轮里同时发出多个工具调用时，各调用之间怎样协调。dLLM 用骨架约束解码：调用数、函数名、参数名、括号和引号都事先写好，只有参数值是被 mask 的槽。同一步提交的槽彼此看不见，就像同时行动、互不通信的决策者。

**已有结果**（BFCL 指 BFCL v4 的 parallel 和 parallel_multiple，共 400 条）：
1. **有序请求、一张画布**：模型几乎总让第 i 个调用取第 i 个被提到的实体（95–97%），与每步提交几个 token 无关。
2. **对称请求（choose-N）**：
   - 一张画布上，同一步提交的槽会撞车。
   - 实验 B 的 agent 团队：同时行动就撞车；prompt 里写编号（“assistant i of n”）几乎没用；轮流行动从不撞车。
3. **槽长**：
   - 每个槽多一个 mask，Dream 的 set accuracy 就从 89.8% 降到 15.5%（k=1）。
   - 槽长互换时，79.4% 的槽写入长度合适的兄弟值。
   - teacher-forced 探针：槽里只剩 1–2 个 mask 时，模型几乎不写收尾符。
   - 论文把这概括为“mask 是长度承诺”。

**模拟审稿指出了两个问题，实验 C 要回答它们，并补上论文 2×2 设计里空着的一格**：
- **缺口**：还没有“agent 团队 × 有序请求”的数据（C1）。
- **收尾符混淆**：现有骨架在每个字符串槽后面**预先写好了收尾引号**；数值槽后面预先写好了 `,` 或 `}`。模型要提前结束一个值，就得在槽里再写一个收尾符，剩下的位置填 EOS。于是画布上出现“值 + 收尾符 + EOS… + 预先写好的收尾符”，这是训练中没见过的形式。所以“模型不愿提前收尾”可能只是在避免连写两个收尾符，而不是把 mask 数当成承诺。C3 用一个界面变体来区分这两种解释。

## 2. 设计

### 2.1 数据

BFCL v4 的 parallel + parallel_multiple，用现有加载器 `load_examples("bfcl:parallel,parallel_multiple")`（会自动下载到 `data/bfcl`）。所有 400 条都只有一条 user 消息。

- **团队对称子集（C1 用，104 条）**：
  - **定义**：对每个函数名，这个函数的所有参考调用的单调用骨架签名都相同，并且至少有一个函数有 ≥2 个调用。
  - **签名**：`tuple(sorted((p, L[ci, p]) for p in 调用 ci 的骨架参数))`，其中 `L = oracle_lengths(tokenizer, ex)`；骨架参数就是出现在 `L` 里的参数。
  - **为什么这样选**：同一函数的各 agent 拿到的骨架完全相同，槽长不会泄露谁该取哪个实体。
  - **核对数字**：10-03 用 Dream 和 Qwen 两个分词器各算一遍，都得到同样的 104 条（parallel 84、parallel_multiple 20），共 289 个 agent；各请求的调用数为 2 个调用 57 条、3 个 21 条、4 个 24 条、8 个 2 条。
- **互换子集（C2、C3 用，165 条）**：`swapped_lengths(tok, ex) != oracle_lengths(tok, ex)` 的请求，与论文里槽长互换实验的 165 条相同；两个分词器结果一致。
- **choose-N**（C1 的正对照、C4 用）：`data/choose.jsonl`（`run_minimal.sh` 里的 `choose_data` 会生成它）。

### 2.2 C1：agent 团队做有序请求（顺序线索）

- 每条请求一个团队，n = 参考调用数。agent i（从 1 开始）负责第 i 个调用：`ground_truth = [ex.ground_truth[i-1]]`，所以它的骨架就是参考调用 i 的单调用骨架（函数名、参数名、槽长）。
- **协议**：
  - 实验 B 的四种协议，**文本一字不改**（用 `ptcdiag/data/agents.py` 的常量、`instruction()` 和 `format_calls()`）；
  - 再加一个正对照协议 `sim-rule`：把约定明确告诉 agent，看它们能不能照做。文本：
    ```text
    sim-rule:   You are assistant {i} of {n} answering this request at the same time. Each assistant makes exactly one of the {n} calls, and the assistants cannot see each other's calls. By convention, assistant {i} makes the {ordinal} of the calls, in the order in which the request mentions them. Make your one call.
    ```
    `{ordinal}` 写成 first、second、third、fourth、fifth、sixth、seventh、eighth。
- **user 消息** = BFCL 原请求 + `"\n\n"` + 协议说明；system 消息是统一系统提示。
- **轮流协议的 `previous`**：之前各 agent 解析出的单个调用，或者原始文本，规则与 B 相同。
- **条件**：团队对称子集 104 条 × 五种协议 × {Qwen（AR，`run_example_ar(..., mode="skeleton")`），Dream（`run_example`，k=1，confidence 顺序，不分块）}，exact 槽长，贪心解码。
- **choose-N 上的正对照**：list 变体 60 条 × `sim-rule` × 两个模型（choose-N 的骨架照实验 B，用 `ground_truth[:1]`）。用来回答：编号没用，是因为没有共同知道的映射，还是 agent 给了规则也照样做不到？

### 2.3 C2：长度信号对单个 dLLM agent（长度线索）

- 互换子集 165 条 × 实验 B 的四种协议 × {Dream exact、Dream swap、Qwen exact}，贪心解码。
- **swap 的做法**：先在完整请求上算 `S = swapped_lengths(tok, ex)`；agent i 的槽长 = `{(0, p): S[i-1, p]}`，即调用 i−1 的那几项，调用编号改成 0。通过 `run_example(..., lengths=..., length_mode="swap")` 传入。
- **Qwen 只做 exact**：AR 的骨架解码没有逐槽长度参数，而且它写到收尾符就停，读不到槽长信号，只作“看不到长度线索”的对照。`--backend ar --length-mode swap` 要直接报错。
- **要回答的问题**：同时行动的 Dream agent 能不能靠正确的槽长分工？错误的槽长会不会让单个 agent 去拿兄弟的值？

### 2.4 C3：收尾符对照（界面变体，Dream，一张画布）

**界面变体 `closer_in_slot`**：收尾符由模型在槽里写，骨架不再预写；值结束后用空格填充。
- **字符串槽**：骨架写到 `"param": "` 为止，**不再**在槽后预写收尾引号。下一段固定文本照旧从 `, "next":` 或 `}}` 开始。
- **非字符串槽**（数值、布尔、数组、字典）：骨架**不再**预写槽后的那个 `,`（不是最后一个参数时），或 `}}` 里的第一个 `}`（是最后一个参数时）。下一段固定文本相应改成 ` "next":` 或 `}`。
- **槽长**：变体里的 exact = 参考值的 token 数 + 1（留给收尾符）；`+s` 再加 s 个；swap = 互换后的长度 + 1。
- **结束规则**：模型必须在槽里写出收尾符（字符串是 `"`，其他类型是 `,` 或 `}`，判断规则同 `value_end`），值才算结束。之后槽里还没提交的位置立即强制填成**空格 token**（分词器里单个空格 `" "` 的 id），而不是 EOS。空格不算结束符，模型不能直接写空格来结束值。
  
  这样提前结束的值在画布上就是 `"HSBC"   , "amount": ` 这样完全合法的 JSON。
- **解码输出**：保留收尾符，去掉空格填充（保留也可以，都是合法 JSON），然后照常解析和诊断。
- **溢出**：如果模型把槽写满都没写收尾符，就算语法错误，照实记录。
- **实现要求**：给 `build_skeleton` 和 `SkeletonConstraint` 加可选参数 `closer_in_slot=False`，并在 `run_dllm.py` 和 `length_prior.py` 加 `--closer-in-slot` 开关。**默认值时所有行为必须与现在逐字节相同**（关卡 1）。
- **条件**（Dream，贪心）：

  | 槽长 | k | 请求 |
  |---|---|---|
  | exact、+1、+2、+8 | 1 | 400 |
  | exact、+1 | 16 | 400 |
  | swap | 1 | 165 |
  | 单侧加长（见下） | 1 | 165 |

  **单侧加长（两种界面都跑）**：
  - **原界面**：不加 `--closer-in-slot`，新文件 `results/dream/bfcl_onesided.jsonl`；
  - **变体**：新文件 `results/dream/bfcl_closer_onesided.jsonl`。
  
  **做法**：对每个兄弟槽长不全相等的兄弟组 (f, p)，设 j* 为该组里参考值最长的调用（并列取第一个），i* 为该组里按顺序第一个比 j* 短的调用。只把槽 (i*, p) 的长度改成 L[j*, p]（变体里再 +1），其余所有槽保持 exact。
  
  **为什么要做**：swap 时两个槽同时换了长度。在 k=1 下，较长的槽写入兄弟值，可能只是因为兄弟的槽太短、被迫先写了自己的值，较长的槽于是避开了重复。单侧加长时兄弟的槽是精确的，会写自己的值；如果加长的槽仍写入兄弟的值，两个调用就出现相同的值，这无法用“避免重复”解释。
  
  **统计**：只看被加长的槽，分为四类：
  - 自己的值（提前收尾）；
  - 自己的值后面接更多内容（overfill）；
  - j* 的值（`sibling_fit`，即重复）；
  - 其他。
  
  同时报告 j* 的槽是否仍是它自己的值。

- **变体下的探针**：`length_prior.py --closer-in-slot`。把参考值 teacher-force 写进槽里后，槽里还剩 1+s 个位置（s ∈ {0, 1, 2, 8}），记录第一个剩余位置上收尾符的概率。
  - 与原探针的对应关系：原探针的“surplus s”是 s 个 mask 之后接预写的收尾符；变体的“surplus s”是 1+s 个位置之后接下一段文本。
  - s=0 时只剩一个位置，它只能是收尾符。

### 2.5 C4：画布 choose-N 的置信阈值和从左到右（很便宜）

Dream 一张画布做 choose-N（105 条），贪心：
```bash
run_dllm.py --data probe:data/choose.jsonl --mode skeleton --k 1 --threshold 0.9 --out results/dream/choose_tau_ltr.jsonl
run_dllm.py --data probe:data/choose.jsonl --mode skeleton --k 1 --order left_to_right --out results/dream/choose_tau_ltr.jsonl
```
用 `choose_analysis.py` 汇总。

**要回答的问题**：论文建议“有序的调用用置信阈值并行提交”，但置信阈值在对称请求上会不会让各槽同时、而且同样自信地选中同一个城市？

### 2.6 团队输出和指标（C1、C2）

- **组装团队输出**：
  - 如果每个 agent 都恰好解析出一个调用：`text = json.dumps([call_1, ..., call_n])`，`parsed = parse_tool_calls(text)`，`diagnosis = diagnose(完整请求 ex, parsed)`。这和一张画布用同一套判分。
  - 否则这个团队记为 `unparsed`：`correct = False`，标签为 `["syntax_error"]`。
- **团队层面的指标**（按 模型 × 子集 × 槽长 × 协议 统计）：
  - `set_acc`：`diagnosis.correct`；
  - `ccer`：`is_cross_call(labels)`；
  - `duplicate`：标签里有 `duplicate_call`；
  - `in_order`：对每个 i，agent i 的调用都通过 `simple_function_checker`（参考调用 i 的函数描述、agent i 的调用、参考调用 i 的可接受值）；
  - `first_mention`（agent 层面，只看兄弟组）：调用通过该组**第一个**参考调用检查的 agent 所占比例；
  - `unparsed`：至少一个 agent 没有恰好解析出一个调用。
- **槽层面的指标**（C2）：只看互换会改变长度的槽（Dream 共 514 个，即论文表 3 用的那些槽）。对 agent i 的每个这样的参数 p，把值分为四类：
  - `own`：通过参考调用 i 的检查（`value_ok`）；
  - `sibling_fit`：通过某个兄弟 j≠i 的检查，且 `oracle_lengths[j, p]` 等于该槽在本条件下的长度；
  - `sibling_other`：通过另一个兄弟的检查，但长度不等；
  - `other`：其他情况，包括缺失和解析失败。
  
  exact 条件下统计同一批槽，便于对照。
- **一张画布的对照行**：从 `release/results_2026-10-03.tar.gz` 解压出的原始记录里取，**只取同一子集**，用同一套函数算：
  - C1：Dream `bfcl_skel_k.jsonl` 的 k=1 和 k=16、LLaDA2.0 `bfcl_skel_k.jsonl` 的 k=1 和 k=16、Qwen `bfcl_skeleton.jsonl`；
  - C2：Dream k=1 exact，以及 Dream `bfcl_swap.jsonl` 的 k=1。
- **C3 的指标**：口径与论文相同。
  - set accuracy、`ccer`；
  - 槽层面的 own / overfill / sibling_fit / truncated / other / unparsed，用 `scripts/slot_errors.py`，必要时让它认识变体的槽长；
  - swap 时被互换槽的分类，与论文表 3 对照；
  - 探针的平均收尾概率。
  
  并排给出原界面的数字：`results/summary/length.md`、`slots_dream.md`、`swap_dream.md`、`length_prior.md`。

### 2.7 预先写下的预测（结果相反时原样报告，不要为了符合预测去调设置）

**C1（顺序线索，104 条，exact）**

| 协议 | 预测 | 依据 |
|---|---|---|
| sim-anon | 同一函数的兄弟 agent 输入完全相同，贪心下输出也完全相同（**理智检查**）；duplicate ≈100%，set_acc ≈0 | 输入相同 |
| sim-label | 不确定。预测 `in_order` 明显低于画布（画布 95–97%）；Qwen 的 duplicate 很高，Dream 较低 | 实验 B：编号几乎没用 |
| sim-rule | `in_order` 和 set_acc 明显高于 sim-label：规则给出了共同知道的映射 | 正对照 |
| turn-anon / turn-label | set_acc 接近画布 k=1 在同一子集上的值；`in_order` 高 | 实验 B：轮流从不撞车 |

如果 sim-label 的 `in_order` 达到画布水平（≥ 90%），或者 sim-rule 也没有改善，都要原样报告。

**C2（长度线索，165 条）**

| 条件 | 预测 |
|---|---|
| Dream exact，sim-anon | 槽长能区分兄弟：duplicate 明显低于 C1 的 sim-anon |
| Dream swap，四种协议 | 被互换的槽大多写入长度合适的兄弟值（`sibling_fit`），比例接近画布（79.4%） |
| Qwen exact | 不受槽长影响；sim-anon 的 duplicate 与 C1 相近 |

**C3（收尾符对照）**：两种假设，预测相反，结果哪种都原样报告。

| 假设 | 预测 |
|---|---|
| 长度承诺：模型把 mask 数当成值的长度，与收尾符在哪里无关 | 变体里 +1、+2 仍然大幅降低 set accuracy；overfill 仍然多；swap 下 `sibling_fit` 仍然高；探针在 s=1、2 时收尾概率仍然低 |
| 局部语法：模型只是不愿连写两个收尾符 | 变体里 +s 的代价小得多；swap 下 `sibling_fit` 明显下降；探针在 s=1、2 时收尾概率明显升高 |

单侧加长：如果是“读槽长”，加长的槽多写入 j* 的值（重复）或 overfill；如果是“轮流时避免重复”，加长的槽多写入自己的值后提前收尾。两种界面分别报告。

**C4**：置信阈值在 choose-N open 变体上会让槽同时选中同一个城市，duplicate 远高于 k=1 的 0%；从左到右不撞车。

## 3. 实现规格（只加文件或加可选参数，不改变现有默认行为）

1. **`ptcdiag/data/agents_bfcl.py`**
   - `team_symmetric(tokenizer, ex) -> bool`、`swap_changes(tokenizer, ex) -> bool`：2.1 节的定义。
   - `bfcl_agent_example(ex, i, n, protocol, previous=())`：与 `agent_example` 相同，但有以下区别：
     - `ground_truth=[ex.ground_truth[i-1]]`；
     - `meta` 加上 `agent=i`、`protocol`、`call=i-1`；
     - `id = f"{ex.id}_a{i}"`。
   - `agent_lengths(tokenizer, ex, i, mode)`：mode 为 `"oracle"` 时返回 `None`，为 `"swap"` 时返回 `{(0, p): S[i-1, p]}`。
   - `sim-rule` 的文本常量和 `ordinal()`。可以加在 `ptcdiag/data/agents.py`（只新增，不改已有常量），并在 `PROTOCOLS` 之外单独登记，保证实验 B 的行为不变。
2. **`scripts/run_agents_bfcl.py`**
   - **参数**：`--model`、`--backend {dllm,ar}`、`--data`、`--bfcl-dir`、`--subset {sym,swap}`、`--length-mode {oracle,swap}`、`--protocols`、`--out`、`--limit`、`--dry-run`、`--device`。
   - **复用**：从 `scripts/run_agents.py` 导入 `load_model`；Dream 的解码配置用 `DecodeConfig(block_length=None, k=1, order="confidence")`。
   - **每个团队写一条 JSONL 记录**：
     ```json
     {"id", "category", "subset", "length_mode", "n", "protocol", "model_id", "backend", "decoding",
      "agents": [{"i", "user_text", "text", "syntax_ok", "call", "lengths"}],
      "team": {"syntax_ok", "text", "calls", "diagnosis"}, "seconds"}
     ```
     `user_text` 必须原样保存。
   - **断点续跑**：跳过已经存在的 (id, protocol, length_mode)。
   - `--dry-run` 不加载模型，只打印 user 文本。
   - choose-N 上的 `sim-rule`：可以让 `scripts/run_agents.py` 接受这个协议名（只新增），输出到新文件 `results/{qwen,dream}/agents_rule.jsonl`。
3. **`closer_in_slot` 变体**：见 2.4 节。改动集中在 `constraints.py`，加可选参数，默认关闭；在 `run_dllm.py` 和 `length_prior.py` 加开关；记录里写 `"closer_in_slot": true`。输出写到新文件：
   - `results/dream/bfcl_closer.jsonl`（exact、surplus、k=16）；
   - `results/dream/bfcl_closer_swap.jsonl`；
   - `results/dream/bfcl_onesided.jsonl`（原界面）和 `results/dream/bfcl_closer_onesided.jsonl`（变体）；
   - `results/dream/length_prior_closer_s{0,1,2,8}.jsonl`。
   
   单侧加长的槽长由 `ptcdiag/decoding/constraints.py` 里新增的 `onesided_lengths(tokenizer, example)` 给出（只新增函数），通过 `run_dllm.py --length-mode onesided` 传入，记录里写 `length_mode: "onesided"`。
4. **`scripts/agents_bfcl_analysis.py`** 输出：
   - `results/summary/agents_bfcl.md` 和 `.csv`：C1 团队表（含 choose-N `sim-rule` 一行）、C2 团队表、C2 槽层面表，每张表后附画布对照行；
   - `results/summary/agents_bfcl_examples.md`：每个 模型 × 条件 × 协议 一个团队，带完整的输入和输出。
5. **C3、C4 的汇总**：`results/summary/closer.md` 和 `.csv`，变体与原界面并排；`results/summary/choose_tau_ltr.md`。
6. **测试**：
   - `tests/test_agents_bfcl.py`：
     - `team_symmetric` 的三种情况：签名相同为真、兄弟槽长不同为假、没有兄弟组为假；
     - `bfcl_agent_example` 的 ground_truth 正好是第 i 个调用；
     - C1 sim-anon 的兄弟 agent 文本完全相同；`sim-rule` 文本的序数正确；
     - 轮流协议的文本包含之前的调用；
     - `agent_lengths` 的 swap 映射正确；
     - 团队组装、`duplicate`、`in_order`、`first_mention` 在玩具团队上的值正确。
   - `tests/test_closer.py`：
     - 变体骨架里槽后不再预写收尾符；
     - 槽长 = 参考长度 + 1 + s；
     - 提前收尾后的填充是空格；
     - 解码输出能解析；
     - **默认参数下骨架与现在逐 token 相同**。
7. **`scripts/run_minimal.sh`**：在 case 语句末尾（`*)` 之前）新增以下阶段，不要改动已有阶段：
   - `agents_bfcl_ar`、`agents_bfcl_dream`、`agents_bfcl_swap`、`agents_rule`；
   - `closer_dream`、`closer_probe`、`choose_tau_ltr`；
   - `summary_c`。

## 4. 执行步骤与关卡（每个关卡不过就停下汇报）

1. **CPU**：
   - `python -m pytest tests -q` 全部通过，包括新测试；
   - 用两个分词器重算子集，必须得到 104 条（84 + 20，289 个 agent）和 165 条；
   - `--dry-run --limit 2` 检查五种协议的提示拼接。
2. **默认行为不变**（GPU）：不加 `--closer-in-slot`，用 Dream 重跑 5 条 BFCL（k=1 和 k=16）。输出 `text` 必须与 `release/results_2026-10-03.tar.gz` 里 `results/dream/bfcl_skel_k.jsonl` 对应记录逐字相同。
3. **冒烟**（GPU）：
   - agent 团队：两个模型、五种协议，两个子集各 3 条，Dream 的 swap 也跑。打印每个 agent 的 `user_text`、槽长和输出，人工确认提示拼接、轮流协议带上了之前的调用、swap 的槽长确实换了。
   - 变体：exact 和 +1 各 5 条。打印画布最终状态，确认槽后没有预写的收尾符、填充是空格、输出能解析。
   - **变体 exact 的 set accuracy 不能比原界面低很多**（20 条上相差超过 20 个点就停下汇报），否则说明变体实现有问题。
4. **理智检查**：C1 的 sim-anon 下，两个模型里同一函数的兄弟 agent 输出必须完全相同。
5. **主实验，按这个顺序跑**（时间不够时，前面的优先）：
   1. C1；
   2. C3（含单侧加长）；
   3. C2；
   4. C4。
   
   最后运行 `summary_c`。
6. **汇报**。

## 5. 资源和预算

- 一张 40GB 以上的卡就够；Dream 和 Qwen 依次加载。
- **工作量估计**：
  - C1：约 2,900 次很短的单调用解码（289 个 agent × 5 种协议 × 2 个模型），再加 choose-N 的 `sim-rule`；
  - C2：约 6,000 次；
  - C3：约 2,900 条完整请求的画布解码（含两种界面的单侧加长），主要是 k=1，再加探针；
  - C4：210 条。
- 加上安装和下载模型，**估计 4–5 GPU 小时**。这是估计值，冒烟时测一下速度再更新。
- 累计 GPU 时间超过 **6 小时**时，先停下来问用户。
- 跑完立即把结果拉回本地，然后停机（HANDOFF 第 9 节），不要让 pod 空转。

## 6. 交付

推到 `exp-c-agents`：
- 新代码和测试；
- `results/summary/` 下的 `agents_bfcl.md`、`agents_bfcl.csv`、`agents_bfcl_examples.md`、`closer.md`、`closer.csv`、`choose_tau_ltr.md`。都用 `git add -f`；
- 原始 JSONL 打包成 `release/exp_c_<日期>.tar.gz` 一并提交。

**不要改动 `paper/` 下的任何文件，也不要改已有的结果文件**，写作那边会同时在改。

向用户汇报：
1. C1 表（含 choose-N 的 `sim-rule`），附画布对照行；
2. C2 团队表和槽层面表，附画布对照行；
3. C3 表：变体与原界面的 set accuracy、槽层面分类、swap 分类、单侧加长分类、探针，并排；
4. C4 表；
5. 与预测（2.7 节）逐条对照，哪里相反；
6. 每种协议一个代表性团队的输入和输出；变体的两个画布样例；
7. 实际花费的 GPU 时间，以及任何偏离本方案的地方及原因。

## 7. 规则

- 只跑本方案里的配置；需要改设计时先问。
- 没跑出来的数字写 TBD，不要估计或外推。
- 结果和预测相反时原样报告。
- 每处代码修改单独提交，commit message 写清原因。
- 改 `constraints.py` 时只加可选参数，默认路径一行都不要动；关卡 2 不过就停。

---
