# Masks Are Length Promises: Diagnosing Parallel Tool Calls in Diffusion Language Models

AAMAS 2027 论文大纲（工作稿，2026-10-03，所有实验已跑完）。截稿时间为 10 月 9 日 20:00（北京时间）。

规则：所有数字都来自 `results/summary/`（由 `bash scripts/run_minimal.sh summary_length` 生成）。结果与预期相反时原样报告，并写明论点因此要怎么改。

---

## 一句话论点

用骨架约束 dLLM 生成并行工具调用时，并行提交在请求给出实体顺序时几乎没有代价。只有当没有任何信息能区分几个调用时（choose-N），并行提交才会导致重复。

真正普遍而严重的问题是槽长。模型把 mask 个数当作值的长度：每个槽只多一个 mask，set accuracy 就从 0.897 掉到 0.155。槽长互换之后，79% 的槽写进了长度正好合适的兄弟值，这在逐 token 串行解码时也一样。这些错误看起来像跨调用协调失败，很容易被误算到并行解码头上。

## 贡献

1. **误归因**：长度错配本身就会产生“跨调用错绑”，与并行度无关。
   - 长度互换：Dream 被互换的槽里有 79.4% 写进了长度正好合适的兄弟值（k=1），exact 长度下这一比例是 0%。
   - surplus 会推高 CCER：Dream s=1 时为 1.3% → 5.8%；LLaDA2.0 s=8 时为 2% → 36%。
2. **定量对比和边界**：k 的代价比长度错配小一个数量级以上。
   - k 的代价：k 从 1 到 16，Dream −5.2 个点。
   - 长度错配的代价：s 从 0 到 1，−74.2 个点。
   - 原因是模型按提及顺序绑定实体（95–97%）。
   - 边界：请求不规定选哪些实体时，Dream 的并行提交会在同一步里产生重复（open 变体：k≥2 时 100%，k=1 时 0%）。
3. **严重程度、非单调性和缓解手段**：
   - 只差一点最危险：Dream k=1 时 s=2 为 0.033，s=8 为 0.600。
   - 收尾偏置在余量大时有效（s=8：0.570 → 0.723），在小错配时基本无效。
   - 一次前向的长度估计（CAL 式）只能恢复到 0.42，离 exact 长度的 0.90 还很远。

**和已有工作的关系**：“mask 个数被当作长度”这一现象已有报道，见 CAL 的 Length Bias（arXiv 2602.00476）、ρ-EOS（2601.22527）和 DAEDAL（2508.00819）。我们不声称发现了这个现象。我们的贡献在于：
- 在多槽结构化输出中测量这个现象；
- 证明它会伪装成跨调用错误；
- 把它和并行解码代价放在同一尺度上比较。

---

## 章节（正文 8 页；按第二份审稿意见附录也算在内，以 CFP 为准）

### 1 引言（约 1 页）
- agent 越来越常在一轮里同时调用多个工具，dLLM 的并行解码看起来正好合适。担心在于：同一步提交的 token 互相看不见，可能导致重复调用或参数错绑。DiffuAgent（2601.12979）也报告了 dLLM 调用工具失败。
- **要先讲清楚长度错配是常态**：用 dLLM 填模板或骨架时，槽长必须事先给定，而值的 token 数在解码前是未知的。
- 我们的三点发现见上文的“贡献”。
- 图 1：同一个双调用例子在 exact / +1 / swap 三种槽长下的输出。+1 用真实样例“Taylor Swift” → “Taylor Swift. Swift”；swap 的样例从 `results/dream/bfcl_swap.jsonl` 里挑一个。

### 2 背景与相关工作（约 0.75 页）
- **dLLM**：LLaDA、Dream-v0-Instruct-7B、LLaDA2.0-mini（16B MoE，32 token 的块因果注意力）；并行解码的因子化误差。
- **变长问题**：
  - CAL：Length Bias / Oracle Peak，以及基于 first-step 的长度估计；
  - ρ-EOS、DAEDAL：整段末尾的变长控制；
  - S³：null token。
  - 本文区别于以上工作的地方：多槽、槽内长度、错误归因。
- **工具调用评测**：BFCL v4 的 parallel / parallel_multiple，以及错误分类；约束扩散解码（constrained-diffusion）。

### 3 设定（约 1.25 页）
- **oracle 骨架**：调用数、函数名、参数名、括号和引号都给定，只有值是 mask。
- **收尾规则**：字符串在 `"` 处结束，其他类型在深度 0 的 `,` 或 `}` 处结束；收尾后槽内剩余位置强制填 pad。AR 基线遵守同一条规则，并且最多只写到槽长（只截断、不强制写满）。
- **槽长条件**：
  - exact：每个槽等于自身 gold 值的 token 数；
  - +s：每个槽多 s 个 mask，s ∈ {1, 2, 4, 8}；
  - swap：同一兄弟组内槽长轮换一位；
  - estimate：一次前向估计，不使用 gold 长度。
- **解码**：k ∈ {1, 2, 4, 8, 16}，τ=0.9，left-to-right；LLaDA2.0 的块长为 32；全部贪心解码。
- **模型**：Dream-7B（AR 对照为同 tokenizer 的 Qwen2.5-7B）、LLaDA2.0-mini。
- **数据**：
  - BFCL parallel + parallel_multiple，共 400 条。LLaDA2.0 的 surplus、swap、estimate 实验用其中 100 条子集。
  - choose-N 探针，105 条。
- **指标**：
  - set_acc：BFCL 集合匹配；
  - CCER / SCER：分别是含跨调用错误、含单调用错误的题目占比；
  - 槽级分类：own / sibling_fit / sibling / overfill / truncated / other / unparsed（`scripts/slot_errors.py`）。

### 4 RQ1：并行提交会破坏跨调用协调吗？（约 1.25 页）

**表 1**（exact 长度，BFCL 400 条）

| | k=1 | k=2 | k=4 | k=8 | k=16 | left-to-right | τ=0.9 |
|---|---|---|---|---|---|---|---|
| Dream set_acc | 0.897 | 0.880 | 0.870 | 0.863 | 0.845 | 0.892 | 0.890 |
| Dream CCER | 1.3% | 1.3% | 1.5% | 1.8% | 2.0% | 1.0% | 1.3% |
| LLaDA2.0 set_acc | 0.877 | 0.875 | 0.865 | 0.833 | 0.797 | — | — |
| LLaDA2.0 CCER | 2.5% | 2.8% | 3.3% | 4.0% | 4.5% | — | — |

Qwen2.5（AR）：set_acc 0.865，CCER 1.3%。

- **长度对称子集**：在这些题里，画布本身无法区分各个调用。CCER 依然很低，所以 CCER 低不是长度泄露造成的。
  - Dream（213 条）：CCER 1.9% → 2.8%（k=1 → 16）。
  - LLaDA2.0（216 条）：CCER 0.9% → 1.9%（k=1 → 16）。
- **提及顺序**：在对称的 BFCL parallel 题中，第 i 个调用填入的正是第 i 个被提到的实体。
  - 这一比例在 Dream 上为 0.950–0.975，在 LLaDA2.0 上为 0.938–0.963，与 k 无关。
- **choose-N（边界情形）**：请求不规定选哪几个实体。

**表 2**

| | list：重复率 | list：按列出顺序 | open：重复率 |
|---|---|---|---|
| Dream k=1 | 0% | 48% | 0% |
| Dream k=2 | 10% | 32% | 100% |
| Dream k=4 / 16 | 15% | 12% | 100% |
| LLaDA2.0 k=1 / 4 / 16 | 0% | 97–100% | 0% / 4.4% / 4.4% |
| Qwen2.5（AR） | 0% | 82% | 0% |

- Dream 的所有重复都来自同一步提交的两个槽，最典型的是 open 变体在 k≥2 时写出 (Chicago, Chicago, …)。这是并行解码因子化失败的直接证据，原始担心在这里确实成立。
- LLaDA2.0 基本不重复。原因有两个：
  - 32 token 的块因果注意力把各个调用分进不同的块，实际上变成了串行解码；
  - list 变体里它几乎总是按列出顺序选择（97–100%）。
- **结果与原先预期相反，论点据此修改为**：只要请求给出了实体顺序，并行几乎没有代价；什么都区分不了各个槽时，全双向注意力的 dLLM（Dream）会在同一步写出重复值。
- **注意**：open 变体的槽长只有 1 个 token，多 token 的城市名会被截断（“New”）或改写成缩写（“LA”、“NY”）。所以 open 变体的有效率不可用，只报告重复率。

### 5 RQ2：模型从 mask 个数里读到了什么？（约 0.5 页）
做法：先把 gold 值 teacher-force 进槽，后面再放 s 个 mask，测量下一个位置的概率分布（共 3063 个槽）。

| s | Dream P(内容) | Dream P(收尾) | LLaDA2.0 P(内容) | LLaDA2.0 P(收尾) |
|---|---|---|---|---|
| 1 | 0.990 | 0.010 | 0.807 | 0.193 |
| 2 | 0.967 | 0.033 | — | — |
| 4 | 0.746 | 0.254 | 0.467 | 0.533 |
| 8 | 0.260 | 0.740 | 0.082 | 0.918 |

- P(pad) 很小：Dream 的中位数约为 1e-5，LLaDA2.0 约为 1e-6。所以仅仅允许 null/pad token（S³ 式）并不起作用，本文的约束本来就允许 pad。
- 结论：剩余 mask 越少，模型越笃定还要继续写；剩余很多时反而愿意收尾。这与下一节 set_acc 随 s 的非单调变化相吻合。

### 6 RQ3：长度错配的代价，以及它如何伪装成协调失败（约 1.5 页）

**表 3**：set_acc（Dream 400 条；LLaDA2.0 用 100 条子集，s=0 也在同一子集上计算）

| | s=0 | s=1 | s=2 | s=4 | s=8 |
|---|---|---|---|---|---|
| Dream k=1 | 0.897 | 0.155 | 0.033 | 0.170 | 0.600 |
| Dream k=4 | 0.870 | 0.102 | 0.028 | 0.145 | 0.570 |
| Dream k=16 | 0.845 | 0.050 | 0.018 | 0.072 | 0.475 |
| Dream k=1 CCER | 1.3% | 5.8% | 5.2% | 4.3% | 1.8% |
| LLaDA2.0 k=1 | 0.920 | 0.160 | 0.080 | — | 0.380 |
| LLaDA2.0 k=4 | 0.900 | 0.120 | 0.070 | — | 0.350 |
| LLaDA2.0 k=1 CCER | 2.0% | 12.0% | 17.0% | — | 36.0% |
| Qwen2.5（AR） | 0.865 | — | — | — | 0.890 |

- Qwen2.5 不受影响，这是 AR 的设计性质（写到收尾符就停），不算实验发现。
- **对比式结论**：
  - Dream 在 s=0 时 k 从 1 到 16：set_acc −5.2 个点，槽级准确率 −3.4 个点（0.954 → 0.920）。
  - Dream 在 k=1 时 s 从 0 到 1：set_acc −74.2 个点，槽级准确率 −61.4 个点（0.954 → 0.340）。
- **非单调**：s=2 最差，s=8 部分恢复，与第 5 节的探针结果一致。
- **槽级构成**（k=1）：
  - 小幅错配主要导致写长。Dream s=1 时 overfill 占 0.351，抄兄弟值只占 0.005。
  - 余量大时会出现抄兄弟值。LLaDA2.0 s=8 时 sibling 占 0.077，这时 CCER 达到 36%。
- **长度互换**（只统计长度被改动的槽；在同一批槽上与 exact 长度对比）：

**表 4**

| | 槽数 / 题数 | own | 长度合适的兄弟值 | 写长 | 截断 | 其他 | set_acc | CCER |
|---|---|---|---|---|---|---|---|---|
| Dream exact，k=1 | 514 / 165 | 0.963 | 0.000 | 0.002 | 0.000 | 0.035 | 0.933 | 0.6% |
| Dream swap，k=1 | 514 / 165 | 0.054 | **0.794** | 0.023 | 0.023 | 0.097 | 0.673 | 16.4% |
| Dream swap，k=4 | | 0.062 | 0.772 | 0.023 | 0.023 | 0.119 | 0.648 | 19.4% |
| Dream swap，k=16 | | 0.043 | 0.733 | 0.029 | 0.021 | 0.121 | 0.558 | 23.6% |
| LLaDA2.0 exact，k=1 | 120 / 34 | 0.967 | 0.008 | 0.000 | 0.000 | 0.008 | 0.882 | 5.9% |
| LLaDA2.0 swap，k=1 | 120 / 34 | 0.125 | **0.475** | 0.067 | 0.133 | 0.150 | 0.176 | 44.1% |
| LLaDA2.0 swap，k=4 | | 0.092 | 0.400 | 0.067 | 0.117 | 0.283 | 0.147 | 50.0% |

- 槽比自己的值长或短，两个方向的结果一致（Dream k=1 分别为 79.1% 和 79.7%）。槽比自己的值长时，模型本可以写完自己的值就收尾，但仍写入了更长的兄弟值。
- set_acc 的降幅比槽级小，因为当一组只有这一个参数不同时，整组互换后在集合匹配下仍算正确。
- LLaDA2.0 的结论方向相同但更弱，而且只有 34 条题，要在文中注明。

### 7 缓解与建议（约 0.75 页）

**表 5**：Dream，k=4，set_acc

| | s=2 | s=8 |
|---|---|---|
| 无偏置 | 0.028 | 0.570 |
| β=2 | 0.028 | **0.723** |
| β=4 | 0.065 | 0.713 |
| β=8 | 0.138 | 0.203（语法正确率掉到 0.372） |

- **收尾偏置的结论**：余量大时有效，小错配时基本救不回来。β 过大会导致过早收尾。

**一次前向估计槽长**（每个槽先给到类型上限个 mask，跑一次前向，把收尾概率当作 hazard，取众数作为槽长）：
- **估计准确度**：
  - Dream：51.6% 的槽估计完全正确；只有 14.5% 的题全部槽都正确；估计过长占 44%。
  - LLaDA2.0：72.8% 的槽正确，27% 的题全部正确。
- **按估计长度解码后的 set_acc**：
  - Dream：k=1 0.422，k=4 0.390，k=16 0.323。
  - LLaDA2.0：k=1 0.390，k=4 0.380。
  - 对比：比 s=1 或 s=2 好得多，但距 exact 长度（约 0.90）差距仍然很大。
- **结论**：模型对槽长只有部分认知，单靠一次前向并不足以给出可用的槽长。
- **给 agent 开发者的建议**：
  - 槽长要么精确，要么留出大量余量并配合收尾偏置，最危险的是只差一点。
  - 选择对称（choose-N）时要串行解码，或者使用块因果结构。
  - 评测 dLLM 工具调用时，先排除长度错配，再把错误归因到并行解码。
- ρ-EOS、DAEDAL 只做定性讨论（它们针对整段末尾，搬到槽上需要另行改写）。

### 8 讨论与局限（约 0.5 页）
- oracle 骨架给定了调用数和名字；exact 长度本身会泄露信息，已用对称子集和长度互换处理。
- 只测了两个 dLLM。LLaDA2.0 的结果受块因果结构影响，不写块长度对照。
- free 模式只写一句：Dream 不加约束时 set_acc 0.752，骨架下为 0.880（k=2）。
- LLaDA2.0 的长度互换只有 34 条题。
- choose-N 的 open 变体受 1 token 槽长的限制，只报告重复率。

### 9 结论（约 0.25 页）

---

## 摘要草稿

> Diffusion language models (dLLMs) decode many tokens in parallel, which makes them attractive for agents that issue several tool calls at once, and raises a worry: tokens committed in the same step cannot see each other, so parallel calls might duplicate or cross-bind arguments. We test this worry on BFCL parallel tool calls with an oracle call skeleton and find it narrow. When the request orders its entities, committing 16 tokens per step instead of one costs Dream-7B 5 points of set accuracy, also on items whose slots the canvas cannot tell apart, because the models bind the i-th call to the i-th mentioned entity at any degree of parallelism; only when nothing distinguishes the calls ("any two US cities") do same-step commits duplicate. The dominant failure is elsewhere: dLLMs read the number of masks in a slot as the length of its value. One extra mask per slot drops set accuracy from 0.90 to 0.16, and swapping slot lengths between sibling calls makes 79% of the affected slots take the sibling whose value fits, even when decoding one token at a time—errors an evaluator would attribute to parallel decoding. Small mismatches are the most harmful; a logit bias toward closing tokens recovers large surpluses (0.57 to 0.72) but not small ones, and a one-pass length estimate recovers only to 0.42. Template-constrained tool calling with dLLMs must control slot length before it worries about parallelism.

## 图表清单

| 编号 | 内容 | 数据来源（`results/summary/`） |
|---|---|---|
| 图 1 | 示意图：exact / +1 / swap | 原始 JSONL 中的样例 |
| 图 2 | set_acc 随 k 的变化，每个 s 一条线（左 Dream，右 LLaDA2.0），AR 画成水平线 | `length.md` |
| 表 1 | 并行度扫描、对称子集、提及顺序 | `length.md`、`symmetry_*.md` |
| 表 2 | choose-N | `choose.md` |
| 表 3 | 长度先验探针 | `length_prior.md` |
| 表 4 | s × k，以及槽级构成 | `length.md`、`slots_*.md` |
| 表 5 | 长度互换 | `swap_*.md` |
| 表 6 | 缓解：β 和长度估计 | `length.md`、`length_prior.md` |

## 从原 proposal 砍掉的部分
- H1–H4 的原始表述（改为第 4 节的检验）。
- 反事实归因、DVS、命题 1 的形式化表述。
- free 模式的新跑和块长度对照。
- ParaProbe（只在附录写一句：实体绑错为零，CCER 来自默认参数不一致，与 k 无关）。
