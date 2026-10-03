# Masks Are Length Promises: Diagnosing Parallel Tool Calls in Diffusion Language Models

AAMAS 2027 论文大纲（工作稿，2026-10-03）。截稿 10 月 9 日 20:00（北京时间）。

规则：表中的数字都来自已经跑完的结果，文件见 `results/summary/`。没跑完或只跑了一部分的标 **TBD**，不做估计或外推。结果和预期相反时原样报告，并写明论点要怎么改。

---

## 一句话论点

在骨架约束下让扩散语言模型（dLLM）填写并行工具调用时，并行解码的代价很小：k 从 1 增加到 16，Dream 的 set accuracy 下降 5.2 个点。真正的代价来自槽长：每个槽只多一个 mask，set accuracy 就从 0.897 掉到 0.155。而且长度错配会制造出看起来像跨调用协调失败的错误（CCER 1.3% → 5.8%），这些错误很容易被误算到并行解码头上。

## 贡献（按第二份审稿意见重新划定）

1. **误归因（核心）**：长度错配带来的重复调用、错绑和抄兄弟值，看起来像并行解码的协调失败，其实来自长度先验。证据有三项：surplus 下 CCER 上升；参数最大长度设计中 17/18 的跨调用错误是长度引起的；长度互换实验 **TBD**。
2. **定量对比**：并行度（k）的代价比长度错配小一个数量级以上。这个结论在长度对称的子集上同样成立，所以不是长度泄露造成的。并行度代价小的原因是，模型按提及顺序给槽绑定实体（95–97%，和 k 无关）。
3. **agent 场景下的严重程度和缓解办法**：骨架或模板式约束解码必须先定槽长，只差一格就会崩。缓解办法有两种：收尾偏置 β（**TBD**），以及只用一次前向估计槽长、不依赖 gold（**TBD**）。

**和已有工作的关系**：“mask 个数被当作长度”这一现象已有报道，见 CAL 的 Length Bias（arXiv 2602.00476）、ρ-EOS（2601.22527）和 DAEDAL（2508.00819）。本文不声称发现了这个现象，贡献在于以下三点：
- 在多槽结构化输出（并行工具调用）里测量它；
- 证明它会伪装成跨调用协调错误；
- 量化它和并行解码代价的相对大小。

**风险**：如果长度互换显示模型并不按长度给槽绑定值，贡献 1 就只剩“surplus 让 CCER 上升”这一条。论文会退化成定量对比加严重程度，和 CAL 的区分度会变弱。

---

## 章节（正文 8 页；按第二份审稿意见附录也算在内，以 CFP 为准）

### 1 引言（约 1 页）
- agent 越来越多地在一轮里并行调用多个工具；dLLM 并行解码看起来正好合适，但大家担心同时提交会破坏调用之间的协调（重复、错绑）。DiffuAgent（2601.12979）报告 dLLM 在 agent 工作流里调用工具失败。
- **要先讲清楚**：长度错配在实际中是常态。用 dLLM 填模板或骨架时，槽长必须事先给定，而值的 token 数在解码前是未知的。所以长度错配是任何骨架式 dLLM 工具调用都会遇到的问题，不是我们人为制造出来的。
- 我们的发现：
  - 并行提交基本无害；
  - 长度先验才是主要矛盾；
  - 长度先验产生的错误会被误认为协调失败。
- 图 1（示意）：一个两调用的例子，比较 exact 长度、+1 mask 和长度互换三种情况。+1 mask 的真实样例已有：“Taylor Swift” → “Taylor Swift. Swift”，500000 → 5000000000；长度互换的样例 **TBD**。

### 2 背景与相关工作（约 0.75 页）
- **dLLM**：LLaDA、Dream-v0-Instruct-7B、LLaDA2.0-mini（16B MoE，32 token 块因果）。并行解码的因子化误差。
- **变长和长度问题**：
  - CAL（Length Bias / Oracle Peak，免训练长度估计）；
  - ρ-EOS、DAEDAL（整段末尾的变长控制）；
  - S³（null token 吸收多余长度）。
  - 本文区别在于：多槽、槽内部的长度，以及错误归因。
- **工具调用评测**：BFCL v4 parallel / parallel_multiple，以及错误分类方面的相关工作。
- **约束解码**：constrained-diffusion（eth-sri）。我们的骨架是一个 oracle 上界设定。

### 3 设定（约 1.25 页）
- **把并行调用看作填槽**：oracle 骨架给定调用数、函数名、参数名、括号和引号，只有值是 mask。
- **收尾规则**：字符串遇到 `"` 结束，其他类型在深度 0 遇到 `,` 或 `}` 结束。收尾之后槽里剩下的位置强制填 pad，AR 基线用同一条规则。
- **槽长条件**：
  - exact：每个值的 gold token 长度；
  - +s：每个槽多 s 个 mask，s ∈ {1, 2, 4, 8}；
  - swap：兄弟槽之间轮换长度；
  - estimate：一次前向估计长度，不用 gold。
- **解码旋钮**：每步提交 k ∈ {1, 2, 4, 8, 16}、阈值 τ=0.9、left-to-right、块长度（LLaDA2.0 固定为 32）。
- **模型**：Dream-7B（AR 对照用 Qwen2.5-7B，同一个 tokenizer）和 LLaDA2.0-mini。全部用贪心解码。
- **数据**：
  - BFCL parallel + parallel_multiple，共 400 条；
  - choose-N 探针，105 条；
  - ParaProbe（840 条）只在附录或一句话里提。
- **指标**：
  - set_acc（BFCL 集合匹配）；
  - CCER / SCER（分别是跨调用和单调用错误的题目占比）；
  - 槽级分类（own / sibling / overfill / truncated / other / unparsed，见 `scripts/slot_errors.py`）。
- 命题 1 不写形式化表述；对称情形用 choose-N 实验和一段文字交代。

### 4 RQ1：并行提交会破坏跨调用协调吗？（约 1 页）

**表 1 / 图 2**（exact 长度，s=0，BFCL 400 条）：

| 模型 | k=1 | k=2 | k=4 | k=8 | k=16 | left-to-right | τ=0.9 |
|---|---|---|---|---|---|---|---|
| Dream set_acc | 0.897 | 0.880 | 0.870 | 0.863 | 0.845 | 0.892 | 0.890 |
| Dream CCER | 1.3% | 1.3% | 1.5% | 1.8% | 2.0% | 1.0% | 1.3% |
| LLaDA2.0 set_acc | 0.877 | 0.875 | 0.865 | 0.833 | TBD（只有 259/400，正在补跑） | — | — |
| LLaDA2.0 CCER | 2.5% | 2.8% | 3.3% | 4.0% | TBD | — | — |
| Qwen2.5（AR） | TBD（371/400，正在补跑） | | | | | | |

- **长度对称子集**（每个兄弟组的槽长都相同，画布无法区分各个调用）：
  - Dream（213 条）：CCER 从 1.9%（k=1）到 2.8%（k=16），set_acc 从 0.869 到 0.817。
  - LLaDA2.0（216 条）：CCER 从 0.9%（k=1）到 1.4%（k=8）。
  - 所以 CCER 低不是因为长度泄露。
- **提及顺序**：在对称的 BFCL parallel 题里，第 i 个调用填的就是第 i 个被提到的实体。
  - Dream 的比例是 0.950–0.975，k=1、k=16、left-to-right、τ 都在这个范围。
  - LLaDA2.0 是 0.938–0.963。
  - 这说明每个槽的预测在提交之前就已经由位置决定了，同时提交不会撞车（描述性证据）。
- **choose-N 探针**（边界情形，请求不规定调用顺序）：Dream k∈{1,2,4,16}、LLaDA2.0 k∈{1,4,16}、Qwen 的重复率，以及同一步提交造成的重复 **全部 TBD**。
  - 结果如果显示 open 变体的重复率随 k 上升：论点改为“只要请求给出了实体顺序，并行几乎没有代价；没给顺序时，并行代价就出现了”。
  - 结果如果显示不随 k 上升：论点保持不变，并补充“即使没有顺序，模型也会通过别的先验把各个槽区分开”。

### 5 RQ2：模型把 mask 个数读成什么？（约 0.5 页）
- **teacher-forced 探针**：每个槽先填入 gold 值，后面再接 s 个 mask，测下一个位置是内容、收尾符还是 pad 的概率（`scripts/length_prior.py`，3063 个槽）。

| s | Dream P(内容) | Dream P(收尾) | LLaDA2.0 P(内容) |
|---|---|---|---|
| 1 | 0.990 | 0.010 | TBD |
| 2 | 0.967 | 0.033 | — |
| 4 | 0.746 | 0.254 | 0.467 |
| 8 | 0.260 | 0.740 | TBD |

- P(pad) 中位数约 1e-5。模型几乎不在值的中间放 pad，所以“允许 null/pad token”（S³ 式做法）本身不起作用。
- **非单调**：只多一两个 mask 时，模型几乎一定会继续写内容；多出很多时，反而更愿意收尾。这一点和 CAL 的 Length Bias 一致，本文把它测到了槽的粒度上。

### 6 RQ3：长度错配的代价，以及它伪装成协调失败（约 1.5 页）

**表 2**：s × k（Dream 400 条；LLaDA2.0 用 100 条子集，同一子集上的 s=0 一并列出）

| | s=0 | s=1 | s=2 | s=4 | s=8 |
|---|---|---|---|---|---|
| Dream k=1 set_acc | 0.897 | 0.155 | 0.033 | TBD | TBD |
| Dream k=4 | 0.870 | 0.102 | 0.028 | TBD | TBD |
| Dream k=16 | 0.845 | 0.050 | 0.018 | TBD | TBD |
| Dream k=1 CCER | 1.3% | 5.8% | 5.2% | TBD | TBD |
| LLaDA2.0 k=1 set_acc（100 条） | 0.920 | TBD | 0.080 | — | TBD |
| LLaDA2.0 k=4 set_acc（100 条） | 0.900 | TBD | 0.070 | — | 0.350 |
| LLaDA2.0 k=4 CCER（100 条） | 3.0% | TBD | 12.0% | — | 31.0% |
| Qwen2.5（AR）s=8 | — | — | — | — | TBD（预期不变：AR 遇到收尾符就停，这是设计上的性质，不算实验发现） |

- **对比式结论**：Dream 在 s=0 时 k 从 1 到 16，set_acc −5.2 个点；k=1 时 s 从 0 到 1，set_acc −74.2 个点。槽级准确率分别是 −3.4 和 −61.4 个点（0.954 → 0.920；0.954 → 0.340）。
- **槽级构成**（Dream k=1，s=1）：
  - own 0.340，overfill 0.351，other 0.117，unparsed 0.184；
  - 抄兄弟值只占 0.005（sibling_fit 0.002，sibling 0.003）。
  - 小幅错配主要导致写长，而不是抄兄弟值。图 1 和正文要按这个写，不能写成“多数错误是抄兄弟值”。
- **s=8 时机制改变**：LLaDA2.0 k=4、s=8 时，sibling 占 0.080（s=0 时为 0），CCER 是 31%（s=0 时 3%）。多出很多 mask 的槽会去抄一个更长的兄弟值。最终数字等 k=1 跑完，**TBD**。
- **自然发生的错配**：之前用过的“同参数共用最长长度”设计，在 20 条的 pilot 里，18 个跨调用错误中有 17 个落在长度不对的槽上（较短的值被写成较长兄弟的值，如 sea_level 0 → 1000）。写进正文作为动机例子。
- **长度互换实验（关键，TBD）**：同一兄弟组内把槽长轮换一位（`--lengths swap`），只统计长度被改动的槽。
  - 判据：写入的是“长度正好装得下的兄弟值”（sibling_fit，按长度绑定），还是自己的值被截断或写长（按位置绑定）。
  - 拆成两种方向：槽比自己的值长、槽比自己的值短。
  - 和同一批槽在 exact 长度下的表现对比。
  - Dream k∈{1,4,16}，LLaDA2.0 k∈{1,4}。

### 7 缓解与建议（约 0.75 页）
- **收尾偏置 β**：给 pad 和收尾符的 logit 加 β，β ∈ {2,4,8}，s ∈ {2,8}，Dream k=4。**TBD**
- **一次前向估计槽长**（CAL 的 first-step 估计，按槽做）：
  - 做法：每个槽先给到类型上限个 mask，跑一次前向，用收尾概率当作 hazard，取众数作为槽长，再按这个长度解码。
  - 要报告的量：
    - 估计长度和 gold 长度完全一致的比例；
    - 解码后的 set_acc，对比 exact 长度和 +s。
  - Dream k∈{1,4,16} 和 LLaDA2.0 k∈{1,4} 的结果都是 **TBD**。
- ρ-EOS 和 DAEDAL 做的是整段末尾的长度控制，放到槽级需要我们自己改写，不能算公平对比，只做定性讨论。
- **给 agent 开发者的建议**（等第 7 节结果出来再定稿）：
  - 槽长要么精确，要么留出大量余量，最危险的是只差一点；
  - 或者先估计长度再解码；
  - 评估 dLLM 工具调用时，先排除长度错配，再把错误归因到并行解码。

### 8 讨论与局限（约 0.5 页）
- **oracle 骨架**：调用数、函数名和参数名都是给定的。exact 长度本身就泄露了信息，这一点已经用对称子集和长度互换来处理。
- **只测了两个 dLLM**。LLaDA2.0 的 32 token 块因果结构会把相距较远的槽串行化；块长度对照不写进正文。
- **free 模式只写一句**：Dream 不加约束时 set_acc 0.752，骨架下 0.880（k=2），用来说明骨架设定的必要性。
- **ParaProbe**：实体绑错几乎为零，CCER 主要来自不一致的默认参数，和 k 无关。放附录或写一句。
- **反事实归因和 DVS 已删除**，不再使用。

### 9 结论（约 0.25 页）

---

## 摘要草稿（数字待定的地方标 TBD）

> Diffusion language models (dLLMs) decode many tokens in parallel, which makes them attractive for agents that issue several tool calls at once, and raises a worry: tokens committed in the same step cannot see each other, so parallel calls might duplicate or cross-bind arguments. We test this worry on BFCL parallel tool calls with an oracle call skeleton, and find it largely unfounded: committing 16 tokens per step instead of one costs Dream-7B 5 points of set accuracy, also on items whose slots the canvas cannot tell apart, because the models bind the i-th call to the i-th mentioned entity at every degree of parallelism. The dominant failure is elsewhere. dLLMs read the number of masks in a slot as the length of its value: one extra mask per slot drops set accuracy from 0.90 to 0.16 and quadruples cross-call errors, which an evaluator would attribute to parallel decoding. Swapping slot lengths between sibling calls TBD. A logit bias toward closing tokens TBD and a one-forward-pass length estimate TBD recover TBD. Template-constrained tool calling with dLLMs must control slot length before it worries about parallelism.

## 图表清单

| 编号 | 内容 | 数据 |
|---|---|---|
| 图 1 | 示意图：exact / +1 / swap 三种槽长下的同一个双调用例子 | +1 已有，swap TBD |
| 图 2 | set_acc 随 k 的变化，s 用不同的线表示（左 Dream，右 LLaDA2.0；AR 用水平线），一张图同时展示 k 的影响很小、s 的影响很大 | s=4、8 和 LLaDA2.0 k=16 TBD |
| 表 1 | RQ1：s=0 时的 k 扫描、对称子集、提及顺序 | LLaDA2.0 k=16 和 Qwen 全量 TBD |
| 表 2 | teacher-forced 长度先验探针 | LLaDA2.0 s=1、8 TBD |
| 表 3 | 槽级错误构成（s=0、1、2、8） | s=4、8 TBD |
| 表 4 | 长度互换 | TBD |
| 表 5 | 缓解：β 和一次前向长度估计 | TBD |
| 表 6 | choose-N 探针 | TBD |

## 从原 proposal 砍掉的部分
- H1–H4 的原始表述，改成第 4 节的检验和负面结果。
- 反事实归因、DVS、命题 1 的形式化表述。
- free 模式的新跑（Qwen、LLaDA2.0），以及块长度对照。

## 复现
- **代码**：分支 `gpu-run`。
- **汇总表**：`results/summary/`，由 `bash scripts/run_minimal.sh summary_length` 生成。
- **原始 JSONL**：不入库。
