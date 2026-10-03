# 论文结构方案（v3，2026-10-03）

v3 按用户 10-03 的决定改写：做 A（把论文改成 agent 论文的框架）和 B（多 agent 对照实验，执行方案见根目录 `EXP_B_PROMPT.md`，由另一个会话在 `exp-b-agents` 分支上跑）。`paper/outline.md` 仍是旧数字的来源；本文件是结构的依据，写作时以本文件为准。

## 0. 结论先行

- 主线不变：先检验“并行解码导致跨调用错误”这个担心，发现范围很窄；再指出真正的原因是槽长，并证明它会伪装成并行错误；最后讨论缓解办法。
- **变的是研究对象**：不再写成“dLLM 的解码诊断”，而是写成“dLLM agent 怎样在一步里选出多个动作（工具调用）”。
  - 协调分析从 0.3 页的补充，变成贯穿全文的主干。
  - 实验 B 让真正的多个 LLM agent 做同一任务，把“同一步提交的槽像不能交流的 agent”从比喻变成证据。
- **原因**：AAMAS 2027 的 GAAI 领域明确写了：“Submissions where the core contribution is not specifically about agents or multiagent systems are out of scope.” 旧框架有被判为不在范围内的风险。

## 1. 定位：这篇论文为什么是 agent 论文

| 审稿人会问 | 我们的回答 |
|---|---|
| agent 在哪里 | 以 dLLM 为决策核心的工具调用 agent。它在一轮里的动作是一组工具调用，也就是一个联合动作。工具调用的接口（骨架）和决策过程（去噪调度）都是 agent 设计的一部分 |
| 和多智能体有什么关系 | 同一步提交的槽是一组同时决策、互不可见的决策者，解码调度就是它们之间的行动协议：k=1 是轮流行动；k≥2 是同时行动；块因果结构是让各个调用轮流行动。我们用多智能体协调的标准概念来分析和预测它们的行为：焦点和约定（提及顺序）、对称情形下的撞车（choose-N）、角色标签（槽位置）、误导性的公共信号（槽长） |
| 这个类比只是比喻吗 | 实验 B 用真正的 n 个 LLM agent，在 {同时行动, 轮流行动} × {匿名, 有编号} 四种协议下做同一个 choose-N 任务，与单画布的 dLLM 并排比较 |
| 对 agent 社区有什么用 | 三点。(1) 设计：动作接口的槽长怎么定、对称的调用要不要串行。(2) 评测：把 dLLM agent 的错误归因到并行之前，必须先控制槽长。(3) 推广：并行决策只有在共享约定能分配角色时才安全，这同样适用于并行领取子任务的多 agent LLM 系统 |

投稿领域选 GAAI。关键词：LLM agents、tool use、parallel action selection、coordination without communication、focal points、diffusion language models。

## 2. 标题与贡献

**标题**（主标题保留，副标题加上 agent 和协调）：
- 推荐：*Masks Are Length Promises: How Diffusion-LM Agents Coordinate Parallel Tool Calls*
- 备选：*Acting in Parallel Without Talking: Focal Points, Collisions, and Length Promises in Diffusion-LM Agents*

**一句话论点**：dLLM agent 在一步里发出多个工具调用时，各调用像不能交流的同时行动者。请求里的提及顺序充当焦点，所以并行几乎没有代价。没有焦点时它们会撞车，和真正的不通信 agent 一样，轮流行动可以避免撞车。最常见、代价最大的失败来自动作接口上的槽长：槽长是一种它们会遵守的承诺，槽长一错，即使每步只提交一个 token，也会绑错值，而且看起来像并行造成的协调失败。

**贡献**（按正文顺序）：
1. **一个可检验的协调分析**：把 dLLM agent 的并行工具调用建模为同时决策，并用实验检验它的预测。
   - 请求有顺序时，k 从 1 到 16 只降 5.2 个点，提及顺序的遵守率为 94–98%，与 k 无关。
   - 请求对称时，同一步提交会撞车：k≥2 时 100% 重复，k=1 时为 0%。
   - 这一点与真正的不通信 agent 团队一致（实验 B，TBD），轮流行动则能消除撞车。
2. **mask 是长度承诺**：动作接口的槽长被当成值的长度。每个槽多一个 mask，set_acc 就从 0.90 掉到 0.16；只差一点最危险，探针解释了原因。
3. **误归因**（新颖性最强）：错误的槽长会在 k=1 下制造出被归咎于并行的协调失败。
   - swap 时 79% 的槽写入长度合适的兄弟值；
   - 用估计长度时，k=1 的跨调用错误约为 k=16 + exact 长度的 4 倍。
   
   由此给出 dLLM agent 的评测规程。
4. **设计建议**：收尾偏置只能补救大余量；一次前向估计只能恢复到 0.42；对称的调用应该轮流行动。

贡献最后一句说明与已有工作的关系：我们不声称发现了长度偏差（CAL、ρ-EOS、DAEDAL、S³），而是在 agent 的多槽动作中测量它，并证明它会伪装成协调失败。

## 3. 章节结构（正文 8 页，参考文献另计）

### §1 Introduction（约 1 页，含图 1）
1. agent 常在一轮里同时发出多个工具调用，这是一个联合动作。dLLM 每步提交多个 token，看起来很适合做 agent 的决策核心。担心在于：同一步提交的 token 是同时行动、互不可见的决定，可能重复或错绑。Lu et al. (2026) 报告 dLLM agent 调用工具失败，归因于扩散噪声，但没有受控实验。
2. 没人控制的混杂因素：动作接口（骨架）的槽长必须事先定好，而值的长度是未知的，所以长度错配是常态。三种定长方式对应三种错误。
3. 我们的做法：把一轮并行调用看作同时决策，从协调理论推出预测；独立操控行动协议（k、块因果）和接口（槽长）；再用真正的多 agent 团队做对照。
4. 发现（三小段）和贡献（第 2 节）。
- 图 1：`parallel_20` 在三种槽长下的输出，标注 “k=1: one token per step”。

### §2 Related Work（约 0.6 页）
- **LLM agent 与并行工具调用**：LLMCompiler（并行函数调用）、BFCL 的 parallel 类别、dLLM agent（Lu et al. 2026；DLLM Agent；DLLM-Searcher；Agents of Diffusion，AAMAS 2026）。
- **无通信协调**：Schelling 焦点、Lewis 的约定、dispersion games、zero-shot coordination（Other-Play）。我们把这些概念用到一个 agent 内部的并行决策上，并用真正的 agent 团队做对照。
- **dLLM 的并行解码与长度**（压缩）：因子化误差（Fast-dLLM、ParallelBench、EB-Sampler）；长度问题（CAL 只做单段、ρ-EOS、DAEDAL、S³）；约束解码（Dang & Ermon、Mündler et al. 等）。
- 每篇文献引用前都要逐篇核对。

### §3 A dLLM Agent's Parallel Turn as Simultaneous Decisions（约 1 页）
- **3.1 agent 与动作接口**：请求、工具、一轮的动作 = 一组调用；骨架把调用数、函数名和参数名固定下来，只把值留成槽；收尾规则；AR agent 用同一接口。
- **3.2 决策过程即行动协议**：
  - 槽是决策者，同一步提交是同时行动；
  - k=1 是逐个行动；k≥2 是成组同时行动；
  - LLaDA2.0 的块因果结构让各个调用轮流行动；
  - τ=0.9 是按置信度自适应地决定谁先行动。
- **3.3 协调线索与预测**：能区分各个槽的只有共享线索：请求的提及顺序（焦点）、槽的位置（角色标签）、槽长（公共信号）。
  - **P1**：有顺序的请求下，k 影响很小。
  - **P2**：对称请求下，同一步提交会撞车，轮流行动不会；行为与不能交流、只有角色标签的 agent 团队一致。
  - **P3**：槽长是会被遵守的信号；信号错了就会绑错值，与 k 无关。
- **3.4 模型、数据、指标、两个操控变量**：k、槽长（exact / +s / swap / estimate）。

### §4 Ordered Requests: Same-Step Commits Coordinate on Mention Order（约 0.8 页；P1）
- 表 1：k 扫描，加上长度对称子集、提及顺序和 NFE。
- 要点：
  - Dream −5.2 个点，CCER ≤ 2.0%；
  - 前向次数少约 13 倍，τ=0.9 只要 1.86 次前向，只降 0.7 个点；
  - 提及顺序遵守率 94–98%，与 k 无关。

### §5 Symmetric Requests: Same-Step Slots Collide Like Agents That Cannot Talk（约 0.9 页；P2，含实验 B）
- 表 2 合并了 choose-N 和实验 B：

| | list：重复 | list：按列出顺序 | open：重复 |
|---|---|---|---|
| n 个 Qwen agent：sim-anon / sim-label / turn-anon / turn-label | TBD | TBD | TBD |
| n 个 Dream agent：同上四种协议 | TBD | TBD | TBD |
| 一张 Dream 画布，同一步提交（k=2/4/16） | 10% / 15% / 15% | 32% / 12% / 12% | 100% |
| 一张 Dream 画布，每步一个 token（k=1） | 0% | 48% | 0% |
| 一张 LLaDA2.0 画布（块因果） | 0% | 97–100% | 0% / 4.4% / 4.4% |
| 一次 Qwen 生成（AR） | 0% | 82% | 0% |

- 叙述顺序：先讲单画布的结果（所有重复都来自同一步提交），再讲 agent 团队的四种协议，最后逐行对应：同一步提交的槽 ↔ 有角色标签、但看不到彼此的 agent；k=1 和块因果 ↔ 轮流行动。
- B 的结果与预测相反时原样报告，并相应收窄“类比成立”的表述。
- 注意：open 变体受 1 token 槽长限制，只报告重复率。

### §6 Masks Are Length Promises（约 0.9 页；P3 的前一半）
- 图 2a：set_acc 随 s 的变化。Dream k=1 时为 0.897/0.155/0.033/0.170/0.600；槽级准确率从 0.954 降到 0.340；同一尺度对比：k 降 5.2 个点，s 降 74.2 个点。
- 多出来的 mask 被填了什么：overfill 0.351，抄兄弟值 0.005。图 1 第二行是金额 ×10。
- 图 2b：探针显示 P(收尾) 在 s=1 时只有 0.010，在 s=8 时为 0.740，解释了“只差一点最危险”。这里写关于 S³ 的论证（第 6.4 节）。

### §7 Length Errors Masquerade as Coordination Failures（约 1 页；P3 的后一半，核心贡献）
- **7.1 swap**（表 3）：79.4% 写入兄弟值（exact 下为 0%）；长或短两个方向都一样；LLaDA2.0 为 47.5%（34 条）。
- **7.2 信号压过焦点**：两种线索冲突时，模型约 15 : 1 跟随长度（第 6.3 节）。
- **7.3 评测者会看到什么**（表 4，伪装表）：估计长度下约为 4 倍；嵌合值和重复调用在 k=1 下也会出现。
- **7.4 对 agent 的后果**：错误的动作做了什么。例如作用到错的实体、同一动作执行两次、金额差 10 倍以上。各类的比例用新脚本统计（第 7 节待做第 2 项）。
- **7.5 评测规程**：把错误归因到并行之前，先用 swap 或 surplus 做长度对照。

### §8 Designing and Evaluating dLLM Agents（约 0.6 页）
- 表 5：收尾偏置 β × s，以及一次前向估计。
- 三条建议：
  1. 接口：不要“差一点”，要么精确，要么留出大量余量并配合收尾偏置。
  2. 协议：对称的调用要轮流行动（k=1 或块因果），或者给它们可以区分的线索。
  3. 评测：先控制长度，再归因。

### §9 Discussion, Limitations, Conclusion（约 0.5 页）
- 局限：
  - oracle 骨架：free 模式下为 0.752，骨架下为 0.880；
  - 只测了两个 dLLM；LLaDA2.0 用了子集；
  - 只用贪心解码（B2 的采样结果视完成情况补充）；
  - choose-N open 变体受 1 token 槽长限制；没有跑 S³ 式 prompt 基线；骨架的分词边界与自然生成不同。
- 推广：多 agent LLM 系统并行领取子任务时，会遇到同样的撞车和信号问题。

## 4. 图表清单

| 编号 | 内容 | 位置 | 来源 |
|---|---|---|---|
| 图 1 | `parallel_20` 在 exact / +1 / swap 下的输出，k=1 | §1，通栏 teaser | 原始 JSONL |
| 图 2 | (a) set_acc 随 s 的变化（k ∈ {1,4,16}，两个模型，AR 用标记点）；(b) P(收尾) 随 s 的变化 | §6 | `length.md`、`length_prior.md` |
| 表 1 | k 扫描 | §4 | `length.md`、`symmetry_*.md` |
| 表 2 | choose-N 与 agent 团队 | §5 | `choose.md`、`agents.md`（B，TBD） |
| 表 3 | swap 的槽级构成 | §7 | `swap_*.md` |
| 表 4 | 伪装表 | §7 | `masquerade.md` |
| 表 5 | 缓解 | §8 | `length.md`、`length_prior.md` |

篇幅紧张时，表 3 可以压成正文里的一句话加表 4 里的一行。

## 5. 篇幅预算

| 部分 | 页数 |
|---|---|
| 标题、摘要、关键词 | 0.35 |
| §1 | 1.0 |
| §2 | 0.6 |
| §3 | 1.0 |
| §4 | 0.8 |
| §5 | 0.9 |
| §6 | 0.9 |
| §7 | 1.0 |
| §8 | 0.6 |
| §9 | 0.5 |
| **合计** | **7.65** |

## 6. 核对结果与新数据

### 6.1 图 1 的样例：`parallel_20`（Dream，k=1，即每步只提交一个 token）

请求：*Can you check my loan eligibility for a home loan of amount $500,000 from HSBC with annual income $100,000 and for Wells Fargo for a amount of $700,000 with annual income of $120,000?*

| 槽长 | 输出（两个调用的 institution / loan_amount / annual_income） | 判定 |
|---|---|---|
| exact | HSBC / 500000 / 100000；Wells Fargo / 700000 / 120000 | 正确（k=4、16 也正确） |
| +1 | **HSBC Bank** / **5000000** / **1000000**；**Wells Fargo Bank** / **7000000** / **1200000** | 每个值都写长了；金额变成 10 倍 |
| swap | **Wells Fargo** / 500000 / 100000；**HSBC** / 700000 / 120000 | 两家银行错绑。这正是“并行解码造成的跨调用错误”的样子，但这里是 k=1 |

- 这个样例同时解决了审稿 B 指出的问题：原定的 “Taylor Swift. Swift” 是自我重复，不是抄兄弟值，和图 1 想说明的事对不上。
- 金额 ×10 有现实风险，引言可以直接用：一个多余的 mask 把 50 万的贷款申请变成了 500 万。
- 备选：`parallel_47`（Barcelona / Man United 的联赛互换）、`parallel_52`（Vegetarian / Delivery 互换），两者在 swap、k=1 下都错绑。
- LLaDA2.0 在同一题 +1、k=1 下写出 “HSBC Fargo” / “Wells Fargo Fargo”，是嵌合值。§7 可以引用。

### 6.2 新表：协调失败标签的“伪装”（`scripts/masquerade.py` → `results/summary/masquerade.md`）

只用现有 JSONL 里的诊断标签计算（CPU，不需要 tokenizer）。每一对比较都在同一批题上计算。

| 模型 | 条件 | 题数 | 重复调用 | 参数错绑 | 嵌合值 | 共享参数不一致 | CCER | set_acc |
|---|---|---|---|---|---|---|---|---|
| Dream | **exact，k=16**（最并行） | 400 | 0.5% | 0.7% | 0.0% | 0.7% | 2.0% | 0.845 |
| Dream | estimate，k=1 | 400 | 1.5% | 3.5% | 1.0% | 2.2% | **7.5%** | 0.422 |
| Dream | +1，k=1 | 400 | 0.5% | 0.7% | 3.0% | 1.5% | 5.8% | 0.155 |
| Dream | exact，k=16 | 165 | 0.0% | 0.6% | 0.0% | 0.6% | 1.2% | 0.891 |
| Dream | swap，k=1 | 165 | 0.0% | 15.2% | 1.8% | 0.0% | **16.4%** | 0.673 |
| LLaDA2.0 | **exact，k=16** | 100 | 1.0% | 1.0% | 0.0% | 1.0% | 3.0% | 0.870 |
| LLaDA2.0 | estimate，k=1 | 100 | 0.0% | 6.0% | 2.0% | 7.0% | **14.0%** | 0.390 |
| LLaDA2.0 | +1，k=1 | 100 | 2.0% | 1.0% | 7.0% | 4.0% | 12.0% | 0.160 |
| LLaDA2.0 | +8，k=1 | 100 | 11.0% | 26.0% | 1.0% | 3.0% | 36.0% | 0.380 |
| LLaDA2.0 | exact，k=16 | 34 | 2.9% | 2.9% | 0.0% | 2.9% | 8.8% | 0.794 |
| LLaDA2.0 | swap，k=1 | 34 | 0.0% | 29.4% | 17.6% | 0.0% | **44.1%** | 0.176 |

（+2、+4 的行见 `masquerade.md`。Dream +8，k=1 的 CCER 为 1.8%，与 k=16 的 2.0% 持平；这是唯一不高于参照的一行，要原样报告。）

要点：
- **最现实的条件**（不用 gold 长度、一次前向估计槽长）下，每步只提交一个 token 产生的跨调用错误，Dream 是 16 token/步 + exact 长度的 3.8 倍（7.5% vs 2.0%），LLaDA2.0 是 4.7 倍（14% vs 3%）。
- ParallelBench 所说的“并行特有”的嵌合值（“New City”）在 k=1 下也会出现：Dream +1 为 3.0%（k=16、exact 为 0%），LLaDA2.0 +2 为 11%。
- 并行担心里最典型的重复调用：LLaDA2.0 +8、k=1 时为 11%，k=16、exact 时为 1%。

### 6.3 长度线索压过顺序线索

swap 时两种线索互相冲突：顺序线索指向自己的值（own），长度线索指向长度合适的兄弟值（sibling_fit）。
- Dream k=1：sibling_fit 79.4%，own 5.4%，约 15 : 1。
- LLaDA2.0 k=1：47.5% vs 12.5%，约 4 : 1。

同一批槽在 exact 长度下 own 为 96.3%。在 §4 中，模型在 94% 以上的情况下按提及顺序填槽。所以两种线索冲突时，模型跟随的是长度。这一点直接对应 P3，写进 §7.2。来源：`swap_*.md`。

### 6.4 核对文献和规则后要改的地方

| 事项 | 原写法 | 核对结果 | 怎么改 |
|---|---|---|---|
| arXiv 2601.12979 | 写作 “DiffuAgent” | 论文是 Lu et al., *The Bitter Lesson of Diffusion Language Models for Agentic Workflows*, ACL 2026；DiffuAgent 是其中的评测框架名 | 引作 Lu et al. (2026)。他们把失败归因于 “fail to maintain symbolic precision … under diffusion noise”，没有受控实验 |
| CAL（2602.00476） | “CAL 式一次前向估计” | CAL 只做**单段**填空（代码和文本），用校准后的首步置信度做双向爬山搜索，平均每段多做 11–18 次前向；论文明确把多段填空列为未来工作 | 我们的估计改称 “a one-forward closing-hazard estimate”，**不能**说成 CAL。要说明 CAL 的搜索可能更准，但每个槽要多花约 10 倍前向，我们没有测。CAL 自己承认多段是空白，引言和相关工作可以直接引用这一点 |
| S³（2507.04504） | “null token” | S³ 在 prompt 里要求模型用语义词 `null` 占位，只测了 LLaDA 和 WikiBio；它也报告过 mask 给多时会产生幻觉内容 | 原论证“P(pad)≈1e-5，所以 null 没用”**并不能覆盖 S³**，因为它的 null 是模型会自然写出的词。改用这条论证：收尾符本身就是 JSON 里自然的“值结束”记号，写出一次之后剩余位置全部填 pad，模型只需要做一次“停”的决定；探针测的正是这一决定，在 s=1 时概率只有 0.010。另外在局限里写明没跑 S³ 式 prompt 基线 |
| AAMAS 2027 篇幅 | “以 CFP 为准” | 官方说明：至多 8 页，另加任意页数的参考文献；补充材料为单个 zip，不超过 25MB，审稿人可以不看；必须用 LaTeX（`aamas_2027_template.zip`）；双盲 | 8 页包括一切，没有附录空间 |
| AI 使用声明 | 没有提到 | AAMAS 2027 规定：AI 工具参与假设或方法设计时，必须在正文或补充材料中详细说明，包括所用 prompt、工具和版本 | 本项目的代码、实验设计和运行都大量使用了 Claude。需要写声明，建议放在补充材料，正文加一句指向它。**要用户决定写什么** |
| Lu et al. 的“没有受控实验” | 写作 “without a controlled experiment” | 原文附录 D.4 有一个 100 条 BFCL 输出的受控研究（schema guardrails），附录 D.1 比较了 APD、D2F、DCD 三种解码器；但没有改变槽长、每步提交的 token 数或并行程度。归因原话：“fail to maintain symbolic precision (e.g. strict JSON schemas) under diffusion noise”，§4.4 又说 “parallel decoding weakens causal dependency” | §1、§2 写成：他们报告 dLLM agent 调用工具失败（多为 JSON 格式错误和参数错误），归因于扩散噪声和并行解码削弱因果依赖，但没有改变能隔离原因的解码因素（槽长、每步提交的 token 数） |
| Dream 的初始化 | “Qwen2.5-7B-Instruct, the model Dream starts from” | Dream 从 Qwen2.5-7B **base** 初始化（官方博客链接到 Qwen/Qwen2.5-7B） | §3 已改为 “the instruction-tuned version of the base model that Dream is initialized from” |
| LLaDA2.0-mini | “1.4B active … trained for tool use” | 论文只写 16B MoE；“1.4B activated” 出自官方模型卡；论文没说为工具调用训练，只报告 BFCL v3 70.90；模型卡写 “Supports tool calling”。官方推理设置：块长 32、阈值 0.95 | §3 已改为 “supports tool calling” |
| ParallelBench 与“嵌合值” | “the error most specific to parallel decoding” | 论文没有 chimera 这个词，也没有说它“最特有”；例子是 “New City”（来自 New York / Mexico City），归因于并行解码的条件独立 | §7 已改：chimera 是我们的术语，引用他们的例子和归因 |
| DAEDAL | “adjust the end of the whole sequence” | 第一阶段在末尾追加 mask；第二阶段还会在低置信位置插入 mask。两者都只对单个回复定长，不处理多个字段 | §2 写成：从 EOS 信号给单个回复定长（ρ-EOS 伸缩尾部；DAEDAL 扩展尾部并在低置信位置插入 mask） |
| Agents of Diffusion（AAMAS 2026） | “dLLM agents” | dLLM 本身不是 agent：两个 AR LLM agent（prompt 优化器和评判者）用语言反馈的多智能体 RL 引导冻结的 LLaDA-8B 生成符合 schema 的 JSON 数据 | §2 按此描述，不能说成 dLLM agent |
| BFCL 的匹配 | “matched one to one, whatever their order” | 成立（App. H）；官方代码是贪心的先到先配，不是最大二分匹配，只在一个预测调用能满足多个参考调用时有差别 | 不用改 |

## 7. 待做

CPU 部分由本会话做，实验 B 由另一个会话做。

1. ✅ `scripts/masquerade.py`（表 4）。
2. ✅ `scripts/consequences.py`：统计错误动作的后果，按条件分别计算。类别包括：作用到错的实体、同一动作执行两次、数值参数差 10 倍以上、其他错误值。用于 §7.4。
3. `scripts/stats.py`：对关键比较做配对 bootstrap 置信区间和 McNemar 检验，包括 LLaDA2.0 的子集，以及 B 里各协议之间的比较。
4. ✅ 图 1、图 2 的生成脚本，输出到 `paper/figures/`。表 5 由 `paper_tables.py` 的 `table_mitigation` 生成，§8 引用的估计误差分布在 `paper/tables/estimate.md`。
5. 引用前复核：
   - ✅ 引用清单定稿，见 `paper/references.md`（43 条，均已核对）。
   - ✅ ParaProbe 查无此文，“实体绑错为零”删掉不写。
   - ✅ LLaDA2.0 兄弟槽的同块比例已在当前骨架下重算（`block_share.md`：3.1% / 2.3%）。
   - 同一参数取最长值的 pilot（17/18）是我们自己的数据，要用的话先在结果里复核。
6. 实验 B：方案见 `EXP_B_PROMPT.md`，结果推到 `exp-b-agents`，最晚 10-05 22:00。拿到后合并 `results/summary/agents*.md`，填表 2。
   - ✅ 10-03 结果已合并（`results/summary/agents.md`、`choose_sample.md`、`choose_focal.md`），表 2 由 `paper_tables.py` 的 `table_choose` 生成，§5 已写。
   - 与预测对照：同时匿名 100% 撞车（理智检查通过）；轮流 0%（符合）；**同时有编号在 list 上与预测相反**（Qwen 100%、Dream 78.3% 撞车，按列表顺序的只有 3.3% / 0%）。同时行动的 Qwen agent 全部选列表第一个城市；T=0.7 采样也打不破对称（list 75–100%）。
   - 论点的改法（用户 10-03 同意）：§5 标题改为 “Same-Step Slots Collide When Nothing Orders the Choices”；协议维度可以类比，标签维度不能：prompt 里的编号几乎分不开 agent，而画布上的槽位置能分开同一步提交的槽（k=16 时 list 槽对 94.5% 不同）。P2 加上编号那半句，并在 §5 如实报告它不成立。
   - **B 推迟时的预案**：10-06 中午仍没有结果，§5 只写单画布的 choose-N，删掉表 2 里的 agent 行，把“与 agent 团队一致”改为“与预测一致”，第 1 节和 §1 里提到实验 B 的地方也一并删掉。

## 8. 审稿人可能追问的问题和回应位置

| 追问 | 回应位置 |
|---|---|
| 核心贡献不是关于 agent 的 | 第 1 节的定位；§3 的协调分析；§5 的 agent 团队对照；§8 的设计建议；§9 的推广 |
| 把槽说成 agent 只是比喻 | §5 的实验 B；正文用词上把槽称为 decider，不叫 agent |
| 长度错配是你们人为造出来的 | §1 第 2 段；§3.1；estimate 条件（§7.3、§8） |
| oracle 骨架不现实 | §3.1 的设计理由；§9 给出 free 模式的数字 |
| 长度偏差已有人报道（CAL、S³） | §2；贡献最后一句；第 6.4 节 |
| 为什么没有 null token 基线 | §6 的收尾符论证；§8 的 β；§9 局限 |
| LLaDA2.0 样本太少 | 标出题数，给置信区间 |
| 集合匹配掩盖了错误 | 所有主要结论都同时报告槽级指标 |
| 只用贪心解码 | §9；如果做了 B2，补充采样结果 |

## 9. 时间表（北京时间；截稿 10-09 20:00）

| 日期 | 任务 |
|---|---|
| 10-03（六） | A 的结构改写 ✅；B 的执行方案 ✅（交给另一个会话） |
| 10-04（日） | 搭 AAMAS 2027 LaTeX 骨架；`consequences.py`、`stats.py`、图表脚本；写 §3 |
| 10-05（一） | §4、§6；B 的结果晚上到 |
| 10-06（二） | §5（填入 B 的结果）、§7、§8 |
| 10-07（三） | §1、§2、摘要；逐篇核对参考文献 |
| 10-08（四） | §9；通读；压页；模拟审稿；AI 使用声明；补充材料 zip（匿名化） |
| 10-09（五） | 缓冲；更新 OpenReview 上的标题和摘要；20:00 前提交 |

## 10. 写作约定

- **agent 一词只用于整个 LLM agent**，包括实验 B 里的 agent。槽称为 *slot-level deciders*，同一步提交称为 *simultaneous moves*，以免和实验 B 里真正的 agent 混淆。
- 术语：value slot、sibling group、surplus *s*、closer、pad、exact / +*s* / swap / estimate、*k*（tokens committed per step）。“k=1” 写成 *one token per step*，不写 *sequential*，以免和 left-to-right 混淆。
- 数字格式：准确率和各种比率统一用百分数，保留一位小数；差值用 points；只有探针的概率用小数。
- 没跑出来的数字一律标 TBD；结果与预期相反时原样报告。

## 11. 需要用户决定

1. 标题：推荐版还是备选版（第 2 节）。
2. LaTeX 写在本仓库 `paper/` 下，还是 Overleaf。
3. AI 使用声明的内容（AAMAS 2027 强制要求）。
