# 研究方案：语法正确之后——扩散语言模型并行工具调用的残余错误及其与并行去掩码的因果关系

> 目标会议：**AAMAS 2027**（第 26 届，越南河内，2027-05-03 ~ 05-07），主轨道，主投 **Generative and Agentic AI (GAAI)** 领域
> 版本：v0.1（2026-09-30）

---

## 0. 先看时间线（最重要）

| 事项 | 官方截止（AoE, UTC−12） | 北京时间 | 状态 |
|---|---|---|---|
| 所有作者 OpenReview 账号注册 | 2026-09-17 | 09-18 20:00 | **已过**，必须立刻核对所有合作者账号是否已激活 |
| 摘要注册（100–300 词纯文本） | **2026-10-01** | **10-02 20:00** | 约 2.5 天后 |
| 全文提交 | **2026-10-08** | **10-09 20:00** | 约 9.5 天后 |
| Rebuttal | 11-20 ~ 11-24 | | |
| 通知 | 12-21 | | |

- 篇幅：正文 **≤ 8 页**，参考文献不限；**附录算入 8 页**；补充材料单独提交 zip（≤ 25MB），审稿人不必看。
- 双盲；允许挂 arXiv；需声明 AI 工具使用；有 Reciprocal Reviewer 义务。
- 未进入正式论文集的稿件会自动进入 **Findings of AAMAS 2027**（可选择退出）——这降低了“赶稿”的风险。

**判断**：这个题目的核心是一篇**诊断/分析型论文**，不需要大规模训练，8 天内可以做完。前提是：(1) 手上至少有 2–4 张 A100/H100（80GB）；(2) 范围严格控制在下面的“必做”部分。第 9 节是逐日计划，第 11 节是赶不上时的备选路线。

---

## 1. 一句话选题与贡献

**一句话**：在约束解码已经保证语法正确的前提下，扩散语言模型（dLLM）并行工具调用中剩下的错误主要是**调用之间的协调错误**（重复、遗漏、参数错绑、“嵌合”取值），这类错误由**同一步同时去掩码的相互依赖 token** 造成。我们通过干预实验和逐实例反事实重放证明这一点，并用一个轻量的结构感知调度来修复它。

**预期贡献（写进 Introduction 的 4 条）**：
1. **问题与分类法**：第一个针对“语法已保证”条件下 dLLM 并行工具调用的**残余错误分类法**，区分单调用错误和跨调用协调错误。
2. **理论视角（AAMAS 卖点）**：把一步内并行去掩码建模为**同时行动的去中心化团队决策**：每个被掩码位置是一个“智能体”，只看共享上下文、看不到同伴本步的选择，联合动作按边缘分布之积采样。跨调用错误就是经典的**协调失败（miscoordination）**。由此得到可检验的预测，例如 n 个对称调用在一步内同时提交时，得到合法（无重复、无遗漏）分配的概率上界为 n!/nⁿ。
3. **因果归因方法**：三件套——(a) 剂量–响应干预（只改并行度，其余全固定）；(b) **逐实例反事实重放**（把出错那一步改成逐 token 串行提交，看错误是否消失）；(c) **依赖违背分数 DVS**（同一步提交的 token 对之间的条件依赖强度）。这三者把“并行去掩码导致的错误”与“模型能力不足”“生成顺序”分开。
4. **缓解**（可选但强烈建议）：**冲突组串行化调度**（同一“协调组”内每步最多提交一个 token），在准确率–NFE（函数评估次数）平面上给出 Pareto 改进。

---

## 2. 相关工作与空白

（待文献检索结果补全，见本文件后续版本。）

---

## 3. 理论框架：并行去掩码 = 同时行动博弈

### 3.1 形式化

- 输出画布 x = (x₁, …, x_L)，第 t 步的已提交集合 Cₜ，本步提交集合 Sₜ（|Sₜ| = kₜ）。
- dLLM 在第 t 步对每个 i ∈ Sₜ 给出边缘分布 p_θ(xᵢ | x_{Cₜ}, c)，并**独立**地（按边缘之积）为 Sₜ 中所有位置取值。
- 真实（或模型自身的）联合条件分布为 p_θ(x_{Sₜ} | x_{Cₜ}, c)。二者之差是这一步的**因子化误差**：

  Δₜ = KL( p(x_{Sₜ} | ·) ‖ ∏_{i∈Sₜ} p(xᵢ | ·) ) = Sₜ 上的**总相关（total correlation）**。

- 约束解码只保证最终 x 属于语法语言 𝓛(G)。**语法无法表达“两个调用的 city 不能相同”“每个实体恰好出现一次”这类跨位置的语义约束**，所以 Δₜ 造成的错误在约束之下依然存在。

### 3.2 与多智能体协调的对应（写论文时的核心叙事）

| dLLM 并行解码 | 多智能体系统 |
|---|---|
| 同一步内被提交的各个掩码位置 | 同时行动的智能体 |
| 已提交的上下文 x_{Cₜ} | 公共知识/共享观察 |
| 按边缘之积采样 | 独立（非相关）策略 |
| 跨调用重复/遗漏/错绑 | 协调失败（如对称协调博弈中的错配） |
| 先提交一个调用，再提交其余 | 顺序行动/相关化装置（correlation device）打破对称 |
| 冲突组串行化调度 | 在冲突组内引入“轮流行动”的协调协议 |

### 3.3 可检验的定量预测

**命题 1（对称分配）**：用户请求包含 n 个彼此可交换的实体（如“查询巴黎、东京、纽约的天气”），模型对调用顺序没有偏好（联合分布在 n! 个排列上均匀）。此时每个参数槽的边缘都是 n 个实体上的均匀分布。若这 n 个槽在同一步按边缘之积**采样**，输出是合法排列（无重复、无遗漏）的概率为

| n | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| n!/nⁿ | 0.500 | 0.222 | 0.094 | 0.038 | 0.015 |

若用**贪心**（argmax）提交，完全对称时平局按同一规则打破，所有槽取同一实体，重复率 → 100%。

**推论（调度器依赖性，很重要）**：
- **置信度阈值调度**（如 Fast-dLLM，τ≈0.9）：对称槽的边缘最大值 ≤ 1/n < τ，因此它们*不会*在同一步被提交。完全对称的情况反而被保护。风险出现在**部分对称**的情况：例如多 token 实体共享前缀（"San Francisco" / "San Diego"，"New York" / "New Delhi"），前缀 token 高置信、一起提交后，后缀才出现分歧。这会产生**嵌合取值**（如 "San Diego" 被写成 "San Francisco" 的一半）。
- **固定 k 的低置信度重掩码调度**（LLaDA 默认）：k 大时会强制把低置信的对称槽一起提交，预测重复/错绑显著增加。
- **半自回归分块**（block diffusion / LLaDA block_length）：两个耦合槽若落在不同块中，就不会被同时提交。**块长度是一个天然的干预旋钮**。

**命题 2（顺序先验）**：若提示中实体的提及顺序给了模型强烈的“按提及顺序排列调用”的先验，联合分布集中在恒等排列上，边缘之积≈联合，因子化误差很小。**预测：错误率随“槽分配歧义度”（排列熵）单调上升**。合成探针会显式操控这个变量（见 6.3）。

---

## 4. 研究问题与假设

- **RQ1（是什么）**：语法被保证之后，dLLM 并行工具调用还剩哪些错误？分布与同规模自回归（AR）模型有何不同？
  - **H1**：dLLM 的残余错误中，**跨调用协调错误**的占比显著高于匹配的 AR 模型；单调用错误（函数选错、参数类型错）两者相近。
- **RQ2（是不是并行造成的）**：这些错误在多大程度上由并行去掩码导致，而不是由模型能力或任意顺序生成导致？
  - **H2（剂量–响应）**：跨调用错误率随每步提交 token 数 k（或 1/τ）单调上升；k=1 时降到接近 AR 的水平；单调用错误对 k 不敏感。
  - **H3（逐实例归因）**：在出错的轨迹中，把出错步改成逐 token 串行提交，可以修复相当大比例的跨调用错误；对照组（随机选一个正确步串行化）几乎不改变结果。
  - **H4（机制）**：同一步提交的 token 对的依赖违背分数（DVS）越高，出错概率越高。
- **RQ3（什么时候最严重）**：调用数量 n、槽分配歧义度、实体多 token 程度、跨调用共享参数如何调节并行错误？
  - **H5**：错误率随 n 和歧义度上升，在对称设置下接近命题 1 的预测曲线。
- **RQ4（能否低成本修复）**（可选）：利用语法状态得知“哪些位置属于同一协调组”，限制组内同步提交，能否在保持大部分加速的前提下消除大部分协调错误？
  - **H6**：冲突组串行化在准确率–NFE 上严格优于同 NFE 的基线调度器。

---

## 5. 错误分类法（语法已保证之后）

评估采用**集合级匹配**：把预测调用集合与标准答案调用集合做最大二分匹配（匈牙利算法，代价 = 函数名 + 参数的 AST 不一致程度）。这样“调用顺序不同”本身不算错。这一点必须在论文中明确，因为并行调用本来就是无序集合。

| 大类 | 错误类型 | 自动判定规则（在匹配之后） | 预测是否并行敏感 |
|---|---|---|---|
| **跨调用（协调）** | **重复调用** Duplicate | 两个预测调用函数名与参数完全相同，而标准答案中不存在这样的重复 | 高 |
| | **遗漏调用** Omission | 预测调用数 < 标准数，且缺失的调用所对应的实体被别的调用“占用” | 高 |
| | **参数错绑/交叉** Swap / Cross-binding | 调用 i 的某参数值恰好等于标准答案中调用 j 的对应值（i≠j） | 高 |
| | **嵌合取值** Chimera | 字符串值不在允许集合中，但它是两个正确值的 token 级拼接（如前缀来自值 A、后缀来自值 B） | 很高（dLLM 特有） |
| | **共享参数不一致** Inconsistent shared arg | 标准答案中各调用共享同一取值（如同一 unit、同一 date），预测中不一致 | 中–高 |
| | **调用数错误** Count | 多出调用（非重复）或少于标准数（非错绑造成） | 中 |
| **单调用** | 函数选错 Wrong function | 匹配调用的函数名不同 | 低 |
| | 缺参/多参 Missing / Extra arg | required 参数缺失或出现无关参数 | 低–中 |
| | 取值错误 Wrong value | 值错，但不属于错绑或嵌合 | 低–中 |
| | 类型/格式错 Type / Format | 语法合法但与 schema 类型或格式不符（如日期格式） | 低（约束解码应大幅降低） |
| **拒答/过度调用** | 不该调用却调用 / 该调用却不调用 | 与 BFCL irrelevance / relevance 类似 | — |

- 分类器以规则为主。随机抽 200–300 条由两名作者独立人工标注，报告 Cohen's κ 和规则分类器的准确率。
- 关键指标：**跨调用错误率（CCER）**、**单调用错误率（SCER）**，以及各细类的占比。

---

## 6. 实验设计

### 6.1 模型

原则：**每个 dLLM 尽量配一个“同源” AR 基线**，以排除数据和规模差异。

| dLLM | 匹配的 AR 基线 | 说明 |
|---|---|---|
| Dream-v0-Instruct-7B | Qwen2.5-7B-Instruct | Dream 由 Qwen2.5-7B 初始化，是最干净的对照 |
| LLaDA-8B-Instruct（及 LLaDA-1.5） | LLaMA-3-8B-Instruct（规模、数据量近似） | 从头训练的 MDM，原生 dLLM |
| （可选）块扩散/半 AR 模型，如 SDAR / Fast-dLLM v2 等 | 其初始化来源的 AR 模型 | 用于检验“块长度”这个旋钮 |
| （可选）LLaDA2.0 系列等原生支持工具调用的新模型 | 同规模 Qwen3 | 若本地显存允许 |

具体型号、HF ID 和是否原生支持 tool calling 待文献检索结果确认，后续补入。

**是否微调**：主实验不做训练，直接用 instruct 模型加 few-shot 工具调用模板。如果某 dLLM 在 BFCL simple 上准确率过低（< 40%）导致分析没有意义，再用同一份工具调用数据（如 xLAM/APIGen 子集）对 dLLM 和 AR **做同样的 LoRA SFT**，保持对照公平。这一步列为“风险预案”，不作为必做项。

### 6.2 约束解码

- 输出格式统一为 JSON 数组：`[{"name": ..., "arguments": {...}}, ...]`，由函数 schema 生成 JSON-Schema / 文法。
- 实现优先级：
  1. **复用已有的 dLLM 约束解码实现**（正则/CFG 版本，见第 2 节文献），保证“语法合法”是**硬保证**。
  2. 若 1 来不及：采用**骨架填空（scaffolded infilling）**，这正是 dLLM 的天然能力。先在受限集合中确定调用数和函数名，再把 JSON 骨架（括号、键名）写入画布，只对值槽加掩码；值槽内按类型做 token 级约束（数字/枚举/布尔/字符串）。
  3. 兜底的稳健性检验：不加约束，只保留语法合法的输出做“条件于语法合法”的分析，并报告选择偏差。
- **额外设置——Oracle 骨架**（强烈推荐，也最容易实现）：直接给定调用数和函数名，只让模型填参数值。这时剩下的错误**全部是参数层面的语义错误**，跨调用错绑/重复/嵌合可以被最干净地测量，是 RQ2/RQ3 的主战场。

### 6.3 数据

1. **BFCL（v3/v4）**：`parallel`、`parallel_multiple`，加上 `live_parallel`、`live_parallel_multiple`；`simple`、`multiple` 作单调用对照组。AST 评估改为上文的集合级匹配。
2. **ParaProbe（自建合成可控探针，本文贡献之一）**：程序化生成，每种配置 100–200 条，覆盖以下可控因子：
   - 调用数 n ∈ {2, 3, 4, 6}
   - **槽分配歧义度**：显式列举（“依次查询 A、B、C”）/ 无序提及 / 隐式推导（“北欧五国首都”）
   - **实体 token 结构**：单 token 实体 / 共享前缀的多 token 实体（San Francisco vs San Diego）/ 无共享前缀
   - **跨调用共享参数**：有（同一 date/unit）/ 无
   - **调用间依赖**：独立 / 同一函数不同参数 / 不同函数（parallel_multiple 风格）
   - 标准答案由生成程序给出，可做精确的错误类型判定。
3. （可选）NESTFUL 或类似的“有依赖的调用序列”，作为外部效度检验。

### 6.4 解码配置矩阵（核心干预）

| 因子 | 取值 |
|---|---|
| 调度器 | 低置信度重掩码 + 固定 k（LLaDA 默认）；随机重掩码；从左到右；置信度阈值（Fast-dLLM 风格）；熵/margin 调度（Dream 的 entropy/topk_margin） |
| 并行度 | 每步提交 k ∈ {1, 2, 4, 8, 16}；阈值 τ ∈ {0.99, 0.95, 0.9, 0.7, 0.5} |
| 块长度 | {8, 32, 128, 全长}（半 AR 块使耦合槽“不可能同时提交”） |
| 采样 | 贪心（反事实分析用，可复现）；温度采样（验证命题 1），3 个种子 |
| 约束 | 全文法约束 / Oracle 骨架 / 无约束（稳健性） |

**关键控制**：比较 k=1（完全串行但任意顺序）和“从左到右 k=1”（等价于 AR 顺序），可以把**“任意顺序”效应**与**“同时性”效应**分开。这是回答“是不是*并行*去掩码造成的”的必要对照，否则审稿人会质疑错误来自生成顺序而非同时性。

### 6.5 因果归因三件套

**(A) 剂量–响应干预**：固定模型、提示、种子和约束，只改 k（或 τ、块长度）。画 CCER 和 SCER 随 k 的曲线。H2 预测 CCER 单调上升、SCER 基本平坦。

**(B) 逐实例反事实重放（Counterfactual Sequentialization）**：
1. 贪心解码，记录完整轨迹：每个位置被提交的步号 tᵢ、每步的边缘分布。
2. 对每个出错实例，定位“出错 token”（如重复调用中第二个调用的实体 token）被提交的那一步 t*。
3. 从 t* 之前的状态重新开始，只把 t* 这一步改成**按置信度逐个提交、每提交一个就重新前向一次**，之后仍用原调度继续。
4. 若错误消失 → 归因为“同时性”；若不消失 → 归因为模型/顺序。
5. **安慰剂对照**：在正确实例上随机挑一步做同样处理，统计被“改坏”的比例，作为噪声基线。
6. 报告：可归因比例 = P(修复 | 出错) − P(改坏 | 正确)，按错误类型分解。

**(C) 依赖违背分数 DVS 与共提交分析**：
- 对第 t 步同时提交的 token 对 (i, j)，额外做一次前向：把 xᵢ 填为已提交的值，得到 p(xⱼ | xᵢ, ·)，计算
  DVSₜ(i, j) = KL( p(xⱼ | xᵢ, ·) ‖ p(xⱼ | ·) )，步级 DVSₜ = 对该步所有 token 对求和（或取最大值）。
- 检验：出错步的 DVS 分布显著高于正确步（AUROC）。
- **混杂提醒**（论文里要主动写）：在阈值调度下，“两个槽同时被提交”本身说明它们都是高置信，这是简单样本的标志。所以观察性的“共提交 vs 出错”相关会被样本难度混杂，只能作为辅助证据。**因果结论以 (A)(B) 的干预结果为准**。

### 6.6 缓解方法（RQ4，可选）

1. **冲突组串行化调度（Coordination-Group Scheduling, CGS）**：约束解码器的文法状态已知每个位置属于哪个调用、哪个键。定义协调组 = {各调用的函数名槽}、{各调用中同一键的值槽}。调度规则：**每一步每个协调组内最多提交 1 个 token**，组间仍然并行。几行代码即可实现，插在任何调度器之上。
2. **校验触发的定向重掩码**：检测到重复调用或嵌合值（值不在 prompt 可抽取的实体集合中）时，只重掩码该值槽并串行重解码。
3. 报告准确率–NFE / 墙钟时间的 Pareto 曲线，与 AR、各基线调度器对比。

### 6.7 统计

- 所有比较用配对设计（同一实例、不同解码配置），McNemar 检验或配对 bootstrap（10k 次）给 95% CI。
- 剂量–响应用 logistic 回归：error ~ log k + 错误类型 + 模型 + (1 | 实例)。
- 多重比较用 Holm 校正。

---

## 7. 预期图表（8 页的骨架）

- **图 1（首页 teaser）**：同一请求，AR 输出正确；dLLM 在约束下语法完全合法，但出现重复调用或嵌合值。旁边画出轨迹，标注两个城市槽在同一步被提交。
- **图 2**：错误构成堆叠条形图（dLLM vs 匹配 AR，BFCL parallel / parallel_multiple）→ RQ1。
- **图 3**：剂量–响应曲线，x = 每步 token 数 k（对数轴），y = CCER / SCER，含 k=1 任意顺序和左到右两条参考线 → RQ2。
- **图 4**：ParaProbe 上重复/错绑率随 n 的变化，叠加命题 1 的理论曲线 n!/nⁿ → RQ3。
- **表 1**：反事实重放的归因结果（按错误类型；含安慰剂对照）。
- **图 5**：DVS 的 AUROC / 分布对比。
- **图 6**：准确率–NFE Pareto（含 CGS）→ RQ4。

---

## 8. 论文结构（8 页）

1. Introduction（1 页）：问题、并行工具调用对智能体的重要性、“语法正确 ≠ 调用正确”、贡献列表
2. Background & Related Work（0.75 页）
3. Parallel Unmasking as a Simultaneous-Move Decision（0.75 页）：形式化、命题 1、调度器依赖推论
4. Taxonomy & Evaluation Protocol（0.75 页）：集合级匹配、分类法、ParaProbe
5. Experimental Setup（0.5 页）
6. Results（2.5 页）：RQ1–RQ3
7. Mitigation（0.75 页）：CGS
8. Discussion, Limitations, Conclusion（0.5 页）

**AAMAS 定位**：主领域选 GAAI（涵盖 orchestration and workflows、agentic benchmarks）。叙事上强调：**工具调用是智能体的动作，并行工具调用是联合动作，并行去掩码是“无通信的同时行动”**。关键词：agentic AI, tool use, diffusion language models, parallel action selection, coordination failure。

---

## 9. 8 天执行计划（北京时间）

| 日期 | 任务 | 产出/里程碑 |
|---|---|---|
| **9/30（今天）** | 核对所有作者 OpenReview 账号；确定标题；搭环境，下载模型；跑通 Dream / LLaDA 的 BFCL parallel 推理 | 标题定稿；每个模型 10 条样例能跑出结果 |
| **10/1** | 接入约束解码（或骨架填空）；实现集合级匹配 + 错误分类器；**提交摘要** | 摘要最迟 10/2 20:00 提交；pipeline 端到端可用 |
| **10/2** | ParaProbe 生成器；启动主实验矩阵（BFCL × 模型 × k × 调度器）；AR 基线 | 图 2、图 3 的原始数据开始产出 |
| **10/3** | ParaProbe 实验（n、歧义度、token 结构）；温度采样验证命题 1；人工标注 200 条 | 图 4；κ 值 |
| **10/4** | 反事实重放 + 安慰剂；DVS 计算（在子集上） | 表 1、图 5 |
| **10/5** | CGS 缓解 + Pareto；补跑缺失的格子 | 图 6；**实验冻结** |
| **10/6** | 写作：方法、实验、结果（边写边补图） | 完整初稿 |
| **10/7** | 写作：引言、相关工作、理论节；内部审阅 | 第二稿 |
| **10/8** | 精修、压页、补充材料（代码 + ParaProbe + 完整结果表）、AI 使用声明 | 定稿 |
| **10/9 白天** | 缓冲；**20:00 前提交** | 提交 |

**算力估计**：BFCL parallel 类约 400–450 条，ParaProbe 约 3–5k 条。按 3 个 dLLM × 5 个 k × 3 个调度器 × 贪心 + 3 种子采样子集，约 5–8 万次生成，每次 256 token、单卡 A100 约 2–6 秒（Fast-dLLM 缓存可再加速），共约 50–120 GPU·h。反事实重放和 DVS 只在出错子集上做，另需约 20–40 GPU·h。**4 张卡约 1.5–2 天可跑完**，与上面的日程吻合。

**分工建议（若有 3 人）**：A 负责解码器（约束 + 调度 + 轨迹记录 + 重放）；B 负责数据与评估（BFCL 适配、ParaProbe、分类器、标注）；C 负责理论节、AR 基线、写作主线。

---

## 10. 风险与预案

| 风险 | 预案 |
|---|---|
| dLLM 基础工具调用能力太弱，错误被“啥都不会”淹没 | 以 Oracle 骨架设置为主战场（只填参数，任务更容易）；必要时同数据 LoRA SFT dLLM 与 AR |
| 约束解码实现来不及 | 骨架填空（6.2 第 2 种）；或无约束 + 条件于语法合法（稳健性检验） |
| 阈值调度下几乎看不到并行错误 | 这本身是一个发现（阈值保护对称槽），与命题 1 的推论一致。把重点放在固定 k 调度和“部分对称/共享前缀”的嵌合错误上，并报告速度–错误的权衡 |
| 审稿人质疑“只是 NLP 分析，不是 AAMAS” | 第 3 节的同时行动博弈框架 + 智能体联合动作叙事；GAAI 领域明确欢迎 agentic benchmarks 和 orchestration |
| 已有工作抢先（见第 2 节） | 以文献检索结果为准，调整贡献表述 |
| 8 天写不完 | 砍掉 RQ4（CGS），论文保留 RQ1–RQ3 仍然完整；或按第 11 节换投 |

---

## 11. 备选路线

- **赶不上 10/8**：不要硬交半成品，因为摘要注册不等于必须交全文。可考虑：
  - IJCAI-ECAI 2027 / ICML 2027（通常 1 月截止），或 ACL 系列经 ARR 投稿；
  - 先投 AAMAS / NeurIPS / ICLR 的相关 workshop 占坑，同时完善。
- **交了但被拒**：自动进入 Findings of AAMAS 2027（可选退出），仍是可引用的正式发表。

---

## 12. 今天就要做的事（清单）

- [ ] 所有合作者在 OpenReview 上的账号是否已于 9/17 前完成注册/激活？没有的话马上联系 PC chairs 询问
- [ ] 确认算力（卡数、型号、可用时长）
- [ ] 确定标题，填入第 13 节摘要草稿，10/2 20:00（北京时间）前提交
- [ ] 下载 Dream-v0-Instruct-7B、LLaDA-8B-Instruct、Qwen2.5-7B-Instruct、LLaMA-3-8B-Instruct；拉取 BFCL 数据
- [ ] 分工

---

## 13. 摘要草稿（英文，可直接用于 10/1 摘要注册，约 200 词）

**标题候选**
1. *Syntactically Valid, Semantically Uncoordinated: Diagnosing Parallel Tool-Call Errors in Diffusion Language Model Agents*
2. *When Tokens Move Simultaneously: Parallel Unmasking as a Coordination Failure in Diffusion LLM Tool Calling*

**Abstract**
> Diffusion language models (dLLMs) decode many tokens in parallel, which makes them look naturally suited to agents that issue several tool calls in a single turn. Constrained decoding can already guarantee that dLLM tool calls are syntactically valid, yet syntactic validity does not imply that a set of calls is correct. We study what goes wrong after syntax is guaranteed, and why. We introduce a taxonomy of residual errors that separates single-call errors from cross-call coordination errors (duplicated, omitted, and cross-bound calls, and "chimera" argument values stitched from two valid values), together with a set-level evaluation protocol and a controllable probe suite. We cast a parallel unmasking step as a simultaneous-move decision in which each masked position acts on the shared context without observing its peers, which predicts when factorized sampling breaks the coordination between calls. Using dose–response interventions on the degree of parallelism, per-instance counterfactual sequentialization of decoding steps, and a dependency-violation score, we test whether residual errors are caused by parallel unmasking rather than by model capability or generation order. Finally, we show that a lightweight coordination-group scheduler, which forbids simultaneous commitment within groups of interdependent positions, removes much of this error at a small cost in speed.

> ⚠️ 摘要里的结论性语句（如 “removes much of this error”）在实验结果出来之前是假设，10/8 提交全文时要按实际结果修改。OpenReview 通常允许在全文截止前修改摘要，但请以当年说明为准。

---

## 附：参考链接

- AAMAS 2027 主轨道征稿：https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/call-for-main-track/
- 投稿说明（8 页 + 参考文献、LaTeX、双盲）：https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/instructions/
- Q&A（附录算入 8 页、允许 arXiv、Findings）：https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/qa/
- OpenReview：https://openreview.net/group?id=ifaamas.org/AAMAS/2027/Conference
