# 研究方案：语法正确之后——扩散语言模型并行工具调用的残余错误及其与并行去掩码的因果关系

> 目标会议：**AAMAS 2027**（第 26 届，越南河内，2027-05-03 ~ 05-07），主轨道，主投 **Generative and Agentic AI (GAAI)** 领域
> 版本：v0.5（2026-09-30）——采用 minimal 实验方案（Dream-7B + LLaDA2.0-mini），LLaDA-8B 移出；摘要同步

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

**一句话**：在约束解码已经保证语法正确的前提下，扩散语言模型（dLLM）并行工具调用中剩下的错误主要是**调用之间的协调错误**（重复、遗漏、参数错绑、“嵌合”取值），这类错误由**同一步同时去掩码的相互依赖 token** 造成。我们通过干预实验和逐实例反事实重放来检验这一点。

**预期贡献（写进 Introduction 的 3 条）**：
1. **问题与分类法**：第一个**定量的**、按 BFCL 类别拆分的“语法已保证”条件下 dLLM 工具调用**残余错误分类法**，重点是并行调用特有的跨调用协调错误。已有工作只做了定性举例（Dang & Ermon 2026）或错误被格式错误主导（Lu et al., ACL 2026），见第 2 节。
2. **理论视角（AAMAS 卖点）**：把一步内并行去掩码建模为**同时行动的去中心化团队决策**：每个被掩码位置是一个“智能体”，只看共享上下文、看不到同伴本步的选择，联合动作按边缘分布之积采样。跨调用错误就是经典的**协调失败（miscoordination）**。由此得到可检验的预测，例如 n 个对称调用在一步内同时提交时，若联合分布对调用顺序对称，得到合法（无重复、无遗漏）分配的概率只有 n!/nⁿ。
3. **因果归因方法**：三件套——(a) 剂量–响应干预（只改并行度，其余全固定）；(b) **逐实例反事实重放**（把出错那一步改成逐 token 串行提交，看错误是否消失）；(c) **依赖违背分数 DVS**（同一步提交的 token 对之间的条件依赖强度）。这三者把“并行去掩码导致的错误”与“模型能力不足”“生成顺序”分开。**核心论点：并行度可能不改变总准确率，却改变错误的构成**，这正好回应 Dang & Ermon 在总准确率层面观察到的“约束解码对步数不敏感”。

---

## 2. 相关工作与空白（新颖性检查结果）

> 标 ✅ 的已由我直接在 arXiv 核对过标题、作者、日期和摘要；其余来自检索代理的汇总，**引用前请逐篇打开核对**，特别是数字。

### 2.1 最有威胁的两篇（论文中必须正面引用和区分）

**A1. Dang & Ermon, “Constrained Decoding for Diffusion Language Models via Efficient Inference over Finite Automata”，arXiv 2607.07026（2026-07-08）✅**
- 内容：对任意有限自动机约束，给出从**约束化的平均场（mean-field）后验**中精确采样的算法；支持任意重掩码调度下的并行和分块解码。在 Dream-7B / LLaDA-8B 上评测 BFCL、xLAM 等。BFCL-Live：贪心 63.9 → 71.5，采样 22.3 → 69.0。
- **威胁 1**：据代理报告，附录 D 讨论了“约束满足后仍然出错”的样例（取值不匹配、参数过度具体化、枚举合法但错误）。**只有定性举例，没有计数，没有按 BFCL 类别拆分，没有并行调用专门分析，也没有归因到并行去掩码。**
- **威胁 2**：据代理报告，其图 4 显示约束解码下准确率**对去噪步数不敏感**。在总准确率层面，这是一条**不利于我们假设**的证据。
- **我们的回应**：(1) 我们测的是**错误类型的构成**如何随并行度变化，而不仅是总准确率；(2) 我们专门看 parallel / parallel_multiple，那里调用之间的依赖最强，而总体准确率被 simple 类主导；(3) 除步数外，我们还干预块长度、调度器和置信度阈值；(4) 他们采样的仍是**平均场**（各位置独立）后验，自动机约束只保证语法，并不强制跨调用一致（如“两个调用的 city 不能相同”），所以协调错误在他们的方法下依然可能存在。**如果他们开源了代码，直接用它作为我们的约束解码器**，最有说服力。

**A2. Lu, Ding, Zhang, Zhang, Tao, “The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive Reality Check”，ACL 2026 Main，arXiv 2601.12979 ✅**
- 内容：在 AgentBoard 和 BFCL-v3（含 parallel / parallel-multiple）上评测 LLaDA、Dream 等 dLLM，结论是 dLLM 目前不适合作智能体骨干，工具调用中“在扩散噪声下无法保持符号精度（如严格 JSON schema）”。提出 DiffuAgent 框架。
- 据代理报告：其错误分析以 JSON 格式错误和参数错误为主；把原因**笼统地**归于“并行解码削弱因果依赖”，**没有做受控实验**。
- **区别**：他们**没有保证语法**，所以错误被格式错误淹没；他们的“并行导致”是断言，我们要做的是检验。

### 2.2 机制层面最相关

**ParallelBench（Kang et al., ICLR 2026, arXiv 2510.04767）✅**：从信息论角度分析并行解码的条件独立假设，在可解析的合成列表任务上展示强依赖时的质量退化（例如 “New York” + “Mexico City” → “New City” 这类嵌合，以及去重/排列类错误随并行度上升）。**这是我们“嵌合取值”和“重复/遗漏”预测的直接先例，但它不涉及工具调用，也没有约束解码。** 我们的定位：把 ParallelBench 的机制带到智能体工具调用场景，并在语法已保证的条件下做逐实例因果归因。

**并行解码误差的理论**（用于第 3 节）：
- **Fast-dLLM**（ICLR 2026, arXiv 2505.22618）定理 1：若同时提交的各位置边缘概率都 > 1−ε 且 (n+1)ε ≤ 1，则边缘之积的贪心结果与联合分布的贪心结果一致。这正是 3.3 中“阈值调度保护对称槽”推论的理论依据，直接引用。
- **EB-Sampler**（NeurIPS 2025, arXiv 2505.24857）把误差分解为“模型误差 + 联合依赖误差”。这与我们“模型能力 vs 并行去掩码”的归因框架同构，可以直接借用它的术语。
- ParallelBench 定理 1、Li & Cai（arXiv 2505.21400）、Chen et al.（arXiv 2511.04647）、Wen et al.（arXiv 2608.25505）：误差下界 = 各步条件总相关之和。**第 3.1 节的 Δₜ 就是这个量，要引用而不是当成自己的贡献。**
- Feng et al.（arXiv 2502.09622）：要控制**序列级**错误率，步数须随长度线性增长。JSON 工具调用正是“一处错就全错”的序列级任务。

**依赖度量（与我们的 DVS 最接近，必须引用）**：
- **DEMASK**（arXiv 2604.02560）定义 D_ij = 揭示 j 后 i 的预测分布的 TV 变化，并开源代码。
- **Wen et al.**（arXiv 2608.25505）的 pseudo-cost：对同一轮揭示的 token 逐个重新打分，求 log 概率差之和。
- 两者都**只用于设计调度器或衡量整体质量，没有用于逐个错误的归因**。所以 DVS 不宣称是新度量，而定位为“把 DEMASK / pseudo-cost 用于错误归因”。**我们的方法新意在 (A) 剂量–响应 + (B) 反事实重放 + 与工具调用错误类型挂钩。**

依赖感知解码器（DAPD、DAWN、CoCommit、Mean-Field Parallel Decoding 等）是方法类工作，没有工具调用评测，作为背景引用。

### 2.3 其他相关（中低威胁，待核对）

- **LLaDA2.0**（arXiv 2512.15745 ✅，16B-A1.4B 的 mini 版原生支持工具调用，模型卡报告 BFCL 70.90 ✅）；**LLaDA2.1**（arXiv 2602.08676）据报告在 speedy / quality 两种模式下 BFCL v3 有 74.86 vs 75.61 的差异（每次前向提交 token 数不同）。这是“并行度越高，工具调用略差”的**总量层面**证据，但没有错误分析。
- **DLLM Agent**（arXiv 2602.07451）、**DLLM-Searcher**（arXiv 2602.07035）：dLLM 搜索智能体，错误分析停留在格式层（无法解析、缺少 tool_call 等），单轮单调用。
- **Chavan**（arXiv 2609.23742，2026-09-20）：“约束解码消除结构错误后暴露语义差距”，做了约束后的错误分类，但对象似乎是 AR 小模型。说明“语法之后还剩什么”这一问法本身不新，**我们的新意必须落在 dLLM 特有的并行去掩码归因上**。
- **Khanal et al., “Agents of Diffusion”**（arXiv 2601.07152，**AAMAS 2026**）：用多智能体 RL 引导 dLLM 生成符合 schema 的 JSON。说明 AAMAS 接收 dLLM 相关工作，可以引用来论证与会议的契合度。
- dLLM 约束解码系列（均未对 BFCL 并行调用做分类分析）：
  - DINGO（NeurIPS 2025, arXiv 2505.23061）：正则约束，代码 github.com/uiuc-focal-lab/DINGO
  - **Mündler, Dekoninck, Vechev**（ICLR 2026, arXiv 2508.10111）：CFG 约束，含 JSON Schema 实验，代码 github.com/eth-sri/constrained-diffusion
  - LAVE（arXiv 2602.00612）：CFG，代码 github.com/zhangyitonggg/CD4dLLM
  - EPIC（arXiv 2606.00722）：CFG + 并行提交的兼容子集选择，代码 github.com/hyundong98/EPIC-Decoding
- **FactorDLM**（Su & Zhang, arXiv 2609.32900，2026-09-26 ✅）：用因子图表达**跨字段关系约束**（如 JSON 中的交叉引用）并精确解码。**这对我们的论点有影响**：如果事先知道“各调用的实体必须互不相同”，理论上可以写成关系约束来消除重复。我们的回应是：(1) 哪个实体对应哪个调用、调用数是多少，正是任务本身要解决的，无法事先写成约束；(2) all-different 这类启发式约束在合法的重复调用场景下会出错。可以把“加 all-different 约束”作为一个对照实验，展示它只能修掉重复，修不掉错绑和嵌合。

### 2.4 结论：空白在哪里

原来的说法“没人研究语法正确之后还剩哪些错误”**不能这样写**，会被 A1 和 Chavan 直接反驳。改为：

> 已有工作要么在没有语法保证的条件下评测 dLLM 工具调用，导致错误被格式问题主导（A2）；要么在保证语法后只定性列举残余错误，并在总准确率层面报告对步数的鲁棒性（A1）。**没有工作定量刻画语法保证下 dLLM 并行工具调用的残余错误构成，也没有工作检验这些错误是否由并行去掩码造成。**

真正开放的问题：(1) 按 BFCL 类别拆分的定量分类；(2) 并行调用特有的错误（重复、遗漏、跨调用错绑、嵌合、共享参数不一致）；(3) 固定约束、只改并行度的因果检验，看**错误类型分布**如何变化；(4) 覆盖原生支持工具调用的新 dLLM（LLaDA2.x）。

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

---

## 5. 错误分类法（语法已保证之后）

评估采用**集合级匹配**：把预测调用集合与标准答案调用集合做最大二分匹配（匈牙利算法，代价 = 函数名 + 参数的 AST 不一致程度）。这样“调用顺序不同”本身不算错。这一点必须在论文中明确，因为并行调用本来就是无序集合。

与官方 BFCL AST checker 的关系（论文里要写清楚）：官方 checker 先检查调用数是否等于标准答案（否则记 `wrong_count`，重复和遗漏都落在这里），然后做**贪心**的无序匹配（不是最优匹配），任何一个标准调用找不到匹配就记 `cannot_find_match`，逐样本全对或全错。它的错误字符串（`wrong_func_name`、`missing_required`、`unexpected_param`、`type_error:*`、`value_error:*`）只到单调用粒度，**无法区分重复、遗漏、错绑、嵌合**。所以：(1) 总准确率一律报告官方 checker 的结果，保证与已有工作可比；(2) 错误分类用我们的匈牙利匹配 + 细分规则，这是分类法的增量。

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
- 与已有分类法对齐（相关工作里放一张映射表）：ToolScan（arXiv 2411.13547）的 *Repeated API Calls* / *Insufficient API Calls*、HiTEC（arXiv 2506.00042）的 *Wrong Number of Tools*、BFCL 的 `wrong_count` 分别对应我们的重复、遗漏和调用数错误。**错绑、嵌合、共享参数不一致在已有分类法中没有对应项**，这是本文的新增类别。注意：重复调用在 AR 模型里也有记录（例如 OpenAI 文档提到 gpt-4.1-nano 会重复调用同一工具），所以我们的论点是 dLLM 的**比例更高且可归因于同时提交**，而不是“只有 dLLM 会重复”。
- 关键指标：**跨调用错误率（CCER）**、**单调用错误率（SCER）**，以及各细类的占比。

---

## 6. 实验设计

### 6.1 模型

原则：**每个 dLLM 尽量配一个“同源” AR 基线**，以排除数据和规模差异。

| dLLM | 匹配的 AR 基线 | 说明 |
|---|---|---|
| Dream-v0-Instruct-7B | Qwen2.5-7B-Instruct | Dream 由 Qwen2.5-7B 初始化，是最干净的对照 |
| **LLaDA2.0-mini**（inclusionAI/LLaDA2.0-mini，16B 总参/1.4B 激活，MoE，原生支持工具调用，BFCL 70.90，默认块长 32） | **Ling-mini-2.0**（inclusionAI/Ling-mini-2.0，16.3B，MIT；模型卡说明 LLaDA2.0 在 Ling2.0 系列上继续训练，并直接与其对比）；辅以 Qwen3-8B | **强烈建议纳入**：回应“你们的 dLLM 根本没学过工具调用”的质疑。有同源 AR 模型，是继 Dream/Qwen2.5 之后的第二个干净对照。块扩散结构使“块长度”成为天然干预旋钮。SGLang 原生支持（`--dllm-algorithm LowConfidence / JointThreshold`） |

**已定：minimal 方案**，只用上表中两组同源对照：Dream-v0-Instruct-7B ↔ Qwen2.5-7B-Instruct，LLaDA2.0-mini ↔ Ling-mini-2.0。LLaDA-8B（没有同源 AR 模型）、其他块扩散模型和 Llama-3.1 对照都不做。

Dream-7B 在无约束时 BFCL 很弱（A2 报告总分十几分），但在 A1 的约束解码下 BFCL-Live 可达约 71。因此约束（本文用 Oracle 骨架）是让分析有意义的前提。

**是否微调**：主实验不做训练，直接用 instruct 模型加统一的系统提示。如果某 dLLM 在骨架模式下准确率过低（< 40%）导致分析没有意义，再用同一份工具调用数据（如 xLAM/APIGen 子集）对 dLLM 和 AR **做同样的 LoRA SFT**，保持对照公平。这一步列为“风险预案”，不作为必做项。

### 6.2 约束解码

- 输出格式统一为 JSON 数组：`[{"name": ..., "arguments": {...}}, ...]`。
- **主设置（已实现）：Oracle 骨架**。调用数、函数名、参数键、括号和引号预先写在画布上，只对取值槽加掩码；每个槽按 JSON 类型限制 token（字符串 / 整数 / 浮点 / 布尔 / 其他）。语法由构造保证，剩下的错误**全部是取值层面的语义错误**，跨调用的错绑、重复、嵌合能被最干净地测量。它是 RQ1–RQ3 的主战场，dLLM 和 AR 用同一套骨架。
- **对照：自由生成（无约束）**。只报告语法错误率，并在语法合法的子集上看错误构成（注明选择偏差）。
- **暂不做：全文法约束**（例如接入 eth-sri/constrained-diffusion 的可补全性检查）。在代码里留了接口；如果审稿人要求“非 Oracle”的语法保证设置，可以作为 rebuttal 或终稿阶段的补充实验。Dang & Ermon（A1）的代码目前没有找到公开链接。

### 6.3 数据

1. **BFCL（v4，Apache-2.0）**：minimal 方案只用 `parallel`、`parallel_multiple`（共 400 条）。`live_parallel` / `live_parallel_multiple`（共 40 条）和单调用的 `simple` / `multiple` 不跑；单调用错误率在并行题内部就能测到。评估方式见第 5 节（官方 checker + 集合级匹配）。检索代理直接统计了数据文件（引用前请自己复核一遍）：

| 类别 | 条数 | 每条调用数 | 平均 | 不同函数数 | ≥2 个调用共享某个相同参数值 |
|---|---|---|---|---|---|
| parallel | 200 | 2:109, 3:52, 4:36, 6:1, 8:2 | 2.70 | 恒为 1 | 139/200 |
| parallel_multiple | 200 | 2–5 | 3.04 | 1:9, 2:105, 3:67, 4:19 | 74/200 |
| live_parallel | 16 | | 2.44 | | 13/16 |
| live_parallel_multiple | 24 | | 2.29 | | 10/24 |

   - `parallel` 类全是同一函数、不同参数，正是“对称槽”的天然实例，**是验证命题 1 的主场**。
   - “共享参数值”这一列可以直接用来判定“共享参数不一致”错误，不需要额外标注。
   - BFCL 并行题**没有调用之间的输入输出依赖**，所以我们研究的是“同一轮内的协调”，不是嵌套调用；论文里要明确这个范围。
   - live 的并行类样本太少（共 40 条），只作补充，不单独下结论。
   - **BFCL 的 `main` 分支没有 dLLM 的 model handler**，需要自己写一个 handler，或者把 dLLM 包成 OpenAI 兼容接口，用 prompt 模式评测。
2. **ParaProbe（自建合成可控探针，本文贡献之一）**：程序化生成，168 个因子组合 × 每格 5 条 = 840 条（minimal 方案），覆盖以下可控因子：
   - 调用数 n ∈ {2, 3, 4, 6}
   - **槽分配歧义度**：显式列举（“依次查询 A、B、C”）/ 无序提及 / 隐式推导（“北欧五国首都”）
   - **实体 token 结构**：单 token 实体 / 共享前缀的多 token 实体（San Francisco vs San Diego）/ 无共享前缀
   - **跨调用共享参数**：有（同一 date/unit）/ 无
   - **调用间依赖**：独立 / 同一函数不同参数 / 不同函数（parallel_multiple 风格）
   - 标准答案由生成程序给出，可做精确的错误类型判定。
3. （可选）NESTFUL 或类似的“有依赖的调用序列”，作为外部效度检验。

### 6.4 解码配置矩阵（核心干预，minimal 方案）

所有 dLLM 实验都在 Oracle 骨架约束下进行（语法由构造保证），贪心解码。

| 数据 | 配置 | 用途 |
|---|---|---|
| BFCL 并行两类 | confidence 顺序，k ∈ {1, 2, 4, 8, 16} | 剂量–响应（图 3） |
| BFCL 并行两类 | left_to_right，k = 1 | AR 顺序对照：和 confidence k=1 一起，把“任意顺序”与“同时性”分开 |
| BFCL 并行两类 | k = 1 + 阈值 τ = 0.9（Fast-dLLM 式） | 阈值调度的一个工作点（检验 3.3 的“阈值保护”推论） |
| ParaProbe | confidence 顺序，k ∈ {1, 2, 4, 8} | 调节因子（图 4） |
| BFCL 并行两类 | 自由生成（无约束），各模型默认式解码 | 对照：语法错误率和自然生成下的错误构成 |

- Dream 不分块；LLaDA2.0 固定块长 32（其训练设置，按绝对位置对齐）。块长度不作为干预变量。
- 反事实重放和 DVS 在 k = 4 的 BFCL 结果上做。
- 不做：温度采样多种子（命题 1 的采样版在玩具模型上已验证，见 `scripts/sim_prop1.py`）、更多 τ、块长度扫描。

### 6.5 因果归因三件套

**(A) 剂量–响应干预**：固定模型、提示、种子和约束，只改 k（或 τ、块长度）。画 CCER 和 SCER 随 k 的曲线。H2 预测 CCER 单调上升、SCER 基本平坦。

**(B) 逐实例反事实重放（Counterfactual Sequentialization）**：
1. 贪心解码，记录完整轨迹：每个位置被提交的步号 tᵢ、每步的边缘分布。
2. 对每个出错实例，定位“出错 token”（如重复调用中第二个调用的实体 token）被提交的那一步 t*。
3. 从 t* 之前的状态重新开始，只把 t* 这一步改成**按置信度逐个提交、每提交一个就重新前向一次**，之后仍用原调度继续。
4. 若错误消失 → 归因为“同时性”；若不消失 → 归因为模型/顺序。
5. **安慰剂对照**：在正确实例上随机挑一步做同样处理，统计被“改坏”的比例，作为噪声基线。
6. 报告：可归因比例 = P(修复 | 出错) − P(改坏 | 正确)，按错误类型分解。

**(C) 依赖违背分数 DVS 与共提交分析**（度量本身沿用 DEMASK 的 D_ij 和 Wen et al. 的 pseudo-cost，见 2.2；我们的贡献是把它和具体错误挂钩）：
- 对第 t 步同时提交的 token 对 (i, j)，额外做一次前向：把 xᵢ 填为已提交的值，得到 p(xⱼ | xᵢ, ·)，计算
  DVSₜ(i, j) = KL( p(xⱼ | xᵢ, ·) ‖ p(xⱼ | ·) )，步级 DVSₜ = 对该步所有 token 对求和（或取最大值）。
- 检验：出错步的 DVS 分布显著高于正确步（AUROC）。
- **混杂提醒**（论文里要主动写）：在阈值调度下，“两个槽同时被提交”本身说明它们都是高置信，这是简单样本的标志。所以观察性的“共提交 vs 出错”相关会被样本难度混杂，只能作为辅助证据。**因果结论以 (A)(B) 的干预结果为准**。

### 6.6 统计

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

---

## 8. 论文结构（8 页）

1. Introduction（1 页）：问题、并行工具调用对智能体的重要性、“语法正确 ≠ 调用正确”、贡献列表
2. Background & Related Work（0.75 页）
3. Parallel Unmasking as a Simultaneous-Move Decision（0.75 页）：形式化、命题 1、调度器依赖推论
4. Taxonomy & Evaluation Protocol（0.75 页）：集合级匹配、分类法、ParaProbe
5. Experimental Setup（0.5 页）
6. Results（3 页）：RQ1–RQ3
7. Discussion, Limitations, Conclusion（0.75 页）：含对解码器设计的启示（例如耦合位置不应同步提交），但不作为贡献

**AAMAS 定位**：主领域选 GAAI（涵盖 orchestration and workflows、agentic benchmarks）。叙事上强调：**工具调用是智能体的动作，并行工具调用是联合动作，并行去掩码是“无通信的同时行动”**。关键词：agentic AI, tool use, diffusion language models, parallel action selection, coordination failure。

---

## 9. 8 天执行计划（北京时间）

| 日期 | 任务 | 产出/里程碑 |
|---|---|---|
| **9/30（今天）** | 核对所有作者 OpenReview 账号；确定标题；搭环境，下载模型；跑通 Dream / LLaDA 的 BFCL parallel 推理 | 标题定稿；每个模型 10 条样例能跑出结果 |
| **10/1** | 接入约束解码（或骨架填空）；实现集合级匹配 + 错误分类器；**提交摘要** | 摘要最迟 10/2 20:00 提交；pipeline 端到端可用 |
| **10/2** | ParaProbe 生成器；启动主实验矩阵（BFCL × 模型 × k × 调度器）；AR 基线 | 图 2、图 3 的原始数据开始产出 |
| **10/3** | ParaProbe 实验（n、歧义度、token 结构）；人工标注 200 条 | 图 4；κ 值 |
| **10/4** | 反事实重放 + 安慰剂；DVS 计算（在子集上） | 表 1、图 5 |
| **10/5** | 补跑缺失的格子；自由生成对照 | **实验冻结** |
| **10/6** | 写作：方法、实验、结果（边写边补图） | 完整初稿 |
| **10/7** | 写作：引言、相关工作、理论节；内部审阅 | 第二稿 |
| **10/8** | 精修、压页、补充材料（代码 + ParaProbe + 完整结果表）、AI 使用声明 | 定稿 |
| **10/9 白天** | 缓冲；**20:00 前提交** | 提交 |

**算力估计（minimal 方案）**：约 **31 A100·小时**（含 30% 重跑余量），2 张卡并行约 15 小时墙钟时间。明细和估算方法见 README 与 `scripts/estimate_cost.py`；冒烟测试实测速度后重算。LLaDA2.0-mini 需要 80GB 卡。

**分工建议（若有 3 人）**：A 负责解码器（约束 + 调度 + 轨迹记录 + 重放）；B 负责数据与评估（BFCL 适配、ParaProbe、分类器、标注）；C 负责理论节、AR 基线、写作主线。

---

## 10. 风险与预案

| 风险 | 预案 |
|---|---|
| dLLM 基础工具调用能力太弱，错误被“啥都不会”淹没 | 以 LLaDA2.0-mini（原生工具调用）和 Oracle 骨架设置为主战场；必要时用同一份数据对 dLLM 与 AR 做 LoRA SFT |
| 结果对提示格式和解码设置极度敏感：Dream-7B 的 BFCL 分数在 A1（BFCL-Live 约 64–72）和 A2（整体十几分）之间差异巨大 | 第一天就固定提示模板（JSON 格式、系统提示写法）、gen_length、步数和块长度，在论文里完整报告；主结论只依赖**同一配置下只改并行度**的配对比较，不依赖绝对分数 |
| 约束解码实现来不及 | 骨架填空（6.2 第 2 种）；或无约束 + 条件于语法合法（稳健性检验） |
| 阈值调度下几乎看不到并行错误 | 这本身是一个发现（阈值保护对称槽），与命题 1 的推论一致。把重点放在固定 k 调度和“部分对称/共享前缀”的嵌合错误上，并报告速度–错误的权衡 |
| 审稿人质疑“只是 NLP 分析，不是 AAMAS” | 第 3 节的同时行动博弈框架 + 智能体联合动作叙事；GAAI 领域明确欢迎 agentic benchmarks 和 orchestration |
| A1 的“约束解码对步数鲁棒”在我们的设置中复现，即总准确率随并行度基本不变 | 这不否定我们的问题，反而是论文的看点：检查**错误构成**是否变化、parallel 类是否与 simple 类表现不同。若错误构成也不变，那就是一个干净的**负面结果**：“语法保证后，并行去掩码不是主要错误来源”，同样可发表，但要在 10/5 前决定叙事 |
| 被审稿人认为与 A1 / A2 / ParallelBench 重叠 | 引言第二段就明确三者的差异（见 2.4），并在相关工作中逐条对比 |
| 8 天写不完 | 优先保证 RQ1 + RQ2（分类法 + 剂量–响应 + 反事实重放）；DVS 和 ParaProbe 的部分因子可移入补充材料；或按第 11 节换投 |

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
- [ ] 下载 Dream-v0-Instruct-7B、LLaDA2.0-mini、Qwen2.5-7B-Instruct、Ling-mini-2.0；拉取 BFCL 数据（本地运行见 `LOCAL_CLAUDE_PROMPT.md`）
- [ ] 查 Dang & Ermon（arXiv 2607.07026）是否开源代码（没有就发邮件）；通读其附录 D 和图 4
- [ ] 克隆 eth-sri/constrained-diffusion，在 Dream-7B 上跑通一个 BFCL parallel 样例
- [ ] 读 FactorDLM（arXiv 2609.32900）：确认它能否表达跨调用约束，决定是否把 all-different 约束作为对照
- [ ] 通读 Lu et al.（ACL 2026）的 BFCL 错误分析部分，以及 ParallelBench
- [ ] 分工

---

## 13. 摘要（英文，296 词；v0.5 已按 minimal 方案修改）

**标题**：*Syntactically Valid, Semantically Uncoordinated: Diagnosing Parallel Tool-Call Errors in Diffusion Language Model Agents*

**Abstract**（纯文本，直接复制到 OpenReview 替换旧版）：

```text
Diffusion language models (dLLMs) commit several tokens per denoising step, which makes them look well suited to agents that issue multiple tool calls in one turn. Automaton- and grammar-constrained decoding now guarantees that dLLM tool calls are syntactically valid. Yet prior evaluations either report errors dominated by malformed output or describe the remaining semantic errors only qualitatively, and none tests whether parallel unmasking causes them. We ask which errors remain once syntax is guaranteed, and whether parallel unmasking changes which errors occur, not only how many. We model an unmasking step as a simultaneous-move decision: each masked position chooses a token without observing what its peers commit in the same step, so co-committed tokens follow a product of marginals, and schema constraints do not enforce agreement across calls. For n interchangeable calls whose argument slots are committed together under a permutation-symmetric joint distribution, this predicts a valid assignment (no duplicated or omitted call) with probability only n!/n^n, e.g. 0.22 for three calls. We study Dream-v0-Instruct-7B and the tool-use-trained LLaDA2.0-mini, each paired with an autoregressive model from the same base family (Qwen2.5-7B-Instruct and Ling-mini-2.0). We use the parallel and parallel-multiple categories of the Berkeley Function Calling Leaderboard and a controllable probe suite that varies the number of calls, slot-assignment ambiguity, and shared-prefix entity names. We propose an order-invariant, set-level evaluation and a taxonomy separating single-call errors from cross-call coordination errors: duplicated, omitted, and cross-bound calls, inconsistent shared arguments, and chimera values spliced from two valid values. To attribute errors to parallelism rather than to model capability or generation order, we hold the constraint fixed and combine dose-response interventions on the number of tokens committed per step, with a left-to-right control; counterfactual re-decoding of single steps one token at a time; and a dependency-violation score over co-committed tokens.
```

**相对上一版的三处修改**（都是为了和 minimal 方案一致）：
1. 模型：删去 LLaDA-8B-Instruct，改为“Dream 和 LLaDA2.0-mini 各配一个同源 AR 模型”。
2. 数据：BFCL 只写 parallel 和 parallel-multiple 两类（simple / multiple 不跑）。
3. 干预：删去“置信度阈值、块长度”作为剂量–响应变量，改为“每步提交 token 数 + 从左到右对照”（阈值只有一个工作点，块长度不扫）。

**说明**
- 摘要只陈述问题、定位、理论预测和实验设计，**不包含尚未得到的实验结论**。10/8 提交全文前，把最后一句替换成实际发现（例如跨调用错误占比、错误构成随 k 的变化、反事实重放的修复比例）。
- 第二、三句是对 A1（Dang & Ermon）和 A2（Lu et al.）的定位，没有点名，也没有说“从来没人研究过”。

## 附：参考链接

- AAMAS 2027 主轨道征稿：https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/call-for-main-track/
- 投稿说明（8 页 + 参考文献、LaTeX、双盲）：https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/instructions/
- Q&A（附录算入 8 页、允许 arXiv、Findings）：https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/qa/
- OpenReview：https://openreview.net/group?id=ifaamas.org/AAMAS/2027/Conference
- A1 Dang & Ermon 2026：https://arxiv.org/abs/2607.07026
- A2 Lu et al., ACL 2026：https://arxiv.org/abs/2601.12979
- ParallelBench：https://arxiv.org/abs/2510.04767
- LLaDA2.0：https://arxiv.org/abs/2512.15745 ，模型卡 https://huggingface.co/inclusionAI/LLaDA2.0-mini
- Ling-mini-2.0：https://huggingface.co/inclusionAI/Ling-mini-2.0
- BFCL：https://gorilla.cs.berkeley.edu/leaderboard.html
- ToolScan：https://arxiv.org/abs/2411.13547 ；HiTEC：https://arxiv.org/abs/2506.00042
