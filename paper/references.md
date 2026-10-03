# 引用清单（2026-10-03 定稿）

`references.bib` 共 37 条，每一条都对照了出版方或 arXiv 的记录（标题、全部作者、年份、出处）。BibTeX 只输出正文中实际 `\cite` 的条目。

**核对方法**：
- arXiv 条目：用 arXiv API 逐条比对。
- 正式发表的条目：PMLR、ACL Anthology、AAMAS 2026 目录、Crossref（DOI）、iclr.cc 和 neurips.cc 的录用数据、proceedings.neurips.cc、ML Anthology、RePEc、HathiTrust。
- 正文里对文献的说法：子 agent 读原文核对，结论记在 `structure.md` 第 6.4 节。

**新增条目的规则**：只能加同样核对过的文献；描述一篇文献之前，先读它的原文。

**状态**：已引 = 正文已 `\cite`（10-03 §1、§2 写完后全部引用）。

## 1. 模型、数据、评测

| 键 | 文献 | 出处（已核对） | 用途 | 状态 |
|---|---|---|---|---|
| patil2025bfcl | BFCL | ICML 2025，PMLR 267:48371–48392 | 数据；按集合评分（App. H：“any predicted call ai may match any ground-truth call bj”） | 已引 |
| ye2025dream | Dream 7B | arXiv 2508.15487 | 模型；从 Qwen2.5-7B **base** 初始化，全注意力 | 已引 |
| bie2025llada2 | LLaDA2.0 | arXiv 2512.15745 | 模型；块扩散；论文报告 BFCL v3 70.90；“1.4B active”出自模型卡 | 已引 |
| nie2025llada | LLaDA | NeurIPS 2025，DOI 10.52202/085713-1689 | dLLM 背景 | 已引 |
| qwen2024qwen25 | Qwen2.5 技术报告 | arXiv 2412.15115（2024-12） | AR 参照 | 已引 |
| sahoo2024mdlm | MDLM | NeurIPS 2024，DOI 10.52202/079017-4135 | 掩码扩散背景 | 已引 |

## 2. LLM agent 与工具调用

| 键 | 文献 | 出处 | 用途 | 状态 |
|---|---|---|---|---|
| kim2024llmcompiler | LLMCompiler | ICML 2024，PMLR 235:24370–24391 | 并行函数调用（planner + task fetching + executor） | 已引 |
| yao2023react | ReAct | ICLR 2023 | agent 背景 | 已引 |
| schick2023toolformer | Toolformer | NeurIPS 2023，DOI 10.52202/075280-2997 | 工具使用背景 | 已引 |

## 3. dLLM agent

| 键 | 文献 | 出处 | 用途（按原文） | 状态 |
|---|---|---|---|---|
| lu2026bitter | Lu et al., Bitter Lesson | ACL 2026 Long，pp. 43997–44020 | dLLM agent 调用工具失败（JSON 格式错误、参数幻觉），归因于扩散噪声和并行解码削弱因果依赖；附录有受控研究，但没改变槽长或每步提交的 token 数 | 已引 |
| zhen2026dllmagent | DLLM Agent | arXiv 2602.07451 | 同一工作流、匹配微调下比较 dLLM 和 AR agent：端到端快 30% 以上；工具调用更容易出结构性错误 | 已引 |
| zhao2026dllmsearcher | DLLM-Searcher | arXiv 2602.07035 | dLLM 搜索 agent；P-ReAct 先解码 tool_call | 已引 |
| qiu2026reverse | Qiu et al., Reverse Scoring | arXiv 2609.38536 | dLLM agent 的重试循环来自“先提交最有把握的位置”，上下文中显眼的动作被重新提交 | 已引 |
| khanal2026agents | Agents of Diffusion | AAMAS 2026 Research Track，pp. 449–458，DOI 10.65109/GGJL7344 | **dLLM 本身不是 agent**：两个 AR LLM agent 用多智能体 RL 引导冻结的 LLaDA-8B 生成 JSON 数据 | 已引 |

## 4. 并行解码与因子化误差

| 键 | 文献 | 出处 | 用途 | 状态 |
|---|---|---|---|---|
| wu2026fastdllm | Fast-dLLM | ICLR 2026 | 置信阈值并行解码，默认阈值 0.9 | 已引 |
| kang2026parallelbench | ParallelBench | ICLR 2026 | “New City”例子（来自 New York / Mexico City），归因于条件独立；“chimera”是我们的术语 | 已引 |
| benhamu2025ebsampler | EB-Sampler | NeurIPS 2025，DOI 10.52202/085713-1874 | 按熵上界一次解掩多个 token | 已引 |
| liu2025copula | Discrete Copula Diffusion | ICLR 2025，pp. 88953–88979 | 同一步解码的变量之间的依赖缺失 | 已引 |

## 5. dLLM 的长度问题（与我们的发现最近）

| 键 | 文献 | 出处 | 用途（按原文） | 状态 |
|---|---|---|---|---|
| han2026dia | Dynamic Infilling Anchors (DIA) | ACL 2026 Long，pp. 26213–26227 | **最接近的已有工作**：固定锚点“impose rigid spans, leading to truncated reasoning or redundant content”；在填空前估计结束锚点的位置。只处理单个回复的格式模板 | 已引 |
| wu2026dreamon | DreamOn | ICLR 2026 | 代码填空：mask 长度与理想长度不符时性能严重下降；加入两个长度控制状态，让模型自己伸缩 | 已引 |
| liu2026cal | CAL | arXiv 2602.00476 | 单段填空；校准后的首步置信度 + 双向爬山；代码填空平均多 11–18 次前向；多段列为未来工作 | 已引 |
| yang2026rhoeos | ρ-EOS | arXiv 2601.22527 | 伸缩整段的尾部 | 已引 |
| li2026daedal | DAEDAL | ICLR 2026 | 扩展尾部，并在低置信位置插入 mask；只处理单个回复 | 已引 |
| xiong2026s3 | S³ | ICLR 2026 | mask 给多时过度生成；用 `null` 占位；LLaDA-1.5，WikiBio | 已引 |

**定位**：DIA 和 DreamOn 已经报告过“固定长度导致截断或冗余”。我们不声称发现这一点，新意在两处：
1. 在多槽的 agent 动作中，错误的长度会让值绑到兄弟调用（swap 下 79.4%），而且只差一点最危险。
2. 这些错误看起来像并行协调失败，即使每步只提交一个 token 也会出现。

## 6. 约束解码

| 键 | 文献 | 出处 | 用途 | 状态 |
|---|---|---|---|---|
| dang2026constrained | Dang & Ermon | NeurIPS 2026 录用（论文集未出，标 To appear） | dLLM 的有限自动机约束解码；在 BFCL-Live 上把 Dream-7B 从 63.9% 提到 71.5% | 已引 |
| mundler2026cfg | Mündler et al. | ICLR 2026 | dLLM 的上下文无关文法约束解码，涵盖多区域填空 | 已引 |
| geng2023grammar | Geng et al. | EMNLP 2023，pp. 10932–10952 | AR 模型的文法约束解码 | 已引 |

## 7. 无通信协调

| 键 | 文献 | 出处 | 用途 | 状态 |
|---|---|---|---|---|
| schelling1960strategy | Schelling | Harvard University Press, 1960 | 焦点 | 已引 |
| lewis1969convention | Lewis | Harvard University Press, 1969 | 约定 | 已引 |
| mehta1994salience | Mehta, Starmer & Sugden | AER 84(3):658–673, 1994 | 纯协调博弈中显著性的实验（注意：不要和同一组作者 1994 年在 Theory and Decision 上的另一篇混淆） | 已引 |
| grenager2002dispersion | Dispersion games | AAAI 2002，pp. 398–403 | 对称选择下的分散与撞车 | 已引 |
| hu2020otherplay | Other-Play | ICML 2020，PMLR 119:4399–4410 | 利用对称性的零样本协调 | 已引 |

## 8. LLM 与协调、并行多 agent 系统

| 键 | 文献 | 出处 | 用途（按摘要） | 状态 |
|---|---|---|---|---|
| aharon2026tacit | Tacit Coordination of LLMs（Wooldridge、Kraus） | arXiv 2601.22184 | 20 多个 LLM 的焦点研究：常能不通信地协调，但需要数字常识或文化知识时失败 | 已引 |
| agashe2025llmcoordination | LLM-Coordination | Findings of NAACL 2025，pp. 8053–8072 | LLM 在纯协调博弈上的基准 | 已引 |
| ballestero2026monoculture | Strategic Algorithmic Monoculture | arXiv 2604.09502 | LLM 的选择高度相似；擅长选相同的动作，但需要彼此不同时不如人类 | 已引 |
| mao2026delm | DeLM | arXiv 2606.10662 | 并行 LLM agent 重复同伴的工作；用共享上下文和任务队列异步领取任务 | 已引 |
| rodionov2025hogwild | Hogwild! Inference | NeurIPS 2025，DOI 10.52202/085713-1551 | 同一 LLM 的多个实例共享 KV cache 并行生成，自己决定如何分工 | 已引 |

## 已删除或不引用

- **核对过但决定不引（用户 10-03 决定）**：D3PM（NeurIPS 2021）、Gorilla（NeurIPS 2024）、Breaking the Factorization Barrier（arXiv 2603.00045）、Outlines（arXiv 2307.09702）、Crawford & Haller（Econometrica 1990）、Akata et al.（Nature Human Behaviour 2025）。已从 bib 删除。

- **ParaProbe**：arXiv、Crossref 和网页搜索都找不到这样一篇论文（唯一同名的是原子探针层析的工具）。“实体绑错为零”这条说法没有来源，**不引用、不写**。
- **MetaGPT、AutoGen**：没有核对，也不切合 §9 那句话，不引用。
- **同一参数取最长值的 pilot（17/18）**：这是我们自己的数据，不是文献；要用的话，先在结果里复核。
