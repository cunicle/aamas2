# ptcdiag：dLLM 并行工具调用的残余错误诊断

研究方案见 [`proposal.md`](proposal.md)。本仓库是配套的实验代码（初版）。

## 当前状态

| 部分 | 状态 |
|---|---|
| BFCL 评测（Python 部分移植） | ✅ 已测：约 3,000 条 BFCL 样例上，集合级判定与官方判定逐条一致 |
| 错误分类法（§5 的 11 类） | ✅ 已测：手工用例 + BFCL 扰动 |
| ParaProbe 生成器（§6.3） | ✅ 已测：标准答案全部通过，“derived”设置确实不泄露城市名 |
| 统一采样器（k / τ / 块长度 / 顺序 / 轨迹记录 / 串行化） | ✅ 已测：玩具模型上与命题 1 的闭式解一致（误差 < 0.03） |
| 反事实重放、安慰剂、DVS | ✅ 已测：真实分词器 + 玩具 logits 的端到端用例 |
| Oracle 骨架约束（§6.2 第 2 种） | ✅ 已测（同上） |
| Dream / LLaDA2.0 适配器（以及保留的 LLaDA-8B 适配器） | ⚠️ **按官方参考代码编写，但还没在真实权重上跑过**（这台机器没有 GPU）。上 GPU 后先跑冒烟测试 |
| AR 基线（free / skeleton） | ⚠️ 同上，未在真实模型上跑过 |
| 全文法约束（free 模式下保证语法） | ❌ 未实现，只留了接口（见“已知限制”） |

测试（CPU 即可，约 10 秒）：

```bash
pip install -r requirements.txt
python scripts/prepare_data.py          # 下载 BFCL v4 + 生成 ParaProbe
python -m pytest tests -q               # 38 个测试；test_pipeline 需要能下载 Dream 分词器
python scripts/sim_prop1.py             # 命题 1：闭式解 vs 真实采样器
python scripts/estimate_cost.py          # A100 小时估计（见下）
```

## 目录

```
ptcdiag/
  types.py                 Example（BFCL 格式）
  prompting.py             统一的系统提示；带字符区间的 JSON 解析（把错误定位回 token）
  data/bfcl.py             BFCL v4 下载与加载
  data/paraprobe.py        ParaProbe：n / 实体类型 / 歧义度 / 共享参数 / 函数混合 五个因子
  eval/bfcl_checker.py     BFCL AST checker 的 Python 部分（Apache-2.0，保留出处）
  eval/taxonomy.py         匈牙利匹配 + 残余错误分类
  decoding/sampler.py      统一的带轨迹采样器（所有干预旋钮都在这里）
  decoding/adapters.py     LLaDA / Dream / LLaDA2.0 / 玩具模型
  decoding/constraints.py  Oracle 骨架约束，以及约束接口
  decoding/ar.py           AR 基线（同一提示、同两种模式）
  analysis/counterfactual.py  逐实例反事实串行化 + 安慰剂
  analysis/dependency.py      DVS（pseudo-cost）与两两 TV
  theory.py                命题 1 及其按调度推广的闭式解
  pipeline.py              单条样例：提示 → 解码 → 解析 → 诊断 → 记录
scripts/                   见下
tests/
```

## 在 GPU 上运行（minimal 方案）

实验方案已定为 minimal：dLLM 用 Dream-v0-Instruct-7B 和 LLaDA2.0-mini，对应的 AR 对照是 Qwen2.5-7B-Instruct 和 Ling-mini-2.0；数据是 BFCL parallel / parallel_multiple 和 ParaProbe（840 条）。全部步骤都封装在 `scripts/run_minimal.sh`，交给本地 Claude 执行的完整说明见 [`LOCAL_CLAUDE_PROMPT.md`](LOCAL_CLAUDE_PROMPT.md)。

**环境**：各模型 `config.json` 里记录的 transformers 版本不同，建议建两个环境。

| 环境 | 模型 | transformers |
|---|---|---|
| `PY_DREAM` | Dream-7B、Qwen2.5-7B | 4.46.2 |
| `PY_LLADA2` | LLaDA2.0-mini、Ling-mini-2.0 | 4.57.1 |

**按阶段执行**（每个阶段都能断点续跑）：

```bash
export PY_DREAM=~/envs/dream/bin/python PY_LLADA2=~/envs/llada2/bin/python
bash scripts/run_minimal.sh setup     # 数据 + CPU 测试
bash scripts/run_minimal.sh smoke     # 冒烟测试，必须通过才能继续
bash scripts/run_minimal.sh pilot     # 试跑 80 条 × k∈{1,4,16}：准确率、跨调用错误、测速；看完再决定开跑
CUDA_VISIBLE_DEVICES=0 bash scripts/run_minimal.sh dream &
CUDA_VISIBLE_DEVICES=1 bash scripts/run_minimal.sh llada2 &   # 需要 80GB 卡
wait
bash scripts/run_minimal.sh ar
bash scripts/run_minimal.sh attr
bash scripts/run_minimal.sh summary   # 表格写入 results/summary/
```

冒烟测试要看到：
- LLaDA2.0：`LLaDA2.0 reference match (up to EOS): True`。官方采样器在 temperature=0 时仍然用 `torch.multinomial` 采样，所以冒烟测试会先把它换成贪心再比较；我们的主实验一律用真正的贪心。
- 所有模型：`determinism (resume at step 3): True`。这是反事实重放的前提；如果是 False，先试 `torch.use_deterministic_algorithms(True)`。LLaDA2.0 还会多跑一次整段一块的骨架模式（`[skeleton block=full]`），同样要求语法正确、确定性为 True。
- Dream：官方调度不是固定 k，没有可逐 token 对齐的参考，只看端到端输出是否像样（JSON 结构、调用数合理）。

单独运行某一项时，各脚本的用法写在文件开头的 docstring 里（`run_dllm.py`、`run_ar.py`、`attribute.py`、`dvs.py`、`summarize.py`）。LLaDA-8B 的适配器还保留在代码里，但不在 minimal 方案中。

## 算力估计（A100 小时）

`scripts/estimate_cost.py` 用真实提示和骨架算出每个配置的前向次数；固定 k 的配置是精确值，块按模型各自的规则切分。再乘以每次前向的耗时得到小时数。耗时默认按 FLOPs 在约 40% MFU 下估算（8B 稠密模型每 1k token 约 125 ms），**不是实测值**。冒烟测试跑完后，把实测值代进去重算：

```bash
python scripts/estimate_cost.py --plan minimal
python scripts/estimate_cost.py --plan full --timing dream=110:20 --timing llada2=45:30   # 实测后
```

| 方案 | 内容 | 估计 |
|---|---|---|
| minimal | Dream + LLaDA2.0-mini；BFCL parallel / parallel_multiple；ParaProbe 840 条；k∈{1,2,4,8,16}、左到右 k=1、τ=0.9；free 对照 1 个配置；k=4 下做归因和 DVS；LLaDA2.0 整段一块的块长度对照（BFCL k∈{1,4,16}、ParaProbe k∈{1,4}、k=4 归因）；AR 对照为 Qwen2.5-7B 和 Ling-mini-2.0 | 约 27 h，加 30% 重跑余量后约 **35 A100·h**；2 卡并行约 18 h |
| full | 在 minimal 基础上增加 LLaDA-8B（轻量版）、live 并行题、ParaProbe 1,680 条、更多 τ 和块长度、左到右 k=4、命题 1 的采样验证、k=8 下的归因、Llama-3.1-8B | 约 54 h，加余量后约 **70 A100·h** |

- 大头是 k=1 的对照（每条约 80–100 次前向）和 ParaProbe；按因子砍 ParaProbe 是最有效的省钱方式。
- 为什么 LLaDA2.0 要加整段一块的对照：按原生的 32 token 分块，BFCL 并行题里不同调用的同名参数槽只有 0.8% 落在同一块，ParaProbe 为 0%，所以原生设置下它们几乎不可能被同时提交，理论预测跨调用错误对 k 不敏感（proposal H2b）。整段一块时这个比例是 100%，同一权重上形成块长度 × k 的对照。
- 显存：Dream / LLaDA-8B 用 40GB 卡即可；LLaDA2.0-mini 的 bf16 权重约 32GB，需要 80GB 卡。
- 前向耗时的不确定性约 ±50%（取决于 attention 实现和 MFU），所以冒烟测试实测速度后一定要重算。

## 已知限制 / 下一步

1. **全文法约束没做**。free 模式目前不保证语法。计划是把 eth-sri/constrained-diffusion 的“可补全性检查”包装成 `Constraint` 接口：`filter` 屏蔽会导致不可补全的 token；或者在提交后检查、不合法就重采。它自带的生成循环不能直接用，因为我们需要自己控制调度和轨迹。
2. **Dream 没有能逐 token 对齐的官方参考**：它的官方调度按时间步分配提交数，与固定 k 不同。Dream 的冒烟测试只做端到端检查。
3. **骨架的分词边界**：骨架文本和取值是分开分词的，与模型自然生成时的分词可能不同（例如 `"Paris` 被拆成 `"` + `Paris`）。dLLM 和 AR 用的是同一套骨架，比较仍然公平，但论文里要说明。
4. **槽长度固定**（字符串 14、整数 10、浮点数 12、数组/字典 48 个 token），多余位置用 EOS 填充，解码时丢弃。长度按 BFCL 并行题标准答案取值的分布选定：覆盖字符串的 p99，以及整数、浮点数的最大值。约 1% 的超长字符串、数组仍会被截断；需要时可通过 `slot_lengths` 调整。
5. **效率**：batch=1、没有 KV cache。如果时间不够，可以按提示长度分桶做 batch（LLaDA / Dream 需要 attention mask）。
6. BFCL 里有少数样例本身无解（标准答案用了 schema 里不存在的参数，或必填参数没有可接受值），`tests/test_taxonomy.py` 里有说明。所有模型同样受影响，不影响相对比较。
