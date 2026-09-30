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
| LLaDA / Dream / LLaDA2.0 适配器 | ⚠️ **按官方参考代码编写，但还没在真实权重上跑过**（这台机器没有 GPU）。上 GPU 后先跑 `scripts/smoke_test.py` |
| AR 基线（free / skeleton） | ⚠️ 同上，未在真实模型上跑过 |
| 全文法约束（free 模式下保证语法） | ❌ 未实现，只留了接口（见“已知限制”） |

测试（CPU 即可，约 10 秒）：

```bash
pip install -r requirements.txt
python scripts/prepare_data.py          # 下载 BFCL v4 + 生成 ParaProbe
python -m pytest tests -q               # 36 个测试；test_pipeline 需要能下载 Dream 分词器
python scripts/sim_prop1.py             # 命题 1：闭式解 vs 真实采样器
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

## 在 GPU 上的执行顺序

### 0. 冒烟测试（每个模型各一次，第一步必须做）

```bash
python scripts/smoke_test.py --model GSAI-ML/LLaDA-8B-Instruct
python scripts/smoke_test.py --model Dream-org/Dream-v0-Instruct-7B
python scripts/smoke_test.py --model inclusionAI/LLaDA2.0-mini
```

需要看到：
- LLaDA：`LLaDA reference match: True`（我们的采样器与官方 `generate` 逐 token 一致）
- LLaDA2.0：`LLaDA2.0 reference match (up to EOS): True`
- 所有模型：`determinism (resume at step 3): True`（反事实重放的前提；bf16 下如果是 False，要打开 `torch.use_deterministic_algorithms` 或改用 fp32 做归因实验）

各模型的 remote code 可能依赖不同版本的 transformers（例如 Dream 发布时用的是 4.46.x）。如果一个版本跑不通全部模型，就为每个模型单独建一个环境。

### 1. RQ1：错误构成（图 2）

```bash
D=bfcl:parallel,parallel_multiple,live_parallel,live_parallel_multiple,simple_python,multiple
python scripts/run_dllm.py --model Dream-org/Dream-v0-Instruct-7B --data $D \
    --mode skeleton --k 1,4 --out results/dream_skel.jsonl
python scripts/run_ar.py --model Qwen/Qwen2.5-7B-Instruct --data $D \
    --mode skeleton --out results/qwen_skel.jsonl
python scripts/summarize.py results/dream_skel.jsonl results/qwen_skel.jsonl --by category --csv results/rq1.csv
```

`--mode free` 是不加约束的对照：语法错误会单独记为 `syntax_error`，报告时两种模式都给。

### 2. RQ2：剂量–响应（图 3），固定约束、只改并行度

```bash
M=Dream-org/Dream-v0-Instruct-7B; D=bfcl:parallel,parallel_multiple
# 每步提交 token 数；confidence = 任意顺序，left_to_right = AR 顺序（§6.4 的关键对照）
python scripts/run_dllm.py --model $M --data $D --mode skeleton \
    --k 1,2,4,8,16 --order confidence,left_to_right --out results/dream_k.jsonl
# 置信度阈值（Fast-dLLM 式，至少提交 1 个）
python scripts/run_dllm.py --model $M --data $D --mode skeleton \
    --k 1 --threshold 0.99,0.95,0.9,0.7,0.5 --out results/dream_tau.jsonl
# 块长度
python scripts/run_dllm.py --model $M --data $D --mode skeleton \
    --k 4 --block-length 8,32,none --out results/dream_block.jsonl
python scripts/summarize.py results/dream_k.jsonl --by category
```

LLaDA2.0 必须带块（例如 `--block-length 32`），块边界按绝对位置对齐，与官方实现一致。

### 3. RQ3：ParaProbe（图 4）

```bash
python scripts/prepare_data.py --per-cell 20       # 3,360 条
python scripts/run_dllm.py --model $M --data probe:data/paraprobe.jsonl --mode skeleton \
    --k 1,2,4,8 --out results/dream_probe.jsonl
# 验证命题 1：温度采样、多个种子
python scripts/run_dllm.py --model $M --data probe:data/paraprobe.jsonl --mode skeleton \
    --k 8 --temperature 1.0 --seeds 0,1,2 --out results/dream_probe_T1.jsonl
python scripts/summarize.py results/dream_probe.jsonl --by meta.n --by meta.entities
```

### 4. 归因（表 1）与 DVS（图 5）

只能用确定性配置（温度 0、顺序不是 random）。`--cfg-tag` 取 summarize 输出里的 `cfg` 列：

```bash
python scripts/attribute.py --model $M --results results/dream_k.jsonl --data $D \
    --cfg-tag confidence_k4_tnone_bfull_T0.0 --out results/dream_attr_k4.jsonl
python scripts/dvs.py --model $M --results results/dream_k.jsonl --data $D \
    --cfg-tag confidence_k4_tnone_bfull_T0.0 --out results/dream_dvs_k4.jsonl
```

`attribute.py` 打印每类错误的修复比例、安慰剂的“改坏”比例，以及两者之差（可归因比例）。只在单 token 步中提交的错误单独计数：它们按定义不可能来自同时提交。

## 算力粗估

每次运行都是 batch=1、没有 KV cache，一次前向约等于 8B 模型处理一遍“提示 + 画布”（约 1k token）。

- skeleton 模式只对取值槽加掩码（BFCL 并行题约 60 个掩码位置），k=1 时约 60 次前向，**便宜**。
- free 模式 256 个生成位置，k=1 时 256 次前向，贵 4 倍左右。
- 所以主实验（RQ2、RQ3）放在 skeleton 模式，free 模式只跑少数配置作对照。

## 已知限制 / 下一步

1. **全文法约束没做**。free 模式目前不保证语法。计划是把 eth-sri/constrained-diffusion 的“可补全性检查”包装成 `Constraint` 接口：`filter` 屏蔽会导致不可补全的 token；或者在提交后检查、不合法就重采。它自带的生成循环不能直接用，因为我们需要自己控制调度和轨迹。
2. **Dream 没有能逐 token 对齐的官方参考**：它的官方调度按时间步分配提交数，与固定 k 不同。Dream 的冒烟测试只做端到端检查。
3. **骨架的分词边界**：骨架文本和取值是分开分词的，与模型自然生成时的分词可能不同（例如 `"Paris` 被拆成 `"` + `Paris`）。dLLM 和 AR 用的是同一套骨架，比较仍然公平，但论文里要说明。
4. **槽长度固定**（字符串 12、整数 6、数组 32 个 token），多余位置用 EOS 填充，解码时丢弃。超长取值会被截断；如有需要，可通过 `slot_lengths` 调整。
5. **效率**：batch=1、没有 KV cache。如果时间不够，可以按提示长度分桶做 batch（LLaDA / Dream 需要 attention mask）。
6. BFCL 里有少数样例本身无解（标准答案用了 schema 里不存在的参数，或必填参数没有可接受值），`tests/test_taxonomy.py` 里有说明。所有模型同样受影响，不影响相对比较。
