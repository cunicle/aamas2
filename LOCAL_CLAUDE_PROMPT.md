# 给本地 Claude Code 的运行 prompt

把下面分隔线之间的内容整段粘贴给 GPU 机器上的 Claude Code。

---

你在一台有 GPU 的机器上，帮我跑一篇 AAMAS 2027 投稿论文的全部实验。论文研究扩散语言模型（dLLM）并行工具调用中的残余错误：约束保证语法正确之后还剩哪些错误，以及这些错误是否由“同一步同时去掩码”造成。代码已经写好并在 CPU 上测过；你的任务是**在真实模型上跑通、按计划执行、盯住异常、最后汇报**，而不是重新设计实验。

## 仓库

- `git clone https://github.com/cunicle/aamas2.git`，切到分支 `claude/wonderful-wozniak-9f67th`，再从它新建分支 `gpu-run`，你的所有提交都推到 `gpu-run`。
- 先读 `README.md`，再读 `proposal.md` 的第 4–6 节（研究问题、错误分类、实验设计）。执行入口是 `scripts/run_minimal.sh`。

## 硬约束

1. **预算**：整套实验估计约 31 A100·小时（含余量）。实际用量预计超过 **40 A100·小时**时，先停下来问我。
2. **时间**：实验要在北京时间 **10 月 5 日**之前全部跑完（10 月 9 日 20:00 交论文）。
3. **不改科学设计**：模型、数据、解码配置、错误分类规则、提示模板都不要改。可以修的是让代码在真实模型上跑起来的工程问题，例如适配器、版本兼容、显存。每一处代码修改都要单独提交，commit message 写清原因。
4. **只用贪心解码**（temperature 0），这是反事实归因的前提。
5. 原始结果 `results/**/*.jsonl` 很大，不要提交；只提交汇总（见“交付”）。`.gitignore` 忽略了 `results/`，需要时用 `git add -f`。

## 环境

两个模型族依赖的 transformers 版本不同（见各模型 `config.json`），建两个虚拟环境：

```bash
python -m venv ~/envs/dream  && ~/envs/dream/bin/pip install torch "transformers==4.46.2" accelerate numpy scipy pytest
python -m venv ~/envs/llada2 && ~/envs/llada2/bin/pip install torch "transformers==4.57.1" accelerate numpy scipy pytest
export PY_DREAM=~/envs/dream/bin/python PY_LLADA2=~/envs/llada2/bin/python
```

不要在这两个环境里执行 `pip install -r requirements.txt`，那样会覆盖锁定的 transformers 版本。

- torch 的 CUDA 版本按本机驱动选。
- 在中国大陆下载慢的话，设置 `export HF_ENDPOINT=https://hf-mirror.com`。
- 提前把四个模型下载好：`Dream-org/Dream-v0-Instruct-7B`、`inclusionAI/LLaDA2.0-mini`、`Qwen/Qwen2.5-7B-Instruct`、`inclusionAI/Ling-mini-2.0`。
- LLaDA2.0-mini 和 Ling-mini-2.0 的 bf16 权重约 32GB，**必须用 80GB 卡**；Dream 和 Qwen 用 40GB 卡即可。

## 执行步骤与关卡

每一步做完先检查关卡，**关卡不过就停下来汇报，不要继续往下跑**。

**第 1 步：`bash scripts/run_minimal.sh setup`**
- 关卡：`pytest` 全部通过（36 个）。

**第 2 步：`bash scripts/run_minimal.sh smoke`**
- 关卡（看 `results/smoke_*.txt`）：
  - LLaDA2.0：`LLaDA2.0 reference match (up to EOS): True`；
  - 两个模型的两种模式都是 `determinism (resume at step 3): True`；
  - skeleton 模式 `syntax_ok=True`，而且输出文本是合理的 JSON 工具调用。
- 如果 reference match 为 False：找出第一个不一致的 token 位置，检查注意力掩码、position_ids、块边界和阈值比较，修好再重跑。
- 如果 determinism 为 False：先试 `torch.use_deterministic_algorithms(True)` 加 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；还是不行就汇报，不要跳过。
- 模型加载失败：大多是 transformers 版本问题，按模型卡说明修；修不好就汇报完整报错。

**第 3 步：`bash scripts/run_minimal.sh timing`**
- 它会打印每个模型的 `--timing` 参数，用这些参数重算预算：
  `$PY_LLADA2 scripts/estimate_cost.py --plan minimal --timing dream=..:15 --timing llada2=..:15`
- 关卡：估计值 ≤ 40 A100·小时。超了就停下来，把估计明细发给我。我倾向的缩减办法是用 `$PY_DREAM scripts/prepare_data.py --per-cell 3` 缩小 ParaProbe，但要我确认后才能做。
- 同时看 `results/timing/*.jsonl` 前 20 条：skeleton 模式的语法正确率应接近 100%，集合级准确率（`diagnosis.correct`）应明显大于 0。如果接近 0，打印 3 条输出文本，排查提示模板或适配器，不要直接开跑。

**第 4 步：两个 dLLM 并行跑**
```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_minimal.sh dream  > results/log_dream.txt  2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_minimal.sh llada2 > results/log_llada2.txt 2>&1 &
```
- 只有一张卡就顺序跑。
- 跑的过程中定期检查：
  - 日志里的 `s/run` 是否与测速一致；
  - `$PY_DREAM scripts/summarize.py results/dream/bfcl_skel_k.jsonl --by category` 开头有没有 `WARNING ... raised an exception`。
- 关卡：异常记录不超过 2%。超了就停下来，修复原因后续跑（脚本会跳过已完成的条目）。

**第 5 步：`bash scripts/run_minimal.sh ar`**
- AR 对照（Qwen2.5-7B、Ling-mini-2.0）；Ling 需要 80GB 卡。

**第 6 步：`bash scripts/run_minimal.sh attr`**
- 反事实归因 + DVS，都用 k=4 的结果。
- 关卡：`determinism check` 那一行应该是全部复现；否则归因结果不可信，停下来汇报。

**第 7 步：`bash scripts/run_minimal.sh summary`**

## 交付

1. 在 `results/REPORT.md` 写一份报告，包含：
   - **用量**：每个阶段的实际 GPU 小时和总计。
   - **冒烟测试**：关键输出。
   - **主表**：`results/summary/bfcl_*.md` 的内容。重点看每个模型的跨调用错误率（`ccer`）和单调用错误率（`scer`）怎样随 k 变化，以及 confidence k=1 与 left_to_right k=1 的对比。
   - **ParaProbe**：按 `meta.n`、`meta.entities`、`meta.ambiguity` 的表。
   - **归因与 DVS**：`attr_k4.txt` 和 `dvs_k4.txt` 的输出（每类错误的修复比例、安慰剂比例，以及 DVS 的 AUROC）。
   - **错误样例**：每个跨调用错误类别（`duplicate_call`、`cross_binding`、`chimera_value`、`omitted_call`、`inconsistent_shared_arg`）各挑 3 条 k=4 下的真实样例，给出用户请求、模型输出和标准答案，供论文图 1 选用。
   - **异常与代码修改**：遇到的所有异常、你做的代码修改和原因。
   - **你的判断**：结果是否支持 `proposal.md` 第 4 节的 H1–H4。不支持的地方如实写，不要美化。
2. `git add -f results/REPORT.md results/summary results/*/attr_k4.txt results/*/dvs_k4.txt results/smoke_*.txt`，连同代码修改一起提交，推到 `gpu-run`。
3. 原始 JSONL 打包成 `results_raw.tar.gz` 留在本机，不要提交。

遇到这份说明没覆盖的决定（改配置、删数据、换模型、超预算），先问我。
