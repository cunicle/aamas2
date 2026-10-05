# 实验 E 执行方案：补齐图 3 没跑的点（给另一个 Claude）

把下面分隔线之间的内容整段粘贴给负责实验 E 的 Claude。论文写作在另一个会话里进行，两边通过 git 分支交接。实验 E 不需要新代码：两个运行阶段和快速汇总脚本**已经写好并推到写作分支**，你只需要运行、检查关卡、汇报。

---

你负责一篇 AAMAS 2027 投稿论文的一组补充实验（实验 E）。论文作者已经批准了这组实验和它的 GPU 时间。截稿是北京时间 2026-10-09 20:00；**结果最晚在 10-06 18:00（北京时间）前推上来，越早越好**。你的任务是：在 GPU 上运行、检查关卡、汇报结果。不要改动实验设计或代码；遇到问题先停下来问。

## 1. 背景（只需要知道这些）

仓库：`https://github.com/cunicle/aamas2`。从分支 `claude/elegant-goldberg-jjk1e0` 的最新提交新建分支 `exp-e`，你的所有提交都推到 `exp-e`。

开工前先读：
- `HANDOFF.md` 第 9 节：RunPod 的操作注意事项（两个 Python 环境、下载、拉回结果、停机）；
- `EXP_D_PROMPT.md`：上一轮实验的方案，环境和规则相同；
- `scripts/run_minimal.sh` 里的两个新阶段 `fill_fig3_dream`、`fill_fig3_llada2`，以及它们参照的旧阶段 `closer_dream_main`、`closer_probe`、`surplus_ar`、`surplus_llada2`、`surplus1_llada2`、`lensweep_llada2`；
- `scripts/exp_e_quick.py`。

**论文研究什么**：dLLM 工具调用 agent 用骨架约束解码，参数值是被 mask 的槽，槽长（mask 个数）事先固定。图 3 给每个槽多加 s 个 mask（s = 0, 1, 2, 4, 8），看 set accuracy 和“值写完后下一个 token 是收尾符”的概率怎样随 s 变化。

**实验 E 只做一件事**：把图 3 里没跑的点补上。条件、代码、请求集合都和已经跑过的相邻点完全相同，只换 s 或 k。不涉及新方法。

## 2. 条件（贪心解码，除非另注）

| 图 3 | 模型 | 条件 | 请求 | 阶段 | 输出文件（追加） |
|---|---|---|---|---|---|
| (a) 虚线 | Dream | 收尾符在槽内，s=4，k=1 | 400 | `fill_fig3_dream` | `results/dream/bfcl_closer.jsonl` |
| (c) 虚线 | Dream | 收尾符在槽内的探针，s=4 | 3,063 个槽 | `fill_fig3_dream` | `results/dream/length_prior_closer_s4.{jsonl,txt}`（新文件） |
| (a) 菱形 | Qwen2.5 AR | s = 1, 2, 4 | 400 | `fill_fig3_dream` | `results/qwen/bfcl_surplus.jsonl` |
| (b) | LLaDA2.0 | s=4，k = 4, 1, 16 | 100（`--per-data 50`） | `fill_fig3_llada2` | `results/llada2/bfcl_surplus.jsonl` |
| (b) | LLaDA2.0 | k=16，s = 2, 8 | 100 | `fill_fig3_llada2` | 同上 |
| (c) | LLaDA2.0 | 探针，s=2 | 3,063 个槽 | `fill_fig3_llada2` | `results/llada2/length_prior_s2.{jsonl,txt}`（新文件） |

`fill_fig3_dream` 用 `PY_DREAM` 的环境（transformers 4.46.2），`fill_fig3_llada2` 用 `PY_LLADA2` 的环境（4.57.1）。两个阶段互不依赖，一张 80GB 的卡可以并行跑。

**新结果必须追加到旧文件里。** 补充材料打包时，后解压的记录包会覆盖同名文件，所以交回去的必须是“旧记录 + 新记录”的完整文件。开工时先把旧记录解压到 `results/`：
```bash
tar xzf release/results_2026-10-03.tar.gz   # qwen/llada2 的 bfcl_surplus.jsonl、探针
tar xzf release/exp_c_2026-10-04.tar.gz     # dream/bfcl_closer.jsonl、收尾符在槽内的探针
```
然后记下三个要追加的文件的行数，并各存一份原样副本，供关卡 4 比对：
```bash
mkdir -p /tmp/old
for f in results/dream/bfcl_closer.jsonl results/qwen/bfcl_surplus.jsonl results/llada2/bfcl_surplus.jsonl; do
  wc -l $f; cp $f /tmp/old/$(echo $f | tr / _)
done
```
运行脚本会按条件（含 s、k、接口变体）跳过已经跑过的部分，所以只会追加新条件。

## 3. 执行步骤与关卡（每个关卡不过就停下汇报）

1. **CPU**：
   - BFCL 数据放到 `data/bfcl`：运行 `scripts/prepare_data.py`，或者等加载器自动下载。
   - `python -m pytest tests -q` 必须全部通过：**121 passed**。
   - `bash -n scripts/run_minimal.sh` 无输出。
2. **复现旧结果**（GPU）：用已经跑过的相邻条件各重跑几条，输出到临时文件，与旧记录对比。
   ```bash
   T=/tmp/gate_e; mkdir -p $T
   $PY_DREAM scripts/run_dllm.py --model Dream-org/Dream-v0-Instruct-7B --data bfcl:parallel,parallel_multiple \
       --mode skeleton --closer-in-slot --surplus 2 --k 1 --order confidence --limit 5 --out $T/dream_closer_s2.jsonl
   $PY_DREAM scripts/run_ar.py --model Qwen/Qwen2.5-7B-Instruct --data bfcl:parallel,parallel_multiple \
       --mode skeleton --surplus 8 --limit 5 --out $T/qwen_s8.jsonl
   $PY_LLADA2 scripts/run_dllm.py --model inclusionAI/LLaDA2.0-mini --data bfcl:parallel --data bfcl:parallel_multiple \
       --per-data 50 --mode skeleton --block-length 32 --surplus 2 --k 4 --order confidence --limit 5 \
       --out $T/llada2_s2_k4.jsonl
   $PY_LLADA2 scripts/length_prior.py --model inclusionAI/LLaDA2.0-mini --data bfcl:parallel,parallel_multiple \
       --surplus 1 --block-length 32 --limit 3 --out $T/llada2_probe_s1.jsonl
   ```
   - **前三个**：每条记录的 `text` 必须与旧文件里 `id`、`surplus`、`cfg_tag` 都相同的记录逐字相同。旧文件分别是 `results/dream/bfcl_closer.jsonl`、`results/qwen/bfcl_surplus.jsonl`、`results/llada2/bfcl_surplus.jsonl`。
   - **探针**：与 `results/llada2/length_prior_s1.jsonl` 里同一个槽（`id`、`call`、`param` 相同）的 `p_close` 之差都不超过 0.01。
   - **注意 `--limit`**：先确认它取的是和旧运行相同的前几条请求。LLaDA2.0 的 100 条是 `--per-data 50` 选出来的，确认 `--limit 5` 取的是这 100 条里的前 5 条。如果对不上，换成按 `id` 对比。不同就停下汇报：列出不同的条数和一个样例。
3. **主实验**：两个阶段可以并行，都能断点续跑。
   ```bash
   bash scripts/run_minimal.sh fill_fig3_dream    2>&1 | tee results/log_e_dream.txt
   bash scripts/run_minimal.sh fill_fig3_llada2   2>&1 | tee results/log_e_llada2.txt
   ```
4. **旧记录没有被改动**：三个追加的文件里，前 N 行（N 是第 2 节记下的行数）必须与 `/tmp/old/` 里的副本逐字节相同（`head -n N 新文件 | cmp - 旧副本`）。新增行数必须正好等于：
   | 文件 | 新增行数 | 计算 |
   |---|---|---|
   | `dream/bfcl_closer.jsonl` | 400 | |
   | `qwen/bfcl_surplus.jsonl` | 1,200 | 3 × 400 |
   | `llada2/bfcl_surplus.jsonl` | 500 | 3 × 100 + 2 × 100 |

   新增的记录里不能有 `error` 字段；有的话列出来。
5. **汇报**：
   ```bash
   python scripts/exp_e_quick.py --results results > results/summary/exp_e_quick.md
   ```

## 4. 预先写下的预测（结果相反时原样报告，不要为了符合预测去调设置）

新点都是已有曲线中间或旁边的点，预测是：它们落在相邻两点之间（单调插值）。不单调本身就是结果，原样报告。

| 条件 | 预测 | 依据（已有的相邻点） |
|---|---|---|
| Qwen AR，s = 1, 2, 4 | set accuracy 在 86.5% 到 89.5% 之间，大致随 s 不降 | AR 写到收尾符就停，多余的 mask 只会放开截断；s=0 是 86.5%，s=8 是 89.0% |
| Dream 收尾符在槽内，s=4，k=1 | 在 16.8% 与 39.5% 之间 | s=2 是 16.8%，s=8 是 39.5% |
| Dream 收尾符在槽内的探针，s=4 | 平均 P(收尾) 在 34.5% 与 81.6% 之间 | s=2 是 34.5%，s=8 是 81.6% |
| LLaDA2.0，s=4，k=1 / k=4 | 分别在 8%–38% / 7%–35% 之间 | s=2 和 s=8 的值 |
| LLaDA2.0，k=16，s = 2, 4, 8 | 不高于同一 s 下 k=1 的值 | Dream 的 k=16 在每个 s 上都不高于 k=1 |
| LLaDA2.0 探针，s=2 | 平均 P(收尾) 在 19.3% 与 53.3% 之间 | s=1 是 19.3%，s=4 是 53.3% |

## 5. 资源和预算

- 一张 40GB 以上的卡就够；80GB 的卡可以两个阶段并行。
- **工作量**（按旧记录里每条的实测耗时估算）：
  | 阶段 | 内容 | 估计时间 |
  |---|---|---|
  | `fill_fig3_dream` | Dream 400 条 | 约 0.5 小时 |
  | | Qwen 1,200 条 | 约 0.4 小时 |
  | | 探针 | 约 0.1 小时 |
  | `fill_fig3_llada2` | s=4 三个 k | 约 0.9 小时 |
  | | k=16 两个 s | 约 0.1 小时 |
  | | 探针 | 约 0.1 小时 |
- **合计约 2–3 GPU 小时**，含装环境和加载模型。冒烟时测一下速度，再更新这个估计。
- 累计 GPU 时间超过 **4 小时**时，先停下来问用户。
- 跑完立即把结果拉回本地，然后停机（HANDOFF 第 9 节），不要让 pod 空转。

## 6. 交付

推到 `exp-e`：
- 打包成 `release/exp_e_<日期>.tar.gz`，路径保持 `results/...`，只包含下面这些文件：
  - `results/dream/bfcl_closer.jsonl`（完整文件：旧记录加 s=4）；
  - `results/dream/length_prior_closer_s4.jsonl` 和 `.txt`；
  - `results/qwen/bfcl_surplus.jsonl`（完整文件：旧记录加 s=1、2、4）；
  - `results/llada2/bfcl_surplus.jsonl`（完整文件：旧记录加新条件）；
  - `results/llada2/length_prior_s2.jsonl` 和 `.txt`；
  - 日志 `results/log_e_*.txt`；
  - 关卡输出：把 `/tmp/gate_e` 拷到 `results/gate_e/`。
- `results/summary/exp_e_quick.md`，用 `git add -f`。

**不要改动 `paper/` 下的任何文件，也不要改已有的代码和结果文件**，写作那边会同时在改。如果必须修代码（例如 GPU 上才暴露的错误），单独提交，commit message 写清原因，并在汇报里说明。

向用户汇报：
1. 关卡 1、2、4 的结果；
2. `exp_e_quick.md` 的四张表；
3. 与预测（第 4 节）逐条对照，哪里不符；
4. 实际花费的 GPU 时间，以及任何偏离本方案的地方及原因。

## 7. 规则

- 只跑本方案里的配置；需要改设计时先问。
- 没跑出来的数字写 TBD，不要估计或外推。
- 结果和预测不符时原样报告。

---
