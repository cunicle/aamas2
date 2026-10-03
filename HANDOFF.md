# 交接文档（2026-10-03）

本文供接手的人（或 Claude）阅读，说明这个项目做到了哪一步、数据在哪里、怎么复现，以及还剩什么没做。

- 工作分支是 `gpu-run`，2026-10-03 已快进合并进默认分支 `claude/wonderful-wozniak-9f67th`，两者内容相同。
- `LOCAL_CLAUDE_PROMPT.md` 是最初的实验 runbook，已经过时：论文的论点和实验设计都改了，以本文和 `paper/outline.md` 为准。

## 1. 现状

- **原论点已被推翻**：原来认为“并行解码造成跨调用错误”，数据不支持（Dream k 从 1 到 16，CCER 只从 1.3% 升到 2.0%）。
- **论文已收窄为长度先验研究**：
  - 暂定标题：*Masks Are Length Promises: Diagnosing Parallel Tool Calls in Diffusion Language Models*
  - 目标：AAMAS 2027，达到投稿底线即可。
- **所有实验都已跑完**，GPU pod 已停机，汇总表和原始数据都在仓库里。
- **剩下的工作是写论文**：
  - 大纲：`paper/outline.md`，里面的数字已全部填好，并标注了来源表。
  - 审稿意见原文：`paper/reviews.md`。

## 2. 用户的决定和约束

| 事项 | 决定 |
|---|---|
| 截稿 | 2026-10-09 20:00（北京时间） |
| 预算 | 总花费不超过 40 美元。实验已结束，RunPod 实际花费以账单为准；10-03 那台 pod 开了约 11.3 小时，其中约 7 小时是跑完后空转 |
| 注册的标题和摘要 | 可以改（原摘要提到的反事实归因和 DVS 已经砍掉） |
| 写作规则 | 没跑出来的数字一律标 TBD；结果与预期相反时原样报告，并说明论点怎么改；超预算的任务先问用户 |
| 原始数据 | 用户在 10-03 要求上传到仓库，见 `release/` |

## 3. 仓库地图

| 路径 | 内容 |
|---|---|
| `paper/outline.md` | 论文大纲：贡献、各节内容、全部数字、表和图的清单、摘要草稿 |
| `paper/reviews.md` | 两份模拟审稿意见原文 |
| `paper/structure.md` | 论文结构方案 v2（10-03 接手后写，待用户确认后并回 `outline.md`） |
| `results/summary/*.md, *.csv` | 所有汇总表（由 `bash scripts/run_minimal.sh summary_length` 生成） |
| `release/results_2026-10-03.tar.gz` | 全部原始结果（10-03 pod 上的 `results/` 目录，解压后约 190 MB）：28 个 JSONL、日志和汇总表 |
| `release/pod_2026-10-01_raw.tar.gz` | 10-01 旧 pod 的归档，包含原始结果、`aamas2_diag/`（pilot、参数最长长度设计的 pilot、AR 小测试）和 setup 日志 |
| `ptcdiag/` | 代码：采样器、骨架约束、AR 基线、数据、评测和错误分类 |
| `scripts/run_minimal.sh` | 所有实验阶段的入口（见第 4 节） |
| `scripts/*.py` | 运行和分析脚本（见第 5 节） |
| `scripts/pod/` | RunPod 环境安装、环境变量、状态脚本，以及当时实际使用的任务链 `chains/` |
| `scripts/diag/` | 一次性诊断脚本，里面写死了 `/workspace` 路径，已被正式脚本取代，仅作记录 |
| `tests/` | 58 个 CPU 测试（需要 Dream tokenizer），在 pod 上全部通过 |

解压原始结果：

```bash
tar xzf release/results_2026-10-03.tar.gz
```

解压后得到 `results/`。`.gitignore` 忽略了 `/results/`，所以解压后的文件不会被意外提交；`results/summary/` 是用 `git add -f` 加进仓库的。

## 4. 实验清单

均为贪心解码，使用 oracle 骨架，并且槽在收尾符处结束（字符串遇 `"`，其他类型在深度 0 遇 `,` 或 `}`），之后剩余位置强制填 pad。

BFCL 指 `bfcl:parallel,parallel_multiple`，共 400 条。“LLaDA2 子集”指 `--per-data 50`，即每类均匀取 50 条，共 100 条。

| 阶段（`run_minimal.sh`） | 模型 | 配置 | 输出 | 条数 |
|---|---|---|---|---|
| `dream` | Dream-v0-Instruct-7B | skeleton，k ∈ {1,2,4,8,16}；left-to-right；τ=0.9；ParaProbe k ∈ {1,2,4,8}；free 模式 k=2 | `dream/bfcl_skel_k`, `_ltr`, `_tau`, `probe_skel_k`, `bfcl_free` | 400 / 840 |
| `llada2` | LLaDA2.0-mini | skeleton，块长 32，k ∈ {1,2,4,8,16}（k=16 由 chainN5 补齐） | `llada2/bfcl_skel_k` | 400 |
| `pilot`（只有 pilot） | LLaDA2.0-mini | 块长 none，k ∈ {1,4,16} | `llada2/bfcl_skel_bfull` | 80 |
| `surplus_ar` | Qwen2.5-7B-Instruct（AR） | skeleton：s=0 和 s=8 | `qwen/bfcl_skeleton`, `qwen/bfcl_surplus` | 400 |
| `lenprior_*`, `lensweep_*` | Dream / LLaDA2 | teacher-forced 长度先验探针。Dream：s ∈ {1,2,4,8}；LLaDA2：s ∈ {1,4,8} | `*/length_prior*.jsonl/.txt` | 3063 个槽 |
| `surplus_dream` | Dream | s ∈ {1,2,4,8} × k ∈ {1,4,16} | `dream/bfcl_surplus` | 400 |
| `surplus_llada2`, `surplus1_llada2` | LLaDA2 | s=1：k ∈ {1,4,16}；s ∈ {2,8}：k ∈ {1,4} | `llada2/bfcl_surplus` | 100 |
| `endbias_dream` | Dream | β ∈ {2,4,8} × s ∈ {2,8}，k=4 | `dream/bfcl_endbias` | 400 |
| `swap_dream`, `swap_llada2` | Dream / LLaDA2 | `--lengths swap`（兄弟组内槽长轮换一位），只跑有槽被改动的题。Dream：k ∈ {1,4,16}；LLaDA2：k ∈ {1,4} | `*/bfcl_swap` | Dream 165，LLaDA2 34 |
| `estimate_dream`, `estimate_llada2` | Dream / LLaDA2 | 先用 `length_estimate.py` 一次前向估计槽长，再按估计长度解码。Dream：k ∈ {1,4,16}；LLaDA2：k ∈ {1,4} | `*/length_estimate.jsonl`, `*/bfcl_estimate` | 400 / 100 |
| `choose_dream`, `choose_llada2`, `choose_ar` | 三个模型 | choose-N 探针（`data/choose.jsonl`，由 `ptcdiag/data/choose.py` 确定性生成）。Dream：k ∈ {1,2,4,16}；LLaDA2：k ∈ {1,4,16}；Qwen：AR | `*/choose.jsonl` | 105 |
| `summary_length` | 只用 CPU | 生成全部汇总表 | `results/summary/` | — |

**没做的**：
- Ling-mini-2.0（LLaDA2 的 AR 对照）只在旧 pod 上做过小测试，归档在 `aamas2_diag/ar_test` 里。
- LLaDA2 的 ParaProbe 实验，以及 LLaDA2 和 Qwen 的 free 模式。
- 完整的块长度对照。
- 反事实归因和 DVS（已从论文中删除；代码 `scripts/attribute.py` 和 `dvs.py` 还在）。
- S³ 式 null token 对照：没有单独跑。理由是骨架约束本来就允许在任何位置写 pad，而模型几乎从不在值内部放 pad（P(pad) 约 1e-5），β 偏置相当于它的加强版。审稿人可能还会追问这一点。

## 5. 分析脚本

均需 `transformers`（加载 tokenizer），在 Dream 环境下运行即可。

| 脚本 | 产出 |
|---|---|
| `scripts/length_analysis.py` | 按模型 × 长度模式 × s × β × k 统计 set_acc、syntax、CCER、SCER、overfill、NFE；同一模型只在各设置共有的题目上比较 → `length.md` |
| `scripts/slot_errors.py` | 逐槽分类（own / sibling_fit / sibling / overfill / truncated / other / unparsed）。`--common` 只用共有题；`--swap-slots` 只统计长度互换时被改动的槽，并按“槽比自己的值长 / 短”拆分 → `slots_*.md`, `swap_*.md` |
| `scripts/symmetry.py` | 长度对称子集上的 CCER，以及对称 BFCL parallel 题里“按提及顺序填槽”的比例 → `symmetry_*.md` |
| `scripts/choose_analysis.py` | choose-N 的重复率、同一步产生的重复、无效率、是否按列出顺序 → `choose.md`。open 变体的无效率反映的是 1 token 槽长的限制，只能看重复率 |
| `scripts/length_prior.py` | teacher-forced 长度先验探针（用 GPU） |
| `scripts/length_estimate.py` | 一次前向估计槽长（用 GPU），并输出估计准确度表 |
| `scripts/masquerade.py` | 协调失败标签（重复、错绑、嵌合、共享参数不一致）：k=1 + 错误槽长 vs k=16 + exact 槽长，在同一批题上比较 → `masquerade.md`。只读诊断标签，不需要 tokenizer |

## 6. 主要结果

完整表格见 `paper/outline.md` 和 `results/summary/`。

1. **k 的代价很小**：Dream 的 set_acc 从 0.897（k=1）降到 0.845（k=16），LLaDA2 从 0.877 降到 0.797。在长度对称子集上 CCER 也很低。两个模型约 95% 按提及顺序填槽，这一比例和 k 无关。
2. **choose-N（与预期相反）**：请求不规定选哪几个实体时，Dream 在同一步提交中写出重复。
   - open 变体：k≥2 时 100% 重复，k=1 时 0%。
   - list 变体：重复率最高 15%。
   - LLaDA2 的 32 token 块因果注意力使各调用实际串行解码，基本不重复；Qwen 不重复。
   - 论点因此改为：有顺序线索时并行无害，没有时会重复绑定。
3. **长度错配代价大，且不单调**：Dream k=1 时，s 从 0 到 8 的 set_acc 依次为 0.897 / 0.155 / 0.033 / 0.170 / 0.600。小幅错配主要导致写长。
4. **长度互换（误归因的直接证据）**：
   - Dream 被改动的槽中 79% 写进了长度正好合适的兄弟值，k=1 时也是如此；exact 长度下这一比例为 0%。
   - LLaDA2 为 48%，但只有 34 条题。
5. **缓解效果有限**：
   - β 偏置只对大余量有效（s=8：0.570 → 0.723）。
   - 一次前向估计槽长后解码只到 0.42（exact 长度为 0.90）。

## 7. 审稿意见处理情况（原文见 `paper/reviews.md`）

| 意见 | 处理情况 |
|---|---|
| A-P0a：LLaDA2 的 s=1 | 已跑。k=1/4/16 的 set_acc 为 0.160 / 0.120 / 0.040（s=0 为 0.920 / 0.900 / 0.870），降幅和 Dream 相当 |
| A-P0b：宽松指标和错误三分类 | 三分类已由 `slot_errors.py` 覆盖（own / overfill / sibling / other）。宽松 set_acc 只在 10-03 上午用 `scripts/diag/p0_check.py` 对部分数据算过，**最终数据没有重算**，需要的话在解压后的结果上重跑这个脚本（要改路径） |
| A-P0c：采样器是否允许提前收尾 | 允许，有测试（`tests/test_closers.py`）。正确收尾位置上的 P(收尾) 见 `length_prior.md`：Dream s=1 时为 0.010，s=8 时为 0.740 |
| A-P1：对称情形 | 用 choose-N 实验完成，结果见第 6 节第 2 条，与预期相反，已原样写入大纲 |
| A-P2：null token 基线 | 没有单独跑，理由见第 4 节“没做的”。需要时可以在骨架 prompt 里加 null 指令，作为一组新实验 |
| A-P3：free 模式 | 在第 8 节写一句：Dream 0.752，对比骨架下的 0.880 |
| B-1：新颖性 | 已按意见调整贡献，并引用 CAL、ρ-EOS、DAEDAL、S³、DiffuAgent |
| B-2：长度泄露和位置先验 | 对称子集 + 提及顺序检查（`symmetry.py`）+ choose-N |
| B-3：单槽准确率、抄兄弟值还是写长、长度互换、AR、引言 | 已全部做了或写进大纲。小错配以写长为主；长度互换支持“按长度绑定” |
| B-4：篇幅 | 已删除命题 1、块长度对照和 free 模式新跑 |

## 8. 注意事项（写论文时要说明）

- LLaDA2 的 surplus、swap 和 estimate 实验只用了 100 条子集（swap 只有 34 条），同一子集上 s=0 的 k=1 / k=4 为 0.920 / 0.900。
- AR 基线在骨架里最多写到槽长，只截断不强制写满；遇到收尾符就停。所以“AR 不受 surplus 影响”是设计性质，不是实验发现。
- 第 6 节中 set_acc 的降幅比槽级准确率小：长度互换时，整组兄弟值一起轮换后在集合匹配下仍算对。
- 10-03 pod 重跑的输出和 10-01 旧 pod 逐位相同，两边的数据可以混用。
- “参数最长长度”设计下 18 个跨调用错误中有 17 个是长度造成的，这个数字来自旧 pod 的 pilot（归档里的 `aamas2_diag/pilot_param_max/`）。

## 9. 如果还要在 RunPod 上跑

1. **连接**：
   - 网关 `ssh <pod>@ssh.runpod.io` 需要 PTY。先通过网关读出 `RUNPOD_PUBLIC_IP` 和 `RUNPOD_TCP_PORT_22`，再直连：`ssh -F /dev/null -p <port> root@<ip>`。
   - 用户的 `~/.ssh/config` 带 BOM，Git Bash 的 ssh 会报错，所以必须加 `-F /dev/null`。
2. **安装**：
   - 没有网络卷时用 `scripts/pod/setup_no_volume.sh`：模型放在 `/dev/shm/hf`，环境装在容器盘上。
   - 两个 venv：`envs.sh`。
     - Dream 用 torch 2.5.1 + transformers 4.46.2。
     - LLaDA2 用 torch 2.8.0 + transformers 4.57.1。
   - 运行前 `source env.sh`，其中必须设置 `OMP_NUM_THREADS`。pod 报告的核数有 128–255 个，但实际配额只有十几个，不限线程的话 CPU 部分会极慢。
3. **同步代码**：在本地 `git bundle`，到 pod 上 `git fetch` 后 `merge --ff-only`。不要直接打包 Windows 的工作区，会带进 CRLF。
   - 有任务在跑时，不要在 pod 仓库里 stash 或 checkout。
   - 不要用 `pkill -f`，模式会匹配到自己的 shell。
4. **排队**：用 `scripts/pod/chains/` 里的写法，把多个阶段串成一条链，再用 `nohup bash chainX.sh &` 后台运行；后一条链等前一条的 `results/CHAIN_*_DONE` 标记文件。
   - 不要修改正在运行的链文件：bash 是边执行边读文件的。
   - 一张 80GB 卡可以同时跑两条链。
5. **停机**：务必把 `scripts/pod/chains/finish_and_stop.sh` 这样的自动停机步骤排在最后，在 pod 内用 `runpodctl stop pod $RUNPOD_POD_ID` 停机。没有网络卷的 pod 一停，`/dev/shm` 和容器盘上的数据就没了，所以要先把结果拉回本地。
6. **推送**：本地推送时会卡在 git-credential-manager，改用 `git -c credential.helper= -c "credential.helper=!gh auth git-credential" push`。
