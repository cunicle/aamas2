# 实验 F 执行方案：角色提示的其他措辞（给另一个 Claude）

把下面分隔线之间的内容整段粘贴给负责实验 F 的 Claude。Rebuttal 的写作在另一个会话里进行，两边通过 git 分支交接。实验 F 的代码、测试、关卡脚本和汇总脚本**已经写好并提交在分支 `exp-f` 上**，你只需要运行、检查关卡、汇报。

---

你负责一篇 AAMAS 2027 投稿论文在 rebuttal 阶段的一组补充实验（实验 F）。论文作者已经批准了这组实验和它的 GPU 时间。Rebuttal 截止是 2026-11-24 23:59 AoE，即北京时间 11-25 19:59；**结果最晚在北京时间 11-24 12:00 前推上来，越早越好**。你的任务是：在 GPU 上运行、检查关卡、汇报结果。不要改动实验设计或代码逻辑；遇到问题先停下来问。

## 1. 背景（只需要知道这些）

仓库：`https://github.com/cunicle/aamas2`。代码在分支 `exp-f` 上（从投稿版分支 `claude/elegant-goldberg-jjk1e0` 的 1df2892 分出），你的所有提交都推到 `exp-f`。

开工前先读：
- `HANDOFF.md` 第 9 节：RunPod 的操作注意事项（环境、同步代码、拉回结果、停机）；
- `EXP_E_PROMPT.md`：上一轮实验的方案，环境和规则相同；
- `ptcdiag/data/agents.py` 里的 `WORDINGS`、`wording_texts`、`agent_example`；
- `scripts/run_minimal.sh` 末尾的阶段 `wording_ar`、`wording_dream`；
- `scripts/pod/exp_f/` 下的七个脚本，以及 `scripts/exp_f_quick.py`；
- `tests/test_wordings.py`。

**论文里被质疑的结论**：一组 7B agent 同时行动、每个 agent 做一个调用时，只在 prompt 里给它一个编号（`sim-label`），或者编号加上一条规则（`sim-rule`：“第 i 个 assistant 做请求里提到的第 i 个调用”），大多不能让它们各做各的调用；而只给它整轮骨架上的位置（位置 agent，`pos-anon`）就可以。在 104 条团队对称的 BFCL 请求上，set accuracy 是：

| | 编号 | 编号 + 规则 | 位置 agent | 轮流（带编号） |
|---|---|---|---|---|
| Dream | 28.8% | 48.1% | 85.6% | 85.6% |
| Qwen2.5 | 0% | 34.6% | 54.8% | 88.5% |

**两位审稿人的质疑**：每种协议只用了一种措辞，结论可能取决于措辞。

**实验 F 只做一件事**：给编号和规则各换三种措辞，其余条件（模型、请求、骨架、解码）与实验 B、C 完全相同。

## 2. 条件（贪心解码，exact 槽长；Dream 为 k=1、confidence 顺序）

六种新措辞，协议名是“原协议@措辞”。每种措辞携带的信息与原协议相同，只改一样东西：

| 协议名 | 改了什么 |
|---|---|
| `sim-label@para` | 换一种说法（“handled by n assistants working in parallel, and you are number i”） |
| `sim-label@ordinal` | 编号改成序数词（“You are the second of 3 assistants”） |
| `sim-label@first` | 角色说明放在请求**之前**，请求之后只留“Make your one call.” |
| `sim-rule@para` | 换一种说法，规则写成“The assistants share one rule: …” |
| `sim-rule@direct` | 规则改成直接的指令（“Make only the second of the calls that the request asks for …”） |
| `sim-rule@first` | 角色说明和规则放在请求之前 |

确切文本见 `tests/test_wordings.py::test_wording_texts_verbatim`。

| 数据 | 请求 | 模型 | 阶段 | 输出文件（新文件） | 记录数 |
|---|---|---|---|---|---|
| BFCL 团队对称子集 | 104 | Qwen2.5（AR） | `wording_ar` | `results/qwen/agents_f1.jsonl` | 6 × 104 = 624 |
| choose-N 的 list 请求 | 60 | Qwen2.5（AR） | `wording_ar` | `results/qwen/agents_f_choose.jsonl` | 6 × 60 = 360 |
| BFCL 团队对称子集 | 104 | Dream | `wording_dream` | `results/dream/agents_f1.jsonl` | 624 |
| choose-N 的 list 请求 | 60 | Dream | `wording_dream` | `results/dream/agents_f_choose.jsonl` | 360 |

两个阶段都只用 `PY_DREAM` 的环境（transformers 4.46.2），不需要 LLaDA2.0。原措辞不重跑（关卡 2 除外），汇总时从旧记录里读。

## 3. 执行步骤与关卡（每个关卡不过就停下汇报）

脚本都在 `scripts/pod/exp_f/`，写法与实验 E 相同。

0. **装环境、同步代码**：按 HANDOFF 第 9 节。
   - 用 `scripts/pod/setup_no_volume.sh`（与实验 E 相同；它也会装 LLaDA2.0 的环境和模型，用不到，不影响）。
   - 代码用 `git bundle` 同步到 pod，不要打包 Windows 工作区。
   - `gate_f.sh` 会等 `/workspace/aamas2_logs/setup2_done`，所以可以在装环境的同时把下面三条都挂上：
   ```bash
   nohup bash scripts/pod/exp_f/gate_f.sh   > /workspace/gate_f.log 2>&1 &
   nohup bash scripts/pod/exp_f/chain_f.sh  > /workspace/chain_f.log 2>&1 &
   nohup bash scripts/pod/exp_f/finish_f.sh > /workspace/finish_f.log 2>&1 &
   ```
   - 本地同时启动 `POD_HOST=<ip> POD_PORT=<port> bash scripts/pod/exp_f/pull_and_stop.sh`（后台），它等结果打包后拉回并停机。
   - 进度用 `bash scripts/pod/exp_f/status_f.sh` 看。
1. **关卡 1（CPU，`gate_f.sh`）**：
   - `scripts/prepare_data.py` 退出码 0；数据文件的 md5 与实验 D 的相同。
   - `python -m pytest tests -q` 必须是 **132 passed**（原有 121 个加 `tests/test_wordings.py` 的 11 个）。
   - `bash -n scripts/run_minimal.sh` 无输出。
   - 不加载模型打印六种新措辞的文本（`--dry-run --limit 2`），存到 `/tmp/gate_f/texts_bfcl.txt` 和 `texts_choose.txt`。**汇报前读一遍这两个文件**，确认每个 agent 的文本里请求只出现一次、编号和序数词正确。
2. **关卡 2（GPU，复现旧结果）**：两个模型各用**原措辞**的 `sim-label`、`sim-rule` 重跑 BFCL 的前 5 条和 choose-N list 的前 5 条。每个 agent 的 `user_text` 和输出 `text` 必须与旧记录逐字相同。旧记录是 `results/{qwen,dream}/agents_c1.jsonl`（BFCL）、`agents.jsonl` 和 `agents_rule.jsonl`（choose-N），来自 `release/` 下实验 B、C 的压缩包，`gate_f.sh` 会解压。
   - `gate_check_f.py` 的最后一行是 `ALL GATES PASS` 才算过；`chain_f.sh` 只在过了之后才跑。
   - 不同就停下汇报：列出不同的条数和一个样例。
3. **主实验（`chain_f.sh`）**：先 `wording_ar`，再 `wording_dream`，都能断点续跑。日志在 `results/log_f_ar.txt`、`results/log_f_dream.txt`。
4. **关卡 4（`finish_f.sh` 调用 `gate4_check_f.py`）**：四个文件的记录数正好是第 2 节的数；每个（请求，措辞）正好一条；没有 `error` 字段；请求集合与旧的 `sim-label` 运行相同；每个 agent 的文本里请求正好出现一次。最后一行是 `GATE 4 PASS` 才算过。
5. **汇总**：`finish_f.sh` 会运行
   ```bash
   python scripts/exp_f_quick.py --results results > results/summary/exp_f_quick.md
   ```
   然后打包、等 45 分钟、在 pod 内停机。

## 4. 预先写下的预测和判定规则（结果相反时原样报告，不要为了符合预测去调设置）

预测来自论文的结论“prompt 里的编号或规则是弱角色”：

| 条件 | 预测 | 依据（原措辞） |
|---|---|---|
| BFCL，Dream，编号的三种新措辞 | set accuracy 都不超过 50% | 原措辞 28.8% |
| BFCL，Qwen2.5，编号的三种新措辞 | 都不超过 20% | 原措辞 0% |
| BFCL，Dream，规则的三种新措辞 | 都在 30% 到 70% 之间 | 原措辞 48.1% |
| BFCL，Qwen2.5，规则的三种新措辞 | 都在 15% 到 60% 之间 | 原措辞 34.6% |
| BFCL，规则的三种新措辞之间 | `direct` 最高 | 直接的指令比“约定”更明确 |
| BFCL，Dream，全部六种新措辞 | 都比 Dream 的位置 agent（85.6%）低 20 个百分点以上 | 论文的主要对比 |
| choose-N，编号的三种新措辞 | 有重复城市的团队，Dream 不少于 50%，Qwen2.5 不少于 80% | 原措辞 78.3%、100% |
| choose-N，规则的三种新措辞 | 有重复城市的团队，Dream 不少于 25%，Qwen2.5 不少于 40% | 原措辞 45.0%、70.0% |
| `first`（角色放在请求之前） | 不作方向预测 | 没有相邻的已有结果 |

**判定规则（事先定好）**：
- 如果 Dream 的六种新措辞相对位置 agent 的差都显著（Holm 校正后 p < 0.05，校正范围是汇总表里的 12 个检验），论文的结论在这些措辞下成立。
- 如果有任何一种措辞与位置 agent 的差不显著，或者 set accuracy 达到 70% 以上，就原样报告“该措辞下编号 / 规则能起到角色的作用”，论文的结论要相应收窄。

两种结果都会写进 rebuttal。

## 5. 资源和预算

- 一张 40GB 以上的卡就够；之前用的是 A100 80GB。
- **工作量**（按实验 B、C、D 记录里的实测耗时估算）：
  | 步骤 | 内容 | 估计时间 |
  |---|---|---|
  | 装环境、下载模型 | `setup_no_volume.sh` | 约 10 分钟 |
  | 关卡 1、2 | 测试，两个模型各 20 个团队 | 约 10 分钟 |
  | `wording_ar` | Qwen2.5：BFCL 1,734 个 agent，choose-N 1,080 个 | 约 15 分钟 |
  | `wording_dream` | Dream：同样的 agent 数，k=1 | 约 45 分钟 |
- **合计约 1.5 GPU 小时**。关卡 2 跑完后按实测的每个 agent 耗时更新这个估计。
- 累计 GPU 时间超过 **3 小时**时，先停下来问用户。
- 跑完立即把结果拉回本地，然后停机，不要让 pod 空转。`finish_f.sh` 在 pod 内排了 45 分钟后的自动停机，`pull_and_stop.sh` 在本地拉回后立即停机；两条都要挂上。

## 6. 交付

推到 `exp-f`：
- `release/exp_f_<日期>.tar.gz`（`finish_f.sh` 打好的包），路径保持 `results/...`，包含：
  - `results/{qwen,dream}/agents_f1.jsonl`、`results/{qwen,dream}/agents_f_choose.jsonl`；
  - `results/summary/exp_f_quick.md`；
  - 日志 `results/log_f_*.txt`；
  - 关卡输出 `results/gate_f/`。
- `results/summary/exp_f_quick.md`，用 `git add -f`。

**不要改动 `paper/` 下的任何文件，也不要改已有的代码和结果文件。** 如果必须修代码（例如 GPU 上才暴露的错误），单独提交，commit message 写清原因，并在汇报里说明。

向用户汇报：
1. 关卡 1、2、4 的结果；
2. `exp_f_quick.md` 的四张表和每个模型的范围行；
3. 与预测（第 4 节）逐条对照，哪里不符；按判定规则，结论是成立还是要收窄；
4. 实际花费的 GPU 时间，以及任何偏离本方案的地方及原因。

## 7. 规则

- 只跑本方案里的配置；需要改设计时先问。
- 没跑出来的数字写 TBD，不要估计或外推。
- 结果和预测不符时原样报告。

---
