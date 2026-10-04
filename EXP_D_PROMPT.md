# 实验 D 执行方案：格式宽容的槽，以及位置 agent（给另一个 Claude）

把下面分隔线之间的内容整段粘贴给负责实验 D 的 Claude。论文写作在另一个会话里进行，两边通过 git 分支交接。实验 D 的代码和测试**已经写好并推到写作分支**，你只需要运行、检查关卡、汇报。

---

你负责一篇 AAMAS 2027 投稿论文的一组补充实验（实验 D）。论文作者已经批准了这组实验和它的 GPU 时间。截稿是北京时间 2026-10-09 20:00；**结果最晚在 10-05 18:00（北京时间）前推上来，越早越好**。你的任务是：在 GPU 上运行、检查关卡、汇报结果。不要改动实验设计或代码逻辑；遇到问题先停下来问。

## 1. 背景（只需要知道这些）

仓库：`https://github.com/cunicle/aamas2`。从分支 `claude/elegant-goldberg-jjk1e0` 的最新提交新建分支 `exp-d`，你的所有提交都推到 `exp-d`。

开工前先读：
- `HANDOFF.md` 第 9 节：RunPod 的操作注意事项（环境、下载、拉回结果、停机）；
- `EXP_C_PROMPT.md`：上一轮实验的方案，环境和规则相同；
- `ptcdiag/decoding/constraints.py` 开头的说明（Format-tolerant variant、Position agents 两段），以及 `tolerant_classes`、`normalize_value`、`SkeletonConstraint` 的 `tolerant` 和 `active_call` 参数；
- `ptcdiag/data/agents.py` 的 `POSITION_PROTOCOLS`、`position_agent_example`；
- `scripts/run_minimal.sh` 末尾的两个新阶段 `tolerant_dream`、`position_agents`；
- `tests/test_exp_d.py`。

**论文研究什么**：dLLM 工具调用 agent 用骨架约束解码，只有参数值是被 mask 的槽。同一步提交的槽像互不通信、同时行动的决策者。论文还把同样的请求交给一组 LLM agent，每个 agent 做一个调用。

**模拟审稿提出的两个质疑，实验 D 回答它们：**

- **A. 类型掩码**：现有的槽只接受该参数 JSON 类型的 token（字符串不能含引号和反斜杠；整数只能是数字）。teacher-forced 探针显示，槽里多一个 mask 时，模型最想写的下一个 token 有 59% 被类型掩码禁止：字符串里多是转义换行 `\n`，整数后面是 `L` 或 `.`；多两个 mask 时，95% 的整数槽想写小数点（`500000.0`）。所以“多一个 mask 就写出更长的值”（`HSBC Bank`、金额多一位）可能主要是类型掩码造成的：模型想写填充，被禁止后只能写内容。

  A 用“格式宽容的槽”回答：槽也接受填充和格式 token（只要不破坏周围的 JSON），解码时再规范化。字符串去掉末尾空白（包括转义换行）；数值和布尔值取最长的合法前缀（`500000.0` → `500000`）。代码是 `--tolerant`。

- **B. 信息不对等**：画布上的槽看得到整个骨架，而团队里的 agent 只看到自己那一个调用的骨架。

  B 用“位置 agent”（协议名 `pos-anon`）回答：agent i 拿到原请求（不加任何说明）和**整轮的骨架**，只填第 i 个调用。Dream 让其他调用的槽始终保持 mask；Qwen 在其他调用的槽里写占位符（字符串写 `...`，其他类型写 `null`）。它看得到自己的调用在整轮里的位置，但看不到其他 agent 的选择。它的角色只来自它在共享骨架上的位置。

## 2. 条件（都是贪心解码，exact 槽长，除非另注）

### A：格式宽容的槽（Dream，一张画布，阶段 `tolerant_dream`）

| 槽长 | k | 请求 | 输出文件 |
|---|---|---|---|
| exact、+1、+2、+8 | 1 | 400 | `results/dream/bfcl_tolerant.jsonl` |
| exact、+1 | 16 | 400 | 同上 |
| swap | 1 | 165 | `results/dream/bfcl_tolerant_swap.jsonl` |
| 单侧加长（onesided） | 1 | 165 | `results/dream/bfcl_tolerant_onesided.jsonl` |
| 模型自估的槽长 | 1、16 | 400 | `results/dream/bfcl_tolerant_estimate.jsonl` |

**自估槽长**：用 10-03 的 `results/dream/length_estimate.jsonl`，来自 `release/results_2026-10-03.tar.gz`。阶段里用环境变量 `EST` 指定路径，默认 `results/dream/length_estimate.jsonl`；文件在别处时这样运行：
```bash
EST=<路径> bash scripts/run_minimal.sh tolerant_dream
```
记录里的 `length_mode` 必须是 `length_estimate`。

### B：位置 agent（阶段 `position_agents`）

| 模型 | 数据 | 输出文件 |
|---|---|---|
| Qwen（AR）、Dream（k=1，confidence 顺序） | BFCL 团队对称子集 104 条 | `results/{qwen,dream}/agents_d.jsonl` |
| 同上 | choose-N 全部 105 条（list 60 + open 45） | `results/{qwen,dream}/agents_pos.jsonl` |

## 3. 执行步骤与关卡（每个关卡不过就停下汇报）

1. **CPU**：
   - 把 BFCL 数据放到 `data/bfcl`：运行 `scripts/prepare_data.py`，或者等加载器自动下载。
   - `python -m pytest tests -q` 必须全部通过：**120 passed**，包括 `tests/test_exp_d.py` 的 15 个。
   - dry run，检查位置 agent 的输入（只有原请求，没有说明）：
     ```bash
     python scripts/run_agents_bfcl.py --subset sym --protocols pos-anon --dry-run --limit 2
     python scripts/run_agents.py --protocols pos-anon --variant list --dry-run --limit 2
     ```
2. **默认行为不变**（GPU）：不加 `--tolerant`，用 Dream 重跑 5 条 BFCL（k=1 和 k=16）。输出 `text` 必须与 `release/results_2026-10-03.tar.gz` 里 `results/dream/bfcl_skel_k.jsonl` 对应记录逐字相同。
3. **冒烟**（GPU）：
   - **宽容的槽**：在前 20 条上跑 exact 和 +1（k=1，`--limit 20`，输出到临时文件）。打印 3 条 +1 的最终画布和解码出的 `text`，确认槽里出现了填充（例如转义换行、小数点），而且 `text` 里已经去掉。exact 的 set accuracy 不能比原界面在同样 20 条上低 10 个点以上，否则停下汇报。
   - **位置 agent**：两个模型各跑 BFCL 3 条、choose-N 3 条（`--limit 3`，输出到临时文件）。打印每个 agent 的输出 `text`，确认：
     - 其他调用的值是占位符（Qwen）或为空（Dream 的 mask 解码为占位符 `...` / `null`）；
     - 自己那个调用有值。
4. **理智检查**（GPU，主实验跑完后做）：Dream 位置 agent 在 choose-N 上选出的城市，必须与 Dream 画布 k=16 的输出逐条相同（`release/results_2026-10-03.tar.gz` 里的 `results/dream/choose.jsonl`，`cfg_tag` 为 `confidence_k16_tnone_bfull_T0.0`）。
   - **为什么必须相同**：choose-N 的城市槽各只有一个 token。k=16 时它们在第一步都从全 mask 的画布上提交；位置 agent i 在同一张画布上只提交第 i 个槽，所以 logits 相同，选出的 token 也相同。
   - **不同怎么办**：报告不同的条数和样例，再停下来问。
5. **主实验**，按这个顺序跑（时间不够时前面的优先）：
   1. `bash scripts/run_minimal.sh position_agents`；
   2. `bash scripts/run_minimal.sh tolerant_dream`。
   
   两个阶段都能断点续跑。
6. **汇报**：用下面的命令出快速结果，连同关卡记录一起汇报：
   ```bash
   python scripts/exp_d_quick.py results/dream/bfcl_tolerant*.jsonl results/*/agents_d.jsonl results/*/agents_pos.jsonl
   ```

## 4. 预先写下的预测（结果相反时原样报告，不要为了符合预测去调设置）

**A（宽容的槽）**：两种假设，预测相反。

| 假设 | 预测 |
|---|---|
| 类型掩码 | 模型想用填充补满多余的 mask，被类型掩码禁止后才写出内容。<br>宽容的槽里 +1、+2 的 set accuracy 大幅回升（原界面分别是 15.5% 和 3.3%，exact 是 89.8%）；<br>swap 下被加长的槽写兄弟值的比例下降。 |
| 长度信号 | 模型把 mask 数当作值的长度，有填充也照样写更长的内容。<br>宽容的槽里 +1、+2 仍然很低。 |

**B（位置 agent）**：
- **Dream**：如果共享骨架上的位置本身就是角色，那么即使看不到别人的选择，BFCL 104 条上的 set accuracy 也接近画布 k=16 的 81.7%，远高于编号 agent 的 28.8%。
- **Qwen**：结果不确定，这正是要回答的问题。参照值：Qwen 编号 agent 0%、告知规则 34.6%、单个 agent 91.3%。
- **choose-N**：Dream 与画布 k=16 相同（理智检查）。Qwen 没有预测。

## 5. 资源和预算

- 一张 40GB 以上的卡就够；Dream 和 Qwen 依次加载，用 `PY_DREAM` 的环境（transformers 4.46.2）。
- **工作量估计**：
  - A：约 3,500 条完整请求的画布解码，主要是 k=1；
  - B：约 1,200 次 agent 解码。
- **估计 2–3 GPU 小时**。冒烟时测一下速度，再更新这个估计。
- 累计 GPU 时间超过 **4 小时**时，先停下来问用户。
- 跑完立即把结果拉回本地，然后停机（HANDOFF 第 9 节），不要让 pod 空转。

## 6. 交付

推到 `exp-d`：
- 原始 JSONL 打包成 `release/exp_d_<日期>.tar.gz`：
  - `results/dream/bfcl_tolerant*.jsonl`；
  - `results/{qwen,dream}/agents_d.jsonl`；
  - `results/{qwen,dream}/agents_pos.jsonl`；
  - 关卡的输出和日志。
- `exp_d_quick.py` 的输出：`results/summary/exp_d_quick.md`，用 `git add -f`。

**不要改动 `paper/` 下的任何文件，也不要改已有的代码和结果文件**，写作那边会同时在改。如果必须修代码（例如 GPU 上才暴露的错误），单独提交，commit message 写清原因，并在汇报里说明。

向用户汇报：
1. 关卡 1–4 的结果；
2. `exp_d_quick.md` 的三张表；
3. 与预测（第 4 节）逐条对照，哪里相反；
4. 宽容的槽：+1 的两个画布样例（填充和规范化前后的值）；
5. 位置 agent：一个 BFCL 团队和一个 choose-N 团队的输出；
6. 实际花费的 GPU 时间，以及任何偏离本方案的地方及原因。

## 7. 规则

- 只跑本方案里的配置；需要改设计时先问。
- 没跑出来的数字写 TBD，不要估计或外推。
- 结果和预测相反时原样报告。

---
