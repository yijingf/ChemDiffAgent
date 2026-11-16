# NPU inference

## 失败工具修复后继续生成 prediction

```bash
bash scripts/run_inference_npu.sh inference/configs/inference_npu_continue.yaml
```

将 `inference.retry_results` 指向原始 inference 的完整结果（不要使用工具 replay 合并文件），
`model.path` 使用同一个 checkpoint，保留原来的生成参数。所有原样本按原顺序写到
`res/<datetime>-inference_npu_continue/single_test.jsonl`。
没有 success=false 工具调用的样本原样复制；其余样本复用首次失败工具调用之前及该轮
已保存的 raw_generations，复用之前成功的工具结果，只从首次失败处重新执行工具并让
模型继续生成。失败之后的旧模型回复全部丢弃，不重新生成失败之前的模型文本。
模型仍需重新加载并 prefill 已保存上下文，不复用跨进程的 KV cache。
选择失败样本严格依据 success=false；工具返回错误提示但被标为 success=true 的情况
不会自动识别。主结果不添加重跑索引或标记；单独的 `single_test.continuation.jsonl`
保存重跑样本的 index（从 0 开始）、sample_id、roll_idx、旧 prediction、工具链、时延
及续跑起点。索引对应原文件/完整新结果的样本顺序。
新 timing 仅覆盖本次续跑，不包含历史生成耗时。没有新增模型生成时的旧样本保留原 timing。
max_turns 是整个轨迹的轮数上限，必须大于首次失败的工具轮次，给新答案留出生成机会。

## SMILES 参数的引号兼容

单分子 SMILES 工具（包括六类性质预测和逆合成）执行前允许去掉首尾的单/双引号，
但仅在去掉后 RDKit 能解析为有效分子时采用。未改变分子内部字符，不处理名称、
化学式、正向反应字符串或不合法结构。缺少 RDKit 时不清理参数。
结果的 `arguments` 和 `raw_generations` 继续保留模型原始输出，
`executed_arguments` 记录实际执行参数；replay 及合并结果也保留该字段。
这属于执行侧输入兼容策略，比较 benchmark 时应对所有模型使用相同策略，并将
原始参数合法性与修正后的执行成功率区分报告。旧结果不会自动改变，须重新执行工具。

## 仅补执行工具，不运行主模型

```bash
python3 inference/replay_tools.py output/single_test.jsonl \
  --config inference/configs/inference_npu.yaml --dry-run
python3 inference/replay_tools.py output/single_test.jsonl \
  --config inference/configs/inference_npu.yaml
```

默认只补执行 tool_use_chain 中 success=false 的调用，按工具名称+参数去重，
保留所有原始样本/轮次引用。`--missing-only` 只选缺依赖/文件等错误。
缺少 tool_use_chain 时自动解析 raw_generations；`--source raw` 强制从原文的
tool_call 标签提取，但跳过已记录成功的轮次。无法解析的调用写入 parse_errors，
不会猜测参数。安装工具不能解决错误参数、404 或网络不可达等问题。
工具路径读取 YAML runtime.env 或容器环境，默认 T5Chem 使用 CPU；不加载主模型，
不需要 model.path。输出为 `res/<datetime>-tool_replay/plan.json`、
`tool_results.jsonl`、`summary.json` 和合并后的完整 `single_test.jsonl`；dry-run 只保存 plan。
完整文件保留全部原始样本，替换重放调用的工具 output/success/seconds，旧观察保存在
`replay_history` 中。原文件不修改。prediction、messages、raw_generations、原时延不重新计算，
`tool_replay` 明确标注 prediction/trajectory 未重算，并保留原 evaluation。
合并后 all_tool_calls_succeeded 根据更新的工具结果重算，不代表新的端到端生成成功率。
失败的重放也保留，不自动修正模型给出的无效 SMILES 或其他参数。

已有 replay 结果可以只合并，不再次运行工具：

```bash
python3 inference/replay_tools.py output/single_test.jsonl \
  --merge-only res/<之前replay目录>/tool_results.jsonl
```

`--merge-only` 也可合并已完成且每行完整的部分 replay 结果。其余原样本保留。
重放旧参数只能恢复工具输出，不能恢复本应基于新工具结果生成的后续模型决策。

## 补跑缺失工具导致失败的样本

先等待原推理结束或正常停止，使结果 JSONL 不再写入，再执行：

```bash
python3 inference/prepare_tool_retry.py res/<原运行目录>/single_test.jsonl
```

脚本从失败工具调用的错误信息筛选缺包、缺源码、缺权重/文件等样本，使用
`original_data` 还原输入，并输出补跑 YAML 和启动命令。默认读取原结果旁的
`config.yaml`，也可用 `--config <yaml>` 指定更新后的工具路径配置。
加 `--all-tool-errors` 可选择所有工具失败样本（包括超时等）。格式解析失败且没有
失败工具调用的样本不会被选中。补跑是从该问题开头重新生成整个 rollout，
每个样本一次；不只是重放工具调用。默认关闭 prof，原结果不修改、不自动合并，
新结果保存在 `res/<datetime>-inference_npu_tool_retry/`。

每条结果的 `timing.forward_steps` 记录全部模型 forward 时延，包含 `phase`
（prefill / denoise / cache_commit）、`turn`（从 1 开始）、`block_index` 和
`step_index`（从 0 开始，不适用时为 null）、`fallback` 以及 `seconds`。
去噪 step 时延只覆盖模型 forward，不含 token 采样、remasking 或工具执行。
NPU/CUDA 前后同步设备后测量 wall time，CPU 使用 perf_counter；因此每步同步
会影响端到端吞吐。prof 的启停/导出不计入该 forward 时延，但采集本身可能带来开销。
`timing.total_seconds` 仍保存完整样本耗时，即使输出格式无效也会保留已有 step 记录。

## 性能采集配置

```bash
bash scripts/run_inference_npu.sh inference/configs/inference_npu_smoke.yaml
```

smoke 配置按数据集顺序跑 single_test 的前 8 条，每条 1 次 rollout，保留正式
推理的模型、block length、去噪步数和生成长度上限。prefill / denoise 各自先预热
2 次 forward，再分别采集 5 次，之后继续完成这 8 条。若实际执行次数不足，
对应阶段只能产生已执行次数的 trace；性能采集开销不能当作端到端吞吐。
输出位于启动目录的 `res/<datetime>-inference_npu_smoke/`。

完整测试仍使用 `inference_npu.yaml`（max_samples=0），输出在
`res/<datetime>-inference_npu/`。其中 `single_test.jsonl` 每完成一条 rollout 就写入并
flush，包含逐条推理结果；`run.log` 保存终端输出和最终计数，`config.yaml` 和
`commands.txt` 保存启动配置与命令，`prof/` 保存 trace。

## YAML 一键启动

编辑 `inference/configs/inference_npu.yaml` 中的 Python、CANN、模型和数据路径，然后：

```bash
bash scripts/run_inference_npu.sh inference/configs/inference_npu.yaml
# 只查看实际命令，不加载模型或启动 NPU：
bash scripts/run_inference_npu.sh inference/configs/inference_npu.yaml --dry-run
```

启动脚本使用 `${PYTHON:-python3}` 解析 YAML，需要安装 PyYAML；真正推理使用
YAML 的 `runtime.python`。`runtime.cann_env` 会自动 source，已加载环境时可设为 null。
配置中的相对模型、数据路径基于仓库根目录。YAML 启动结果固定写入启动命令时
当前目录的 `res/<YYYYMMDD_HHMMSS_ffffff>-<config文件名>/`。
无需配置顶层 `output_dir`；旧配置须删除该字段。每次运行新建目录，保存
`single_test.jsonl`、`config.yaml`、`commands.txt`、`run.log` 和 `prof/`。
prof 相对路径基于本次结果目录。每次新建目录意味着 `resume` 不会续跑历史目录；
若要续跑已有结果，可直接使用目录内 commands.txt 中的命令并加 `--resume`。
`runtime.env` 可设置工具路径、`ASCEND_RT_VISIBLE_DEVICES` 等环境变量。
工具路径默认相对于仓库。外部安装可通过 `runtime.env` 中的
`RXN_T5CHEM_ROOT`、`RXN_T5CHEM_MODEL_PATH`、`UNICORE_ROOT`、
`CHEMTOOL_AGENT_ROOT`、`CONTROLLED_CHEMICALS_CSV` 覆盖；已有容器环境变量也会保留。
YAML 中的模型、数据、CANN 绝对路径属于部署配置，迁移容器时需按实际目录修改。
正式全量推理将 `inference.max_samples` 设为 0，按需关闭 `profiling.enabled`。

`profiling.phase` 可选 `both`、`prefill`、`denoise`、`cache_commit` 或 `all`，默认
YAML 选择 `both`，一次完整推理中分别采集 prefill 和去噪 decode，各阶段独立
预热、计数和保存，不采 cache commit。每个阶段只计对应的 forward，先运行 skip_first + warmup
次而不记录，再采 active 次。每次 active forward 独立启停 profiler、同步 NPU，
保证其他阶段不会混入 trace；因此会产生采集开销，不适合衡量端到端吞吐。
结果分开保存在 `prof/<dataset>/<phase>/forward_<index>/rank_0/`。
这些独立 trace 的 native warmup 为 0；YAML warmup 表示未记录的模型预热次数。
`both` 模式无需重复运行数据集；也可用 prefill / denoise 单独采集。数据仍完整运行。
prefill 次数按 assistant turn 计，短于一个 block 的 prompt 不产生 prefill。
`all` 保留连续窗口及混合 forward 计数行为。

## 环境变量启动（兼容旧用法）

使用与 SFT 相同的 PyTorch / torch-npu / CANN 配套环境，先加载服务器的
CANN 环境。模型须为 SDAR checkpoint（包含 tokenizer 文件）。

```bash
MODEL_PATH=/path/to/checkpoint DATA_DIR=/path/to/data \
CONDA_ENV=chemdiffagent DEVICE=npu:0 MAX_SAMPLES=1 \
bash scripts/run_inference_npu.sh
```

NPU 入口直接加载仓库 `training/sft/models/sdar`，复用 SFT 的
`npu_rms_norm`、`npu_swiglu`、`npu_fusion_attention` 替换。
Attention 保留 block diffusion 的可见性 mask、GQA 和 KV cache 语义；
prefill、去噪以及 cache commit 均走这套算子。暂未切换到 FIA。
显式请求不可用的设备会报错。反应工具 T5Chem 默认使用 CPU，可通过
`RXN_T5CHEM_DEVICE` 单独配置。

NPU 脚本默认开启采集，`PROFILE=0` 关闭。默认跳过 0 次 forward、预热 1 次、
采集 2 次，然后立即停止。通过 `PROFILE_SKIP_FIRST`、`PROFILE_WARMUP`、
`PROFILE_ACTIVE` 调整。一个 step 是一次完整模型 forward，工具调用不计 step。
建议 smoke run 保证至少完成 warmup + active 次 forward。
CPU/NPU 活动采用 Level1 + PipeUtilization，默认结果位于
`outputs/inference/prof/<dataset>/rank_0/`，含版本及 schedule metadata。
重复运行建议设置不同的 OUTPUT_DIR。异常退出也会关闭 profiler。

直接运行 Python 时使用 `--device npu:0 --profile`；可通过 `--profile-dir`
指定目录，相对路径基于输出 jsonl 所在目录。

本地 CPU contract tests 不代表原生 NPU 性能或数值验证。服务器上仍需完成
上述 smoke run，检查 trace 中三个 NPU 算子、生成结果及多 block KV cache 行为。
