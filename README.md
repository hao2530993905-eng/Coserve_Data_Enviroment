# CoServe Data Environment

Python 入口：配置模型 API，在现有 CoServe Agent 框架中执行一个任务，并收集原始轨迹与指定产物。项目保留仓库名 `Coserve_Data_Enviroment`。

调用链：`Python → CoServe Python SDK → CoServe Agent runtime → 模型 API / 工具 → 原始轨迹`。模型服务需要预先启动；这里启动的是 Agent runtime，不负责部署模型权重。CoServe 的提示词、工具、循环和上下文管理均由原框架执行。

## 准备

需要 Python 3.11+、Node.js 24.9+，以及已经执行 `pnpm install` 的 [CoServe checkout](https://github.com/yhz5613813/coserve-agent)。代码从指定 checkout 加载其自带 Python SDK，使用同一 checkout 的源码 runtime，不下载另一个版本的 Harness。

在本项目目录执行：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Python 调用

```python
from coserve_agent import CoServeAgent, ModelAPI

agent = CoServeAgent(
    repo="../",                         # CoServe 仓库
    workspace="examples/workspace",    # 已准备的任务项目
    model=ModelAPI(
        base_url="http://127.0.0.1:8000/v1",
        model="coserve-agent",
        api_key="EMPTY",               # 无鉴权本地服务使用占位值；真实 key 从环境变量读取
        context_window=32768,
        max_tokens=4096,
    ),
    output_dir="runs",
)

rollout = agent.run("读取 input.txt，使用工具把内容复制到 answer.txt。", artifacts=("answer.txt",))
print(rollout.final_response)
print(rollout.status, rollout.directory)
```

调用有鉴权的服务时省略 `api_key`，设置 `COSERVE_MODEL_API_KEY`，或用 `api_key_env` 指定变量名。支持提供工具调用和 SSE 流式响应的 OpenAI Chat Completions 兼容服务；`base_url` 通常以 `/v1` 结尾，模型名必须与服务实际提供的名称一致。当前适配器声明文本输入、关闭 reasoning effort 协商；多模态和厂商专用推理参数需要后续扩展。

也可使用示例脚本：

```powershell
$env:COSERVE_MODEL_API_KEY = "EMPTY"
.venv\Scripts\python examples/run_once.py "读取 input.txt，并把内容复制到 answer.txt" --model coserve-agent --artifact answer.txt
```

`--base-url`、`--model`、`--workspace`、`--repo` 可以指定其他服务和项目。`CoServeAgent(patches=(... ,))` 将额外的 CoServe 配置 overlay 传给 SDK；本项目的模型路由与轨迹存储配置最后应用。CAD/CAE 任务复用已准备好的项目与 CAStudio 服务；这里不创建或重置这些环境。

## 每次运行的输出

```text
runs/run-<id>/
  request.json             任务、模型配置、项目路径和配置引用，不含 API key
  notifications.jsonl      原始 SDK 通知，包含根会话及其后代的事件
  sessions/                CoServe 自身保存的完整、未压缩 Session 文件
  final_response.txt       最终回复
  result.json              运行状态、结束原因、错误和产物摘要
  artifacts/               显式指定的文件或目录副本
  runtime-home/            本次运行的 profile、附件和其他运行数据
  runtime.patch.json       本次模型与日志配置，不含 API key
  dsh.cmd 或 dsh           SDK 启动本地源码 runtime 所需的平台入口
```

每次调用使用新的 runtime 和 Session，工作项目原地执行，文件修改会保留。每个 Python 进程绑定一个 CoServe checkout。默认沿用 `workspace-write` 权限；需要交互的问题或授权没有 Python 回答器，CoServe 会按已有策略处理。当前不设置整次任务的墙钟超时，Ctrl+C 会进入 SDK 关闭流程并记录中断状态。

`completed` 仅表示 Agent 回合正常结束，不表示设计、仿真或任务已经通过工程验收。非正常结束保留记录；SDK 或产物收集异常抛出 `CoServeRunError`，其 `directory` 指向部分记录。缺失的指定产物会使收集失败，不会静默跳过。产物路径相对工作项目，可指定文件或目录，不支持符号链接。

Session 文件保留框架事件，可按 CoServe 的规则重建模型可见历史；它们不是逐请求 HTTP 抓包，也不包含训练后端专用 token IDs / logprobs。需要这些信息时再增加模型网关采集。原始运行记录和附件可能包含项目内容，全部保存在本地并由 Git 忽略；API key 只传给 runtime 的环境变量。

## 验证

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python tests/smoke_real_coserve.py --repo ..
```

第二条命令使用本地测试模型 HTTP 服务驱动真实 CoServe，不需要模型账号或 GPU；检查模型请求确实经过框架、工具执行后的结果进入下一次模型请求，以及原始轨迹落盘。它只验证集成链路，不评价真实模型的任务能力。

## 参考源码

- [Agent Lightning Calc-X](https://github.com/microsoft/agent-lightning/blob/main/examples/calc_x/calc_agent.py)：将模型端点传入 Agent，由框架执行任务。
- [rLLM math tool agent](https://github.com/rllm-org/rllm/blob/main/cookbooks/math_tool_agent/math_tool_agent.py)：以 Task 与 AgentConfig 驱动一次 rollout。
- [SWE-smith trajectory collection](https://github.com/SWE-bench/SWE-smith/blob/main/swesmith/train/traj_mgr/collect_trajs.py)：原始轨迹采集与训练格式转换分开。

本项目复用这些分工思路，不引入它们的训练、网关或调度系统。当前只实现 Python 调用与原始输出收集。
