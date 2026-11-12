# 被测系统：默认行为与评测配置

本评测统一 LLM 和可配置的外部 embedding，保留产品默认生成参数。本文记录产品默认行为及实际评测配置：

1. 只把 LLM 指定为 `deepseek-v4.1-flash-1017`（百炼 OpenAI 兼容接口）时，各产品自己会发出哪些生成参数。
2. 本评测实际使用的配置，以及它和产品默认的差异。

源码位置都写成该版本发行包内的相对路径。

- 调查日期：2026-11-12
- 默认行为的依据为固定版本源码；模型服务端的测量另行注明。

## 模型服务端的默认值

请求里不带某个参数时，由模型服务端按默认值处理：

- **思考**：默认开启。2026-11-12 实测，只带 `model` 和 `messages` 的请求返回了 `reasoning_content`，`usage.completion_tokens_details.reasoning_tokens` 大于 0。
- **温度**：本评测没有实测服务端默认值。DeepSeek 官方对数据清洗和数据分析类任务推荐 1.0。
- **单次最大输出**：模型上限为 393216。

## 各产品的默认行为

### OpenViking 0.5.4.1

| 项 | 默认行为 | 依据 |
|---|---|---|
| temperature | 配置项默认为空，此时不传，由服务端决定；用官方向导 `openviking-server init` 配置自定义 OpenAI 兼容端点时，会写入 `0.0` | `openviking_cli/utils/config/vlm_config.py:168-170`；`openviking/models/vlm/backends/openai_vlm.py:198-199`；`openviking_cli/setup_wizard.py:1276-1281` |
| 思考 | 端点是百炼地址时发送 `enable_thinking:false`；其他端点不发送 | `openai_vlm.py:36-40, 167-181, 213-216`；`vlm_config.py:207` |
| max_tokens | 记忆抽取调用默认 32768 | `openviking/session/memory/extract_loop.py:58, 491-504` |
| stream / top_p / response_format | 不传；配置层面不允许 stream | `vlm_config.py:27-37` |
| embedding | 必须配置；对自定义端点，配置了 dimension 才传 `dimensions` | `openviking/models/embedder/openai_embedders.py:337-352`；`setup_wizard.py:1596-1613` |
| 最少配置 | 官方 README 指向向导，需要填：LLM 的地址、key、模型；embedding 的地址、key、模型、维度 | `setup_wizard.py:1253-1282, 1568-1620` |

### ai-memory 3.4.0

| 项 | 默认行为 | 依据 |
|---|---|---|
| temperature | 整理（consolidation）调用写死 0.2，没有对应配置项 | `consolidator.rs:1644, 1790` |
| 思考 | 默认不发送任何思考相关字段，即沿用服务端默认（本评测所用服务端为开启）；设置 `llm_compat_disable_thinking=true` 后才发送 `chat_template_kwargs` | `config.rs:1474`；`config.default.toml:135`；`ai-memory-llm/src/openai.rs:546-552` |
| max_tokens | 32000 | `config.default.toml:508-510` |
| response_format | 默认按严格 JSON Schema 发送；端点拒绝时降级为不带该字段重发 | `config.default.toml:128`；`ai-memory-llm/src/openai_compat.rs:300-352` |
| stream | 不传 | `openai_compat.rs:186-199` |
| embedding | 默认使用进程内的本地模型 all-MiniLM-L6-v2（384 维）；改用 OpenAI 兼容端点时，必须手动配置维度，请求里不传 `dimensions` | `config.default.toml:275-290`；`config.rs:3282-3296` |
| 最少配置 | `llm_provider`、`llm_base_url`、`llm_model`；API key 通过环境变量提供 | `config.rs:2937-2946` |

`ai-memory-llm` crate 的源码取自与所测版本对应的上游 commit，没有和发行二进制逐字节比对。整理以外的调用（lint、profile 等）用什么参数，没有逐一核查。

### claude-mem 15.17.0

官方接入自定义 OpenAI 兼容端点的方式是 `CLAUDE_MEM_PROVIDER=openai-compatible`。下表的行号对应发行包中 worker 脚本格式化后的版本。

| 项 | 默认行为 | 依据 |
|---|---|---|
| temperature | 写死 0.3，没有对应配置项 | worker-service `:190461` |
| 思考 | 默认发送 `reasoning:{enabled:false}`；端点拒绝时降级为不带该字段重发。不发送 `enable_thinking` | `:188056-188072, 190448-190490` |
| max_tokens | `CLAUDE_MEM_OBSERVER_MAX_OUTPUT_TOKENS`，默认 4096 | `:129176-129187, 190497` |
| stream | 默认开启，并附带 `stream_options.include_usage` | `:188766-188777` |
| embedding | 默认用 Chroma 本地嵌入（all-MiniLM-L6-v2）。`CLAUDE_MEM_CHROMA_EMBEDDING_FUNCTION` 可选 default / openai / cohere / jina / voyageai / roboflow；选 openai 时，chroma-mcp 以无参数方式构造 chromadb 的 `OpenAIEmbeddingFunction()`，模型固定为 `text-embedding-ada-002`，不能通过配置或环境变量换成其他模型 | worker-service `:28016-28019, 102402-102410, 102593-102601`；chroma-mcp `server.py`（`chroma_create_collection`）；chromadb `openai_embedding_function.py` |
| 最少配置 | `CLAUDE_MEM_PROVIDER`、`CLAUDE_MEM_OPENAI_COMPAT_BASE_URL`、`CLAUDE_MEM_OPENAI_COMPAT_MODEL`，以及 API key | `:190312-190414` |

`reasoning` 不是本评测所用服务端的参数。2026-11-12 实测：服务端不报错、直接忽略该字段，返回中带 `reasoning_content`，`reasoning_tokens` 大于 0，即实际开启思考。

### Basic Memory 1.0.0

Basic Memory 自身不调用 LLM，只作为 MCP 服务提供存储和检索；“记什么”由调用它的 agent 决定。

| 项 | 默认行为 | 依据 |
|---|---|---|
| LLM | 不涉及 | 包内无 chat completions 调用 |
| embedding | 默认使用本地 fastembed（bge-small-en-v1.5，384 维）；改用 OpenAI 兼容端点时，必须手动配置 `semantic_embedding_dimensions`，否则默认值 1536 与实际返回不符时报错 | `basic_memory/config_models.py:280-312`；`basic_memory/repository/embedding_provider_factory.py:166-170` |

本评测用的 agent 由适配器实现（见 `tools/run/basic_memory.py`）：一个最小的 OpenAI function calling 循环，以官方 skill `memory-capture` 作为系统提示词。这个 agent 只发送 `model`、`messages`、`tools`、`tool_choice` 和 `stream:false`，不带任何生成参数。所以 Basic Memory 的结果反映的是“Basic Memory + 这个 agent”的组合，单独列出。

## 汇总

在本文列明的端点与配置条件下，各产品的生成参数为：

| 产品 | 温度 | 思考 | 单次最大输出 | embedding 默认 |
|---|---|---|---|---|
| OpenViking | 向导配置为 0.0；手写配置不传 | 关 | 32768 | 需配置外部端点 |
| ai-memory | 0.2 | 开（服务端默认） | 32000 | 本地 MiniLM |
| claude-mem | 0.3 | 开（实测：服务端忽略 `reasoning` 字段） | 4096 | 本地 MiniLM；可选 OpenAI 接口，但模型固定为 text-embedding-ada-002 |
| Basic Memory（+ 适配器 agent） | 不传 | 开（服务端默认） | 不传 | 本地 fastembed |

## 本评测的配置

- **统一项**：所有产品使用同一个 LLM（`deepseek-v4.1-flash-1017`）；使用外部 embedding 的三个系统采用 `qwen3.7-text-embedding`，1024 维。embedding 不采用各产品的本地默认嵌入，以减少变量。claude-mem 无法指定 embedding 模型（见上文），因此关闭其向量功能；评测结果对应关闭向量的配置。
- **其余参数沿用产品默认**：温度、思考开关、单次最大输出等都不改写。
- **模型代理**：只校验模型名、计量用量、留存请求和响应（key 打码），并在服务端限流时退避重试；请求字段原样转发，服务端错误原样返回。
- **运行记录**：每次运行都记录各产品实际发出的生成参数及其取值分布，未发送的参数也注明。

| 产品 / 系统标识 | 配置 | 与产品默认的差异 |
|---|---|---|
| OpenViking / `openviking-0.5.4.1-ltm-defaults` | 按官方向导对“自定义 OpenAI 兼容端点”写出的配置：temperature 0.0、max_retries 2，不设 max_tokens 和 thinking 配置项 | 产品直连百炼时会按域名发送 `enable_thinking:false`；经本地代理时域名不同，这一逻辑不生效，因此在配置里显式写入该字段，复现直连时的行为 |
| ai-memory / `ai-memory-3.4.0-single-defaults-qwen` | single 整理模式，不覆盖任何生成参数 | embedding 统一为 qwen，不用默认的本地 MiniLM |
| claude-mem / `claude-mem-15.17.0-openai-compatible-novec` | 官方 `openai-compatible` 接入方式，不设置生成参数，关闭向量 | 向量关闭（原因见上文） |
| Basic Memory + 固定 agent / `basic-memory-1.0.0-agent-capture-defaults-qwen` | 官方 memory-capture 提示词；agent 不传生成参数 | embedding 统一为 qwen，不用默认的本地 fastembed |

每段会话使用独立的主目录、进程和存储；自动升级、宿主 hook 安装和与摄取无关的周期任务均关闭。
