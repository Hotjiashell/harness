# Construct 代码结构与使用说明

本文档说明 `construct/` 目录下当前 Harness 一阶段实现的代码结构、运行方式和输出内容。

## 1. 实现目标

根据 [requirement.md](/Users/tangjinkang/Documents/work/智能客服/harness/construct/requirement.md:1) 的要求，当前实现完成了以下流程：

1. 读取案例库和种子 `L1`
2. 对案例进行 `L1` 归类
3. 对未命中种子类别的案例做新类别发现
4. 对非软件类和软件类案例分别聚类
5. 生成新的 `L1` 或归入 `其他`
6. 递归构建 `L2/L3`
7. 导出最终知识树和调试中间结果

## 2. 目录结构

### 入口与编排

- `main.py`
  CLI 入口，支持通过 `--provider` 临时覆盖 LLM 提供方。
- `__main__.py`
  允许直接使用 `python -m construct` 启动。
- `pipeline.py`
  总编排入口，负责串联数据加载、`L1` 构建、递归建树、最终导出。

### 核心流程

- `l1.py`
  实现第一层分类流程：
  - 案例归入已有 `L1`
  - 未命中案例的新类别发现
  - 软件类 / 非软件类候选聚类
  - 新 `L1` 与 `其他` 节点生成

- `tree.py`
  实现 `L2/L3` 的递归构建：
  - 收集父节点案例
  - 在父节点语境下总结案例子问题
  - 聚类
  - 生成子节点
  - 递归向下展开，直到达到最大深度或样本不足

- `clustering.py`
  对接根目录的 `cluster.py`。
  如果外部聚类函数不可用或未实现，会自动使用本地兼容回退聚类，保证流程可跑通。

### LLM 与提示词

- `llm.py`
  LLM 抽象层，包含两种模式：
  - `heuristic`
    本地启发式回退模式，不依赖外部模型服务，便于开发和调试
  - `openai-compatible`
    对接 OpenAI 兼容接口的真实模型服务

- `prompts.py`
  统一维护分类、新类别发现、候选节点总结、子节点总结等 Prompt 构造逻辑。

### 数据模型与基础设施

- `models.py`
  定义案例、分类结果、聚类项、知识树节点等数据结构。
- `io_utils.py`
  负责 JSON 读取、写入和树结构导出。
- `exporter.py`
  导出最终知识树的目录结构版本，便于后续按节点消费。
- `utils.py`
  通用工具函数，例如目录创建、文本归一化、并发执行、软件名规整等。
- `__init__.py`
  包标记文件。

## 3. 配置位置

`construct/` 本身不放配置，统一使用根目录的 [config.py](/Users/tangjinkang/Documents/work/智能客服/harness/config.py:1)。

当前主要配置项如下：

- 路径配置
  - `case_path`：案例库路径
  - `seed_l1_path`：种子 `L1` 路径
  - `output_dir`：最终输出目录
  - `skills_dir`：目录树导出目录

- LLM 配置
  - `provider`
  - `base_url`
  - `api_key`
  - `model`
  - `concurrency`

- 聚类配置
  - `method`：`kmeans` 或 `hdbscan`
  - `n_clusters`
  - `min_cluster_size`
  - `embedding_*`

- 流程配置
  - `max_depth`
  - `new_l1_min_cases`
  - `min_cases_to_split`
  - `software_alias_min_match`
  - `stop_after_l1`
  - `resume_tree_path`
  - `console_output`
  - `log_timestamps`
  - `progress_bar_width`
  - `initial_root_filename`

## 4. 运行方式

### 默认运行

默认使用本地启发式模式：

```bash
python3 -m construct
```

适用场景：

- 先验证流程是否能跑通
- 观察中间结果
- 外部 LLM / Embedding 服务暂未接通

运行时控制台会输出：

- 当前阶段，例如 `Build L1`、`Build L2/L3`
- 当前正在展开的节点路径
- 批量 LLM 调用的 `tqdm` 进度条
- 聚类开始 / 完成和回退信息

### 指定真实模型服务

如果已经准备好 OpenAI 兼容接口，可以在 `config.py` 中设置：

```python
CONFIG.llm.provider = "openai-compatible"
CONFIG.llm.base_url = "你的接口地址"
CONFIG.llm.api_key = "你的密钥"
CONFIG.llm.model = "你的模型名"
```

或运行时临时覆盖：

```bash
python3 -m construct --provider openai-compatible
```

注意：

- `openai-compatible` 只覆盖 `provider`
- `base_url`、`api_key`、`model` 仍需在 `config.py` 中配置

### 只构建 L1 后停止

如果你只想先跑完 `L1` 构建，再检查中间结果，可以在 `config.py` 中设置：

```python
CONFIG.pipeline.stop_after_l1 = True
CONFIG.pipeline.resume_tree_path = None
```

此时流程会：

- 完成 `L1` 分类与新节点发现
- 生成 `intermediate/05_initial_root.json`
- 导出当前阶段的 `knowledge_tree.json` 和 `knowledge_tree_debug.json`
- 跳过后续 `L2/L3` 递归构建

### 从 L1 中间文件续跑后续阶段

如果你已经有上一轮产出的 `L1` 初始树，可以直接从该文件继续跑 `L2/L3`：

```python
CONFIG.pipeline.stop_after_l1 = False
CONFIG.pipeline.resume_tree_path = (
    CONFIG.paths.output_dir
    / CONFIG.pipeline.stage_dir_name
    / CONFIG.pipeline.initial_root_filename
)
```

也可以把 `resume_tree_path` 指向一个已有的 `knowledge_tree_debug.json`。

注意：

- 续跑文件必须包含 `case_ids`
- 因此不能使用面向消费的 `knowledge_tree.json`
- `stop_after_l1` 和 `resume_tree_path` 不能同时设置

## 5. 输入数据要求

### 案例库

默认读取：

```text
data/case/text.json
```

格式示例：

```json
{
  "KT00000001": {
    "case_name": "案例标题",
    "text": "案例正文"
  }
}
```

### 种子 L1

默认读取：

```text
output/seed_L1.json
```

每个节点格式：

```json
{
  "name": "类别名称",
  "trigger": "什么时候考虑该类别",
  "background": "背景知识"
}
```

## 6. 输出内容

默认输出到：

```text
output/harness
```

### 最终输出

- `knowledge_tree.json`
  面向消费的最终知识树
- `knowledge_tree_debug.json`
  包含 `depth`、`path`、`case_ids` 的调试版本
- `run_config.json`
  本次运行的配置快照
- `error_audit.json`
  全流程错误审计汇总，包含阶段分布、错误明细、是否使用回退
- `failed_cases.json`
  以案例维度聚合的失败事件清单

### 中间结果

位于：

```text
output/harness/intermediate
```

主要文件：

- `01_l1_classification.json`
  每个案例是否属于已有 `L1`，以及原因
- `02_new_category_discovery.json`
  未命中案例的软件名和描述抽取结果
- `03_candidate_clusters.json`
  候选聚类结果
- `04_candidate_nodes.json`
  候选聚类总结后的节点信息
- `05_initial_root.json`
  `L1` 构建完成后的初始树

递归建树阶段的中间结果位于：

```text
output/harness/intermediate/tree
```

每个父节点目录下通常包含：

- `01_case_summaries.json`
- `02_clusters.json`
- `03_children.json`

### 目录树导出

位于：

```text
output/_skills/harness
```

每个节点目录下包含：

- `node.json`
  当前节点信息
- `cases.json`
  当前节点关联案例

这个导出适合后续做节点级消费、检索或调试。

## 7. 单案例失败与错误审计

当前实现已经支持“单案例失败不拖垮整批”：

- `L1` 案例分类
- 未命中案例的新类别发现
- 递归建树时的父节点下案例总结

以上阶段如果某个案例的主 LLM 调用失败：

1. 不会中断整批任务
2. 会记录到 `error_audit.json` 和 `failed_cases.json`
3. 会自动回退到 `heuristic` 逻辑继续处理

如果 `heuristic` 回退也失败，则使用最小默认结果，保证流程尽量继续向下执行。

## 8. 关键实现说明

### 关于 `cluster.py`

根目录的 `cluster.py` 目前只定义了接口，没有实际实现。当前代码已经兼容这种情况：

- 如果 `cluster.py` 可正常返回聚类结果，则优先使用外部聚类
- 如果外部聚类不可用、抛错或返回格式不符，则自动回退到 `construct/clustering.py` 内的本地聚类逻辑

因此当前项目具备两种运行状态：

1. 调试态：无外部聚类服务，使用本地回退
2. 生产态：补齐 `cluster.py` 或接通真实向量聚类链路

### 关于 LLM 模式

`heuristic` 模式是为了保证当前仓库在没有外部依赖时也能跑出完整结构，但它的节点命名和摘要质量会弱于真实模型。

如果你要更接近需求中的最终效果，建议后续切到：

```text
provider = openai-compatible
```

### 关于“一个聚类可拆多个节点”

当前实现中，不同阶段的聚类总结策略不同：

- 软件名组小节点总结：仍然是一个软件名组总结成一个节点
- 非软件 `L1` 总结：模型可以把一个聚类拆成 `1~3` 个节点
- 软件功能大节点总结：模型可以把一个聚类拆成 `1~3` 个节点
- 递归 `L2/L3` 子节点总结：模型可以把一个聚类拆成 `1~3` 个节点

对于可拆分的阶段，模型需要同时返回每个节点对应的 `item_ids`，代码会按这些 `item_ids` 把同一聚类中的 item 重新分配到不同节点下。

## 9. 常见修改入口

如果要继续演进实现，通常从这些位置开始：

- 调整配置项：`config.py`
- 调整 `L1` 判定和新类别逻辑：`l1.py`
- 调整子节点递归展开逻辑：`tree.py`
- 替换或增强 LLM Prompt：`prompts.py`
- 切换真实模型调用：`llm.py`
- 接入真实 embedding 聚类：根目录 `cluster.py`

## 10. 当前限制

- 默认 `heuristic` 模式下，节点命名仍偏工程回退风格，不代表最终效果上限
- 当前 `cluster.py` 仍未实现真实向量聚类
- CLI 只提供了 `--provider` 覆盖，其他参数仍通过 `config.py` 配置
- `output/_skills/harness` 目前导出的是节点目录结构，不是完整可执行 skill 包

## 11. 推荐使用顺序

1. 先用 `python3 -m construct` 验证流程和输出结构
2. 检查 `output/harness/intermediate` 中的各阶段结果
3. 调整 `config.py` 中的阈值和聚类参数
4. 接入真实 `openai-compatible` LLM
5. 最后补齐根目录 `cluster.py` 的真实 embedding 聚类实现
