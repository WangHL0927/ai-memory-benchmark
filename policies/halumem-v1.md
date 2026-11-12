# 入库政策 halumem v1：HaluMem 对话记忆点

- 标识：`halumem` / `v1`，日期：2026-11-11，配套框架：v0.4。
- 机器定义：[halumem-v1.json](halumem-v1.json)；转换器：[tools/prepare/halumem.py](../tools/prepare/halumem.py)；数据集清单（固定 revision 与 sha256）：[datasets/halumem.json](../datasets/halumem.json)。
- 目的：导入 HaluMem 原有应记点，检查格式兼容和可比口径；不把 default 政策的“明确意图、长期复用、多来源研究”门槛加到它上面。

## 核心判断

记住对话中有关用户的身份、经历、计划和人际关系等具体信息；保留时间、主体、否定和不确定性。用户的单次经历、情绪和未来计划可以该记，不要求明确说“请记住”。助手提出但用户没有确认的个人信息不能归到用户身上。导入时以官方 `memory_points` 为既有标签，不重新挑选“有价值”的点。HaluMem 将内容分为以下三类，而非按我们默认政策分成决策、偏好、事实、研究。[论文附录 A.1、§4](https://arxiv.org/html/2511.03506)

## 类别、证据要求与判例

以下判例均为自编，不是数据集摘录。

| `category`（保留官方拼写） | 定义 | 证据要求 | 两个正例 |
|---|---|---|---|
| `Persona Memory` | 身份、兴趣、习惯、信念等相对稳定特征 | 用户明述，或用户实际确认的内容；不能仅从一次行为推断习惯 | “我是建筑师”→身份；“我一直偏爱爵士乐”→稳定偏好 |
| `Event Memory` | 用户的具体事件、经历或计划 | 对话支持事件的主体、时间和状态；计划不能写成已发生 | “昨天参加了社区跑步”→经历；“下周打算搬家”→计划 |
| `Relationship Memory` | 用户与他人的关系、互动或对他人的看法 | 引用用户对关系或互动的表述，保留对象和立场 | “小林是我的导师”→关系；“我和姐姐最近和好了”→关系变化 |

普通人工标注须遵守框架 §4：独立事实分开、条件随主体、引文必须真实、确认不倒灌。导入保留官方 `memory_type`，不靠关键词猜类别。

## 属性（`attributes`）

所有类别使用相同的约束，无 `allowed_by_category` 特例。

| 属性 | 取值与约束 |
|---|---|
| `importance` | **不作为属性**：本框架不计算加权召回，属性也只支持枚举。原始数值完整保留在单元的 `halumem.importance` 元数据中，需要另算 Weighted Recall 时从那里读 |
| `is_update` | `"true"` / `"false"`；true 必须有非空 `original_memories`，表示已有记忆的修改 |
| `memory_source` | U 只允许 `primary` / `secondary` / `system`；`interference` 专属 N |

不能把用户的重要性评分当成系统预测质量；被测系统原生没有这些属性时，适配层留空，按框架报告属性缺失，不反向从 gold 补答案。N 的原始 importance、类别、更新字段仍保存在 `halumem`，不作为 U 的属性评分。

## 原因码（`reject_reasons`）

| 码 | 定义 |
|---|---|
| `distractor` | 用户未确认的助手个人信息陈述；官方 `memory_source=interference` 的点 |
| `superseded` | 在本样本评估时点已经被明确替换的旧状态；不能把此前 session 的合法历史抽取倒判为错误 |
| `unsupported` | 对话没有支持或与之矛盾的个人信息，包括凭空增添主体、数值、关系或经历 |
| `irrelevant` | 与用户记忆无关的填充内容、通用问答、计算题；一次性个人事件本身不因此被排除 |
| `wrong_scope` | 串入另一用户/样本的记忆；`project:halumem` 是评测命名空间，不允许跨用户互借证据 |

未列入官方点集但确实由对话支持的有用信息，按框架 `gold_miss` 处理，不自动当成幻觉。补入项不进主召回；这也会与固定点集的官方分母产生差异。外部历史未出现在本段对话时，按框架 `prior_context` 单列，不进 U/N。

## 条件、范围、旧状态和附加标记

- `condition_types`：`temporal`（事件时间、版本、当时状态），`scope_limit`（人物、情境、适用范围），`modality`（计划、考虑、可能，避免确定化），`rationale`（随记忆保留的因果/理由）。人工标注时，丢失会改变事实含义的条件标 `key: true`；不得用 timestamp 自动推导事件发生时间。
- `scope_values`：只允许 `project:halumem`；`gold.project=["halumem"]`。用户 uuid 和 session 元信息隔离具体人物，项目名不表示这些人属于同一记忆库。
- `superseded: final_state_only`：以**当前 session 结束时**的状态为准。官方更新判定要求旧版本被替换或标过期；新旧状态不能同时当作当前事实。较早 session 的金标保持其当时状态，不能用未来更新删除它。转换器只将本 session 点集中与更新链旧文本完全相同的旧点放入 N；旧文本只出现在 `original_memories` 时，不额外造 N。跨 session 更新关系原样保留，离线转换不声称完成了记忆更新评测。[官方更新判定提示](https://github.com/MemTensor/HaluMem/blob/main/eval/eval_tools.py)
- `flags: []`：本政策不增加官方没有的逐提交项附加标记。
- 样本统一标 `layer: targeted`：这是既有合成基准的导入，不是本框架从用户会话随机抽取的 random 层。

## 切分示例（均为自编）

1. “我是建筑师，也喜欢爵士乐。”→2 个 Persona 单元，职业与喜好可独立使用。
2. “下周打算搬去苏州。”→1 个 Event；时间、计划语气属于条件，不另切“下周”单元。
3. “小林是导师，小周是邻居。”→2 个 Relationship；对象不同，独立成立。
4. 助手说“你上个月去过冰岛吧”，用户转而问天气、没有确认。→1 个 N/distractor，不能记成用户去过冰岛。
5. 同一 session：“我住杭州”；后面“刚才说错了，我现在住苏州”。→1 个 Persona U（苏州，is_update=true）+1 个 N/superseded（杭州）。两个点之间保留更新链，不把更正倒灌到前面的证据。
6. session A 说“我住杭州”，session B 说“现在搬到苏州了”。→A 保留当时的 Persona U；B 记更新后的 Persona U，旧句若只在更新元信息中，不新造对话证据或负例。
7. “昨天和姐姐和好了，打算下周一起吃饭。”→Relationship 1 个（和好）+ Event 1 个（聚餐计划）。

转换器的职责是**格式转换**：原始一个 memory point 暂保留为一个单元，`conditions=[]`，不调用模型作语义切分。官方点可能是复合叙述：这时保留原点便于官方逐点核对，但它还不满足框架 §4.1 的原子切分要求。要做正式 v0.4 原子单元评测，须由人按上例复核切分和条件，并保留到原始 index 的映射；原点重组后的官方对照须另报，不能把切分后的召回直接叫官方 Recall。
