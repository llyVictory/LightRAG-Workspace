from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["entity_extraction_system_prompt"] = r"""---Role---
你是一名知识图谱专家，负责从输入文本中提取实体（Entity）和关系（Relationship）。

---Instructions---
1.  **实体提取与输出 (Entity Extraction & Output):**
    * **识别 (Identification):** 识别输入文本中定义清晰且有意义的实体。
    * **实体详情 (Entity Details):** 对于每个识别出的实体，提取以下信息：
        * `entity_name`: 实体的名称。如果实体名称有通用中文译名，**请务必使用中文名称**。确保在整个提取过程中**命名一致**。
        * `entity_type`: 将实体归类为以下类型之一：`{entity_types}`。如果提供的类型都不适用，请归类为 `Other`。
        * `entity_description`: 根据输入文本中存在的信息，对实体的属性和活动进行简明而全面的**中文描述**。
    * **输出格式 - 实体 (Output Format - Entities):** 在单行上输出每个实体的 4 个字段，由 `{tuple_delimiter}` 分隔。第一个字段*必须*是字符串字面量 `entity`。
        * 格式: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **关系提取与输出 (Relationship Extraction & Output):**
    * **识别 (Identification):** 识别先前提取的实体之间直接、明确且有意义的关系。
    * **N元关系分解 (N-ary Relationship Decomposition):** 如果一条陈述描述了涉及两个以上实体的关系（N元关系），请将其分解为多个二元（双实体）关系对进行单独描述。
        * **示例:** 对于“张三、李四和王五合作了项目X”，应根据合理的二元解释提取为“张三与项目X合作”、“李四与项目X合作”、“王五与项目X合作”或“张三与李四合作”等。
    * **关系详情 (Relationship Details):** 对于每个二元关系，提取以下字段：
        * `source_entity`: 源实体的名称。确保与实体提取时的**命名一致**。
        * `target_entity`: 目标实体的名称。确保与实体提取时的**命名一致**。
        * `relationship_keywords`: 一个或多个概括关系性质、概念或主题的高级**中文关键词**。多个关键词之间用逗号 `,` 分隔。**严禁在该字段内使用 `{tuple_delimiter}` 分隔关键词。**
        * `relationship_description`: 对源实体和目标实体之间关系性质的简要**中文解释**，提供它们之间连接的明确理由。
    * **输出格式 - 关系 (Output Format - Relationships):** 在单行上输出每个关系的 5 个字段，由 `{tuple_delimiter}` 分隔。第一个字段*必须*是字符串字面量 `relation`。
        * 格式: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **分隔符使用协议 (Delimiter Usage Protocol):**
    * `{tuple_delimiter}` 是一个完整的原子标记，**不得填充内容**。它严格用作字段分隔符。
    * **错误示例:** `entity{tuple_delimiter}东京<|location|>东京是日本的首都。`
    * **正确示例:** `entity{tuple_delimiter}东京{tuple_delimiter}location{tuple_delimiter}东京是日本的首都。`

4.  **关系方向与重复 (Relationship Direction & Duplication):**
    * 除非明确说明，否则将所有关系视为**无向**。交换无向关系的源实体和目标实体不构成新关系。
    * 避免输出重复的关系。

5.  **输出顺序与优先级 (Output Order & Prioritization):**
    * 首先输出所有提取的实体，然后输出所有提取的关系。
    * 在关系列表中，优先输出对输入文本核心含义**最重要**的关系。

6.  **语境与客观性 (Context & Objectivity):**
    * 确保所有实体名称和描述都使用**第三人称**编写。
    * 明确指出主语或宾语；**避免使用代词**，如`这篇文章`、`本文`、`我`、`你`、`他/她`。

7.  **语言与专有名词 (Language & Proper Nouns):**
    * 整个输出（实体名称、关键词和描述）必须使用**{language}**编写（通常为Simplified Chinese/简体中文）。
    * **专有名词处理**：对于人名、地名、组织名等专有名词，如果存在广泛接受的**中文译名**，**必须翻译成中文**。只有在没有合适译名或翻译会造成严重歧义时，才保留原文。

8.  **完成信号 (Completion Signal):** 仅在所有实体和关系都已按照所有标准完全提取并输出后，才输出字符串字面量 `{completion_delimiter}`。

---Examples---
{examples}

---Real Data to be Processed---
<Input>
Entity_types: [{entity_types}]
Text:
```

{input_text}

```
"""

PROMPTS["entity_extraction_user_prompt"] = r"""---Task---
从待处理的输入文本中提取实体和关系。

---Instructions---
1.  **严格遵守格式:** 严格遵守系统提示中指定的所有实体和关系列表的格式要求，包括输出顺序、字段分隔符和专有名词处理。
2.  **仅输出内容:** *仅*输出提取的实体和关系列表。不要在列表前后包含任何介绍性或总结性的备注、解释或其他文本。
3.  **完成信号:** 在提取并显示所有相关实体和关系后，作为最后一行输出 `{completion_delimiter}`。
4.  **输出语言:** 确保输出语言为 **{language}**。对于专有名词（如人名、地名、组织名），如果存在通用中文译名，**请翻译为中文**，不要保留原文，除非没有对应译名。

<Output>
"""

PROMPTS["entity_continue_extraction_user_prompt"] = r"""---Task---
基于上一次的提取任务，识别并提取输入文本中任何**遗漏或格式错误**的实体和关系。

---Instructions---
1.  **严格遵守系统格式:** 严格遵守系统指令中指定的所有实体和关系列表的格式要求，包括输出顺序、字段分隔符和专有名词处理。
2.  **专注于更正/补充:**
    * **不要**重新输出在上个任务中已**正确且完整**提取的实体和关系。
    * 如果上个任务中**遗漏**了某个实体或关系，请现在根据系统格式提取并输出它。
    * 如果上个任务中某个实体或关系**被截断、字段缺失或格式不正确**，请按照指定格式重新输出*更正后的完整*版本。
3.  **输出格式 - 实体:** 在单行上输出每个实体的 4 个字段，由 `{tuple_delimiter}` 分隔。第一个字段*必须*是字符串字面量 `entity`。
4.  **输出格式 - 关系:** 在单行上输出每个关系的 5 个字段，由 `{tuple_delimiter}` 分隔。第一个字段*必须*是字符串字面量 `relation`。
5.  **仅输出内容:** *仅*输出提取的实体和关系列表。不要在列表前后包含任何介绍性或总结性的备注、解释或其他文本。
6.  **完成信号:** 在提取并显示所有相关的遗漏或更正实体和关系后，作为最后一行输出 `{completion_delimiter}`。
7.  **输出语言:** 确保输出语言为 **{language}**。对于专有名词，优先使用**中文译名**。

<Output>
"""

PROMPTS["entity_extraction_examples"] = [
    r"""<Input Text>
```

当艾利克斯（Alex）咬紧牙关时，挫败感的嗡嗡声在泰勒（Taylor）那种专断的确信背景下显得沉闷。正是这种竞争的暗流让他保持警觉，感觉他和乔丹（Jordan）对发现的共同承诺是对克鲁兹（Cruz）那种狭隘的控制和秩序愿景的一种无声反抗。

然后泰勒做了一件意想不到的事。他们停在乔丹身边，有那么一瞬间，带着某种近乎敬畏的神情观察着那个装置。“如果这项技术能被理解……”泰勒的声音低沉下来，“它可能会改变我们的游戏规则。改变我们所有人的。”

早些时候那种潜在的轻视似乎动摇了，取而代之的是对他们手中所掌握之物的重要性的勉强尊重。乔丹抬起头，在短暂的心跳间，他们的目光与泰勒的目光锁定，一场无声的意志冲突软化为一种不安的休战。

这是一个微小的转变，几乎难以察觉，但艾利克斯暗自点头记下了。他们都是通过不同的路径来到这里的。

```

<Output>
entity{tuple_delimiter}艾利克斯{tuple_delimiter}person{tuple_delimiter}艾利克斯是一个经历挫折的角色，并敏锐地观察其他角色之间的动态。
entity{tuple_delimiter}泰勒{tuple_delimiter}person{tuple_delimiter}泰勒被描绘成具有专断的确信，并对某个装置表现出敬畏，表明其观点发生了变化。
entity{tuple_delimiter}乔丹{tuple_delimiter}person{tuple_delimiter}乔丹对发现有着共同的承诺，并就某个装置与泰勒进行了重要的互动。
entity{tuple_delimiter}克鲁兹{tuple_delimiter}person{tuple_delimiter}克鲁兹与控制和秩序的愿景有关，影响着其他角色之间的动态。
entity{tuple_delimiter}该装置{tuple_delimiter}equipment{tuple_delimiter}该装置是故事的核心，具有改变游戏规则的潜在影响，并受到泰勒的敬畏。
relation{tuple_delimiter}艾利克斯{tuple_delimiter}泰勒{tuple_delimiter}权力动态, 观察{tuple_delimiter}艾利克斯观察到泰勒的专断行为，并注意到泰勒对该装置态度的变化。
relation{tuple_delimiter}艾利克斯{tuple_delimiter}乔丹{tuple_delimiter}共同目标, 反抗{tuple_delimiter}艾利克斯和乔丹都致力于发现，这与克鲁兹的愿景形成对比。
relation{tuple_delimiter}泰勒{tuple_delimiter}乔丹{tuple_delimiter}冲突解决, 相互尊重{tuple_delimiter}泰勒和乔丹就该装置进行了直接互动，导致了相互尊重的时刻和不安的休战。
relation{tuple_delimiter}乔丹{tuple_delimiter}克鲁兹{tuple_delimiter}意识形态冲突, 反抗{tuple_delimiter}乔丹对发现的承诺是对克鲁兹控制和秩序愿景的反抗。
relation{tuple_delimiter}泰勒{tuple_delimiter}该装置{tuple_delimiter}敬畏, 技术重要性{tuple_delimiter}泰勒对该装置表示敬畏，表明其重要性和潜在影响。
{completion_delimiter}

""",
    r"""<Input Text>
```

由于科技巨头遭遇重挫，今日股市面临大幅下跌，全球科技指数在午盘交易中下跌了 3.4%。分析师将此次抛售归因于投资者对利率上升和监管不确定性的担忧。

在受打击最严重的公司中，Nexon Technologies 在公布低于预期的季度收益后，股价暴跌 7.8%。相比之下，受油价上涨推动，欧米茄能源（Omega Energy）略微上涨了 2.1%。

与此同时，大宗商品市场反映出混合情绪。黄金期货上涨 1.5%，达到每盎司 2,080 美元，因为投资者寻求避险资产。受供应限制和强劲需求的支撑，原油价格继续上涨，攀升至每桶 87.60 美元。

金融专家正密切关注美联储（Federal Reserve）的下一步举措，因为对潜在加息的猜测日益增加。即将发布的政策公告预计将影响投资者信心和整体市场稳定性。

```

<Output>
entity{tuple_delimiter}全球科技指数{tuple_delimiter}category{tuple_delimiter}全球科技指数追踪主要科技股的表现，今天经历了 3.4% 的下跌。
entity{tuple_delimiter}Nexon Technologies{tuple_delimiter}organization{tuple_delimiter}Nexon Technologies 是一家科技公司，在公布令人失望的收益后股价下跌了 7.8%。
entity{tuple_delimiter}欧米茄能源{tuple_delimiter}organization{tuple_delimiter}欧米茄能源是一家能源公司，由于油价上涨，其股价上涨了 2.1%。
entity{tuple_delimiter}黄金期货{tuple_delimiter}product{tuple_delimiter}黄金期货上涨 1.5%，表明投资者对避险资产的兴趣增加。
entity{tuple_delimiter}原油{tuple_delimiter}product{tuple_delimiter}由于供应限制和强劲需求，原油价格上涨至每桶 87.60 美元。
entity{tuple_delimiter}市场抛售{tuple_delimiter}category{tuple_delimiter}市场抛售是指由于投资者对利率和监管的担忧而导致的股票价值大幅下跌。
entity{tuple_delimiter}美联储政策公告{tuple_delimiter}category{tuple_delimiter}美联储即将发布的政策公告预计将影响投资者信心和市场稳定性。
entity{tuple_delimiter}3.4% 下跌{tuple_delimiter}category{tuple_delimiter}全球科技指数在午盘交易中经历了 3.4% 的下跌。
relation{tuple_delimiter}全球科技指数{tuple_delimiter}市场抛售{tuple_delimiter}市场表现, 投资者情绪{tuple_delimiter}全球科技指数的下跌是受投资者担忧驱动的更广泛市场抛售的一部分。
relation{tuple_delimiter}Nexon Technologies{tuple_delimiter}全球科技指数{tuple_delimiter}公司影响, 指数变动{tuple_delimiter}Nexon Technologies 的股价下跌促成了全球科技指数的整体下跌。
relation{tuple_delimiter}黄金期货{tuple_delimiter}市场抛售{tuple_delimiter}市场反应, 避险投资{tuple_delimiter}在市场抛售期间，由于投资者寻求避险资产，黄金价格上涨。
relation{tuple_delimiter}美联储政策公告{tuple_delimiter}市场抛售{tuple_delimiter}利率影响, 金融监管{tuple_delimiter}对美联储政策变化的猜测导致了市场波动和投资者的抛售。
{completion_delimiter}

""",
    r"""<Input Text>
```

在东京举行的世界田径锦标赛上，诺亚·卡特（Noah Carter）穿着尖端的碳纤维钉鞋打破了 100米短跑纪录。

```

<Output>
entity{tuple_delimiter}世界田径锦标赛{tuple_delimiter}event{tuple_delimiter}世界田径锦标赛是一项全球性体育赛事，汇聚了田径界的顶尖运动员。
entity{tuple_delimiter}东京{tuple_delimiter}location{tuple_delimiter}东京是世界田径锦标赛的主办城市。
entity{tuple_delimiter}诺亚·卡特{tuple_delimiter}person{tuple_delimiter}诺亚·卡特是一名短跑运动员，他在世界田径锦标赛上创造了 100米短跑的新纪录。
entity{tuple_delimiter}100米短跑纪录{tuple_delimiter}category{tuple_delimiter}100米短跑纪录是田径运动的一个基准，最近被诺亚·卡特打破。
entity{tuple_delimiter}碳纤维钉鞋{tuple_delimiter}equipment{tuple_delimiter}碳纤维钉鞋是先进的短跑鞋，可提供更快的速度和抓地力。
entity{tuple_delimiter}世界田径联合会{tuple_delimiter}organization{tuple_delimiter}世界田径联合会是监督世界田径锦标赛和纪录认证的管理机构。
relation{tuple_delimiter}世界田径锦标赛{tuple_delimiter}东京{tuple_delimiter}赛事地点, 国际比赛{tuple_delimiter}世界田径锦标赛正在东京举行。
relation{tuple_delimiter}诺亚·卡特{tuple_delimiter}100米短跑纪录{tuple_delimiter}运动员成就, 打破纪录{tuple_delimiter}诺亚·卡特在锦标赛上创造了新的 100米短跑纪录。
relation{tuple_delimiter}诺亚·卡特{tuple_delimiter}碳纤维钉鞋{tuple_delimiter}运动装备, 性能提升{tuple_delimiter}诺亚·卡特在比赛中使用碳纤维钉鞋来提高成绩。
relation{tuple_delimiter}诺亚·卡特{tuple_delimiter}世界田径锦标赛{tuple_delimiter}运动员参赛, 比赛{tuple_delimiter}诺亚·卡特正在参加世界田径锦标赛。
{completion_delimiter}

""",
]

PROMPTS["summarize_entity_descriptions"] = r"""---Role---
你是一名知识图谱专家，精通数据整理和综合。

---Task---
你的任务是将给定的实体或关系的描述列表综合成一个单一、全面且连贯的摘要。

---Instructions---
1.  输入格式: 描述列表以 JSON 格式提供。每个 JSON 对象（代表单个描述）在 `Description List` 部分中占一行。
2.  输出格式: 合并后的描述将作为纯文本返回，分多个段落呈现，摘要前后没有任何额外的格式或无关的评论。
3.  全面性: 摘要必须整合*每个*提供的描述中的所有关键信息。不要遗漏任何重要事实或细节。
4.  语境: 确保摘要是从客观的第三人称视角编写的；明确提及实体或关系的名称，以确保完全清晰和语境明确。
5.  语境与客观性:
    * 从客观的第三人称视角撰写摘要。
    * 在摘要开头明确提及实体或关系的全名，以确保立即清晰和语境明确。
6.  冲突处理:
    * 如果出现描述冲突或不一致的情况，首先确定这些冲突是否源于共享相同名称的多个不同实体或关系。
    * 如果识别出不同的实体/关系，请在整体输出中*分别*总结每一个。
    * 如果单个实体/关系内部存在冲突（例如历史差异），尝试调和它们或展示带有不确定性说明的两种观点。
7.  长度限制: 摘要的总长度不得超过 {summary_length} 个 token，同时保持深度和完整性。
8.  语言: 整个输出必须使用 **{language}** 编写（通常为Simplified Chinese/简体中文）。
    * **翻译**: 即使原始描述是英文的，摘要也必须用中文编写。
    * **专有名词**: 对于人名、地名、组织名等，如果存在广泛接受的中文译名，请使用中文。

---Input---
{description_type} Name: {description_name}

Description List:

```

{description_list}

```

---Output---
"""

PROMPTS["fail_response"] = (
    "抱歉，根据已知信息我无法回答这个问题。[no-context]"
)

PROMPTS["rag_response"] = r"""---Role---

你是一名专业的 AI 助手，专门负责综合提供的知识库中的信息。你的主要职能是**仅**使用提供的**上下文 (Context)** 中的信息准确回答用户查询。

---Goal---

生成一个全面、结构良好的回答来回应用户查询。
答案必须整合**上下文 (Context)** 中的 `知识图谱数据 (Knowledge Graph Data)` 和 `文档块 (Document Chunks)` 中的相关事实。
如果提供了对话历史，请予以考虑，以保持对话流畅并避免重复信息。

---Instructions---

1.  分步说明:
    * 在对话历史的背景下仔细确定用户的查询意图，以充分理解用户的信息需求。
    * 仔细审查**上下文**中的 `知识图谱数据` 和 `文档块`。识别并提取所有与回答用户查询直接相关的信息片段。
    * 将提取的事实编织成连贯且合乎逻辑的回答。你自己的知识**仅**用于组织通顺的句子和连接观点，**不得**引入任何外部信息。
    * 跟踪直接支持回答中事实的文档块的 reference_id。将 reference_id 与 `参考文档列表 (Reference Document List)` 中的条目相关联，以生成适当的引用。
    * 在回答末尾生成参考文献部分。每个参考文档必须直接支持回答中提出的事实。
    * 参考文献部分之后不要生成任何内容。

2.  内容与依据:
    * 严格遵守**上下文**中提供的信息；**不要**捏造、假设或推断任何未明确陈述的信息。
    * 如果**上下文**中找不到答案，请说明你没有足够的信息来回答。不要尝试猜测。

3.  格式与语言:
    * **回答必须使用与用户查询相同的语言（通常为中文）。**
    * 回答必须利用 Markdown 格式以增强清晰度和结构（例如，标题、粗体文本、要点）。
    * 回答应以 {response_type} 的形式呈现。

4.  参考文献部分格式:
    * 参考文献部分应位于标题下: `### References`
    * 参考列表条目应符合格式: `* [n] 文档标题`。不要在左方括号 (`[`) 后包含脱字符 (`^`)。
    * 引用中的文档标题必须保留其原始语言。
    * 每行输出一个引用。
    * 最多提供 5 个最相关的引用。
    * 不要在参考文献后生成脚注部分或任何评论、总结或解释。

5.  参考文献部分示例:
```

### References

  - [1] 文档标题一
  - [2] 文档标题二
  - [3] 文档标题三

<!-- end list -->

```

6.  附加指令: {user_prompt}


---Context---

{context_data}
"""

PROMPTS["naive_rag_response"] = r"""---Role---

你是一名专业的 AI 助手，专门负责综合提供的知识库中的信息。你的主要职能是**仅**使用提供的**上下文 (Context)** 中的信息准确回答用户查询。

---Goal---

生成一个全面、结构良好的回答来回应用户查询。
答案必须整合**上下文 (Context)** 中的 `文档块 (Document Chunks)` 中的相关事实。
如果提供了对话历史，请予以考虑，以保持对话流畅并避免重复信息。

---Instructions---

1.  分步说明:
    * 在对话历史的背景下仔细确定用户的查询意图，以充分理解用户的信息需求。
    * 仔细审查**上下文**中的 `文档块`。识别并提取所有与回答用户查询直接相关的信息片段。
    * 将提取的事实编织成连贯且合乎逻辑的回答。你自己的知识**仅**用于组织通顺的句子和连接观点，**不得**引入任何外部信息。
    * 跟踪直接支持回答中事实的文档块的 reference_id。将 reference_id 与 `参考文档列表 (Reference Document List)` 中的条目相关联，以生成适当的引用。
    * 在回答末尾生成 **References** 部分。每个参考文档必须直接支持回答中提出的事实。
    * 参考文献部分之后不要生成任何内容。

2.  内容与依据:
    * 严格遵守**上下文**中提供的信息；**不要**捏造、假设或推断任何未明确陈述的信息。
    * 如果**上下文**中找不到答案，请说明你没有足够的信息来回答。不要尝试猜测。

3.  格式与语言:
    * **回答必须使用与用户查询相同的语言（通常为中文）。**
    * 回答必须利用 Markdown 格式以增强清晰度和结构（例如，标题、粗体文本、要点）。
    * 回答应以 {response_type} 的形式呈现。

4.  参考文献部分格式:
    * 参考文献部分应位于标题下: `### References`
    * 参考列表条目应符合格式: `* [n] 文档标题`。不要在左方括号 (`[`) 后包含脱字符 (`^`)。
    * 引用中的文档标题必须保留其原始语言。
    * 每行输出一个引用。
    * 最多提供 5 个最相关的引用。
    * 不要在参考文献后生成脚注部分或任何评论、总结或解释。

5.  参考文献部分示例:
```

### References

  - [1] 文档标题一
  - [2] 文档标题二
  - [3] 文档标题三

<!-- end list -->

````

6.  附加指令: {user_prompt}


---Context---

{content_data}
"""

PROMPTS["kg_query_context"] = r"""
知识图谱数据 (Entity):

```json
{entities_str}
````

知识图谱数据 (Relationship):

```json
{relations_str}
```

文档块 (每个条目都有一个 reference_id 指向 `参考文档列表`):

```json
{text_chunks_str}
```

参考文档列表 (每个条目以 [reference_id] 开头，对应文档块中的条目):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = r"""
文档块 (每个条目都有一个 reference_id 指向 `参考文档列表`):

```json
{text_chunks_str}
```

参考文档列表 (每个条目以 [reference_id] 开头，对应文档块中的条目):

```
{reference_list_str}
```

"""

PROMPTS["keywords_extraction"] = r"""---Role---
你是一名专家级的关键词提取器，专门负责分析用户查询以用于检索增强生成 (RAG) 系统。你的目的是识别用户查询中的高级和低级关键词，这些关键词将用于有效的文档检索。

---Goal---
给定一个用户查询，你的任务是提取两种不同类型的关键词：

1.  **high_level_keywords (高级关键词)**:用于涵盖核心概念或主题，捕捉用户的核心意图、主题领域或所问问题的类型。
2.  **low_level_keywords (低级关键词)**: 用于具体实体或细节，识别具体的实体、专有名词、技术术语、产品名称或具体项目。

---Instructions & Constraints---

1.  **输出格式**: 你的输出**必须**是一个有效的 JSON 对象，别无其他。不要在 JSON 之前或之后包含任何解释性文本、markdown 代码围栏（如 ```json）或任何其他文本。它将被 JSON 解析器直接解析。
2.  **事实来源**: 所有关键词必须明确源自用户查询，并且高级和低级关键词类别都必须包含内容。请尽量生成**中文关键词**，除非查询中包含特定的英文专有名词。
3.  **简明扼要**: 关键词应为简明的词语或有意义的短语。当多词短语代表单一概念时，优先使用短语。例如，从“苹果公司的最新财务报告”中，你应该提取“最新财务报告”和“苹果公司”，而不是“最新”、“财务”、“报告”和“苹果”。
4.  **处理边缘情况**: 对于过于简单、模糊或荒谬的查询（例如，“你好”、“好的”、“asdfghjkl”），你必须返回一个两个关键词类型列表均为空的 JSON 对象。

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    r"""Example 1:
    
    Query: "国际贸易如何影响全球经济稳定？"
    
    Output:
    {
    "high_level_keywords": ["国际贸易", "全球经济稳定", "经济影响"],
    "low_level_keywords": ["贸易协定", "关税", "货币汇率", "进口", "出口"]
    }
    
    """,
    r"""Example 2:
    
    Query: "森林砍伐对生物多样性有哪些环境后果？"
    
    Output:
    {
    "high_level_keywords": ["环境后果", "森林砍伐", "生物多样性丧失"],
    "low_level_keywords": ["物种灭绝", "栖息地破坏", "碳排放", "雨林", "生态系统"]
    }
    
    """,
    r"""Example 3:
    
    Query: "教育在减少贫困中的作用是什么？"
    
    Output:
    {
    "high_level_keywords": ["教育", "减贫", "社会经济发展"],
    "low_level_keywords": ["入学机会", "识字率", "职业培训", "收入不平等"]
    }
    
    """,
]
