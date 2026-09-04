"""
人格提取引擎 (Persona Extractor)

参考 immortal-skill 的 3 个提取器 prompt，适配为 LoveMender 伴侣场景：
1. 互动风格提取 → 从 interaction-extractor.md 改造（7 维度通用+伴侣特化）
2. 依恋风格与情感需求 → 从 personality-extractor.md 改造（6 维度行为证据）
3. 关系记忆图谱 → 从 memory-extractor.md 改造（5 视角叙事分析）

核心流程：
    清洗后的语料 (corpus_cleaner.py 的输出)
        ↓
    分块（避免超出 LLM 上下文窗口）
        ↓
    每块执行 3 个维度的 LLM 提取
        ↓
    合并多块结果（去重 + 证据分级）
        ↓
    输出结构化 Markdown（供 persona_prompt.py 使用）
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from logger import logger
from model_factory import create_summary_llm
from config import TEXT_MODEL_NAME, DASHSCOPE_BASE_URL, request_timeout


# ============================================================
# 常量
# ============================================================
CHUNK_SIZE = 4000          # 每块最大字符数（约 2000-3000 token，安全在 qwen3.8-27b 的 8K 窗口内）
CHUNK_OVERLAP = 200        # 块间重叠字符数，避免在消息中间切断丢失上下文


# ============================================================
# 提取维度
# ============================================================
class ExtractionDimension(str, Enum):
    """提取维度 — 对应 immortal-skill 的 3 个提取器

    伴侣场景下 procedure（程序性知识）不适用，只提取 3 个维度。
    """
    INTERACTION = "interaction"    # 互动风格（TA 怎么说话、怎么回应）
    PERSONALITY = "personality"    # 性格与价值观（TA 是什么样的人）
    MEMORY = "memory"              # 记忆与经历（TA 经历过什么）


# ============================================================
# 提取结果
# ============================================================
@dataclass
class ExtractionResult:
    """单次提取结果

    每个维度提取后生成一个 ExtractionResult，
    content 是 LLM 输出的结构化 Markdown，
    可以直接保存到文件或传给 persona_prompt.py。
    """
    dimension: ExtractionDimension
    content: str                          # Markdown 格式的提取结果
    chunks_processed: int = 1             # 处理了多少个语料块
    raw_chars: int = 0                     # 原始语料字符数

    def save(self, output_dir: str) -> Path:
        """保存到文件（按维度名命名）"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        f = out / f"{self.dimension.value}.md"
        f.write_text(self.content, encoding="utf-8")
        logger.info("[人格提取] %s 结果已保存到 %s", self.dimension.value, f)
        return f


# ============================================================
# 人格提取引擎
# ============================================================
class PersonaExtractor:
    """人格提取引擎

    将清洗后的语料送入 LLM，提取 3 个维度的人格画像。

    模型路由策略（复用 model_factory.py 的三级路由）：
    - 提取：create_text_llm (qwen3.8-27b) — 纯文本分析，中等能力，省 token
    - 合并：create_summary_llm (qwen3.7-flash) — 快速合并去重，最轻量

    使用方式：
        extractor = PersonaExtractor(api_key="sk-xxx")
        results = extractor.extract_all(corpus_md, target_name="小美")
        for dim, result in results.items():
            result.save("persona_data/")
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        # 提取用中等模型（需要理解能力，但不需要多模态）
        # 专用实例：增大 max_tokens（提取需要长输出），禁用 thinking（避免思考过程占满 token）
        self._extract_llm = ChatOpenAI(
            model=TEXT_MODEL_NAME,
            api_key=api_key,
            base_url=DASHSCOPE_BASE_URL,
            timeout=request_timeout,
            temperature=0.3,           # 提取任务需要稳定输出，降低随机性
            max_tokens=4096,          # 提取需要长输出，给足够空间
            extra_body={"enable_thinking": False},
        )
        # 合并用轻量模型（只是去重和整理，不需要强推理）
        self._merge_llm = create_summary_llm(api_key)

    # ============================================================
    # 公开方法
    # ============================================================

    def extract_all(
        self,
        corpus_md: str,
        target_name: str = "TA",
    ) -> Dict[ExtractionDimension, ExtractionResult]:
        """执行全部 3 个维度的提取

        参数：
            corpus_md: 语料 Markdown（来自 corpus_cleaner.py 的 to_corpus_markdown）
            target_name: 分析对象的代号（如"小美"）

        返回：
            {维度: ExtractionResult} 字典
        """
        results: Dict[ExtractionDimension, ExtractionResult] = {}

        for dim in ExtractionDimension:
            logger.info("[人格提取] 开始 %s 维度提取（目标：%s）...", dim.value, target_name)
            results[dim] = self._extract_dimension(dim, corpus_md, target_name)
            logger.info(
                "[人格提取] %s 完成：%d 字符结果，处理 %d 块",
                dim.value, len(results[dim].content), results[dim].chunks_processed,
            )

        return results

    def extract_interaction(
        self, corpus_md: str, target_name: str = "TA"
    ) -> ExtractionResult:
        """只提取互动风格维度"""
        return self._extract_dimension(ExtractionDimension.INTERACTION, corpus_md, target_name)

    def extract_personality(
        self, corpus_md: str, target_name: str = "TA"
    ) -> ExtractionResult:
        """只提取性格与价值观维度"""
        return self._extract_dimension(ExtractionDimension.PERSONALITY, corpus_md, target_name)

    def extract_memory(
        self, corpus_md: str, target_name: str = "TA"
    ) -> ExtractionResult:
        """只提取记忆与经历维度"""
        return self._extract_dimension(ExtractionDimension.MEMORY, corpus_md, target_name)

    # ============================================================
    # 内部方法：分块 → 提取 → 合并
    # ============================================================

    def _extract_dimension(
        self,
        dim: ExtractionDimension,
        corpus_md: str,
        target_name: str,
    ) -> ExtractionResult:
        """单维度提取的完整流程：分块 → 逐块提取 → 合并"""

        # Step 1: 分块
        chunks = self._chunk_corpus(corpus_md)
        logger.info("[人格提取] %s：语料 %d 字符 → %d 块", dim.value, len(corpus_md), len(chunks))

        # Step 2: 获取该维度的 system prompt
        system_prompt = self._build_prompt(dim, target_name)

        # Step 3: 逐块提取
        chunk_results: List[str] = []
        for i, chunk in enumerate(chunks):
            logger.info("[人格提取] %s：处理第 %d/%d 块...", dim.value, i + 1, len(chunks))
            result = self._run_extraction(chunk, system_prompt, target_name)
            if result:
                chunk_results.append(result)

        # Step 4: 合并结果
        if len(chunk_results) == 0:
            content = f"# {dim.value}：提取失败\n\nLLM 未返回有效结果。"
        elif len(chunk_results) == 1:
            content = chunk_results[0]
        else:
            content = self._merge_results(chunk_results, dim, target_name)

        return ExtractionResult(
            dimension=dim,
            content=content,
            chunks_processed=len(chunks),
            raw_chars=len(corpus_md),
        )

    def _chunk_corpus(self, corpus_md: str) -> List[str]:
        """将语料 Markdown 分块

        策略：按字符数切分，在块之间保留 overlap 重叠区域，
        确保不会在一条消息的中间硬切断导致语义丢失。

        如果语料很短（< CHUNK_SIZE），直接返回一个块。
        """
        if len(corpus_md) <= CHUNK_SIZE:
            return [corpus_md]

        chunks: List[str] = []#变量类型注解，确保返回的是字符串列表
        start = 0
        while start < len(corpus_md):
            end = start + CHUNK_SIZE
            chunk = corpus_md[start:end]

            # 尽量在换行处切断，避免切断一句话
            if end < len(corpus_md):
                last_newline = chunk.rfind("\n")
                if last_newline > CHUNK_SIZE // 2:
                    end = start + last_newline + 1
                    chunk = corpus_md[start:end]

            chunks.append(chunk)

            # 下一块从 overlap 处开始
            start = end - CHUNK_OVERLAP if end < len(corpus_md) else end

        return chunks

    def _build_prompt(self, dim: ExtractionDimension, target_name: str) -> str:
        """根据维度构建 system prompt

        3 个维度的 prompt 分别改编自 immortal-skill 的：
        - interaction-extractor.md
        - personality-extractor.md
        - memory-extractor.md

        改编要点：
        1. 固定 persona 为 partner（伴侣场景）
        2. 维度 6/7 始终输出（immortal-skill 中按 persona 条件输出）
        3. 证据分级规则保持一致（verbatim > artifact > impression）
        """
        if dim == ExtractionDimension.INTERACTION:
            return self._interaction_prompt(target_name)
        if dim == ExtractionDimension.PERSONALITY:
            return self._personality_prompt(target_name)
        return self._memory_prompt(target_name)

    def _run_extraction(
        self,
        corpus_chunk: str,
        system_prompt: str,
        target_name: str,
    ) -> Optional[str]:
        """执行单次 LLM 提取

        使用 LangChain 的 invoke 模式：
        - SystemMessage: 提取指令（维度 prompt）
        - HumanMessage: 语料块

        返回 LLM 输出的 Markdown 文本，失败返回 None。

        注意：qwen3.x 模型默认开启 thinking 模式，
        响应可能将思考过程放在 reasoning_content，
        最终回答放在 content。两者都检查。
        """
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"以下是聊天记录语料：\n\n{corpus_chunk}"),
            ]
            response = self._extract_llm.invoke(messages)

            # qwen3.x thinking 模式：content 是最终回答，reasoning_content 是思考过程
            content = ""
            if hasattr(response, "content") and response.content:
                content = response.content

            # 如果 content 为空，尝试从 additional_kwargs 获取
            if not content and hasattr(response, "additional_kwargs"):
                content = response.additional_kwargs.get("reasoning_content", "")

            if not content:
                logger.warning("[人格提取] LLM 返回空内容，response 类型：%s", type(response).__name__)
                return None

            return content
        except Exception as e:
            logger.error("[人格提取] LLM 调用失败：%s", e, exc_info=True)
            return None

    def _merge_results(
        self,
        chunk_results: List[str],
        dim: ExtractionDimension,
        target_name: str,
    ) -> str:
        """合并多个语料块的提取结果

        当语料被分成多块时，每块独立提取会产生重复或片段化的结果。
        用轻量模型（qwen3.7-flash）做一次合并去重。

        合并策略（参考 immortal-skill merge-policy.md）：
        1. 同维度重复条目去重
        2. 证据级别高者优先（verbatim > artifact > impression）
        3. 矛盾项并列保留，标明来源块
        """
        merge_prompt = (
            f"你是一个人格画像合并器。以下是针对「{target_name}」的"
            f"{dim.value}（{self._dim_name_zh(dim)}）维度的多段提取结果，"
            f"来自不同语料块的分块提取。\n\n"
            f"请合并为一份完整的 Markdown 文档，遵循以下规则：\n"
            f"1. 去重：完全重复的条目只保留一条\n"
            f"2. 合并：同一特征的不同证据片段合并为一条，保留所有证据级别标注\n"
            f"3. 排序：按原始结构顺序排列\n"
            f"4. 保留所有 `verbatim`/`artifact`/`impression` 标注\n"
            f"5. 不要添加新信息，只整理已有内容\n"
            f"6. 如果有矛盾，并列保留并标注「⚠️ 矛盾」\n\n"
            f"输出合并后的完整 Markdown："
        )

        combined_input = "\n\n---\n\n".join(
            f"【第 {i+1} 块提取结果】\n{r}" for i, r in enumerate(chunk_results)
        )

        try:
            messages = [
                SystemMessage(content=merge_prompt),
                HumanMessage(content=combined_input),
            ]
            response = self._merge_llm.invoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error("[人格提取] 合并失败，回退为简单拼接：%s", e)
            # 回退：直接拼接，用分隔线隔开
            return "\n\n---\n\n".join(chunk_results)

    # ============================================================
    # 3 个维度的 Prompt 模板（改编自 immortal-skill）
    # ============================================================

    def _interaction_prompt(self, name: str) -> str:
        """互动风格提取 prompt — 改编自 interaction-extractor.md

        适配 LoveMender 伴侣场景：
        - 维度 1-5：通用沟通维度（保留）
        - 维度 6：情感互动（伴侣核心，始终输出）
        - 维度 7：关系动态（伴侣特有，始终输出）
        """
        return f"""你是互动风格分析专家。从聊天记录中提取「{name}」的互动风格。

## 核心约束
- 只基于材料中的可观测行为，不臆测
- 有材料才写，没有写「材料不足」
- 区分证据级别：verbatim（原话）> artifact（客观推断）> impression（主观印象）
- 正文只放 verbatim 和 artifact，印象放在分隔线以下

## 提取维度

### 维度 1：默认沟通方式
- 消息长度（长段还是短句？）
- 句式结构（完整句/碎片/表情包为主？）
- 响应速度（秒回？拖延？忽快忽慢？）
- 主动发起 vs 被动回应的比例

### 维度 2：提问与推回
- 什么情况下会追问？
- 推回/拒绝时的措辞和方式（直接还是委婉？）

### 维度 3：质疑与挑战
- 对什么话题会表达异议？
- 质疑时的语气和句式

### 维度 4：冲突场景应对
- 意见不合时的模式（回避/正面/冷战/讲道理？）
- 争吵升级还是降温？

### 维度 5：排斥场景
- 绝不做的沟通类型（如：从不爆粗口/从不冷暴力？）

### 维度 6：情感互动（伴侣核心）
- 表达关心的方式（言语/行动/转红包？）
- 撒娇/示弱/逞强的模式
- 开玩笑的边界与禁区
- 沉默时的含义（生气？思考？需要空间？）

### 维度 7：关系动态（伴侣特有）
- 互动节奏变化（热恋期 vs 平稳期 vs 冷淡期）
- 吵架/冷战/复合的模式
- 对关系议题的立场（未来规划/承诺/责任）

## 输出格式

严格按以下 Markdown 结构输出：

```markdown
# 互动与态度：{name}

## 默认沟通方式
- [条目]（`verbatim` | `artifact`，来源简述）

## 提问与推回
- [条目]（标注）

## 质疑与挑战
- [条目]（标注）

## 冲突场景应对
- [条目]（标注）

## 排斥场景
- [条目]（标注）

## 情感互动
- [条目]（标注）

## 关系动态
- [条目]（标注）

---

> 以下为数据提供者主观印象，非当事人自述：
- [印象]（`impression`）
```

若某维度材料不足，写「材料不足」。"""

    def _personality_prompt(self, name: str) -> str:
        """性格与价值观提取 prompt — 改编自 personality-extractor.md

        适配 LoveMender 伴侣场景：
        - 维度 1：核心价值观 → 依恋安全感来源
        - 维度 2-5：保留（口头禅/情绪模式/社交偏好/兴趣审美）
        - 维度 6：自我认知 vs 他人印象
        - 禁止心理诊断标签，聚焦可观测行为
        """
        return f"""你是性格分析专家。从聊天记录中提取「{name}」的性格特征与价值观。

## 核心约束
- 禁止使用心理学诊断标签（如「回避型依恋」「INTJ」）——聚焦可观测行为
- 允许记录矛盾面——人本身就是复杂的
- 有材料才写，没有写「材料不足」
- 证据级别：verbatim（原话）> artifact（客观推断）> impression（主观印象）
- 正文只放 verbatim 和 artifact，印象放分隔线以下

## 6 个提取维度

### 维度 1：核心价值观与安全感
- TA 在面对选择时，什么排在第一位？
- 有没有明确说过「我在乎 X」「Y 是不可接受的」？
- 什么给 TA 安全感？什么让 TA 不安？（从行为推断）

### 维度 2：口头禅与表达习惯
- 高频词汇、句式、语气词
- 特有的比喻或类比
- 强调/生气/开心时的用词变化

### 维度 3：情绪模式
- 什么场景触发积极情绪？表现是什么？
- 什么场景触发消极情绪？表现是什么？
- 压力下的行为变化
- 情绪恢复的方式（如有）

### 维度 4：社交偏好
- 在群体中的角色
- 对社交频率和深度的偏好
- 维护关系的方式

### 维度 5：兴趣与审美
- 持续投入时间精力的领域
- 消费/创作/评价中反映的审美倾向

### 维度 6：自我认知 vs 他人印象
- TA 怎么描述自己
- 他人怎么评价 TA（如有材料）
- 差异点

## 输出格式

```markdown
# 性格与价值观：{name}

## 核心价值观与安全感
- [价值观] —— [行为证据]（`verbatim` | `artifact`，来源）

## 口头禅与表达习惯
- [语言特征]（标注）

## 情绪模式
### 积极情绪
- [触发场景 → 表现]（标注）
### 消极情绪
- [触发场景 → 表现]（标注）
### 压力反应
- [描述]（标注）

## 社交偏好
- [偏好描述]（标注）

## 兴趣与审美
- [描述]（标注）

## 自我认知 vs 他人印象
- TA 说自己：...（`verbatim`）
- 他人评价：...（`impression`）
- 差异：...

---

> 以下为数据提供者的主观印象：
- [印象]（`impression`）
```

若某维度材料不足，写「材料不足」。"""

    def _memory_prompt(self, name: str) -> str:
        """关系记忆图谱提取 prompt — 改编自 memory-extractor.md

        适配 LoveMender 伴侣场景：
        - 视角 1：人生转折点 → 关系转折点
        - 视角 2：反复讲述的故事 → 反复提起的共同经历
        - 视角 3：共同记忆（核心维度）
        - 视角 4：情感地图
        - 视角 5：时代与环境 → 关系阶段特征
        """
        return f"""你是记忆分析专家。从聊天记录中提取与「{name}」相关的记忆与经历。

## 核心约束
- 有材料才写，没有写「材料不足」
- 区分「TA 自己讲的」（verbatim/artifact）和「他人补充的」（impression）
- 如果发现 TA 回避某话题，仅标注回避存在，不深入挖掘
- 证据级别：verbatim > artifact > impression

## 5 个提取视角

### 视角 1：关系转折点
- 关系中的重大变化（相识/确定关系/争吵/分离等）
- 转折前后的对比
- TA 自己对转折的评价（如有）

### 视角 2：反复提起的故事
- TA 在不同场合重复提到的经历
- 不同版本之间的差异
- 故事背后隐含的价值观

### 视角 3：共同记忆
- 一起经历的事件
- 各方对同一事件的不同叙述
- 「内部梗」和共同语言

### 视角 4：情感地图
- 提到某事时的语气变化（热情/低落/回避/怀念）
- 经常以正面情绪提及的话题
- 似乎不愿谈及的方向（仅标注，不追问）

### 视角 5：关系阶段特征
- 不同关系阶段的表现差异
- 特定事件对关系的影响
- 代际或环境特征的体现

## 输出格式

```markdown
# 记忆与经历：{name}

## 关系转折点
- [事件] —— [时间/时期]（`verbatim` | `artifact` | `impression`，来源简述）

## 反复提起的故事
### <故事标题>
- 核心内容：...
- 提及场景：...
- 隐含价值观：...
- 证据级别：...

## 共同记忆
- [经历]（标注）

## 情感地图
- [正面话题] ❤️（标注）
- [回避话题] 🚫 仅标注存在

## 关系阶段特征
- [描述]（标注）

---

> 以下为数据提供者的补充回忆，非当事人自述：
- [印象]（`impression`）
```

若某视角材料不足，写「材料不足」。"""

    # ============================================================
    # 工具方法
    # ============================================================

    @staticmethod
    def _dim_name_zh(dim: ExtractionDimension) -> str:
        """维度的中文名称"""
        names = {
            ExtractionDimension.INTERACTION: "互动风格",
            ExtractionDimension.PERSONALITY: "性格与价值观",
            ExtractionDimension.MEMORY: "记忆与经历",
        }
        return names.get(dim, dim.value)
