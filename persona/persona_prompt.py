"""
人格化提示词生成器 (Persona Prompt Generator)

Step 4：人格分析系统的最后一步——把前面 3 步的成果转化为可用于 LLM 对话的系统提示词。

核心思想：
    以前的 LoveMender：固定 4 种人设模板 → 回复千篇一律，有"AI 味"
    现在的 LoveMender：从真实聊天记录提取的人格画像 → 回复风格贴合真实人物

接入方式：
    在 main.py 中，原来的代码是：
        sys_prompt = role_prompt_dict[role_select].format(context=full_context)

    接入人格化后：
        persona_prompt = persona_prompt_generator.generate(user_text, emotion="angry")
        sys_prompt = base_persona + persona_prompt + common_suffix

本模块设计原则：
1. 不侵入现有代码 — 可以独立使用，也可以和原有角色模板叠加
2. 智能裁剪 — 画像可能很长，根据对话场景动态选择相关片段，控制 token
3. 分层注入 — 核心人格（必选）+ 场景相关片段（可选）+ 记忆锚点（可选）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from logger import logger
from persona.persona_manager import PersonaManager
from persona.persona_extractor import ExtractionDimension


# ============================================================
# 常量
# ============================================================
# 各维度在提示词中的最大字符数（控制 token，避免画像内容太长）
MAX_INTERACTION_CHARS = 1200      # 互动风格：最影响回复方式，给最多空间
MAX_PERSONALITY_CHARS = 800       # 性格价值观：影响回复立场
MAX_MEMORY_CHARS = 600            # 记忆经历：只取最相关的片段

# 情绪场景关键词映射（用于从记忆中找相关片段）
EMOTION_KEYWORDS = {
    "angry": ["生气", "吵架", "冷暴力", "道歉", "认错", "矛盾", "不满"],
    "sad": ["难过", "委屈", "哭", "伤心", "不开心", "失落"],
    "happy": ["开心", "高兴", "快乐", "幸福", "甜", "惊喜"],
    "anxious": ["担心", "害怕", "不安", "焦虑", "紧张", "压力"],
    "neutral": [],
}


# ============================================================
# 人格化提示词结果
# ============================================================
@dataclass
class PersonaPrompt:
    """人格化提示词结果

    包含三部分，可以灵活拼接：
    - core: 核心人格描述（互动风格 + 性格特征，必选注入）
    - memory: 相关记忆片段（根据对话情绪选择，可选注入）
    - conflicts: 待决冲突提醒（可选，防止 AI 说错话）
    """
    core: str = ""           # 核心人格（互动+性格）
    memory: str = ""         # 相关记忆
    conflicts: str = ""      # 冲突提醒

    def to_system_prompt_addon(self) -> str:
        """拼接成系统提示词的附加段（可以直接追加到基础人设后面）

        返回格式：
            ## 对方人格画像
            ### 互动风格
            ...
            ### 性格特征
            ...

            ## 相关记忆锚点
            ...

            ## ⚠️ 注意事项
            ...
        """
        parts = []

        if self.core:
            parts.append("## 对方人格画像\n" + self.core)

        if self.memory:
            parts.append("## 相关记忆锚点\n" + self.memory)

        if self.conflicts:
            parts.append("## ⚠️ 注意事项（存在矛盾信息，不要笃定）\n" + self.conflicts)

        if not parts:
            return ""

        return "\n\n".join(parts) + "\n"

    def total_chars(self) -> int:
        """总字符数（用于 token 估算）"""
        return len(self.core) + len(self.memory) + len(self.conflicts)


# ============================================================
# 人格化提示词生成器
# ============================================================
class PersonaPromptGenerator:
    """人格化提示词生成器

    从 PersonaManager 加载画像，根据当前对话场景生成精简的人格提示词，
    注入到 LLM 的系统提示词中。

    使用方式：
        generator = PersonaPromptGenerator("xiao-mei", manager)

        # 生成人格化提示词
        prompt = generator.generate(
            user_message="她又生气了，说我不在乎她",
            emotion="angry",
        )

        # 拼接到系统提示词
        full_sys_prompt = base_prompt + "\n\n" + prompt.to_system_prompt_addon()
    """

    def __init__(
        self,
        persona_slug: str,
        manager: PersonaManager,
    ):
        """
        参数：
            persona_slug: 画像标识（对应 persona_data/ 下的目录名）
            manager: PersonaManager 实例
        """
        self.slug = persona_slug
        self.manager = manager
        self._manifest = None  # 懒加载

        # 缓存各维度的完整内容（避免每次都读文件）
        self._cache: Dict[str, str] = {}

    # ============================================================
    # 主方法：生成人格化提示词
    # ============================================================

    def generate(
        self,
        user_message: str = "",
        emotion: str = "neutral",
        include_memory: bool = True,
        include_conflicts: bool = True,
    ) -> PersonaPrompt:
        """根据当前对话场景生成人格化提示词

        参数：
            user_message: 用户当前输入（用于关键词匹配记忆）
            emotion: 情绪标签（angry/sad/happy/anxious/neutral）
            include_memory: 是否包含相关记忆片段
            include_conflicts: 是否包含冲突提醒

        返回：
            PersonaPrompt 对象，可调用 to_system_prompt_addon() 转成字符串
        """
        result = PersonaPrompt()

        # Step 1: 核心人格（互动风格 + 性格特征，必选）
        result.core = self._build_core_persona()

        # Step 2: 相关记忆（根据情绪和关键词筛选，可选）
        if include_memory:
            result.memory = self._build_relevant_memory(user_message, emotion)

        # Step 3: 冲突提醒（可选）
        if include_conflicts:
            result.conflicts = self._build_conflict_warnings()

        logger.info(
            "[人格提示词] 生成完成：核心%d字 + 记忆%d字 + 冲突%d字 = %d字",
            len(result.core), len(result.memory), len(result.conflicts),
            result.total_chars(),
        )
        return result

    # ============================================================
    # 内部方法：核心人格
    # ============================================================

    def _build_core_persona(self) -> str:
        """构建核心人格描述（互动风格 + 性格特征）

        策略：
        - 互动风格：取前 MAX_INTERACTION_CHARS 字符（最核心的沟通特征在前）
        - 性格特征：取前 MAX_PERSONALITY_CHARS 字符
        - 用小标题分隔，方便 LLM 理解结构

        为什么不全部塞进去？
        - 画像可能有几千字，全塞进去太费 token
        - LLM 对提示词前面的内容更关注（首因效应），前面放最核心的
        """
        parts = []

        # 互动风格
        interaction = self._load_dimension_cached(ExtractionDimension.INTERACTION)
        if interaction:
            # 提取核心部分（去掉标题，取主体内容的前 N 字符）
            content = self._extract_markdown_body(interaction)
            trimmed = self._smart_trim(content, MAX_INTERACTION_CHARS)
            parts.append(f"### 互动风格\n{trimmed}")

        # 性格特征
        personality = self._load_dimension_cached(ExtractionDimension.PERSONALITY)
        if personality:
            content = self._extract_markdown_body(personality)
            trimmed = self._smart_trim(content, MAX_PERSONALITY_CHARS)
            parts.append(f"### 性格特征\n{trimmed}")

        return "\n\n".join(parts)

    # ============================================================
    # 内部方法：相关记忆
    # ============================================================

    def _build_relevant_memory(self, user_message: str, emotion: str) -> str:
        """从记忆维度中筛选与当前场景相关的片段

        策略（简单有效，不依赖向量检索，减少依赖）：
        1. 从记忆维度中提取所有条目
        2. 根据情绪关键词 + 用户输入关键词匹配
        3. 返回最相关的前几条，控制在 MAX_MEMORY_CHARS 内

        为什么不用向量检索？
        - 记忆画像本身不大（通常几千字），关键词匹配足够
        - 避免额外的 embedding 调用，省 token 省时间
        - 实现简单，可解释性强
        """
        memory = self._load_dimension_cached(ExtractionDimension.MEMORY)
        if not memory:
            return ""

        # 收集关键词
        keywords = set(EMOTION_KEYWORDS.get(emotion, []))

        # 从用户输入中提取关键词（简单分词：按空格和标点切）
        if user_message:
            import re
            words = re.findall(r'[\u4e00-\u9fa5]{2,}', user_message)  # 提取 2 字以上的中文词
            keywords.update(words)

        if not keywords:
            # 没有关键词，返回记忆的前 N 字符（概览）
            content = self._extract_markdown_body(memory)
            return self._smart_trim(content, MAX_MEMORY_CHARS)

        # 按行匹配关键词
        relevant_lines = []
        for line in memory.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # 如果行内容包含任意关键词
            if any(kw in line_stripped for kw in keywords):
                relevant_lines.append(line_stripped)

        if not relevant_lines:
            # 没匹配到，返回概览
            content = self._extract_markdown_body(memory)
            return self._smart_trim(content, MAX_MEMORY_CHARS // 2)

        # 把相关行拼起来，限制长度
        result = "\n".join(relevant_lines)
        if len(result) > MAX_MEMORY_CHARS:
            result = result[:MAX_MEMORY_CHARS] + "..."

        return result

    # ============================================================
    # 内部方法：冲突提醒
    # ============================================================

    def _build_conflict_warnings(self) -> str:
        """构建冲突提醒

        告诉 LLM：哪些信息有矛盾，不要笃定地说，保持开放性。
        只取前 3 条，避免信息过载。
        """
        conflicts = self.manager.load_conflicts(self.slug)
        if not conflicts:
            return ""

        # 简单提取冲突条目（找 C1、C2... 的标题）
        import re
        conflict_items = re.findall(r'## C\d+：(.+?)\n', conflicts)

        if not conflict_items:
            return ""

        # 取前 3 条
        top_items = conflict_items[:3]
        return "\n".join(f"- 关于「{item}」的信息存在矛盾，不要笃定" for item in top_items)

    # ============================================================
    # 工具方法
    # ============================================================

    def _load_dimension_cached(self, dim: ExtractionDimension) -> str:
        """加载维度内容（带缓存，避免重复读文件）"""
        key = dim.value
        if key not in self._cache:
            content = self.manager.load_dimension(self.slug, dim)
            self._cache[key] = content or ""
        return self._cache[key]

    @staticmethod
    def _extract_markdown_body(md_text: str) -> str:
        """从 Markdown 中提取主体内容（去掉一级标题）

        因为注入到提示词时会有自己的标题结构，不需要重复的一级标题。
        """
        lines = md_text.split("\n")
        # 跳过开头的一级标题（# 开头的行）和空行
        start = 0
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith("# ") or line_stripped == "":
                start = i + 1
            else:
                break

        return "\n".join(lines[start:]).strip()

    @staticmethod
    def _smart_trim(text: str, max_chars: int) -> str:
        """智能截断：在完整的条目处截断，不把一条信息切两半

        策略：
        - 如果总长度 <= max_chars，直接返回
        - 否则从前往后找最近的换行（条目分隔），在那里截断
        - 至少保留 max_chars // 2 的内容
        """
        if len(text) <= max_chars:
            return text

        # 找 max_chars 位置之前最后一个换行
        cut_pos = text.rfind("\n", 0, max_chars)

        # 如果找不到换行，或者截断太少，就在 max_chars 处硬截断
        if cut_pos < max_chars // 2:
            return text[:max_chars] + "..."

        return text[:cut_pos] + "\n..."

    def refresh_cache(self):
        """刷新缓存（画像更新后调用）"""
        self._cache.clear()
        self._manifest = None
        logger.info("[人格提示词] 缓存已刷新")
