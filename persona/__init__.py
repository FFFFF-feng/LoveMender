"""
LoveMender 人格分析系统（Persona System）

从聊天记录中提取人物画像，生成个性化的情感修复回复。

模块结构：
    corpus_cleaner.py    — 语料采集与清洗（统一数据格式）[Step 1 ✅]
    wechat_importer.py   — 微信本地数据库导入器
    persona_extractor.py — 人格提取引擎（LLM 驱动的多维度分析）[Step 2 ✅]
    persona_manager.py   — 画像管理（存储、合并、冲突处理）[Step 3 ✅]
    persona_prompt.py    — 人格化提示词生成器 [Step 4 ✅]
"""

from .corpus_cleaner import (
    Message,
    CorpusCleaner,
    ManualImporter,
    EvidenceLevel,
)
from .wechat_importer import (
    WeChatDBImporter,
    WeChatInfo,
)
from .wechat_ui_importer import (
    WeChatImporter,
)
from .persona_extractor import (
    PersonaExtractor,
    ExtractionDimension,
    ExtractionResult,
)
from .persona_manager import (
    PersonaManager,
    PersonaManifest,
)
from .persona_prompt import (
    PersonaPromptGenerator,
    PersonaPrompt,
)

__all__ = [
    # Step 1: 语料清洗
    "Message",
    "CorpusCleaner",
    "ManualImporter",
    "WeChatDBImporter",
    "WeChatInfo",
    "WeChatImporter",
    "EvidenceLevel",
    # Step 2: 人格提取
    "PersonaExtractor",
    "ExtractionDimension",
    "ExtractionResult",
    # Step 3: 画像管理
    "PersonaManager",
    "PersonaManifest",
    # Step 4: 人格化提示词
    "PersonaPromptGenerator",
    "PersonaPrompt",
]
