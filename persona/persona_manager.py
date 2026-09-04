"""
画像管理器 (Persona Manager)

参考 immortal-skill 的 merge-policy.md 和 manifest.json 设计，
负责人格画像的存储、增量合并和冲突处理。

核心功能：
1. 画像持久化 — 把 Step 2 的提取结果保存到目录，含 manifest 元数据
2. 增量合并 — 追加新聊天记录时，智能合并已有画像，而不是从头重跑
3. 冲突处理 — 证据分级 + 时间优先 + 待决冲突表
4. 跨维度一致性检查 — 不同维度的信息是否矛盾

目录结构（一个人物一个目录）：
    persona_data/
    └── xiao-mei/                 ← 人物 slug（唯一标识）
        ├── manifest.json          ← 元数据（创建时间、来源、维度、指纹）
        ├── interaction.md         ← 互动风格
        ├── personality.md         ← 性格与价值观
        ├── memory.md              ← 记忆与经历
        └── conflicts.md           ← 待决冲突表
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from logger import logger
from persona.persona_extractor import ExtractionDimension, ExtractionResult


# ============================================================
# 常量
# ============================================================
DEFAULT_PERSONA_ROOT = "persona_data"   # 默认画像根目录
CONFLICT_FILENAME = "conflicts.md"       # 冲突表文件名
MANIFEST_FILENAME = "manifest.json"      # 元数据文件名


# ============================================================
# 画像元数据
# ============================================================
@dataclass
class PersonaManifest:
    """画像元数据 — 对应 immortal-skill 的 manifest.json

    记录画像的基本信息和版本指纹，用于增量更新时判断哪些文件变了。
    """
    slug: str                                   # 唯一标识（如 "xiao-mei"）
    name: str                                   # 显示名（如 "小美"）
    persona_type: str = "partner"              # 角色类型（partner/friend/mentor...）
    created_at: str = ""                       # 创建时间 ISO 格式
    updated_at: str = ""                       # 最后更新时间
    sources: List[str] = field(default_factory=list)   # 数据来源列表
    platforms: List[str] = field(default_factory=list)  # 来源平台
    dimensions: List[str] = field(default_factory=list)  # 已提取的维度
    fingerprints: Dict[str, str] = field(default_factory=dict)  # 各文件指纹（用于检测变更）
    total_messages: int = 0                    # 累计消息数

    @classmethod
    def from_dict(cls, data: dict) -> "PersonaManifest":
        """从字典加载"""
        return cls(
            slug=data.get("slug", ""),
            name=data.get("name", ""),
            persona_type=data.get("persona_type", "partner"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            sources=data.get("sources", []),
            platforms=data.get("platforms", []),
            dimensions=data.get("dimensions", []),
            fingerprints=data.get("fingerprints", {}),
            total_messages=data.get("total_messages", 0),
        )

    def to_dict(self) -> dict:
        """转字典"""
        return {
            "slug": self.slug,
            "name": self.name,
            "persona_type": self.persona_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sources": self.sources,
            "platforms": self.platforms,
            "dimensions": self.dimensions,
            "fingerprints": self.fingerprints,
            "total_messages": self.total_messages,
        }


# ============================================================
# 画像管理器
# ============================================================
class PersonaManager:
    """画像管理器

    负责人格画像的 CRUD、增量合并和冲突处理。

    使用方式：
        manager = PersonaManager()

        # 首次创建画像
        manager.create_persona(
            slug="xiao-mei",
            name="小美",
            results={dim: extraction_result},
        )

        # 读取画像
        persona = manager.load_persona("xiao-mei")

        # 增量合并新提取结果
        manager.merge_results("xiao-mei", new_results)
    """

    def __init__(self, root_dir: str = DEFAULT_PERSONA_ROOT):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[画像管理] 根目录：%s", self.root_dir.resolve())

    # ============================================================
    # 基础 CRUD
    # ============================================================

    def create_persona(
        self,
        slug: str,
        name: str,
        results: Dict[ExtractionDimension, ExtractionResult],
        sources: Optional[List[str]] = None,
        platforms: Optional[List[str]] = None,
        persona_type: str = "partner",
        total_messages: int = 0,
    ) -> PersonaManifest:
        """创建一个新的人物画像

        参数：
            slug: 唯一标识（英文/拼音，用作目录名）
            name: 显示名
            results: Step 2 提取的各维度结果
            sources: 数据来源描述列表
            platforms: 来源平台列表
            persona_type: 角色类型（partner/friend/mentor...）
            total_messages: 累计消息数

        返回：
            PersonaManifest 元数据对象
        """
        persona_dir = self.root_dir / slug
        if persona_dir.exists():
            raise FileExistsError(f"画像已存在：{slug}，请使用 merge_results 增量更新")

        persona_dir.mkdir(parents=True)

        now = datetime.now(timezone.utc).isoformat()
        manifest = PersonaManifest(
            slug=slug,
            name=name,
            persona_type=persona_type,
            created_at=now,
            updated_at=now,
            sources=sources or [],
            platforms=platforms or [],
            dimensions=[],
            fingerprints={},
            total_messages=total_messages,
        )

        # 保存各维度文件
        for dim, result in results.items():
            self._save_dimension(persona_dir, dim, result.content, manifest)

        # 初始化冲突表（空）
        conflicts_path = persona_dir / CONFLICT_FILENAME
        conflicts_path.write_text(
            "# 待决冲突\n\n暂无冲突。\n",
            encoding="utf-8",
        )
        manifest.fingerprints[CONFLICT_FILENAME] = self._fingerprint(
            conflicts_path.read_text(encoding="utf-8")
        )

        # 保存 manifest
        self._save_manifest(persona_dir, manifest)

        logger.info(
            "[画像管理] 创建画像 %s：%d 个维度，%d 条消息",
            slug, len(results), total_messages,
        )
        return manifest

    def load_persona(self, slug: str) -> Optional[PersonaManifest]:
        """加载画像元数据

        返回 None 表示画像不存在。
        """
        persona_dir = self.root_dir / slug
        manifest_path = persona_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            return None
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return PersonaManifest.from_dict(data)

    def load_dimension(self, slug: str, dim: ExtractionDimension) -> Optional[str]:
        """加载某个维度的内容

        返回 Markdown 文本，不存在返回 None。
        """
        filepath = self._dimension_path(slug, dim)
        if not filepath.exists():
            return None
        return filepath.read_text(encoding="utf-8")

    def load_conflicts(self, slug: str) -> Optional[str]:
        """加载冲突表"""
        filepath = self.root_dir / slug / CONFLICT_FILENAME
        if not filepath.exists():
            return None
        return filepath.read_text(encoding="utf-8")

    def list_personas(self) -> List[str]:
        """列出所有画像的 slug"""
        return [
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and (d / MANIFEST_FILENAME).exists()
        ]

    def delete_persona(self, slug: str) -> bool:
        """删除一个画像（慎用！）
        返回 True 表示删除成功。
        """
        persona_dir = self.root_dir / slug
        if not persona_dir.exists():
            return False

        # 递归删除目录
        for f in persona_dir.rglob("*"):
            if f.is_file():
                f.unlink()
        persona_dir.rmdir()

        logger.info("[画像管理] 删除画像：%s", slug)
        return True

    # ============================================================
    # 增量合并（核心功能）
    # ============================================================

    def merge_results(
        self,
        slug: str,
        new_results: Dict[ExtractionDimension, ExtractionResult],
        api_key: str,
        new_messages: int = 0,
    ) -> PersonaManifest:
        """增量合并新的提取结果

        这是画像管理器的核心功能。当用户追加新聊天记录时，
        不需要从头重跑全部提取，而是把新结果和已有结果合并。

        合并策略（参考 merge-policy.md）：
        1. 按维度逐一合并
        2. 证据级别高者优先（verbatim > artifact > impression）
        3. 同级矛盾：较新者为主 / 并列保留 / 进入冲突表
        4. 跨维度一致性检查
        5. 更新 manifest 元数据

        参数：
            slug: 画像标识
            new_results: 新提取的维度结果
            api_key: LLM API Key（合并需要调用 LLM）
            new_messages: 新增消息数
        """
        from persona.persona_extractor import PersonaExtractor

        persona_dir = self.root_dir / slug
        if not persona_dir.exists():
            raise FileNotFoundError(f"画像不存在：{slug}，请先 create_persona")

        manifest = self.load_persona(slug)
        if not manifest:
            raise FileNotFoundError(f"无法加载画像元数据：{slug}")

        extractor = PersonaExtractor(api_key)

        for dim, new_result in new_results.items():
            dim_name = dim.value
            old_content = self.load_dimension(slug, dim)

            if old_content is None:
                # 该维度之前没有，直接保存
                self._save_dimension(persona_dir, dim, new_result.content, manifest)
                logger.info("[画像管理] %s：新增维度 %s", slug, dim_name)
                continue

            # 已有该维度，调用 LLM 合并
            logger.info("[画像管理] %s：合并维度 %s...", slug, dim_name)
            merged_content = self._merge_dimension_content(
                extractor, old_content, new_result.content, dim, manifest.name,
            )

            self._save_dimension(persona_dir, dim, merged_content, manifest)

        # 跨维度一致性检查
        self._check_cross_dimension_consistency(persona_dir, manifest, extractor)

        # 更新元数据
        manifest.updated_at = datetime.now(timezone.utc).isoformat()
        manifest.total_messages += new_messages
        self._save_manifest(persona_dir, manifest)

        logger.info(
            "[画像管理] %s：合并完成，新增 %d 条消息，累计 %d 条",
            slug, new_messages, manifest.total_messages,
        )
        return manifest

    def _merge_dimension_content(
        self,
        extractor: "PersonaExtractor",
        old_content: str,
        new_content: str,
        dim: ExtractionDimension,
        target_name: str,
    ) -> str:
        """调用 LLM 合并同一维度的新旧内容

        合并规则：
        - 去重：完全重复的条目只保留一条
        - 证据优先级：verbatim > artifact > impression
        - 同级矛盾：较新者优先，无法裁决则写入冲突表
        - 保留所有证据标注
        """
        merge_prompt = f"""你是人格画像合并器。以下是针对「{target_name}」的
{dim.value}（{self._dim_name_zh(dim)}）维度的两份画像：

- 旧版：之前提取的结果
- 新版：从新聊天记录中提取的结果

请合并为一份完整的画像，遵循以下规则：

1. **去重**：完全重复的条目只保留一条
2. **证据优先级**：verbatim > artifact > impression，高级别证据覆盖低级别
3. **同级矛盾处理**：
   - 能判断时间先后 → 以较新者为准，旧版标注「已过期」
   - 场景不同 → 并列保留，标明适用场景
   - 无法裁决 → 标注「⚠️ 待决冲突」
4. **保留所有证据级别标注**（verbatim/artifact/impression）
5. **不要添加新信息**，只整理已有内容
6. 保持原有的 Markdown 结构

输出合并后的完整 Markdown："""

        combined = (
            f"【旧版画像】\n{old_content}\n\n"
            f"【新版画像】\n{new_content}\n"
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=merge_prompt),
                HumanMessage(content=combined),
            ]
            # 用提取模型合并（max_tokens=4096），因为合并输出可能很长
            # 轻量模型（300 token）不够用
            response = extractor._extract_llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            return content if content else old_content  # 合并失败回退到旧版
        except Exception as e:
            logger.error("[画像管理] 合并 %s 失败：%s", dim.value, e)
            return old_content  # 回退

    # ============================================================
    # 冲突处理
    # ============================================================

    def add_conflict(
        self,
        slug: str,
        title: str,
        claim_a: str,
        claim_b: str,
        dimension_a: str = "",
        dimension_b: str = "",
        verdict: str = "待用户决定",
        reason: str = "",
    ) -> str:
        """添加一条待决冲突

        参数：
            slug: 画像标识
            title: 冲突主题
            claim_a: 说法 A
            claim_b: 说法 B
            dimension_a/b: 所属维度
            verdict: 临时裁定
            reason: 裁定理由
        """
        persona_dir = self.root_dir / slug
        conflicts_path = persona_dir / CONFLICT_FILENAME

        existing = conflicts_path.read_text(encoding="utf-8") if conflicts_path.exists() else "# 待决冲突\n\n"

        # 计算冲突编号
        conflict_num = existing.count("## C") + 1

        new_conflict = f"""
## C{conflict_num}：{title}
- 说法 A：{claim_a}（{dimension_a}）
- 说法 B：{claim_b}（{dimension_b}）
- 临时裁定：{verdict}
- 裁定理由：{reason}
"""

        updated = existing.rstrip() + "\n" + new_conflict + "\n"
        conflicts_path.write_text(updated, encoding="utf-8")

        # 更新指纹
        manifest = self.load_persona(slug)
        if manifest:
            manifest.fingerprints[CONFLICT_FILENAME] = self._fingerprint(updated)
            manifest.updated_at = datetime.now(timezone.utc).isoformat()
            self._save_manifest(persona_dir, manifest)

        logger.info("[画像管理] %s：新增冲突 C%d - %s", slug, conflict_num, title)
        return f"C{conflict_num}"

    def _check_cross_dimension_consistency(
        self,
        persona_dir: Path,
        manifest: PersonaManifest,
        extractor: "PersonaExtractor",
    ):
        """跨维度一致性检查（简化版）

        参考 merge-policy.md 第 3 节：
        - interaction 中的沟通风格 vs personality 中的价值观是否吻合？
        - memory 中的自述 vs personality 中的特征是否矛盾？

        当前实现：调用 LLM 快速扫描，发现明显矛盾时写入 conflicts.md
        """
        # 只有当至少有 2 个维度时才检查
        if len(manifest.dimensions) < 2:
            return

        # 加载已有内容
        contents = {}
        for dim_str in manifest.dimensions:
            try:
                dim = ExtractionDimension(dim_str)
                path = persona_dir / f"{dim.value}.md"
                if path.exists():
                    contents[dim_str] = path.read_text(encoding="utf-8")
            except ValueError:
                continue

        if len(contents) < 2:
            return

        check_prompt = f"""你是画像一致性检查员。以下是「{manifest.name}」的人格画像，
包含 {len(contents)} 个维度。请检查不同维度之间是否存在明显矛盾。

检查要点：
1. 互动风格 vs 性格价值观：行为模式是否吻合？
2. 记忆经历 vs 性格价值观：自述特征是否一致？
3. 如果发现矛盾，用简短语言描述，不超过 3 条

如果没有明显矛盾，只回答「无明显矛盾」。
如果有矛盾，按以下格式输出：
```
矛盾1：<主题>
  - 维度A：<内容摘要>
  - 维度B：<内容摘要>
```

只输出矛盾列表，不要其他分析。"""

        all_content = "\n\n".join(
            f"【{dim}】\n{content[:2000]}" for dim, content in contents.items()
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=check_prompt),
                HumanMessage(content=all_content),
            ]
            response = extractor._merge_llm.invoke(messages)
            result = response.content if hasattr(response, "content") else str(response)

            if "无明显矛盾" not in result and result.strip():
                # 有矛盾，写入冲突表
                conflicts_path = persona_dir / CONFLICT_FILENAME
                existing = conflicts_path.read_text(encoding="utf-8")

                conflict_section = "\n## 跨维度一致性检查\n\n" + result + "\n"
                if "跨维度一致性检查" not in existing:
                    updated = existing.rstrip() + conflict_section
                else:
                    # 替换旧的检查结果
                    import re
                    updated = re.sub(
                        r"## 跨维度一致性检查\n.*?(?=\n## |\Z)",
                        "## 跨维度一致性检查\n\n" + result.strip() + "\n",
                        existing,
                        flags=re.DOTALL,
                    )

                conflicts_path.write_text(updated, encoding="utf-8")
                manifest.fingerprints[CONFLICT_FILENAME] = self._fingerprint(updated)
                logger.info("[画像管理] %s：跨维度一致性检查发现矛盾", manifest.slug)
            else:
                logger.info("[画像管理] %s：跨维度一致性检查通过", manifest.slug)

        except Exception as e:
            logger.warning("[画像管理] 跨维度一致性检查失败：%s", e)

    # ============================================================
    # 工具方法
    # ============================================================

    def _dimension_path(self, slug: str, dim: ExtractionDimension) -> Path:
        """获取维度文件路径"""
        return self.root_dir / slug / f"{dim.value}.md"

    def _save_dimension(
        self,
        persona_dir: Path,
        dim: ExtractionDimension,
        content: str,
        manifest: PersonaManifest,
    ):
        """保存单个维度文件并更新 manifest"""
        filepath = persona_dir / f"{dim.value}.md"
        filepath.write_text(content, encoding="utf-8")

        # 更新指纹和维度列表
        manifest.fingerprints[f"{dim.value}.md"] = self._fingerprint(content)
        if dim.value not in manifest.dimensions:
            manifest.dimensions.append(dim.value)

    def _save_manifest(self, persona_dir: Path, manifest: PersonaManifest):
        """保存 manifest.json"""
        manifest_path = persona_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _fingerprint(content: str) -> str:
        """计算内容指纹（MD5），用于检测文件是否变更"""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _dim_name_zh(dim: ExtractionDimension) -> str:
        """维度的中文名称"""
        names = {
            ExtractionDimension.INTERACTION: "互动风格",
            ExtractionDimension.PERSONALITY: "性格与价值观",
            ExtractionDimension.MEMORY: "记忆与经历",
        }
        return names.get(dim, dim.value)
