"""
语料采集与清洗模块

参考 immortal-skill 的 collectors/base.py + manual.py 设计，
负责将不同来源的聊天记录统一转换成标准格式。

核心设计：
1. Message 数据类 — 所有平台的消息最终都转成这个结构
2. CorpusCleaner 基类 — 定义清洗流水线的统一接口（模板方法模式）
3. ManualImporter — 最通用的导入方式（粘贴文本/文件导入）
4. EvidenceLevel — 证据分级（verbatim / artifact / impression）
"""

from __future__ import annotations

import csv
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, List

from logger import logger


# ============================================================
# 证据分级：verbatim > artifact > impression
# ============================================================
class EvidenceLevel(str, Enum):
    """证据级别 — 参考 immortal-skill 的 merge-policy.md

    用于标注人格信息的可信度：
    - VERBATIM: 对方的原话，可信度最高
    - ARTIFACT: 从行为/文档中客观推断
    - IMPRESSION: 主观印象，可信度最低，需要单独存放
    """
    VERBATIM = "verbatim"      # 原话（聊天记录、邮件原文等）
    ARTIFACT = "artifact"      # 客观推断（从行为/作品中推导）
    IMPRESSION = "impression"  # 主观印象（我觉得TA...）


# ============================================================
# 统一消息格式
# ============================================================
@dataclass
class Message:
    """统一消息格式 — 所有平台的消息最终都转换成这个结构

    参考 immortal-skill/collectors/base.py 中的 Message 类
    设计思想：不管数据来自微信、QQ、短信还是手动粘贴，
    都先转成统一的 Message 格式，后续处理只认这一种结构。
    """
    sender: str                       # 发送者昵称
    text: str                         # 消息内容
    timestamp: Optional[datetime] = None  # 发送时间
    channel_name: str = ""            # 会话名称（如：张三的私聊）
    msg_type: str = "text"            # 消息类型（text/image/emoji等）
    platform: str = ""                # 来源平台（wechat/manual/qq等）
    evidence_level: str = EvidenceLevel.VERBATIM.value  # 证据级别
    raw: dict = field(default_factory=dict)  # 原始数据（保留原始格式，方便排查）

    @property
    def date_str(self) -> str:
        """格式化的日期字符串，用于展示"""
        if self.timestamp:
            return self.timestamp.strftime("%Y-%m-%d %H:%M")
        return "未知时间"

    def to_dict(self) -> dict:
        """转成字典，方便 JSON 序列化"""
        d = asdict(self)
        if self.timestamp:
            d["timestamp"] = self.timestamp.isoformat()
        return d


# ============================================================
# 清洗器基类（模板方法模式）
# ============================================================
class CorpusCleaner(ABC):
    """语料清洗器基类 — 参考 immortal-skill 的 BaseCollector

    设计模式：模板方法模式（Template Method Pattern）
    基类定义了「清洗流水线」的固定步骤，
    子类只需要实现特定平台的数据解析逻辑。

    这样做的好处：
    1. 新增平台（比如QQ、短信）很简单，只要实现 2 个方法
    2. 所有平台的输出格式统一，后续模块不用关心来源
    3. 通用逻辑（保存、格式化、去重）只写一次
    """

    platform_name: str = "unknown"
    platform_name_zh: str = "未知平台"

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    @abstractmethod
    def parse(self, raw_data) -> List[Message]:
        """【子类实现】解析原始数据，返回 Message 列表

        这是唯一需要子类实现的方法 —— 把各种乱七八糟的原始数据
        （微信导出文件、QQ 聊天记录、用户粘贴的文本等）
        统一解析成标准的 Message 列表。
        """
        ...

    # --------------------------------------------------------
    # 以下都是通用方法，子类不用改
    # --------------------------------------------------------

    def clean(self, messages: List[Message]) -> List[Message]:
        """清洗消息：去重、过滤空消息、按时间排序

        这是模板方法的「通用步骤」，所有平台都一样。
        """
        # 过滤空消息
        messages = [m for m in messages if m.text and m.text.strip()]

        # 按时间排序（有时间的排前面，没有时间的排后面）
        messages.sort(
            key=lambda m: m.timestamp or datetime.min.replace(tzinfo=timezone.utc)
        )

        # 简单去重（同一发送者 + 同一时间 + 同一内容）
        seen = set()#无序,但能判断去重
        unique = []#有序,能保持原始顺序
        for msg in messages:
            key = (msg.sender, msg.date_str, msg.text[:100])
            if key not in seen:
                seen.add(key)
                unique.append(msg)

        logger.info(
            "[语料清洗] %s：原始 %d 条 → 清洗后 %d 条",
            self.platform_name_zh, len(messages), len(unique)
        )
        return unique

    def to_corpus_markdown(
        self,
        messages: List[Message],
        target_person: str = "",
    ) -> str:
        """将消息列表转成 LLM 友好的 Markdown 格式

        参考 immortal-skill 的 to_corpus() 方法
        格式里包含元数据头（平台、证据级别、目标人物），
        方便后续 Prompt 提取时了解数据来源和可信度。
        """
        header = (
            f"<!-- 来源平台: {self.platform_name_zh} | "
            f"证据级别: {messages[0].evidence_level if messages else 'unknown'} | "
            f"消息数量: {len(messages)} -->\n"
            f"# 聊天记录语料\n\n"
        )
        if target_person:
            header += f"**分析对象：** {target_person}\n\n"
        header += "---\n\n"

        lines = [header]
        for msg in messages:
            prefix = f"[{msg.date_str}]" if msg.timestamp else "[]"
            sender_tag = f" **{msg.sender}**:" if msg.sender else ""
            lines.append(f"{prefix}{sender_tag} {msg.text}")

        return "\n".join(lines) + "\n"

    def save_json(self, messages: List[Message], output_path: str) -> Path:
        """保存为 JSON 文件（方便后续处理）"""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = [msg.to_dict() for msg in messages]
        output.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output


# ============================================================
# 手动导入清洗器（最通用的方式）
# ============================================================
class ManualImporter(CorpusCleaner):
    """手动导入清洗器 — 参考 immortal-skill 的 ManualCollector

    支持 4 种输入方式：
    1. 粘贴纯文本（最常用，用户直接复制聊天记录粘贴进来）
    2. TXT/MD 文件
    3. JSON 文件（结构化数据）
    4. CSV 文件（表格格式）

    这是用户最容易上手的导入方式，
    不需要任何第三方工具，复制粘贴就能用。
    """

    platform_name = "manual"
    platform_name_zh = "手动导入"

    def parse(self, raw_data) -> List[Message]:
        """统一入口：根据数据类型自动选择解析方式"""
        if isinstance(raw_data, str):
            return self.from_text(raw_data)
        if isinstance(raw_data, Path):
            return self.from_file(raw_data)
        if isinstance(raw_data, list):
            # 已经是 Message 列表了，直接返回
            return [m for m in raw_data if isinstance(m, Message)]
        raise ValueError(f"不支持的数据类型: {type(raw_data)}")

    # --------------------------------------------------------
    # 方式 1：从纯文本解析（用户粘贴的聊天记录）
    # --------------------------------------------------------
    @classmethod
    def from_text(cls, text: str, source_label: str = "粘贴导入") -> List[Message]:
        """从粘贴的文本中解析消息

        支持两种格式：
        A. 每行一条（微信导出的常见格式）:
            [时间] 发送者: 内容
            [时间] 发送者: 内容

        B. 按空行分段（用户自己整理的）:
            发送者说的话...

            另一个人说的话...

        设计思想：尽量兼容用户的各种粘贴格式，
        能识别多少算多少，识别不了的就当纯文本处理。
        """
        messages: List[Message] = []

        lines = text.strip().split('\n')

        # 先尝试解析「时间+发送者」格式（微信导出风格）
        pattern = re.compile(
            r'^\[?(\d{4}[-/]\d{2}[-/]\d{2}\s*\d{1,2}:\d{2}(?::\d{2})?)\]?\s*'  # 时间
            r'([^:：]+)[:：]\s*'  # 发送者
            r'(.*)'  # 内容
        )

        current_sender = ""
        current_text = []
        current_time = None

        for line in lines:
            line = line.strip()
            if not line:
                # 空行：如果有累积的内容，保存一条消息
                if current_text:
                    messages.append(Message(
                        sender=current_sender or "未知",
                        text="\n".join(current_text),
                        timestamp=current_time,
                        channel_name=source_label,
                        platform="manual",
                        evidence_level=EvidenceLevel.VERBATIM.value,
                    ))
                    current_text = []
                continue

            match = pattern.match(line)
            if match:
                # 匹配到了「时间+发送者+内容」格式
                # 先保存之前累积的消息
                if current_text:
                    messages.append(Message(
                        sender=current_sender or "未知",
                        text="\n".join(current_text),
                        timestamp=current_time,
                        channel_name=source_label,
                        platform="manual",
                        evidence_level=EvidenceLevel.VERBATIM.value,
                    ))
                    current_text = []

                time_str = match.group(1).replace('/', '-').strip()
                sender = match.group(2).strip()
                content = match.group(3).strip()

                # 解析时间
                ts = None
                for fmt in (
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M",
                ):
                    try:
                        ts = datetime.strptime(time_str, fmt).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue

                current_sender = sender
                current_time = ts
                if content:
                    current_text.append(content)
            else:
                # 没匹配到时间格式，当作上一条消息的续行
                if line:
                    current_text.append(line)

        # 处理最后一条
        if current_text:
            messages.append(Message(
                sender=current_sender or "未知",
                text="\n".join(current_text),
                timestamp=current_time,
                channel_name=source_label,
                platform="manual",
                evidence_level=EvidenceLevel.VERBATIM.value,
            ))

        # 如果一条都没解析出来（格式不对），
        # 退化成「按空行分段」的简单模式
        if len(messages) <= 1:
            paragraphs = re.split(r'\n\s*\n', text.strip())
            messages = []
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                messages.append(Message(
                    sender="未知",
                    text=para,
                    timestamp=None,
                    channel_name=source_label,
                    platform="manual",
                    evidence_level=EvidenceLevel.VERBATIM.value,
                ))

        logger.info(
            "[语料导入] 文本粘贴：%d 条消息", len(messages)
        )
        return messages

    # --------------------------------------------------------
    # 方式 2：从文件导入（自动识别后缀）
    # --------------------------------------------------------
    @classmethod
    def from_file(cls, file_path: str) -> List[Message]:
        """根据文件后缀自动选择解析方式"""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".json":
            return cls._from_json(path)
        if suffix == ".csv":
            return cls._from_csv(path)
        # .txt / .md 都按纯文本处理
        return cls._from_text_file(path)

    @classmethod
    def _from_text_file(cls, path: Path) -> List[Message]:
        """从纯文本/Markdown 文件导入"""
        content = path.read_text(encoding="utf-8")
        return cls.from_text(content, source_label=path.name)

    @classmethod
    def _from_json(cls, path: Path) -> List[Message]:
        """从 JSON 文件导入

        支持两种格式：
        1. 数组格式: [{sender, text, time}, ...]
        2. 对象格式: {messages: [...]}

        容错设计：自动尝试多种字段名
        （text/content/message, time/date/timestamp, sender/from）
        这样用户随便导出的格式大概率也能解析。
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("messages", [])

        messages: List[Message] = []
        for item in items:
            if isinstance(item, str):
                messages.append(Message(
                    sender="未知",
                    text=item,
                    platform="manual",
                    evidence_level=EvidenceLevel.VERBATIM.value,
                ))
                continue
            if not isinstance(item, dict):
                continue

            # 容错：尝试多种字段名
            text = (
                item.get("text")
                or item.get("content")
                or item.get("message")
                or item.get("msg", "")
            )
            if not text:
                continue

            sender = item.get("sender") or item.get("from", "未知")

            # 尝试解析时间
            ts = None
            ts_val = (
                item.get("timestamp")
                or item.get("date")
                or item.get("time")
                or item.get("datetime", "")
            )
            if isinstance(ts_val, str) and ts_val:
                ts = cls._parse_time(ts_val)

            messages.append(Message(
                sender=sender,
                text=text,
                timestamp=ts,
                channel_name=path.name,
                platform="manual",
                evidence_level=EvidenceLevel.VERBATIM.value,
            ))

        logger.info("[语料导入] JSON 文件 %s：%d 条消息", path.name, len(messages))
        return messages

    @classmethod
    def _from_csv(cls, path: Path) -> List[Message]:
        """从 CSV 文件导入

        预期列: sender, text, time
        容错：列名不区分大小写，自动匹配多种变体
        """
        messages: List[Message] = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 容错：不区分大小写地查找字段
                row_lower = {k.lower(): v for k, v in row.items()}
                text = (
                    row_lower.get("text")
                    or row_lower.get("content")
                    or row_lower.get("message")
                    or ""
                )
                if not text.strip():
                    continue

                sender = row_lower.get("sender") or row_lower.get("from", "未知")
                time_str = (
                    row_lower.get("time")
                    or row_lower.get("date")
                    or row_lower.get("timestamp")
                    or ""
                )
                ts = cls._parse_time(time_str) if time_str else None

                messages.append(Message(
                    sender=sender.strip(),
                    text=text.strip(),
                    timestamp=ts,
                    channel_name=path.name,
                    platform="manual",
                    evidence_level=EvidenceLevel.VERBATIM.value,
                ))

        logger.info("[语料导入] CSV 文件 %s：%d 条消息", path.name, len(messages))
        return messages

    # --------------------------------------------------------
    # 工具方法
    # --------------------------------------------------------
    @staticmethod
    def _parse_time(time_str: str) -> Optional[datetime]:
        """尝试多种格式解析时间字符串"""
        time_str = time_str.strip()
        formats = (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
        )
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None
