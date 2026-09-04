"""
微信聊天记录导入器 (WeChatImporter)

参考以下优秀开源项目实现：

1. LangChain WeChatChatLoader
   - 剪贴板复制格式解析（昵称+日期+时间同行）
   - regex: r"(?P<sender>.+?) (?P<timestamp>\\d{4}/\\d{2}/\\d{2} \\d{1,2}:\\d{2} (?:AM|PM))"
   - https://python.langchain.ac.cn/docs/integrations/chat_loaders/wechat/

2. immortal-skill collectors/wechat.py
   - parse_exported_csv(): WeChatMsg 导出 CSV 解析
   - parse_exported_txt(): 微信原生 TXT 导出解析
   - https://github.com/agenmod/immortal-skill

3. WeClone (18k stars)
   - PyWxDump CSV 列格式: type_name, is_sender, talker, msg, CreateTime
   - 数据预处理流程: CSV → 清洗 → QA 配对
   - https://github.com/xming521/WeClone

4. chatlog-keeper
   - WeChat 4.0+ 内存扫描提取，支持 4.1.x
   - https://github.com/labazhou2024/chatlog-keeper

导入方式：
1. 文件上传 — CSV / JSON / TXT（WeChatMsg/PyWxDump/chatlog-keeper 导出）
2. 剪贴板粘贴 — 从微信复制后粘贴（最简单，无需任何工具）
3. 手动输入 — 直接粘贴文本
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

from logger import logger
from persona.corpus_cleaner import CorpusCleaner, Message, EvidenceLevel


class WeChatImporter(CorpusCleaner):
    """微信聊天记录导入器

    参考 LangChain/immortal-skill/WeClone 的开源实现，
    支持多种格式导入，适配 WeChat 4.0+。

    用法：
        importer = WeChatImporter()

        # 方式1：从文件导入（CSV/JSON/TXT）
        messages = importer.parse_file(Path("wechat_export.csv"))

        # 方式2：从剪贴板文本导入（微信复制格式）
        messages = importer.parse(clipboard_text)

        # 方式3：直接解析文本
        messages = importer.parse("2025-01-15 14:30 张三\\n你好啊")
    """

    platform_name = "wechat"
    platform_name_zh = "微信"

    # ================================================================
    # 核心：微信剪贴板复制格式（参考 LangChain WeChatChatLoader）
    # ================================================================
    # 微信复制格式：昵称和日期时间在同一行，空行后是消息内容
    #
    # 示例：
    #   女朋友 2023/09/16 2:51 PM
    #
    #   天气有点凉
    #
    #   男朋友 2023/09/16 2:51 PM
    #
    #   珍簟凉风著，瑶琴寄恨生。
    #
    # 中文版微信可能使用：
    #   张三 2025/1/15 14:30          （24小时制）
    #   张三 2025/1/15 下午2:51       （中文AM/PM）
    #   张三 2025年1月15日 14:30      （中文日期）
    # ================================================================

    # LangChain 原始正则（英文 AM/PM）
    _RE_LANGCHAIN = re.compile(
        r"^(?P<sender>.+?)\s+"
        r"(?P<timestamp>\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}\s*(?:AM|PM))\s*$"
    )

    # 24小时制：张三 2025/1/15 14:30  或  张三 2025-01-15 14:30:00
    _RE_24H_SLASH = re.compile(
        r"^(?P<sender>.+?)\s+"
        r"(?P<timestamp>\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)\s*$"
    )

    # 中文日期：张三 2025年1月15日 14:30
    _RE_CN_DATE = re.compile(
        r"^(?P<sender>.+?)\s+"
        r"(?P<timestamp>\d{4}年\d{1,2}月\d{1,2}日\s+\d{1,2}:\d{2}(?::\d{2})?)\s*$"
    )

    # 中文AM/PM：张三 2025/1/15 下午2:51
    _RE_CN_AMPM = re.compile(
        r"^(?P<sender>.+?)\s+"
        r"(?P<timestamp>\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+"
        r"(?:上午|下午)\d{1,2}:\d{2}(?::\d{2})?)\s*$"
    )

    # 英文AM/PM变体：张三 2025/1/15 2:51 PM
    _RE_EN_AMPM = re.compile(
        r"^(?P<sender>.+?)\s+"
        r"(?P<timestamp>\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+"
        r"\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\s*$"
    )

    # ================================================================
    # immortal-skill TXT 格式（日期时间+发送者同行）
    # ================================================================
    # 2025-01-15 14:30:00 张三
    # 消息内容
    # ================================================================
    _RE_IMMORTAL_TXT = re.compile(
        r"^(\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+"
        r"\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)$"
    )

    # ================================================================
    # 简单聊天格式：[14:30] 张三: 消息
    # ================================================================
    _RE_BRACKET = re.compile(
        r"^\[(?:(?P<date>\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+)?"
        r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\]\s*"
        r"(?P<sender>.+?):\s*(?P<content>.*)$"
    )

    # ================================================================
    # CorpusCleaner 接口实现
    # ================================================================

    def parse(self, raw_data: Union[str, bytes]) -> List[Message]:
        """解析文本/字节流为 Message 列表

        自动检测格式：
        - JSON 字符串 → 按消息数组解析
        - CSV 文本 → 按列解析
        - 微信复制格式 → 按 LangChain 方案解析
        - immortal-skill TXT 格式 → 按 immortal-skill 方案解析
        - 纯文本 → 按行模式匹配
        """
        if isinstance(raw_data, bytes):
            try:
                raw_data = raw_data.decode("utf-8")
            except UnicodeDecodeError:
                raw_data = raw_data.decode("gbk", errors="ignore")

        raw_data = raw_data.strip()
        if not raw_data:
            return []

        # JSON 格式检测
        if raw_data.startswith("[") or raw_data.startswith("{"):
            return self._parse_json_text(raw_data)

        # CSV 格式检测（第一行有逗号 + 第二行也有逗号）
        lines = raw_data.split("\n")
        if len(lines) >= 2 and "," in lines[0] and "," in lines[1]:
            return self._parse_csv_text(raw_data)

        # 纯文本格式 → 自动检测子格式
        return self._parse_txt_text(raw_data)

    # ================================================================
    # 文件导入
    # ================================================================

    def parse_file(self, file_path: Union[str, Path]) -> List[Message]:
        """从文件导入聊天记录

        根据扩展名自动选择解析器：
        - .csv → CSV 解析（兼容 WeChatMsg/PyWxDump/chatlog-keeper）
        - .json → JSON 解析
        - .txt / 其他 → TXT 解析（兼容微信复制/immortal-skill 格式）
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        # 尝试多种编码（参考 immortal-skill 的做法）
        content = None
        for encoding in ["utf-8", "utf-8-sig", "gbk", "gb2312", "utf-16"]:
            try:
                content = path.read_text(encoding=encoding)
                logger.info("[微信导入] 文件编码: %s", encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            raise ValueError("无法识别文件编码，请转换为 UTF-8 后重试")

        ext = path.suffix.lower()
        logger.info("[微信导入] 文件: %s (%s, %d 字符)", path.name, ext, len(content))

        if ext == ".csv":
            return self._parse_csv_text(content)
        elif ext == ".json":
            return self._parse_json_text(content)
        else:
            return self._parse_txt_text(content)

    # ================================================================
    # CSV 格式解析
    # 参考：WeClone（PyWxDump CSV）、immortal-skill（WeChatMsg CSV）
    # ================================================================

    def _parse_csv_text(self, text: str) -> List[Message]:
        """解析 CSV 格式

        兼容多种工具导出的 CSV：
        - WeChatMsg: 发送者,内容,时间
        - PyWxDump/WeClone: type_name,is_sender,talker,msg,CreateTime
        - chatlog-keeper: 自定义列名
        """
        messages: List[Message] = []

        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)

        if not header:
            return []

        # 找到关键列的索引（兼容不同列名）
        col_map = self._find_csv_columns(header)

        if col_map["sender"] is None or col_map["content"] is None:
            logger.warning("[微信导入] CSV 列名不匹配，尝试按位置解析（前3列）")
            col_map = {"sender": 0, "content": 1, "time": 2, "is_sender": None}

        for row in reader:
            if len(row) <= max(col_map["sender"] or 0, col_map["content"] or 0):
                continue

            sender = row[col_map["sender"]].strip() if col_map["sender"] is not None and col_map["sender"] < len(row) else ""
            text_content = row[col_map["content"]].strip() if col_map["content"] is not None and col_map["content"] < len(row) else ""

            if not text_content:
                continue

            # 解析时间
            ts = None
            if col_map["time"] is not None and col_map["time"] < len(row):
                time_val = row[col_map["time"]].strip()
                ts = self._parse_time_str(time_val) or self._parse_timestamp(time_val)

            # 解析 is_sender（WeClone 格式有此字段）
            if col_map["is_sender"] is not None and col_map["is_sender"] < len(row):
                is_sender_val = row[col_map["is_sender"]].strip()
                if is_sender_val == "1":
                    sender = "我"
                elif is_sender_val == "0" and not sender:
                    sender = "对方"

            messages.append(Message(
                sender=sender or "未知",
                text=text_content,
                timestamp=ts,
                channel_name="微信聊天",
                platform="wechat",
                evidence_level=EvidenceLevel.VERBATIM.value,
            ))

        logger.info("[微信导入] CSV 解析: %d 条消息", len(messages))
        return messages

    @staticmethod
    def _find_csv_columns(header: List[str]) -> dict:
        """从 CSV 表头找出各字段列索引

        兼容列名：
        - 发送者: 发送者, 昵称, name, sender, talker, from, 对方
        - 内容: 内容, 消息, message, content, text, msg, 正文
        - 时间: 时间, 日期, date, time, timestamp, createTime, 时间戳
        - 是否本人: is_sender, IsSender, isSend, IsSelf
        """
        result = {"sender": None, "content": None, "time": None, "is_sender": None}

        sender_keywords = {"发送者", "昵称", "name", "sender", "talker", "from", "对方"}
        content_keywords = {"内容", "消息", "message", "content", "text", "msg", "正文"}
        time_keywords = {"时间", "日期", "date", "time", "timestamp", "createtime", "时间戳"}
        is_sender_keywords = {"is_sender", "issender", "issend", "isself"}

        for i, col in enumerate(header):
            col_lower = col.strip().lower()
            if result["sender"] is None and (col_lower in sender_keywords or any(k in col_lower for k in sender_keywords)):
                result["sender"] = i
            elif result["content"] is None and (col_lower in content_keywords or any(k in col_lower for k in content_keywords)):
                result["content"] = i
            elif result["time"] is None and (col_lower in time_keywords or any(k in col_lower for k in time_keywords)):
                result["time"] = i
            elif result["is_sender"] is None and (col_lower in is_sender_keywords or any(k in col_lower for k in is_sender_keywords)):
                result["is_sender"] = i

        return result

    # ================================================================
    # JSON 格式解析
    # ================================================================

    def _parse_json_text(self, text: str) -> List[Message]:
        """解析 JSON 格式

        兼容：
        - WeChatMsg JSON: [{"sender": "...", "content": "...", "createTime": ...}]
        - Wechat-exporter JSON: 嵌套结构
        - chatlog-keeper JSON: 自定义结构
        """
        messages: List[Message] = []

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("[微信导入] JSON 解析失败")
            return []

        # 统一成列表
        if isinstance(data, dict):
            if "messages" in data:
                data = data["messages"]
            elif "msg_list" in data:
                data = data["msg_list"]
            elif "data" in data and isinstance(data["data"], list):
                data = data["data"]
            else:
                data = [data]

        if not isinstance(data, list):
            return []

        for item in data:
            if not isinstance(item, dict):
                continue

            # 兼容不同字段名
            sender = (
                item.get("sender") or item.get("name") or item.get("nickname")
                or item.get("talker") or item.get("from") or "未知"
            )

            content = (
                item.get("content") or item.get("text") or item.get("message")
                or item.get("msg") or item.get("msg_content") or ""
            )

            if not content.strip():
                continue

            # is_sender 字段（WeClone 格式）
            is_sender = item.get("is_sender") or item.get("IsSender")
            if is_sender == 1 or is_sender == "1":
                sender = "我"
            elif is_sender == 0 or is_sender == "0":
                if not sender or sender == "未知":
                    sender = "对方"

            # 解析时间
            ts = None
            time_val = (
                item.get("createTime") or item.get("timestamp")
                or item.get("time") or item.get("date") or item.get("CreateTime")
            )
            if time_val:
                ts = self._parse_timestamp(time_val)
                if ts is None and isinstance(time_val, str):
                    ts = self._parse_time_str(time_val)

            messages.append(Message(
                sender=str(sender),
                text=str(content).strip(),
                timestamp=ts,
                channel_name="微信聊天",
                platform="wechat",
                evidence_level=EvidenceLevel.VERBATIM.value,
            ))

        logger.info("[微信导入] JSON 解析: %d 条消息", len(messages))
        return messages

    # ================================================================
    # TXT 格式解析（核心：微信复制格式）
    # ================================================================

    def _parse_txt_text(self, text: str) -> List[Message]:
        """解析纯文本格式

        自动检测子格式：
        1. 微信复制格式（参考 LangChain）: 昵称 日期时间 → 空行 → 内容
        2. immortal-skill 格式: 日期时间 发送者 → 内容
        3. 方括号格式: [14:30] 张三: 消息
        4. 简单格式: 昵称: 消息
        5. 兜底: 不规则文本
        """
        lines = text.strip().split("\n")
        lines = [l.rstrip("\r") for l in lines]

        # 先检测微信复制格式（最常见）
        if self._detect_wechat_copy(lines):
            return self._parse_wechat_copy(lines)

        # immortal-skill 格式
        if any(self._RE_IMMORTAL_TXT.match(line.strip()) for line in lines[:20]):
            return self._parse_immortal_txt(lines)

        # 方括号格式
        if any(self._RE_BRACKET.match(line.strip()) for line in lines[:20]):
            return self._parse_bracket(lines)

        # 兜底
        return self._parse_loose(lines)

    def _detect_wechat_copy(self, lines: List[str]) -> bool:
        """检测是否为微信复制格式

        判断逻辑：前20行中是否有匹配
        "昵称 日期 时间" 格式的行
        """
        patterns = [
            self._RE_LANGCHAIN,
            self._RE_24H_SLASH,
            self._RE_CN_DATE,
            self._RE_CN_AMPM,
            self._RE_EN_AMPM,
        ]
        count = 0
        for line in lines[:30]:
            line = line.strip()
            for pat in patterns:
                if pat.match(line):
                    count += 1
                    break
        return count >= 2

    def _parse_wechat_copy(self, lines: List[str]) -> List[Message]:
        """解析微信复制格式（参考 LangChain WeChatChatLoader）

        格式：
            昵称 日期 时间
            （空行）
            消息内容（可能多行）
            （空行）
            昵称 日期 时间
            ...
        """
        messages: List[Message] = []
        patterns = [
            self._RE_LANGCHAIN,
            self._RE_24H_SLASH,
            self._RE_CN_DATE,
            self._RE_CN_AMPM,
            self._RE_EN_AMPM,
        ]

        current_sender = ""
        current_ts: Optional[datetime] = None
        current_lines: List[str] = []

        for line in lines:
            line_stripped = line.strip()

            # 尝试匹配 "昵称 日期 时间" 行
            matched = False
            for pat in patterns:
                m = pat.match(line_stripped)
                if m:
                    # 保存上一条消息
                    if current_lines and current_sender:
                        messages.append(self._make_message(
                            current_sender,
                            "\n".join(current_lines),
                            current_ts,
                        ))

                    current_sender = m.group("sender").strip()
                    timestamp_str = m.group("timestamp").strip()
                    current_ts = self._parse_time_str(timestamp_str)
                    current_lines = []
                    matched = True
                    break

            if matched:
                continue

            # 空行 → 可能是消息分隔
            if not line_stripped:
                if current_lines and current_sender:
                    messages.append(self._make_message(
                        current_sender,
                        "\n".join(current_lines),
                        current_ts,
                    ))
                    current_lines = []
                    current_sender = ""
                    current_ts = None
                continue

            # 内容行
            current_lines.append(line_stripped)

        # 最后一条
        if current_lines and current_sender:
            messages.append(self._make_message(
                current_sender,
                "\n".join(current_lines),
                current_ts,
            ))

        logger.info("[微信导入] 微信复制格式解析: %d 条消息", len(messages))
        return messages

    def _parse_immortal_txt(self, lines: List[str]) -> List[Message]:
        """解析 immortal-skill TXT 格式

        格式：
            2025-01-15 14:30:00 张三
            消息内容
        """
        messages: List[Message] = []
        current_sender = ""
        current_ts: Optional[datetime] = None
        current_lines: List[str] = []

        for line in lines:
            m = self._RE_IMMORTAL_TXT.match(line.strip())
            if m:
                # 保存上一条
                if current_lines and current_sender:
                    messages.append(self._make_message(
                        current_sender,
                        "\n".join(current_lines),
                        current_ts,
                    ))

                time_str = m.group(1)
                current_sender = m.group(2).strip()
                current_lines = []
                current_ts = self._parse_time_str(time_str)
            else:
                stripped = line.strip()
                if stripped:
                    current_lines.append(stripped)

        # 最后一条
        if current_lines and current_sender:
            messages.append(self._make_message(
                current_sender,
                "\n".join(current_lines),
                current_ts,
            ))

        logger.info("[微信导入] immortal-skill TXT 格式解析: %d 条消息", len(messages))
        return messages

    def _parse_bracket(self, lines: List[str]) -> List[Message]:
        """解析方括号格式: [14:30] 张三: 消息"""
        messages: List[Message] = []

        for line in lines:
            m = self._RE_BRACKET.match(line.strip())
            if m:
                date_str = m.group("date") or ""
                time_str = m.group("time")
                sender = m.group("sender").strip()
                content = m.group("content").strip()

                if not content:
                    continue

                full_time = f"{date_str} {time_str}".strip() if date_str else time_str
                ts = self._parse_time_str(full_time) if full_time else None

                messages.append(self._make_message(sender, content, ts))

        logger.info("[微信导入] 方括号格式解析: %d 条消息", len(messages))
        return messages

    def _parse_loose(self, lines: List[str]) -> List[Message]:
        """兜底解析：不规则文本

        策略：
        - 找 "昵称: 内容" 或 "昵称：内容" 格式
        - 找不到就整段当一个消息
        """
        messages: List[Message] = []
        pattern = re.compile(r"^(.{1,20})[:：]\s*(.+)$")

        for line in lines:
            m = pattern.match(line.strip())
            if m:
                sender = m.group(1).strip()
                content = m.group(2).strip()
                if content:
                    messages.append(self._make_message(sender, content, None))

        if not messages and lines:
            full_text = "\n".join(l.strip() for l in lines if l.strip())
            if full_text:
                messages.append(self._make_message("未知", full_text, None))

        logger.info("[微信导入] 松散解析: %d 条消息", len(messages))
        return messages

    # ================================================================
    # 剪贴板读取（可选功能）
    # ================================================================

    @staticmethod
    def get_clipboard_text() -> str:
        """读取当前剪贴板文本"""
        try:
            import win32clipboard

            win32clipboard.OpenClipboard()
            if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
                text = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
                win32clipboard.CloseClipboard()
                return text
            win32clipboard.CloseClipboard()
            return ""
        except Exception as e:
            logger.warning("[微信导入] 读取剪贴板失败: %s", e)
            try:
                win32clipboard.CloseClipboard()
            except Exception:
                pass
            return ""

    # ================================================================
    # 工具方法
    # ================================================================

    @staticmethod
    def _parse_time_str(time_str: str) -> Optional[datetime]:
        """解析时间字符串，支持多种格式

        支持：
        - 2023/09/16 2:51 PM（英文AM/PM，参考 LangChain）
        - 2025/1/15 14:30（24小时制）
        - 2025-01-15 14:30:00（ISO格式）
        - 2025年1月15日 14:30（中文日期）
        - 2025/1/15 下午2:51（中文AM/PM）
        - 纯时间 14:30（用今天日期补全）
        """
        if not time_str:
            return None

        time_str = time_str.strip()

        # 处理中文 AM/PM
        cn_pm = False
        cn_am = False
        if "下午" in time_str or "PM" in time_str or "pm" in time_str:
            cn_pm = True
            time_str = time_str.replace("下午", "").replace("PM", "").replace("pm", "").strip()
        elif "上午" in time_str or "AM" in time_str or "am" in time_str:
            cn_am = True
            time_str = time_str.replace("上午", "").replace("AM", "").replace("am", "").strip()

        # 中文日期转斜杠
        time_str = time_str.replace("年", "/").replace("月", "/").replace("日", "")

        formats = [
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %I:%M",  # 12小时制
            "%m-%d %H:%M",
            "%m月%d日 %H:%M",
            "%H:%M:%S",
            "%H:%M",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(time_str, fmt)
                # 12小时制 AM/PM 处理
                if cn_pm and dt.hour < 12:
                    dt = dt.replace(hour=dt.hour + 12)
                elif cn_am and dt.hour == 12:
                    dt = dt.replace(hour=0)
                # 如果只有时间没有日期，用今天的日期
                if "%Y" not in fmt:
                    now = datetime.now()
                    dt = dt.replace(year=now.year, month=now.month, day=now.day)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        return None

    @staticmethod
    def _parse_timestamp(ts) -> Optional[datetime]:
        """解析时间戳（秒或毫秒）"""
        if isinstance(ts, str):
            try:
                ts = int(ts)
            except ValueError:
                return None
        if not isinstance(ts, (int, float)):
            return None

        # 毫秒时间戳
        if ts > 1e12:
            ts = ts / 1000

        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OSError):
            return None

    @staticmethod
    def _make_message(
        sender: str, text: str, ts: Optional[datetime]
    ) -> Message:
        """构造 Message 对象"""
        return Message(
            sender=sender or "未知",
            text=text.strip(),
            timestamp=ts,
            channel_name="微信聊天",
            platform="wechat",
            evidence_level=EvidenceLevel.VERBATIM.value,
        )
