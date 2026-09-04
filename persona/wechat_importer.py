"""
微信本地数据库导入器 (WeChatDBImporter)

利用 PyWxDump 工具读取本地微信数据库，导出聊天记录为统一 Message 格式。

使用前提：
1. 微信电脑版已登录并运行
2. 已安装 pywxdump 包（pip install pywxdump）

工作流程：
1. pywxdump info → 获取微信信息、数据库密钥、数据库路径
2. pywxdump decrypt → 解密 MSG.db
3. sqlite3 读取解密后的数据库 → 查询消息
4. 转换为统一 Message 格式
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict

from logger import logger
from persona.corpus_cleaner import CorpusCleaner, Message, EvidenceLevel


@dataclass
class WeChatInfo:
    """微信信息（从 pywxdump info 获取）"""
    wxid: str = ""           # 微信 ID
    nickname: str = ""       # 昵称
    mobile: str = ""         # 手机号
    key: str = ""            # 数据库密钥
    db_path: str = ""        # 数据库目录路径
    version: str = ""        # 微信版本


class WeChatDBImporter(CorpusCleaner):
    """微信本地数据库导入器

    直接从微信电脑版的本地数据库读取聊天记录，
    不需要用户手动导出，一键获取所有消息。

    用法：
        importer = WeChatDBImporter()
        contacts = importer.list_contacts()  # 列出所有联系人
        messages = importer.get_messages(wxid="wxid_xxx")  # 获取某个人的消息
    """

    platform_name = "wechat-db"
    platform_name_zh = "微信本地数据库"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self._wx_info: Optional[WeChatInfo] = None
        self._decrypted_dir: Optional[Path] = None
        self._pywxdump_cli = self._find_pywxdump_cli()

    # ============================================================
    # 公共 API
    # ============================================================

    def get_wechat_info(self) -> WeChatInfo:
        """获取微信信息（ID、密钥、数据库路径等）

        缓存结果，重复调用不会重复执行。
        """
        if self._wx_info and self._wx_info.key:
            return self._wx_info

        if not self._pywxdump_cli:
            raise RuntimeError("未找到 pywxdump，请先安装：pip install pywxdump")

        logger.info("[微信导入] 正在获取微信信息...")
        output = self._run_pywxdump(["info"])
        self._wx_info = self._parse_info_output(output)

        if not self._wx_info.key:
            raise RuntimeError(
                "未能获取微信数据库密钥，请确保微信电脑版已登录并运行。\n"
                "如果微信已启动仍失败，可能是微信版本不兼容。"
            )

        logger.info(
            "[微信导入] 微信信息获取成功：%s (%s)",
            self._wx_info.nickname or self._wx_info.wxid,
            self._wx_info.wxid,
        )
        return self._wx_info

    def list_contacts(self) -> List[Dict[str, str]]:
        """列出所有聊天联系人（按消息数量排序）

        返回: [{"wxid": "...", "nickname": "...", "msg_count": N}, ...]
        """
        info = self.get_wechat_info()
        db_path = self._get_msg_db_path()

        if not db_path.exists():
            raise RuntimeError(f"MSG 数据库不存在：{db_path}")

        # 解密数据库
        decrypted_db = self._decrypt_db(db_path)

        # 查询所有会话（按消息数量排序）
        conn = sqlite3.connect(str(decrypted_db))
        cursor = conn.cursor()

        # 微信 MSG 表中，StrTalker 是对方的 wxid
        # 我们统计每个 StrTalker 的消息数量
        try:
            cursor.execute("""
                SELECT StrTalker, COUNT(*) as cnt
                FROM MSG
                WHERE Type IN (1, 10000)  -- 文本消息和系统消息
                GROUP BY StrTalker
                ORDER BY cnt DESC
            """)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("[微信导入] MSG 表查询失败: %s，尝试其他表结构", e)
            # 可能表名或列名不同，试试其他常见结构
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]
            logger.info("[微信导入] 数据库表: %s", tables)
            raise RuntimeError(f"MSG 表结构不兼容，可用表：{tables}")

        conn.close()

        # 尝试获取联系人昵称（从 MicroMsg.db 的 Contact 表）
        nicknames = self._get_contact_nicknames()

        contacts = []
        for wxid, count in rows:
            # 过滤掉自己和群聊（简单过滤：群聊通常以 @chatroom 结尾）
            if wxid == info.wxid:
                continue
            if "@chatroom" in wxid:
                # 群聊也先跳过，专注于私聊
                continue

            nickname = nicknames.get(wxid, "")
            contacts.append({
                "wxid": wxid,
                "nickname": nickname or wxid,
                "msg_count": count,
            })

        logger.info("[微信导入] 找到 %d 个私聊联系人", len(contacts))
        return contacts

    def get_messages(
        self,
        wxid: str,
        limit: Optional[int] = None,
        nickname: str = "",
    ) -> List[Message]:
        """获取指定联系人的聊天记录

        Args:
            wxid: 对方的微信 ID
            limit: 最多获取多少条消息（None 表示全部）
            nickname: 对方昵称（用于显示）

        Returns:
            Message 列表，按时间升序排列
        """
        info = self.get_wechat_info()
        db_path = self._get_msg_db_path()

        if not db_path.exists():
            raise RuntimeError(f"MSG 数据库不存在：{db_path}")

        # 解密数据库
        decrypted_db = self._decrypt_db(db_path)

        # 查询消息
        conn = sqlite3.connect(str(decrypted_db))
        cursor = conn.cursor()

        # 微信 MSG 表关键字段：
        #   MsgSvrID - 消息 ID
        #   CreateTime - 创建时间（秒级时间戳）
        #   StrTalker - 对话者 wxid
        #   IsSender - 是否是自己发送的（1=自己发的，0=对方发的）
        #   Content - 消息内容
        #   Type - 消息类型（1=文本，3=图片，34=语音，43=视频，47=表情，10000=系统消息等）
        #   DisplayContent - 显示内容（有时 Content 是 XML，DisplayContent 是纯文本）
        query = """
            SELECT
                CreateTime,
                IsSender,
                Content,
                DisplayContent,
                Type
            FROM MSG
            WHERE StrTalker = ?
            AND Type IN (1, 10000, 47, 1052, 1053)
            ORDER BY CreateTime ASC
        """
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, (wxid,))
        rows = cursor.fetchall()
        conn.close()

        # 转换为 Message 格式
        messages: List[Message] = []
        my_name = info.nickname or "我"
        other_name = nickname or wxid

        for create_time, is_sender, content, display_content, msg_type in rows:
            # 选择可用的文本内容
            text = content or display_content or ""
            text = text.strip()

            # 过滤掉空消息和 XML 系统消息
            if not text:
                continue
            if text.startswith("<?xml"):
                # XML 格式的消息（图片、视频、红包等），提取显示内容
                if display_content and display_content.strip():
                    text = display_content.strip()
                else:
                    # 用类型标记替代
                    type_label = self._msg_type_label(msg_type)
                    if type_label:
                        text = f"[{type_label}]"
                    else:
                        continue

            # 解析时间（CreateTime 是秒级时间戳）
            ts = None
            try:
                ts = datetime.fromtimestamp(create_time, tz=timezone.utc)
            except (ValueError, OSError):
                pass

            # 判断发送者
            sender = my_name if is_sender == 1 else other_name
            msg_type_str = self._msg_type_label(msg_type) or "text"

            messages.append(Message(
                sender=sender,
                text=text,
                timestamp=ts,
                channel_name=f"{other_name}的私聊",
                msg_type=msg_type_str,
                platform="wechat-db",
                evidence_level=EvidenceLevel.VERBATIM.value,
                raw={
                    "wxid": wxid,
                    "msg_type_code": msg_type,
                    "is_sender": is_sender,
                },
            ))

        logger.info(
            "[微信导入] 获取 %s 的聊天记录：%d 条",
            other_name, len(messages),
        )
        return messages

    # ============================================================
    # CorpusCleaner 接口实现
    # ============================================================

    def parse(self, raw_data) -> List[Message]:
        """实现 CorpusCleaner 的 parse 接口

        raw_data 可以是：
        - str: 对方的 wxid
        - dict: {"wxid": "...", "limit": N, "nickname": "..."}
        """
        if isinstance(raw_data, str):
            return self.get_messages(raw_data)
        if isinstance(raw_data, dict):
            return self.get_messages(
                wxid=raw_data["wxid"],
                limit=raw_data.get("limit"),
                nickname=raw_data.get("nickname", ""),
            )
        raise ValueError(f"不支持的数据类型: {type(raw_data)}")

    # ============================================================
    # 内部工具方法
    # ============================================================

    @staticmethod
    def _find_pywxdump_cli() -> Optional[str]:
        """找到 pywxdump 的 cli.py 路径"""
        # 先找当前虚拟环境里的
        venv_cli = Path(__file__).parent.parent / ".venv" / "Lib" / "site-packages" / "pywxdump" / "cli.py"
        if venv_cli.exists():
            return str(venv_cli)

        # 找系统 Python 的
        try:
            import pywxdump
            return str(Path(pywxdump.__file__).parent / "cli.py")
        except ImportError:
            pass

        # 用 python -c 查找
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import pywxdump; print(pywxdump.__file__)"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                cli_path = Path(result.stdout.strip()).parent / "cli.py"
                if cli_path.exists():
                    return str(cli_path)
        except Exception:
            pass

        return None

    def _run_pywxdump(self, args: List[str]) -> str:
        """运行 pywxdump 命令并返回输出"""
        cmd = [sys.executable, self._pywxdump_cli] + args
        logger.debug("[微信导入] 执行命令: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        # pywxdump 输出到 stdout 和 stderr 都有可能
        output = result.stdout + result.stderr
        return output

    @staticmethod
    def _parse_info_output(output: str) -> WeChatInfo:
        """解析 pywxdump info 的输出"""
        info = WeChatInfo()

        # 尝试从输出中提取关键字段
        patterns = {
            "wxid": [r"微信ID[：:]\s*(\S+)", r"wxid[：:]\s*(\S+)"],
            "nickname": [r"昵称[：:]\s*(\S+)", r"nickname[：:]\s*(\S+)"],
            "mobile": [r"手机号[：:]\s*(\S+)", r"mobile[：:]\s*(\S+)"],
            "key": [r"密钥[：:]\s*([0-9a-fA-F]+)", r"key[：:]\s*([0-9a-fA-F]+)"],
            "db_path": [r"数据库路径[：:]\s*(\S+)", r"db_path[：:]\s*(\S+)", r"路径[：:]\s*(\S+)"],
            "version": [r"微信版本[：:]\s*(\S+)", r"version[：:]\s*(\S+)"],
        }

        for field, pats in patterns.items():
            for pat in pats:
                match = re.search(pat, output)
                if match:
                    setattr(info, field, match.group(1).strip())
                    break

        return info

    def _get_msg_db_path(self) -> Path:
        """获取 MSG.db 的路径"""
        info = self.get_wechat_info()
        db_dir = Path(info.db_path)

        # 微信数据库通常在 Msg 目录下，文件名是 MSG.db 或 MSG0.db, MSG1.db 等
        # 先找最常见的
        msg_dir = db_dir / "Msg"
        if msg_dir.exists():
            # 找 MSG.db 或 MSG*.db
            for f in msg_dir.glob("MSG*.db"):
                if f.is_file():
                    return f

        # 也可能直接在 db_path 下
        for f in db_dir.glob("MSG*.db"):
            if f.is_file():
                return f

        # 再试试子目录
        for sub in db_dir.iterdir():
            if sub.is_dir():
                for f in sub.glob("MSG*.db"):
                    if f.is_file():
                        return f

        raise FileNotFoundError(f"未找到 MSG 数据库，路径：{db_dir}")

    def _decrypt_db(self, db_path: Path) -> Path:
        """解密微信数据库

        如果已经解密过，直接返回缓存的路径。
        """
        info = self.get_wechat_info()

        # 解密输出目录
        if not self._decrypted_dir:
            self._decrypted_dir = Path(tempfile.mkdtemp(prefix="wechat_decrypted_"))

        out_db = self._decrypted_dir / db_path.name
        if out_db.exists():
            return out_db

        logger.info("[微信导入] 正在解密数据库：%s", db_path.name)
        self._run_pywxdump([
            "decrypt",
            "-k", info.key,
            "-i", str(db_path),
            "-o", str(self._decrypted_dir),
        ])

        if not out_db.exists():
            # 可能输出在子目录里
            for f in self._decrypted_dir.rglob(db_path.name):
                if f.is_file():
                    return f
            raise RuntimeError(f"数据库解密失败：{db_path}")

        return out_db

    def _get_contact_nicknames(self) -> Dict[str, str]:
        """从 MicroMsg.db 获取联系人昵称映射"""
        info = self.get_wechat_info()
        db_dir = Path(info.db_path)

        # 找 MicroMsg.db
        micro_msg_db = None
        for pattern in ["MicroMsg*.db", "microMsg*.db"]:
            for f in db_dir.glob(pattern):
                if f.is_file():
                    micro_msg_db = f
                    break
            if micro_msg_db:
                break

        # 也可能在子目录
        if not micro_msg_db:
            for sub in db_dir.iterdir():
                if sub.is_dir():
                    for pattern in ["MicroMsg*.db", "microMsg*.db"]:
                        for f in sub.glob(pattern):
                            if f.is_file():
                                micro_msg_db = f
                                break
                        if micro_msg_db:
                            break
                    if micro_msg_db:
                        break

        if not micro_msg_db:
            logger.info("[微信导入] 未找到 MicroMsg.db，无法获取联系人昵称")
            return {}

        # 解密
        try:
            decrypted_db = self._decrypt_db(micro_msg_db)
        except Exception as e:
            logger.warning("[微信导入] MicroMsg.db 解密失败: %s", e)
            return {}

        # 查询联系人
        nicknames: Dict[str, str] = {}
        try:
            conn = sqlite3.connect(str(decrypted_db))
            cursor = conn.cursor()

            # 常见的联系人表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {r[0] for r in cursor.fetchall()}

            contact_table = None
            for candidate in ["Contact", "contact", "rcontact", "Rcontact"]:
                if candidate in tables:
                    contact_table = candidate
                    break

            if contact_table:
                # 尝试常见列名
                cursor.execute(f"PRAGMA table_info({contact_table})")
                columns = {r[1] for r in cursor.fetchall()}

                wxid_col = None
                nickname_col = None
                for col in columns:
                    if col.lower() in ("username", "wxid", "UserName"):
                        wxid_col = col
                    elif col.lower() in ("nickname", "NickName", "remark", "Remark"):
                        nickname_col = col

                if wxid_col and nickname_col:
                    cursor.execute(f"SELECT {wxid_col}, {nickname_col} FROM {contact_table}")
                    for wxid, nick in cursor.fetchall():
                        if nick:
                            nicknames[wxid] = nick
                elif wxid_col:
                    cursor.execute(f"SELECT {wxid_col} FROM {contact_table}")
                    for (wxid,) in cursor.fetchall():
                        nicknames[wxid] = wxid

            conn.close()
        except sqlite3.Error as e:
            logger.warning("[微信导入] 查询联系人失败: %s", e)

        logger.info("[微信导入] 获取到 %d 个联系人昵称", len(nicknames))
        return nicknames

    @staticmethod
    def _msg_type_label(type_code: int) -> str:
        """微信消息类型转文本标签"""
        type_map = {
            1: "text",       # 文本
            3: "image",      # 图片
            34: "voice",     # 语音
            43: "video",     # 视频
            47: "emoji",     # 表情
            49: "link",      # 链接/小程序/文件
            10000: "system",  # 系统消息
            1052: "text",     # 引用/回复
            1053: "text",     # 编辑过的消息
        }
        return type_map.get(type_code, "text")
