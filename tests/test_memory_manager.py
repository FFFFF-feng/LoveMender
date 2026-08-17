"""
test_memory_manager.py
======================
测试 memory_manager.py 的记忆管理逻辑
- 短期记忆: 增删、长度限制、压缩触发
- 长期记忆: 摘要存储、相关检索
- 消息组装: SystemMessage / HumanMessage / AIMessage
- 会话统计: 轮次、使用比例
- 会话重置: 清空短期、保留长期
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from memory_manager import MemoryManager


# ==================== 短期记忆: 增删与长度 ====================

class TestAddExchange:
    """测试对话追加功能"""

    def test_single_exchange(self, mock_llm, mock_embedding, tmp_chroma_db):
        """一轮对话 → 短期记忆 2 条"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.add_exchange("你好", "你好呀")

        assert len(mm.short_term_memory) == 2
        assert mm.short_term_memory[0]["role"] == "human"
        assert mm.short_term_memory[0]["content"] == "你好"
        assert mm.short_term_memory[1]["role"] == "ai"
        assert mm.short_term_memory[1]["content"] == "你好呀"

    def test_multiple_exchanges(self, mock_llm, mock_embedding, tmp_chroma_db):
        """多轮对话 → 短期记忆按序追加"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.add_exchange("第一轮", "回复1")
        mm.add_exchange("第二轮", "回复2")
        mm.add_exchange("第三轮", "回复3")

        assert len(mm.short_term_memory) == 6
        assert mm.short_term_memory[0]["content"] == "第一轮"
        assert mm.short_term_memory[-1]["content"] == "回复3"

    def test_empty_text_still_added(self, mock_llm, mock_embedding, tmp_chroma_db):
        """空文本也能追加(不报错)"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.add_exchange("", "")
        assert len(mm.short_term_memory) == 2


# ==================== 压缩机制 ====================

class TestCompression:
    """测试短期记忆超限时的压缩机制"""

    def test_compression_triggers_at_limit(self, mock_llm, mock_embedding, tmp_chroma_db):
        """超过 max_history_len → 触发压缩"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.max_history_len = 4  # 设置小限制: 2 轮就触发压缩

        mm.add_exchange("第一轮用户", "第一轮AI")
        # 此时 short_term_memory 有 2 条,未超限
        assert len(mm.short_term_memory) == 2

        mm.add_exchange("第二轮用户", "第二轮AI")
        # 此时 short_term_memory 有 4 条,未超限(等于不触发)
        assert len(mm.short_term_memory) == 4

        mm.add_exchange("第三轮用户", "第三轮AI")
        # 此时 short_term_memory 有 6 条 > 4,触发压缩
        # 压缩后: 删除 4 条旧消息,插入 1 条摘要 → 6 - 4 + 1 = 3 条
        assert len(mm.short_term_memory) == 3
        # 摘要应插入到最前面
        assert mm.short_term_memory[0]["role"] == "system"

    def test_compression_calls_llm(self, mock_llm, mock_embedding, tmp_chroma_db):
        """压缩时调用 LLM 生成摘要"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.max_history_len = 4

        # 添加足够多的对话触发压缩
        for i in range(3):
            mm.add_exchange(f"用户{i}", f"AI{i}")

        # LLM 应被调用至少 1 次(用于生成摘要)
        assert mock_llm.invoke_count >= 1

    def test_compression_summary_stored_in_long_term(self, mock_llm, mock_embedding, tmp_chroma_db):
        """压缩产生的摘要存入长期记忆"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.max_history_len = 4

        for i in range(3):
            mm.add_exchange(f"用户消息{i}", f"AI回复{i}")

        # 验证长期记忆集合中有数据
        count = mm.long_term_collection.count()
        assert count >= 1

    def test_compression_llm_failure_graceful(self, mock_embedding, tmp_chroma_db):
        """LLM 调用失败时降级处理(不崩溃)"""
        failing_llm = MagicMock()
        failing_llm.invoke.side_effect = Exception("API error")

        mm = MemoryManager(failing_llm, mock_embedding)
        mm.max_history_len = 4

        # 应该不报错
        for i in range(3):
            mm.add_exchange(f"用户{i}", f"AI{i}")

        # 摘要应为降级文本
        assert mm.short_term_memory[0]["role"] == "system"
        assert "失败" in mm.short_term_memory[0]["content"]


# ==================== 长期记忆检索 ====================

class TestRetrieveRelevantMemory:
    """测试长期记忆检索"""

    def test_retrieve_after_compression(self, mock_llm, mock_embedding, tmp_chroma_db):
        """压缩后能检索到相关摘要"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.max_history_len = 4

        # 添加对话触发压缩,摘要存入长期记忆
        mm.add_exchange("我今天很焦虑", "试试深呼吸")
        mm.add_exchange("工作压力很大", "可以适当休息")
        mm.add_exchange("睡不好觉", "睡前放松一下")

        # 检索长期记忆
        result = mm.retrieve_relevant_memory("焦虑怎么办")
        assert isinstance(result, str)

    def test_retrieve_empty_query(self, mock_llm, mock_embedding, tmp_chroma_db):
        """空查询 → 返回空字符串"""
        mm = MemoryManager(mock_llm, mock_embedding)
        result = mm.retrieve_relevant_memory("")
        assert result == ""

    def test_retrieve_whitespace_query(self, mock_llm, mock_embedding, tmp_chroma_db):
        """纯空格查询 → 返回空字符串"""
        mm = MemoryManager(mock_llm, mock_embedding)
        result = mm.retrieve_relevant_memory("   ")
        assert result == ""

    def test_retrieve_no_long_term_memory(self, mock_llm, mock_embedding, tmp_chroma_db):
        """长期记忆为空 → 返回空字符串"""
        mm = MemoryManager(mock_llm, mock_embedding)
        result = mm.retrieve_relevant_memory("测试查询")
        assert result == ""


# ==================== 消息组装 ====================

class TestBuildMessages:
    """测试消息组装为 LangChain 格式"""

    def test_basic_message_structure(self, mock_llm, mock_embedding, tmp_chroma_db):
        """组装的消息包含 SystemMessage + 当前输入"""
        mm = MemoryManager(mock_llm, mock_embedding)
        messages = mm.build_messages("你是助手", "用户问题")

        # 第一个是 SystemMessage,最后一个是 HumanMessage
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[-1], HumanMessage)
        assert messages[-1].content == "用户问题"

    def test_messages_include_history(self, mock_llm, mock_embedding, tmp_chroma_db):
        """组装的消息包含历史对话"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.add_exchange("历史问题", "历史回答")

        messages = mm.build_messages("系统提示", "当前问题")

        # SystemMessage + HumanMessage(历史) + AIMessage(历史) + HumanMessage(当前)
        assert len(messages) == 4
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "历史问题"
        assert isinstance(messages[2], AIMessage)
        assert messages[2].content == "历史回答"

    def test_messages_with_compressed_summary(self, mock_llm, mock_embedding, tmp_chroma_db):
        """压缩后的摘要作为 SystemMessage 出现在历史中"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.max_history_len = 4

        # 触发压缩
        for i in range(3):
            mm.add_exchange(f"用户{i}", f"AI{i}")

        messages = mm.build_messages("系统提示", "当前问题")

        # 应包含至少 2 个 SystemMessage(系统提示 + 摘要)
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(system_msgs) >= 2


class TestGetChatHistoryMessages:
    """测试获取历史消息(供 AgentExecutor 使用)"""

    def test_empty_history(self, mock_llm, mock_embedding, tmp_chroma_db):
        """无历史 → 返回空列表"""
        mm = MemoryManager(mock_llm, mock_embedding)
        messages = mm.get_chat_history_messages()
        assert messages == []

    def test_history_excludes_current_input(self, mock_llm, mock_embedding, tmp_chroma_db):
        """历史消息不包含当前输入"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.add_exchange("问题1", "回答1")

        messages = mm.get_chat_history_messages()
        assert len(messages) == 2
        assert "当前" not in messages[-1].content


# ==================== 会话统计 ====================

class TestGetSessionStats:
    """测试会话统计信息"""

    def test_initial_stats(self, mock_llm, mock_embedding, tmp_chroma_db):
        """初始状态: 0 轮,使用比例 0"""
        mm = MemoryManager(mock_llm, mock_embedding)
        stats = mm.get_session_stats()

        assert stats["turns"] == 0
        assert stats["max_turns"] == 10  # max_history_len=20, 每轮 2 条
        assert stats["usage_ratio"] == 0.0

    def test_stats_after_exchanges(self, mock_llm, mock_embedding, tmp_chroma_db):
        """几轮对话后统计正确"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.add_exchange("问题1", "回答1")
        mm.add_exchange("问题2", "回答2")

        stats = mm.get_session_stats()
        assert stats["turns"] == 2
        assert stats["usage_ratio"] == 0.2  # 2/10

    def test_stats_max_turns(self, mock_llm, mock_embedding, tmp_chroma_db):
        """max_turns = max_history_len // 2"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.max_history_len = 30

        stats = mm.get_session_stats()
        assert stats["max_turns"] == 15


# ==================== 会话重置 ====================

class TestResetSession:
    """测试会话重置"""

    def test_clears_short_term(self, mock_llm, mock_embedding, tmp_chroma_db):
        """重置后短期记忆清空"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.add_exchange("问题1", "回答1")
        mm.add_exchange("问题2", "回答2")

        mm.reset_session()
        assert len(mm.short_term_memory) == 0

    def test_preserves_long_term(self, mock_llm, mock_embedding, tmp_chroma_db):
        """重置后长期记忆保留"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.max_history_len = 4

        # 触发压缩,生成长期记忆
        for i in range(3):
            mm.add_exchange(f"用户{i}", f"AI{i}")

        long_term_count_before = mm.long_term_collection.count()
        assert long_term_count_before > 0

        mm.reset_session()

        # 长期记忆数量不变
        long_term_count_after = mm.long_term_collection.count()
        assert long_term_count_after == long_term_count_before

    def test_stats_after_reset(self, mock_llm, mock_embedding, tmp_chroma_db):
        """重置后统计归零"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.add_exchange("问题1", "回答1")

        mm.reset_session()
        stats = mm.get_session_stats()
        assert stats["turns"] == 0
        assert stats["usage_ratio"] == 0.0

    def test_can_add_after_reset(self, mock_llm, mock_embedding, tmp_chroma_db):
        """重置后能继续添加新对话"""
        mm = MemoryManager(mock_llm, mock_embedding)
        mm.add_exchange("旧问题", "旧回答")
        mm.reset_session()

        mm.add_exchange("新问题", "新回答")
        assert len(mm.short_term_memory) == 2
        assert mm.short_term_memory[0]["content"] == "新问题"
