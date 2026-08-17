"""
test_rag_service.py
===================
测试 rag_service.py 的核心逻辑
- MD5 精确去重
- 语义近似去重(同源内查重)
- 文档入库流程(分割 → 去重 → 存储)
- 检索服务(相关度过滤 + 截断)
- 上下文拼接与截断
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from rag_service import (
    _compute_md5,
    _is_md5_duplicate,
    _is_semantic_duplicate,
    ingest_txt_vector_store,
    search_vector_store,
    get_context_from_docs,
)


# ==================== 第一层: MD5 精确去重 ====================

class TestComputeMD5:
    """测试 MD5 哈希计算"""

    def test_md5_consistent(self):
        """相同文本 → 相同哈希"""
        text = "今天心情很糟糕"
        assert _compute_md5(text) == _compute_md5(text)

    def test_md5_different_text(self):
        """不同文本 → 不同哈希"""
        assert _compute_md5("你好") != _compute_md5("你好呀")

    def test_md5_known_value(self):
        """验证 MD5 值与 hashlib 结果一致"""
        import hashlib
        text = "test"
        expected = hashlib.md5(text.encode("utf-8")).hexdigest()
        assert _compute_md5(text) == expected

    def test_md5_empty_string(self):
        """空字符串也能计算哈希(不报错)"""
        result = _compute_md5("")
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 固定 32 位十六进制

    def test_md5_unicode(self):
        """中文字符正常计算"""
        result = _compute_md5("情绪修复助手")
        assert isinstance(result, str)
        assert len(result) == 32


class TestIsMD5Duplicate:
    """测试 MD5 去重检查"""

    def test_duplicate_found(self):
        """向量库中已存在相同 MD5 → 返回 True"""
        mock_vs = MagicMock()
        mock_vs._collection.get.return_value = {"ids": ["doc_1"]}

        assert _is_md5_duplicate(mock_vs, "abc123") is True

    def test_no_duplicate(self):
        """向量库中不存在该 MD5 → 返回 False"""
        mock_vs = MagicMock()
        mock_vs._collection.get.return_value = {"ids": []}

        assert _is_md5_duplicate(mock_vs, "abc123") is False

    def test_no_collection_attr(self):
        """vector_store 没有 _collection 属性 → 降级返回 False"""
        mock_vs = MagicMock(spec=[])  # 空规格,没有任何属性

        assert _is_md5_duplicate(mock_vs, "abc123") is False

    def test_exception_returns_false(self):
        """查询异常时降级返回 False(不阻塞入库)"""
        mock_vs = MagicMock()
        mock_vs._collection.get.side_effect = Exception("DB error")

        assert _is_md5_duplicate(mock_vs, "abc123") is False


# ==================== 第二层: 语义近似去重 ====================

class TestIsSemanticDuplicate:
    """测试语义去重(同一来源内查重)"""

    def test_above_threshold_is_duplicate(self):
        """余弦相似度高于阈值 → 判定重复"""
        mock_vs = MagicMock()
        doc = Document(page_content="如何缓解焦虑")
        # score=0.05 → cosine_sim = 1 - 0.05 = 0.95 > 0.9
        mock_vs.similarity_search_with_score.return_value = [(doc, 0.05)]

        assert _is_semantic_duplicate(mock_vs, "如何缓解焦虑", 0.9, "test.txt") is True

    def test_below_threshold_not_duplicate(self):
        """余弦相似度低于阈值 → 不重复"""
        mock_vs = MagicMock()
        doc = Document(page_content="完全不同的内容")
        # score=0.8 → cosine_sim = 1 - 0.8 = 0.2 < 0.9
        mock_vs.similarity_search_with_score.return_value = [(doc, 0.8)]

        assert _is_semantic_duplicate(mock_vs, "如何缓解焦虑", 0.9, "test.txt") is False

    def test_empty_results_not_duplicate(self):
        """向量库为空 → 不重复(首次入库)"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = []

        assert _is_semantic_duplicate(mock_vs, "新内容", 0.9, "test.txt") is False

    def test_uses_source_filter(self):
        """查重时传入 source 过滤条件(避免跨文件误杀)"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = []

        _is_semantic_duplicate(mock_vs, "测试文本", 0.9, "knowledge.txt")

        # 验证 similarity_search_with_score 被调用时传入了 filter
        call_kwargs = mock_vs.similarity_search_with_score.call_args
        assert call_kwargs.kwargs.get("filter") == {"source": "knowledge.txt"}

    def test_no_source_no_filter(self):
        """未提供 source_name 时不传 filter(兼容旧逻辑)"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = []

        _is_semantic_duplicate(mock_vs, "测试文本", 0.9, None)

        call_kwargs = mock_vs.similarity_search_with_score.call_args
        assert "filter" not in call_kwargs.kwargs

    def test_exception_returns_false(self):
        """异常时降级返回 False"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.side_effect = Exception("DB error")

        assert _is_semantic_duplicate(mock_vs, "测试", 0.9, "test.txt") is False


# ==================== 文档入库 ====================

class TestIngestTxtVectorStore:
    """测试文档入库流程(分割 → 去重 → 存储)"""

    def test_basic_ingest(self):
        """正常文本入库 → 返回入库数量 > 0"""
        mock_vs = MagicMock()
        mock_vs._collection.get.return_value = {"ids": []}
        mock_vs.similarity_search_with_score.return_value = []

        text = "这是一段关于情绪管理的知识文本,包含深呼吸、正念冥想等技巧。"
        count = ingest_txt_vector_store(text, "test.txt", mock_vs)

        assert count > 0
        mock_vs.add_documents.assert_called_once()

    def test_empty_text_returns_zero(self):
        """空文本入库 → 返回 0"""
        mock_vs = MagicMock()
        count = ingest_txt_vector_store("", "test.txt", mock_vs)
        assert count == 0
        mock_vs.add_documents.assert_not_called()

    def test_md5_dedup_skips_duplicate(self):
        """MD5 去重: 已存在的切片被跳过"""
        mock_vs = MagicMock()
        # 模拟所有切片的 MD5 都已存在
        mock_vs._collection.get.return_value = {"ids": ["existing"]}
        mock_vs.similarity_search_with_score.return_value = []

        text = "这是一段文本内容"
        count = ingest_txt_vector_store(text, "test.txt", mock_vs)

        assert count == 0
        mock_vs.add_documents.assert_not_called()

    def test_batch_ingest_large_text(self, monkeypatch):
        """大批量入库: 超过 25 条时分批处理"""
        mock_vs = MagicMock()
        mock_vs._collection.get.return_value = {"ids": []}
        mock_vs.similarity_search_with_score.return_value = []

        # 设置小 chunk_size 产生大量切片
        monkeypatch.setattr("rag_service.chunk_size", 10)
        monkeypatch.setattr("rag_service.chunk_overlap", 2)

        # 生成足够长的文本(会产生 >25 个切片)
        long_text = "情绪修复技巧。" * 100
        count = ingest_txt_vector_store(long_text, "big.txt", mock_vs)

        assert count > 25
        # 验证分批调用(add_documents 被调用多次)
        assert mock_vs.add_documents.call_count > 1

    def test_source_metadata_stored(self):
        """入库文档携带 source 和 md5_hash 元数据"""
        mock_vs = MagicMock()
        mock_vs._collection.get.return_value = {"ids": []}
        mock_vs.similarity_search_with_score.return_value = []

        text = "这是一段文本"
        ingest_txt_vector_store(text, "knowledge.txt", mock_vs)

        # 检查 add_documents 收到的 Document 列表
        added_docs = mock_vs.add_documents.call_args[0][0]
        for doc in added_docs:
            assert doc.metadata["source"] == "knowledge.txt"
            assert "md5_hash" in doc.metadata


# ==================== 检索服务 ====================

class TestSearchVectorStore:
    """测试检索服务(相关度过滤 + 截断)"""

    def test_filters_low_similarity(self, monkeypatch):
        """低于阈值的结果被过滤"""
        monkeypatch.setattr("rag_service.retrieve_top_k", 1)
        mock_vs = MagicMock()
        docs = [
            (Document(page_content="高相关"), 0.1),   # sim=0.9
            (Document(page_content="低相关"), 0.9),   # sim=0.1
        ]
        mock_vs.similarity_search_with_score.return_value = docs

        results = search_vector_store(mock_vs, "测试查询")

        # 只有高相关的被保留(过滤后1条 >= retrieve_top_k=1,不触发fallback)
        assert len(results) == 1
        assert results[0].page_content == "高相关"

    def test_all_above_threshold(self, monkeypatch):
        """全部高于阈值 → 全部保留(截断为 top_k)"""
        monkeypatch.setattr("rag_service.retrieve_top_k", 3)
        mock_vs = MagicMock()
        docs = [
            (Document(page_content=f"doc{i}"), 0.1) for i in range(5)
        ]
        mock_vs.similarity_search_with_score.return_value = docs

        results = search_vector_store(mock_vs, "测试查询")

        # retrieve_top_k=3,只返回前 3 条
        assert len(results) == 3

    def test_empty_results(self):
        """向量库返回空 → 返回空列表"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = []

        results = search_vector_store(mock_vs, "测试查询")
        assert results == []

    def test_search_exception_returns_empty(self):
        """检索异常 → 返回空列表(不崩溃)"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.side_effect = Exception("DB error")

        results = search_vector_store(mock_vs, "测试查询")
        assert results == []

    def test_fallback_when_filtered_too_few(self, monkeypatch):
        """过滤后不足 top_k → 放宽阈值补充"""
        monkeypatch.setattr("rag_service.retrieve_top_k", 3)
        mock_vs = MagicMock()
        docs = [
            (Document(page_content="高相关"), 0.1),   # sim=0.9, 通过过滤
            (Document(page_content="低相关1"), 0.95),  # sim=0.05, 不通过
            (Document(page_content="低相关2"), 0.96),  # sim=0.04, 不通过
        ]
        mock_vs.similarity_search_with_score.return_value = docs

        results = search_vector_store(mock_vs, "测试查询")

        # 过滤后只有 1 条 < retrieve_top_k=3,触发放宽逻辑
        # 放宽后取 results[:3],返回 3 条(包含低相关的)
        assert len(results) == 3


# ==================== 上下文拼接与截断 ====================

class TestGetContextFromDocs:
    """测试上下文拼接与 Token 截断"""

    def test_single_doc(self):
        """单个文档正常拼接"""
        docs = [Document(page_content="情绪修复建议")]
        result = get_context_from_docs(docs)
        assert result == "情绪修复建议"

    def test_multiple_docs_joined(self):
        """多个文档用双换行拼接"""
        docs = [
            Document(page_content="第一段"),
            Document(page_content="第二段"),
        ]
        result = get_context_from_docs(docs)
        assert "第一段" in result
        assert "第二段" in result
        assert "\n\n" in result

    def test_empty_docs(self):
        """空文档列表 → 返回空字符串"""
        result = get_context_from_docs([])
        assert result == ""

    def test_truncation_when_exceeds_limit(self, monkeypatch):
        """超过 MAX_CONTEXT_CHARS → 截断并加省略标记"""
        monkeypatch.setattr("rag_service.MAX_CONTEXT_CHARS", 10)
        docs = [Document(page_content="这是一段超过十个字符的长文本内容")]
        result = get_context_from_docs(docs)

        assert len(result) <= 20  # 10 字符 + "…(已截断)"
        assert "已截断" in result

    def test_no_truncation_when_under_limit(self, monkeypatch):
        """未超过 MAX_CONTEXT_CHARS → 不截断"""
        monkeypatch.setattr("rag_service.MAX_CONTEXT_CHARS", 1000)
        docs = [Document(page_content="短文本")]
        result = get_context_from_docs(docs)

        assert "已截断" not in result
        assert result == "短文本"

    def test_truncation_exact_boundary(self, monkeypatch):
        """正好等于 MAX_CONTEXT_CHARS → 不截断"""
        text = "正好十个字"
        monkeypatch.setattr("rag_service.MAX_CONTEXT_CHARS", len(text))
        docs = [Document(page_content=text)]
        result = get_context_from_docs(docs)

        assert "已截断" not in result
        assert result == text
