"""
pytest 全局配置文件 (conftest.py)
================================
职责:
1. 将项目根目录加入 sys.path,让测试能 import 项目模块
2. 设置测试用环境变量,避免调用真实 API
3. 提供 Mock LLM / Mock Embedding / Mock VectorStore fixture

运行方式:
    cd D:\\develop\\python project\\LoveMender
    python -m pytest tests/ -v
"""

import sys
import os
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ===== 1. 路径设置 (必须在 import 项目模块之前) =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ===== 2. 测试环境变量 =====
# config.py 的 load_dotenv(override=False) 不会覆盖已设置的变量
# 所以这里设置的假 Key 会优先于 .env 文件中的真实 Key
os.environ.setdefault("DASHSCOPE_API_KEY", "test_key_not_real_12345")


# ===== 3. Mock 对象定义 =====

class MockLLM:
    """
    模拟大语言模型
    - invoke() 返回带 content 属性的 MagicMock,模拟 AIMessage
    - 记录调用次数,方便断言
    """

    def __init__(self, response_text="这是一条测试摘要"):
        self.response_text = response_text
        self.invoke_count = 0

    def invoke(self, messages):
        self.invoke_count += 1
        return MagicMock(content=self.response_text)


class MockEmbedding:
    """
    模拟嵌入模型
    - 基于文本 MD5 生成确定性向量(不同文本 → 不同向量)
    - 不依赖外部 API,测试速度快
    """

    def __init__(self, dim=16):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        return [int(h[i % len(h):(i % len(h)) + 2], 16) / 255.0 for i in range(self.dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# ===== 4. 公共 Fixture =====

@pytest.fixture
def mock_llm():
    """提供 Mock LLM 实例"""
    return MockLLM()


@pytest.fixture
def mock_embedding():
    """提供 Mock Embedding 实例"""
    return MockEmbedding()


@pytest.fixture
def tmp_chroma_db(tmp_path, monkeypatch):
    """
    提供临时向量数据库目录,测试结束后自动清理
    同时 patch memory_manager 中导入的 CHROMA_DB_DIR,确保不影响真实数据
    """
    db_dir = tmp_path / "test_chroma_db"
    db_dir.mkdir()
    db_path = str(db_dir)
    # patch memory_manager 模块中的 CHROMA_DB_DIR
    monkeypatch.setattr("memory_manager.CHROMA_DB_DIR", db_path)
    return db_path
