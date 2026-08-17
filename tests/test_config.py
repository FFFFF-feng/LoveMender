"""
test_config.py
==============
测试 config.py 配置参数加载逻辑
- 验证所有必要配置项存在且类型正确
- 验证默认值在合理范围内
- 验证环境变量能正确覆盖默认值
"""

import importlib


class TestConfigAttributes:
    """测试 config 模块包含所有必要的配置项"""

    def test_has_rag_params(self):
        """RAG 相关参数存在"""
        import config
        assert hasattr(config, "CHROMA_DB_DIR")
        assert hasattr(config, "chunk_size")
        assert hasattr(config, "chunk_overlap")
        assert hasattr(config, "retrieve_top_k")

    def test_has_model_params(self):
        """模型路由相关参数存在"""
        import config
        assert hasattr(config, "model_name")
        assert hasattr(config, "TEXT_MODEL_NAME")
        assert hasattr(config, "SUMMARY_MODEL_NAME")
        assert hasattr(config, "embedding_model_name")
        assert hasattr(config, "DASHSCOPE_BASE_URL")

    def test_has_rerank_params(self):
        """重排和去重参数存在"""
        import config
        assert hasattr(config, "RERANK_TOP_K")
        assert hasattr(config, "SIMILARITY_THRESHOLD")
        assert hasattr(config, "DEDUP_SIMILARITY_THRESHOLD")

    def test_has_token_params(self):
        """Token 压缩参数存在"""
        import config
        assert hasattr(config, "MAX_CONTEXT_CHARS")
        assert hasattr(config, "MAX_REPLY_TOKENS")

    def test_has_agent_params(self):
        """Agent 配置参数存在"""
        import config
        assert hasattr(config, "AGENT_MAX_ITERATIONS")


class TestConfigTypes:
    """测试配置项的类型正确"""

    def test_int_params(self):
        """整型参数类型正确"""
        import config
        assert isinstance(config.chunk_size, int)
        assert isinstance(config.chunk_overlap, int)
        assert isinstance(config.retrieve_top_k, int)
        assert isinstance(config.RERANK_TOP_K, int)
        assert isinstance(config.MAX_CONTEXT_CHARS, int)
        assert isinstance(config.MAX_REPLY_TOKENS, int)
        assert isinstance(config.AGENT_MAX_ITERATIONS, int)

    def test_float_params(self):
        """浮点型参数类型正确"""
        import config
        assert isinstance(config.request_timeout, float)
        assert isinstance(config.temperature, float)
        assert isinstance(config.SIMILARITY_THRESHOLD, float)
        assert isinstance(config.DEDUP_SIMILARITY_THRESHOLD, float)

    def test_str_params(self):
        """字符串参数类型正确"""
        import config
        assert isinstance(config.CHROMA_DB_DIR, str)
        assert isinstance(config.model_name, str)
        assert isinstance(config.DASHSCOPE_BASE_URL, str)


class TestConfigValueRanges:
    """测试配置项的值在合理范围内"""

    def test_chunk_params_valid(self):
        """切片参数: size > 0, overlap >= 0, overlap < size"""
        import config
        assert config.chunk_size > 0
        assert config.chunk_overlap >= 0
        assert config.chunk_overlap < config.chunk_size

    def test_retrieve_params_valid(self):
        """检索参数: top_k > 0, rerank_top_k >= retrieve_top_k"""
        import config
        assert config.retrieve_top_k > 0
        assert config.RERANK_TOP_K >= config.retrieve_top_k

    def test_threshold_in_range(self):
        """阈值参数在 [0, 1] 范围内"""
        import config
        assert 0.0 <= config.SIMILARITY_THRESHOLD <= 1.0
        assert 0.0 <= config.DEDUP_SIMILARITY_THRESHOLD <= 1.0

    def test_temperature_in_range(self):
        """温度参数在 [0, 1] 范围内"""
        import config
        assert 0.0 <= config.temperature <= 1.0

    def test_token_limits_positive(self):
        """Token 限制为正数"""
        import config
        assert config.MAX_CONTEXT_CHARS > 0
        assert config.MAX_REPLY_TOKENS > 0

    def test_agent_iterations_positive(self):
        """Agent 迭代次数为正数"""
        import config
        assert config.AGENT_MAX_ITERATIONS > 0


class TestConfigEnvOverride:
    """测试环境变量覆盖机制"""

    def test_env_override_chunk_size(self, monkeypatch):
        """环境变量 CHUNK_SIZE 能覆盖默认值"""
        monkeypatch.setenv("CHUNK_SIZE", "256")
        import config
        importlib.reload(config)
        try:
            assert config.chunk_size == 256
        finally:
            importlib.reload(config)

    def test_env_override_temperature(self, monkeypatch):
        """环境变量 TEMPERATURE 能覆盖默认值"""
        monkeypatch.setenv("TEMPERATURE", "0.3")
        import config
        importlib.reload(config)
        try:
            assert config.temperature == 0.3
        finally:
            importlib.reload(config)

    def test_env_override_top_k(self, monkeypatch):
        """环境变量 RETRIEVE_TOP_K 能覆盖默认值"""
        monkeypatch.setenv("RETRIEVE_TOP_K", "7")
        import config
        importlib.reload(config)
        try:
            assert config.retrieve_top_k == 7
        finally:
            importlib.reload(config)
