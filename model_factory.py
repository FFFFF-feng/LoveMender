from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import (
    model_name,
    TEXT_MODEL_NAME,  # 【新增】纯文本模型名称
    SUMMARY_MODEL_NAME,  # 【新增】摘要压缩模型名称
    embedding_model_name,
    DASHSCOPE_BASE_URL,
    request_timeout,
    temperature,
    MAX_REPLY_TOKENS,#最大回复token数
)

# 第一级：多模态对话模型（有图片时用，最强）
# 替代旧版 qwen-vl-max
def create_llm(api_key: str):
    llm = ChatOpenAI(
        model=model_name,  # qwen3.7-plus
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
        timeout=request_timeout,
        temperature=temperature,
        max_tokens=MAX_REPLY_TOKENS,
    )
    return llm

# 第二级：纯文本对话模型（无图片时用，省 token）
# 替代旧版 qwen-plus，比第一级便宜
def create_text_llm(api_key: str):
    llm = ChatOpenAI(
        model=TEXT_MODEL_NAME,  # qwen3.8-27b
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
        timeout=request_timeout,
        temperature=temperature,
        max_tokens=MAX_REPLY_TOKENS,
    )
    return llm

# 第三级：摘要压缩专用模型（最轻最快，用于记忆管理）
# 替代旧版 qwen-turbo
def create_summary_llm(api_key: str):
    llm = ChatOpenAI(
        model=SUMMARY_MODEL_NAME,  # qwen3.7-flash
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
        timeout=request_timeout,
        temperature=0.3,
        max_tokens=300,
    )
    return llm

# 文本嵌入模型
def create_embedding(api_key: str):
    embeddings = OpenAIEmbeddings(
        model=embedding_model_name,  # 使用嵌入模型名称
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
        # ==========【关键修复】==========
        # OpenAIEmbeddings 默认 check_embedding_ctx_length=True，
        # 会用 tiktoken 把文本编码成 token 整数列表再发给 API。
        # 但 DashScope 的 OpenAI 兼容接口不支持 token 输入，只接受文本字符串，
        # 导致服务端返回错误。设为 False 后直接发送原始文本，跳过 token 编码。
        check_embedding_ctx_length=False,
    )
    return embeddings
