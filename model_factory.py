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

# 对话大模型（多模态，有图片时用）
def create_llm(api_key: str):
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
        timeout=request_timeout,
        temperature=temperature,
        max_tokens=MAX_REPLY_TOKENS,#限制回复长度,防止废话浪费
    )
    return llm

# 【新增】纯文本大模型（无图片时用，省 token）
# 记忆管理器的摘要压缩也用这个模型，比 qwen-vl-max 便宜约 50%
def create_text_llm(api_key: str):
    llm = ChatOpenAI(
        model=TEXT_MODEL_NAME,
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
        timeout=request_timeout,
        temperature=temperature,
        max_tokens=MAX_REPLY_TOKENS,#限制回复长度,防止废话浪费
    )
    return llm

# 【新增】摘要压缩专用模型(qwen-turbo,最便宜)
#用于记忆管理功能的对话摘要
def create_summary_llm(api_key: str):
    llm = ChatOpenAI(
        model=SUMMARY_MODEL_NAME,
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
        timeout=request_timeout,
        temperature=0.3, #摘要压缩需要精准度，避免生成不相关的内容,所有降低温度值
        max_tokens=300, #摘要不能过长,防止浪费
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
