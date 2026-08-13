# 全局配置常量
# 从.env文件中读取配置
import os
# dotenv模块用于加载环境变量，.env文件中的变量会被加载到os.environ字典中
from dotenv import load_dotenv
# 从.env文件中加载环境变量
#os.environ字典中全是字符串类型，需要转换为其他类型
load_dotenv()
# Chroma向量数据库的持久化目录
# 如果.env文件中没有指定，默认使用./chroma_rag_db
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR","./chroma_rag_db")

# 文本切片的参数
chunk_size = int(os.environ.get("CHUNK_SIZE", 100))
chunk_overlap =int(os.environ.get("CHUNK_OVERLAP", 80))  # 文本切片的重叠长度

# 向量检索返回topK片段数量
retrieve_top_k = int(os.environ.get("RETRIEVE_TOP_K", 10))

# ========== 三级模型路由 ==========
# qwen-vl-max: 多模态(图片) → 最贵但最强
# qwen-plus:  纯文本对话    → 中等性价比
# qwen-turbo: 摘要/简单任务  → 最便宜，摘要够用
SUMMARY_MODEL_NAME=os.environ.get("SUMMARY_MODEL_NAME", "qwen-turbo")#摘要压缩用的轻量模型

# 通义千问兼容OpenAI的接口地址
# LangChain默认会访问OpenAI官方地址，需要手动指定base_url转发到阿里云通义千问网关
DASHSCOPE_BASE_URL =os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# 模型名称
model_name=os.environ.get("MODEL_NAME", "qwen-vl-max")
TEXT_MODEL_NAME = os.environ.get("TEXT_MODEL_NAME", "qwen-plus")  # 【新增】纯文本模型（无图片时用，省token）
embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-v1") # 文本转向量模型

# 请求超时时间设置
# 底层http库部分场景要求浮点类型，使用60.0避免潜在类型报错
request_timeout = float(os.environ.get("REQUEST_TIMEOUT", 60.0))

# 温度参数，控制模型输出随机性
temperature = float(os.environ.get("TEMPERATURE", 0.7))
# 取值区间0~1
# 越接近0：输出稳定、逻辑严谨，随机性低
# 越接近1：回答发散、创意更强，但更容易产生幻觉

#RAG去重+重排参数:
#1.重排前扩大检索的数量,选取top前20个片段进行重排
# 重排的意义是根据片段的相似度和相关性，对检索到的片段进行排序，使更相关的片段排在前面，更不相关的片段排在后面
# 因为粗排会出现几个词重复但内容不符合的情况,更多的是距离,没考虑相关性逻辑被筛选出来
RERANK_TOP_K=int(os.environ.get("RERANK_TOP_K", 20))
#2.重排阈值,把相关性低于0.3的片段筛选出来,只保留相关性高于阈值的片段
SIMILARITY_THRESHOLD=float(os.environ.get("SIMILARITY_THRESHOLD", 0.3))
#3.语义去重阈值,把相似度高于阈值的片段筛选出来,只保留相似度低于阈值的片段
DEDUP_SIMILARITY_THRESHOLD=float(os.environ.get("DEDUP_SIMILARITY_THRESHOLD", 0.95))

# ========== Token 压缩参数 ==========
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", 2000))  # RAG上下文最大字符数，超长截断
MAX_REPLY_TOKENS = int(os.environ.get("MAX_REPLY_TOKENS", 1024))  # LLM回复最大token数，防止废话

# ========== Agent 配置 ==========
AGENT_MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", 3))  # Agent最大推理循环次数