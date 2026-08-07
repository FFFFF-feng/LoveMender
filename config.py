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

# 通义千问兼容OpenAI的接口地址
# LangChain默认会访问OpenAI官方地址，需要手动指定base_url转发到阿里云通义千问网关
DASHSCOPE_BASE_URL =os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# 模型名称
model_name=os.environ.get("MODEL_NAME", "qwen-vl-max")
embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-v1") # 文本转向量模型（修正拼写texr → text）

# 请求超时时间设置
# 底层http库部分场景要求浮点类型，使用60.0避免潜在类型报错
request_timeout = float(os.environ.get("REQUEST_TIMEOUT", 60.0))

# 温度参数，控制模型输出随机性
temperature = float(os.environ.get("TEMPERATURE", 0.7))
# 取值区间0~1
# 越接近0：输出稳定、逻辑严谨，随机性低
# 越接近1：回答发散、创意更强，但更容易产生幻觉