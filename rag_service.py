# RAG服务
# 向量库,文档入库,检索服务
# 用于文档去重,重排
import hashlib
from pathlib import Path  # 用于路径操作
from langchain_community.vectorstores import Chroma  # 用于向量数据库
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 用于文本分割
from langchain_core.documents import Document  # 用于文档操作
from config import (
    CHROMA_DB_DIR,
    chunk_size,
    chunk_overlap,
    retrieve_top_k,
    # RAG去重+重排参数:
    SIMILARITY_THRESHOLD,
    DEDUP_SIMILARITY_THRESHOLD,
    RERANK_TOP_K,
)


def init_vector_store(embeddings):
    """
    初始化向量库
    如果目录不存在,则创建,已存在则自动加载持久化数据
    :return:
    """
    Path(CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,  # 向量嵌入函数,用于将文本转换为向量表示
    )
    return vector_store


# TODO: 1.MD5精确去重(入库时)--------------------------------------

def _compute_md5(text: str):
    """
    计算文本的MD5值,一个32位的字符串,用于唯一标识文本
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()
    # hexdigest()方法将字节对象转换为16进制字符串


def _is_md5_duplicate(vector_store, chunk_hash: str) -> bool:
    """
    检查改MD5值是否已存在向量数据库中
    利用向量数据库的metadata字段,检查是否存在相同的MD5值的文档
    :param vector_store: Chroma向量数据库,创建的一个实例
    :param chunk_hash: 文档的MD5值
    :return: 是否存在
    """
    # TODO: where查询返回结果格式:
    """{
        "ids": ["uuid-1", "uuid-2"],      # 找到了几个文档，就有几个ID
        "embeddings": None,                # 这里不需要返回向量，所以通常是None
        "metadatas": [{"source": "a.txt"}],# 对应的元数据
        "documents": ["文本内容..."]       # 对应的文本内容
        }
    """
    try:
        # Chroma允许通过metadata字段进行查询
        # 查询中'md5_hash'等于当前的hash的文档
        # 如果存在,则返回True,否则返回False
        existing = vector_store.get(where={"md5_hash": chunk_hash})
        if existing and existing.get("ids"):
            # 如果当前文档在hash列表中存在了对应的ID,则说明已存在,即重复文档
            return True
    except Exception as e:
        print(f"md5查询向量数据库出错:{e}")
        return False
    return False  # 【补充】如果没报错也没找到，应该返回False


# TODO: 2.语义近似去重(入库时)--------------------------------------------------
def _is_semantic_duplicate(vector_store, text: str, threshold=float, source_name: str = None) -> bool:
    """
    检查文本是否与已有的内容语义上高度重复
    原理:拿当前文本去库里面搜索出最相似的一条内容,如果相似度极高,达到预设的阈值,则说明已存在,即重复文档
    【关键修复】只在同一 source 内查重，不同文件允许内容相似，避免文件B被文件A的内容"误杀"导致返回0
    """
    try:
        # 搜索最相似的一条:
        # 【关键修复】只在同一来源内搜索，避免跨文件误杀
        if source_name:
            result = vector_store.similarity_search_with_score(text, k=1, filter={"source": source_name})
        else:
            result = vector_store.similarity_search_with_score(text, k=1)

        if result:
            # 返回结果是列表的列表results = [
            # (Document对象_1, 0.35),   # 第1个结果：(文档内容, 距离分数)
            # (Document对象_2, 0.42),   # 第2个结果
            #         ]
            doc, score = result[0]

            # 【关键修复】ChromaDB默认返回平方L2距离（不是L2距离）
            # 对归一化向量：平方L2 = 2 - 2 * 余弦相似度
            # 所以：余弦相似度 = 1 - score / 2
            cosine_sim = max(0.0, 1 - score / 2)

            # 如果相似度高于阈值,则说明已存在,即重复文档
            if cosine_sim > threshold:
                print(
                    f"  [语义去重] 余弦相似度={cosine_sim:.4f} > {threshold}, 来源={source_name}, 跳过: {text[:40]}...")
                return True
    except Exception:
        pass
    return False


# 文档入库函数,输入原始文本,文档来源,向量库
def ingest_txt_vector_store(raw_text: str, source_name: str, vector_store):
    """
    包含双重去重:MD5精确去重+语义近似去重
    文档入库完整的一个流程:文本分割->封装成Document->向量嵌入->向量数据库存储->持久化
    :param raw_text: txt全文本
    :param source_name: 文档名称(元数据溯源)
    :param vector_store: Chroma向量数据库,创建的一个实例
    :return: 切片总数量
    """
    # 递归文本分割器
    # 下面是初始化文本分割器,设置每个切片的最大字符数和切片之间的重叠字符数
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # 每个切片的最大字符数
        chunk_overlap=chunk_overlap,  # 切片之间的重叠字符数
    )
    # 将文本分割成多个切片
    # ==========【修复】split() 方法不存在，正确方法是 split_text() ==========
    text_chunks = text_splitter.split_text(raw_text)  # 分割成字符串列表[切片1,切片2,切片3,...]
    # 入库前去重:
    valid_docs = []  # 有效切片列表
    md5_count = 0  # 记录MD5去重的重复切片数量
    semantic_count = 0  # 记录语义去重的重复切片数量

    for chunk in text_chunks:
        # 1.MD5去重
        chunk_hash = _compute_md5(chunk)  # 【修复】修正拼写 conpute -> compute
        if _is_md5_duplicate(vector_store, chunk_hash):
            md5_count += 1  # 记录重复切片数量
            continue
        # 2.语义去重
        # 【关键修复】这里必须传入 chunk (原始文本)，而不是 chunk_hash (MD5值)
        # 同时传入配置中的阈值 DEDUP_SIMILARITY_THRESHOLD 和当前来源 source_name
        if _is_semantic_duplicate(vector_store, chunk, DEDUP_SIMILARITY_THRESHOLD, source_name):
            semantic_count += 1  # 记录重复切片数量
            continue  # 【关键修复】命中语义去重后，必须跳过当前循环，不能加入valid_docs

        # 通过双重检查,确保当前切片不是重复的后,构建成Document对象,并写入md5_hash元数据
        doc = Document(
            page_content=chunk,
            metadata={"source": source_name, "md5_hash": chunk_hash},
        )
        valid_docs.append(doc)
    # 批量写入,将最终的有效切片列表封装成Document对象:
    if valid_docs:
        vector_store.add_documents(valid_docs)
        print(f"[入库成功]原文本切片{len(text_chunks)}")
        print(f"MD5去重文本{md5_count}个,语义去重文本{semantic_count}个")
        print(f"实际入库切片{len(valid_docs)}个")
    else:
        print(f"[入库提示]所有切片都被去重,无需入库")

    return len(valid_docs)  # 返回实际切片数量


# 设置向量数据库的检索参数
def search_vector_store(vector_store: Chroma, query: str):
    # k表示返回的文档数量,相似度检索的方法是余弦相似度
    # 检索优化:扩大召回->过滤噪声->截取Top-K
    # 1.扩大检索返回:
    search_k = max(retrieve_top_k * 3, RERANK_TOP_K)  # 取最大值,确保检索到足够的文档
    try:
        # 获取带分数的搜索结果:[(Document对象_1, 0.35), (Document对象_2, 0.42), ...]
        results = vector_store.similarity_search_with_score(query, k=search_k)
    except Exception as e:
        print(f"[检索错误]{e}")
        return []

    filtered_docs = []  # 过滤后的文档列表

    # 2.相关度过滤:剔除低分噪声
    for doc, score in results:
        # 【关键修复】ChromaDB返回平方L2距离，余弦相似度 = 1 - score/2
        cosine_sim = max(0.0, 1 - score / 2)
        if cosine_sim > SIMILARITY_THRESHOLD:
            filtered_docs.append(doc)

    # 3.截断重排:只保留Top-K个文档
    # 如果过滤后文档数量不足Top-K,为了保证有内容返回,可以放宽策略,这里简单处理为直接截取
    final_docs = filtered_docs[:retrieve_top_k]
    print(f"[检索成功]共检索到{len(results)}个文档,过滤后{len(filtered_docs)}个,截取Top-{retrieve_top_k}个")
    return final_docs


# 将返回的文档列表,转换为字符串
def get_context_from_docs(docs: list[Document]) -> str:
    """
    将检索到的Document列表拼接成一段上下文文本字符串
    :param docs: 检索到的Document列表
    :return: 上下文文本字符串
    """
    # join是啥? 将列表中的元素用指定的分隔符拼接成一个字符串
    # 这里用\n\n表示每个文档之间用两个换行符隔开
    return "\n\n".join([doc.page_content for doc in docs])