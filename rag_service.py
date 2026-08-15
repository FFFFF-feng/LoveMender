# RAG服务（增强版）
# 向量库,文档入库(含去重),检索服务(含相关度过滤+重排)
import hashlib  # 用于计算MD5哈希值
from pathlib import Path  #用于路径操作
from langchain_community.vectorstores import Chroma  #用于向量数据库
from langchain_text_splitters import RecursiveCharacterTextSplitter  #用于文本分割
from langchain_core.documents import Document  #用于文档操作
from logger import logger  #用于日志记录

from config import (
    CHROMA_DB_DIR,
    chunk_size,
    chunk_overlap,
    retrieve_top_k,
    RERANK_TOP_K,                # 重排前扩大检索的候选数量
    SIMILARITY_THRESHOLD,        # 相关度过滤阈值
    DEDUP_SIMILARITY_THRESHOLD,  # 语义去重阈值
    MAX_CONTEXT_CHARS,           # 【新增】RAG上下文最大字符数
)


def init_vector_store(embeddings):
    """
    初始化向量库
    如果目录不存在,则创建,已存在则自动加载持久化数据
    【关键修复】使用 cosine 距离度量，避免向量未归一化时 L2 距离失效
    :return:
    """
    Path(CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},  # cosine距离，不受向量归一化影响
    )
    return vector_store


# ==================== 第一层：MD5 精确去重 ====================

def _compute_md5(text: str) -> str:
    """计算文本的MD5哈希值，用于精确去重"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _is_md5_duplicate(vector_store, chunk_hash: str) -> bool:
    """
    检查该MD5是否已存在于向量库中
    通过Chroma底层collection的where元数据过滤查询
    :return: True表示已存在（重复）
    """
    try:
        # vector_store._collection 是 Chroma 底层的 chromadb collection 对象
        # where 参数支持按 metadata 字段过滤
        if hasattr(vector_store, "_collection"):
            existing = vector_store._collection.get(where={"md5_hash": chunk_hash})
            if existing and existing.get("ids"):
                return True
    except Exception:
        pass  # 查重失败不阻塞入库流程，降级为不去重
    return False


# ==================== 第二层：语义近似去重（仅同一来源内）====================

def _is_semantic_duplicate(vector_store, text: str, threshold: float, source_name: str = None) -> bool:
    """
    检查文本是否与【同一来源】的已有内容语义重复
    用相似度搜索找最接近的已有文档，如果余弦相似度超过阈值则判定重复

    【关键修复】只在同一 source 内查重，不同文件允许内容相似
    避免文件B被文件A的内容"误杀"导致返回0

    :param threshold: 余弦相似度阈值（0~1），超过则判定为重复
    :param source_name: 来源文件名，用于限定查重范围
    :return: True表示语义重复
    """
    try:
        # 只在同一来源内搜索，避免跨文件误杀
        if source_name:
            results = vector_store.similarity_search_with_score(
                text, k=1, filter={"source": source_name}
            )
        else:
            results = vector_store.similarity_search_with_score(text, k=1)

        if results:
            doc, score = results[0]
            # 【关键修复】ChromaDB 使用 cosine 距离度量
            # cosine_distance = 1 - cosine_similarity
            # 所以：cosine_similarity = 1 - score
            cosine_sim = max(0.0, 1 - score)
            if cosine_sim > threshold:
                print(f"  [语义去重] 余弦相似度={cosine_sim:.4f} > {threshold}, "
                      f"来源={source_name}, 跳过: {text[:40]}...")
                return True
    except Exception:
        pass  # 向量库为空时首次查询会失败，忽略即可
    return False


# ==================== 文档入库（含两层去重）====================

def ingest_txt_vector_store(raw_text: str, source_name: str, vector_store):
    """
    文档入库（增强版）：文本分割 → MD5去重 → 语义去重 → 向量存储

    与原版区别：
    - 每个 chunk 计算 MD5 存入 metadata，入库前查重
    - 语义相似度查重，过滤"措辞不同但内容相同"的重复
    - 返回实际入库数量（去重后的）

    :param raw_text: txt全文本
    :param source_name: 文档名称(元数据溯源)
    :param vector_store: Chroma向量数据库实例
    :return: 实际入库的切片数量
    """
    # 1. 递归文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    text_chunks = text_splitter.split_text(raw_text)

    # 2. 逐条去重
    doc_list = []
    md5_dedup_count = 0      # MD5去重命中次数
    semantic_dedup_count = 0  # 语义去重命中次数

    for chunk in text_chunks:
        # 2.1 第一层：MD5精确去重（成本最低，先做）
        chunk_hash = _compute_md5(chunk)
        if _is_md5_duplicate(vector_store, chunk_hash):
            md5_dedup_count += 1
            continue  # 跳过，不入库

        # 2.2 第二层：语义近似去重（仅在同一来源内查重，避免跨文件误杀）
        if _is_semantic_duplicate(vector_store, chunk, DEDUP_SIMILARITY_THRESHOLD, source_name):
            semantic_dedup_count += 1
            continue  # 跳过，不入库

        # 通过两层去重，封装为Document
        doc = Document(
            page_content=chunk,
            metadata={
                "source": source_name,
                "md5_hash": chunk_hash,  # 存入MD5供后续查重
            },
        )
        doc_list.append(doc)

    # 3. 向量入库（DashScope API限制：每次最多25条，需分批入库）
    if doc_list:
        BATCH_SIZE = 25
        for i in range(0, len(doc_list), BATCH_SIZE):
            batch = doc_list[i:i + BATCH_SIZE]
            vector_store.add_documents(batch)
            logger.debug("[RAG入库] 批次 %d/%d 入库 %d 条", i // BATCH_SIZE + 1,
                         (len(doc_list) + BATCH_SIZE - 1) // BATCH_SIZE, len(batch))

    # 打印去重统计（面试时可以说清楚优化效果）
    logger.info("[RAG入库] 总切片:%d, MD5去重:%d, 语义去重:%d, 实际入库:%d",
                len(text_chunks), md5_dedup_count, semantic_dedup_count, len(doc_list))
    return len(doc_list)


# ==================== 检索服务（含相关度过滤+重排）====================

def search_vector_store(vector_store: Chroma, query: str):
    """
    增强检索：扩大召回 → 相关度过滤 → 取top-K

    第三层：用 similarity_search_with_score 获取带分数的结果，过滤低分噪声
    第四层：先检索更多候选（RERANK_TOP_K），过滤后只取 retrieve_top_k 条

    与原版区别：
    - 原版直接 similarity_search(query, k=3)，无法过滤不相关结果
    - 增强版先检索 top-20，过滤掉低于阈值的结果，再取 top-3
    - 召回率高（不会漏掉相关内容），精度也高（噪声被过滤）

    :param vector_store: 向量库
    :param query: 查询文本
    :return: 过滤后的Document列表
    """
    # 扩大检索范围，给过滤留余量
    search_k = max(retrieve_top_k * 3, RERANK_TOP_K)

    try:
        results = vector_store.similarity_search_with_score(query, k=search_k)
    except Exception as e:
        logger.error("[RAG检索] 向量检索失败: %s", e)
        return []

    if not results:
        return []

    # 第三层：相关度过滤
    # 【关键修复】ChromaDB 使用 cosine 距离，cosine_similarity = 1 - score
    # 低于阈值的不送入LLM（避免噪声干扰）
    filtered_docs = []
    for doc, score in results:
        cosine_sim = max(0.0, 1 - score)
        # 打印每条结果的分数，方便排查阈值问题
        logger.info("[RAG检索] score=%.4f, 余弦相似度=%.4f, 内容: %s",
                    score, cosine_sim, doc.page_content[:30])
        if cosine_sim >= SIMILARITY_THRESHOLD:
            filtered_docs.append(doc)

    # 第四层：截断取top-K
    # 如果过滤后结果不足retrieve_top_k，放宽阈值取top-K（保证LLM至少有上下文）
    if len(filtered_docs) < retrieve_top_k:
        print(f"[RAG检索] 过滤后仅{len(filtered_docs)}条，放宽阈值取top-{retrieve_top_k}")
        filtered_docs = [doc for doc, _ in results[:retrieve_top_k]]
    else:
        filtered_docs = filtered_docs[:retrieve_top_k]
    # 打印过滤后结果（面试时可以说清楚优化效果）
    logger.info("[RAG检索] 候选%d条, 过滤后%d条", len(results), len(filtered_docs))
    return filtered_docs


#将返回的文档列表,转换为字符串（含截断）
def get_context_from_docs(docs: list[Document]) -> str:
    """
    将检索到的Document列表拼接成一段上下文文本字符串
    【Token优化】超过 MAX_CONTEXT_CHARS 时截断，防止上下文过长浪费token

    :param docs: 检索到的Document列表
    :return: 上下文文本字符串
    """
    #join是啥? 将列表中的元素用指定的分隔符拼接成一个字符串
    #这里用\n\n表示每个文档之间用两个换行符隔开
    context = "\n\n".join([doc.page_content for doc in docs])

    # 超长截断：超过最大字符数时裁剪，末尾加省略标记
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "…(已截断)"
        logger.info("[RAG上下文] 原始%d字, 截断至%d字", len(''.join([doc.page_content for doc in docs])), MAX_CONTEXT_CHARS)
    return context
