#RAG服务
# 向量库,文档入库,检索服务
from pathlib import Path  #用于路径操作
from langchain_community.vectorstores import Chroma  #用于向量数据库
from langchain_text_splitters import RecursiveCharacterTextSplitter  #用于文本分割
from langchain_core.documents import Document  #用于文档操作
from config import (
    CHROMA_DB_DIR,
    chunk_size,
    chunk_overlap,
    retrieve_top_k
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
        embedding_function=embeddings,
    )
    return vector_store


#文档入库函数,输入原始文本,文档来源,向量库
def ingest_txt_vector_store(raw_text: str, source_name: str, vector_store):
    """
    文档入库完整的一个流程:文本分割->封装成Document->向量嵌入->向量数据库存储->持久化
    :param raw_text: txt全文本
    :param source_name: 文档名称(元数据溯源)
    :param vector_store: Chroma向量数据库,创建的一个实例
    :return: 切片总数量
    """
    #递归文本分割器
    #下面是初始化文本分割器,设置每个切片的最大字符数和切片之间的重叠字符数
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  #每个切片的最大字符数
        chunk_overlap=chunk_overlap,  #切片之间的重叠字符数
    )
    #将文本分割成多个切片
    # ==========【修复】split() 方法不存在，正确方法是 split_text() ==========
    text_chunks = text_splitter.split_text(raw_text)  #分割成字符串列表[切片1,切片2,切片3,...]
    #封装成Document对象:
    doc_list = []
    for chunk in text_chunks:
        #与纯字符文本不同,切片之间有重叠,并且还有任意自定义元数据
        doc = Document(
            page_content=chunk,  #文档内容
            metadata={"source": source_name},  #文档元数据
        )
        doc_list.append(doc)  # 把文档加入列表

    #将Document对象添加到向量数据库
    vector_store.add_documents(doc_list)
    # vector_store.persist()  # 新版langchain已废弃，注释或删除
    return len(doc_list)  #返回切片总数量


#设置向量数据库的检索参数
def search_vector_store(vector_store: Chroma, query: str):
    #k表示返回的文档数量,相似度检索的方法是余弦相似度
    docs = vector_store.similarity_search(query, k=retrieve_top_k)
    return docs


#将返回的文档列表,转换为字符串
def get_context_from_docs(docs: list[Document]) -> str:
    """
    将检索到的Document列表拼接成一段上下文文本字符串
    :param docs: 检索到的Document列表
    :return: 上下文文本字符串
    """
    #join是啥? 将列表中的元素用指定的分隔符拼接成一个字符串
    #这里用\n\n表示每个文档之间用两个换行符隔开
    return "\n\n".join([doc.page_content for doc in docs])
