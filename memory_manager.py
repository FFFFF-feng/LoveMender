# memory_manager.py
# 承担短期记忆管理、长期记忆检索以及消息的构建（组装成 LLM 能理解的格式）

import time
import chromadb
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import CHROMA_DB_DIR
from logger import logger

# 定义与 RAG 知识库数据存储独立的长期记忆集合名称
MEMORY_COLLECTION_NAME = "long_term_memory"


class MemoryManager:
    def __init__(self, llm, embedding_model):
        """
        初始化记忆管理器
        :param llm: 纯文本大语言模型，用于生成对话摘要
        :param embedding_model: 嵌入模型，用于将文本转向量（长期记忆检索用）
        """
        self.llm = llm
        self.embedding_model = embedding_model

        # 1. 短期记忆：保存在内存列表中，记录最近的对话
        self.short_term_memory = []
        self.max_history_len = 20  # 短期记忆最大条数（即 10 轮对话）

        # 2. 长期记忆：初始化 ChromaDB 持久化客户端，指向统一的数据库目录
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

        # 获取或创建"长期记忆"专用的集合（防止首次运行因集合不存在而报错）
        # 注意：不指定 embedding_function，add/query 时手动传入预计算的向量
        self.long_term_collection = self.chroma_client.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # 使用cosine距离，与RAG保持一致
        )

    def add_exchange(self, user_text, ai_text):
        """
        每追加一轮对话，并在超限时触发压缩存入长期记忆
        :param user_text: 用户的输入
        :param ai_text: AI 的回复
        """
        # 追加对话到短期记忆列表
        self.short_term_memory.append({"role": "human", "content": user_text})
        self.short_term_memory.append({"role": "ai", "content": ai_text})

        # 如果短期记忆列表超过最大长度，触发压缩机制
        if len(self.short_term_memory) > self.max_history_len:
            self._compress_old_memory()

    def _compress_old_memory(self):
        """
        压缩最早的两轮对话（4条消息），生成摘要存入短期记忆开头及长期记忆数据库中
        """
        # 提取最早的两轮对话（4条消息）
        batch = self.short_term_memory[:4]

        # 将对话内容转换为字符串（注意：Python 换行符是 \n）
        conversation_text = "\n".join(
            [f"{'用户' if m['role'] == 'human' else 'AI'}: {m['content']}" for m in batch]
        )

        # 生成摘要提示词
        summary_prompt = (
            "请将以下对话内容压缩为一段不要超过100字的摘要，保留关键情绪、问题和解决建议。"
            "不要加任何前缀或解释，直接输出摘要。\n\n"
            f"{conversation_text}"
        )

        # 调用 LLM 生成摘要
        try:
            # invoke 必须传入消息列表，不能传纯字符串
            summary = self.llm.invoke([HumanMessage(content=summary_prompt)])
            # 防御性编程：兼容返回 AIMessage 或纯字符串的情况
            summary_text = summary.content if hasattr(summary, "content") else str(summary)
        except Exception as e:
            logger.error("[Memory] 压缩旧记忆时出错: %s", e)
            summary_text = "历史对话摘要生成失败"

        # 将摘要作为系统消息插入到短期记忆的最开头
        self.short_term_memory.insert(0, {"role": "system", "content": summary_text})

        # 删除最早的两轮对话（4条消息）
        del self.short_term_memory[:4]

        # 将摘要存入长期记忆向量数据库
        if summary_text.strip():
            try:
                # 【关键修复】用 embedding_model 预计算向量
                # 之前直接传 documents 让 chromadb 自动向量化，
                # 但 collection 没有指定 embedding_function，
                # chromadb 会尝试加载默认 ONNX 模型，导致卡死
                summary_embedding = self.embedding_model.embed_documents([summary_text])[0]
                self.long_term_collection.add(
                    documents=[summary_text],
                    embeddings=[summary_embedding],  # 传入预计算的向量
                    metadatas=[{"type": "summary", "timestamp": time.time()}],
                    ids=[f"mem_{int(time.time() * 1000)}"]  # 使用时间戳生成唯一 ID
                )
            except Exception as e:
                logger.error("[Memory] 存入长期记忆失败: %s", e)

    def retrieve_relevant_memory(self, query_text, top_k=3):
        """
        根据当前用户输入，从长期记忆中检索最相关的历史摘要
        :param query_text: 用户当前输入
        :param top_k: 返回的最相关记忆数量，默认 3 条
        :return: 拼接后的摘要字符串
        """
        if not query_text.strip():
            return ""

        try:
            # 【关键修复】用 embedding_model 预计算查询向量
            # 之前用 query_texts 让 chromadb 自动向量化，
            # 但 collection 没有指定 embedding_function，
            # chromadb 会尝试加载默认 ONNX 模型，导致卡死或长时间无响应
            query_embedding = self.embedding_model.embed_query(query_text)
            results = self.long_term_collection.query(
                query_embeddings=[query_embedding],  # 传入预计算的向量
                n_results=top_k
            )

            # 格式化返回结果
            if results['documents'] and results['documents'][0]:
                return "\n".join(results['documents'][0])
        except Exception as e:
            logger.error("[Memory] 检索长期记忆失败: %s", e)

        return ""

    def build_messages(self, system_prompt, current_text):
        """
        组装最终的提示词，包含系统提示词、历史消息和当前用户输入
        格式: [SystemMessage, 历史消息..., 当前用户输入]
        :param system_prompt: 角色设定与上下文提示词
        :param current_text: 当前用户最新输入
        :return: 组装好的 LangChain 消息列表
        """
        messages = [SystemMessage(content=system_prompt)]

        # 遍历短期记忆，将其转换为 LangChain 消息对象
        for msg in self.short_term_memory:
            if msg['role'] == 'human':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'ai':
                messages.append(AIMessage(content=msg['content']))
            elif msg['role'] == 'system':
                # 压缩产生的摘要，作为系统消息插入，提供背景信息
                messages.append(SystemMessage(content=msg['content']))

        # 加入当前用户最新输入
        messages.append(HumanMessage(content=current_text))

        return messages

    def get_chat_history_messages(self):
        """
        返回短期记忆的消息列表（仅历史，不含系统提示和当前输入）
        供 AgentExecutor 的 chat_history 参数使用
        """
        messages = []
        for msg in self.short_term_memory:
            if msg['role'] == 'human':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'ai':
                messages.append(AIMessage(content=msg['content']))
            elif msg['role'] == 'system':
                messages.append(SystemMessage(content=msg['content']))
        return messages

    def get_session_stats(self):
        """
        返回当前会话的统计信息，用于 UI 展示和提醒
        :return: dict {"turns": 对话轮次, "max_turns": 最大轮次, "usage_ratio": 使用比例}
        """
        turns = len(self.short_term_memory) // 2  # 每轮 = 用户 + AI 两条消息
        max_turns = self.max_history_len // 2
        return {
            "turns": turns,
            "max_turns": max_turns,
            "usage_ratio": turns / max_turns if max_turns > 0 else 0,
        }

    def reset_session(self):
        """
        重置会话：清空短期记忆，保留长期记忆
        用于用户新建会话时调用，长期记忆（历史摘要）跨会话保留
        """
        self.short_term_memory = []
        logger.info("[Memory] 会话已重置，长期记忆保留")
