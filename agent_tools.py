# agent_tools.py
# Agent 工具定义 + AgentExecutor 工厂
# 让 LLM 自主决定调用哪些工具（Function Calling + ReAct 推理循环）
#
# 工具清单：
#   1. analyze_emotion(text)        - 情绪分析（关键词匹配，无 LLM 调用）
#   2. get_repair_suggestions(type) - 修复建议（预定义知识库）
#   3. get_current_time()           - 时间问候
#   4. search_knowledge_base(query) - RAG 知识库检索（工厂函数，需 vector_store）

from datetime import datetime
from langchain_core.tools import tool
from langchain.agents import create_agent

from logger import logger


# ==================== 工具 1：情绪分析 ====================

@tool
def analyze_emotion(text: str) -> str:
    """
    分析用户输入文本中的情绪状态，返回情绪类型和强度。
    当用户表达负面情绪或倾诉烦恼时调用此工具。
    """
    emotion_keywords = {
        "愤怒": ["生气", "愤怒", "气死", "恼火", "暴怒", "烦死", "讨厌", "凭什么"],
        "悲伤": ["难过", "伤心", "哭", "心痛", "崩溃", "绝望", "失落", "想哭"],
        "焦虑": ["焦虑", "紧张", "害怕", "担心", "恐惧", "不安", "压力", "心慌"],
        "委屈": ["委屈", "不公平", "不被理解", "付出", "白费", "没人懂"],
        "嫉妒": ["嫉妒", "羡慕", "比较", "不如", "凭什么他"],
        "孤独": ["孤独", "寂寞", "一个人", "没人陪", "空虚"],
    }

    detected = []
    for emotion, keywords in emotion_keywords.items():
        count = sum(1 for kw in keywords if kw in text)
        if count > 0:
            intensity = "强烈" if count >= 2 else "一般"
            detected.append(f"{emotion}(强度:{intensity},命中{count}个关键词)")

    if not detected:
        result = "未检测到明显负面情绪，用户可能处于平静或倾诉状态"
    else:
        result = f"检测到情绪: {'; '.join(detected)}"

    logger.info("[工具] analyze_emotion → %s", result)
    return result


# ==================== 工具 2：修复建议 ====================

@tool
def get_repair_suggestions(emotion_type: str) -> str:
    """
    根据情绪类型生成具体的修复建议。
    emotion_type 参数应为情绪类型关键词，如：愤怒、悲伤、焦虑、委屈、嫉妒、孤独。
    """
    suggestions = {
        "愤怒": [
            "深呼吸练习：吸气4秒 → 屏气4秒 → 呼气6秒，重复5次",
            "物理释放：快走15分钟或做20个俯卧撑，用运动代谢肾上腺素",
            "书写发泄：把愤怒的想法写在纸上，写完撕掉，象征性释放",
        ],
        "悲伤": [
            "自我拥抱：双臂交叉抱住自己，轻拍肩膀30秒，激活安抚反射",
            "温暖疗法：喝一杯热可可或热茶，用温度带来生理性慰藉",
            "社交连接：给信任的朋友发一条消息，不需要长聊，一句就好",
        ],
        "焦虑": [
            "5-4-3-2-1接地法：说出5个看到的、4个触摸到的、3个听到的、2个闻到的、1个尝到的",
            "渐进放松：从脚趾到头顶，逐个部位收紧5秒再放松",
            "信息断食：关闭手机通知15分钟，切断焦虑源",
        ],
        "委屈": [
            "情绪日记：写下事件经过和感受，不评判对错，只记录",
            "自我肯定：列出自己的三个优点或近期成就，对抗自我否定",
            "安全倾诉：找一个安全空间哭一场或大声说出来",
        ],
        "嫉妒": [
            "自我觉察：写下嫉妒的根源，区分「想要」和「需要」",
            "感恩练习：列出三件目前拥有且感恩的事",
            "行动转化：把嫉妒的能量转化为提升自己的动力",
        ],
        "孤独": [
            "自我对话：对着镜子说三件今天做得好的事",
            "环境改变：去咖啡馆或公园，感受他人的存在",
            "线上社区：浏览感兴趣的论坛或社群，发一条帖子",
        ],
    }

    for key, tips in suggestions.items():
        if key in emotion_type:
            result = f"针对【{key}】的修复建议:\n" + "\n".join(
                f"  {i + 1}. {s}" for i, s in enumerate(tips)
            )
            logger.info("[工具] get_repair_suggestions → %s", key)
            return result

    result = "通用建议: 深呼吸3次，给自己一个拥抱，告诉自己「这都会过去的」"
    logger.info("[工具] get_repair_suggestions → 通用建议")
    return result


# ==================== 工具 3：时间问候 ====================

@tool
def get_current_time() -> str:
    """
    获取当前时间和时段问候语。
    当需要在回复中给出时段问候（如"晚上好"）时调用。
    """
    now = datetime.now()
    hour = now.hour

    if 5 <= hour < 11:
        greeting = "早上好，新的一天充满了可能性"
    elif 11 <= hour < 14:
        greeting = "中午好，记得吃饭和休息"
    elif 14 <= hour < 18:
        greeting = "下午好，辛苦了"
    elif 18 <= hour < 23:
        greeting = "晚上好，今天过得怎么样"
    else:
        greeting = "夜深了，注意休息，明天还有希望"

    result = f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}，{greeting}"
    logger.info("[工具] get_current_time → %s", result)
    return result


# ==================== 工具 4：知识库检索（工厂函数）====================

def create_knowledge_search_tool(vector_store):
    """
    创建知识库检索工具（需要传入 vector_store 实例）
    用工厂函数是因为 vector_store 在运行时才创建，不能直接用 @tool
    """
    @tool
    def search_knowledge_base(query: str) -> str:
        """
        搜索情感修复知识库，获取心理学和情感修复的专业知识。
        当需要专业知识或方法来帮助用户时调用此工具。
        """
        from rag_service import search_vector_store, get_context_from_docs

        docs = search_vector_store(vector_store, query)
        if not docs:
            logger.info("[工具] search_knowledge_base → 无结果")
            return "知识库中未找到相关内容"

        context = get_context_from_docs(docs)
        logger.info("[工具] search_knowledge_base → 检索到%d条结果", len(docs))
        return f"知识库检索结果:\n{context}"

    return search_knowledge_base


# ==================== Agent 工具使用指令 ====================
# 拼接到角色提示词后面，告诉 LLM 如何使用工具

TOOL_INSTRUCTIONS = """

【Agent工具使用指令】
你可以调用以下工具来更好地帮助用户：
1. analyze_emotion(text) - 分析用户情绪状态
2. search_knowledge_base(query) - 搜索情感知识库
3. get_repair_suggestions(emotion_type) - 获取具体修复建议
4. get_current_time() - 获取当前时间和问候

工具使用策略：
- 用户表达情绪时，先调用 analyze_emotion 分析情绪类型和强度
- 需要专业知识时，调用 search_knowledge_base 搜索
- 根据情绪类型，调用 get_repair_suggestions 获取建议
- 可以在回复开头用 get_current_time 给出时段问候

最终回复要求：
- 结合工具返回的信息，给出温暖、有共情力的回复
- 不要机械罗列工具结果，要自然融入对话
- 回复控制在300字以内
""".strip()


# ==================== Agent 工厂（langchain 1.x API）====================

def create_agent_executor(llm, tools, system_prompt, max_iterations=3):
    """
    创建 Agent（langchain 1.x create_agent API）
    底层基于 LangGraph，支持 Function Calling + ReAct 推理循环

    :param llm: 支持工具调用的 LLM（如 qwen-plus）
    :param tools: 工具列表
    :param system_prompt: 系统提示词（角色人设 + 记忆上下文 + 工具指令）
    :param max_iterations: 最大推理循环次数（通过 recursion_limit 控制）
    :return: CompiledStateGraph（可直接 .invoke）
    """
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    # langchain 1.x: 通过 recursion_limit 控制最大循环次数
    # 每次工具调用 = 2步（LLM决策 + 工具执行），+5 缓冲
    agent._recursion_limit = max_iterations * 2 + 5

    logger.info("[Agent] Agent创建成功，工具数: %d, 最大迭代: %d", len(tools), max_iterations)
    return agent
