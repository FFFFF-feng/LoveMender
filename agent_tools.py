#Agent工具模块+AgentExecutor工厂
#让LLM自主决定调用哪些工具(Function Calling+ReAct模式)

#工具清单:
"""
1.analyze_emotion(text) -情绪分析,关键词匹配,不需要调用llm
2.get_repair_suggestion(text) -根据情绪分析结果,给出修复建议,根据知识库内容和截图描述
3.get_current_time() -获取当前时间,格式为YYYY-MM-DD HH:MM:SS,不需要调用llm
4.search_knowledge_base(text) -根据用户问题,搜索知识库,返回相关文档
"""
from datetime import datetime
#导入工具装饰器,用于定义工具
from langchain_core.tools import tool
#导入提示词模板,用于生成模型的提示词
#from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
#导入AgentExecutor工厂,用于创建智能体执行器
from langchain.agents import create_agent
from logger import logger

#工具1:情绪分析
@tool
def analyze_emotion(text)->str:
    """
    分析用户文本中的情绪状态,返回情绪类型和强度
    当用户表达负面情绪或倾诉烦恼时调用此工具
    """
    emotion_keywords = {
        "愤怒": ["生气", "愤怒", "气死", "恼火", "暴怒", "烦死", "讨厌", "凭什么"],
        "悲伤": ["难过", "伤心", "哭", "心痛", "崩溃", "绝望", "失落", "想哭"],
        "焦虑": ["焦虑", "紧张", "害怕", "担心", "恐惧", "不安", "压力", "心慌"],
        "委屈": ["委屈", "不公平", "不被理解", "付出", "白费", "没人懂"],
        "嫉妒": ["嫉妒", "羡慕", "比较", "不如", "凭什么他"],
        "孤独": ["孤独", "寂寞", "一个人", "没人陪", "空虚"],
    }
    #初始化检测到的情绪列表
    detecteed=[]
    #遍历每个情绪类型
    for emotion,keywords in emotion_keywords.items():
        #统计文本中包含该情绪的关键词数量
        count=sum(1 for keyword in keywords if keyword in text)
        if count>0:
            #根据关键词数量判断情绪强度
            if count >= 2:
                intensity="强烈"
            else:
                intensity="中等"
            #将检测到的情绪类型和强度添加到列表中
            detecteed.append(f"{emotion}(强度:{intensity},命中:{count}个关键词)")
    if not detecteed:
        result=f"未检测到任何情绪,用户可能没有表达情绪或表达的情绪不明显"
    else:
        result=f"检测到以下情绪:{'; '.join(detecteed)}"
    #日志记录
    logger.info("[工具] analyze_emotion-%s", result)
    return result

#工具2:根据情绪分析结果,给出修复建议
@tool
def get_repair_suggestion(emotion_type:str,text:str)->str:
    """
    根据情绪分析结果,给出修复建议
    emotion_type参数为情绪类型关键词,如"愤怒","悲伤"等
    """
    suggestion={
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
    for key,tips in suggestion.items():
        if key in emotion_type:
            result=f"针对{key}情绪,建议:\n"+"\n".join(
                #enumerate()函数用于将可迭代对象(如列表、元组等)转换为一个索引序列,同时返回数据的索引和数据
                #i+1是为了将索引从1开始,而不是从0开始
                f"{i+1}.{s}" for i,s in enumerate(tips)
                #也就是比如:
                #1.深呼吸练习：吸气4秒 → 屏气4秒 → 呼气6秒，重复5次
                #2.物理释放：快走15分钟或做20个俯卧撑，用运动代谢肾上腺素
                #3.书写发泄：把愤怒的想法写在纸上，写完撕掉，象征性释放
            )
            logger.info("[工具] get_repair_suggestion-%s", key)
            return result
    result="通用建议:深呼吸三次,给自己一个拥抱,告诉自己[这都会过去的]"
    logger.info("[工具] get_repair_suggestion->通用建议")
    return result

#工具3:时间问候
@tool
def get_current_time():
    """
    获取当前时间和时段问候语.
    当需要在回复中给出时段问候时调用
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
    result=f"当前时间:{now.strftime('%Y-%m-%d %H:%M:%S')}, {greeting}"
    logger.info("[工具] get_current_time-%s", result)
    return result

#工具4:知识库引用
def create_knowledge_search_tool(vector_store):
    """
    创建知识库检索工具(需要传入向量数据库的实例对象)
    用工厂函数是因为vector_store在运行时才创建,不能直接用@tool装饰器
    """
    @tool
    def knowledge_search_base(query: str) -> str:
        """
        搜索情感修复知识库,获取心理学和情感修复的专业知识.
        当需要专业知识或方法来帮助用户调用时使用
        """
        from rag_service import search_vector_store,get_context_from_docs
        docs=search_vector_store(vector_store,query)
        if not docs:
            logger.info("[工具] knowledge_search_base->未找到相关文档")
            return "知识库未找到相关文档"
        context=get_context_from_docs(docs)
        logger.info("[工具] knowledge_search_base ->检索到%d条结果",len(docs))
        return f"知识库检索结果:\n{context}"

    return knowledge_search_base

#Agent工具使用指令:
#拼接到角色提示词后面,格式为:
TOOL_INSTRUCTIONS="""
你可以调用以下工具来更好的帮助用户:
1.get_repair_suggestion(text) -根据用户情绪,给出情感修复建议
2.get_current_time() -获取当前时间和时段问候语
3.knowledge_search_base(text) -根据用户问题,搜索知识库,返回相关文档
4.analyze_emotion(text) -分析用户情绪状态

工具使用策略:
- 用户表达情绪时，先调用 analyze_emotion 分析情绪类型和强度
- 需要专业知识时，调用 search_knowledge_base 搜索
- 根据情绪类型，调用 get_repair_suggestions 获取建议
- 可以在回复开头用 get_current_time 给出时段问候

最终回复要求:
- 结合工具返回的信息，给出温暖、有共情力的回复
- 不要机械罗列工具结果，要自然融入对话
- 回复控制在300字以内
""".strip()

#拼接到角色提示词后面,格式为:
#max_iterations: 最大迭代次数,默认3次
def create_agent_executor(llm,tools,system_prompt,max_iterations=3):
    """
    创建Agent执行器
    底层基于LangGraph,支持Function Calling+ReAct推理循环
    :param llm: 语言模型实例对象
    :param tools: 工具列表
    :param system_prompt: 系统提示词
    :param max_iterations: 最大迭代次数
    :return: CompiledGraph 可直接.invoke()调用
    """
    agent=create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )
    logger.info("Agent创建完成,工具数:%d",len(tools))
    return agent
