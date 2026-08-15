# streamlit 主入口,网页应用入口
import streamlit as st
import tempfile  # 用于创建临时文件,用于存储用户上传的文件
import os
import base64
from pathlib import Path  # 用于处理文件路径
from langchain_core.messages import HumanMessage  # 用于创建用户消息

# 导入自定义模块
from utils import base64_encode_image
from model_factory import create_llm, create_text_llm, create_summary_llm, create_embedding
from rag_service import init_vector_store, ingest_txt_vector_store, search_vector_store, get_context_from_docs
from prompt_template import role_prompt_dict
from memory_manager import MemoryManager
from logger import logger
from agent_tools import (
    analyze_emotion,
    get_repair_suggestions,
    get_current_time,
    create_knowledge_search_tool,
    create_agent_executor,
    TOOL_INSTRUCTIONS,
)
from config import AGENT_MAX_ITERATIONS

# ========== 缓存模型实例（避免每次点击按钮都重新创建）==========
@st.cache_resource
def _get_embedding(api_key):
    return create_embedding(api_key)

@st.cache_resource
def _get_llm(api_key):
    return create_llm(api_key)

@st.cache_resource
def _get_text_llm(api_key):
    return create_text_llm(api_key)

@st.cache_resource
def _get_summary_llm(api_key):
    return create_summary_llm(api_key)

@st.cache_resource
def _get_vector_store(api_key):
    emb = _get_embedding(api_key)
    return init_vector_store(emb)

# ========== API Key 自动获取 ==========
def _get_api_key():
    """优先从 Streamlit Secrets / .env 读取 API Key，找不到则返回空字符串"""
    # 1. Streamlit Secrets（部署在 Streamlit Cloud 时）
    try:
        if "DASHSCOPE_API_KEY" in st.secrets:
            return st.secrets["DASHSCOPE_API_KEY"]
    except Exception:
        pass
    # 2. 环境变量（本地 .env 文件，config.py 中 load_dotenv 已加载）
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if key.strip():
        return key
    # 3. 都没有，返回空（后面用侧边栏输入兜底）
    return ""

# ========== 页面配置 ==========
st.set_page_config(
    page_title="情绪修复助手",
    page_icon="💔❤️",
)


# ========== 背景图片 + 自定义样式 ==========
def _load_bg_image():
    """加载背景图片，返回 base64 编码（找不到则返回 None，不影响运行）"""
    bg_path = Path("assets/background.jpg")
    if bg_path.exists():
        with open(bg_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


bg_base64 = _load_bg_image()

# 构建 CSS
css = """
<style>
/* ===== 全局字体 ===== */
html, body, .stApp {
    font-family: "Microsoft YaHei", "PingFang SC", "Helvetica Neue", sans-serif;
}

/* ===== 主标题渐变 ===== */
.main-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ff6b6b, #ee5a6f, #c44569);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 0.8rem 0 0.3rem;
}

/* ===== 副标题 ===== */
.sub-title {
    text-align: center;
    color: #aaa;
    font-size: 0.9rem;
    margin-bottom: 1.8rem;
}

/* ===== 按钮圆角 + hover 动效 ===== */
.stButton > button {
    border-radius: 25px;
    font-weight: 600;
    font-size: 1.05rem;
    height: 3rem;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(238, 90, 111, 0.4);
}

/* ===== 输入框圆角 ===== */
.stTextArea > div > div > textarea,
.stTextInput > div > div > input {
    border-radius: 12px;
}

/* ===== 侧边栏半透明毛玻璃 ===== */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(10px);
}

/* ===== 分隔线颜色 ===== */
hr {
    border-color: rgba(238, 90, 111, 0.15) !important;
    margin: 1rem 0 !important;
}

/* ===== 聊天气泡圆角 ===== */
.stChatMessage {
    border-radius: 16px;
}
</style>
"""

# 如果有背景图，追加背景 CSS
if bg_base64:
    css += f"""
<style>
.stApp {{
    background-image: url("data:image/jpeg;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
/* 半透明白色遮罩，保证内容可读 */
.stApp::before {{
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(255, 255, 255, 0.85);
    z-index: 0;
}}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# ========== Session State 初始化 ==========
# 防止用户每次刷新页面都会把聊天记录清空
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # UI 展示用
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = None  # AI 用的记忆管理器

# ========== 侧边栏 ==========
with st.sidebar:
    # 优先从 Secrets / .env 自动获取 API Key
    _auto_key = _get_api_key()
    if _auto_key:
        api_key = _auto_key
        st.success("✅ API 已自动配置，可直接开始对话")
    else:
        st.header("🗝️ API 配置")
        if "api_key_input" not in st.session_state:
            st.session_state.api_key_input = ""
        api_key = st.text_input(
            "请输入阿里云API密钥",
            type="password",
            value=st.session_state.api_key_input,
            help="阿里云平台申请的API密钥,用于调用通义千问模型",
        )
        st.session_state.api_key_input = api_key
    st.divider()

    # RAG 知识库上传
    st.subheader("📖 RAG 知识库上传")
    uploader_txt = st.file_uploader("上传情感知识库txt文件", type=["txt"])
    if uploader_txt and api_key.strip():
        if st.session_state.get("uploaded_txt_name") != uploader_txt.name:
            txt_content = uploader_txt.read().decode("utf-8")
            vec_store = _get_vector_store(api_key)
            chunk_count = ingest_txt_vector_store(txt_content, uploader_txt.name, vec_store)
            st.session_state.uploaded_txt_name = uploader_txt.name
            st.success(f"成功导入{chunk_count}条文本片段存入向量数据库")
        else:
            st.info("该文件已导入知识库，无需重复上传")
    elif uploader_txt and not api_key.strip():
        st.warning("请先输入API密钥,再上传txt文件")

    st.divider()

    # ===== 会话状态监控 + 新建会话 =====
    if st.session_state.memory_manager:
        stats = st.session_state.memory_manager.get_session_stats()
        turns = stats["turns"]
        max_turns = stats["max_turns"]

        st.subheader("📊 会话状态")
        st.progress(
            min(turns / max_turns, 1.0),
            text=f"对话轮次: {turns} / {max_turns}",
        )

        # 根据使用比例显示不同级别的提醒
        if turns >= max_turns:
            st.error("⚠️ 当前会话已达上限，建议新建会话！")
        elif turns >= max_turns * 0.8:
            st.warning("💡 会话即将达到上限，可考虑新建会话")

        # 新建会话按钮（清空短期记忆 + UI 历史，保留长期记忆）
        if st.button("🔄 新建会话", use_container_width=True):
            st.session_state.memory_manager.reset_session()
            st.session_state.chat_history = []
            st.rerun()

    _tip_step1 = "1.直接输入文字或上传截图开始" if _auto_key else "1.先填写API密钥"
    st.info(
        "💡 使用提示:\n"
        f"{_tip_step1}\n"
        "2.上传情感知识库(可选)\n"
        "3.纯文本走Agent:自动调用\n"
        "  情绪分析/知识检索/修复建议\n"
        "4.图片走qwen-vl-max直接对话\n"
        "5.会话达上限时点击「新建会话」\n"
        "  长期记忆保留，不影响跨会话回忆"
    )

# ========== 初始化记忆管理器 ==========
if api_key.strip() and st.session_state.memory_manager is None:
    try:
        # 【Token优化】摘要压缩用 qwen-turbo（最便宜），不需要强模型
        llm_summary = _get_summary_llm(api_key)
        emb = _get_embedding(api_key)
        st.session_state.memory_manager = MemoryManager(llm_summary, emb)
    except Exception as e:
        st.error(f"记忆系统初始化失败: {e}")

# ========== 主界面 ==========
st.markdown('<div class="main-title">💢❤️💢 情绪修复助手</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Agent + RAG + 记忆机制 · 工具调用 · 多轮对话 · 知识检索 · 多模态</div>', unsafe_allow_html=True)

# 两栏布局：左 3 份输入区，右 2 份角色选择 + 按钮
col_left, col_right = st.columns([3, 2])

with col_left:
    user_text = st.text_area("写下你的心情,我们来修复它", height=140)
    upload_img = st.file_uploader("上传聊天截图(可选)", type=["jpg", "jpeg", "png"])

with col_right:
    role_list = list(role_prompt_dict.keys())
    role_select = st.radio(
        "选择助手角色",
        role_list,
        index=role_list.index("温柔体贴男"),
    )
    st.write("")  # 留白，让按钮和左侧输入框底部对齐
    st.write("")
    send_btn = st.button("开始修复❤️!", type="primary", use_container_width=True)

# ========== 展示历史对话（聊天气泡）==========
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg.get("image"):
            st.image(msg["image"])
        st.markdown(msg["content"])

# ========== 点击按钮逻辑 ==========
if send_btn:
    # 基础校验
    if not api_key.strip():
        st.warning("请先输入API密钥!")
        st.stop()
    if not user_text.strip() and upload_img is None:
        st.warning("请输入文字描述，或者上传聊天截图！")
        st.stop()

    memory_mgr = st.session_state.memory_manager
    if memory_mgr is None:
        st.error("记忆系统未初始化，请检查API密钥是否正确")
        st.stop()

    # --- 展示用户消息 ---
    with st.chat_message("user"):
        if upload_img:
            st.image(upload_img, caption="你上传的截图")
        if user_text.strip():
            st.markdown(user_text)

    # 存入 UI 历史
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_text if user_text.strip() else "(图片消息)",
        "image": upload_img.getvalue() if upload_img else None,
    })

    # --- 初始化向量库 ---
    vec_store = _get_vector_store(api_key)

    # ===== 两条路径：图片走直接调用，纯文本走 Agent =====
    temp_path = None  # 临时图片路径，finally 中统一清理
    ai_reply = None   # 最终回复内容

    if upload_img:
        # ===== 图片路径：直接用 qwen-vl-max，不走 Agent =====
        llm = _get_llm(api_key)

        # RAG 预检索（图片路径仍用预检索，Agent 不支持多模态工具调用）
        if user_text.strip():
            search_docs = search_vector_store(vec_store, user_text)
            context_text = get_context_from_docs(search_docs)
        else:
            context_text = ""

        long_term_context = memory_mgr.retrieve_relevant_memory(
            user_text if user_text.strip() else "图片对话"
        )

        full_context = ""
        if context_text:
            full_context += f"【知识库参考】\n{context_text}\n\n"
        if long_term_context:
            full_context += f"【历史对话摘要】\n{long_term_context}\n\n"

        raw_sys_prompt = role_prompt_dict[role_select]
        sys_prompt = raw_sys_prompt.format(context=full_context)

        est_tokens = int(len(sys_prompt) * 1.5 + len(user_text) * 1.5)
        logger.info("[Token估算] 模型=qwen-vl-max, 系统提示=%d字, 用户输入=%d字, 估算≈%dtoken",
                     len(sys_prompt), len(user_text), est_tokens)

        messages = memory_mgr.build_messages(sys_prompt, user_text)

        # 处理图片（压缩后再发给API，避免DashScope文件大小限制）
        from PIL import Image
        import io

        img = Image.open(upload_img)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        max_dim = 1024
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64_img = base64.b64encode(buf.getvalue()).decode()
        mime_type = "image/jpeg"
        logger.info("[图片处理] 压缩后尺寸=%dx%d, 大小≈%.0fKB", img.size[0], img.size[1], len(buf.getvalue()) / 1024)

        human_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}},
        ]
        messages[-1] = HumanMessage(content=human_content)

        try:
            with st.chat_message("assistant"):
                def _stream_image():
                    for chunk in llm.stream(messages):
                        if chunk.content:
                            yield chunk.content
                ai_reply = st.write_stream(_stream_image())
            if ai_reply:
                logger.info("[LLM] 图片对话完成，回复长度=%d字", len(ai_reply))
        except Exception as e:
            logger.error("[LLM] 图片对话请求异常: %s", e)
            st.error(f"请求异常：{e}")

    else:
        # ===== 纯文本路径：Agent + 工具调用（Function Calling + ReAct）=====
        llm = _get_text_llm(api_key)  # qwen-plus（支持 function calling）

        # 检索长期记忆（注入系统提示词，不走工具）
        long_term_context = memory_mgr.retrieve_relevant_memory(user_text)

        # 组装系统提示词 = 角色人设 + 历史记忆 + 工具使用指令
        full_context = long_term_context if long_term_context else "（暂无历史摘要）"
        raw_sys_prompt = role_prompt_dict[role_select]
        sys_prompt = raw_sys_prompt.format(context=full_context)
        sys_prompt += "\n\n" + TOOL_INSTRUCTIONS

        # 构建工具列表
        tools = [
            analyze_emotion,
            get_repair_suggestions,
            get_current_time,
            create_knowledge_search_tool(vec_store),
        ]

        # 获取聊天历史（供 Agent 理解上下文）
        chat_history = memory_mgr.get_chat_history_messages()

        est_tokens = int(len(sys_prompt) * 1.5 + len(user_text) * 1.5)
        logger.info("[Token估算] 模型=qwen-plus(Agent), 系统提示=%d字, 用户输入=%d字, 估算≈%dtoken",
                     len(sys_prompt), len(user_text), est_tokens)

        # 创建并执行 Agent（langchain 1.x create_agent API，流式输出）
        try:
            agent_messages = chat_history + [HumanMessage(content=user_text)]
            executor = create_agent_executor(llm, tools, sys_prompt, max_iterations=AGENT_MAX_ITERATIONS)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_text = ""
                for chunk, meta in executor.stream(
                    {"messages": agent_messages},
                    config={"recursion_limit": getattr(executor, "_recursion_limit", 25)},
                    stream_mode="messages",
                ):
                    if hasattr(chunk, "content") and chunk.content and not hasattr(chunk, "tool_call_id"):
                        full_text += chunk.content
                        response_placeholder.markdown(full_text)
                ai_reply = full_text if full_text else None

            if ai_reply:
                logger.info("[Agent] 执行完成，回复长度=%d字", len(ai_reply))
            else:
                raise Exception("Agent未返回内容")
        except Exception as e:
            logger.error("[Agent] 执行失败，回退到直接调用: %s", e)
            # 降级：直接用 LLM 调用（不用工具），同样流式输出
            try:
                messages = memory_mgr.build_messages(sys_prompt, user_text)
                with st.chat_message("assistant"):
                    def _stream_fallback():
                        for chunk in llm.stream(messages):
                            if chunk.content:
                                yield chunk.content
                    ai_reply = st.write_stream(_stream_fallback())
                logger.info("[Agent降级] 直接调用成功")
            except Exception as e2:
                logger.error("[Agent降级] 直接调用也失败: %s", e2)
                st.error(f"请求异常：{e2}")

    # ===== 共通：存入记忆（展示已在上方流式输出中完成）=====
    if ai_reply:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_reply,
            "image": None,
        })

        memory_mgr.add_exchange(
            user_text if user_text.strip() else "(图片消息)",
            ai_reply,
        )

    # 清理临时图片文件
    if temp_path and os.path.exists(temp_path):
        os.unlink(temp_path)
