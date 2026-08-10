# streamlit 主入口,网页应用入口
import streamlit as st
import tempfile  # 用于创建临时文件,用于存储用户上传的文件
import os
import base64
from pathlib import Path  # 用于处理文件路径
from langchain_core.messages import HumanMessage  # 用于创建用户消息

# 导入自定义模块
from utils import base64_encode_image
from model_factory import create_llm, create_text_llm, create_embedding
from rag_service import init_vector_store, ingest_txt_vector_store, search_vector_store, get_context_from_docs
from prompt_template import role_prompt_dict
from memory_manager import MemoryManager

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
            emb = create_embedding(api_key)
            vec_store = init_vector_store(emb)
            chunk_count = ingest_txt_vector_store(txt_content, "情感知识库", vec_store)
            st.session_state.uploaded_txt_name = uploader_txt.name
            st.success(f"成功导入{chunk_count}条文本片段存入向量数据库")
        else:
            st.info("该文件已导入知识库，无需重复上传")
    elif uploader_txt and not api_key.strip():
        st.warning("请先输入API密钥,再上传txt文件")

    st.divider()

    # 清空对话按钮
    if st.session_state.memory_manager and st.button("🗑️ 清空对话记忆"):
        st.session_state.memory_manager.short_term_memory = []
        st.session_state.chat_history = []
        st.rerun()

    st.info(
        "💡 使用提示:\n"
        "1.先填写API密钥\n"
        "2.上传情感知识库(可选)\n"
        "3.对话自动保留多轮记忆\n"
        "4.纯文本自动用qwen-plus省token"
    )

# ========== 初始化记忆管理器 ==========
if api_key.strip() and st.session_state.memory_manager is None:
    try:
        llm_text = create_text_llm(api_key)
        emb = create_embedding(api_key)
        st.session_state.memory_manager = MemoryManager(llm_text, emb)
    except Exception as e:
        st.error(f"记忆系统初始化失败: {e}")

# ========== 主界面 ==========
st.markdown('<div class="main-title">💢❤️💢 情绪修复助手</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">LangChain + RAG + 记忆机制 · 多轮对话 · 知识检索 · 多模态</div>', unsafe_allow_html=True)

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

    # --- 初始化模型与向量库 ---
    llm = create_llm(api_key)
    emb = create_embedding(api_key)
    vec_store = init_vector_store(emb)

    # --- RAG 检索（只有文本非空才执行）---
    if user_text.strip():
        search_docs = search_vector_store(vec_store, user_text)
        context_text = get_context_from_docs(search_docs)
    else:
        context_text = ""

    # --- 检索长期记忆 ---
    long_term_context = memory_mgr.retrieve_relevant_memory(
        user_text if user_text.strip() else "图片对话"
    )

    # --- 组装系统提示词 ---
    full_context = ""
    if context_text:
        full_context += f"【知识库参考】\n{context_text}\n\n"
    if long_term_context:
        full_context += f"【历史对话摘要】\n{long_term_context}\n\n"

    raw_sys_prompt = role_prompt_dict[role_select]
    sys_prompt = raw_sys_prompt.format(context=full_context)

    # --- 构建消息列表 ---
    messages = memory_mgr.build_messages(sys_prompt, user_text)

    # --- 处理图片（多模态）---
    temp_path = None
    if upload_img:
        file_ext = os.path.splitext(upload_img.name)[1] or ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        temp_file.write(upload_img.read())
        temp_file.close()
        temp_path = Path(temp_file.name)
        b64_img = base64_encode_image(str(temp_path))

        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        mime_type = mime_map.get(file_ext.lower(), "image/jpeg")

        human_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}},
        ]
        # 用带图片的消息替换最后一条纯文本 HumanMessage
        messages[-1] = HumanMessage(content=human_content)

    # --- 调用模型生成回复 ---
    try:
        with st.spinner("AI正在思考中..."):
            response = llm.invoke(messages)
            ai_reply = response.content

            # 展示 AI 回复（聊天气泡）
            with st.chat_message("assistant"):
                st.markdown(ai_reply)

            # 存入 UI 历史
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_reply,
                "image": None,
            })

            # 存入记忆管理器（超限自动触发摘要压缩）
            memory_mgr.add_exchange(
                user_text if user_text.strip() else "(图片消息)",
                ai_reply,
            )

    except Exception as e:
        st.error(f"请求异常：{e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
