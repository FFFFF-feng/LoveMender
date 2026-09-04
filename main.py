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
    MCP_TOOL_GUIDANCE,
)
from config import AGENT_MAX_ITERATIONS
from mcp_manager import get_mcp_tools
from persona import (
    ManualImporter,
    WeChatImporter,
    WeChatDBImporter,
    PersonaExtractor,
    PersonaManager,
    PersonaPromptGenerator,
    ExtractionDimension,
)

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

@st.cache_resource
def _get_mcp_tools():
    """
    初始化 MCP 工具并缓存
    注意：MCP 初始化是异步的，用 asyncio.run 在同步环境里跑
    返回 LangChain 格式的工具列表（可能为空，取决于配置和 Token）
    """
    import asyncio
    try:
        # asyncio.run 会创建一个新的事件循环来跑异步函数
        # 跑完后自动关闭循环，不会影响 Streamlit 的主线程
        tools = asyncio.run(get_mcp_tools())
        return tools
    except Exception as e:
        from logger import logger
        logger.error("[MCP] 初始化失败: %s", e)
        return []

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
if "persona_enabled" not in st.session_state:
    st.session_state.persona_enabled = False  # 人格化开关
if "persona_slug" not in st.session_state:
    st.session_state.persona_slug = ""  # 当前使用的画像标识
if "persona_manager" not in st.session_state:
    st.session_state.persona_manager = None  # 画像管理器实例
if "persona_generator" not in st.session_state:
    st.session_state.persona_generator = None  # 提示词生成器实例

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

    # ===== MCP 工具状态显示 =====
    if api_key.strip():
        mcp_tools = _get_mcp_tools()
        if mcp_tools:
            st.success(f"🛠️ GitHub 工具已加载 ({len(mcp_tools)}个)")
        else:
            github_token = os.environ.get("GITHUB_TOKEN", "")
            if not github_token.strip():
                st.info("💡 配置 GITHUB_TOKEN 后可启用 GitHub 开发工具")
            else:
                st.warning("⚠️ GitHub 工具加载失败，检查 Node.js 是否安装")

    st.divider()

    # ===== 人格画像管理 =====
    st.subheader("🎭 人格画像（对方性格克隆）")

    # 初始化画像管理器
    if api_key.strip() and st.session_state.persona_manager is None:
        st.session_state.persona_manager = PersonaManager(root_dir="persona_data")

    persona_mgr = st.session_state.persona_manager

    if persona_mgr:
        # 列出已有画像
        existing_personas = persona_mgr.list_personas()

        if existing_personas:
            selected = st.selectbox(
                "选择对方画像",
                options=["（不使用）"] + existing_personas,
                index=0 if not st.session_state.persona_enabled else (
                    existing_personas.index(st.session_state.persona_slug) + 1
                    if st.session_state.persona_slug in existing_personas else 0
                ),
            )
            if selected == "（不使用）":
                if st.session_state.persona_enabled:
                    st.session_state.persona_enabled = False
                    st.session_state.persona_slug = ""
                    st.session_state.persona_generator = None
            else:
                if not st.session_state.persona_enabled or st.session_state.persona_slug != selected:
                    st.session_state.persona_enabled = True
                    st.session_state.persona_slug = selected
                    st.session_state.persona_generator = PersonaPromptGenerator(
                        selected, persona_mgr
                    )
                # 显示画像信息
                manifest = persona_mgr.load_persona(selected)
                if manifest:
                    st.info(
                        f"👤 {manifest.name} · "
                        f"{manifest.total_messages}条消息 · "
                        f"{len(manifest.dimensions)}个维度"
                    )
        else:
            st.info("暂无画像，上传聊天记录生成一个吧")

        st.divider()

        # 生成新画像（参考 LangChain/immortal-skill/WeClone 方案）
        st.markdown("**生成新画像**")
        import_tab = st.radio(
            "导入方式",
            ["上传文件(CSV/JSON/TXT)", "粘贴聊天记录"],
            horizontal=True,
            label_visibility="collapsed",
        )

        # 共用的画像生成函数
        def _generate_persona(importer, raw_data, target_name, source_label):
            """通用画像生成流程：解析→清洗→提取→保存"""
            import re as _re
            with st.spinner("正在分析聊天记录...这可能需要 1-3 分钟"):
                # Step 1: 解析+清洗
                if isinstance(raw_data, str) and raw_data.startswith("__file__:"):
                    file_path = Path(raw_data[7:])
                    messages = importer.parse_file(file_path)
                else:
                    messages = importer.parse(raw_data)

                clean_msgs = importer.clean(messages)
                corpus_md = importer.to_corpus_markdown(
                    clean_msgs, target_person=target_name
                )
                st.info(f"📝 解析完成：{len(clean_msgs)} 条消息")

                if len(clean_msgs) < 5:
                    st.warning("消息太少（不足5条），可能无法有效提取人格特征")
                    return

                # Step 2: 人格提取
                extractor = PersonaExtractor(api_key)
                results = extractor.extract_all(
                    corpus_md, target_name=target_name
                )
                st.info("🧠 人格提取完成")

                # Step 3: 生成 slug
                slug_base = _re.sub(r'[^\w]', '', target_name)
                if not slug_base:
                    slug_base = "persona"
                slug = slug_base
                counter = 1
                while slug in existing_personas:
                    slug = f"{slug_base}_{counter}"
                    counter += 1

                # Step 4: 保存画像
                manifest = persona_mgr.create_persona(
                    slug=slug,
                    name=target_name,
                    results=results,
                    sources=[source_label],
                    platforms=["wechat"],
                    persona_type="partner",
                    total_messages=len(clean_msgs),
                )

                st.session_state.persona_enabled = True
                st.session_state.persona_slug = slug
                st.session_state.persona_generator = PersonaPromptGenerator(
                    slug, persona_mgr
                )

                st.success(f"✅ 画像「{target_name}」生成完成！已自动启用")
                logger.info("[人格画像] 生成完成：%s (%d条消息)", slug, len(clean_msgs))

        if import_tab == "上传文件(CSV/JSON/TXT)":
            new_persona_name = st.text_input(
                "对方昵称/称呼",
                value=st.session_state.get("new_persona_name", ""),
                placeholder="如：小美、宝宝",
                help="用于画像命名和提示词中指代对方",
            )
            st.session_state.new_persona_name = new_persona_name

            chat_upload = st.file_uploader(
                "上传聊天记录文件",
                type=["csv", "json", "txt"],
                help=(
                    "支持 WeChatMsg/PyWxDump/chatlog-keeper 等工具导出的文件\n"
                    "CSV: 发送者,内容,时间 或 PyWxDump 格式\n"
                    "JSON: 消息数组格式\n"
                    "TXT: 微信复制格式或 immortal-skill 格式"
                ),
            )

            if chat_upload and new_persona_name.strip() and api_key.strip():
                if st.button(
                    "✨ 生成对方人格画像",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.get("persona_generating", False),
                ):
                    st.session_state.persona_generating = True
                    try:
                        # 保存上传文件到临时路径，再用 WeChatImporter 解析
                        tmp_dir = Path(tempfile.mkdtemp())
                        tmp_path = tmp_dir / chat_upload.name
                        tmp_path.write_bytes(chat_upload.getvalue())

                        importer = WeChatImporter()
                        _generate_persona(
                            importer,
                            f"__file__:{tmp_path}",
                            new_persona_name.strip(),
                            chat_upload.name,
                        )
                    except Exception as e:
                        logger.error("[人格画像] 生成失败：%s", e, exc_info=True)
                        st.error(f"生成失败：{e}")
                    finally:
                        st.session_state.persona_generating = False
                        st.rerun()

            elif chat_upload and not new_persona_name.strip():
                st.warning("请先填写对方昵称")
            elif chat_upload and not api_key.strip():
                st.warning("请先配置 API Key")

        else:  # 粘贴聊天记录
            st.caption(
                "💡 操作方法：在微信中选中消息 → Ctrl+C 复制 → 粘贴到下方文本框\n"
                "参考 LangChain WeChatChatLoader 方案，支持微信复制格式"
            )

            new_persona_name = st.text_input(
                "对方昵称/称呼",
                value=st.session_state.get("new_persona_name_paste", ""),
                placeholder="如：小美、宝宝",
                key="persona_name_paste",
            )
            st.session_state.new_persona_name_paste = new_persona_name

            pasted_text = st.text_area(
                "粘贴聊天记录",
                height=250,
                placeholder=(
                    "在此粘贴从微信复制的聊天记录...\n\n"
                    "示例格式：\n"
                    "张三 2025/1/15 14:30\n\n"
                    "你好啊\n\n"
                    "李四 2025/1/15 14:31\n\n"
                    "最近怎么样"
                ),
            )

            if pasted_text.strip() and new_persona_name.strip() and api_key.strip():
                if st.button(
                    "✨ 生成对方人格画像",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.get("persona_generating", False),
                ):
                    st.session_state.persona_generating = True
                    try:
                        importer = WeChatImporter()
                        _generate_persona(
                            importer,
                            pasted_text.strip(),
                            new_persona_name.strip(),
                            "微信粘贴导入",
                        )
                    except Exception as e:
                        logger.error("[人格画像] 生成失败：%s", e, exc_info=True)
                        st.error(f"生成失败：{e}")
                    finally:
                        st.session_state.persona_generating = False
                        st.rerun()

            elif pasted_text.strip() and not new_persona_name.strip():
                st.warning("请先填写对方昵称")
            elif pasted_text.strip() and not api_key.strip():
                st.warning("请先配置 API Key")

    else:
        st.info("💡 配置 API Key 后可使用人格画像功能")

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


# ========== 人格化提示词辅助 ==========
def _build_persona_addon(user_text: str = "") -> str:
    """如果开启了人格画像，生成人格化提示词附加段

    返回可直接拼接到系统提示词后的字符串，未开启则返回空。
    """
    if not st.session_state.persona_enabled or not st.session_state.persona_generator:
        return ""

    # 简单情绪判断（关键词匹配，避免额外调用 LLM）
    emotion = "neutral"
    angry_kw = ["生气", "吵架", "骂", "冷暴力", "不理", "道歉", "认错", "矛盾", "不满", "气", "火"]
    sad_kw = ["难过", "委屈", "哭", "伤心", "不开心", "失落", "难受"]
    happy_kw = ["开心", "高兴", "快乐", "幸福", "甜", "想你", "爱"]
    anxious_kw = ["担心", "害怕", "不安", "焦虑", "紧张", "压力", "怕"]

    text_lower = user_text
    if any(kw in text_lower for kw in angry_kw):
        emotion = "angry"
    elif any(kw in text_lower for kw in sad_kw):
        emotion = "sad"
    elif any(kw in text_lower for kw in happy_kw):
        emotion = "happy"
    elif any(kw in text_lower for kw in anxious_kw):
        emotion = "anxious"

    persona_prompt = st.session_state.persona_generator.generate(
        user_message=user_text,
        emotion=emotion,
    )
    return persona_prompt.to_system_prompt_addon()

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

        # 注入人格化提示词（如果开启了画像）
        persona_addon = _build_persona_addon(user_text)
        if persona_addon:
            sys_prompt += "\n\n" + persona_addon

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

        # 组装系统提示词 = 角色人设 + 人格画像 + 历史记忆 + 工具使用指令
        full_context = long_term_context if long_term_context else "（暂无历史摘要）"
        raw_sys_prompt = role_prompt_dict[role_select]
        sys_prompt = raw_sys_prompt.format(context=full_context)

        # 注入人格化提示词（如果开启了画像）
        persona_addon = _build_persona_addon(user_text)
        if persona_addon:
            sys_prompt += "\n\n" + persona_addon

        sys_prompt += "\n\n" + TOOL_INSTRUCTIONS

        # 获取 MCP 工具（如果有 GitHub Token 就会加载，没有则返回空列表，不影响原有功能）
        mcp_tools = _get_mcp_tools()
        if mcp_tools:
            # 有 MCP 工具时，追加上 GitHub 工具的使用说明
            sys_prompt += "\n\n" + MCP_TOOL_GUIDANCE

        # 构建工具列表 = 内置工具 + MCP 工具
        tools = [
            analyze_emotion,
            get_repair_suggestions,
            get_current_time,
            create_knowledge_search_tool(vec_store),
        ] + mcp_tools

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
