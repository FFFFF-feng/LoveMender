# streamlit 主入口,网页应用入口
import streamlit as st
import tempfile  # 用于创建临时文件,用于存储用户上传的文件
import os
from pathlib import Path  # 用于处理文件路径
from langchain_core.messages import HumanMessage  # 用于创建用户消息

# 导入自定义模块
from utils import base64_encode_image
# 导入模型工厂模块 (新增导入 create_text_llm，用于生成摘要)
from model_factory import create_llm, create_text_llm, create_embedding
# 导入RAG服务模块
from rag_service import init_vector_store, ingest_txt_vector_store, search_vector_store, get_context_from_docs
# 导入提示词模板
from prompt_template import role_prompt_dict
# 导入记忆管理模块 (修正：文件名应为 memory_manager)
from memory_manager import MemoryManager

# ================= 改动1：Session State 初始化 =================
# streamlit页面设计:
# 设置页面标题
st.set_page_config(
    page_title="💢❤️💢情绪修复助手"
)

# 防止用户每次刷新页面都会把聊天记录清空
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 给用户看的聊天记录
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = None  # 给 AI 用的记忆管理器

# 侧边栏设计:
with st.sidebar:
    st.header("🗝️API配置")
    # st.session_state 用于存储用户输入的API密钥
    # 如果用户没有输入API密钥,则提示用户输入API密钥
    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = ""
    api_key = st.text_input(
        "请输入阿里云API密钥",
        type="password",  # 密码输入框,用于隐藏用户输入的API密钥
        value=st.session_state.api_key_input,  # 默认值为用户输入的API密钥
        help="阿里云平台申请的API密钥,用于调用通义千问模型"  # 提示信息
    )
    st.session_state.api_key_input = api_key  # 如果用户输入了API密钥,则将API密钥存储到st.session_state中
    st.divider()  # 添加分隔线,用于分隔API配置和模型配置

    # RAG知识库上传模块:
    st.subheader("📖RAG知识库上传(txt格式)")
    uploader_txt = st.file_uploader("上传情感知识库txt文件", type=["txt"])
    # strip()方法用于移除字符串首尾的空格
    if uploader_txt and api_key.strip():
        # ==========【修复】用文件名做标记，避免每次页面刷新重复导入同一文件 ==========
        if st.session_state.get("uploaded_txt_name") != uploader_txt.name:
            # 读取用户上传的txt文件内容
            txt_content = uploader_txt.read().decode("utf-8")
            # 初始化文本嵌入模型
            emb = create_embedding(api_key)  # 返回一个embedding模型
            # 初始化向量数据库
            vec_store = init_vector_store(emb)  # 返回一个向量数据库
            chunk_count = ingest_txt_vector_store(txt_content, "情感知识库", vec_store)
            st.session_state.uploaded_txt_name = uploader_txt.name
            st.success(f"成功导入{chunk_count}条文本片段存入向量数据库")
        else:
            st.info("该文件已导入知识库，无需重复上传")
    elif uploader_txt and not api_key.strip():
        st.warning("请先输入API密钥,再上传txt文件")
    st.info("💡使用提示:\n1.先填写API密钥\n2.再上传情感知识库txt文件,文件内容为情感文本\n3.支持文字+图片格式")

# ================= 改动2：初始化记忆管理器 =================
# 如果用户输入了API密钥,且内存管理器为空,则初始化内存管理器
if api_key.strip() and st.session_state.memory_manager is None:
    # 为什么用 create_text_llm 而不是 create_llm？因为记忆管理器只做摘要压缩，不需要看图片。
    # 用 qwen-plus（纯文本模型）比 qwen-vl-max 便宜约 50%，摘要质量没有区别。
    llm_text = create_text_llm(api_key)
    emb = create_embedding(api_key)  # 创建一个embedding模型
    st.session_state.memory_manager = MemoryManager(llm_text, emb)  # 创建一个内存管理器

# 主页面设计:
st.header("💢❤️💢情绪修复助手(LangChain+RAG模块化项目)")
# st.markdown 用于渲染Markdown格式的文本,用于添加段落、标题、列表等元素
st.markdown("四位情感修复助手,用于修复用户的情感问题")
# st.text_area 用于创建多行文本输入框,用于用户输入文本
user_text = st.text_area("写下你的心情,我们来修复它", height=140)
upload_img = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
# st.radio 用于创建单选框,用于用户选择情感修复助手
role_list = list(role_prompt_dict.keys())
target_default = "温柔体贴男"
select_index = role_list.index(target_default)
role_select = st.radio(
    "请选择你想要的情感修复助手:",
    role_list,
    index=select_index,
)
# st.button 用于创建按钮,用于用户点击提交表单
send_btn = st.button("开始修复❤️!", type="primary")

# 点击按钮后逻辑运行:
if send_btn:
    # 基础的一个校验:
    if not api_key.strip():
        st.warning("请先输入API密钥!")
        st.stop()

    # 新增校验：文字为空并且没有上传图片，禁止提交
    if not user_text.strip() and upload_img is None:
        st.warning("请输入文字描述，或者上传聊天截图！")
        st.stop()

    # 【修复】从 session_state 中取出记忆管理器，否则下面会报变量未定义的错误
    memory_mgr = st.session_state.memory_manager

    # 1.初始化模型与向量库
    llm = create_llm(api_key)
    emb = create_embedding(api_key)
    vec_store = init_vector_store(emb)

    # ==========【核心修复】只有文本非空，才执行RAG检索！！==========
    if user_text.strip():
        search_docs = search_vector_store(vec_store, user_text)
        context_text = get_context_from_docs(search_docs)
    else:
        # 文本空白，直接跳过向量检索，context置空，不会调用embedding接口
        context_text = ""

    # ================= 改动3：检索长期记忆 =================
    # .retrieve_relevant_memory方法用于检索与用户输入相关的记忆,返回值是一个列表,每个元素是一个字典,包含记忆的文本内容和相关度
    long_term_context = memory_mgr.retrieve_relevant_memory(
        user_text if user_text.strip() else "图片对话"
    )

    # ================= 改动4：组装系统提示词 =================
    # 填充角色提示词，包含知识库参考和长期记忆
    full_context = ""  # 初始化一个空字符串,用于存储用户输入和长期记忆的组合
    if context_text:  # 如果用户输入了文本,则将文本添加到系统提示词中
        full_context += f"【知识库参考】\n{context_text}\n\n"  # 将用户输入的文本添加到系统提示词中
    if long_term_context:
        full_context += f"【历史对话摘要】\n{long_term_context}\n\n"  # 将长期记忆添加到系统提示词中

    raw_sys_prompt = role_prompt_dict[role_select]  # 获取用户选择的情感修复助手的系统提示词模板
    sys_prompt = raw_sys_prompt.format(context=full_context)  # 将知识库参考和长期记忆替换到系统提示词中
    # .format()是一个字符串方法,用于将字符串中的占位符替换为实际值

    # ================= 改动5：用 build_messages 替代手动构建 =================
    # 组装消息列表,用build_messages方法代替手动构建
    messages = memory_mgr.build_messages(sys_prompt, user_text)

    # 处理图像上传并追加到消息列表中
    temp_path = None
    if upload_img:
        # ==========【修复】根据实际上传文件类型动态设置后缀，而非硬编码 .jpg ==========
        file_ext = os.path.splitext(upload_img.name)[1] or ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        temp_file.write(upload_img.read())
        temp_file.close()
        temp_path = Path(temp_file.name)
        b64_img = base64_encode_image(str(temp_path))

        # ==========【修复】MIME类型与后缀保持一致 ==========
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        mime_type = mime_map.get(file_ext.lower(), "image/jpeg")

        human_content = [
            {
                "type": "text", "text": user_text,
            },
            {
                "type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"},
            }
        ]
        # 【修复】：因为 build_messages 里已经加了一条纯文本的 HumanMessage，
        # 如果上传图片，我们需要用带图片的消息【替换】掉最后一条，而不是【追加】，
        # 否则大模型会收到两条人类消息，导致逻辑混乱。
        messages[-1] = HumanMessage(content=human_content)
        st.image(upload_img, caption="你上传的截图")
    else:
        st.write(user_text)

    # 5.调用模型生成回复
    try:
        with st.spinner("AI正在思考中..."):
            response = llm.invoke(messages)
            st.markdown("🤖情感修复助手回复:")
            st.markdown(response.content)

            # 【修复】：AI 回复后，必须把这一轮对话存入记忆，实现真正的记忆闭环！
            memory_mgr.add_exchange(user_text, response.content)

    except Exception as e:
        st.error(f"请求异常：{e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)