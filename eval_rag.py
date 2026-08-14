# eval_system.py
# 全系统验证脚本 — 一键检测所有模块健康度
# 用法: python eval_system.py
#
# 检测项目:
#   1. 系统健康检查（模块导入、配置加载、数据库连接）
#   2. RAG 检索质量（Recall@K, Precision@K, MRR, 相似度）
#   3. 记忆机制（短期记忆、长期记忆、自动压缩）
#   4. Agent 工具（4个工具逐一调用验证）
#   5. 模型路由（三级模型创建验证）
#   6. 缓存性能（创建耗时对比）

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()


# ==================== 工具函数 ====================

def print_header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_section(title):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def status_icon(passed):
    return "✓" if passed else "✗"


# ==================== 1. 系统健康检查 ====================

def check_system_health():
    print_header("1. 系统健康检查")
    checks = []

    # 1.1 配置加载
    try:
        from config import (
            CHROMA_DB_DIR, chunk_size, chunk_overlap, retrieve_top_k,
            SIMILARITY_THRESHOLD, DEDUP_SIMILARITY_THRESHOLD,
            model_name, TEXT_MODEL_NAME, SUMMARY_MODEL_NAME,
            AGENT_MAX_ITERATIONS, MAX_CONTEXT_CHARS, MAX_REPLY_TOKENS,
        )
        checks.append(("配置加载", True, f"chunk_size={chunk_size}, top_k={retrieve_top_k}, threshold={SIMILARITY_THRESHOLD}"))
    except Exception as e:
        checks.append(("配置加载", False, str(e)))
        return checks  # 后续检查依赖配置，直接返回

    # 1.2 模块导入
    modules = [
        ("model_factory", "create_llm, create_text_llm, create_summary_llm, create_embedding"),
        ("rag_service", "init_vector_store, ingest_txt_vector_store, search_vector_store"),
        ("memory_manager", "MemoryManager"),
        ("agent_tools", "analyze_emotion, get_repair_suggestions, create_agent_executor"),
        ("prompt_template", "role_prompt_dict"),
        ("logger", "logger"),
    ]
    for mod_name, attrs in modules:
        try:
            mod = __import__(mod_name)
            for attr in attrs.split(", "):
                getattr(mod, attr)
            checks.append((f"模块导入: {mod_name}", True, ""))
        except Exception as e:
            checks.append((f"模块导入: {mod_name}", False, str(e)))

    # 1.3 数据库连接
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collections = [c.name for c in client.list_collections()]
        checks.append(("数据库连接", True, f"集合: {collections}"))

        # 检查 RAG 集合是否有数据
        for col_name in collections:
            col = client.get_collection(col_name)
            count = col.count()
            checks.append((f"  集合 '{col_name}'", count > 0, f"记录数: {count}"))
    except Exception as e:
        checks.append(("数据库连接", False, str(e)))

    # 1.4 .env 文件检查
    env_vars = ["DASHSCOPE_API_KEY", "CHUNK_SIZE", "MODEL_NAME", "TEXT_MODEL_NAME"]
    for var in env_vars:
        val = os.environ.get(var, "")
        if var == "DASHSCOPE_API_KEY":
            checks.append((f"环境变量 {var}", bool(val), f"{'已设置 (' + val[:6] + '...)' if val else '未设置'}"))
        else:
            checks.append((f"环境变量 {var}", bool(val), f"值: {val}" if val else "使用默认值"))

    # 输出
    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    for name, ok, detail in checks:
        print(f"  {status_icon(ok)} {name} {f'→ {detail}' if detail else ''}")
    print(f"\n  通过: {passed}/{total}")

    return all(ok for _, ok, _ in checks)


# ==================== 2. RAG 检索质量评估 ====================

def evaluate_rag(api_key):
    print_header("2. RAG 检索质量评估")

    from model_factory import create_embedding
    from rag_service import init_vector_store, search_vector_store
    from config import retrieve_top_k, SIMILARITY_THRESHOLD

    test_cases = [
        ("我好生气，怎么办", ["愤怒", "生气", "情绪"], "愤怒情绪"),
        ("最近总是很焦虑", ["焦虑", "紧张", "放松"], "焦虑情绪"),
        ("如何控制自己的情绪", ["情绪", "控制", "管理"], "情绪管理"),
        ("心情不好做什么能开心", ["心情", "开心", "好心情", "习惯"], "心情低落"),
        ("跟男朋友吵架了很伤心", ["沟通", "关系", "吵架", "伤"], "人际冲突"),
        ("感觉自己很抑郁走不出来", ["抑郁", "疗愈", "自救"], "抑郁倾向"),
        ("怎么走出情绪低谷", ["情绪", "低谷", "死胡同"], "情绪低谷"),
        ("有什么放松身心的方法", ["放松", "方法", "练习"], "放松方法"),
        ("情绪管理有什么技巧", ["情绪", "管理", "技巧", "颗粒度"], "情绪技巧"),
        ("如何提升情绪颗粒度", ["情绪颗粒度", "颗粒度", "镜头"], "颗粒度提升"),
        ("写日记能帮助情绪修复吗", ["书写", "日记", "表达", "坦诚"], "表达性书写"),
        ("情绪背后有什么需求", ["需求", "背后", "想法"], "情绪需求"),
    ]

    try:
        emb = create_embedding(api_key)
        vs = init_vector_store(emb)
    except Exception as e:
        print(f"  ✗ 初始化失败: {e}")
        return False

    count = vs._collection.count()
    print(f"  向量库记录数: {count}")
    if count == 0:
        print("  ✗ 向量库为空，请先上传知识库文件")
        return False

    all_recall = []
    all_precision = []
    all_mrr = []
    all_top_sim = []
    pass_count = 0

    for query, keywords, desc in test_cases:
        docs = search_vector_store(vs, query)
        if not docs:
            print(f"  ✗ {desc}: 检索结果为空")
            all_recall.append(0)
            all_precision.append(0)
            all_mrr.append(0)
            all_top_sim.append(0)
            continue

        relevant = 0
        first_relevant_rank = 0
        for i, doc in enumerate(docs):
            if any(kw in doc.page_content for kw in keywords):
                relevant += 1
                if first_relevant_rank == 0:
                    first_relevant_rank = i + 1

        recall = 1.0 if relevant > 0 else 0.0
        precision = relevant / len(docs)
        mrr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0

        all_recall.append(recall)
        all_precision.append(precision)
        all_mrr.append(mrr)
        if recall > 0:
            pass_count += 1

        status = "✓" if recall > 0 else "✗"
        print(f"  {status} {desc}: Recall={recall:.0%} P={precision:.0%} MRR={mrr:.2f}")

    n = len(test_cases)
    avg_recall = sum(all_recall) / n
    avg_precision = sum(all_precision) / n
    avg_mrr = sum(all_mrr) / n

    print(f"\n  Recall@{retrieve_top_k}:    {avg_recall:.1%}  ({pass_count}/{n})")
    print(f"  Precision@{retrieve_top_k}: {avg_precision:.1%}")
    print(f"  MRR:              {avg_mrr:.3f}")

    if avg_recall >= 0.8:
        print(f"  评级: A (优秀)")
    elif avg_recall >= 0.6:
        print(f"  评级: B (良好)")
    else:
        print(f"  评级: C (需优化)")

    return avg_recall >= 0.6


# ==================== 3. 记忆机制验证 ====================

def evaluate_memory(api_key):
    print_header("3. 记忆机制验证")

    from model_factory import create_summary_llm, create_embedding
    from memory_manager import MemoryManager

    try:
        llm = create_summary_llm(api_key)
        emb = create_embedding(api_key)
        mgr = MemoryManager(llm, emb)
    except Exception as e:
        print(f"  ✗ 初始化失败: {e}")
        return False

    results = []

    # 3.1 短期记忆 — 添加对话
    print_section("3.1 短期记忆 — 添加对话")
    mgr.add_exchange("我好生气", "我理解你的感受，深呼吸...")
    mgr.add_exchange("还是很烦", "试着写下你的想法...")
    stats = mgr.get_session_stats()
    ok = stats["turns"] == 2
    results.append(ok)
    print(f"  {status_icon(ok)} 添加2轮对话后 turns={stats['turns']} (期望=2)")

    # 3.2 短期记忆 — 消息构建
    print_section("3.2 短期记忆 — 消息构建")
    messages = mgr.build_messages("你是助手", "新消息")
    ok = len(messages) >= 5  # SystemMessage + 4 history + 1 current
    results.append(ok)
    print(f"  {status_icon(ok)} 消息列表长度={len(messages)} (期望>=5)")

    # 3.3 短期记忆 — 会话重置
    print_section("3.3 短期记忆 — 会话重置")
    mgr.reset_session()
    stats = mgr.get_session_stats()
    ok = stats["turns"] == 0
    results.append(ok)
    print(f"  {status_icon(ok)} 重置后 turns={stats['turns']} (期望=0)")

    # 3.4 长期记忆 — 检索
    print_section("3.4 长期记忆 — 检索")
    try:
        result = mgr.retrieve_relevant_memory("测试查询")
        ok = isinstance(result, str)
        results.append(ok)
        print(f"  {status_icon(ok)} 检索返回类型=str, 内容长度={len(result)}")
    except Exception as e:
        results.append(False)
        print(f"  ✗ 检索失败: {e}")

    # 3.5 长期记忆 — cosine 距离验证
    print_section("3.5 长期记忆 — cosine 距离验证")
    try:
        meta = mgr.long_term_collection.metadata
        if meta is None:
            meta = {}
        space = meta.get("hnsw:space", "l2")
        ok = space == "cosine"
        results.append(ok)
        print(f"  {status_icon(ok)} 距离度量={space} (期望=cosine)")
    except Exception as e:
        results.append(False)
        print(f"  ✗ 检查失败: {e}")

    passed = sum(1 for r in results if r)
    print(f"\n  通过: {passed}/{len(results)}")
    return all(results)


# ==================== 4. Agent 工具验证 ====================

def evaluate_agent_tools(api_key):
    print_header("4. Agent 工具验证")

    from agent_tools import (
        analyze_emotion,
        get_repair_suggestions,
        get_current_time,
        create_knowledge_search_tool,
        create_agent_executor,
        TOOL_INSTRUCTIONS,
    )
    from model_factory import create_text_llm, create_embedding
    from rag_service import init_vector_store

    results = []

    # 4.1 情绪分析工具
    print_section("4.1 情绪分析工具 (analyze_emotion)")
    try:
        result = analyze_emotion.invoke({"text": "我好生气，气死了"})
        ok = "愤怒" in result
        results.append(ok)
        print(f"  {status_icon(ok)} 输入='我好生气' → {result[:50]}")
    except Exception as e:
        results.append(False)
        print(f"  ✗ 调用失败: {e}")

    # 4.2 修复建议工具
    print_section("4.2 修复建议工具 (get_repair_suggestions)")
    try:
        result = get_repair_suggestions.invoke({"emotion_type": "愤怒"})
        ok = "深呼吸" in result
        results.append(ok)
        print(f"  {status_icon(ok)} 输入='愤怒' → 包含建议: {len(result)}字")
    except Exception as e:
        results.append(False)
        print(f"  ✗ 调用失败: {e}")

    # 4.3 时间问候工具
    print_section("4.3 时间问候工具 (get_current_time)")
    try:
        result = get_current_time.invoke({})
        ok = "当前时间" in result
        results.append(ok)
        print(f"  {status_icon(ok)} 返回: {result[:50]}")
    except Exception as e:
        results.append(False)
        print(f"  ✗ 调用失败: {e}")

    # 4.4 知识库检索工具
    print_section("4.4 知识库检索工具 (search_knowledge_base)")
    try:
        emb = create_embedding(api_key)
        vs = init_vector_store(emb)
        if vs._collection.count() > 0:
            search_tool = create_knowledge_search_tool(vs)
            result = search_tool.invoke({"query": "生气怎么办"})
            ok = len(result) > 10
            results.append(ok)
            print(f"  {status_icon(ok)} 检索结果: {len(result)}字")
        else:
            results.append(True)
            print(f"  ⚠ 向量库为空，跳过知识库检索测试")
    except Exception as e:
        results.append(False)
        print(f"  ✗ 调用失败: {e}")

    # 4.5 Agent 创建验证
    print_section("4.5 Agent 创建验证 (create_agent_executor)")
    try:
        from model_factory import create_text_llm
        llm = create_text_llm(api_key)
        tools = [analyze_emotion, get_repair_suggestions, get_current_time]
        executor = create_agent_executor(llm, tools, "你是助手", max_iterations=3)

        has_recursion_limit = hasattr(executor, "_recursion_limit")
        results.append(has_recursion_limit)
        print(f"  {status_icon(has_recursion_limit)} Agent创建成功, "
              f"工具数={len(tools)}, recursion_limit={getattr(executor, '_recursion_limit', '未设置')}")
    except Exception as e:
        results.append(False)
        print(f"  ✗ 创建失败: {e}")

    # 4.6 工具指令完整性
    print_section("4.6 工具指令完整性 (TOOL_INSTRUCTIONS)")
    required_tools = ["analyze_emotion", "search_knowledge_base", "get_repair_suggestions", "get_current_time"]
    missing = [t for t in required_tools if t not in TOOL_INSTRUCTIONS]
    ok = len(missing) == 0
    results.append(ok)
    print(f"  {status_icon(ok)} 指令包含所有工具名 ({'完整' if ok else f'缺失: {missing}'})")

    passed = sum(1 for r in results if r)
    print(f"\n  通过: {passed}/{len(results)}")
    return all(results)


# ==================== 5. 模型路由验证 ====================

def evaluate_model_routing(api_key):
    print_header("5. 模型路由验证")

    from model_factory import create_llm, create_text_llm, create_summary_llm, create_embedding

    results = []

    # 5.1 多模态模型 (qwen-vl-max)
    print_section("5.1 多模态模型 (qwen-vl-max)")
    try:
        llm = create_llm(api_key)
        model_id = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
        ok = "vl" in str(model_id).lower() or "max" in str(model_id).lower()
        results.append(ok)
        print(f"  {status_icon(ok)} model={model_id}")
    except Exception as e:
        results.append(False)
        print(f"  ✗ 创建失败: {e}")

    # 5.2 纯文本模型 (qwen-plus)
    print_section("5.2 纯文本模型 (qwen-plus)")
    try:
        llm = create_text_llm(api_key)
        model_id = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
        ok = "plus" in str(model_id).lower()
        results.append(ok)
        print(f"  {status_icon(ok)} model={model_id}")
    except Exception as e:
        results.append(False)
        print(f"  ✗ 创建失败: {e}")

    # 5.3 摘要模型 (qwen-turbo)
    print_section("5.3 摘要模型 (qwen-turbo)")
    try:
        llm = create_summary_llm(api_key)
        model_id = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
        ok = "turbo" in str(model_id).lower()
        results.append(ok)
        print(f"  {status_icon(ok)} model={model_id}")
    except Exception as e:
        results.append(False)
        print(f"  ✗ 创建失败: {e}")

    # 5.4 Embedding 模型
    print_section("5.4 Embedding 模型")
    try:
        emb = create_embedding(api_key)
        test_vec = emb.embed_query("测试")
        ok = len(test_vec) > 0
        results.append(ok)
        print(f"  {status_icon(ok)} 向量维度={len(test_vec)}")
    except Exception as e:
        results.append(False)
        print(f"  ✗ 创建失败: {e}")

    passed = sum(1 for r in results if r)
    print(f"\n  通过: {passed}/{len(results)}")
    return all(results)


# ==================== 6. 缓存性能验证 ====================

def evaluate_caching(api_key):
    print_header("6. 缓存性能验证")

    from model_factory import create_embedding, create_text_llm

    results = []

    # 6.1 Embedding 创建耗时
    print_section("6.1 Embedding 创建耗时")
    try:
        t0 = time.time()
        emb1 = create_embedding(api_key)
        t1 = time.time()
        elapsed1 = t1 - t0
        ok = elapsed1 < 5.0
        results.append(ok)
        print(f"  首次创建: {elapsed1:.3f}s")
    except Exception as e:
        results.append(False)
        print(f"  ✗ 创建失败: {e}")
        return all(results)

    # 6.2 Embedding 重复创建（应该很快，因为 Streamlit 缓存）
    print_section("6.2 Embedding 重复创建（验证对象复用）")
    try:
        t0 = time.time()
        emb2 = create_embedding(api_key)
        t1 = time.time()
        elapsed2 = t1 - t0
        is_same = emb1 is emb2
        ok = is_same or elapsed2 < 0.1
        results.append(ok)
        print(f"  第二次创建: {elapsed2:.3f}s, 同一对象={is_same}")
    except Exception as e:
        results.append(False)
        print(f"  ✗ 创建失败: {e}")

    # 6.3 向量库查询耗时
    print_section("6.3 向量库查询耗时")
    try:
        from rag_service import init_vector_store
        vs = init_vector_store(emb1)
        if vs._collection.count() > 0:
            t0 = time.time()
            vs.similarity_search_with_score("测试查询", k=3)
            t1 = time.time()
            elapsed = t1 - t0
            ok = elapsed < 2.0
            results.append(ok)
            print(f"  {status_icon(ok)} 查询耗时: {elapsed:.3f}s")
        else:
            results.append(True)
            print(f"  ⚠ 向量库为空，跳过查询测试")
    except Exception as e:
        results.append(False)
        print(f"  ✗ 查询失败: {e}")

    passed = sum(1 for r in results if r)
    print(f"\n  通过: {passed}/{len(results)}")
    return all(results)


# ==================== 主函数 ====================

def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("[错误] 请在 .env 中设置 DASHSCOPE_API_KEY")
        sys.exit(1)

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  LoveMender 全系统验证报告".center(56) + "║")
    print("║" + f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}".center(56) + "║")
    print("╚" + "═" * 68 + "╝")

    all_passed = []

    # 1. 系统健康检查
    all_passed.append(("系统健康检查", check_system_health()))

    # 2. RAG 检索质量
    all_passed.append(("RAG 检索质量", evaluate_rag(api_key)))

    # 3. 记忆机制
    all_passed.append(("记忆机制", evaluate_memory(api_key)))

    # 4. Agent 工具
    all_passed.append(("Agent 工具", evaluate_agent_tools(api_key)))

    # 5. 模型路由
    all_passed.append(("模型路由", evaluate_model_routing(api_key)))

    # 6. 缓存性能
    all_passed.append(("缓存性能", evaluate_caching(api_key)))

    # 汇总
    print_header("汇总")
    for name, passed in all_passed:
        print(f"  {status_icon(passed)} {name}")

    total_passed = sum(1 for _, p in all_passed if p)
    total = len(all_passed)
    print(f"\n  总通过: {total_passed}/{total}")

    if total_passed == total:
        print("\n  🎉 全部通过！系统状态良好。")
    else:
        failed = [name for name, p in all_passed if not p]
        print(f"\n  ⚠ 未通过: {', '.join(failed)}")

    print("=" * 70)
    return total_passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
