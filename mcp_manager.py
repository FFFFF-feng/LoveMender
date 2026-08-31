# mcp_manager.py
# MCP (Model Context Protocol) 工具管理模块
# 作用：启动 MCP Server（如 GitHub），并把 MCP 工具转换成 LangChain Agent 可用的工具
#
# 原理图解：
# ┌─────────────┐   stdio    ┌──────────────────┐   MCP协议    ┌──────────────────┐
# │  Python端   │◄──────────►│  GitHub MCP Server│◄───────────►│   GitHub API     │
# │ (我们的代码) │            │  (Node.js进程)    │              │  (实际干活的地方) │
# └─────────────┘            └──────────────────┘              └──────────────────┘

import os
import asyncio
from typing import List
from langchain_core.tools import BaseTool
from dotenv import load_dotenv  # 加载 .env 文件

# 先加载 .env，再读取环境变量
load_dotenv()
from logger import logger


# ==================== MCP 配置 ====================
# 配置想要接入的 MCP Server
# 目前只接入 GitHub，以后想加其他的，在这个字典里加就行
MCP_SERVER_CONFIG = {
    "github": {
        # 用 npx 运行（Node.js 的包运行工具）
        # -y 表示自动确认，不提示用户
        # @modelcontextprotocol/server-github 是官方的 GitHub MCP Server
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "transport": "stdio",  # 通信方式：标准输入输出（进程间通信）
        # GitHub Token 从环境变量读取
        "env": {
            "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
        },
    }
}

# 是否启用 MCP（没有 Token 时自动禁用，不影响原有功能）
MCP_ENABLED = os.environ.get("MCP_ENABLED", "true").lower() == "true"


# ==================== 核心类：MCP 管理器 ====================
class MCPManager:
    """
    MCP 工具管理器
    - 负责启动/关闭 MCP Server
    - 把 MCP 工具转换成 LangChain 格式
    - 单例模式，全局只有一个实例
    """

    def __init__(self):
        self._client = None       # MCP 客户端
        self._tools = []          # 已加载的工具列表
        self._initialized = False # 是否已初始化

    async def initialize(self) -> List[BaseTool]:
        """
        初始化 MCP 客户端，加载所有 MCP 工具
        返回 LangChain 格式的工具列表
        """
        if self._initialized:
            logger.info("[MCP] 已初始化，直接返回缓存工具（%d个）", len(self._tools))
            return self._tools

        if not MCP_ENABLED:
            logger.info("[MCP] MCP 已禁用（MCP_ENABLED=false），跳过初始化")
            self._initialized = True
            return []

        # 检查 GitHub Token
        github_token = os.environ.get("GITHUB_TOKEN", "")
        if not github_token.strip():
            logger.warning("[MCP] 未检测到 GITHUB_TOKEN，GitHub MCP 工具将不可用")
            self._initialized = True
            return []

        try:
            # 延迟导入，避免没装包时 import 就报错
            from langchain_mcp_adapters.client import MultiServerMCPClient

            logger.info("[MCP] 正在启动 MCP Server...")

            # 创建多服务器 MCP 客户端
            # MultiServerMCPClient 可以同时管理多个 MCP Server
            self._client = MultiServerMCPClient(MCP_SERVER_CONFIG)

            # 获取所有 MCP 工具
            # get_tools() 会自动：
            # 1. 启动所有配置的 MCP Server 进程
            # 2. 通过 MCP 协议查询每个 Server 提供了哪些工具
            # 3. 把 MCP 工具转换成 LangChain 的 BaseTool 格式
            self._tools = await self._client.get_tools()
            self._initialized = True

            # 打印加载的工具名，方便调试
            tool_names = [t.name for t in self._tools]
            logger.info("[MCP] 初始化成功，加载了 %d 个工具: %s",
                        len(self._tools), ", ".join(tool_names))

            return self._tools

        except Exception as e:
            logger.error("[MCP] 初始化失败: %s", e)
            # 失败了也不抛出异常，让主流程继续运行（降级：不用 MCP 工具）
            self._initialized = True
            return []

    async def close(self):
        """关闭 MCP 客户端，清理子进程"""
        if self._client:
            try:
                await self._client.close()
                logger.info("[MCP] 已关闭 MCP 客户端")
            except Exception as e:
                logger.error("[MCP] 关闭客户端失败: %s", e)
            finally:
                self._client = None
                self._tools = []
                self._initialized = False

    @property
    def tools(self) -> List[BaseTool]:
        """获取已加载的工具列表"""
        return self._tools

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# ==================== 全局单例 ====================
# 整个应用共用一个 MCPManager 实例
_mcp_manager = MCPManager()


def get_mcp_manager() -> MCPManager:
    """获取全局 MCP 管理器单例"""
    return _mcp_manager


async def get_mcp_tools() -> List[BaseTool]:
    """
    便捷函数：初始化并获取 MCP 工具列表
    直接在 main.py 里调用这个就行
    """
    mgr = get_mcp_manager()
    return await mgr.initialize()


async def close_mcp():
    """关闭 MCP（程序退出时调用）"""
    mgr = get_mcp_manager()
    await mgr.close()


# ==================== 测试入口 ====================
#运行该文件可以测试 MCP 工具是否能正常加载

if __name__ == "__main__":
    async def test_mcp():
        print("=" * 60)
        print("MCP 工具加载测试")
        print("=" * 60)

        tools = await get_mcp_tools()

        if not tools:
            print("\n❌ 没有加载到任何 MCP 工具")
            print("   可能原因：")
            print("   1. 未设置 GITHUB_TOKEN 环境变量")
            print("   2. 未安装 Node.js（npx 命令不可用）")
            print("   3. 网络问题导致 npx 下载包失败")
        else:
            print(f"\n✅ 成功加载 {len(tools)} 个 MCP 工具：\n")
            for i, tool in enumerate(tools, 1):
                print(f"  {i}. {tool.name}")
                print(f"     描述: {tool.description[:80]}...")
                print()

        await close_mcp()
        print("=" * 60)
        print("测试完成")
        print("=" * 60)

    asyncio.run(test_mcp())
