"""物流 MCP Server:查询物流轨迹(mock 随机数据,Streamable HTTP)。

独立进程:`python -m mcp_servers.logistics_server`。
"""

import random

from mcp.server import MCPServer

mcp = MCPServer("logistics")


@mcp.tool()
def query_logistics(order_id: str) -> str:
    """根据订单号查询物流/包裹运输状态。当用户询问包裹到哪了、运输或配送进度时使用。"""
    statuses = ["已揽收", "运输中", "已到达派送点", "派送中", "已签收"]
    return f"订单 {order_id} 物流状态:{random.choice(statuses)}。"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8100)
