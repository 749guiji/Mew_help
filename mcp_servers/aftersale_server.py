"""售后 MCP Server:查在保 / 查退货进度(mock 随机数据,Streamable HTTP)。

独立进程:`python -m mcp_servers.aftersale_server`。
"""

import random

from mcp.server import MCPServer

mcp = MCPServer("aftersale")


@mcp.tool()
def check_warranty(order_id: str) -> str:
    """根据订单号查询商品是否在保修期内。当用户询问是否在保、保修状态时使用。"""
    states = ["在保", "已过保", "保修期内(剩余 3 个月)"]
    return f"订单 {order_id} 保修状态:{random.choice(states)}。"


@mcp.tool()
def check_return_progress(order_id: str) -> str:
    """根据订单号查询退货进度。当用户询问退货到哪一步、退款进度时使用。"""
    stages = ["已提交退货申请", "已寄回待验收", "已验收待退款", "退款已到账"]
    return f"订单 {order_id} 退货进度:{random.choice(stages)}。"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8101)
