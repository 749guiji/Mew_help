"""Mock business tools that return random plausible Chinese answers."""

import random

from langchain_core.tools import tool


@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单状态。"""
    statuses = ["已支付，待发货", "已发货，运输中", "已签收", "已完成", "退款中"]
    return f"订单 {order_id} 当前状态:{random.choice(statuses)}。"


@tool
def query_product(product_name: str) -> str:
    """根据商品名称查询商品信息。"""
    infos = ["有现货，库存充足", "好评率 98%", "支持 7 天无理由退货", "销量领先，价格实惠"]
    return f"商品「{product_name}」:{random.choice(infos)}。"


@tool
def query_logistics(order_id: str) -> str:
    """根据订单号查询物流状态。"""
    statuses = ["已揽收", "运输中", "已到达派送点", "派送中", "已签收"]
    return f"订单 {order_id} 物流状态:{random.choice(statuses)}。"
