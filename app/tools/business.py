"""Mock business tools that return random plausible Chinese answers."""

import random

from langchain_core.tools import tool


@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单状态(商品名/金额/状态/物流单号)。当用户询问某个订单的当前状态时使用。"""
    from app.orders import get_order

    o = get_order(order_id)
    return f"订单 {order_id}:{o['product']} 金额{o['amount']} 状态:{o['status']}。"


@tool
def query_product(product_name: str) -> str:
    """根据商品名称查询商品信息(价格、库存、评价等)。当用户询问某个具体商品时使用。"""
    infos = ["有现货，库存充足", "好评率 98%", "支持 7 天无理由退货", "销量领先，价格实惠"]
    return f"商品「{product_name}」:{random.choice(infos)}。"


@tool
def query_logistics(order_id: str) -> str:
    """根据订单号查询物流/包裹运输状态。当用户询问包裹到哪了、运输或配送进度时使用。"""
    statuses = ["已揽收", "运输中", "已到达派送点", "派送中", "已签收"]
    return f"订单 {order_id} 物流状态:{random.choice(statuses)}。"
