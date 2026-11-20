"""Tool registry for the chapter 2 tool-calling chain."""

from app.tools.business import query_logistics, query_order, query_product, submit_refund
from app.tools.context import current_conversation_id
from app.tools.db_tools import create_ticket, query_faq

ALL_TOOLS = [query_order, query_product, query_logistics, query_faq, create_ticket, submit_refund]

TOOL_REGISTRY = {tool.name: tool for tool in ALL_TOOLS}

__all__ = [
    "ALL_TOOLS",
    "TOOL_REGISTRY",
    "current_conversation_id",
    "query_order",
    "query_product",
    "query_logistics",
    "query_faq",
    "create_ticket",
]
