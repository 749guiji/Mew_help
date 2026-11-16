"""模型选工具正确率评估(非单测,跑脚本看输出)。

每条标注 (话术, 期望工具)。脚本对每条调 llm.bind_tools(ALL_TOOLS).ainvoke,
取首个 tool_call.name 与期望比对;期望 None 表示不应调任何工具。
"""
import asyncio
import sys

from langchain_core.messages import HumanMessage

from app.core.llm import get_llm
from app.tools import ALL_TOOLS

CASES = [
    ("订单 1001 的物流到哪了", "query_logistics"),
    ("退货政策是什么", "query_faq"),
    ("邮费是多少", "query_faq"),  # 负例:应选 query_faq 但 LIKE 查不到(漏召回,预期)
    ("帮我查一下订单 1001 的状态", "query_order"),
    ("手机壳这个商品多少钱", "query_product"),
    ("我要投诉,帮我开个工单", "create_ticket"),
    ("你好", None),  # 不应调工具
    ("订单 1001 现在到哪了", "query_logistics"),
]


async def main() -> int:
    llm = get_llm()
    model = llm.bind_tools(ALL_TOOLS)
    correct = 0
    for text, expected in CASES:
        msg = await model.ainvoke([HumanMessage(content=text)])
        tc = msg.tool_calls
        actual = tc[0]["name"] if tc else None
        ok = actual == expected
        if ok:
            correct += 1
        print(f"[{'OK ' if ok else 'X  '}] {text!r:30} 期望={expected!r:20} 实际={actual!r}")
    print(f"\n正确率: {correct}/{len(CASES)}")
    return 0 if correct == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
