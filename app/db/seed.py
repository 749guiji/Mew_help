"""Seed data for the FAQ table."""

from app.db.models import Faq

FAQ_SEED = [
    {
        "question": "退货政策是什么？",
        "answer": "7 天无理由退货，商品保持完好、不影响二次销售即可申请退款。",
        "category": "售后",
    },
    {
        "question": "如何申请退款？",
        "answer": "在订单详情页点击申请售后，选择退款原因并提交，审核通过后原路退回。",
        "category": "售后",
    },
    {
        "question": "商品多久发货？",
        "answer": "现货商品付款后 48 小时内发出，预售商品按页面标注时间发货。",
        "category": "物流",
    },
    {
        "question": "可以开发票吗？",
        "answer": "可以，下单时勾选开具发票，或到订单页补开发票。",
        "category": "发票",
    },
    {
        "question": "收到的商品有问题怎么办？",
        "answer": "请拍照留证并联系在线客服，核实后为您安排换货或退款。",
        "category": "售后",
    },
]


def seed_faq(session) -> int:
    """Insert FAQ seed rows and return the number of rows inserted."""
    faqs = [Faq(**row) for row in FAQ_SEED]
    session.add_all(faqs)
    session.commit()
    return len(faqs)
