import asyncio

import pytest
from pydantic import ValidationError

from app.schemas.extraction import AfterSalesExtraction
from app.services.extraction_service import ExtractionService


class FakeChain:
    def __init__(self, result):
        self.result = result
        self.seen = None

    async def ainvoke(self, input):
        self.seen = input
        return self.result


def test_extract_returns_parsed():
    fixed = AfterSalesExtraction(order_id="123", request_type="退货", expected_resolution="退货退款")
    fc = FakeChain(fixed)
    svc = ExtractionService(fc)
    result = asyncio.run(svc.extract("订单 123 想退货"))
    assert result == fixed
    assert fc.seen == {"input": "订单 123 想退货"}


def test_schema_rejects_bad_enum():
    with pytest.raises(ValidationError):
        AfterSalesExtraction(order_id="1", request_type="白嫖", expected_resolution="x")


def test_schema_orders_id_optional():
    e = AfterSalesExtraction(request_type="其他", expected_resolution="暂无")
    assert e.order_id is None
