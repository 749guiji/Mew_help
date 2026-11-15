from typing import Literal

from pydantic import BaseModel


class AfterSalesExtraction(BaseModel):
    order_id: str | None = None
    request_type: Literal["退款", "退货", "换货", "维修", "物流查询", "其他"]
    expected_resolution: str
