from pydantic import BaseModel, Field
from typing import Literal, Optional


class EquityDecision(BaseModel):
    symbol: str = Field(..., description="NSE Yahoo ticker ending in .NS")
    risk_tier: Literal["LOW", "MID", "HIGH"]
    action: Literal["BUY", "SKIP"]
    conviction: int = Field(..., ge=0, le=100)
    entry_price: float
    stop_loss: float
    target_price: float
    news_status: Literal["CLEAR", "POSITIVE_CATALYST", "NEGATIVE_RISK"]
    reasoning: str


class FnODecision(BaseModel):
    underlying: Literal["NIFTY", "BANKNIFTY"]
    contract_symbol: str = Field(..., description="Exact option contract or spread label e.g. NIFTY 24200 CE")
    strategy_type: Literal["SINGLE_LEG_BUY", "DEBIT_SPREAD"]
    option_type: Literal["CE", "PE"]
    strike: int
    hedge_strike: Optional[int] = Field(None, description="OTM short leg strike if DEBIT_SPREAD, else null")
    lot_size: int
    action: Literal["BUY", "SKIP"]
    conviction: int = Field(..., ge=0, le=100)
    entry_premium: float
    sl_premium: float
    target_premium: float
    delta: float
    iv: float
    reasoning: str


class UnifiedCommitteeOutput(BaseModel):
    market_regime: Literal["TRENDING_BULL", "TRENDING_BEAR", "RANGE_CHOP", "HIGH_VOL_EVENT"]
    vix_assessment: str
    equity_trades: list[EquityDecision]
    fno_trades: list[FnODecision]
