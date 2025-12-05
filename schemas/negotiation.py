from pydantic import BaseModel
from typing import List, Optional


class NegotiationResponse(BaseModel):
    response: str
    conversation_id: int
    # messages: List[message]


class NegotiationRequest(BaseModel):
    message: str
    load_offer_id: str
    carrier_id: str
    broker_email: str
    min_price: int
    max_price: int
    broker_id: Optional[str] = None


class NegotiationStrategy(BaseModel):
    """Strategy settings for negotiation"""
    max_rounds: int = 5
    min_price: float = 4000
    max_price: float = 6000
    initial_discount: float = 0.05  # 5% drop per round


class NegotiationState(BaseModel):
    # Core negotiation details
    load_offer_id: Optional[str] = None
    thread_id: Optional[str] = None
    is_negotiation_active: bool = True
    negotiation_round: int = 0
    info_exchange_round: int = 0

    # Price tracking
    initial_broker_price: Optional[float] = None
    initial_ask_price: Optional[float] = None
    last_broker_price: Optional[float] = None
    last_supertruck_price: Optional[float] = None

    # Strategy
    strategy: Optional[NegotiationStrategy] = None

    # Communication state
    last_message: str = ""
    response_needed: bool = False
    suggested_response: str = ""