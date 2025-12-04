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