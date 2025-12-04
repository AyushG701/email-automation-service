from typing import Optional
from pydantic import BaseModel

class ConversationHistoryQuery(BaseModel):
    load_offer_id: str
    carrier_id: str
    broker_email: str
    conversation_id: Optional[int] = None

class IdentifyConversationParams(BaseModel):
    load_offer_id: str
    carrier_id: str
    broker_email: str
    broker_id: Optional[str] = None