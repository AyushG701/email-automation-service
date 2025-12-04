from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class EmailClassificationRequest(BaseModel):
    """Request model for email classification."""
    created_at: Optional[str] = Field(None, alias="Created At")
    updated_at: Optional[str] = Field(None, alias="Updated At")
    sender: Optional[str] = Field(None, alias="Sender")
    receiver_to: Optional[str] = Field(None, alias="Receiver (To)")
    subject: Optional[str] = Field(None, alias="Subject")
    short_preview: Optional[str] = Field(None, alias="Short Preview")
    main_body_trimmed: Optional[str] = Field(None, alias="Main Body (Trimmed)")
    status: Optional[str] = Field(None, alias="Status")
    message_id: Optional[str] = Field(None, alias="Message ID")
    user_id: Optional[str] = Field(None, alias="User ID")
    read_at: Optional[str] = Field(None, alias="Read At")
    carrier_id_context_id: Optional[str] = Field(
        None, alias="Carrier ID / Context ID")
    template_name: Optional[str] = Field(None, alias="Template Name")
    retries: Optional[int] = Field(None, alias="Retries")
    is_failed: Optional[bool] = Field(None, alias="Is Failed")
    is_deleted: Optional[bool] = Field(None, alias="Is Deleted")
    snippet: Optional[str] = Field(None, alias="Snippet")

    # To allow any other fields that might be present in the payload
    class Config:
        extra = "allow"
        populate_by_name = True  # Allows using field names or aliases

    
    
class EmailClassificationResponse(BaseModel):
    """Response model for email classification and detail extraction."""
    classification: str
    extracted_details: Optional[Dict[str, Any]] = {}
    reason: Optional[str] = None  # For errors or unknown classification
