from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union


# --- Pydantic Models ---
class EmailAttachment(BaseModel):
    """Model for email attachment details."""
    filename: str = Field(..., description="Name of the attachment file")
    mimeType: str = Field(..., description="MIME type of the attachment")
    size: int = Field(..., description="Size of the attachment in bytes")
    attachmentId: str = Field(...,
                              description="Unique identifier for the attachment")


class LoadBidEmailDetails(BaseModel):
    rate: Optional[str] = Field(
        None, description="Rate mentioned in negotiation emails")
    conditions: Optional[str] = Field(
        None, description="Any price-related or logistical conditions for negotiation, acceptance, or rejection")
    questions: Optional[List[str]] = Field(
        default=None, description="List of information requests or questions")
    confirmed_rate: Optional[str] = Field(
        None, description="The final confirmed rate for the load")
    pickup_details: Optional[str] = Field(
        None, description="Key information about pickup for rate confirmation")
    delivery_details: Optional[str] = Field(
        None, description="Key information about delivery for rate confirmation")
    load_id: Optional[str] = Field(
        None, description="Any identifiable load or tracking number")
    broker_email: Optional[str] = Field(
        None, description="Email address of the broker/company requesting setup")
    broker_company: Optional[str] = Field(
        None, description="Name of the broker company requesting setup")
    setup_link: Optional[str] = Field(
        None, description="Setup, onboarding, or registration link provided")


class LoadBidEmailClassificationRequest(BaseModel):
    email: str = Field(...,
                       description="The email content received from the broker")
    attachments: Optional[List[EmailAttachment]] = Field(
        None, description="Optional list of email attachments"
    )


class LoadBidEmailClassificationResponseStructured(BaseModel):
    intent: Literal["negotiation", "information_seeking", "bid_acceptance",
                    "bid_rejection", "rate_confirmation", "broker_setup"]
    details: LoadBidEmailDetails
