import json
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from integration.supertruck import SuperTruck

# Assuming the InformationSeeker is imported
# from services.information_seeker import InformationSeeker

logger = logging.getLogger(__name__)


# ==================== REQUEST SCHEMAS ====================

class CarrierData(BaseModel):
    """Carrier information"""
    companyName: Optional[str] = None
    mcNumber: Optional[str] = None
    dotNumber: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    equipmentTypes: Optional[list[str]] = None
    serviceAreas: Optional[list[str]] = None
    insurance: Optional[dict] = None
    operatingAuthority: Optional[str] = None
    fleetSize: Optional[int] = None
    yearsInBusiness: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {
                "companyName": "Swift Logistics LLC",
                "mcNumber": "MC-789456",
                "dotNumber": "DOT-123789",
                "email": "dispatch@swiftlogistics.com",
                "phone": "+1-555-0123",
                "equipmentTypes": ["53' Dry Van", "48' Flatbed"],
                "serviceAreas": ["Texas", "Oklahoma"],
                "insurance": {
                    "cargo": "$100,000",
                    "liability": "$1,000,000"
                }
            }
        }


class AskQuestionRequest(BaseModel):
    """Request to ask a question about carrier"""
    carrier_data: CarrierData
    question: str = Field(..., min_length=1)
    previous_conversation: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "carrier_data": {
                    "companyName": "Swift Logistics LLC",
                    "mcNumber": "MC-789456",
                    "equipmentTypes": ["53' Dry Van"]
                },
                "question": "What's your MC number?",
                "previous_conversation": "Broker: Hi, looking for a carrier.\nCarrier: Sure, what do you need?"
            }
        }


class ConversationRequest(BaseModel):
    """Request for multi-turn conversation"""
    carrier_data: CarrierData
    conversation_history: list[dict] = Field(default_factory=list)
    new_question: str = Field(..., min_length=1)

    class Config:
        json_schema_extra = {
            "example": {
                "carrier_data": {
                    "companyName": "Swift Logistics LLC",
                    "mcNumber": "MC-789456"
                },
                "conversation_history": [
                    {"role": "broker", "message": "Hi, what's your MC?"},
                    {"role": "carrier", "message": "MC-789456"}
                ],
                "new_question": "Do you have cargo insurance?"
            }
        }


class AskByEmailRequest(BaseModel):
    """Request to ask question by fetching carrier via email"""
    tenant_id: str
    carrier_email: str = Field(..., description="Carrier email to lookup")
    question: str = Field(..., min_length=1)
    previous_conversation: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "tenant_id": "your-tenant-id",
                "carrier_email": "400dyforever@gmail.com",
                "question": "What's your MC number?",
                "previous_conversation": "Broker: Hi\nCarrier: Hello, looking for loads"
            }
        }
# ==================== RESPONSE SCHEMAS ====================

class AskQuestionResponse(BaseModel):
    """Response with carrier's answer"""
    answer: str
    carrier_company: Optional[str] = None


class ConversationResponse(BaseModel):
    """Response for conversation"""
    answer: str
    updated_conversation: list[dict]
    carrier_company: Optional[str] = None


# ==================== CONTROLLER ====================
supertruck = SuperTruck()
class InformationSeekerController:
    """Controller for testing InformationSeeker"""

     
    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger(__name__)
        self.router = APIRouter(prefix='/api/v1/carrier-info', tags=['Carrier Information'])
        
        # Register routes
        self._register_routes()
        app.include_router(self.router)

    def _register_routes(self):
        """Register API endpoints"""
        
        # Endpoint 1: Simple question/answer
        self.router.add_api_route(
            '/ask',
            self.ask_question,
            methods=["POST"],
            response_model=AskQuestionResponse,
            summary="Ask a question about carrier information"
        )

        # Endpoint 2: Multi-turn conversation
        self.router.add_api_route(
            '/conversation',
            self.handle_conversation,
            methods=["POST"],
            response_model=ConversationResponse,
            summary="Handle multi-turn conversation with carrier"
        )
        self.router.add_api_route(
            '/ask-by-email',
            self.ask_question_by_email,
            methods=["POST"],
            response_model=AskQuestionResponse,
            summary="Ask a question using carrier email (fetches carrier data)"
        )

    async def ask_question(self, request: AskQuestionRequest) -> AskQuestionResponse:
        """
        ENDPOINT 1: Ask a single question about carrier
        
        Example:
        POST /api/v1/carrier-info/ask
        {
          "carrier_data": {
            "companyName": "Swift Logistics",
            "mcNumber": "MC-789456",
            "equipmentTypes": ["53' Dry Van"]
          },
          "question": "What equipment do you have?",
          "previous_conversation": "Broker: Hi\nCarrier: Hello, looking for loads"
        }
        """
        try:
            self.logger.info(f"Processing question: {request.question}")
            
            # Import here to avoid circular imports
            from services.information_seek import InformationSeeker
            
            # Convert Pydantic model to dict
            carrier_dict = request.carrier_data.model_dump(exclude_none=True)
            
            # Initialize InformationSeeker
            seeker = InformationSeeker(data=carrier_dict)
            
            # Get answer
            answer = seeker.ask(
                question=request.question,
                prev_conversation=request.previous_conversation or ""
            )
            
            self.logger.info(
                "Question answered",
                question=request.question,
                answer_length=len(answer)
            )
            
            return AskQuestionResponse(
                answer=answer,
                carrier_company=carrier_dict.get("companyName")
            )

        except Exception as e:
            self.logger.exception("Failed to process question")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process question: {str(e)}"
            )

    async def handle_conversation(self, request: ConversationRequest) -> ConversationResponse:
        """
        ENDPOINT 2: Handle multi-turn conversation
        
        Example:
        POST /api/v1/carrier-info/conversation
        {
          "carrier_data": {
            "companyName": "Swift Logistics",
            "mcNumber": "MC-789456"
          },
          "conversation_history": [
            {"role": "broker", "message": "What's your MC?"},
            {"role": "carrier", "message": "MC-789456"}
          ],
          "new_question": "Do you have insurance?"
        }
        """
        try:
            self.logger.info(f"Processing conversation turn: {request.new_question}")
            
            from services.information_seek import InformationSeeker
            
            carrier_dict = request.carrier_data.model_dump(exclude_none=True)
            seeker = InformationSeeker(data=carrier_dict)
            
            # Build conversation context
            prev_conversation = self._format_conversation_history(
                request.conversation_history
            )
            
            # Get answer
            answer = seeker.ask(
                question=request.new_question,
                prev_conversation=prev_conversation
            )
            
            # Update conversation history
            updated_history = request.conversation_history.copy()
            updated_history.append({
                "role": "broker",
                "message": request.new_question
            })
            updated_history.append({
                "role": "carrier",
                "message": answer
            })
            
            self.logger.info(
                "Conversation turn completed",
                total_turns=len(updated_history)
            )
            
            return ConversationResponse(
                answer=answer,
                updated_conversation=updated_history,
                carrier_company=carrier_dict.get("companyName")
            )

        except Exception as e:
            self.logger.exception("Failed to handle conversation")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to handle conversation: {str(e)}"
            )

    def _format_conversation_history(self, history: list[dict]) -> str:
        """
        Format conversation history for context
        """
        if not history:
            return ""
        
        formatted = []
        for turn in history:
            role = turn.get("role", "unknown").capitalize()
            message = turn.get("message", "")
            formatted.append(f"{role}: {message}")
        
        return "\n".join(formatted)








        # **NEW ENDPOINT METHOD**
    
    
    
    async def ask_question_by_email(self, request: AskByEmailRequest) -> AskQuestionResponse:
        """
        Ask question by fetching carrier data from email
        
        Example:
        POST /api/v1/carrier-info/ask-by-email
        {
          "tenant_id": "some-tenant-id",
          "carrier_email": "400dyforever@gmail.com",
          "question": "What's your MC number?",
          "previous_conversation": "Broker: Hi\nCarrier: Hello"
        }
        """
        try:
            self.logger.info(f"Fetching carrier by email: {request.carrier_email}")
            
            # **FETCH CARRIER DATA**
            # from services import   # Import your supertruck service
            
            carrier = await supertruck.find_carrier_by_email(
                tenant_id=request.tenant_id,
                email=request.carrier_email
            )
            
            if not carrier:
                raise HTTPException(
                    status_code=404,
                    detail=f"No carrier found for email: {request.carrier_email}"
                )
            
            # **LOG CARRIER DATA STRUCTURE**
            self.logger.info("="*50)
            self.logger.info("CARRIER DATA STRUCTURE:")
            self.logger.info(json.dumps(carrier, indent=2, default=str))
            self.logger.info("="*50)
            
            # **USE INFORMATION SEEKER**
            from services.information_seek import InformationSeeker
            
            seeker = InformationSeeker(data=carrier)
            
            answer = seeker.ask(
                question=request.question,
                prev_conversation=request.previous_conversation or ""
            )
            
            self.logger.info(
                "Question answered from carrier data",
                carrier_company=carrier.get("companyName", "Unknown"),
                answer_length=len(answer)
            )
            
            return AskQuestionResponse(
                answer=answer,
                carrier_company=carrier.get("companyName")
            )

        except HTTPException:
            raise
        except Exception as e:
            self.logger.exception("Failed to process question by email")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process question: {str(e)}"
            )