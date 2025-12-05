"""
Negotiation Controller - PRODUCTION READY
File: controllers/negotiation_controller.py

Key Improvements:
- Handles partial load data gracefully
- Robust chat history processing
- Real-world style responses
- Comprehensive error handling
- Production-grade logging
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from services.negotiation_orchestrator import NegotiationOrchestrator
from services.load_info_seeker import LoadInfoSeekerOpenAI
from services.bid_calculator import FreightBidCalculator


from integration.supertruck import SuperTruck
from integration.load_board import LoadBoard
from integration.geo_code import GeocodingService
from email_reply_parser import EmailReplyParser


logger = logging.getLogger(__name__)

negotiation_orchestrator = NegotiationOrchestrator()
supertruck = SuperTruck()
load_board = LoadBoard()
distance = GeocodingService()
calculator = FreightBidCalculator()


class NegotiationHistoryItem(BaseModel):
    """Single negotiation message with flexible parsing"""
    negotiationDirection: str = Field(
        ..., description="'incoming' (broker) or 'outgoing' (carrier)")
    negotiationRawEmail: str = Field(...,
                                     description="The actual message text")
    rate: Optional[float] = Field(None, description="Price mentioned (if any)")


class NegotiationResponse(BaseModel):
    """Concise real-world response format"""
    response: str = Field(..., max_length=9000)
    proposed_price: Optional[str] = None
    status: str = Field(..., pattern="^(negotiating|accepted|rejected)$")
    min_price: Optional[float] = None
    max_price: Optional[float] = None

    # negotiation_round: int = Field(..., ge=1)

# ============================================================================
# CONTROLLER - PRODUCTION GRADE
# ============================================================================


class ProcessBrokerResponseRequest(BaseModel):
    """Request to process broker's response and update load offer"""
    broker_message: str
    curr_load_offer_id: str


class ProcessBrokerResponseResponse(BaseModel):
    """Response after processing broker's message"""
    extraction_successful: bool
    fields_extracted: List[str]
    extracted_values: Dict[str, Any]
    updated_load_offer: Dict[str, Any]
    still_missing_critical_fields: List[str]
    can_proceed_to_negotiation: bool
    message: str


class NegotiationSchema(BaseModel):
    broker_id: Optional[str] = None
    carrier_id: str
    entity_id: str
    message: str
    max_price: Optional[int] = None
    min_price: Optional[int] = None
    broker_email: str

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Responding to your offer",
                "entity_id": "e2b47fe4-573f-45a1-97fc-d2ff3a2171e0",
                "carrier_id": "CARR-001",
                "broker_email": "broker@example.com",
                "max_price": 1800,
                "min_price": 2500,
                "broker_id": "BROKER-555"
            }
        }


class NegotiationController:
    def __init__(self):
        self.router = APIRouter(prefix="/api/v1", tags=["Negotiation"])
        self.negotiation_orchestrator = NegotiationOrchestrator()
        self.logger = logging.getLogger(__name__)
        self.info_seeker = LoadInfoSeekerOpenAI()

        self.router.add_api_route(
            "/negotiate",
            self.bid_negotiation,
            methods=["POST"],
            response_model=NegotiationResponse
        )

    async def process_broker_response(self, request) -> ProcessBrokerResponseResponse:
        """
        ENDPOINT 3: Process broker's response and update load offer

        Example:
        POST /api/v1/load-info/process-broker-response
        {
          "broker_message": "PU: Dallas, TX 75201, DEL: Atlanta, GA 30301, 1/20-1/21, 53' DV, 44k lbs",
          "current_load_offer": {
            "pickupLocation": "Dallas",
            "brokerCompany": "ABC Freight"
          }
        }

        Response will show:
        - What information was extracted from broker's message
        - Updated load offer with extracted info
        - What's still missing (if anything)
        - Whether you can now proceed to negotiation
        """
        try:
            self.logger.info("Processing broker response")
            self.logger.info(request)
            curr_offer = request["load_offer"]

            # # current_load_dict = request.current_load_offer.model_dump()
            # curr_offer = await load_board.get_load_offer_by_id(request["curr_load_offer_id"])
            if not curr_offer:
                raise ValueError("No current load offer found.")
            # Extract information from broker's message
            extraction_result = self.info_seeker.extract_info_from_response(
                request["broker_message"],
                curr_offer,
                None  # Will use current date
            )
            # Check if extraction was successful
            if not extraction_result.get('extraction_successful', False):
                self.logger.warning(
                    "No information extracted from broker message")

                # Identify missing fields even if extraction fails
                missing_info = self.info_seeker.identify_missing_fields(
                    curr_offer, "offer")
                still_missing_critical = missing_info.get(
                    'missing_critical_fields', [])
                
                from services.load_info_seeker import FIELD_DESCRIPTIONS
                missing_readable = [FIELD_DESCRIPTIONS.get(f, f) for f in still_missing_critical]
                message = f"Thanks for your interest. To give you a quote, I need a few more details: {', '.join(missing_readable)}. Could you please provide them?" if missing_readable else "Thanks for your interest. Please provide the load details so I can give you a quote."

                return ProcessBrokerResponseResponse(
                    extraction_successful=False,
                    fields_extracted=[],
                    extracted_values={},
                    updated_load_offer=curr_offer,
                    still_missing_critical_fields=still_missing_critical,
                    can_proceed_to_negotiation=False,
                    message=message
                )

            # Merge extracted info into current load offer
            updated_load = self.info_seeker.merge_extracted_info(
                curr_offer,
                extraction_result.get('extracted_fields', {})
            )

            # Check what's still missing
            still_missing_info = self.info_seeker.identify_missing_fields(
                updated_load, "offer")
            still_missing_critical = still_missing_info.get(
                'missing_critical_fields', [])
            can_proceed = still_missing_info.get(
                'has_sufficient_info_for_negotiation', False)

            # Create response message
            if can_proceed:
                message = "Great! All critical information received. You can now proceed to negotiation."
            else:
                from services.load_info_seeker import FIELD_DESCRIPTIONS
                missing_readable = [FIELD_DESCRIPTIONS.get(
                    f, f) for f in still_missing_critical]
                message = f"Thanks for the information. I just need a few more details: {', '.join(missing_readable)}. Could you please provide them?"

            # self.logger.info(
            #     "Broker response processed",
            #     extracted_count=len(extraction_result.get('fields_found', [])),
            #     still_missing_count=len(still_missing_critical),
            #     can_proceed=can_proceed
            # )

            return ProcessBrokerResponseResponse(
                extraction_successful=True,
                fields_extracted=extraction_result.get('fields_found', []),
                extracted_values=extraction_result.get('extracted_fields', {}),
                updated_load_offer=updated_load,
                still_missing_critical_fields=still_missing_critical,
                can_proceed_to_negotiation=can_proceed,
                message=message
            )

        except Exception as e:
            self.logger.exception("Failed to process broker response")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process broker response: {str(e)}"
            )

    async def bid_negotiation(self, payload: NegotiationSchema):
        try:
            load_offer = await load_board.get_load_offer_by_id(id=payload.entity_id)

            if (load_offer["loadConfidentialScore"] > 90):
                total_distance = await distance.calculate_distance(
                    load_offer["pickupLocation"],
                    load_offer.get('dropoffLocation') or load_offer.get('deliveryLocation')
                )
                self.logger.info(
                    "Distance calculated")
                rate_calc = await calculator.calculate_bid(
                    float(total_distance["distance"]),
                    float(load_offer.get("requestedRate", 0))
                )
                self.logger.info(
                    "Min & Max rate calculated")

                parsed_email_body = EmailReplyParser.parse_reply(
                    load_offer["offerEmail"])
                # *AI NEGOTIATION
                result = negotiation_orchestrator.process_message(
                    broker_message=parsed_email_body,
                    negotiation_history=[],
                    load_offer=load_offer,
                    pricing={
                        'min_price': rate_calc.min_rate,
                        'max_price': rate_calc.max_rate
                    }
                )
                return NegotiationResponse(
                    response=result.response,
                    proposed_price=str(
                        result.proposed_price) if result.proposed_price else None,
                    status=result.status,
                    min_price=rate_calc.min_rate,
                    max_price=rate_calc.max_rate
                    # negotiation_round=negotiation_round
                )
            else:
                info_check = await self.process_broker_response({
                    "broker_message": load_offer["offerEmail"],
                    "load_offer": load_offer
                })
                self.logger.info(
                    f"Load info checked for negotiation readiness {info_check}")
                if info_check.can_proceed_to_negotiation:
                    total_distance = await distance.calculate_distance(
                        info_check.updated_load_offer["pickupLocation"],
                        info_check.updated_load_offer["dropoffLocation"]
                    )
                    self.logger.info(
                        "Distance calculated")
                    if not total_distance:
                        raise ValueError("Could not calculate distance. Please check pickup and delivery locations.")
                    
                    rate_calc = await calculator.calculate_bid(
                        float(total_distance["distance"]),
                        float(load_offer.get("requestedRate", 0))
                    )
                    self.logger.info(
                        "Min & Max rate calculated")
                    parsed_email_body = EmailReplyParser.parse_reply(
                        load_offer["offerEmail"])
                    # *AI NEGOTIATION
                    result = negotiation_orchestrator.process_message(
                        broker_message=parsed_email_body,
                        negotiation_history=[],
                        load_offer=info_check.updated_load_offer,
                        pricing={
                            'min_price': rate_calc.min_rate,
                            'max_price': rate_calc.max_rate
                        }
                    )
                    self.logger.info(
                        "Info checked and continue for conversation")
                    # ASK The detail with broker
                    # Return response
                    return NegotiationResponse(
                        response=result.response,
                        proposed_price=str(
                            result.proposed_price) if result.proposed_price else None,
                        status=result.status,
                        min_price=rate_calc.min_rate,
                        max_price=rate_calc.max_rate
                        # negotiation_round=negotiation_round
                    )

                else:
                    self.logger.info(
                        "Missing critical info, cannot proceed to negotiation")
                    return NegotiationResponse(
                        response=info_check.message,
                        proposed_price=None,
                        status="negotiating",
                        # negotiation_round=negotiation_round
                    )

        except ValueError as ve:
            self.logger.warning(f"Validation error: {str(ve)}")
            raise HTTPException(status_code=400, detail=str(ve))

        except Exception as e:
            self.logger.exception(f"Critical negotiation failure,{e}")
            raise HTTPException(
                status_code=500,
                detail="Negotiation service unavailable. Please try again."
            )
