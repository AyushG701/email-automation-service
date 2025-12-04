

import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.load_info_seeker import LoadInfoSeekerOpenAI

logger = logging.getLogger(__name__)


# ==================== REQUEST SCHEMAS ====================

class LoadOfferData(BaseModel):
    """Load offer data structure"""
    pickupLocation: Optional[str] = None
    dropoffLocation: Optional[str] = None
    pickupDate: Optional[str] = None
    deliveryDate: Optional[str] = None
    equipmentType: Optional[str] = None
    weightLbs: Optional[float] = None
    equipmentSize: Optional[str] = None
    commodity: Optional[str] = None
    rate: Optional[float] = None
    brokerContactPhone: Optional[str] = None
    brokerContactEmail: Optional[str] = None
    brokerCompany: Optional[str] = None


class CheckMissingFieldsRequest(BaseModel):
    """Request to check what fields are missing"""
    load_offer: LoadOfferData

    class Config:
        json_schema_extra = {
            "example": {
                "load_offer": {
                    "pickupLocation": "Dallas",
                    "dropoffLocation": None,
                    "pickupDate": "2025-01-20",
                    "deliveryDate": None,
                    "equipmentType": None,
                    "weightLbs": None,
                    "brokerCompany": "ABC Freight"
                }
            }
        }


class GenerateMessageRequest(BaseModel):
    """Request to generate message for broker"""
    load_offer: LoadOfferData
    broker_company: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "load_offer": {
                    "pickupLocation": "Dallas",
                    "dropoffLocation": None,
                    "pickupDate": "2025-01-20",
                    "deliveryDate": None,
                    "equipmentType": None,
                    "weightLbs": None
                },
                "broker_company": "ABC Freight"
            }
        }


class ProcessBrokerResponseRequest(BaseModel):
    """Request to process broker's response and update load offer"""
    broker_message: str
    current_load_offer: LoadOfferData

    class Config:
        json_schema_extra = {
            "example": {
                "broker_message": "PU: Dallas, TX 75201, DEL: Atlanta, GA 30301, 1/20-1/21, 53' DV, 44k lbs",
                "current_load_offer": {
                    "pickupLocation": "Dallas",
                    "brokerCompany": "ABC Freight"
                }
            }
        }


# ==================== RESPONSE SCHEMAS ====================

class CheckMissingFieldsResponse(BaseModel):
    """Response showing what fields are missing"""
    missing_critical_fields: List[str]
    missing_optional_fields: List[str]
    field_descriptions: Dict[str, str]
    has_sufficient_info: bool
    priority_level: str
    total_missing: int


class GenerateMessageResponse(BaseModel):
    """Response with generated message for broker"""
    message: str
    should_send: bool
    missing_fields: List[str]


class ProcessBrokerResponseResponse(BaseModel):
    """Response after processing broker's message"""
    extraction_successful: bool
    fields_extracted: List[str]
    extracted_values: Dict[str, Any]
    updated_load_offer: Dict[str, Any]
    still_missing_critical_fields: List[str]
    can_proceed_to_negotiation: bool
    message: str


# ==================== CONTROLLER ====================

class LoadInfoSeekController:
    """Simple controller with 3 clear endpoints"""

    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger(__name__)
        self.router = APIRouter(prefix='/api/v1/load-info', tags=['Load Info Seeker'])
        self.info_seeker = LoadInfoSeekerOpenAI()

        # Register routes
        self._register_routes()
        app.include_router(self.router)

    def _register_routes(self):
        """Register the 3 main endpoints"""
        
        # Endpoint 1: Check what's missing
        self.router.add_api_route(
            '/check-missing',
            self.check_missing_fields,
            methods=["POST"],
            response_model=CheckMissingFieldsResponse,
            summary="Check what fields are missing in load offer"
        )

        # Endpoint 2: Generate message to send to broker
        self.router.add_api_route(
            '/generate-message',
            self.generate_message,
            methods=["POST"],
            response_model=GenerateMessageResponse,
            summary="Generate message to request missing info from broker"
        )

        # Endpoint 3: Process broker's response and update load offer
        self.router.add_api_route(
            '/process-broker-response',
            self.process_broker_response,
            methods=["POST"],
            response_model=ProcessBrokerResponseResponse,
            summary="Process broker's response and update load offer"
        )

    async def check_missing_fields(self, request: CheckMissingFieldsRequest) -> CheckMissingFieldsResponse:
        """
        ENDPOINT 1: Check what fields are missing
        
        Example:
        POST /api/v1/load-info/check-missing
        {
          "load_offer": {
            "pickupLocation": "Dallas",
            "dropoffLocation": null,
            "pickupDate": "2025-01-20",
            "deliveryDate": null,
            "equipmentType": null,
            "weightLbs": null,
            "brokerCompany": "ABC Freight"
          }
        }
        
        Response will show:
        - What critical fields are missing
        - What optional fields are missing
        - Whether you have enough info to negotiate
        """
        try:
            self.logger.info("Checking missing fields")
            
            load_offer_dict = request.load_offer.model_dump()
            
            # Call service to identify missing fields
            missing_info = self.info_seeker.identify_missing_fields(load_offer_dict, "offer")
            
            # Get field descriptions
            from services.load_info_seeker import FIELD_DESCRIPTIONS
            all_missing = (missing_info.get('missing_critical_fields', []) + 
                          missing_info.get('missing_optional_fields', []))
            
            field_descriptions = {
                field: FIELD_DESCRIPTIONS.get(field, field) 
                for field in all_missing
            }
            
            self.logger.info(
                "Missing fields identified",
                critical_count=len(missing_info.get('missing_critical_fields', [])),
                optional_count=len(missing_info.get('missing_optional_fields', [])),
                sufficient=missing_info.get('has_sufficient_info_for_negotiation', False)
            )
            
            return CheckMissingFieldsResponse(
                missing_critical_fields=missing_info.get('missing_critical_fields', []),
                missing_optional_fields=missing_info.get('missing_optional_fields', []),
                field_descriptions=field_descriptions,
                has_sufficient_info=missing_info.get('has_sufficient_info_for_negotiation', False),
                priority_level=missing_info.get('priority_level', 'medium'),
                total_missing=len(all_missing)
            )

        except Exception as e:
            self.logger.exception("Failed to check missing fields")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to check missing fields: {str(e)}"
            )

    async def generate_message(self, request: GenerateMessageRequest) -> GenerateMessageResponse:
        """
        ENDPOINT 2: Generate message to send to broker
        
        Example:
        POST /api/v1/load-info/generate-message
        {
          "load_offer": {
            "pickupLocation": "Dallas",
            "dropoffLocation": null,
            "pickupDate": "2025-01-20",
            "deliveryDate": null,
            "equipmentType": null,
            "weightLbs": null
          },
          "broker_company": "ABC Freight"
        }
        
        Response will give you:
        - Natural message to send to broker
        - List of what you're asking for
        """
        try:
            self.logger.info("Generating message for broker")
            
            load_offer_dict = request.load_offer.model_dump()
            
            # First identify what's missing
            missing_info = self.info_seeker.identify_missing_fields(load_offer_dict, "offer")
            
            # Generate natural message
            message_result = self.info_seeker.generate_request_message(
                missing_info,
                broker_company=request.broker_company or request.load_offer.brokerCompany
            )
            
            missing_fields = missing_info.get('missing_critical_fields', []) + missing_info.get('missing_optional_fields', [])
            
            self.logger.info(
                "Message generated",
                message_length=len(message_result.get('message', '')),
                should_send=message_result.get('should_send', False)
            )
            
            return GenerateMessageResponse(
                message=message_result.get('message', ''),
                should_send=message_result.get('should_send', False),
                missing_fields=missing_fields
            )

        except Exception as e:
            self.logger.exception("Failed to generate message")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate message: {str(e)}"
            )

    async def process_broker_response(self, request: ProcessBrokerResponseRequest) -> ProcessBrokerResponseResponse:
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
            
            current_load_dict = request.current_load_offer.model_dump()
            
            # Extract information from broker's message
            extraction_result = self.info_seeker.extract_info_from_response(
                request.broker_message,
                current_load_dict,
                None  # Will use current date
            )
            
            # Check if extraction was successful
            if not extraction_result.get('extraction_successful', False):
                self.logger.warning("No information extracted from broker message")
                return ProcessBrokerResponseResponse(
                    extraction_successful=False,
                    fields_extracted=[],
                    extracted_values={},
                    updated_load_offer=current_load_dict,
                    still_missing_critical_fields=[],
                    can_proceed_to_negotiation=False,
                    message="Could not extract any information from broker's message. Please check the message format."
                )
            
            # Merge extracted info into current load offer
            updated_load = self.info_seeker.merge_extracted_info(
                current_load_dict,
                extraction_result.get('extracted_fields', {})
            )
            
            # Check what's still missing
            still_missing_info = self.info_seeker.identify_missing_fields(updated_load, "offer")
            still_missing_critical = still_missing_info.get('missing_critical_fields', [])
            can_proceed = still_missing_info.get('has_sufficient_info_for_negotiation', False)
            
            # Create response message
            if can_proceed:
                message = "Great! All critical information received. You can now proceed to negotiation."
            else:
                from services.load_info_seeker import FIELD_DESCRIPTIONS
                missing_readable = [FIELD_DESCRIPTIONS.get(f, f) for f in still_missing_critical]
                message = f"Still missing: {', '.join(missing_readable)}. Need to request more information."
            
            self.logger.info(
                "Broker response processed",
                extracted_count=len(extraction_result.get('fields_found', [])),
                still_missing_count=len(still_missing_critical),
                can_proceed=can_proceed
            )
            
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

