"""
Negotiation Action - Refactored with Database Persistence
File: actions/negotiation.py

Key Fixes:
1. Uses NegotiationOrchestrator for unified state management
2. PERSISTS metadata with each negotiation via supertruck API
3. Proper round counting from database-backed state
4. Context-aware intent classification
"""

import logging
import json
import re
from datetime import datetime
from email_reply_parser import EmailReplyParser
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException

from integration.supertruck import SuperTruck
from integration.load_board import LoadBoard
from integration.geo_code import GeocodingService

from services.email_intent_classifier import IntentClassifierService
from services.negotiation_orchestrator import (
    NegotiationOrchestrator,
    NegotiationAction as OrchestratorAction,
    NegotiationMetadata
)
from services.information_seek import InformationSeeker
from services.load_info_seeker import LoadInfoSeekerOpenAI
from services.bid_calculator import FreightBidCalculator

from constant.enum import Load_Bid_Email_Intent, LOAD_NEGOTIATION_DIRECTION, BIDDING_TYPE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

supertruck = SuperTruck()
load_board = LoadBoard()
distance = GeocodingService()

# Initialize services
email_intent = IntentClassifierService()
loadInfo = LoadInfoSeekerOpenAI()
calculator = FreightBidCalculator()

# Initialize the orchestrator
orchestrator = NegotiationOrchestrator()


class ProcessBrokerResponseResponse(BaseModel):
    """Response after processing broker's message"""
    extraction_successful: bool
    fields_extracted: List[str]
    extracted_values: Dict[str, Any]
    updated_load_offer: Dict[str, Any]
    still_missing_critical_fields: List[str]
    can_proceed_to_negotiation: bool
    message: str


class LoadOfferData(BaseModel):
    pickupLocation: Optional[str] = None
    deliveryLocation: Optional[str] = None
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


class ProcessBrokerResponseRequest(BaseModel):
    """Request to process broker's response and update load offer"""
    broker_message: str
    curr_offer: dict
    tenant_id: str


class NegotiationAction:
    """
    Main negotiation action handler with DATABASE PERSISTENCE.

    KEY: Saves metadata with each negotiation to track:
    - negotiationRound (price discussions only)
    - infoExchangeCount (info questions)
    - conversationState
    - lastCarrierPrice / lastBrokerPrice
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.info_seeker = LoadInfoSeekerOpenAI()
        self.orchestrator = NegotiationOrchestrator()

    async def process_broker_response(self, request) -> ProcessBrokerResponseResponse:
        """Process broker's response and update load offer"""
        try:
            # Handle both dict and Pydantic model
            if isinstance(request, dict):
                broker_message = request.get('broker_message', '')
                curr_offer = request.get('curr_offer', {})
                tenant_id = request.get('tenant_id', '')
            else:
                broker_message = request.broker_message
                curr_offer = request.curr_offer
                tenant_id = request.tenant_id

            self.logger.info("="*60)
            self.logger.info("PROCESSING BROKER RESPONSE")
            self.logger.info(f"Message: {broker_message[:100]}...")

            if not curr_offer:
                raise ValueError("No current load offer found.")

            self.logger.info(f"Offer ID: {curr_offer.get('id')}")
            self.logger.info(f"Confidence score: {curr_offer.get('loadConfidentialScore', 'N/A')}")

            # Extract information from broker's message
            extraction_result = self.info_seeker.extract_info_from_response(
                broker_message,
                curr_offer,
                None
            )

            self.logger.info(f"Extraction: {extraction_result.get('extraction_successful', False)}")

            if not extraction_result.get('extraction_successful', False):
                self.logger.warning("No information extracted from broker message")
                return ProcessBrokerResponseResponse(
                    extraction_successful=False,
                    fields_extracted=[],
                    extracted_values={},
                    updated_load_offer=curr_offer,
                    still_missing_critical_fields=[],
                    can_proceed_to_negotiation=False,
                    message="Thank you for your interest. To proceed, could you please provide more details about the load, such as pickup/delivery locations, dates, and weight?"
                )

            # Merge extracted info
            updated_load = self.info_seeker.merge_extracted_info(
                curr_offer,
                extraction_result.get('extracted_fields', {})
            )

            self.logger.info(f"Updated fields: {list(extraction_result.get('extracted_fields', {}).keys())}")

            # Check what's still missing
            still_missing_info = self.info_seeker.identify_missing_fields(updated_load, "offer")
            still_missing_critical = still_missing_info.get('missing_critical_fields', [])
            can_proceed = still_missing_info.get('has_sufficient_info_for_negotiation', False)

            self.logger.info(f"Still missing: {still_missing_critical}")
            self.logger.info(f"Can proceed: {can_proceed}")

            # Create response message
            if can_proceed:
                message = "Great! All critical information received. Ready to negotiate."
            else:
                from services.load_info_seeker import FIELD_DESCRIPTIONS
                missing_readable = [FIELD_DESCRIPTIONS.get(f, f) for f in still_missing_critical]
                message = f"Thanks for the info. I just need a few more details: {', '.join(missing_readable)}. Could you please provide them?"

            await load_board.update_load_offer(curr_offer["id"], updated_load)
            self.logger.info("Load offer updated successfully")
            self.logger.info("="*60)

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
            self.logger.exception("Error processing broker response")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing broker's response. Please try again."
            )

    async def find_bid(self, arg0: dict):
        """Find existing bid for carrier/thread"""
        try:
            self.logger.info("Searching for bid...")
            tenant_id = arg0.get("tenant_id")
            carrier_id = arg0.get("carrier_id")
            from_email = arg0.get("from")
            thread_id = arg0.get("thread_Id")

            if not tenant_id or not carrier_id or not from_email or not thread_id:
                raise ValueError(f"Missing required fields: {arg0}")

            bid_list = await supertruck.get_bid(
                tenant_id=tenant_id,
                params={
                    "carrierId": carrier_id,
                    "threadId": thread_id,
                }
            )

            if not bid_list or len(bid_list) == 0:
                raise LookupError("No bid found for the given criteria")

            self.logger.info(f"Found bid: {bid_list[0].get('id')}")
            return bid_list[0]

        except Exception as e:
            self.logger.error(f"Error finding bid: {e}")
            raise

    def _get_persisted_state(self, negotiations: List[Dict]) -> Dict[str, Any]:
        """Extract persisted state from last negotiation with metadata"""
        for neg in reversed(negotiations):
            metadata = neg.get('metadata')
            if metadata:
                return {
                    'negotiationRound': metadata.get('negotiationRound', 0),
                    'infoExchangeCount': metadata.get('infoExchangeCount', 0),
                    'conversationState': metadata.get('conversationState', 'initial'),
                    'lastCarrierPrice': metadata.get('lastCarrierPrice'),
                    'lastBrokerPrice': metadata.get('lastBrokerPrice'),
                    'belowMinCount': metadata.get('belowMinCount', 0)
                }
        return {
            'negotiationRound': 0,
            'infoExchangeCount': 0,
            'conversationState': 'initial',
            'lastCarrierPrice': None,
            'lastBrokerPrice': None,
            'belowMinCount': 0
        }

    async def execute_negotiation(self, data: dict):
        """
        Main negotiation execution with DATABASE PERSISTENCE.

        KEY CHANGES:
        1. Reads state from persisted metadata in negotiations
        2. Saves metadata with BOTH incoming and outgoing messages
        3. Metadata tracks negotiationRound, infoExchangeCount, etc.
        """
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("NEGOTIATION ACTION STARTED (With Persistence)")
            self.logger.info("="*80)

            tenant_id = data["tenant_id"]
            thread_id = data.get("thread_Id") or data.get("threadId")
            message_id = data.get("messageId")

            self.logger.info(f"Tenant ID: {tenant_id}")
            self.logger.info(f"Thread ID: {thread_id}")
            self.logger.info(f"From: {data.get('from')}")

            # Extract latest message (remove quoted replies)
            split_text = re.split(r"On .* wrote:", data["body"], maxsplit=1)
            latest = split_text[0].strip()

            self.logger.info("="*60)
            self.logger.info("LATEST MESSAGE:")
            self.logger.info(f"{latest[:200]}...")
            self.logger.info("="*60)

            # Find carrier
            self.logger.info("Finding carrier...")
            carrier = await supertruck.find_carrier_by_email(
                tenant_id=tenant_id,
                email="400dyforever@gmail.com"
            )
            if not carrier:
                raise LookupError("No carrier found for email")

            self.logger.info(f"Carrier found: {carrier.get('id')}")

            # Find bid
            params = {
                "tenant_id": tenant_id,
                "carrier_id": carrier["id"],
                "from": data["from"],
                "thread_Id": thread_id
            }

            currBid = await self.find_bid(arg0=params)
            if not currBid:
                raise ValueError("No current bid found for negotiation.")

            self.logger.info(f"Bid found: {currBid.get('id')}")
            self.logger.info(f"Bid type: {currBid.get('biddingType')}")
            self.logger.info(f"Base rate: ${currBid.get('baseRate')}")

            # Get load offer
            curr_offer = await load_board.get_load_offer_by_id(currBid["entityId"])
            if not curr_offer:
                raise ValueError("No current load offer found.")

            self.logger.info(f"Load offer found: {curr_offer.get('id')}")
            self.logger.info(f"Pickup: {curr_offer.get('pickupLocation')}")
            self.logger.info(f"Delivery: {curr_offer.get('deliveryLocation')}")
            self.logger.info(f"Confidence score: {curr_offer.get('loadConfidentialScore')}")

            # Get conversation history (includes metadata from previous negotiations)
            conversation_history = currBid.get("negotiations", [])
            self.logger.info(f"Conversation history: {len(conversation_history)} messages")

            # Log persisted state from last message
            persisted_state = self._get_persisted_state(conversation_history)
            self.logger.info(f"Persisted state from DB:")
            self.logger.info(f"  - negotiationRound: {persisted_state['negotiationRound']}")
            self.logger.info(f"  - infoExchangeCount: {persisted_state['infoExchangeCount']}")
            self.logger.info(f"  - conversationState: {persisted_state['conversationState']}")
            self.logger.info(f"  - lastCarrierPrice: ${persisted_state['lastCarrierPrice']}")
            self.logger.info(f"  - lastBrokerPrice: ${persisted_state['lastBrokerPrice']}")

            bidding_type = currBid.get("biddingType")

            # Manual bids - save incoming but skip auto-response
            if bidding_type == BIDDING_TYPE.MANUAL.value:
                self.logger.info('Manual bid - saving incoming, skipping auto-response')
                await supertruck.negotiate(tenant_id=tenant_id, data={
                    "rate": currBid.get("baseRate"),
                    "negotiationDirection": "incoming",
                    "bidId": currBid["id"],
                    "negotiationRawEmail": latest,
                    "messageId": message_id,
                    "inReplyTo": data.get("inReplyTo"),
                    "references": data.get("references"),
                    "metadata": {
                        "messageType": "MANUAL_REVIEW",
                        "negotiationRound": persisted_state['negotiationRound'],
                        "infoExchangeCount": persisted_state['infoExchangeCount'],
                        "conversationState": persisted_state['conversationState'],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                })
                return

            # Auto bids - use orchestrator with full persistence
            if bidding_type == BIDDING_TYPE.AUTO.value:
                self.logger.info('Auto bid - processing with orchestrator')

                # Determine pricing
                min_rate = float(currBid.get("minRate", 0))
                max_rate = float(currBid.get("maxRate", 0))

                # If no rates set, calculate them
                if min_rate <= 0 or max_rate <= 0:
                    confidence_score = curr_offer.get("loadConfidentialScore", 0)
                    if confidence_score < 80:
                        self.logger.info(f"Low confidence ({confidence_score}) - checking for info")

                        info_check = await self.process_broker_response({
                            "broker_message": latest,
                            "tenant_id": tenant_id,
                            "curr_offer": curr_offer
                        })

                        if len(info_check.still_missing_critical_fields) > 0:
                            self.logger.info(f"Still need info: {info_check.still_missing_critical_fields}")

                            # Increment info exchange count
                            new_info_count = persisted_state['infoExchangeCount'] + 1

                            # Save incoming with INFO metadata
                            await supertruck.negotiate(tenant_id=tenant_id, data={
                                "rate": None,
                                "negotiationDirection": "incoming",
                                "bidId": currBid["id"],
                                "negotiationRawEmail": latest,
                                "messageId": message_id,
                                "inReplyTo": data.get("inReplyTo"),
                                "references": data.get("references"),
                                "metadata": {
                                    "messageType": "INFO_RESPONSE",
                                    "negotiationRound": persisted_state['negotiationRound'],
                                    "infoExchangeCount": new_info_count,
                                    "conversationState": "info_gathering",
                                    "lastCarrierPrice": persisted_state['lastCarrierPrice'],
                                    "lastBrokerPrice": persisted_state['lastBrokerPrice'],
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            })

                            # Save outgoing info request with metadata
                            await supertruck.negotiate(tenant_id=tenant_id, data={
                                "rate": None,
                                "negotiationDirection": "outgoing",
                                "bidId": currBid["id"],
                                "negotiationRawEmail": info_check.message,
                                "metadata": {
                                    "messageType": "INFO_REQUEST",
                                    "negotiationRound": persisted_state['negotiationRound'],
                                    "infoExchangeCount": new_info_count,
                                    "conversationState": "info_gathering",
                                    "lastCarrierPrice": persisted_state['lastCarrierPrice'],
                                    "lastBrokerPrice": persisted_state['lastBrokerPrice'],
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            })

                            self.logger.info(f"Info request sent with metadata: {info_check.message}")
                            return

                        # Got enough info, update offer
                        curr_offer = info_check.updated_load_offer

                    # Calculate distance and rates
                    total_distance = await distance.calculate_distance(
                        curr_offer.get("pickupLocation", ""),
                        curr_offer.get("deliveryLocation", "")
                    )
                    self.logger.info(f"Distance: {total_distance.get('distance')} miles")

                    rate_calc = await calculator.calculate_bid(
                        float(total_distance.get("distance", 500))
                    )
                    min_rate = rate_calc.min_rate
                    max_rate = rate_calc.max_rate

                    self.logger.info(f"Calculated rates: ${min_rate} - ${max_rate}")

                    # Update bid with rates
                    await supertruck.update_bid(
                        tenant_id=tenant_id,
                        bid_id=currBid["id"],
                        data={
                            "minRate": min_rate,
                            "maxRate": max_rate,
                            "baseRate": curr_offer.get("requestedRate", min_rate)
                        }
                    )

                # ============================================================
                # USE ORCHESTRATOR - Gets metadata from DB, returns new metadata
                # ============================================================
                self.logger.info("\n" + "="*60)
                self.logger.info("USING NEGOTIATION ORCHESTRATOR")
                self.logger.info("="*60)

                result = self.orchestrator.process_message(
                    broker_message=latest,
                    negotiation_history=conversation_history,
                    load_offer=curr_offer,
                    pricing={
                        "min_price": min_rate,
                        "max_price": max_rate,
                        "sweet_spot": min_rate + (max_rate - min_rate) * 0.6
                    },
                    carrier_info=carrier
                )

                self.logger.info(f"Orchestrator result:")
                self.logger.info(f"  Action: {result.action.value}")
                self.logger.info(f"  Status: {result.status}")
                self.logger.info(f"  Price: ${result.proposed_price}")
                self.logger.info(f"  Response: {result.response[:100]}...")
                self.logger.info(f"  Reasoning: {result.reasoning}")

                # Get metadata to persist
                metadata_dict = result.metadata.to_dict() if result.metadata else {}
                self.logger.info(f"New metadata to persist:")
                self.logger.info(f"  - negotiationRound: {metadata_dict.get('negotiationRound')}")
                self.logger.info(f"  - infoExchangeCount: {metadata_dict.get('infoExchangeCount')}")
                self.logger.info(f"  - conversationState: {metadata_dict.get('conversationState')}")
                self.logger.info(f"  - messageType: {metadata_dict.get('messageType')}")

                # ============================================================
                # SAVE INCOMING MESSAGE WITH METADATA
                # ============================================================
                incoming_metadata = metadata_dict.copy()
                incoming_metadata["direction"] = "incoming"

                await supertruck.negotiate(tenant_id=tenant_id, data={
                    "rate": result.metadata.extractedPrice if result.metadata else None,
                    "negotiationDirection": "incoming",
                    "bidId": currBid["id"],
                    "negotiationRawEmail": latest,
                    "messageId": message_id,
                    "inReplyTo": data.get("inReplyTo"),
                    "references": data.get("references"),
                    "metadata": incoming_metadata
                })
                self.logger.info("Incoming negotiation saved WITH METADATA")

                # ============================================================
                # UPDATE BID STATUS IF ACCEPTED/REJECTED
                # ============================================================
                if result.status == "accepted":
                    await supertruck.update_bid(
                        tenant_id=tenant_id,
                        bid_id=currBid["id"],
                        data={"isAcceptedByBroker": True, "status": "accepted"}
                    )
                    self.logger.info("Bid marked as accepted")

                elif result.status == "rejected":
                    await supertruck.update_bid(
                        tenant_id=tenant_id,
                        bid_id=currBid["id"],
                        data={"isAcceptedByBroker": False, "status": "rejected"}
                    )
                    self.logger.info("Bid marked as rejected")

                # ============================================================
                # SAVE OUTGOING RESPONSE WITH METADATA
                # ============================================================
                outgoing_metadata = metadata_dict.copy()
                outgoing_metadata["direction"] = "outgoing"
                if result.proposed_price:
                    outgoing_metadata["lastCarrierPrice"] = result.proposed_price

                await supertruck.negotiate(tenant_id=tenant_id, data={
                    "rate": result.proposed_price,
                    "negotiationDirection": "outgoing",
                    "bidId": currBid["id"],
                    "negotiationRawEmail": result.response,
                    "metadata": outgoing_metadata
                })
                self.logger.info("Outgoing response saved WITH METADATA")

            self.logger.info("\n" + "="*80)
            self.logger.info("NEGOTIATION ACTION COMPLETED")
            self.logger.info("="*80 + "\n")

        except Exception as e:
            self.logger.error("\n" + "="*80)
            self.logger.error("ERROR IN EXECUTE_NEGOTIATION")
            self.logger.error(f"Error: {str(e)}")
            self.logger.exception("Full traceback:")
            self.logger.error("="*80 + "\n")
            raise

    async def execute_negotiation_legacy(self, data: dict):
        """
        Legacy negotiation execution (kept for reference).
        Use execute_negotiation instead.
        """
        self.logger.warning("Using legacy negotiation - consider using execute_negotiation instead")
        # Legacy code omitted - use execute_negotiation instead
        pass
