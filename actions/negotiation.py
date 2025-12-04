# import logging
# import json
# import re
# from email_reply_parser import EmailReplyParser
# from pydantic import BaseModel, Field
# from typing import Optional, List, Dict, Any
# from fastapi import APIRouter, HTTPException

# from integration.supertruck import SuperTruck
# from integration.load_board import LoadBoard
# from integration.geo_code import GeocodingService

# from services.email_intent_classifier import IntentClassifierService
# from services.negotiation import NegotiationService
# from services.information_seek import InformationSeeker
# from services.load_info_seeker import LoadInfoSeekerOpenAI
# from services.bid_calculator import FreightBidCalculator

# from constant.enum import Load_Bid_Email_Intent, LOAD_NEGOTIATION_DIRECTION, BIDDING_TYPE

# supertruck = SuperTruck()
# load_board = LoadBoard()
# distance = GeocodingService()


# negotiation = NegotiationService()
# email_intent = IntentClassifierService()
# loadInfo = LoadInfoSeekerOpenAI()
# calculator = FreightBidCalculator()
# calculator = FreightBidCalculator()


# """
# NEGOTIATION, CLASSIFY AND FIND EMAIL INTENT
# """


# class ProcessBrokerResponseResponse(BaseModel):
#     """Response after processing broker's message"""
#     extraction_successful: bool
#     fields_extracted: List[str]
#     extracted_values: Dict[str, Any]
#     updated_load_offer: Dict[str, Any]
#     still_missing_critical_fields: List[str]
#     can_proceed_to_negotiation: bool
#     message: str


# class LoadOfferData(BaseModel):
#     # """Load offer data structure"""
#     pickupLocation: Optional[str] = None
#     dropoffLocation: Optional[str] = None
#     pickupDate: Optional[str] = None
#     deliveryDate: Optional[str] = None
#     equipmentType: Optional[str] = None
#     weightLbs: Optional[float] = None
#     equipmentSize: Optional[str] = None
#     commodity: Optional[str] = None
#     rate: Optional[float] = None
#     brokerContactPhone: Optional[str] = None
#     brokerContactEmail: Optional[str] = None
#     brokerCompany: Optional[str] = None


# class ProcessBrokerResponseRequest(BaseModel):
#     """Request to process broker's response and update load offer"""
#     broker_message: str
#     curr_offer: dict
#     tenant_id: str


# class NegotiationAction:
#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
#         self.info_seeker = LoadInfoSeekerOpenAI()

#     async def process_broker_response(self, request: ProcessBrokerResponseRequest) -> ProcessBrokerResponseResponse:
#         """
#         ENDPOINT 3: Process broker's response and update load offer

#         Example:
#         POST /api/v1/load-info/process-broker-response
#         {
#           "broker_message": "PU: Dallas, TX 75201, DEL: Atlanta, GA 30301, 1/20-1/21, 53' DV, 44k lbs",
#           "current_load_offer": {
#             "pickupLocation": "Dallas",
#             "brokerCompany": "ABC Freight"
#           }
#         }

#         Response will show:
#         - What information was extracted from broker's message
#         - Updated load offer with extracted info
#         - What's still missing (if anything)
#         - Whether you can now proceed to negotiation
#         """
#         try:
#             self.logger.info("Processing broker response")
#             curr_offer = request["curr_offer"]
#             if not curr_offer:
#                 raise ValueError("No current load offer found.")
#             # Extract information from broker's message
#             extraction_result = self.info_seeker.extract_info_from_response(
#                 request["broker_message"],
#                 curr_offer,
#                 None  # Will use current date
#             )

#             # Check if extraction was successful
#             if not extraction_result.get('extraction_successful', False):
#                 self.logger.warning(
#                     "No information extracted from broker message")
#                 return ProcessBrokerResponseResponse(
#                     extraction_successful=False,
#                     fields_extracted=[],
#                     extracted_values={},
#                     updated_load_offer=curr_offer,
#                     still_missing_critical_fields=[],
#                     can_proceed_to_negotiation=False,
#                     message="Thank you for your interest. To proceed, could you please provide more details about the load, such as pickup/delivery locations, dates, and weight?"
#                 )

#             # Merge extracted info into current load offer
#             updated_load = self.info_seeker.merge_extracted_info(
#                 curr_offer,
#                 extraction_result.get('extracted_fields', {})
#             )

#             # Check what's still missing
#             still_missing_info = self.info_seeker.identify_missing_fields(
#                 updated_load, "offer")
#             still_missing_critical = still_missing_info.get(
#                 'missing_critical_fields', [])
#             can_proceed = still_missing_info.get(
#                 'has_sufficient_info_for_negotiation', False)

#             # Create response message
#             if can_proceed:
#                 message = "Great! All critical information received. You can now proceed to negotiation."
#             else:
#                 from services.load_info_seeker import FIELD_DESCRIPTIONS
#                 missing_readable = [FIELD_DESCRIPTIONS.get(
#                     f, f) for f in still_missing_critical]
#                 message = f"Thanks for the information. I just need a few more details: {', '.join(missing_readable)}. Could you please provide them?"

#             await load_board.update_load_offer(curr_offer["id"], updated_load)
#             self.logger.info(f"Load offer updated")
#             return ProcessBrokerResponseResponse(
#                 extraction_successful=True,
#                 fields_extracted=extraction_result.get('fields_found', []),
#                 extracted_values=extraction_result.get('extracted_fields', {}),
#                 updated_load_offer=updated_load,
#                 still_missing_critical_fields=still_missing_critical,
#                 can_proceed_to_negotiation=can_proceed,
#                 message=message
#             )

#         except Exception as e:
#             self.logger.exception(
#                 "Apologies, we're having trouble processing the broker's response. Please try again later.")
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Apologies, we're having trouble processing the broker's response. Please try again later."
#             )

#     async def find_bid(self, arg0: dict):
#         try:
#             tenant_id = arg0.get("tenant_id")
#             carrier_id = arg0.get("carrier_id")
#             from_email = arg0.get("from")
#             thread_id = arg0.get("thread_Id")

#             # Validate inputs
#             if not tenant_id or not carrier_id or not from_email or not thread_id:
#                 raise ValueError(
#                     f"It seems some required fields are missing in the provided data: {arg0}. Could you please double-check?")

#             # Fetch bids
#             bid_list = await supertruck.get_bid(
#                 tenant_id=tenant_id,
#                 params={
#                     # "brokerContactEmail": "from_email",
#                     "carrierId": carrier_id,
#                     "threadId": thread_id,
#                 }
#             )

#             if not bid_list or len(bid_list) == 0:
#                 raise LookupError(
#                     "I couldn't find a bid for the given load offer and carrier. It might have been updated or is no longer available.")

#             return bid_list[0]

#         except Exception as e:
#             # You can either raise or return a structured error
#             print(f"Error in find_bid: {e}")
#             raise

#     async def execute_negotiation(self, data: dict):
#         """
#         FIND INTENT FROM EMAIL MESSAGE AND NEGOTIATE
#         """
#         try:
#             self.logger.info(f"Negotiation Action Started")
#             tenant_id = data["tenant_id"]
#             split_text = re.split(
#                 r"On .* wrote:", data["body"], maxsplit=1)
#             latest = split_text[0].strip()
#             # Find carrier
#             carrier = await supertruck.find_carrier_by_email(
#                 tenant_id=tenant_id,
#                 email="400dyforever@gmail.com"
#             )
#             if not carrier:
#                 raise LookupError(f"No carrier found for email:")

#             params = {
#                 "tenant_id": tenant_id,
#                 "carrier_id": "41a0eae5-98d1-4db8-ad0c-6833393a2ad4",
#                 # carrier["id"],
#                 "from": data["from"],
#                 "thread_Id": data["thread_Id"]
#             }

#             currBid = await self.find_bid(arg0=params)
#             if not currBid:
#                 raise ValueError("No current bid found for negotiation.")

#             curr_offer = await load_board.get_load_offer_by_id(currBid["entityId"])
#             if not curr_offer:
#                 raise ValueError("No current load offer found.")

#             await supertruck.negotiate(tenant_id=tenant_id, data={
#                 "rate": currBid["baseRate"],
#                 "negotiationDirection": "incoming",
#                 "bidId": currBid["id"],
#                 "negotiationRawEmail": latest,
#                 "messageId": data["messageId"],
#                 "inReplyTo": data["inReplyTo"],
#                 "references": data["references"]
#             })
#             self.logger.info(f"Broker message received")

#             # * AI TO FIND INTENT
#             extracted_info = email_intent.classify_load_bid_email_intent(
#                 input_email=data["body"],
#                 conversation_history=currBid.get("negotiations")
#             )
#             self.logger.info(f"Info Extraction Successful")

#             # intent = "negotiation"
#             # intent = extracted_info.intent
#             # if not intent:
#             #     raise ValueError(
#             #         "Failed to extract intent from email.")
#             # intent = "bid_acceptance"
#             intent = extracted_info.intent
#             if not intent:
#                 raise ValueError(
#                     "I couldn't quite understand the intent from the email. Could you please clarify if you're accepting, rejecting, or proposing a new rate?")

#             bidding_type = currBid.get("biddingType")

#             parsed_email_body = EmailReplyParser.parse_reply(
#                 data["body"])

#             # * START NEGOTIATION OR TAKE NEXT ACTION BASED ON EXTRACTED INFO INTENT
#             if (intent == Load_Bid_Email_Intent.NEGOTIATION.value):
#                 if not bidding_type:
#                     raise ValueError(
#                         "Bid missing 'biddingType' field.")

#                 # *FIND BID TYPE AND RUN ACTION
#                 if (bidding_type == BIDDING_TYPE.MANUAL.value):
#                     self.logger.info('👀 ~ :This is Manual bid request###')
#                     return
#                 if (bidding_type == BIDDING_TYPE.AUTO.value):
#                     # ! IMPORTANT AREA FOR LOAD OFFER RESPONSE AND ALL
#                     if (curr_offer["loadConfidentialScore"] < 80):
#                         info_check = curr_offer
#                         info_check = await self.process_broker_response({"broker_message": latest,  "tenant_id": tenant_id, "curr_offer": curr_offer})
#                         self.logger.info("Info Checked Successfully")
#                         # self.logger.info(
#                         #     info_check.still_missing_critical_fields)
#                         if len(info_check.still_missing_critical_fields) <= 1:
#                             # len(info_check.still_missing_critical_fields) == 0:
#                             self.logger.info(
#                                 "All information requirement is fulfilled. Proceed to negotiation.")
#                             total_distance = await distance.calculate_distance(
#                                 info_check["pickupLocation"],
#                                 info_check["deliveryLocation"]
#                             )
#                             self.logger.info(
#                                 "Distance calculated")
#                             rate_calc = await calculator.calculate_bid(
#                                 float(total_distance["distance"])
#                             )
#                             self.logger.info(
#                                 "Min & Max rate calculated")

#                             # *AI NEGOTIATION
#                             ai_res = negotiation.offer_negotiation(
#                                 parsed_email_body,
#                                 rate_calc.min_rate,
#                                 rate_calc.max_rate,
#                                 currBid.get("negotiations"),
#                                 curr_offer  # if required
#                             )
#                             await supertruck.update_bid(
#                                 tenant_id=tenant_id,
#                                 bid_id=currBid["id"],
#                                 data={"minRate": rate_calc.min_rate,
#                                       "maxRate": rate_calc.max_rate, }
#                             )

#                             self.logger.info(f"AI response")
#                             await supertruck.negotiate(tenant_id=tenant_id, data={
#                                 "rate": currBid["baseRate"],
#                                 "negotiationDirection": "outgoing",
#                                 "bidId": currBid["id"],
#                                 "negotiationRawEmail": ai_res["response"]
#                             })
#                             self.logger.info(f"Negotiation Successful")
#                         else:
#                             await supertruck.negotiate(tenant_id=tenant_id, data={
#                                 "rate": None,
#                                 "negotiationDirection": "outgoing",
#                                 "bidId": currBid["id"],
#                                 "negotiationRawEmail": info_check.message
#                             })
#                             self.logger.info(f"Continue asking load info")
#                     else:
#                         # *AI NEGOTIATION
#                         ai_res = negotiation.offer_negotiation(
#                             parsed_email_body,
#                             float(currBid["minRate"]),
#                             float(currBid["maxRate"]),
#                             currBid.get("negotiations"),
#                             currBid  # if required
#                         )
#                         self.logger.info(f"AI response")
#                         await supertruck.negotiate(tenant_id=tenant_id, data={
#                             "rate": currBid["baseRate"],
#                             "negotiationDirection": "outgoing",
#                             "bidId": currBid["id"],
#                             "negotiationRawEmail": ai_res["response"]
#                         })
#                         self.logger.info(f"Negotiation Successful")

#             elif intent == Load_Bid_Email_Intent.INFORMATION_SEEKING.value:
#                 self.logger.info('👀 ~ : Seeking Information###')
#                 self.logger.info(bidding_type)

#                 # *FIND BID TYPE AND RUN ACTION
#                 if (bidding_type == BIDDING_TYPE.MANUAL.value):
#                     self.logger.info('👀 ~ :This is Manual bid request###')
#                     return
#                 if (bidding_type == BIDDING_TYPE.AUTO.value):
#                     info_seeker = InformationSeeker(data=carrier)
#                     info_res = info_seeker.ask(
#                         question=parsed_email_body)
#                     final_res = await supertruck.negotiate(tenant_id=tenant_id, data={
#                         "rate": currBid["baseRate"],
#                         "negotiationDirection": "outgoing",
#                         "bidId": currBid["id"],
#                         "negotiationRawEmail": info_res
#                     })
#                     self.logger.info(f"Final response: {final_res}")

#             elif intent == Load_Bid_Email_Intent.BID_ACCEPTANCE.value:
#                 self.logger.info(Load_Bid_Email_Intent)
#                 final_res = await supertruck.update_bid(tenant_id=tenant_id, bid_id=currBid["id"], data={"isAcceptedByBroker": True, "status": "accepted"})
#                 self.logger.info(f"👀 ~ Bid accept response: {final_res}")

#             elif intent == Load_Bid_Email_Intent.BID_REJECTION.value:
#                 final_res = await supertruck.update_bid(tenant_id=tenant_id, bid_id=currBid["id"], data={
#                     "isAcceptedByBroker": False
#                 })
#                 self.logger.info(
#                     f"👀 ~ Bid rejection response: {final_res}")

#             elif intent == Load_Bid_Email_Intent.RATE_CONFIRMATION.value:
#                 self.logger.info('👀 ~ : Rate confirmation Information###')

#             elif intent == Load_Bid_Email_Intent.BROKER_SETUP.value:
#                 self.logger.info('👀 ~ : Broker setup###')

#             else:
#                 if extracted_info.details and extracted_info.details.questions and len(extracted_info.details.questions) > 0:
#                     msg = extracted_info.details.questions[0]
#                 else:
#                     msg = "Could you please clarify your request?"
#                 await supertruck.negotiate(tenant_id=tenant_id, data={
#                         "rate": currBid["baseRate"],
#                         "negotiationDirection": "outgoing",
#                         "bidId": currBid["id"],
#                         "negotiationRawEmail": msg
#                     })
#                 self.logger.info(f"Final response: {msg}")
#                 # raise ValueError(f"Unknown intent: {intent}")
#         except Exception as e:
#             self.logger.info(f"❌ Error in execute_negotiation: {str(e)}")
#             raise



import logging
import json
import re
from email_reply_parser import EmailReplyParser
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException

from integration.supertruck import SuperTruck
from integration.load_board import LoadBoard
from integration.geo_code import GeocodingService

from services.email_intent_classifier import IntentClassifierService
from services.negotiation import NegotiationService
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

negotiation = NegotiationService()
email_intent = IntentClassifierService()
loadInfo = LoadInfoSeekerOpenAI()
calculator = FreightBidCalculator()


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
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.info_seeker = LoadInfoSeekerOpenAI()

    async def process_broker_response(self, request: ProcessBrokerResponseRequest) -> ProcessBrokerResponseResponse:
        """Process broker's response and update load offer"""
        try:
            self.logger.info("="*80)
            self.logger.info("🔄 PROCESSING BROKER RESPONSE")
            self.logger.info(f"📧 Broker message: {request['broker_message'][:100]}...")
            
            curr_offer = request["curr_offer"]
            if not curr_offer:
                raise ValueError("No current load offer found.")
            
            self.logger.info(f"📦 Current offer ID: {curr_offer.get('id')}")
            self.logger.info(f"📊 Current load score: {curr_offer.get('loadConfidentialScore', 'N/A')}")

            # Extract information from broker's message
            extraction_result = self.info_seeker.extract_info_from_response(
                request["broker_message"],
                curr_offer,
                None
            )
            
            self.logger.info(f"✅ Extraction result: {extraction_result.get('extraction_successful', False)}")

            if not extraction_result.get('extraction_successful', False):
                self.logger.warning("⚠️ No information extracted from broker message")
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
            
            self.logger.info(f"📝 Updated fields: {list(extraction_result.get('extracted_fields', {}).keys())}")

            # Check what's still missing
            still_missing_info = self.info_seeker.identify_missing_fields(
                updated_load, "offer")
            still_missing_critical = still_missing_info.get('missing_critical_fields', [])
            can_proceed = still_missing_info.get('has_sufficient_info_for_negotiation', False)
            
            self.logger.info(f"❓ Still missing: {still_missing_critical}")
            self.logger.info(f"✅ Can proceed to negotiation: {can_proceed}")

            # Create response message
            if can_proceed:
                message = "Great! All critical information received. You can now proceed to negotiation."
            else:
                from services.load_info_seeker import FIELD_DESCRIPTIONS
                missing_readable = [FIELD_DESCRIPTIONS.get(f, f) for f in still_missing_critical]
                message = f"Thanks for the information. I just need a few more details: {', '.join(missing_readable)}. Could you please provide them?"

            await load_board.update_load_offer(curr_offer["id"], updated_load)
            self.logger.info(f"💾 Load offer updated successfully")
            self.logger.info("="*80)
            
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
            self.logger.exception("❌ Error processing broker response")
            raise HTTPException(
                status_code=500,
                detail=f"Apologies, we're having trouble processing the broker's response. Please try again later."
            )

    async def find_bid(self, arg0: dict):
        try:
            self.logger.info("🔍 Searching for bid...")
            tenant_id = arg0.get("tenant_id")
            carrier_id = arg0.get("carrier_id")
            from_email = arg0.get("from")
            thread_id = arg0.get("thread_Id")

            self.logger.info(f"  Tenant ID: {tenant_id}")
            self.logger.info(f"  Carrier ID: {carrier_id}")
            self.logger.info(f"  Thread ID: {thread_id}")

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

            self.logger.info(f"✅ Found bid: {bid_list[0].get('id')}")
            return bid_list[0]

        except Exception as e:
            self.logger.error(f"❌ Error finding bid: {e}")
            raise

    async def execute_negotiation(self, data: dict):
        """Main negotiation execution with comprehensive logging"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("🚀 NEGOTIATION ACTION STARTED")
            self.logger.info("="*80)
            
            tenant_id = data["tenant_id"]
            thread_id = data.get("thread_Id") or data.get("threadId")
            message_id = data.get("messageId")
            
            self.logger.info(f"📋 Tenant ID: {tenant_id}")
            self.logger.info(f"💬 Thread ID: {thread_id}")
            self.logger.info(f"📨 Message ID: {message_id}")
            self.logger.info(f"👤 From: {data.get('from')}")

            # Extract latest message
            split_text = re.split(r"On .* wrote:", data["body"], maxsplit=1)
            latest = split_text[0].strip()
            
            self.logger.info("="*80)
            self.logger.info("📧 LATEST MESSAGE CONTENT")
            self.logger.info(f"{latest[:200]}...")
            self.logger.info("="*80)

            # Find carrier
            self.logger.info("🔍 Finding carrier...")
            carrier = await supertruck.find_carrier_by_email(
                tenant_id=tenant_id,
                email="400dyforever@gmail.com"
            )
            if not carrier:
                raise LookupError("No carrier found for email")
            
            self.logger.info(f"✅ Carrier found: {carrier.get('id')}")

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
            
            self.logger.info(f"✅ Current bid found: {currBid.get('id')}")
            self.logger.info(f"📊 Bid type: {currBid.get('biddingType')}")
            self.logger.info(f"💰 Base rate: ${currBid.get('baseRate')}")

            # Get load offer
            curr_offer = await load_board.get_load_offer_by_id(currBid["entityId"])
            if not curr_offer:
                raise ValueError("No current load offer found.")
            
            self.logger.info(f"✅ Load offer found: {curr_offer.get('id')}")
            self.logger.info(f"📍 Pickup: {curr_offer.get('pickupLocation')}")
            self.logger.info(f"📍 Delivery: {curr_offer.get('deliveryLocation')}")
            self.logger.info(f"📊 Confidence score: {curr_offer.get('loadConfidentialScore')}")

            # Save incoming negotiation
            await supertruck.negotiate(tenant_id=tenant_id, data={
                "rate": currBid["baseRate"],
                "negotiationDirection": "incoming",
                "bidId": currBid["id"],
                "negotiationRawEmail": latest,
                "messageId": message_id,
                "inReplyTo": data.get("inReplyTo"),
                "references": data.get("references")
            })
            self.logger.info("💾 Incoming negotiation saved")

            # Classify intent with conversation history
            self.logger.info("\n" + "="*80)
            self.logger.info("🤖 CLASSIFYING EMAIL INTENT")
            self.logger.info("="*80)
            
            conversation_history = currBid.get("negotiations", [])
            self.logger.info(f"📜 Conversation history: {len(conversation_history)} messages")
            
            # Log last few messages for context
            if conversation_history:
                self.logger.info("📋 Recent conversation:")
                for idx, msg in enumerate(conversation_history[-8:]):
                    direction = msg.get("negotiationDirection", "unknown")
                    content = msg.get("negotiationRawEmail", "")[:80]
                    self.logger.info(f"  {idx+1}. [{direction}]: {content}...")

            extracted_info = email_intent.classify_load_bid_email_intent(
                input_email=latest,  # Use latest, not full body
                conversation_history=conversation_history
            )
            
            intent = extracted_info.intent
            confidence = extracted_info.confidence
            
            self.logger.info("="*80)
            self.logger.info(f"🎯 INTENT CLASSIFIED: {intent}")
            self.logger.info(f"📊 Confidence: {confidence}")
            self.logger.info(f"📋 Details: {extracted_info.details}")
            self.logger.info("="*80)

            if not intent:
                raise ValueError("Failed to extract intent from email")

            bidding_type = currBid.get("biddingType")

            # Handle different intents
            if intent == Load_Bid_Email_Intent.NEGOTIATION.value:
                self.logger.info("\n" + "="*80)
                self.logger.info("💼 HANDLING NEGOTIATION INTENT")
                self.logger.info("="*80)
                
                if not bidding_type:
                    raise ValueError("Bid missing 'biddingType' field")

                if bidding_type == BIDDING_TYPE.MANUAL.value:
                    self.logger.info('👀 Manual bid - skipping auto-negotiation')
                    return
                    
                if bidding_type == BIDDING_TYPE.AUTO.value:
                    self.logger.info('🤖 Auto bid - processing...')
                    
                    # Check if we need more info
                    if curr_offer["loadConfidentialScore"] < 80:
                        self.logger.info(f"📊 Low confidence score ({curr_offer['loadConfidentialScore']}) - checking for missing info")
                        
                        info_check = await self.process_broker_response({
                            "broker_message": latest,
                            "tenant_id": tenant_id,
                            "curr_offer": curr_offer
                        })
                        
                        self.logger.info(f"✅ Info check complete")
                        self.logger.info(f"   Missing fields: {info_check.still_missing_critical_fields}")
                        self.logger.info(f"   Can proceed: {info_check.can_proceed_to_negotiation}")
                        
                        if len(info_check.still_missing_critical_fields) < 1:
                            self.logger.info("✅ Sufficient info - proceeding to rate calculation")
                            
                            # Calculate distance and rates
                            total_distance = await distance.calculate_distance(
                                info_check.updated_load_offer["pickupLocation"],
                                info_check.updated_load_offer["deliveryLocation"]
                            )
                            self.logger.info(f"📏 Distance calculated: {total_distance.get('distance')} miles")
                            
                            rate_calc = await calculator.calculate_bid(
                                float(total_distance["distance"])
                            )
                            self.logger.info(f"💰 Rate range: ${rate_calc.min_rate} - ${rate_calc.max_rate}")

                            # AI negotiation
                            parsed_email_body = EmailReplyParser.parse_reply(data["body"])
                            ai_res = negotiation.offer_negotiation(
                                parsed_email_body,
                                rate_calc.min_rate,
                                rate_calc.max_rate,
                                currBid.get("negotiations"),
                                curr_offer
                            )
                            
                            self.logger.info(f"🤖 AI response generated: {ai_res['response'][:100]}...")
                            
                            await supertruck.update_bid(
                                tenant_id=tenant_id,
                                bid_id=currBid["id"],
                                data={"minRate": rate_calc.min_rate, "maxRate": rate_calc.max_rate, "baseRate":curr_offer["requestedRate"]}
                            )
                            
                            await supertruck.negotiate(tenant_id=tenant_id, data={
                                "rate": ai_res["proposed_price"],
                                "negotiationDirection": "outgoing",
                                "bidId": currBid["id"],
                                "negotiationRawEmail": ai_res["response"]
                            })
                            self.logger.info(f"✅ Negotiation response sent")
                        else:
                            self.logger.info("❓ Still need more info - asking broker")
                            await supertruck.negotiate(tenant_id=tenant_id, data={
                                "rate": None,
                                "negotiationDirection": "outgoing",
                                "bidId": currBid["id"],
                                "negotiationRawEmail": info_check.message
                            })
                            self.logger.info(f"✅ Info request sent: {info_check.message}")
                    else:
                        self.logger.info(f"✅ High confidence score ({curr_offer['loadConfidentialScore']}) - proceeding directly to negotiation")
                        
                        parsed_email_body = EmailReplyParser.parse_reply(data["body"])
                        ai_res = negotiation.offer_negotiation(
                            parsed_email_body,
                            float(currBid["minRate"]),
                            float(currBid["maxRate"]),
                            currBid.get("negotiations"),
                            curr_offer
                        )
                        
                        self.logger.info(f"🤖 AI response generated")
                        
                        await supertruck.negotiate(tenant_id=tenant_id, data={
                            "rate": ai_res["proposed_price"],
                            "negotiationDirection": "outgoing",
                            "bidId": currBid["id"],
                            "negotiationRawEmail": ai_res["response"]
                        })
                        self.logger.info(f"✅ Negotiation response sent")

            elif intent == Load_Bid_Email_Intent.INFORMATION_SEEKING.value:
                self.logger.info("\n" + "="*80)
                self.logger.info("❓ HANDLING INFORMATION_SEEKING INTENT")
                self.logger.info("="*80)
                
                if bidding_type == BIDDING_TYPE.MANUAL.value:
                    self.logger.info('👀 Manual bid - skipping')
                    return
                    
                if bidding_type == BIDDING_TYPE.AUTO.value:
                    parsed_email_body = EmailReplyParser.parse_reply(data["body"])
                    info_seeker = InformationSeeker(data=carrier)
                    info_res = info_seeker.ask(question=parsed_email_body)
                    
                    self.logger.info(f"📧 Info response: {info_res[:100]}...")
                    
                    final_res = await supertruck.negotiate(tenant_id=tenant_id, data={
                        # "rate": currBid["baseRate"],
                        "negotiationDirection": "outgoing",
                        "bidId": currBid["id"],
                        "negotiationRawEmail": info_res
                    })
                    self.logger.info(f"✅ Information response sent")

            elif intent == Load_Bid_Email_Intent.BID_ACCEPTANCE.value:
                self.logger.info("\n" + "="*80)
                self.logger.info("✅ HANDLING BID_ACCEPTANCE")
                self.logger.info("="*80)
                
                final_res = await supertruck.update_bid(
                    tenant_id=tenant_id,
                    bid_id=currBid["id"],
                    data={"isAcceptedByBroker": True, "status": "accepted"}
                )
                self.logger.info(f"✅ Bid accepted successfully")

            elif intent == Load_Bid_Email_Intent.BID_REJECTION.value:
                self.logger.info("\n" + "="*80)
                self.logger.info("❌ HANDLING BID_REJECTION")
                self.logger.info("="*80)
                
                final_res = await supertruck.update_bid(
                    tenant_id=tenant_id,
                    bid_id=currBid["id"],
                    data={"isAcceptedByBroker": False}
                )
                self.logger.info(f"✅ Bid rejection recorded")

            elif intent == Load_Bid_Email_Intent.RATE_CONFIRMATION.value:
                self.logger.info("📄 Rate confirmation detected")

            elif intent == Load_Bid_Email_Intent.BROKER_SETUP.value:
                self.logger.info("🔧 Broker setup detected")

            else:
                self.logger.warning(f"⚠️ Unclear intent - sending clarification")
                if extracted_info.details and extracted_info.details.questions and len(extracted_info.details.questions) > 0:
                    msg = extracted_info.details.questions[0]
                else:
                    msg = "Could you please clarify your request?"
                    
                await supertruck.negotiate(tenant_id=tenant_id, data={
                    # "rate": currBid["baseRate"],
                    "negotiationDirection": "outgoing",
                    "bidId": currBid["id"],
                    "negotiationRawEmail": msg
                })
                self.logger.info(f"✅ Clarification sent: {msg}")

            self.logger.info("\n" + "="*80)
            self.logger.info("🏁 NEGOTIATION ACTION COMPLETED")
            self.logger.info("="*80 + "\n")

        except Exception as e:
            self.logger.error("\n" + "="*80)
            self.logger.error(f"❌ ERROR IN EXECUTE_NEGOTIATION")
            self.logger.error(f"Error: {str(e)}")
            self.logger.exception("Full traceback:")
            self.logger.error("="*80 + "\n")
            raise