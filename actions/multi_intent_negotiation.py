from services.multi_intent_orchestrator import MultiIntentOrchestrator
from integration.supertruck import SuperTruck
from integration.load_board import LoadBoard
from constant.enum import BIDDING_TYPE
from schemas.negotiation import NegotiationState, NegotiationStrategy
import logging
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

supertruck = SuperTruck()
load_board = LoadBoard()

class MultiIntentNegotiationAction:
    def __init__(self, state=None):
        self.logger = logging.getLogger(__name__)
        self.state = state
        # Don't initialize orchestrator here - we'll create it per negotiation with proper state

    async def find_bid(self, arg0: dict):
        try:
            tenant_id = arg0.get("tenant_id")
            carrier_id = arg0.get("carrier_id")
            thread_id = arg0.get("thread_Id")
            bid_list = await supertruck.get_bid(tenant_id=tenant_id, params={"carrierId": carrier_id, "threadId": thread_id})
            if not bid_list or len(bid_list) == 0:
                raise LookupError("No bid found")
            return bid_list[0]
        except Exception as e:
            self.logger.error(f"Error finding bid: {e}")
            raise

    async def execute_negotiation(self, data: dict):
        try:
            self.logger.info("=== Multi-Intent Negotiation Started ===")
            tenant_id = data["tenant_id"]
            thread_id = data.get("thread_Id") or data.get("threadId")
            message_id = data.get("messageId")

            split_text = re.split(r"On .* wrote:", data["body"], maxsplit=1)
            latest = split_text[0].strip()

            carrier = await supertruck.find_carrier_by_email(tenant_id=tenant_id, email="400dyforever@gmail.com")
            if not carrier:
                raise LookupError("No carrier found")

            currBid = await self.find_bid({"tenant_id": tenant_id, "carrier_id": carrier["id"], "thread_Id": thread_id})
            if not currBid:
                raise ValueError("No bid found")

            curr_offer = await load_board.get_load_offer_by_id(currBid["entityId"])
            conversation_history = currBid.get("negotiations", [])

            bidding_type = currBid.get("biddingType")
            if bidding_type == BIDDING_TYPE.MANUAL.value:
                await supertruck.negotiate(tenant_id=tenant_id, data={
                    "rate": currBid.get("baseRate"),
                    "negotiationDirection": "incoming",
                    "bidId": currBid["id"],
                    "negotiationRawEmail": latest,
                    "messageId": message_id,
                    "metadata": {"messageType": "MANUAL_REVIEW", "timestamp": datetime.utcnow().isoformat()}
                })
                return

            # Initialize negotiation state
            negotiation_state = NegotiationState(
                load_offer_id=currBid["entityId"],
                thread_id=thread_id,
                is_negotiation_active=True,
                negotiation_round=len([msg for msg in conversation_history if msg.get("direction") == "outgoing"]),
                initial_broker_price=currBid.get("baseRate"),
                last_broker_price=currBid.get("baseRate"),
                strategy=NegotiationStrategy(
                    min_price=currBid.get("minRate", 4000),
                    max_price=currBid.get("maxRate", 6000)
                )
            )

            # Initialize orchestrator with proper state and carrier info
            orchestrator = MultiIntentOrchestrator(state=negotiation_state, carrier_info=carrier)

            # Convert conversation_history list to string format
            chat_history_str = "\n".join([
                f"{msg.get('direction', 'unknown')}: {msg.get('message', '')}"
                for msg in conversation_history
            ]) if conversation_history else ""

            # Append the latest message
            full_chat_history = f"{chat_history_str}\n\nBroker: {latest}".strip()

            # Process the email and get updated state
            result_state = orchestrator.process_email(email_body=latest, chat_history=full_chat_history)

            # Log incoming negotiation
            await supertruck.negotiate(tenant_id=tenant_id, data={
                "rate": result_state.last_broker_price or currBid.get("baseRate"),
                "negotiationDirection": "incoming",
                "bidId": currBid["id"],
                "negotiationRawEmail": latest,
                "messageId": message_id,
                "metadata": {
                    "negotiation_round": result_state.negotiation_round,
                    "timestamp": datetime.utcnow().isoformat()
                }
            })

            # Determine negotiation status
            status = "active"
            if not result_state.is_negotiation_active:
                # Check if it was accepted or rejected based on response
                if result_state.suggested_response and "accept" in result_state.suggested_response.lower():
                    status = "accepted"
                else:
                    status = "rejected"

            if status == "accepted":
                await supertruck.update_bid(tenant_id=tenant_id, bid_id=currBid["id"], data={"isAcceptedByBroker": True, "status": "accepted"})
            elif status == "rejected":
                await supertruck.update_bid(tenant_id=tenant_id, bid_id=currBid["id"], data={"isAcceptedByBroker": False, "status": "rejected"})

            # Send outgoing response if needed
            if result_state.response_needed and result_state.suggested_response:
                await supertruck.negotiate(tenant_id=tenant_id, data={
                    "rate": result_state.last_supertruck_price or currBid.get("baseRate"),
                    "negotiationDirection": "outgoing",
                    "bidId": currBid["id"],
                    "negotiationRawEmail": result_state.suggested_response,
                    "metadata": {
                        "negotiation_round": result_state.negotiation_round,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                })

            self.logger.info("=== Multi-Intent Negotiation Completed ===")
        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
            raise
