from services.multi_intent_orchestrator import MultiIntentOrchestrator
from integration.supertruck import SuperTruck
from integration.load_board import LoadBoard
from constant.enum import BIDDING_TYPE
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
        self.orchestrator = MultiIntentOrchestrator(state)

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

            result = self.orchestrator.process_message(broker_message=latest, negotiation_history=conversation_history, load_offer=curr_offer, pricing={"min_price": currBid.get("minRate", 0), "max_price": currBid.get("maxRate", 0)})

            await supertruck.negotiate(tenant_id=tenant_id, data={
                "rate": result.get("extractedPrice"),
                "negotiationDirection": "incoming",
                "bidId": currBid["id"],
                "negotiationRawEmail": latest,
                "messageId": message_id,
                "metadata": result.get("metadata", {})
            })

            if result.get("status") == "accepted":
                await supertruck.update_bid(tenant_id=tenant_id, bid_id=currBid["id"], data={"isAcceptedByBroker": True, "status": "accepted"})
            elif result.get("status") == "rejected":
                await supertruck.update_bid(tenant_id=tenant_id, bid_id=currBid["id"], data={"isAcceptedByBroker": False, "status": "rejected"})

            if result.get("response"):
                await supertruck.negotiate(tenant_id=tenant_id, data={
                    "rate": result.get("proposedPrice"),
                    "negotiationDirection": "outgoing",
                    "bidId": currBid["id"],
                    "negotiationRawEmail": result.get("response"),
                    "metadata": result.get("metadata", {})
                })

            self.logger.info("=== Multi-Intent Negotiation Completed ===")
        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
            raise
