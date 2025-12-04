import logging
import json
from email_reply_parser import EmailReplyParser

from integration.supertruck import SuperTruck
from services.email_intent_classifier import IntentClassifierService
from services.negotiation import NegotiationService
from services.information_seek import InformationSeeker

from constant.enum import Load_Bid_Email_Intent, LOAD_NEGOTIATION_DIRECTION, BIDDING_TYPE

supertruck = SuperTruck()
negotiation = NegotiationService()
email_intent = IntentClassifierService()


"""
NEGOTIATION, CLASSIFY AND FIND EMAIL INTENT
"""


class LoadOfferAction:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def find_load_offer_bid(self, arg0: dict):
        try:
            tenant_id = arg0.get("tenant_id")
            carrier_id = arg0.get("carrier_id")
            from_email = arg0.get("from")
            thread_id = arg0.get("thread_Id")

            # Validate inputs
            if not tenant_id or not carrier_id or not from_email or not thread_id:
                raise ValueError(f"Missing required fields in arg0: {arg0}")

            # Fetch bids
            bid_list = await supertruck.get_load_offer_bid(
                tenant_id=tenant_id,
                params={
                    "brokerContactEmail": from_email,
                    "carrierId": carrier_id,
                    "threadId": thread_id,
                }
            )

            if not bid_list or len(bid_list) == 0:
                raise LookupError(
                    f"No load offer bids found for broker {from_email}, carrier {carrier.get('id')}, thread {thread_id}"
                )

            return bid_list[0]

        except Exception as e:
            # You can either raise or return a structured error
            print(f"Error in find_load_offer_bid: {e}")
            raise

    async def execute_load_offer_negotiation(self, data: dict):
        """
        FIND INTENT FROM EMAIL MESSAGE AND NEGOTIATE
        """
        try:
            tenant_id = data["tenant_id"]
            # Find carrier
            carrier = await supertruck.find_carrier_by_email(
                tenant_id=tenant_id,
                email=data["to"]
            )
            if not carrier:
                raise LookupError(f"No carrier found for email: {to_email}")

            params = {
                "tenant_id": tenant_id,
                "carrier_id": carrier["id"],
                "from": data["from"],
                "thread_Id": data["thread_Id"]
            }

            currBid = await self.find_load_offer_bid(arg0=params)
            if not currBid:
                raise ValueError("No current bid found for negotiation.")

            # * AI TO FIND INTENT
            # email_intent.classify_load_bid_email_intent(input_email=data["body"])
            # print(extracted_info)
            extracted_info = {
                "intent": "information_seeking"
            }
            intent = extracted_info.get("intent")
            if not intent:
                raise ValueError("Failed to extract intent from email.")
            bidding_type = currBid.get("biddingType")

            print(currBid)
            parsed_email_body = EmailReplyParser.parse_reply(data["body"])

            # * START NEGOTIATION OR TAKE NEXT ACTION BASED ON EXTRACTED INFO INTENT
            if intent == Load_Bid_Email_Intent.NEGOTIATION:
                if not bidding_type:
                    raise ValueError("Bid missing 'biddingType' field.")

                # *FIND BID TYPE AND RUN ACTION
                if (bidding_type == BIDDING_TYPE.MANUAL.value):
                    print('👀 ~ :This is Manual bid request###')
                    return
                if (bidding_type == BIDDING_TYPE.AUTO.value):
                    # *AI NEGOTIATION
                    ai_res = negotiation.offer_negotiation({
                        "message": parsed_email_body,
                        "min_price": float(currBid["minRate"]),
                        "max_price": float(currBid["maxRate"])
                    })
                    # ai_res = {'proposed_price': 2500.0, 'response': 'Hi there, thanks for the details! The rate is a bit high for what this load requires. Given the distance and the temperature control needed, I’d need to see at least $2,500 to make this work. Let me know!', 'status': 'negotiating'}
                    print(ai_res)
                    final_res = await supertruck.create_load_offer_negotiation(tenant_id=tenant_id, data={
                        "rate": ai_res["proposed_price"],
                        "negotiationDirection": "outgoing",
                        "loadOfferBidId": currBid["id"],
                        "negotiationRawEmail": ai_res["response"]
                    })
                    print(f"Final response: {final_res}")

            elif intent == Load_Bid_Email_Intent.INFORMATION_SEEKING.value:
                print('👀 ~ : Seeking Information###')
                print(bidding_type)

                # *FIND BID TYPE AND RUN ACTION
                if (bidding_type == BIDDING_TYPE.MANUAL.value):
                    print('👀 ~ :This is Manual bid request###')
                    return
                if (bidding_type == BIDDING_TYPE.AUTO.value):
                    info_seeker = InformationSeeker(data=carrier)
                    info_res = info_seeker.ask(
                        question=parsed_email_body)
                    print(info_res)
                    final_res = await supertruck.create_load_offer_negotiation(tenant_id=tenant_id, data={
                        "rate": None,
                        "negotiationDirection": "outgoing",
                        "loadOfferBidId": currBid["id"],
                        "negotiationRawEmail": info_res
                    })
                    print(f"Final response: {final_res}")

            elif intent == Load_Bid_Email_Intent.BID_ACCEPTANCE.value:
                final_res = await supertruck.update_load_offer_bid(tenant_id=tenant_id, bid_id=currBid["id"], data={
                    "isAcceptedByBroker": True
                })
                print(f"👀 ~ Bid accept response: {final_res}")

            elif intent == Load_Bid_Email_Intent.BID_REJECTION.value:
                final_res = await supertruck.update_load_offer_bid(tenant_id=tenant_id, bid_id=currBid["id"], data={
                    "isAcceptedByBroker": False
                })
                print(f"👀 ~ Bid rejection response: {final_res}")

            elif intent == Load_Bid_Email_Intent.RATE_CONFIRMATION.value:
                print('👀 ~ : Rate confirmation Information###')

            elif intent == Load_Bid_Email_Intent.BROKER_SETUP.value:
                print('👀 ~ : Broker setup###')

            else:
                raise ValueError(f"Unknown intent: {intent}")

        except Exception as e:
            print(f"❌ Error in execute_load_offer_negotiation: {str(e)}")
            raise
