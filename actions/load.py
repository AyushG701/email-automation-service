import logging
import json
from email_reply_parser import EmailReplyParser

from integration.supertruck import SuperTruck
from services.email_intent_classifier import IntentClassifierService
from services.negotiation import NegotiationService


supertruck = SuperTruck()
negotiation = NegotiationService()
email_intent = IntentClassifierService()


""" 
NEGOTIATION, CLASSIFY AND FIND EMAIL INTENT 
"""


class LoadAction:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def find_load_offer_bid(self, arg0: dict):
        carrier = await supertruck.find_carrier_by_email(tenant_id=arg0["tenant_id"], email=arg0["to"])
        bid = await supertruck.get_load_offer_bid(tenant_id=arg0["tenant_id"], params={
            "brokerContactEmail": arg0["from"],
            "carrierId": carrier.get("data", {}).get("id"),
            "threadId": arg0["thread_Id"]
        })
        return bid[0]

    # TODO FUNCTION TO CHECK BID IS VALID?
    # def is_bid_valid(self, arg0: dict):
    #     carrier = await supertruck.find_carrier_by_email(tenant_id=arg0["tenant_id"], email=arg0["to"])
    #     bid = await supertruck.get_load_offer_bid(tenant_id=arg0["tenant_id"], params={
    #         "brokerContactEmail": arg0["from"],
    #         "carrierId": carrier.get("data", {}).get("id"),
    #         "threadId": arg0["thread_Id"]
    #     })
    #     return bid["data"]

    async def execute_load_negotiation(self, data: dict):
        """ 
        FIND INTENT FROM EMAIL MESSAGE AND NEGOTIATE
        """

        # * AI TO FIND INTENT
        # email_intent.classify_load_bid_email_intent(input_email=data["body"])
        # print(extracted_info)
        extracted_info = {
            "intent": "bid_acceptance"
        }
        params = {
            "tenant_id": data["tenant_id"],
            "to": data["to"],
            "from": data["from"],
            "thread_Id": data["thread_Id"]
            # "thread_Id": data.get("email", {}).get("thread_Id")
        }
        currBid = await self.find_load_offer_bid(arg0=params)

        # * START NEGOTIATION OR TAKE NEXT ACTION BASED ON EXTRACTED INFO INTENT
        if (extracted_info["intent"] == "negotiation"):

            print(currBid)
            parsed_email_body = EmailReplyParser.parse_reply(data["body"])

            # *FIND BID TYPE AND RUN ACTION
            if (currBid["biddingType"] == "manual"):
                print('👀 ~ :This is Manual bid request###')
                return
            if (currBid["biddingType"] == "auto"):
                
                # *AI NEGOTIATION
                # ai_res = negotiation.offer_negotiation({
                #     "message": parsed_email_body,
                #     "min_price": float(currBid["minRate"]),
                #     "max_price": float(currBid["maxRate"])
                # })
                ai_res = {'proposed_price': 2500.0, 'response': 'Hi there, thanks for the details! The rate is a bit high for what this load requires. Given the distance and the temperature control needed, I’d need to see at least $2,500 to make this work. Let me know!', 'status': 'negotiating'}
                print(ai_res)
                final_res = await supertruck.create_load_negotiation(tenant_id=data["tenant_id"], data={
                    "rate": ai_res["proposed_price"],
                    "negotiationDirection": "outgoing",
                    "loadOfferBidId": currBid["id"],
                    "negotiationRawEmail": ai_res["response"]
                })
                print(f"Final response: {final_res}")
                return final_res

        if (extracted_info["intent"] == "information_seeking"):
            print('👀 ~ : Seeking Information###')

        if (extracted_info["intent"] == "bid_acceptance"):
            print('👀 ~ : Bid accepted###')
            final_res = await supertruck.update_load_bid(tenant_id=data["tenant_id"], bid_id=currBid["id"], data={
                "isAcceptedByBroker": True
            })
            print(f"👀 ~ Bid accept response: {final_res}")

        if (extracted_info["intent"] == "bid_rejection"):
            print('👀 ~ : Bid rejected###')
            final_res = await supertruck.update_load_bid(tenant_id=data["tenant_id"], bid_id=currBid["id"], data={
                "isAcceptedByBroker": False
            })
            print(f"👀 ~ Bid rejection response: {final_res}")

        if (extracted_info["intent"] == "rate_confirmation"):
            print('👀 ~ : Rate confirmation Information###')

        if (extracted_info["intent"] == "broker_setup"):
            print('👀 ~ : Broker setup###')
