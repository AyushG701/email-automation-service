from fastapi import FastAPI, Body, HTTPException
import httpx
import logging

from config import AppConfig
from core.auth_client import AuthApiClient
from integration.broker import Broker
from services.load_score import LoadScoreService


config = AppConfig()
broker = Broker()
score = LoadScoreService()
authClient = AuthApiClient(
    config.LOAD_SERVICE_URL,
    config.APP_KEY
)


class LoadBoard:
    """ SUPER TRUCK LOAD BOARD INTEGRATION """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def clean_dict(self, d: dict):
        return {k: v for k, v in d.items() if v is not None}

    async def create_load_offer(self, tenant_id: str, data: dict):
        curr_broker = await broker.find_broker(tenant_id=tenant_id, data={
            "companyEmail": data["senderEmail"],
            "contactEmail": data["senderEmail"]
        })
        json_data = {
            "brokerContactEmail": data.get("senderEmail"),
            "deliveryDate": data.get("dropoffTimeExact"),
            "pickupLocation": data.get("pickupLocation"),
            "deliveryLocation": data.get("dropoffLocation"),
            "equipmentType": data.get("equipmentType"),
            "pickupDate": data.get("pickupTimeExact"),
            "weightLbs": data.get("weight"),
            "offerEmail": data.get("emailContentFromBroker"),
            "additionalInstructions": f"{data.get('pickupInstructions')} + {data.get('dropoffInstructions')}",
            "requestedRate": data.get("rate"),
            "brokerCompany": data.get("brokerCompany"),
            "brokerId": curr_broker.get("id") if curr_broker else None,
        }
        total_score = score.load_score(json_data)

        response = authClient.post(
            tenant_id=tenant_id, endpoint='load-offer', json_data={
                "brokerContactEmail": data.get("senderEmail"),
                "deliveryDate": data.get("dropoffTimeExact"),
                "pickupLocation": data.get("pickupLocation"),
                "deliveryLocation": data.get("dropoffLocation"),
                "equipmentType": data.get("equipmentType"),
                "pickupDate": data.get("pickupTimeExact"),
                "weightLbs": data.get("weight"),
                "offerEmail": data.get("emailContentFromBroker"),
                "additionalInstructions": f"{data.get('pickupInstructions')} + {data.get('dropoffInstructions')}",
                "requestedRate": data.get("rate"),
                "brokerCompany": data.get("brokerCompany"),
                "brokerId": curr_broker.get("id") if curr_broker else None,
                "loadConfidentialScore": total_score,
                "loadConfidentialityStatus": "approved" if total_score > 80 else "draft"
            })

        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    async def create_load(self, tenant_id: str, offer: dict):
        response = authClient.post(
            tenant_id=tenant_id, endpoint='load', json_data=offer)
        # response.raise_for_status()
        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    async def get_load_offer_by_id(self, id: str):
        response = authClient.get(
            endpoint=f'load-offer/{id}')
        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    async def update_load_offer(self, id: str, offer: dict):
        raw_payload = {
            "brokerContactEmail": offer.get("brokerContactEmail") or None,
            "deliveryDate": offer.get("deliveryDate") or None,
            "pickupLocation": offer.get("pickupLocation") or None,
            "deliveryLocation": offer.get("dropoffLocation") or None,
            "equipmentType": offer.get("equipmentType") or None,
            "pickupDate": offer.get("pickupDate") or None,
            "weightLbs": int(offer["weightLbs"]) if offer.get("weightLbs") not in [None, ""] and str(offer.get("weightLbs")).isdigit() else None,
            "requestedRate": int(offer["requestedRate"]) if offer.get("requestedRate") not in [None, ""] and str(offer.get("requestedRate")).isdigit() else None,

            "additionalInstructions": offer.get("additionalInstructions") or None,
        }
        safe_payload = self.clean_dict(raw_payload)

        # Compute score
        total_score = score.load_score(safe_payload)
        patch_payload = {
            **safe_payload,
            "loadConfidentialScore": total_score,
            "loadConfidentialityStatus": "approved" if total_score > 80 else "draft",
        }
        response = authClient.patch(
            endpoint=f"load-offer/{id}",
            json_data=patch_payload
        )
        json_resp = response.json()

        if "data" not in json_resp:
            raise ValueError(f"Load offer update failed: {json_resp}")

        return json_resp["data"]
