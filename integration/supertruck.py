from fastapi import FastAPI, Body, HTTPException
import httpx
import logging

from config import AppConfig
from core.auth_client import AuthApiClient

config = AppConfig()
authClient = AuthApiClient(
    config.SUPERTRUCK_SERVICE_URL,
    config.APP_KEY
)


class SuperTruck:
    """ SUPER TRUCK INTEGRATION """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def find_carrier_by_email(self, tenant_id: str, email: str):
        response = authClient.get(
            tenant_id=tenant_id, endpoint=f'carrier/get-by-email/{email}')
        json_resp = response.json()
        print(json_resp)
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    async def get_carrier_by_tenant(self, tenant_id: str, email: str):
        response = authClient.get(
            tenant_id=tenant_id, endpoint=f'carrier/get-by-email/{email}')
        json_resp = response.json()
        print(json_resp)
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    async def find_tenant_by_carrier_email(self, tenant_id: str, carrier_id: str):
        response = authClient.get(
            tenant_id=tenant_id, endpoint=f'carrier/{carrier_id}')
        json_resp = response.json()
        print(json_resp)
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    async def find_email_log(self, tenant_id: str, thread_id: str):
        response = authClient.get(
            tenant_id=tenant_id, endpoint=f'email-log/thread/{thread_id}')
        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    # NEW NEGOTIATION + BID
    async def negotiate(self, tenant_id: str, data: dict):
        response = authClient.post(
            tenant_id=tenant_id, endpoint=f'negotiations', json_data=data)
        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    async def update_bid(self, tenant_id: str, bid_id: str, data: dict):
        print(data)
        response = authClient.patch(
            tenant_id=tenant_id, endpoint=f'bids/{bid_id}', json_data=data)
        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    async def get_bid(self, tenant_id: str, params: dict):
        response = authClient.get(
            tenant_id=tenant_id, endpoint=f'bids', params=params)
        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    async def get_bid_by_id(self, tenant_id: str, bid_id: str):
        response = authClient.get(
            tenant_id=tenant_id, endpoint=f'bids/{bid_id}')
        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    async def get_bid_by_thread(self, tenant_id: str, thread_id: str):
        response = authClient.get(
            tenant_id=tenant_id, endpoint=f'bids/{thread_id}')
        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

  #############################################################
  # PREV LOAD AND LOAD OFFER BIDDING/NEGOTIATION
  # LOAD BID

    async def create_load_negotiation(self, tenant_id: str, data: dict):
        response = authClient.post(
            tenant_id=tenant_id, endpoint=f'load-negotiation', json_data=data)
        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    async def update_load_bid(self, tenant_id: str, bid_id: str, data: dict):
        response = authClient.patch(
            tenant_id=tenant_id, endpoint=f'load-bid/{bid_id}', json_data=data)
        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    async def get_load_bid(self, tenant_id: str, params: dict, data: dict):
        response = authClient.get(
            tenant_id=tenant_id, endpoint=f'load-bid', params=params, json_data=data)
        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    # LOAD OFFER BID
    async def create_load_offer_negotiation(self, tenant_id: str, data: dict):
        response = authClient.post(
            tenant_id=tenant_id, endpoint=f'load-offer-negotiation', json_data=data)
        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    async def update_load_offer_bid(self, tenant_id: str, bid_id: str, data: dict):
        response = authClient.patch(
            tenant_id=tenant_id, endpoint=f'load-offer-bid/{bid_id}', json_data=data)
        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    async def get_load_offer_bid(self, tenant_id: str, params: dict):
        response = authClient.get(
            tenant_id=tenant_id, endpoint=f'load-offer-bid', params=params)
        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]

    async def get_load_offer_bid_by_id(self, tenant_id: str, id: str):
        response = authClient.get(
            tenant_id=tenant_id, endpoint=f'load-offer-bid/{id}')
        json_resp = response.json()
        if "data" not in json_resp:
            raise ValueError(json_resp)

        return json_resp["data"]
