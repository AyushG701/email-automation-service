from fastapi import FastAPI, Body, HTTPException
import httpx
import logging

from config import AppConfig
from core.auth_client import AuthApiClient

config = AppConfig()
authClient = AuthApiClient(
    config.BROKER_SERVICE_URL,
    config.APP_KEY
)


class Broker:
    """ SUPER TRUCK BROKER INTEGRATION """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # async def find_broker(self, tenant_id: str, data: dict):
    #     print(data)
    #     response = authClient.get(
    #         tenant_id=tenant_id, endpoint=f'broker/by-identifier', params=data)
    #     print(response.json())
    #     return response.json()["data"]
    async def find_broker(self, tenant_id: str, data: dict):
        # Bypass external broker lookup for now.
        print(f"[Broker.find_broker] Bypassed broker lookup | tenant={tenant_id} | data={data}")
        return None
