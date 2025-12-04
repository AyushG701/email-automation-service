import httpx
import logging
import asyncio
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AuthApiClient:
    """
    Async HTTP client for interacting with the Auth API.
    Handles headers, retries, and lifecycle automatically.
    """

    def __init__(self, base_url: str, app_key: str,):
        self.base_url = base_url
        self.app_key = app_key
        self.default_timeout = None
        self.client: httpx.AsyncClient | None = None

    def _prepare_headers(self, tenant_id: Optional[str] = None, additional_headers: Optional[Dict] = None) -> Dict[str, str]:
        """Prepare headers with tenant ID and bearer token"""
        # token = self._get_valid_token(tenant_id)
        headers = {
            'Authorization': f'Bearer {self.app_key}',
            'Content-Type': 'application/json'
        }

        if tenant_id is not None:
            headers['x-tenant-id'] = tenant_id
        if additional_headers:
            headers.update(additional_headers)

        return headers

    def get(self,  endpoint: str, params: Optional[Dict] = None, tenant_id: Optional[str] = None,
            headers: Optional[Dict] = None, timeout: Optional[float] = None) -> httpx.Response:
        """GET request with dynamic tenant"""
        url = f"{self.base_url}/{endpoint}"
        request_headers = self._prepare_headers(tenant_id, headers)

        with httpx.Client(timeout=timeout or self.default_timeout) as client:
            return client.get(url, params=params, headers=request_headers)

    def post(self, endpoint: str, tenant_id: Optional[str] = None, json_data: Optional[Dict] = None,
             data: Optional[Union[Dict, str]] = None, headers: Optional[Dict] = None,
             timeout: Optional[float] = None) -> httpx.Response:
        """POST request with dynamic tenant"""
        url = f"{self.base_url}/{endpoint}"
        request_headers = self._prepare_headers(tenant_id, headers)
        with httpx.Client(timeout=timeout or self.default_timeout) as client:
            return client.post(url, json=json_data, data=data, headers=request_headers)

    def put(self, endpoint: str, tenant_id: Optional[str] = None, json_data: Optional[Dict] = None,
            data: Optional[Union[Dict, str]] = None, headers: Optional[Dict] = None,
            timeout: Optional[float] = None) -> httpx.Response:
        """PUT request with dynamic tenant"""
        url = f"{self.base_url}/{endpoint}"
        request_headers = self._prepare_headers(tenant_id, headers)

        with httpx.Client(timeout=timeout or self.default_timeout) as client:
            return client.put(url, json=json_data, data=data, headers=request_headers)

    def patch(self, endpoint: str, tenant_id: Optional[str] = None, json_data: Optional[Dict] = None,
              data: Optional[Union[Dict, str]] = None, headers: Optional[Dict] = None,
              timeout: Optional[float] = None) -> httpx.Response:
        """PATCH request with dynamic tenant"""
        url = f"{self.base_url}/{endpoint}"
        request_headers = self._prepare_headers(tenant_id, headers)

        with httpx.Client(timeout=timeout or self.default_timeout) as client:
            return client.patch(url, json=json_data, data=data, headers=request_headers)

    def delete(self, endpoint: str, tenant_id: Optional[str] = None, headers: Optional[Dict] = None,
               timeout: Optional[float] = None) -> httpx.Response:
        """DELETE request with dynamic tenant"""
        url = f"{self.base_url}/{endpoint}"
        request_headers = self._prepare_headers(tenant_id, headers)

        with httpx.Client(timeout=timeout) as client:
            return client.delete(url, headers=request_headers)
