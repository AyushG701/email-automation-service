from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import math
import re
import json
import urllib.request
import urllib.parse
import logging
from typing import Optional, List, Dict, Any, Tuple
import aiohttp
import urllib.parse


class GeocodingService:
    """
    Service to fetch coordinates from various sources dynamically.
    Supports multiple geocoding APIs and methods.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # @staticmethod

    async def geocode_with_nominatim(self, address: str) -> Optional[Tuple[float, float]]:
        try:
            encoded_address = urllib.parse.quote(address)
            url = f"https://nominatim.openstreetmap.org/search?q={encoded_address}&format=json&limit=1"

            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
            }
            # "FreightDistanceCalculator/1.0 basanta@advikai.com"}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=5) as resp:
                    if resp.status != 200:
                        print(f"Nominatim returned status {resp.status}")
                        return None

                    data = await resp.json()

                    if data and len(data) > 0:
                        lat = float(data[0]["lat"])
                        lon = float(data[0]["lon"])
                        return (lat, lon)
                    else:
                        print("Nominatim returned empty data:", data)

        except Exception as e:
            print(f"Nominatim geocoding failed: {e}")

        return None

    @staticmethod
    def geocode_with_google(address: str, api_key: str) -> Optional[Tuple[float, float]]:
        """
        Geocode using Google Maps Geocoding API (Requires API key)

        Args:
            address: Any address string
            api_key: Your Google Maps API key

        Returns:
            Tuple of (latitude, longitude) or None
        """
        try:
            import urllib.request
            import urllib.parse

            encoded_address = urllib.parse.quote(address)
            url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_address}&key={api_key}"

            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())

                if data['status'] == 'OK' and data['results']:
                    location = data['results'][0]['geometry']['location']
                    return (location['lat'], location['lng'])
        except Exception as e:
            print(f"Google geocoding failed: {e}")

        return None

    @staticmethod
    def geocode_with_radar(address: str, api_key: str = "prj_test_sk_2cd9f9d068d4874ce8c72e48c38ee21a8416d5f1") -> Optional[Tuple[float, float]]:
        """
        Geocode an address using the Radar.io Forward Geocoding API.

        Args:
            address: Any address string
            api_key: Your Radar API key

        Returns:
            Tuple of (latitude, longitude) or None
        """
        try:
            import urllib.request
            import urllib.parse

            encoded_address = urllib.parse.quote(address)
            url = f"https://api.radar.io/v1/geocode/forward?query={encoded_address}"

            req = urllib.request.Request(url)
            req.add_header("Authorization", api_key)

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())

                # Radar format: { "addresses": [ { "latitude": ..., "longitude": ... } ] }
                addresses = data.get("addresses", [])
                if addresses:
                    lat = addresses[0].get("latitude")
                    lng = addresses[0].get("longitude")

                    if lat is not None and lng is not None:
                        return (lat, lng)

        except Exception as e:
            print(f"Radar geocoding failed: {e}")

        return None

    @staticmethod
    def radar_distance(point1: tuple, point2: tuple, radar_key: str = "prj_test_sk_2cd9f9d068d4874ce8c72e48c38ee21a8416d5f1"):
        """
        Calculate driving/walking distance using Radar Route API.
        """

        url = "https://api.radar.io/v1/route/distance"

        payload = json.dumps({
            "origins": [{"latitude": point1[0], "longitude": point1[1]}],
            "destinations": [{"latitude": point2[0], "longitude": point2[1]}],
            "modes": ["truck"],  # other modes: foot, bike, truck
            "units": "imperial"  # or "metric"
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload)
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", radar_key)

        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())

                # miles
                distance_miles = data["routes"]["car"]["distance"]["value"]
                # minutes
                duration_minutes = data["routes"]["car"]["duration"]["value"]

                return distance_miles, duration_minutes

        except Exception as e:
            print("Radar distance error:", e)
            return None

    async def calculate_distance(
        self,
        origin: str,
        destination: str,
        radar_api_key: str = "prj_live_sk_365c94d8ba1bd54e6626143c0d18b1d508d07611"
    ) -> Optional[Dict]:
        """
        Geocode two addresses → call Radar distance API → return distance info
        """

        # 1. Geocode origin
        origin_coords = await self.geocode_with_nominatim(origin)
        if not origin_coords:
            self.logger.error("Origin geocoding failed")
            return None
        self.logger.info(f"Origin geocode: {origin_coords}")

        # 2. Geocode destination
        dest_coords = await self.geocode_with_nominatim(destination)
        if not dest_coords:
            self.logger.error("Destination geocoding failed")
            return None
        self.logger.info(f"Destination geocode: {dest_coords}")

        o_lat, o_lng = origin_coords
        d_lat, d_lng = dest_coords

        radar_url = (
            f"https://api.radar.io/v1/route/distance?"
            f"origin={o_lat},{o_lng}&destination={d_lat},{d_lng}"
            f"&modes=truck&units=imperial&geometry=linestring"
        )

        self.logger.info(f"Radar distance calculation URL: {radar_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    radar_url,
                    headers={"Authorization": radar_api_key},
                    timeout=7
                ) as resp:

                    if resp.status != 200:
                        self.logger.error(
                            f"Radar API returned status {resp.status}")
                        return None

                    data = await resp.json()

                    return {
                        "origin": origin_coords,
                        "destination": dest_coords,
                        "distance": data["routes"]["truck"]["distance"]["text"].replace(" mi", ""),
                        "duration": data["routes"]["truck"]["duration"]["text"]
                    }

        except Exception as e:
            self.logger.error(f"Radar API failed: {e}")
            return None
