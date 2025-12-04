from datetime import datetime
# If you have it, otherwise skip mileage scoring.
# from geopy.distance import geodesic


class LoadScoreService:
    def load_score(self, offer: dict) -> int:
        if not isinstance(offer, dict):
            raise ValueError("Offer must be a dictionary")

        score = 0

        # 1. Pickup & Delivery locations
        if offer.get("pickupLocation") and offer.get("deliveryLocation"):
            score += 20

        # 2. Dates
        if offer.get("pickupDate") and offer.get("deliveryDate"):
            score += 20

        # 3. Rate quality: rate per mile (if possible)
        mileage = self._estimate_mileage(
            offer.get("pickupLocation"),
            offer.get("deliveryLocation")
        )
        # rate = float(offer.get("requestedRate") or 0)

        # if mileage and rate > 0:
        #     rpm = rate / mileage
        #     if rpm >= 2.5:
        #         score += 30
        #     elif rpm >= 2.0:
        #         score += 20
        #     elif rpm >= 1.5:
        #         score += 10
        if offer.get("requestedRate"):
            score += 20

        # 4. Equipment type
        if offer.get("equipmentType"):
            score += 10

        # 5. Weight
        if offer.get("weightLbs") and offer.get("weightLbs") > 0:
            score += 10

        # 6. Instructions
        if offer.get("additionalInstructions"):
            score += 5

        # 7. Broker info
        if offer.get("brokerContactEmail") or offer.get("brokerCompany"):
            score += 10

        # Ensure score is between 0 and 100
        # print(min(score, 100))
        return min(score, 100)

    def _estimate_mileage(self, pickup, delivery):
        """
        Optional: estimates mileage if you have city/state names.
        If no geopy available or just want dummy scoring, return None.
        """
        try:
            # Placeholder: replace with real geocode lookup later.
            # Or remove and use manual mileage input.
            return 1000  # dummy miles
        except:
            return None
