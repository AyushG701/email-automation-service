import os
import json
from openai import OpenAI

REQUIRED_FIELDS = [
    "pickupLocation",
    "dropoffLocation",
    "equipmentType",
    "rate",
    "pickupDate",
    "dropoffDate",
    "weight"
]


class InformationSeeker:
    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.model = model
        self.client = OpenAI(api_key=api_key)

        # Keep your original conversations list
        self.conversations = [
            {
                "id": "bd2fa7a1-72a5-4e4c-9f26-e0f67ce4591d",
                "negotiationDirection": "incoming",
                "negotiationRawEmail": "Hi,Available power-only load for immediate dispatch: Load Details: Pickup: San Antonio, TX – 11/08/2025 08: 00 AM Delivery: Atlanta, GA – 11/12/2025 06: 00 PM Trailer Type: 53' Dry Van(preloaded) Weight: 38,700 lbs Rate: $4000 all-in Notes:Preloaded trailer—ready to roll.Must have clean, odor-free trailer. Let us know if you can take this load."""
            }
        ]

        self.data = self._extract_fields_from_conversation()

    def _extract_fields_from_conversation(self) -> dict:
        """
        Ask the model to infer missing fields from the negotiation history.
        """
        conversation_text = "\n".join(
            f"{m['negotiationDirection']}: {m['negotiationRawEmail']}"
            for m in self.conversations
        )
        prompt = f"""
        You are analyzing a freight load negotiation conversation.

        Extract as many of the following fields as possible:

        pickupLocation
        dropoffLocation
        equipmentType
        rate
        pickupDate
        dropoffDate
        weight

        If unknown or not stated, return null.

        Return ONLY valid JSON.

        Conversation:
        {conversation_text}
        """

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        raw = resp.choices[0].message.content.strip()

        try:
            return json.loads(raw)
        except:
            return {field: None for field in REQUIRED_FIELDS}

    def ask(self) -> str:
        """
        Ask for missing load details using the model to generate the message.
        """

        # Re-extract fields after the current conversation
        self.data = self._extract_fields_from_conversation()

        # Build conversation text
        convo_text = "\n".join(
            f"{m['negotiationDirection']}: {m['negotiationRawEmail']}"
            for m in self.conversations
        )

        # Prompt the model to ask for missing details only
        prompt = f"""You are a trucking dispatcher reviewing a load negotiation conversation.

        Current conversation:
        {convo_text}

        Current extracted data:
        {json.dumps(self.data, indent=2)}

        Some fields may be missing. Generate a short, friendly, natural message asking the broker only for the missing details.
        Do NOT include fields that are already known. Do NOT negotiate — only ask for missing info.
        Do NOT include fields that are already known.
        Do NOT include greetings, signatures, or subject lines.
        Do NOT negotiate — only ask for missing info.
        Return ONLY the message text.
        """

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        print(resp.choices[0].message.content.strip())
        return resp.choices[0].message.content.strip()
