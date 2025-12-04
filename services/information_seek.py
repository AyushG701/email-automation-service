
import os
import json
from openai import OpenAI
from typing import Optional


class InformationSeeker:
    def __init__(self, data: dict, model: str = "gpt-4o-mini"):
        """
        Initialize with carrier data and OpenAI client.
        """
        if not isinstance(data, dict):
            raise ValueError("The data needs to be in a dictionary (JSON) format. Please check your input.")
        self.data = data
        self.model = model

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("It seems the OPENAI_API_KEY is not set. Please add it to your environment variables.")

        self.client = OpenAI(api_key=api_key)

    def update_data(self, new_data: dict):
        """
        Add or update information in the data.
        """
        if not isinstance(new_data, dict):
            raise ValueError("The new data must be a dictionary. Please check the format.")
        self.data.update(new_data)

    def ask(self, question: str, prev_conversation: str = "") -> str:
        """
        Ask a question and get a short, natural response like Parade AI.
        Focus on direct, business-focused answers.
        """
        if not question.strip():
            return "What would you like to know?"

        # Build context from carrier data
        carrier_context = self._build_carrier_context()

        system_prompt = """You are a professional carrier representative in freight/trucking business.

CRITICAL RULES - NEVER VIOLATE:
1. ONLY use information that EXISTS in the CARRIER INFO section
2. If information is NOT in CARRIER INFO, say "I don't have that info currently" or "Let me check on that - what's the load details?"
3. NEVER make up, guess, or assume information (like insurance amounts, equipment details, locations)
4. Answer DIRECTLY and BRIEFLY (1-2 sentences max)
5. Use casual, natural business language
6. NEVER mention you're an AI, assistant, or that you're using data
7. Be friendly but professional
8. Redirect personal/irrelevant questions to business

RESPONSE EXAMPLES:
✅ If MC in data: "Yeah, our MC is 123456"
✅ If insurance NOT in data: "I don't have that info on me right now. What's the load details?"
✅ If equipment in data: "We run 53' dry vans mostly"
✅ If asked about unavailable info: "Let me get back to you on that - got a load for me?"

❌ NEVER say: "We have $100K cargo insurance" (if not in data)
❌ NEVER say: "We cover 48 states" (if not in data)
❌ NEVER make up specific numbers or details
"""

        user_prompt = f"""CARRIER INFO:
{carrier_context}

CONVERSATION HISTORY:
{prev_conversation if prev_conversation else "None"}

BROKER QUESTION:
{question}

IMPORTANT: Only answer if the information EXISTS in CARRIER INFO above. If not, deflect naturally.

Respond naturally and briefly:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature to reduce hallucinations
                max_tokens=150  # Keep responses short
            )
            answer = response.choices[0].message.content.strip()
            return answer

        except Exception as e:
            print(f"It seems there's an issue with the OpenAI API. Here's the error: {str(e)}")
            return "Sorry, can you repeat that? Let's discuss the load details."

    def _build_carrier_context(self) -> str:
        """
        Build a clean context from carrier data for the AI.
        Handles nested structures from supertruck API.
        """
        context_parts = []
        
        # Extract from nested carrierAuthority if exists
        carrier_authority = self.data.get("carrierAuthority", {})
        
        # Company Name
        company_name = carrier_authority.get("companyName") or self.data.get("companyName")
        if company_name:
            context_parts.append(f"Company Name: {company_name}")
        
        # MC Number
        mc_number = carrier_authority.get("mcNumber") or self.data.get("mcNumber")
        if mc_number:
            context_parts.append(f"MC Number: {mc_number}")
        
        # DOT Number
        dot_number = carrier_authority.get("dotNumber") or self.data.get("dotNumber")
        if dot_number:
            context_parts.append(f"DOT Number: {dot_number}")
        
        # Contact Info
        email = carrier_authority.get("companyEmail") or self.data.get("email")
        if email:
            context_parts.append(f"Email: {email}")
        
        phone = carrier_authority.get("companyPhoneNumber") or self.data.get("phone")
        if phone:
            context_parts.append(f"Phone: {phone}")
        
        # Address
        address = carrier_authority.get("authorityAddress") or self.data.get("address")
        if address:
            context_parts.append(f"Address: {address}")
        
        # Company Type
        company_type = self.data.get("companyType")
        if company_type:
            readable_type = company_type.replace("_", " ").title()
            context_parts.append(f"Company Type: {readable_type}")
        
        # Verification Status
        if self.data.get("isMCVerified"):
            context_parts.append("MC: Verified ✓")
        if self.data.get("isDOTVerified"):
            context_parts.append("DOT: Verified ✓")
        
        # Hazmat Certification
        if carrier_authority.get("isHazmatCertified"):
            context_parts.append("Hazmat Certified: Yes")
        
        # Equipment/Trucks
        trucks = self.data.get("trucks", [])
        if trucks:
            context_parts.append(f"\nFleet Size: {len(trucks)} trucks")
            
            # Extract equipment types from trucks
            equipment_types = set()
            for truck in trucks:
                truck_type = truck.get("type")
                if truck_type:
                    equipment_types.add(truck_type.title())
            
            if equipment_types:
                context_parts.append(f"Equipment Types: {', '.join(sorted(equipment_types))}")
            
            # Extract commodities from trucks
            all_commodities = set()
            for truck in trucks:
                commodities = truck.get("preferredCommodity", [])
                for commodity in commodities:
                    readable = commodity.replace("_", " ").title()
                    all_commodities.add(readable)
            
            if all_commodities:
                context_parts.append(f"Preferred Commodities: {', '.join(sorted(all_commodities))}")
        
        # Working Hours
        working_hours = self.data.get("workingHours")
        if working_hours:
            context_parts.append(f"Working Hours: {working_hours}/7 (24/7 available)" if working_hours == "24" else f"Working Hours: {working_hours}")
        
        # Insurance info (if available at root level)
        insurance = self.data.get("insurance", {})
        if insurance:
            context_parts.append(f"Insurance: {json.dumps(insurance)}")
        
        # Service areas / Favorite lanes
        fav_from = self.data.get("favoriteLanesFromCity", [])
        fav_to = self.data.get("favoriteLanesToCity", [])
        if fav_from and fav_from != ['string']:
            context_parts.append(f"Primary Pickup Areas: {', '.join(fav_from)}")
        if fav_to and fav_to != ['string']:
            context_parts.append(f"Primary Delivery Areas: {', '.join(fav_to)}")
        
        # AI Assistant Status
        if self.data.get("isAiAssistantSetupComplete"):
            context_parts.append("AI Assistant: Active")
        
        return "\n".join(context_parts) if context_parts else json.dumps(self.data, indent=2)


# Example usage
if __name__ == "__main__":
    # Sample carrier data
    carrier_data = {
        "companyName": "Swift Logistics LLC",
        "mcNumber": "MC-789456",
        "dotNumber": "DOT-123789",
        "email": "dispatch@swiftlogistics.com",
        "phone": "+1-555-0123",
        "equipmentTypes": ["53' Dry Van", "48' Flatbed"],
        "serviceAreas": ["Texas", "Oklahoma", "Louisiana", "Arkansas"],
        "insurance": {
            "cargo": "$100,000",
            "liability": "$1,000,000"
        },
        "operatingAuthority": "Active",
        "fleetSize": 25,
        "yearsInBusiness": 8
    }

    seeker = InformationSeeker(data=carrier_data)
    
    # Test questions
    print("Q: What's your MC number?")
    print(f"A: {seeker.ask('What is your MC number?')}\n")
    
    print("Q: Do you have cargo insurance?")
    print(f"A: {seeker.ask('Do you have cargo insurance?')}\n")
    
    print("Q: What equipment do you run?")
    print(f"A: {seeker.ask('What equipment do you run?')}\n")
    
    print("Q: What's your favorite color?")
    print(f"A: {seeker.ask('What is your favorite color?')}\n")