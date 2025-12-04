"""Load information seeker using OpenAI for intelligent field identification and extraction."""
from typing import Dict, Any, List, Optional
import json
import structlog
from datetime import datetime

from config import AppConfig as Settings
from core.openai import get_openai_client

logger = structlog.get_logger(__name__)

# Critical fields for LOAD OFFERS (less detailed, just enough to quote)
LOAD_OFFER_CRITICAL_FIELDS = [
    'pickupLocation',
    'dropoffLocation', 
    'pickupDate',
    'deliveryDate',
    'equipmentType',
    'weightLbs'
]

LOAD_OFFER_OPTIONAL_FIELDS = [
    'equipmentSize',
    'commodity',
    'rate',
    'brokerContactPhone',
    'brokerContactEmail'
]

# Critical fields for CONFIRMED LOADS (more detailed, ready for dispatch)
CONFIRMED_LOAD_CRITICAL_FIELDS = [
    'pickupLocation',
    'dropoffLocation',
    'pickupDate',
    'deliveryDate',
    'equipmentType',
    'weightLbs',
    'serviceProviderLoadId',
    'rate'
]

CONFIRMED_LOAD_OPTIONAL_FIELDS = [
    'rpm',
    'distance',
    'equipmentSize',
    'commodity',
    'pickupTimeWindowStart',
    'pickupTimeWindowEnd',
    'dropoffTimeWindowStart',
    'dropoffTimeWindowEnd',
    'brokerContactPhone',
    'brokerContactEmail',
    'pickupFacilityName',
    'dropoffFacilityName',
    'pickupFacilityPhone',
    'dropoffFacilityPhone',
    'pickupInstructions',
    'dropoffInstructions'
]

FIELD_DESCRIPTIONS = {
    # Load Offer fields
    'pickupLocation': 'pickup location (city, state)',
    'dropoffLocation': 'delivery location (city, state)',
    'pickupDate': 'pickup date',
    'deliveryDate': 'delivery date',
    'equipmentType': 'equipment type (Dry Van, Reefer, Flatbed, etc.)',
    'weightLbs': 'weight',
    'equipmentSize': 'trailer size',
    'commodity': 'what you\'re hauling',
    'rate': 'rate',
    'brokerContactPhone': 'phone number',
    'brokerContactEmail': 'email',
    
    # Confirmed Load additional fields
    'serviceProviderLoadId': 'load number',
    'rpm': 'rate per mile',
    'distance': 'total miles',
    'pickupTimeWindowStart': 'pickup time window',
    'pickupTimeWindowEnd': 'pickup time window',
    'dropoffTimeWindowStart': 'delivery time window',
    'dropoffTimeWindowEnd': 'delivery time window',
    'pickupFacilityName': 'pickup facility',
    'dropoffFacilityName': 'delivery facility',
    'pickupFacilityPhone': 'pickup facility phone',
    'dropoffFacilityPhone': 'delivery facility phone',
    'pickupInstructions': 'pickup instructions',
    'dropoffInstructions': 'delivery instructions'
}


class LoadInfoSeekerOpenAI:
    """Intelligently identifies missing load information and extracts it from broker responses using OpenAI."""

    def __init__(self):
        """Initialize OpenAI client."""
        if not Settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set. Please check your .env file.")

        self.client = get_openai_client()
        try:
            self.client.models.list()
            logger.info("Successfully connected to OpenAI API for Load Info Seeker")
        except Exception as e:
            logger.error("Having trouble connecting to the OpenAI API for the Load Info Seeker. Please check your connection and API key.")
            raise

        # System prompt for identifying missing fields
        self.missing_fields_system_prompt = """You are an AI assistant that analyzes load offers and confirmed loads to identify missing information.

**CRITICAL FIELDS FOR LOAD OFFERS (enough to provide a quote):**
- pickupLocation: City and state (e.g., "Dallas, TX")
- dropoffLocation: City and state (e.g., "Atlanta, GA")
- pickupDate: Valid date
- deliveryDate: Valid date
- equipmentType: Specific type (Dry Van, Reefer, Flatbed, etc.)
- weightLbs: Weight in pounds (typically > 1000)

**OPTIONAL FOR LOAD OFFERS:**
- equipmentSize: Trailer size (53ft, 48ft)
- commodity: What's being shipped
- rate: Offered rate
- brokerContactPhone/Email: Contact info

**ADDITIONAL CRITICAL FOR CONFIRMED LOADS:**
- serviceProviderLoadId: Load number/ID
- rate: Confirmed rate
- Time windows, facility details, instructions

**RULES:**
1. Missing = null, empty, or too vague
2. Incomplete = lacks necessary detail (e.g., "Dallas" without state)
3. For load offers, only require the 6 critical fields minimum
4. For confirmed loads, require more details

Respond with JSON:
{
  "missing_critical_fields": ["field1"],
  "missing_optional_fields": ["field2"],
  "incomplete_fields": {"field": "why incomplete"},
  "has_sufficient_info_for_negotiation": boolean,
  "priority_level": "high" | "medium" | "low",
  "load_type": "offer" | "confirmed"
}"""

        # System prompt for extracting information
        self.extraction_system_prompt = """You are an AI assistant that extracts load information from broker messages.

**COMMON VARIATIONS:**
- Pickup: "PU", "P/U", "pickup", "origin", "from"
- Delivery: "DEL", "delivery", "dropoff", "destination", "to"
- Dates: "1/20", "01/20/25", "Jan 20"
- Times: "0700-1500", "7am-3pm"
- Equipment: "DV" (Dry Van), "R" (Reefer), "F" (Flatbed), "53'" (53ft)
- Weight: "44k", "44000", "44,000 lbs", "22 tons"

**RULES:**
1. Only extract explicitly stated info
2. Don't guess or hallucinate
3. Convert units: tons to lbs (1 ton = 2000 lbs)
4. Format locations: "City, State" or "City, State Zip"
5. Format dates: YYYY-MM-DD
6. Map equipment abbreviations to full names

Respond with JSON containing extracted fields (null if not found):
{
  "pickupLocation": "City, State",
  "dropoffLocation": "City, State",
  "pickupDate": "YYYY-MM-DD",
  "deliveryDate": "YYYY-MM-DD",
  "equipmentType": "Type",
  "weightLbs": number,
  "equipmentSize": "size",
  "commodity": "description",
  "rate": number,
  ...
}"""

        # System prompt for generating natural messages
        self.message_generation_system_prompt = """You are a truck carrier/dispatcher writing quick messages to freight brokers. Write like a REAL person - natural, casual but professional.

**STYLE:**
- Keep it SHORT and conversational.
- Sound like you're texting or emailing quickly.
- Don't be overly formal.
- Use contractions (I'm, you're, can't, etc.).
- Get straight to the point.
- Friendly but not chatty.

**BAD EXAMPLES (too formal/robotic):**
❌ "I would be delighted to provide you with a competitive quote."
❌ "I am writing to request additional information."
❌ "Please provide the following details at your earliest convenience."

**GOOD EXAMPLES (natural and varied):**
✅ "Hey! Quick question - what's the pickup date on this?"
✅ "I'm interested but need a couple of details. What's the weight and equipment?"
✅ "Can you send me the pickup/delivery locations? I'll get you a rate right away."
✅ "This looks good. Just need the pickup date and I can quote it."
✅ "Got it, thanks. What are the dimensions of the load?"
✅ "Is there any flexibility on the delivery date?"

**TONE BY URGENCY:**
- 1-2 missing fields: Very brief, almost text-like.
- 3+ missing fields: Slightly longer but still casual.
- Optional fields only: Super casual "if you have it" approach.

Generate ONE natural message. NO bullet points. NO formal language. Sound human.

Respond with JSON:
{
  "message": "your natural message",
  "tone": "casual" | "professional" | "brief"
}"""

    def identify_missing_fields(
        self, 
        load_data: Dict[str, Any],
        load_type: str = "offer"
    ) -> Dict[str, Any]:
        """
        Identify missing or incomplete fields.
        
        Args:
            load_data: The load data structure
            load_type: "offer" or "confirmed" - determines which fields to check
            
        Returns:
            Dict with missing fields analysis
        """
        try:
            # Select appropriate field lists based on load type
            if load_type == "confirmed":
                critical_fields = CONFIRMED_LOAD_CRITICAL_FIELDS
                optional_fields = CONFIRMED_LOAD_OPTIONAL_FIELDS
            else:
                critical_fields = LOAD_OFFER_CRITICAL_FIELDS
                optional_fields = LOAD_OFFER_OPTIONAL_FIELDS
            
            load_data_str = json.dumps(load_data, indent=2, default=str)
            
            user_prompt = f"""Analyze this {load_type} for missing information:

```json
{load_data_str}
```

Load type: {load_type}
Check critical fields: {', '.join(critical_fields)}
Check optional fields: {', '.join(optional_fields)}"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": self.missing_fields_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            result['load_type'] = load_type
            
            logger.info(
                "Identified missing load information",
                load_type=load_type,
                critical_missing=len(result.get('missing_critical_fields', [])),
                optional_missing=len(result.get('missing_optional_fields', [])),
                sufficient=result.get('has_sufficient_info_for_negotiation', False)
            )
            
            return result

        except Exception as e:
            logger.error("It seems there was an issue identifying the missing fields. Please check the data and try again.")
            return self._fallback_missing_fields_check(load_data, load_type)

    def _fallback_missing_fields_check(
        self, 
        load_data: Dict[str, Any],
        load_type: str = "offer"
    ) -> Dict[str, Any]:
        """Fallback rule-based approach."""
        if load_type == "confirmed":
            critical_fields = CONFIRMED_LOAD_CRITICAL_FIELDS
            optional_fields = CONFIRMED_LOAD_OPTIONAL_FIELDS
        else:
            critical_fields = LOAD_OFFER_CRITICAL_FIELDS
            optional_fields = LOAD_OFFER_OPTIONAL_FIELDS
        
        missing_critical = []
        missing_optional = []
        
        for field in critical_fields:
            value = load_data.get(field)
            if value is None or value == "" or (isinstance(value, str) and len(value.strip()) < 3):
                missing_critical.append(field)
        
        for field in optional_fields:
            value = load_data.get(field)
            if value is None or value == "":
                missing_optional.append(field)
        
        critical_for_negotiation = ['pickupLocation', 'dropoffLocation', 'pickupDate', 'deliveryDate']
        has_sufficient = len([f for f in missing_critical if f in critical_for_negotiation]) == 0
        
        return {
            'missing_critical_fields': missing_critical,
            'missing_optional_fields': missing_optional,
            'incomplete_fields': {},
            'has_sufficient_info_for_negotiation': has_sufficient,
            'priority_level': 'high' if len(missing_critical) >= 3 else 'medium' if len(missing_critical) > 0 else 'low',
            'load_type': load_type
        }

    def generate_request_message(
        self, 
        missing_info: Dict[str, Any],
        broker_company: Optional[str] = None,
        broker_contact_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a natural, human-sounding message.
        
        Args:
            missing_info: Output from identify_missing_fields()
            broker_company: Broker company name (optional)
            broker_contact_name: Broker contact person name (optional)
            
        Returns:
            Dict with generated message
        """
        try:
            missing_critical = missing_info.get('missing_critical_fields', [])
            missing_optional = missing_info.get('missing_optional_fields', [])
            
            if not missing_critical and not missing_optional:
                return {
                    'message': '',
                    'tone': 'none',
                    'should_send': False
                }
            
            # Prepare natural field names
            critical_readable = [FIELD_DESCRIPTIONS.get(f, f) for f in missing_critical]
            optional_readable = [FIELD_DESCRIPTIONS.get(f, f) for f in missing_optional[:2]]  # Max 2 optional
            
            context = {
                'broker_company': broker_company,
                'broker_contact_name': broker_contact_name,
                'missing_critical': critical_readable,
                'missing_optional': optional_readable,
                'priority': missing_info.get('priority_level', 'medium'),
                'count': len(missing_critical)
            }
            
            context_str = json.dumps(context, indent=2)
            
            user_prompt = f"""Generate a natural message requesting this info:

{context_str}

Remember: Write like you're quickly texting/emailing a broker. Be casual but professional. Keep it SHORT."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": self.message_generation_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,  # Slightly higher for more natural variation
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            result['should_send'] = True
            
            logger.info(
                "Generated request message",
                tone=result.get('tone'),
                length=len(result.get('message', ''))
            )
            
            return result

        except Exception as e:
            logger.error("An error occurred while generating the request message. Please review the details and try again.")
            return self._fallback_message_generation(missing_info, broker_company)

    def _fallback_message_generation(
        self, 
        missing_info: Dict[str, Any],
        broker_company: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback with more natural messages."""
        missing_critical = missing_info.get('missing_critical_fields', [])
        
        if not missing_critical:
            return {'message': '', 'tone': 'none', 'should_send': False}
        
        # Get readable field names
        fields = [FIELD_DESCRIPTIONS.get(f, f) for f in missing_critical[:3]]
        
        # More natural templates with dynamic sentence construction
        if len(fields) == 1:
            message = f"Hey! Quick question - what's the {fields[0]} on this load?"
        elif len(fields) == 2:
            message = f"Interested in this. Just need the {fields[0]} and {fields[1]} to quote it."
        else:
            message = f"I can cover this but need a few details: {', '.join(fields)}."

        return {
            'message': message,
            'tone': 'casual',
            'should_send': True
        }

    def extract_info_from_response(
        self,
        broker_message: str,
        current_load_data: Dict[str, Any],
        current_date_iso: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract load information from broker's response.
        
        Args:
            broker_message: Broker's response text
            current_load_data: Current load data
            current_date_iso: Current date for context
            
        Returns:
            Dict with extracted information
        """
        try:
            current_load_str = json.dumps(current_load_data, indent=2, default=str)
            date_context = f"\nCurrent date: {current_date_iso}" if current_date_iso else ""
            
            user_prompt = f"""Extract info from this broker message:

**Message:**
{broker_message}

**Current Load Data:**
```json
{current_load_str}
```
{date_context}

Extract any load-related info mentioned."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": self.extraction_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            extracted = json.loads(response.choices[0].message.content)
            extracted = {k: v for k, v in extracted.items() if v is not None}
            
            # Post-process dates
            for date_field in ['pickupDate', 'deliveryDate']:
                if date_field in extracted and extracted[date_field]:
                    try:
                        if 'T' in str(extracted[date_field]):
                            dt_obj = datetime.fromisoformat(extracted[date_field].replace('Z', ''))
                            extracted[date_field] = dt_obj.strftime('%Y-%m-%d')
                    except ValueError:
                        logger.warning(f"It seems the date '{extracted[date_field]}' is not in a recognizable format. Could you please format it as MM/DD/YYYY?")
                        del extracted[date_field]
            
            logger.info(
                "Extracted information",
                fields=list(extracted.keys()),
                count=len(extracted)
            )
            
            return {
                'extracted_fields': extracted,
                'extraction_successful': len(extracted) > 0,
                'fields_found': list(extracted.keys())
            }

        except Exception as e:
            logger.error(f"An unexpected error occurred while extracting information. Please ensure the data is correctly formatted and try again.")
            return {
                'extracted_fields': {},
                'extraction_successful': False,
                'fields_found': [],
                'error': str(e)
            }

    def has_sufficient_information(
        self, 
        load_data: Dict[str, Any],
        load_type: str = "offer"
    ) -> bool:
        """Check if load has sufficient information."""
        missing_info = self.identify_missing_fields(load_data, load_type)
        return missing_info.get('has_sufficient_info_for_negotiation', False)

    def merge_extracted_info(
        self,
        current_load_data: Dict[str, Any],
        extracted_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge extracted information into current load data."""
        updated = current_load_data.copy()
        
        for key, value in extracted_info.items():
            if value is not None and value != "":
                current_value = updated.get(key)
                if current_value is None or current_value == "" or (isinstance(current_value, str) and len(current_value.strip()) < 3):
                    updated[key] = value
                    logger.debug(f"Updated '{key}' with: {value}")
        
        return updated


# Example usage
if __name__ == "__main__":
    info_seeker = LoadInfoSeekerOpenAI()
    
    # Example 1: Load Offer (minimal fields needed)
    print("=== LOAD OFFER Example ===")
    load_offer = {
        "pickupLocation": "Dallas",
        "dropoffLocation": None,
        "pickupDate": "2025-01-20",
        "deliveryDate": None,
        "equipmentType": "trailer",
        "weightLbs": None,
        "rate": 2500,
        "brokerCompany": "ABC Freight"
    }
    
    missing = info_seeker.identify_missing_fields(load_offer, "offer")
    print(json.dumps(missing, indent=2))
    
    message = info_seeker.generate_request_message(missing, "ABC Freight")
    print(f"\nGenerated Message: {message['message']}")
    
    # Example 2: Broker Response Extraction
    print("\n\n=== BROKER RESPONSE Example ===")
    broker_msg = "PU: Dallas, TX 75201, DEL: Atlanta, GA 30301, 1/20-1/21, 53' DV, 44k"
    
    extraction = info_seeker.extract_info_from_response(broker_msg, load_offer, "2025-01-19T10:00:00Z")
    print(json.dumps(extraction, indent=2))
    
    updated = info_seeker.merge_extracted_info(load_offer, extraction['extracted_fields'])
    print("\nUpdated Load Offer:")
    print(json.dumps(updated, indent=2))