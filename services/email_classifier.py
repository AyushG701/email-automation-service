# pylint: disable=line-too-long
"""Email classification and detail extraction using OpenAI."""
from typing import Dict, Any, List, Optional
import json
import re
import structlog

from datetime import datetime, date  # Added date for current year

from config import AppConfig as Settings
from core.openai import get_openai_client

logger = structlog.get_logger(__name__)

# Define the expected schema for load_offer details for post-processing (updated to match ILoad interface)
LOAD_OFFER_SCHEMA_KEYS = [
    "serviceProviderLoadId", "serviceProviderName", "brokerContactEmail", "brokerCompany", "brokerContactPhone",
    "rpm", "distance", "rate", "stops", "pickupTimeType", "pickupTimeExact", "pickupTimeWindowStart", "pickupTimeWindowEnd",
    "isAppointmentRequiredInPickup", "dropoffTimeType", "dropoffTimeExact", "dropoffTimeWindowStart", "dropoffTimeWindowEnd",
    "isAppointmentRequiredInDropoff", "pickupLocation", "dropoffLocation", "pickupDate", "deliveryDate", "weightLbs",
    "pickupFacilityName", "dropoffFacilityName", "pickupFacilityPhone", "dropoffFacilityPhone",
    "pickupInstructions", "dropoffInstructions", "shipper", "consignee",
    "equipmentType", "equipmentSize", "commodity", "totalCommodityCount", "expectedMinTemp", "expectedMaxTemp",
    "isHazmatRequired", "weight", "weightUnit", "size", "pallets", "isLoadExpired", "senderEmail", "emailContentFromBroker"
]

# Define the expected schema for load details for post-processing (updated to match ILoad interface)
LOAD_SCHEMA_KEYS = [
    "serviceProviderLoadId", "serviceProviderName", "brokerContactEmail", "brokerCompany", "brokerContactPhone",
    "rpm", "distance", "rate", "stops", "pickupTimeType", "pickupTimeExact", "pickupTimeWindowStart", "pickupTimeWindowEnd",
    "isAppointmentRequiredInPickup", "dropoffTimeType", "dropoffTimeExact", "dropoffTimeWindowStart", "dropoffTimeWindowEnd",
    "isAppointmentRequiredInDropoff", "pickupLocation", "dropoffLocation", "pickupDate", "deliveryDate", "weightLbs",
    "pickupFacilityName", "dropoffFacilityName", "pickupFacilityPhone", "dropoffFacilityPhone",
    "pickupInstructions", "dropoffInstructions", "shipper", "consignee",
    "equipmentType", "equipmentSize", "commodity", "totalCommodityCount", "expectedMinTemp", "expectedMaxTemp",
    "isHazmatRequired", "weight", "weightUnit", "size", "pallets", "isLoadExpired", "senderEmail", "emailContentFromBroker"
]

# Define the expected schema for broker_setup_request details for post-processing
BROKER_SETUP_REQUEST_SCHEMA_KEYS = [
    "comments", "priority", "brokerEmail", "brokerCompany", "setupLink", "senderEmail", "emailContentFromBroker"
]

# Define expected types for keys to guide OpenAI and for potential future validation (optional)
# This is more for prompting and understanding, strict type conversion isn't implemented here.
LOAD_OFFER_SCHEMA_TYPES_PROMPT_TEXT = """
**MANDATORY/PRIORITY FIELDS (extract with highest accuracy, do NOT guess or hallucinate):**
- equipmentType: string or null (PRIORITY: e.g., 'Dry Van', 'Flatbed', 'Reefer', 'Power Only', 'Step Deck', 'Box Truck', 'Conestoga', 'Double Drop', 'Lowboy') - search thoroughly in email for equipment mentions
- pickupLocation: string or null (PRIORITY: PRIMARY pickup city, state, zipcode - this is the main origin point) - search thoroughly for origin/pickup location
- dropoffLocation: string or null (PRIORITY: FINAL delivery city, state, zipcode - this is the main destination point) - search thoroughly for destination/delivery location  
- pickupDate: string or null (PRIORITY: ISO 8601 Date string e.g., "2025-01-20" - extract pickup/ship date from email, convert to YYYY-MM-DD format)
- deliveryDate: string or null (PRIORITY: ISO 8601 Date string e.g., "2025-01-21" - extract delivery/arrival date from email, convert to YYYY-MM-DD format)
- weightLbs: number or null (PRIORITY: weight in pounds - search for weight mentions, convert to lbs if needed, e.g., 40000 for 40k lbs)

**OTHER FIELDS:**
- serviceProviderLoadId: string or null (CRITICAL: if ANY kind of load ID is found, classify as 'load' instead of 'load_offer')
- serviceProviderName: string or null
- brokerContactEmail: string or null (the email address of the broker/contact)
- brokerCompany: string or null
- brokerContactPhone: string or null
- rpm: number or null (rate per mile)
- distance: number or null
- rate: number or null (total rate for the load offer)
- stops: array of INTERMEDIATE stop objects or null. CRITICAL: This array should ONLY contain stops that occur BETWEEN the main pickup location and the final delivery location. DO NOT include the primary pickup location (goes in pickupLocation) or the final delivery location (goes in dropoffLocation) in this stops array. If there are no intermediate stops along the route, this should be null or an empty array. Each intermediate stop object should have: { stopDate: string|null (ISO 8601 Date string), location: string|null (intermediate stop location), palletInfo: string|null, contactEntityName: string|null, contactEntityPhone: string|null, notes: string|null, direction: string|null (must be 'PICKUP' or 'DROPOFF' indicating what happens at this intermediate stop) }
- pickupTimeType: string, must be 'EXACT' or 'RANGE' or null
- pickupTimeExact: ISO 8601 datetime string or null (use when pickupTimeType is 'EXACT')
- pickupTimeWindowStart: ISO 8601 datetime string or null (use when pickupTimeType is 'RANGE')
- pickupTimeWindowEnd: ISO 8601 datetime string or null (use when pickupTimeType is 'RANGE')
- isAppointmentRequiredInPickup: boolean or null
- dropoffTimeType: string, must be 'EXACT' or 'RANGE' or null
- dropoffTimeExact: ISO 8601 datetime string or null (use when dropoffTimeType is 'EXACT')
- dropoffTimeWindowStart: ISO 8601 datetime string or null (use when dropoffTimeType is 'RANGE')
- dropoffTimeWindowEnd: ISO 8601 datetime string or null (use when dropoffTimeType is 'RANGE')
- isAppointmentRequiredInDropoff: boolean or null
- pickupFacilityName: string or null
- dropoffFacilityName: string or null
- pickupFacilityPhone: string or null
- dropoffFacilityPhone: string or null
- pickupInstructions: string or null
- dropoffInstructions: string or null
- shipper: string or null
- consignee: string or null
- equipmentSize: string or null (e.g., '53ft', '48ft')
- commodity: string or null
- totalCommodityCount: number or null (defaults to 1)
- expectedMinTemp: number or null (for reefer loads)
- expectedMaxTemp: number or null (for reefer loads)
- isHazmatRequired: boolean or null
- weight: number or null (legacy field, prefer weightLbs)
- weightUnit: string or null (e.g., 'lbs', 'kg')
- size: number or null
- pallets: number or null
- isLoadExpired: boolean or null
- senderEmail: string or null (the email address of the person who sent this email)
- emailContentFromBroker: string or null (the actual email body content after cleaning forwarded message headers)

**CRITICAL EXTRACTION RULES:**
1. For PRIORITY fields, search thoroughly throughout the entire email content
2. DO NOT guess or hallucinate values - if not clearly stated, use null
3. Convert units appropriately (tons to lbs, kg to lbs, etc.) but only if conversion is certain
4. Extract dates in YYYY-MM-DD format for pickupDate and deliveryDate
5. Be precise with location formatting: "City, State Zipcode" when possible
"""

LOAD_SCHEMA_TYPES_PROMPT_TEXT = """
**MANDATORY/PRIORITY FIELDS (extract with highest accuracy, do NOT guess or hallucinate):**
- equipmentType: string or null (PRIORITY: e.g., 'Dry Van', 'Flatbed', 'Reefer', 'Power Only', 'Step Deck', 'Box Truck', 'Conestoga', 'Double Drop', 'Lowboy') - search thoroughly in email for equipment mentions
- pickupLocation: string or null (PRIORITY: PRIMARY pickup city, state, zipcode - this is the main origin point) - search thoroughly for origin/pickup location
- dropoffLocation: string or null (PRIORITY: FINAL delivery city, state, zipcode - this is the main destination point) - search thoroughly for destination/delivery location
- pickupDate: string or null (PRIORITY: ISO 8601 Date string e.g., "2025-01-20" - extract pickup/ship date from email, convert to YYYY-MM-DD format)
- deliveryDate: string or null (PRIORITY: ISO 8601 Date string e.g., "2025-01-21" - extract delivery/arrival date from email, convert to YYYY-MM-DD format)
- weightLbs: number or null (PRIORITY: weight in pounds - search for weight mentions, convert to lbs if needed, e.g., 40000 for 40k lbs)

**OTHER FIELDS:**
- serviceProviderLoadId: string or null (CRITICAL: presence of ANY load ID strongly indicates this should be classified as 'load')
- serviceProviderName: string or null
- brokerContactEmail: string or null (the email address of the broker/contact)
- brokerCompany: string or null
- brokerContactPhone: string or null
- rpm: number or null (rate per mile)
- distance: number or null
- rate: number or null (total rate for the load)
- stops: array of INTERMEDIATE stop objects or null. CRITICAL: This array should ONLY contain stops that occur BETWEEN the main pickup location and the final delivery location. DO NOT include the primary pickup location (goes in pickupLocation) or the final delivery location (goes in dropoffLocation) in this stops array. If there are no intermediate stops along the route, this should be null or an empty array. Each intermediate stop object should have: { stopDate: string|null (ISO 8601 Date string), location: string|null (intermediate stop location), palletInfo: string|null, contactEntityName: string|null, contactEntityPhone: string|null, notes: string|null, direction: string|null (must be 'PICKUP' or 'DROPOFF' indicating what happens at this intermediate stop) }
- pickupTimeType: string, must be 'EXACT' or 'RANGE' or null
- pickupTimeExact: ISO 8601 datetime string or null (use when pickupTimeType is 'EXACT')
- pickupTimeWindowStart: ISO 8601 datetime string or null (use when pickupTimeType is 'RANGE')
- pickupTimeWindowEnd: ISO 8601 datetime string or null (use when pickupTimeType is 'RANGE')
- isAppointmentRequiredInPickup: boolean or null
- dropoffTimeType: string, must be 'EXACT' or 'RANGE' or null
- dropoffTimeExact: ISO 8601 datetime string or null (use when dropoffTimeType is 'EXACT')
- dropoffTimeWindowStart: ISO 8601 datetime string or null (use when dropoffTimeType is 'RANGE')
- dropoffTimeWindowEnd: ISO 8601 datetime string or null (use when dropoffTimeType is 'RANGE')
- isAppointmentRequiredInDropoff: boolean or null
- pickupFacilityName: string or null
- dropoffFacilityName: string or null
- pickupFacilityPhone: string or null
- dropoffFacilityPhone: string or null
- pickupInstructions: string or null
- dropoffInstructions: string or null
- shipper: string or null
- consignee: string or null
- equipmentSize: string or null (e.g., '53ft', '48ft')
- commodity: string or null
- totalCommodityCount: number or null (defaults to 1)
- expectedMinTemp: number or null (for reefer loads)
- expectedMaxTemp: number or null (for reefer loads)
- isHazmatRequired: boolean or null
- weight: number or null (legacy field, prefer weightLbs)
- weightUnit: string or null (e.g., 'lbs', 'kg')
- size: number or null
- pallets: number or null
- isLoadExpired: boolean or null
- senderEmail: string or null (the email address of the person who sent this email)
- emailContentFromBroker: string or null (the actual email body content after cleaning forwarded message headers)

**CRITICAL EXTRACTION RULES:**
1. For PRIORITY fields, search thoroughly throughout the entire email content
2. DO NOT guess or hallucinate values - if not clearly stated, use null
3. Convert units appropriately (tons to lbs, kg to lbs, etc.) but only if conversion is certain
4. Extract dates in YYYY-MM-DD format for pickupDate and deliveryDate
5. Be precise with location formatting: "City, State Zipcode" when possible
"""

BROKER_SETUP_REQUEST_SCHEMA_TYPES_PROMPT_TEXT = """
- comments: string or null
- priority: string, must be one of 'high', 'medium', 'low', or null
- brokerEmail: string (email format) or null
- brokerCompany: string or null
- setupLink: string (URL format) or null
- senderEmail: string or null (the email address of the person who sent this email)
- emailContentFromBroker: string or null (the actual email body content after cleaning forwarded message headers)
"""


class EmailClassifierOpenAI:
    """Classifies email content and extracts key details using OpenAI."""

    def __init__(self):
        """Initialize OpenAI client."""
        if not Settings.OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key is not set. Please check your .env file.")

        self.client = get_openai_client()
        try:
            self.client.models.list()  # Test connection
            logger.info(
                "Successfully connected to OpenAI API for Classifier/Extractor")
        except Exception as e:
            logger.error(
                f"Failed to connect to OpenAI API for Classifier/Extractor: {str(e)}")
            raise

        # Construct the system prompt dynamically during initialization
        self.system_prompt_content = f"""You are an AI assistant that processes emails for a logistics company. Your tasks are:

1. Classify the email's primary intent using these DETAILED classification rules:

    **COVERED_LOAD Classification (HIGHEST PRIORITY):**
    - MUST classify as 'covered_load' if the subject line contains ANY of these patterns (case-insensitive):
      * "load covered" or "load is covered"
      * "offer covered" or "offer is offered" 
      * "covered" (when appearing with "load", "offer", or freight-related context)
      * "taken" or "booked" (when referring to a load/offer)
      * "no longer available"
      * "filled" (when referring to a load position)
    - Examples of covered_load subjects: "Re: Offer – Load Covered", "Load Covered - TX to CA", "Offer Covered", "Load is Taken", "Position Filled"
    - This classification takes ABSOLUTE PRECEDENCE even if the email body mentions rate confirmations, documentation, or other follow-up items
    
    **RATE_CONFIRMATION Classification (SECOND HIGHEST PRIORITY):**
    - MUST classify as 'rate_confirmation' if ANY of the following conditions are met:
      * Email contains attachments with "rate confirmation", "rate conf", "RC", or "BOL" in the filename (case-insensitive)
      * Email contains attachments with PDF, DOC, DOCX extensions AND subject mentions load numbers (e.g., "Load# 123", "Load ID", etc.)
      * Email body contains actual rate confirmation details (rates, pickup/delivery locations, terms, etc.)
      * Subject line contains patterns like "Rate Confirmation", "RC for", "BOL", "Bill of Lading"
      * Email explicitly states rate confirmation is attached or included
    - Do NOT classify as 'rate_confirmation' ONLY if the email mentions future rate confirmations without attachments or current details
    - Examples of rate_confirmation:
      * Subject: "Fwd: Spot Freight Inc Load# S2472201" with attachments "Rate Confirmation Spot.pdf", "BOL Spot Inc.jpg"
      * Subject: "Rate Confirmation - Load RC-123" 
      * Body: "Please find attached the signed rate confirmation"
      * Attachments: "RC_LoadABC.pdf", "bill_of_lading.pdf"
    
    **CRITICAL: LOAD vs LOAD_OFFER Classification Rules:**
    
    **LOAD Classification (THIRD HIGHEST PRIORITY):**
    - MUST classify as 'load' if ANY of the following ID patterns are found ANYWHERE in the email:
      * serviceProviderLoadId (e.g., "SP-LOAD-123456", "LOAD-ABC123")
      * Load numbers with specific formats (e.g., "Load# 123", "Load ID: ABC123", "LID: XYZ789")
      * Reference numbers that appear to be formal load identifiers
      * Broker load IDs or tracking numbers
    - Emails describing detailed, confirmed loads ready for dispatch
    - More detailed information than load_offer, often with confirmed booking information
    - Even if email appears to be an offer, presence of formal IDs indicates it's a confirmed load
    
    **LOAD_OFFER Classification:**
    - ONLY classify as 'load_offer' if NO formal load IDs are found in the email
    - Emails presenting new load opportunities seeking carriers
    - Details like origin/destination, dates, equipment type, rate offered
    - Less formal, more exploratory language ("looking for", "need coverage", "can you handle")
    - Subject often contains: "Load Available", "Urgent Load", "FTL", origin-destination pairs
    - If ANY doubt exists about presence of load IDs, classify as 'load' instead
    
    **NEGOTIATION_BIDDING Classification:**
    - Emails discussing terms, rates, or counter-offers for existing loads
    - Contains phrases like: "can you do", "counter offer", "rate adjustment", "best rate"
    - Usually references an existing load ID or previous offer
    
    **BROKER_SETUP_REQUEST Classification:**
    - Emails requesting carrier onboarding or setup completion
    - Contains phrases like: "setup packet", "onboarding", "get started", "complete our forms"
    - Often includes setup links or attachment requests
    
    **OTHER Classification:**
    - Use this only when none of the above categories clearly apply
    - General inquiries, administrative messages, or unclear content

2. Extract key details from the email's subject and body.

**CLASSIFICATION DECISION PROCESS:**
1. FIRST: Check subject line for covered_load patterns (highest priority)
2. SECOND: Check for rate confirmation indicators (attachments, subject patterns, body content)
3. THIRD: Look for ANY formal load IDs - if found, classify as 'load'
4. FOURTH: If no formal IDs found, analyze for load opportunities and classify as 'load_offer'
5. FIFTH: Check for negotiations, setup requests, or other categories
6. LAST: Default to 'other' only if no clear pattern matches

**IMPORTANT: Pay special attention to attachment filenames when classifying emails. Attachments containing "Rate Confirmation", "RC", "BOL", or similar terms strongly indicate rate_confirmation classification.**

Respond with a single JSON object containing two top-level keys:
- "classification": One of the classification strings listed above.
- "extracted_details": A JSON object containing the key details you extracted. Use snake_case for keys in extracted_details.

If the email is classified as 'load_offer', the 'extracted_details' object MUST conform to the following structure. All specified keys must be present. If a piece of information for a key is not found in the email, its value MUST be null. The expected keys and their general types are:
{LOAD_OFFER_SCHEMA_TYPES_PROMPT_TEXT}

If the email is classified as 'load', the 'extracted_details' object MUST conform to the following structure. All specified keys must be present. If a piece of information for a key is not found in the email, its value MUST be null. The expected keys and their general types are:
{LOAD_SCHEMA_TYPES_PROMPT_TEXT}

If the email is classified as 'broker_setup_request', the 'extracted_details' object MUST conform to the following structure. All specified keys must be present. If a piece of information for a key is not found in the email, its value MUST be null. The expected keys and their general types are:
{BROKER_SETUP_REQUEST_SCHEMA_TYPES_PROMPT_TEXT}

Example response for 'rate_confirmation':
{{
  "classification": "rate_confirmation",
  "extracted_details": {{
    "load_id_mentioned": "RC-XYZ123",
    "senderEmail": "broker@example.com",
    "emailContentFromBroker": "Attached is the **signed Rate Confirmation** for load RC-XYZ123 with all details. Please review."
  }}
}}

Example response for 'load_offer':
{{
  "classification": "load_offer",
  "extracted_details": {{
    "serviceProviderLoadId": null,
    "serviceProviderName": null,
    "brokerContactEmail": "sender@example.com",
    "brokerCompany": "ABC Freight Brokers",
    "brokerContactPhone": "555-123-4567",
    "rpm": null,
    "distance": null,
    "rate": 2500.00,
    "stops": null,
    "pickupTimeType": "RANGE",
    "pickupTimeExact": null,
    "pickupTimeWindowStart": "2025-07-14T09:00:00Z",
    "pickupTimeWindowEnd": "2025-07-14T17:00:00Z",
    "isAppointmentRequiredInPickup": false,
    "dropoffTimeType": "EXACT",
    "dropoffTimeExact": "2025-07-15T17:00:00Z",
    "dropoffTimeWindowStart": null,
    "dropoffTimeWindowEnd": null,
    "isAppointmentRequiredInDropoff": false,
    "pickupLocation": "Dallas, TX 75201",
    "dropoffLocation": "Atlanta, GA 30301",
    "pickupDate": "2025-07-14",
    "deliveryDate": "2025-07-15",
    "weightLbs": 40000,
    "pickupFacilityName": null,
    "dropoffFacilityName": null,
    "pickupFacilityPhone": null,
    "dropoffFacilityPhone": null,
    "pickupInstructions": "Pickup available immediately",
    "dropoffInstructions": "Deliver by tomorrow, July 15th, end of day",
    "shipper": "Unknown",
    "consignee": null,
    "equipmentType": "Dry Van",
    "equipmentSize": "53ft",
    "commodity": "general freight",
    "totalCommodityCount": 1,
    "expectedMinTemp": null,
    "expectedMaxTemp": null,
    "isHazmatRequired": false,
    "weight": 40000,
    "weightUnit": "lbs",
    "size": null,
    "pallets": null,
    "isLoadExpired": false,
    "senderEmail": "sender@example.com",
    "emailContentFromBroker": "Good morning, We have an urgent FTL shipment from Dallas, TX 75201 to Atlanta, GA 30301. Weight is 40,000 lbs, commodity is general freight. Needs a 53ft Dry Van. Pickup available immediately (today, July 14th) and delivery required by tomorrow, July 15th, end of day. We are offering $2500. Please call John at 555-123-4567 or reply to this email if you can cover this."
  }}
}}

Example response for 'load' (with intermediate stops):
{{
  "classification": "load",
  "extracted_details": {{
    "serviceProviderLoadId": "SP-LOAD-123456",
    "serviceProviderName": "ABC Logistics",
    "brokerContactEmail": "broker@example.com",
    "brokerCompany": "XYZ Brokerage",
    "brokerContactPhone": "+14155552671",
    "rpm": 2.5,
    "distance": 1200,
    "rate": 3000.75,
    "stops": [
      {{
        "stopDate": "2025-06-02T14:00:00Z",
        "location": "Memphis, TN 38104",
        "palletInfo": "12 pallets",
        "contactEntityName": "Jane Doe",
        "contactEntityPhone": "+19015551234",
        "notes": "Partial delivery at warehouse dock 3",
        "direction": "DROPOFF"
      }}
    ],
    "pickupTimeType": "RANGE",
    "pickupTimeExact": null,
    "pickupTimeWindowStart": "2025-06-01T11:00:00Z",
    "pickupTimeWindowEnd": "2025-06-01T12:00:00Z",
    "isAppointmentRequiredInPickup": true,
    "dropoffTimeType": "EXACT",
    "dropoffTimeExact": "2025-06-05T16:00:00Z",
    "dropoffTimeWindowStart": null,
    "dropoffTimeWindowEnd": null,
    "isAppointmentRequiredInDropoff": false,
    "pickupLocation": "Los Angeles, CA 90001",
    "dropoffLocation": "Chicago, IL 60601",
    "pickupDate": "2025-06-01",
    "deliveryDate": "2025-06-05",
    "weightLbs": 20000,
    "pickupFacilityName": "XYZ Shippers Inc.",
    "dropoffFacilityName": "ABC Consignee Co.",
    "pickupFacilityPhone": "+14155552671",
    "dropoffFacilityPhone": "+17085551234",
    "pickupInstructions": "Call before arrival",
    "dropoffInstructions": "Deliver at back gate",
    "shipper": "XYZ Shippers Inc.",
    "consignee": "ABC Consignee Co.",
    "equipmentType": "Dry Van",
    "equipmentSize": "53ft",
    "commodity": "general freight",
    "totalCommodityCount": 1,
    "expectedMinTemp": null,
    "expectedMaxTemp": null,
    "isHazmatRequired": false,
    "weight": 20000,
    "weightUnit": "lbs",
    "size": 40,
    "pallets": 24,
    "isLoadExpired": false,
    "senderEmail": "broker@example.com",
    "emailContentFromBroker": "Email body content for the load..."
  }}
}}

Example response for 'broker_setup_request':
{{
  "classification": "broker_setup_request",
  "extracted_details": {{
    "comments": "Please complete our setup packet to get started.",
    "priority": "high",
    "brokerEmail": "onboarding@brokerage.com",
    "brokerCompany": "Efficient Brokers LLC",
    "setupLink": "https://brokerage.com/setup/123xyz",
    "senderEmail": "jane.doe@brokerage.com",
    "emailContentFromBroker": "Hi Carrier, We'd love to work with you. Please complete our setup packet here: https://brokerage.com/setup/123xyz - Jane, Efficient Brokers LLC"
  }}
}}

Example for 'negotiation_bidding':
{{
  "classification": "negotiation_bidding",
  "extracted_details": {{
    "load_id": "DALATL7789",
    "counter_offer_usd": 2800,
    "original_offer_usd": 2500,
    "reason_for_counter": "current market conditions and urgency",
    "carrier_name": "Sarah (Carrier Co.)",
    "senderEmail": "sarah@carrierco.com",
    "emailContentFromBroker": "Hi John, Regarding Load ID: DALATL7789 (Dallas to Atlanta)..."
  }}
}}

Example for 'covered_load' (CRITICAL EXAMPLES):
{{
  "classification": "covered_load",
  "extracted_details": {{
    "load_id_mentioned": null,
    "reason_if_stated": "Load covered",
    "follow_up_mentioned": "Rate confirmation will be sent shortly",
    "senderEmail": "cheska.malana@welcompanies.com",
    "emailContentFromBroker": "Hi Damodar,\\n\\nThanks for confirming.\\n\\nSending over the Rate Confirmation shortly.\\n\\nBest,\\nCheska"
  }}
}}

**ADDITIONAL COVERED_LOAD EXAMPLES:**
- Subject: "Re: Offer – Load Covered" → MUST be 'covered_load'
- Subject: "Load Covered - TX to CA" → MUST be 'covered_load'  
- Subject: "Offer is Covered" → MUST be 'covered_load'
- Subject: "Position Filled" → MUST be 'covered_load'

For classification 'other', 'extracted_details' can be a general key-value pair extraction of any logistics-relevant information found (e.g., contact_person, reference_number, general_inquiry_topic, senderEmail, emailContentFromBroker). If no specific details are found, 'extracted_details' can be an empty object {{}}.
"""

    def _extract_email_from_from_field(self, from_field: str) -> Optional[str]:
        """
        Extract email address from the 'from' field.

        Examples:
        - "Damodar <damodar.advikai@gmail.com>" -> "damodar.advikai@gmail.com"
        - "damodar.advikai@gmail.com" -> "damodar.advikai@gmail.com"
        - "Name Only" -> None

        Args:
            from_field: The 'from' field value from the email

        Returns:
            Extracted email address or None if not found
        """
        if not from_field:
            return None

        # Pattern to match email addresses within angle brackets or standalone
        email_pattern = r'<([^>]+@[^>]+)>|([^\s<>]+@[^\s<>]+)'

        match = re.search(email_pattern, from_field)
        if match:
            # Return the first non-None group
            return match.group(1) or match.group(2)

        return None

    def _post_process_load_offer_details(self, extracted_details_from_api: Dict[str, Any], current_date_iso: Optional[str] = None) -> Dict[str, Any]:
        """Ensures the extracted_details for a load_offer conforms to the schema, adding null for missing keys and correcting dates to a fixed year."""
        processed_details = {}
        forced_year = 2025

        date_keys_to_check = ["pickupTimeExact", "pickupTimeWindowStart", "pickupTimeWindowEnd",
                              "dropoffTimeExact", "dropoffTimeWindowStart", "dropoffTimeWindowEnd"]

        # Handle simple date fields (pickupDate, deliveryDate) - keep as date strings, don't add time
        simple_date_keys = ["pickupDate", "deliveryDate"]

        for key in LOAD_OFFER_SCHEMA_KEYS:
            value = extracted_details_from_api.get(key)

            if key in date_keys_to_check and value:
                try:
                    dt_obj = datetime.fromisoformat(value.replace('Z', ''))
                    if dt_obj.year != forced_year:
                        dt_obj = dt_obj.replace(year=forced_year)
                        logger.info(
                            f"Forced year to {forced_year} for '{key}' in 'load_offer'. Original: '{value}', updated: '{dt_obj.isoformat()}Z'.")
                    processed_details[key] = dt_obj.isoformat() + "Z"
                except ValueError:
                    logger.warning(
                        f"Could not parse date string '{value}' for key '{key}' in 'load_offer'. Setting to null.")
                    processed_details[key] = None
            elif key in simple_date_keys and value:
                # Handle simple date fields - ensure YYYY-MM-DD format
                try:
                    # Try to parse the date and reformat to YYYY-MM-DD
                    if 'T' in str(value):  # If it's a datetime, extract just the date part
                        dt_obj = datetime.fromisoformat(value.replace('Z', ''))
                        processed_details[key] = dt_obj.strftime('%Y-%m-%d')
                    else:
                        # Assume it's already a date string, validate and reformat
                        dt_obj = datetime.strptime(value, '%Y-%m-%d')
                        processed_details[key] = dt_obj.strftime('%Y-%m-%d')
                except ValueError:
                    logger.warning(
                        f"Could not parse date string '{value}' for key '{key}' in 'load_offer'. Setting to null.")
                    processed_details[key] = None
            elif key == 'stops' and isinstance(value, list):
                # Process stops with the new structure
                processed_stops = []
                for stop_item in value:
                    if isinstance(stop_item, dict):
                        processed_stop = {}
                        # Handle stopDate - keep as simple date string
                        if 'stopDate' in stop_item and stop_item['stopDate']:
                            try:
                                if 'T' in str(stop_item['stopDate']):
                                    stop_dt_obj = datetime.fromisoformat(
                                        stop_item['stopDate'].replace('Z', ''))
                                    processed_stop['stopDate'] = stop_dt_obj.strftime(
                                        '%Y-%m-%d')
                                else:
                                    stop_dt_obj = datetime.strptime(
                                        stop_item['stopDate'], '%Y-%m-%d')
                                    processed_stop['stopDate'] = stop_dt_obj.strftime(
                                        '%Y-%m-%d')
                            except ValueError:
                                logger.warning(
                                    f"Could not parse date string '{stop_item['stopDate']}' for stopDate in load_offer. Setting to null.")
                                processed_stop['stopDate'] = None
                        else:
                            processed_stop['stopDate'] = stop_item.get(
                                'stopDate')

                        # Copy other stop fields
                        for stop_field in ['location', 'palletInfo', 'contactEntityName', 'contactEntityPhone', 'notes', 'direction']:
                            processed_stop[stop_field] = stop_item.get(
                                stop_field)

                        processed_stops.append(processed_stop)
                processed_details[key] = processed_stops
            elif key == 'pickupTimeType':
                # Ensure pickupTimeType is valid
                if value and value.upper() in ['EXACT', 'RANGE']:
                    processed_details[key] = value.upper()
                else:
                    processed_details[key] = None
            elif key == 'dropoffTimeType':
                # Ensure dropoffTimeType is valid
                if value and value.upper() in ['EXACT', 'RANGE']:
                    processed_details[key] = value.upper()
                else:
                    processed_details[key] = None
            elif key == 'totalCommodityCount':
                # Default to 1 if not specified
                processed_details[key] = value if value is not None else 1
            elif key == 'weightUnit':
                # Default to 'lbs' if not specified and weight is provided
                if value:
                    processed_details[key] = value
                elif processed_details.get('weight') is not None or processed_details.get('weightLbs') is not None:
                    processed_details[key] = 'lbs'
                else:
                    processed_details[key] = None
            else:
                processed_details[key] = value

        # Ensure all schema keys are present
        for schema_key in LOAD_OFFER_SCHEMA_KEYS:
            if schema_key not in processed_details:
                if schema_key == 'totalCommodityCount':
                    processed_details[schema_key] = 1
                elif schema_key == 'weightUnit' and (processed_details.get('weight') is not None or processed_details.get('weightLbs') is not None):
                    processed_details[schema_key] = 'lbs'
                else:
                    processed_details[schema_key] = None

        return processed_details

    def _post_process_load_details(self, extracted_details_from_api: Dict[str, Any], current_date_iso: Optional[str] = None) -> Dict[str, Any]:
        """Ensures the extracted_details for a 'load' conforms to its schema, adding null for missing keys and correcting dates."""
        processed_details = {}
        forced_year = 2025

        date_keys_to_check = ["pickupTimeExact", "pickupTimeWindowStart", "pickupTimeWindowEnd",
                              "dropoffTimeExact", "dropoffTimeWindowStart", "dropoffTimeWindowEnd"]

        # Handle simple date fields (pickupDate, deliveryDate) - keep as date strings, don't add time
        simple_date_keys = ["pickupDate", "deliveryDate"]

        for key in LOAD_SCHEMA_KEYS:
            value = extracted_details_from_api.get(key)

            if key in date_keys_to_check and value:
                try:
                    dt_obj = datetime.fromisoformat(value.replace('Z', ''))
                    if dt_obj.year != forced_year:
                        dt_obj = dt_obj.replace(year=forced_year)
                        logger.info(
                            f"Forced year to {forced_year} for '{key}' in 'load'. Original: '{value}', updated: '{dt_obj.isoformat()}Z'.")
                    processed_details[key] = dt_obj.isoformat() + "Z"
                except ValueError:
                    logger.warning(
                        f"Could not parse date string '{value}' for key '{key}' in 'load'. Setting to null.")
                    processed_details[key] = None
            elif key in simple_date_keys and value:
                # Handle simple date fields - ensure YYYY-MM-DD format
                try:
                    if 'T' in str(value):  # If it's a datetime, extract just the date part
                        dt_obj = datetime.fromisoformat(value.replace('Z', ''))
                        processed_details[key] = dt_obj.strftime('%Y-%m-%d')
                    else:
                        # Assume it's already a date string, validate and reformat
                        dt_obj = datetime.strptime(value, '%Y-%m-%d')
                        processed_details[key] = dt_obj.strftime('%Y-%m-%d')
                except ValueError:
                    logger.warning(
                        f"Could not parse date string '{value}' for key '{key}' in 'load'. Setting to null.")
                    processed_details[key] = None
            elif key == 'stops' and isinstance(value, list):
                # Process stops with the new structure
                processed_stops = []
                for stop_item in value:
                    if isinstance(stop_item, dict):
                        processed_stop = {}
                        # Handle stopDate - keep as simple date string
                        if 'stopDate' in stop_item and stop_item['stopDate']:
                            try:
                                if 'T' in str(stop_item['stopDate']):
                                    stop_dt_obj = datetime.fromisoformat(
                                        stop_item['stopDate'].replace('Z', ''))
                                    processed_stop['stopDate'] = stop_dt_obj.strftime(
                                        '%Y-%m-%d')
                                else:
                                    stop_dt_obj = datetime.strptime(
                                        stop_item['stopDate'], '%Y-%m-%d')
                                    processed_stop['stopDate'] = stop_dt_obj.strftime(
                                        '%Y-%m-%d')
                            except ValueError:
                                logger.warning(
                                    f"Could not parse date string '{stop_item['stopDate']}' for stopDate in load. Setting to null.")
                                processed_stop['stopDate'] = None
                        else:
                            processed_stop['stopDate'] = stop_item.get(
                                'stopDate')

                        # Copy other stop fields
                        for stop_field in ['location', 'palletInfo', 'contactEntityName', 'contactEntityPhone', 'notes', 'direction']:
                            processed_stop[stop_field] = stop_item.get(
                                stop_field)

                        processed_stops.append(processed_stop)
                processed_details[key] = processed_stops
            elif key == 'pickupTimeType':
                # Ensure pickupTimeType is valid
                if value and value.upper() in ['EXACT', 'RANGE']:
                    processed_details[key] = value.upper()
                else:
                    processed_details[key] = None
            elif key == 'dropoffTimeType':
                # Ensure dropoffTimeType is valid
                if value and value.upper() in ['EXACT', 'RANGE']:
                    processed_details[key] = value.upper()
                else:
                    processed_details[key] = None
            elif key == 'totalCommodityCount':
                # Default to 1 if not specified
                processed_details[key] = value if value is not None else 1
            elif key == 'weightUnit':
                # Default to 'lbs' if not specified and weight is provided
                if value:
                    processed_details[key] = value
                elif processed_details.get('weight') is not None or processed_details.get('weightLbs') is not None:
                    processed_details[key] = 'lbs'
                else:
                    processed_details[key] = None
            else:
                processed_details[key] = value

        # Ensure all schema keys are present
        for schema_key in LOAD_SCHEMA_KEYS:
            if schema_key not in processed_details:
                if schema_key == 'totalCommodityCount':
                    processed_details[schema_key] = 1
                elif schema_key == 'weightUnit' and (processed_details.get('weight') is not None or processed_details.get('weightLbs') is not None):
                    processed_details[schema_key] = 'lbs'
                else:
                    processed_details[schema_key] = None

        return processed_details

    def _post_process_broker_setup_details(self, extracted_details_from_api: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures the extracted_details for a 'broker_setup_request' conforms to its schema."""
        processed_details = {}
        for key in BROKER_SETUP_REQUEST_SCHEMA_KEYS:
            value = extracted_details_from_api.get(key)
            if key == "priority":
                # Always set priority to 'high' for broker setup requests
                processed_details[key] = "high"
                if value and value != "high":
                    logger.info(
                        f"Overriding broker_setup_request priority from '{value}' to 'high' (business rule).")
            else:
                processed_details[key] = value

        for schema_key in BROKER_SETUP_REQUEST_SCHEMA_KEYS:
            if schema_key not in processed_details:
                processed_details[schema_key] = None
        return processed_details

    def process_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the intent of an email and extract key details.
        If classified as 'load_offer', 'load', or 'broker_setup_request', ensures extracted_details conform to a specific schema.
        """
        try:
            # Updated to use new payload keys
            subject = email_data.get("subject", "")
            raw_short_preview = email_data.get("snippet", "")
            raw_main_body = email_data.get("body", "")
            current_date_iso_context = email_data.get("date")
            from_field = email_data.get("from", "")

            # Handle attachments information
            attachments = email_data.get("attachments", [])
            attachments_info = ""

            if attachments:
                attachment_names = []
                for attachment in attachments:
                    if isinstance(attachment, dict):
                        filename = attachment.get("filename", "")
                        file_type = attachment.get("type", "")
                        if filename:
                            attachment_names.append(
                                f"{filename} ({file_type})" if file_type else filename)
                    elif isinstance(attachment, str):
                        attachment_names.append(attachment)

                if attachment_names:
                    attachments_info = f"\nAttachments: {', '.join(attachment_names)}"

            # NO CLEANING - just pass everything raw to the AI
            content_to_process = f"Subject: {subject}\nPreview: {raw_short_preview}\nBody: {raw_main_body}{attachments_info}"

            # Only check if we have ANY content at all
            if not subject.strip() and not raw_main_body.strip() and not attachments_info.strip():
                logger.warning("No content at all for email processing.",
                               received_payload_keys=list(email_data.keys()))
                return {"classification": "unknown", "extracted_details": {}, "reason": "No content found in email"}

            user_prompt = self._create_processing_user_prompt(
                content_to_process, current_date_iso_context, from_field)

            response = self.client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt_content
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            api_response_data = json.loads(response.choices[0].message.content)
            classification = api_response_data.get("classification")
            extracted_details_from_api = api_response_data.get(
                "extracted_details", {})

            final_extracted_details = extracted_details_from_api
            if classification == "load_offer":
                final_extracted_details = self._post_process_load_offer_details(
                    extracted_details_from_api, current_date_iso_context)
            elif classification == "load":
                final_extracted_details = self._post_process_load_details(
                    extracted_details_from_api, current_date_iso_context)
            elif classification == "broker_setup_request":
                final_extracted_details = self._post_process_broker_setup_details(
                    extracted_details_from_api)
            elif classification == "covered_load":
                # For covered_load, ensure senderEmail and emailContentFromBroker are added if not present,
                # other details are as-is from LLM.
                if "senderEmail" not in final_extracted_details:
                    # Or extract if possible, but simple for now
                    final_extracted_details["senderEmail"] = None
                if "emailContentFromBroker" not in final_extracted_details:
                    final_extracted_details["emailContentFromBroker"] = None

            result = {
                "classification": classification,
                "extracted_details": final_extracted_details
            }
            # Carry over reason if present (e.g. for 'unknown' or 'error' from LLM)
            if api_response_data.get("reason"):
                result["reason"] = api_response_data.get("reason")

            logger.info(
                "Successfully processed email",
                classification=result.get('classification'),
                details_keys_count=len(result.get(
                    'extracted_details', {}).keys())
            )
            return result

        except Exception as err:
            logger.error("Failed to process email", error=str(
                err), email_subject=email_data.get("subject"))
            return {"classification": "error", "extracted_details": {}, "reason": str(err)}

    def _create_processing_user_prompt(self, email_content: str, current_date_iso: Optional[str] = None, from_field: Optional[str] = None) -> str:
        """Create the user part of the processing prompt for OpenAI.
        Args:
            email_content: The combined text content of the email.
            current_date_iso: Optional current date to provide context to the LLM.
            from_field: Optional 'from' field from the email for sender email extraction.
        Returns:
            Formatted prompt string.
        """
        date_context = ""
        if current_date_iso:
            date_context = f"\n\nFor context, the current date is: {current_date_iso}. Please interpret relative dates like 'today', 'tomorrow', or month/day without a year based on this current date. For example, if the current year is 2025 and the email mentions 'January 13th' for a future event, it should be interpreted as '2026-01-13' if January 13th of 2025 has passed, or '2025-01-13' if it has not yet passed or is the current date. Prefer the closest future date that makes sense."

        from_context = ""
        if from_field:
            from_context = f"\n\nEmail From Field: {from_field}\nPlease extract the sender's email address from this 'from' field and include it in the 'senderEmail' field of your response."

        email_content_context = "\n\nIMPORTANT: For the 'emailContentFromBroker' field, please include the actual email body content (the cleaned message content after removing any forwarding headers). This should be the main message text that contains the load offer or negotiation details."

        return f"""Please classify the intent and extract key details from the following email content according to the rules and formats previously provided in the system message:{date_context}{from_context}{email_content_context}

Email Content:
---
{email_content}
---

Ensure your response is a single valid JSON object."""


# Example usage (optional, for direct testing of this module)
if __name__ == "__main__":
    # from dotenv import load_dotenv
    # load_dotenv(dotenv_path='../../.env') # Adjust path as needed for standalone run

    classifier_extractor = EmailClassifierOpenAI()

    # Updated sample_email_offer to somewhat reflect the new structure for testing
    # Note: The 'date' field is crucial for date interpretation.
    sample_email_offer_new_format = {
        "subject": "URGENT LOAD: Dallas, TX to Atlanta, GA - $2500 - Van",
        "snippet": "Hot load, 40k lbs general freight, pickup today, deliver by tomorrow EOD.",
        "body": "Good morning, We have an urgent FTL shipment from Dallas, TX 75201 to Atlanta, GA 30301. Weight is 40,000 lbs, commodity is general freight. Needs a 53ft Dry Van. Pickup available immediately (today, July 14th) and delivery required by tomorrow, July 15th, end of day. We are offering $2500. Please call John at 555-123-4567 or reply to this email if you can cover this. Load ID: DALATL7789",
        "date": "2025-07-14T08:00:00Z"  # Example: current date for context
    }
    print("--- Processing Load Offer Email (New Format Example) ---")
    result = classifier_extractor.process_email(sample_email_offer_new_format)
    print(json.dumps(result, indent=2))

    # Example of a forwarded email in the new format
    forwarded_email_payload = {
        "id": "196c84fc93581045",
        "threadId": "196c84fc93581045",
        "subject": "Fwd: Asheboro NC to Commerce TX",
        "from": "Sans Kark <skark236@gmail.com>",
        "to": "\"advikai757@gmail.com\" <advikai757@gmail.com>",
        # Date of the outermost forward, or when it's processed
        "date": "2025-01-21T14:58:00Z",
        "snippet": "---------- Forwarded message --------- From: Sudarshan Gaire <gairefreightline@gmail.com> Date: Tue, Jan 21, 2025 at 2:58 PM Subject: Fwd: Asheboro NC to Commerce TX To: <skark236@gmail.com",
        "body": "---------- Forwarded message ---------\r\nFrom: Sudarshan Gaire <gairefreightline@gmail.com>\r\nDate: Tue, Jan 21, 2025 at 2:58 PM\r\nSubject: Fwd: Asheboro NC to Commerce TX\r\nTo: <skark236@gmail.com>\r\n\r\n\r\n\r\n\r\n---------- Forwarded message ---------\r\nFrom: Cheska Malana <cheska@abostonlogistics.com>\r\nDate: Fri, Jan 17, 2025 at 9:28 PM\r\nSubject: Asheboro NC to Commerce TX\r\nTo:\r\n\r\n\r\nHello\r\n\r\nChecking if you have a 53ft Dry Van on Monday\r\n\r\nWould like to offer this load\r\n\r\nAsheboro NC to Commerce TX\r\nPU 1/20 0700-1500 need to set appt\r\nDO 1/21 09:00 strict appt\r\n\r\nWeight 44000\r\n3rd party Scale ticket are required both Empty and Heavy & should be\r\n20-30miles from the PU loc (we can provide location of scaling house)\r\nThis is a double blind shipment - BOL will be provided upon booking\r\n\r\nGive us your best rate\r\n\r\n\r\n\r\n\r\n[image: WEL Companies] <https://www.welcompanies.com/>\r\n Cheska Malana\r\n Operations Manager\r\n(623) 267-3928\r\n"
    }
    print("\n--- Processing Forwarded Email (New Format Example) ---")
    result_fwd = classifier_extractor.process_email(forwarded_email_payload)
    print(json.dumps(result_fwd, indent=2))

    # You can add other sample emails (negotiation, confirmation, other) here
    # using the new payload structure if you want to test them directly.
    # For example:
    # sample_email_negotiation_new_format = {
    #     "subject": "RE: Your offer for DALATL7789",
    #     "snippet": "Thanks for the offer. Can you go up to $2700? Current capacity is tight.",
    #     "body": "Hi John, Regarding Load ID: DALATL7789 (Dallas to Atlanta), we can cover it but $2500 is a bit low for the current market conditions and urgency. Would you consider $2800? Our best driver is available. Let me know. - Sarah (Carrier Co.)",
    #     "date": "2025-07-14T10:00:00Z"
    # }
    # print("\n--- Processing Negotiation Email (New Format Example) ---")
    # result_neg = classifier_extractor.process_email(sample_email_negotiation_new_format)
    # print(json.dumps(result_neg, indent=2))
