# """
# Optimized Email Intent Classifier with improved robustness and edge case handling.
# """
#  # services/email_intent_classifier.py
# from email import parser
# import json
# import re
# from typing import List, Optional, Dict
# from openai import OpenAI, OpenAIError
# from pydantic import BaseModel, Field, ValidationError
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # --- Pydantic Models ---


# class EmailAttachment(BaseModel):
#     """Model for email attachment details."""
#     filename: str = Field(..., description="Name of the attachment file")
#     mimeType: str = Field(..., description="MIME type of the attachment")
#     size: int = Field(..., description="Size of the attachment in bytes")
#     attachmentId: str = Field(...,
#                               description="Unique identifier for the attachment")


# class LoadBidEmailDetails(BaseModel):
#     rate: Optional[str] = Field(
#         None, description="Rate mentioned in negotiation emails")
#     conditions: Optional[str] = Field(
#         None, description="Any price-related or logistical conditions")
#     questions: Optional[List[str]] = Field(
#         default=None, description="List of information requests")
#     accepted_or_rejected_rate: Optional[str] = Field(
#         None, description="Rate being accepted/rejected")
#     confirmed_rate: Optional[str] = Field(
#         None, description="Final confirmed rate")
#     pickup_details: Optional[str] = Field(
#         None, description="Pickup information")
#     delivery_details: Optional[str] = Field(
#         None, description="Delivery information")
#     load_id: Optional[str] = Field(None, description="Load or tracking number")
#     broker_email: Optional[str] = Field(
#         None, description="Broker email address")
#     broker_company: Optional[str] = Field(
#         None, description="Broker company name")
#     setup_link: Optional[str] = Field(
#         None, description="Setup/registration link")


# class LoadBidEmailClassificationRequest(BaseModel):
#     email: str = Field(...,
#                        description="The email content received from the broker")
#     attachments: Optional[List[EmailAttachment]] = Field(
#         None, description="Optional list of email attachments")


# class LoadBidEmailClassificationResponseStructured(BaseModel):
#     intent: str = Field(..., description="Classification intent")
#     details: LoadBidEmailDetails
#     confidence: Optional[str] = Field(
#         None, description="Confidence level of classification")


# # --- Optimized Prompt ---
# OPTIMIZED_LOAD_BID_EMAIL_CLASSIFIER_PROMPT = """
# You are an expert email intent classifier for load bid communications in logistics.

# **CRITICAL INSTRUCTIONS:**
# 1. You MUST respond with ONLY a valid JSON object - no additional text, explanations, or markdown
# 2. Handle ALL types of input: complete emails, fragments, single words, informal text, typos, or unclear messages
# 3. When input is ambiguous, unclear, or lacks context, make your best assessment and use the "confidence" field

# **Intent Categories:**
# - **negotiation**: Discussing/changing terms, rates, conditions (counter-offers, price discussions)
# - **information_seeking**: Requesting details (equipment, availability, load specs, non-price questions)
# - **bid_acceptance**: Agreeing to carrier's rate (explicit/implicit acceptance, "sounds good", "let's do it")
# - **bid_rejection**: Declining the offer ("no thanks", "too high", "found another carrier")
# - **rate_confirmation**: Official rate confirmation document/email (check attachments with "confirmation", "RC", "BOL", "rate con" in filename)
# - **broker_setup**: Setup/onboarding/registration requests
# - **unclear**: Use ONLY when input is completely unintelligible or spam-like (random characters, no discernible meaning)

# **Key Rules:**
# - "Okay, $X works" after carrier proposed $X → bid_acceptance
# - "Can you do $Y?" after carrier proposed $X → negotiation
# - Attachments with "confirmation", "RC", "BOL", "rate_con" → rate_confirmation (high priority)
# - Simple agreement text ("confirmed", "okay") without formal attachment → bid_acceptance
# - For single words/fragments: infer most likely intent (e.g., "yes" → bid_acceptance, "more info?" → information_seeking)
# - For typos/informal text: interpret intent based on context clues

# **Confidence Levels:**
# - "high": Clear, unambiguous intent
# - "medium": Reasonable inference from context
# - "low": Ambiguous or minimal information

# **Edge Cases:**
# - Empty/whitespace-only email: intent="unclear", confidence="low"
# - Single word responses: infer from word meaning
# - Casual/slang language: interpret pragmatically
# - Mixed intents: choose the PRIMARY intent

# **Response Format (JSON only, no markdown):**
# {
#   "intent": "negotiation|information_seeking|bid_acceptance|bid_rejection|rate_confirmation|broker_setup|unclear",
#   "details": {
#     "rate": "rate if negotiation, else null",
#     "conditions": "conditions if applicable, else null",
#     "questions": ["question1", "question2"] or null,
#     "accepted_or_rejected_rate": "specific rate if acceptance/rejection, else null",
#     "confirmed_rate": "rate if rate_confirmation, else null",
#     "pickup_details": "pickup info if rate_confirmation, else null",
#     "delivery_details": "delivery info if rate_confirmation, else null",
#     "load_id": "load ID if available, else null",
#     "broker_email": "email if broker_setup, else null",
#     "broker_company": "company if broker_setup, else null",
#     "setup_link": "link if broker_setup, else null"
#   },
#   "confidence": "high|medium|low"
# }
# """


# class IntentClassifierService:
#     def __init__(self):
#         """Initialize the classifier with OpenAI client."""
#         self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         if not os.getenv("OPENAI_API_KEY"):
#             raise ValueError("OPENAI_API_KEY environment variable not set")

#     def clean_raw_response(self, raw_response: str) -> str:
#         """
#         Enhanced JSON extraction with multiple fallback strategies.

#         Args:
#             raw_response: Raw OpenAI response

#         Returns:
#             Cleaned JSON string

#         Raises:
#             Exception: If valid JSON cannot be extracted
#         """
#         if not raw_response or not raw_response.strip():
#             raise ValueError("Empty response from OpenAI")

#         raw_response = raw_response.strip()

#         # Strategy 1: Remove markdown code blocks
#         json_match = re.search(
#             r'```(?:json)?\s*(.*?)\s*```', raw_response, re.DOTALL)
#         if json_match:
#             potential_json = json_match.group(1).strip()
#         else:
#             # Strategy 2: Find outermost JSON object
#             start_brace = raw_response.find('{')
#             end_brace = raw_response.rfind('}')

#             if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
#                 potential_json = raw_response[start_brace:end_brace + 1]
#             else:
#                 # Strategy 3: Try to find JSON array
#                 start_bracket = raw_response.find('[')
#                 end_bracket = raw_response.rfind(']')

#                 if start_bracket != -1 and end_bracket != -1:
#                     potential_json = raw_response[start_bracket:end_bracket + 1]
#                 else:
#                     raise Exception(
#                         f"No valid JSON structure found in response: '{raw_response[:200]}'")

#         # Validate JSON
#         try:
#             json.loads(potential_json)
#             return potential_json
#         except json.JSONDecodeError as e:
#             raise Exception(
#                 f"Malformed JSON: {e}. Content: '{potential_json[:200]}'")
    
#     def generate_unclear_human_reply(self, original_email: str) -> str:
#         """
#         Let the LLM generate a contextual clarification response.
#         It must stay within load/logistics context but remain natural.
#         """
#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 temperature=0.4,
#                 max_tokens=120,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": (
#                             "You are a friendly freight dispatcher. "
#                             "Your job is to respond politely and naturally "
#                             "when the broker's message has unclear intent. "
#                             "Stay in the context of loads, rates, availability, "
#                             "and freight communication — but DO NOT force the topic unnaturally. "
#                             "Match the broker's tone. Keep replies short and natural."
#                         )
#                     },
#                     {
#                         "role": "user",
#                         "content": (
#                             f"The broker wrote: \"{original_email}\"\n"
#                             "Generate a friendly, human-like clarification response that keeps the context professional."
#                         )
#                     }
#                 ]
#             )
#             return response.choices[0].message.content.strip()
#         except Exception:
#             return "Just to make sure I assist correctly — could you clarify what you need regarding the load?"
    
#     def classify_load_bid_email_intent(
#         self,
#         input_email: str,
#         attachments: Optional[List] = None,
#         conversation_history: Optional[List[Dict[str, str]]] = None
#     ) -> LoadBidEmailClassificationResponseStructured:
#         """
#         Classify email intent with enhanced error handling and edge case support.

#         Args:
#             input_email: Email content to classify
#             attachments: Optional list of attachments
#             conversation_history: Optional list of previous conversation turns

#         Returns:
#             Structured classification response

#         Raises:
#             Exception: If classification fails after retries
#         """
#         raw_response = None
#         cleaned_response = None

#         try:
#             # Validate and sanitize input
#             if not input_email or not input_email.strip():
#                 input_email = "[Empty or whitespace-only email]"

#             # Truncate very long emails to avoid token limits
#             if len(input_email) > 8000:
#                 input_email = input_email[:8000] + \
#                     "\n[...truncated for length...]"

#             # Build prompt with attachment info
#             prompt = OPTIMIZED_LOAD_BID_EMAIL_CLASSIFIER_PROMPT + "\n\n"
#             if conversation_history:
#                 prompt += "**Conversation History (for context):**\n"
#                 for entry in conversation_history[-3:]:  # Limit to last 3 turns
#                     role = "Broker" if entry.get(
#                         "negotiationDirection") == "incoming" else "You (Carrier)"
#                     prompt += f'- {role}: "{entry.get("negotiationRawEmail", "")[:200]}"\n'
#                 prompt += "\n"
#                 prompt += "**Your Task:** Based on the history and the latest email, determine the current intent. If the last message was a question from you, and this email seems to answer it, the intent is likely 'negotiation' to continue the process.\n\n"


#             if attachments:
#                 prompt += "**Attachments:**\n"
#                 for att in attachments:
#                     if isinstance(att, dict):
#                         filename = att.get('filename', 'Unknown')
#                         mime_type = att.get('mimeType', 'Unknown')
#                         size = att.get('size', 'Unknown')
#                     else:
#                         filename = getattr(att, 'filename', 'Unknown')
#                         mime_type = getattr(att, 'mimeType', 'Unknown')
#                         size = getattr(att, 'size', 'Unknown')
#                     prompt += f"- {filename} ({mime_type}, {size} bytes)\n"
#                 prompt += "\n"

#             prompt += f"**Email Content:**\n{input_email}"

#             # API call with retry logic
#             max_retries = 1
#             for attempt in range(max_retries):
#                 try:
#                     response = self.client.chat.completions.create(
#                         model="gpt-4o-mini",  # More reliable than 3.5-turbo
#                         messages=[
#                             {
#                                 "role": "system",
#                                 "content": "You are an expert email classifier. Respond ONLY with valid JSON. No markdown, no explanations, no additional text. Provide the intent based on the entire context."
#                             },
#                             {"role": "user", "content": prompt}
#                         ],
#                         temperature=0.2,  # Lower for more consistent output
#                         max_tokens=2000,
#                         # Force JSON mode
#                         response_format={"type": "json_object"}
#                     )

#                     if not response.choices or not response.choices[0].message.content:
#                         raise ValueError("Empty response from OpenAI API")

#                     raw_response = response.choices[0].message.content
#                     break

#                 except OpenAIError as api_error:
#                     if attempt == max_retries - 1:
#                         raise Exception(
#                             f"OpenAI API failed after {max_retries} attempts: {api_error}")
#                     print(
#                         f"Retry {attempt + 1}/{max_retries} after API error: {api_error}")

#             if not raw_response:
#                 raise ValueError(
#                     "No response received from OpenAI after retries")

#             # Clean and parse response
#             cleaned_response = self.clean_raw_response(raw_response)
#             parsed = json.loads(cleaned_response)

#             # Ensure required fields with defaults
#             if "intent" not in parsed:
#                 parsed["intent"] = "unclear"
#             if "details" not in parsed:
#                 parsed["details"] = {}
#             if "confidence" not in parsed:
#                 parsed["confidence"] = "low"

#             # Validate with Pydantic
#             response_obj = LoadBidEmailClassificationResponseStructured(
#                 **parsed)

#             print(
#                 f"✓ Successfully classified: {response_obj.intent} (confidence: {response_obj.confidence})")
#             return response_obj
        

#         except ValidationError as e:
#             print(f"\n{'='*60}\nPYDANTIC VALIDATION ERROR:\n{e}\n{'='*60}\n")
#             # Return fallback response
#             return LoadBidEmailClassificationResponseStructured(
#                 intent="unclear",
#                 details=LoadBidEmailDetails(
#                     rate=None,
#                     conditions=None,
#                     questions=[self.generate_unclear_human_reply(input_email)],
#                     accepted_or_rejected_rate=None,
#                     confirmed_rate=None,
#                     pickup_details=None,
#                     delivery_details=None,
#                     load_id=None,
#                     broker_email=None,
#                     broker_company=None,
#                     setup_link=None
#                 ),
#                 confidence="low"
#             )

#         except json.JSONDecodeError as e:
#             print(
#                 f"\n{'='*60}\nJSON PARSING ERROR:\n{e}\nRaw: {raw_response}\n{'='*60}\n")
#             raise Exception(f"Failed to parse OpenAI response as JSON: {e}")

#         except Exception as e:
#             print(
#                 f"\n{'='*60}\nCLASSIFICATION ERROR:\n{e}\nRaw: {raw_response}\n{'='*60}\n")
#             raise


# # Example usage
# if __name__ == "__main__":
#     service = IntentClassifierService()

#     # Test cases including edge cases
#     test_emails = [
#         "Can you do $1500 for this load?",
#         "yes",
#         "Sounds good, let's proceed",
#         "",
#         "rate?",
#         "See attached rate confirmation",
#         "What's your MC number?",
#         "asdfghjkl",  # Random text
#         "No thanks, we found another carrier"
#     ]

#     for email in test_emails:
#         print(f"\nTesting: '{email}'")
#         result = service.classify_load_bid_email_intent(email)
#         print(f"Result: {result.intent} ({result.confidence})")

"""
Enhanced Email Intent Classifier for Logistics/Freight Communications
Version 2.0 - Improved context awareness, conversation flow understanding, and edge case handling

Key Improvements:
1. Better conversation history integration
2. Logistics-specific intent patterns
3. Multi-turn conversation understanding
4. Robust rate/price extraction
5. Broker-carrier communication dynamics
"""

import json
import re
from typing import List, Optional, Dict, Any, Tuple
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError, field_validator
import os
from dotenv import load_dotenv
from enum import Enum

load_dotenv()


# --- Enums for Type Safety ---
class IntentType(str, Enum):
    NEGOTIATION = "negotiation"
    INFORMATION_SEEKING = "information_seeking"
    BID_ACCEPTANCE = "bid_acceptance"
    BID_REJECTION = "bid_rejection"
    RATE_CONFIRMATION = "rate_confirmation"
    BROKER_SETUP = "broker_setup"
    LOAD_DETAILS = "load_details"
    AVAILABILITY_CHECK = "availability_check"
    BOOKING_REQUEST = "booking_request"
    UNCLEAR = "unclear"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# --- Pydantic Models ---
class EmailAttachment(BaseModel):
    filename: str = Field(..., description="Name of the attachment file")
    mimeType: str = Field(..., description="MIME type of the attachment")
    size: int = Field(..., description="Size of the attachment in bytes")
    attachmentId: str = Field(..., description="Unique identifier for the attachment")


class LoadBidEmailDetails(BaseModel):
    rate: Optional[str] = Field(None, description="Rate mentioned (e.g., '$1500', '1500')")
    conditions: Optional[str] = Field(None, description="Price-related or logistical conditions")
    questions: Optional[List[str]] = Field(default=None, description="Questions being asked")
    accepted_or_rejected_rate: Optional[str] = Field(None, description="Rate being accepted/rejected")
    confirmed_rate: Optional[str] = Field(None, description="Final confirmed rate")
    pickup_details: Optional[str] = Field(None, description="Pickup location, date, time")
    delivery_details: Optional[str] = Field(None, description="Delivery location, date, time")
    load_id: Optional[str] = Field(None, description="Load or tracking number")
    broker_email: Optional[str] = Field(None, description="Broker email address")
    broker_company: Optional[str] = Field(None, description="Broker company name")
    setup_link: Optional[str] = Field(None, description="Setup/registration link")
    equipment_type: Optional[str] = Field(None, description="Required equipment type")
    weight: Optional[str] = Field(None, description="Load weight")
    commodity: Optional[str] = Field(None, description="Type of freight/commodity")
    miles: Optional[str] = Field(None, description="Distance/miles")
    
    @field_validator('rate', 'accepted_or_rejected_rate', 'confirmed_rate', mode='before')
    @classmethod
    def normalize_rate(cls, v):
        if v is None:
            return None
        # Extract numeric rate from various formats
        if isinstance(v, str):
            # Remove common prefixes and clean up
            v = v.strip()
            # Match patterns like $1500, 1500, $1,500, 1500.00
            match = re.search(r'\$?\s*([\d,]+(?:\.\d{2})?)', v)
            if match:
                return f"${match.group(1).replace(',', '')}"
        return v


class LoadBidEmailClassificationResponse(BaseModel):
    intent: str = Field(..., description="Classification intent")
    details: LoadBidEmailDetails
    confidence: str = Field(default="medium", description="Confidence level")
    reasoning: Optional[str] = Field(None, description="Brief explanation of classification")
    suggested_action: Optional[str] = Field(None, description="Recommended next action")
    
    @field_validator('intent', mode='before')
    @classmethod
    def validate_intent(cls, v):
        valid_intents = [e.value for e in IntentType]
        if v and v.lower() in valid_intents:
            return v.lower()
        return IntentType.UNCLEAR.value


# --- Conversation Context Analyzer ---
class ConversationContextAnalyzer:
    """Analyzes conversation history to provide context for intent classification."""
    
    @staticmethod
    def extract_context(history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract relevant context from conversation history."""
        context = {
            "turn_count": len(history) if history else 0,
            "last_carrier_action": None,
            "last_broker_action": None,
            "mentioned_rates": [],
            "pending_questions": [],
            "negotiation_stage": "initial",
            "has_load_details": False,
            "awaiting_response_to": None
        }
        
        if not history:
            return context
        
        rate_pattern = re.compile(r'\$\s*([\d,]+(?:\.\d{2})?)')
        question_pattern = re.compile(r'\?|what|when|where|how|can you|do you|is there|are there', re.I)
        
        for i, entry in enumerate(history):
            direction = entry.get("negotiationDirection", "")
            content = entry.get("negotiationRawEmail", "")[:500]  # Limit content length
            
            # Extract rates mentioned
            rates = rate_pattern.findall(content)
            for rate in rates:
                context["mentioned_rates"].append({
                    "rate": f"${rate}",
                    "from": "carrier" if direction == "outgoing" else "broker",
                    "turn": i
                })
            
            # Track last actions
            if direction == "outgoing":
                context["last_carrier_action"] = content[:200]
                # Check if carrier asked questions
                if question_pattern.search(content):
                    context["pending_questions"].append(content[:100])
                    context["awaiting_response_to"] = "carrier_question"
            else:
                context["last_broker_action"] = content[:200]
                context["awaiting_response_to"] = None  # Broker responded
            
            # Check for load details
            if any(kw in content.lower() for kw in ['pickup', 'delivery', 'miles', 'weight', 'commodity']):
                context["has_load_details"] = True
        
        # Determine negotiation stage
        if len(context["mentioned_rates"]) > 2:
            context["negotiation_stage"] = "active_negotiation"
        elif context["has_load_details"]:
            context["negotiation_stage"] = "details_exchanged"
        elif context["turn_count"] > 0:
            context["negotiation_stage"] = "initial_contact"
        
        return context


# --- Enhanced Prompt Builder ---
class PromptBuilder:
    """Builds context-aware prompts for the classifier."""
    
    SYSTEM_PROMPT = """You are an expert email intent classifier specialized in freight/logistics broker-carrier communications.

CRITICAL: You must respond with ONLY valid JSON. No markdown, no explanations, no code blocks.

## UNDERSTANDING BROKER-CARRIER COMMUNICATION FLOW:

1. **Initial Contact**: Broker sends load details or carrier responds to load board posting
2. **Information Exchange**: Questions about equipment, availability, load specifics
3. **Rate Negotiation**: Back-and-forth on pricing until agreement or rejection
4. **Acceptance/Rejection**: Clear decision on the load
5. **Confirmation**: Rate confirmation document, booking details

## INTENT CATEGORIES (in order of specificity):

### load_details
Broker providing load information (routes, dates, times, weight, commodity). Look for:
- Pickup/delivery addresses, dates, times
- Load specifications (weight, commodity, equipment)
- This is NOT negotiation - it's information provision
- Example: "Pickup: Dallas TX, Dec 5, 8:00 AM. Delivery: Houston TX, Dec 5, 4:00 PM"

### availability_check  
Broker asking if carrier can handle the load. Look for:
- "Is this load available?"
- "Can you take this?"
- "Are you available for..."
- "Do you have a truck for..."

### booking_request
Broker wants to book/confirm the load. Look for:
- "Let's book this"
- "I'd like to set this up"
- "Ready to dispatch"
- Request for carrier packet/setup docs

### bid_acceptance
Clear agreement to a rate or terms. MUST have explicit acceptance language:
- "Sounds good", "Works for me", "Agreed", "Let's do it", "Confirmed"
- "OK with $X" after carrier proposed $X
- Simple "yes" or "ok" ONLY if responding to a rate proposal

### bid_rejection
Clear decline of offer. Look for:
- "No thanks", "Can't do it", "Too low/high"
- "Found another carrier", "Going with someone else"
- "Not interested"

### negotiation
Active discussion about rates/terms. Look for:
- Counter-offers: "Can you do $X instead?"
- Rate discussion: "That rate is too high/low"
- Conditional acceptance: "I can do $X if..."
- Response to rate questions

### rate_confirmation
Official confirmation document. HIGH PRIORITY indicators:
- Attachment filename contains: "rate con", "RC", "confirmation", "BOL"
- Email mentions "attached confirmation" or "rate confirmation"
- Official booking confirmation language

### information_seeking
Questions not related to rates. Look for:
- Equipment questions: "What type of trailer?"
- Availability: "When can you pick up?"
- Logistics: "Do you have TWIC card?"
- Documentation: "Can you send your MC?"

### broker_setup
Carrier onboarding/setup requests. Look for:
- "Please complete carrier packet"
- "Need to set you up in our system"
- Registration/setup links
- Request for insurance/authority docs

### unclear
Use ONLY when:
- Message is completely unintelligible
- No freight/logistics context at all
- Random characters or spam

## CONTEXT-AWARE CLASSIFICATION RULES:

1. **After carrier asks a question** → Broker's response is likely answering that question
   - If carrier asked about rate → response is likely negotiation
   - If carrier asked about details → response is likely load_details or information_seeking

2. **Repeated load details** → Usually means broker is providing requested info (load_details), NOT unclear

3. **Short responses** ("yes", "ok", "sounds good"):
   - After rate proposal → bid_acceptance
   - After question about availability → availability confirmation (bid_acceptance)
   - Without clear context → information_seeking (ask for clarification)

4. **Messages with emoji/formatting** → Focus on the actual content, ignore styling

5. **Attachments override text**:
   - Rate confirmation attachment + any text → rate_confirmation
   - Setup packet attachment → broker_setup

## OUTPUT FORMAT (JSON only):
{
  "intent": "one of the intent categories above",
  "details": {
    "rate": "extracted rate or null",
    "conditions": "any conditions mentioned or null",
    "questions": ["list of questions"] or null,
    "accepted_or_rejected_rate": "specific rate if acceptance/rejection or null",
    "confirmed_rate": "rate if confirmation or null",
    "pickup_details": "pickup info or null",
    "delivery_details": "delivery info or null",
    "load_id": "load/tracking number or null",
    "broker_email": "email if mentioned or null",
    "broker_company": "company name or null",
    "setup_link": "link if broker_setup or null",
    "equipment_type": "equipment type or null",
    "weight": "weight or null",
    "commodity": "commodity type or null",
    "miles": "distance or null"
  },
  "confidence": "high|medium|low",
  "reasoning": "brief 1-sentence explanation",
  "suggested_action": "what carrier should do next"
}"""

    @staticmethod
    def build_classification_prompt(
        email_content: str,
        attachments: Optional[List[EmailAttachment]] = None,
        conversation_context: Optional[Dict[str, Any]] = None,
        history_summary: Optional[str] = None
    ) -> str:
        """Build the user prompt with all context."""
        
        parts = []
        
        # Add conversation context if available
        if conversation_context:
            parts.append("## CONVERSATION CONTEXT:")
            parts.append(f"- Turn count: {conversation_context.get('turn_count', 0)}")
            parts.append(f"- Negotiation stage: {conversation_context.get('negotiation_stage', 'initial')}")
            
            if conversation_context.get('mentioned_rates'):
                rates_str = ", ".join([f"{r['rate']} (from {r['from']})" for r in conversation_context['mentioned_rates'][-3:]])
                parts.append(f"- Recent rates discussed: {rates_str}")
            
            if conversation_context.get('awaiting_response_to'):
                parts.append(f"- Carrier is awaiting: {conversation_context['awaiting_response_to']}")
            
            if conversation_context.get('last_carrier_action'):
                parts.append(f"- Last carrier message: \"{conversation_context['last_carrier_action'][:150]}...\"")
            
            parts.append("")
        
        # Add conversation history summary
        if history_summary:
            parts.append("## RECENT CONVERSATION:")
            parts.append(history_summary)
            parts.append("")
        
        # Add attachment information
        if attachments:
            parts.append("## ATTACHMENTS:")
            for att in attachments:
                if isinstance(att, dict):
                    fname = att.get('filename', 'Unknown')
                    mtype = att.get('mimeType', 'Unknown')
                    size = att.get('size', 0)
                else:
                    fname = getattr(att, 'filename', 'Unknown')
                    mtype = getattr(att, 'mimeType', 'Unknown')
                    size = getattr(att, 'size', 0)
                
                # Flag potential rate confirmations
                is_rate_con = any(kw in fname.lower() for kw in ['rate', 'confirm', 'rc', 'bol', 'booking'])
                flag = " ⚠️ POSSIBLE RATE CONFIRMATION" if is_rate_con else ""
                parts.append(f"- {fname} ({mtype}, {size} bytes){flag}")
            parts.append("")
        
        # Add the email content
        parts.append("## EMAIL TO CLASSIFY:")
        
        # Clean and truncate email content
        clean_email = email_content.strip() if email_content else "[Empty email]"
        if len(clean_email) > 4000:
            clean_email = clean_email[:4000] + "\n[...truncated...]"
        
        parts.append(clean_email)
        parts.append("")
        parts.append("Classify this email's intent and extract relevant details.")
        
        return "\n".join(parts)


# --- Main Classifier Service ---
class EnhancedIntentClassifierService:
    """Enhanced email intent classifier with conversation context awareness."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize the classifier."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.context_analyzer = ConversationContextAnalyzer()
        self.prompt_builder = PromptBuilder()
    
    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for the prompt."""
        if not history:
            return ""
        
        lines = []
        # Only include last 5 messages for context
        for entry in history[-5:]:
            direction = entry.get("negotiationDirection", "unknown")
            content = entry.get("negotiationRawEmail", "")[:200]
            role = "CARRIER" if direction == "outgoing" else "BROKER"
            lines.append(f"[{role}]: {content}")
        
        return "\n".join(lines)
    
    def _extract_json_from_response(self, raw_response: str) -> dict:
        """Extract and parse JSON from the response."""
        if not raw_response or not raw_response.strip():
            raise ValueError("Empty response from API")
        
        raw_response = raw_response.strip()
        
        # Try direct parse first
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass
        
        # Remove markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Find JSON object boundaries
        start = raw_response.find('{')
        end = raw_response.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_response[start:end + 1])
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not extract valid JSON from response: {raw_response[:200]}")
    
    def _generate_clarification_response(self, email: str, context: Dict[str, Any]) -> str:
        """Generate a contextual clarification response when intent is unclear."""
        try:
            # Build context-aware prompt
            context_info = ""
            if context.get("negotiation_stage") == "active_negotiation":
                context_info = "We're in active rate negotiations. "
            elif context.get("has_load_details"):
                context_info = "We've exchanged load details. "
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.4,
                max_tokens=150,
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a professional freight carrier dispatcher. 
{context_info}Generate a brief, professional clarification response.
Keep it under 2 sentences. Be friendly but business-focused.
Focus on understanding what the broker needs regarding the load."""
                    },
                    {
                        "role": "user",
                        "content": f"The broker wrote: \"{email[:300]}\"\n\nGenerate a clarification response."
                    }
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating clarification: {e}")
            return "Could you please clarify what you need regarding this load? I want to make sure I address your request correctly."
    
    def classify(
        self,
        email_content: str,
        attachments: Optional[List[EmailAttachment]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> LoadBidEmailClassificationResponse:
        """
        Classify email intent with full context awareness.
        
        Args:
            email_content: The email text to classify
            attachments: Optional list of email attachments
            conversation_history: Optional list of previous conversation turns
        
        Returns:
            Structured classification response
        """
        try:
            # Analyze conversation context
            context = self.context_analyzer.extract_context(conversation_history)
            
            # Format conversation history
            history_summary = self._format_conversation_history(conversation_history)
            
            # Build the classification prompt
            user_prompt = self.prompt_builder.build_classification_prompt(
                email_content=email_content,
                attachments=attachments,
                conversation_context=context,
                history_summary=history_summary
            )
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt_builder.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty API response")
            
            raw_response = response.choices[0].message.content
            parsed = self._extract_json_from_response(raw_response)
            
            # Ensure required fields
            if "intent" not in parsed:
                parsed["intent"] = "unclear"
            if "details" not in parsed:
                parsed["details"] = {}
            if "confidence" not in parsed:
                parsed["confidence"] = "medium"
            
            # Generate clarification if unclear
            if parsed["intent"] == "unclear":
                clarification = self._generate_clarification_response(email_content, context)
                if "details" not in parsed:
                    parsed["details"] = {}
                parsed["details"]["questions"] = [clarification]
            
            # Validate and return
            result = LoadBidEmailClassificationResponse(**parsed)
            
            print(f"✓ Classified: {result.intent} (confidence: {result.confidence})")
            if result.reasoning:
                print(f"  Reasoning: {result.reasoning}")
            
            return result
            
        except ValidationError as e:
            print(f"Validation error: {e}")
            return self._create_fallback_response(email_content, context if 'context' in locals() else {})
        
        except Exception as e:
            print(f"Classification error: {e}")
            return self._create_fallback_response(email_content, {})
    
    def _create_fallback_response(
        self, 
        email: str, 
        context: Dict[str, Any]
    ) -> LoadBidEmailClassificationResponse:
        """Create a fallback response when classification fails."""
        clarification = self._generate_clarification_response(email, context)
        
        return LoadBidEmailClassificationResponse(
            intent="unclear",
            details=LoadBidEmailDetails(
                questions=[clarification]
            ),
            confidence="low",
            reasoning="Classification failed, requesting clarification",
            suggested_action="Send clarification message to broker"
        )
    
    # Backwards compatible method name
    def classify_load_bid_email_intent(
        self,
        input_email: str,
        attachments: Optional[List] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> LoadBidEmailClassificationResponse:
        """Backwards compatible method name."""
        return self.classify(input_email, attachments, conversation_history)


# --- Quick Intent Detector for Simple Cases ---
class QuickIntentDetector:
    """Fast pattern-based detector for obvious cases before using LLM."""
    
    ACCEPTANCE_PATTERNS = [
        r'\b(sounds?\s+good|works?\s+for\s+(me|us)|agreed?|confirmed?|let\'?s?\s+do\s+it|deal|perfect)\b',
        r'^(yes|yeah|yep|ok|okay|sure|absolutely|definitely)[\s\.\!]*$',
    ]
    
    REJECTION_PATTERNS = [
        r'\b(no\s+thanks?|can\'?t\s+do|not\s+interested|pass|decline|reject)\b',
        r'\b(too\s+(high|low|much)|found\s+another|going\s+with\s+(someone|another))\b',
    ]
    
    RATE_PATTERNS = [
        r'\$\s*[\d,]+(?:\.\d{2})?',
        r'[\d,]+\s*(?:dollars?|usd)',
    ]
    
    LOAD_DETAIL_PATTERNS = [
        r'(?:pickup|delivery|pick\s*up|drop\s*off)[\s:]*[A-Z][a-z]+,?\s*[A-Z]{2}',
        r'(?:date|time)[\s:]*\d{1,2}[\/\-]\d{1,2}',
        r'\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)',
    ]
    
    @classmethod
    def quick_detect(cls, email: str, has_attachments: bool = False) -> Optional[Tuple[str, str]]:
        """
        Quickly detect obvious intents without LLM.
        
        Returns:
            Tuple of (intent, confidence) or None if LLM needed
        """
        if not email or not email.strip():
            return ("unclear", "low")
        
        email_lower = email.lower().strip()
        
        # Check for rate confirmation attachments
        if has_attachments:
            # Let LLM handle attachment cases
            return None
        
        # Check for clear acceptance
        for pattern in cls.ACCEPTANCE_PATTERNS:
            if re.search(pattern, email_lower, re.IGNORECASE):
                return ("bid_acceptance", "high")
        
        # Check for clear rejection
        for pattern in cls.REJECTION_PATTERNS:
            if re.search(pattern, email_lower, re.IGNORECASE):
                return ("bid_rejection", "high")
        
        # Check for load details (multiple indicators)
        detail_count = sum(1 for p in cls.LOAD_DETAIL_PATTERNS if re.search(p, email, re.IGNORECASE))
        if detail_count >= 2:
            return ("load_details", "medium")
        
        # For complex cases, use LLM
        return None


# Alias for backwards compatibility
IntentClassifierService = EnhancedIntentClassifierService


# --- Test Suite ---
def run_tests():
    """Run test cases to verify classifier behavior."""
    
    test_cases = [
        # Simple acceptance cases
        {
            "email": "Sounds good, let's proceed",
            "expected_intent": "bid_acceptance",
            "description": "Clear acceptance"
        },
        {
            "email": "yes",
            "history": [
                {"negotiationDirection": "outgoing", "negotiationRawEmail": "Can you do $1500 for this load?"}
            ],
            "expected_intent": "bid_acceptance",
            "description": "Short 'yes' after rate proposal"
        },
        
        # Load details cases
        {
            "email": """*Pickup:*
📍 *Dallas, TX 75247*
📅 *Pickup Date:* *Dec 5, 2025*
⏰ *Time:* 08:00 AM

*Delivery:*
📍 *Houston, TX 77001*
📅 *Delivery Date:* *Dec 5, 2025*
⏰ *Time:* 04:00 PM""",
            "expected_intent": "load_details",
            "description": "Formatted load details with emoji"
        },
        
        # Negotiation cases
        {
            "email": "Can you do $1800 instead?",
            "expected_intent": "negotiation",
            "description": "Counter-offer"
        },
        {
            "email": "That rate is a bit high for this lane. Best we can do is $1400.",
            "expected_intent": "negotiation",
            "description": "Rate pushback with counter"
        },
        
        # Rejection cases
        {
            "email": "No thanks, we found another carrier",
            "expected_intent": "bid_rejection",
            "description": "Clear rejection"
        },
        
        # Information seeking
        {
            "email": "What type of trailer do you have available?",
            "expected_intent": "information_seeking",
            "description": "Equipment question"
        },
        
        # Availability check
        {
            "email": "Is this load still available for us to book?",
            "expected_intent": "availability_check",
            "description": "Availability question"
        },
        
        # Context-dependent case
        {
            "email": "Ok $1500 works",
            "history": [
                {"negotiationDirection": "incoming", "negotiationRawEmail": "Load from Dallas to Houston, 450 miles"},
                {"negotiationDirection": "outgoing", "negotiationRawEmail": "I can do this for $1500"},
            ],
            "expected_intent": "bid_acceptance",
            "description": "Acceptance after carrier rate proposal"
        },
    ]
    
    print("=" * 60)
    print("RUNNING INTENT CLASSIFIER TESTS")
    print("=" * 60)
    
    service = EnhancedIntentClassifierService()
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"  Email: {test['email'][:80]}...")
        
        result = service.classify(
            email_content=test['email'],
            conversation_history=test.get('history')
        )
        
        passed = result.intent == test['expected_intent']
        status = "✓ PASS" if passed else "✗ FAIL"
        
        print(f"  Expected: {test['expected_intent']}")
        print(f"  Got: {result.intent} ({result.confidence})")
        print(f"  {status}")
        
        if result.reasoning:
            print(f"  Reasoning: {result.reasoning}")
        
        results.append(passed)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    return all(results)


if __name__ == "__main__":
    # Run tests if executed directly
    run_tests()