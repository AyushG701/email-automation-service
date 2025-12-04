"""
Enhanced Email Intent Classifier for Freight/Logistics Communications
Version 3.0 - Context-aware with State Machine Integration

Key Improvements:
1. Integrates with NegotiationOrchestrator for context-aware classification
2. Non-overlapping intent categories
3. Separate tracking of info exchanges vs price negotiations
4. Robust price extraction
5. Multi-API call approach for accuracy when confidence is low
"""

import json
import re
from typing import List, Optional, Dict, Any, Tuple
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError, field_validator
import os
from dotenv import load_dotenv
from enum import Enum
import logging

load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS - Non-overlapping Intent Categories
# =============================================================================

class IntentType(str, Enum):
    """Non-overlapping intent categories aligned with orchestrator"""
    # Price-related (affects negotiation rounds)
    PRICE_OFFER = "price_offer"
    PRICE_COUNTER = "price_counter"
    PRICE_INQUIRY = "price_inquiry"
    PRICE_ACCEPTANCE = "price_acceptance"
    PRICE_REJECTION = "price_rejection"

    # Legacy mappings for backwards compatibility
    NEGOTIATION = "negotiation"
    BID_ACCEPTANCE = "bid_acceptance"
    BID_REJECTION = "bid_rejection"

    # Info-related (does NOT affect negotiation rounds)
    INFO_REQUEST = "info_request"
    INFO_RESPONSE = "info_response"
    INFORMATION_SEEKING = "information_seeking"  # Legacy
    LOAD_DETAILS = "load_details"
    AVAILABILITY_CHECK = "availability_check"

    # Admin
    RATE_CONFIRMATION = "rate_confirmation"
    BOOKING_REQUEST = "booking_request"
    BROKER_SETUP = "broker_setup"

    # Fallback
    UNCLEAR = "unclear"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

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
        if isinstance(v, str):
            v = v.strip()
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
        # Map intents to valid values
        intent_mapping = {
            'price_offer': 'negotiation',
            'price_counter': 'negotiation',
            'price_inquiry': 'negotiation',
            'price_acceptance': 'bid_acceptance',
            'price_rejection': 'bid_rejection',
            'info_request': 'information_seeking',
            'info_response': 'load_details',
        }

        if v and isinstance(v, str):
            v_lower = v.lower()
            if v_lower in intent_mapping:
                return intent_mapping[v_lower]
            valid_intents = [e.value for e in IntentType]
            if v_lower in valid_intents:
                return v_lower
        return IntentType.UNCLEAR.value


# =============================================================================
# CONVERSATION CONTEXT ANALYZER
# =============================================================================

class ConversationContextAnalyzer:
    """Analyzes conversation history with SEPARATE counters for info vs price"""

    @staticmethod
    def extract_context(history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract context with separate counters - KEY FIX"""
        context = {
            "turn_count": len(history) if history else 0,
            "info_exchanges": 0,
            "negotiation_rounds": 0,
            "last_carrier_action": None,
            "last_broker_action": None,
            "last_carrier_price": None,
            "last_broker_price": None,
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

        # Price-related keywords
        price_keywords = ['$', 'dollar', 'rate', 'price', 'bid', 'offer', 'pay', 'cost']

        for i, entry in enumerate(history):
            direction = entry.get("negotiationDirection", "")
            content = entry.get("negotiationRawEmail", "")[:500]
            content_lower = content.lower()

            # Extract rates
            rates = rate_pattern.findall(content)
            is_price_message = any(kw in content_lower for kw in price_keywords) or len(rates) > 0

            for rate in rates:
                context["mentioned_rates"].append({
                    "rate": f"${rate}",
                    "from": "carrier" if direction == "outgoing" else "broker",
                    "turn": i
                })

            # Track by direction with separate counters
            if direction == "outgoing":  # Carrier
                context["last_carrier_action"] = content[:200]
                if is_price_message and rates:
                    context["negotiation_rounds"] += 1
                    context["last_carrier_price"] = float(rates[0].replace(',', ''))
                elif question_pattern.search(content) and not is_price_message:
                    context["info_exchanges"] += 1
                    context["pending_questions"].append(content[:100])
                    context["awaiting_response_to"] = "carrier_question"
            else:  # Broker
                context["last_broker_action"] = content[:200]
                context["awaiting_response_to"] = None
                if rates:
                    context["last_broker_price"] = float(rates[0].replace(',', ''))

            # Check for load details
            if any(kw in content_lower for kw in ['pickup', 'delivery', 'miles', 'weight', 'commodity']):
                context["has_load_details"] = True

        # Determine stage based on counts
        if context["negotiation_rounds"] > 0:
            context["negotiation_stage"] = "active_negotiation"
        elif context["has_load_details"] or context["info_exchanges"] > 0:
            context["negotiation_stage"] = "info_exchange"
        elif context["turn_count"] > 0:
            context["negotiation_stage"] = "initial_contact"

        return context


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

class PromptBuilder:
    """Builds context-aware prompts for classification"""

    SYSTEM_PROMPT = """You are an expert email intent classifier for freight broker-carrier communications.

CRITICAL: Classification DEPENDS on conversation context. Same words mean different things at different stages.

## INTENT CATEGORIES (in priority order):

### bid_acceptance (HIGHEST PRIORITY)
Broker explicitly agrees to carrier's price. REQUIRES carrier to have proposed a price first.
- "sounds good", "works for me", "deal", "agreed", "let's do it"
- Short "yes", "ok", "okay" ONLY if responding to a price proposal
- Must have context of carrier's price offer

### bid_rejection
Broker explicitly declines.
- "no thanks", "can't do it", "pass", "too high/low"
- "found another carrier", "going with someone else"

### negotiation
Active price discussion. Look for:
- Counter-offers with dollar amounts
- "can you do $X", "best I can do is $X"
- Response to rate questions

### information_seeking
Questions NOT about price:
- Equipment, dates, availability
- MC number, insurance, documentation
- Load logistics questions

### load_details
Broker providing information (not asking):
- Pickup/delivery locations, dates, times
- Weight, commodity, equipment requirements

### rate_confirmation
Rate confirmation document:
- "rate con", "RC attached", "confirmation"
- Attachment mentions

### booking_request
Ready to book:
- "let's book", "send rate con", "ready to dispatch"

### broker_setup
Carrier onboarding:
- "carrier packet", "set you up", "registration"

### unclear
Only if truly cannot determine.

## CONTEXT RULES:

1. After carrier proposes price → short "ok/yes" = bid_acceptance
2. After carrier asks question → broker response = answering that
3. In active negotiation → price mention = counter (negotiation)
4. No prices yet → price mention = initial offer (negotiation)

## OUTPUT (JSON only):
{
  "intent": "one of the intents above",
  "details": {
    "rate": "extracted rate or null",
    "questions": ["questions asked"] or null,
    ...
  },
  "confidence": "high|medium|low",
  "reasoning": "brief explanation"
}"""

    VERIFICATION_PROMPT = """You are verifying an intent classification. The first classifier said:
Intent: {initial_intent}
Confidence: {initial_confidence}
Reasoning: {initial_reasoning}

Review the message and context. Do you agree? If not, what's the correct intent?

CONVERSATION CONTEXT:
{context}

MESSAGE TO CLASSIFY:
{message}

Respond with JSON:
{{
  "agree": true or false,
  "correct_intent": "the correct intent",
  "confidence": "high|medium|low",
  "reasoning": "why you agree or disagree"
}}"""

    @staticmethod
    def build_classification_prompt(
        email_content: str,
        attachments: Optional[List[EmailAttachment]] = None,
        conversation_context: Optional[Dict[str, Any]] = None,
        history_summary: Optional[str] = None
    ) -> str:
        """Build classification prompt with full context"""
        parts = []

        if conversation_context:
            parts.append("## CONVERSATION CONTEXT:")
            parts.append(f"- Total messages: {conversation_context.get('turn_count', 0)}")
            parts.append(f"- Info exchanges: {conversation_context.get('info_exchanges', 0)}")
            parts.append(f"- Negotiation rounds: {conversation_context.get('negotiation_rounds', 0)}")
            parts.append(f"- Stage: {conversation_context.get('negotiation_stage', 'initial')}")

            if conversation_context.get('last_carrier_price'):
                parts.append(f"- Last carrier price: ${conversation_context['last_carrier_price']}")
            if conversation_context.get('last_broker_price'):
                parts.append(f"- Last broker price: ${conversation_context['last_broker_price']}")

            if conversation_context.get('mentioned_rates'):
                rates_str = ", ".join([f"{r['rate']} (from {r['from']})" for r in conversation_context['mentioned_rates'][-3:]])
                parts.append(f"- Recent rates: {rates_str}")

            if conversation_context.get('awaiting_response_to'):
                parts.append(f"- Carrier awaiting: {conversation_context['awaiting_response_to']}")

            if conversation_context.get('last_carrier_action'):
                parts.append(f"- Last carrier message: \"{conversation_context['last_carrier_action'][:150]}...\"")
            parts.append("")

        if history_summary:
            parts.append("## RECENT CONVERSATION:")
            parts.append(history_summary)
            parts.append("")

        if attachments:
            parts.append("## ATTACHMENTS:")
            for att in attachments:
                if isinstance(att, dict):
                    fname = att.get('filename', 'Unknown')
                else:
                    fname = getattr(att, 'filename', 'Unknown')

                is_rate_con = any(kw in fname.lower() for kw in ['rate', 'confirm', 'rc', 'bol', 'booking'])
                flag = " [POSSIBLE RATE CONFIRMATION]" if is_rate_con else ""
                parts.append(f"- {fname}{flag}")
            parts.append("")

        parts.append("## EMAIL TO CLASSIFY:")
        clean_email = email_content.strip() if email_content else "[Empty email]"
        if len(clean_email) > 4000:
            clean_email = clean_email[:4000] + "\n[...truncated...]"
        parts.append(clean_email)

        return "\n".join(parts)


# =============================================================================
# MAIN CLASSIFIER SERVICE
# =============================================================================

class IntentClassifierService:
    """
    Enhanced intent classifier with multi-API approach for accuracy.

    Uses multiple OpenAI calls when confidence is low to ensure accuracy.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.context_analyzer = ConversationContextAnalyzer()
        self.prompt_builder = PromptBuilder()
        self.logger = logging.getLogger(__name__)

    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for prompt"""
        if not history:
            return ""

        lines = []
        for entry in history[-6:]:  # Last 6 messages
            direction = entry.get("negotiationDirection", "unknown")
            content = entry.get("negotiationRawEmail", "")[:200]
            rate = entry.get("rate", "")
            role = "CARRIER" if direction == "outgoing" else "BROKER"
            rate_str = f" [${rate}]" if rate else ""
            lines.append(f"[{role}]{rate_str}: {content}")

        return "\n".join(lines)

    def _extract_json_from_response(self, raw_response: str) -> dict:
        """Extract and parse JSON from response"""
        if not raw_response or not raw_response.strip():
            raise ValueError("Empty response from API")

        raw_response = raw_response.strip()

        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        start = raw_response.find('{')
        end = raw_response.rfind('}')

        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_response[start:end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract JSON from response: {raw_response[:200]}")

    def _initial_classification(
        self,
        email_content: str,
        conversation_context: Dict[str, Any],
        history_summary: str,
        attachments: Optional[List] = None
    ) -> Dict[str, Any]:
        """First classification pass"""
        user_prompt = self.prompt_builder.build_classification_prompt(
            email_content=email_content,
            attachments=attachments,
            conversation_context=conversation_context,
            history_summary=history_summary
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt_builder.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"}
        )

        return self._extract_json_from_response(response.choices[0].message.content)

    def _verify_classification(
        self,
        email_content: str,
        initial_result: Dict[str, Any],
        conversation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Second pass to verify low-confidence classifications"""
        context_str = json.dumps(conversation_context, indent=2)

        verify_prompt = self.prompt_builder.VERIFICATION_PROMPT.format(
            initial_intent=initial_result.get('intent', 'unclear'),
            initial_confidence=initial_result.get('confidence', 'low'),
            initial_reasoning=initial_result.get('reasoning', ''),
            context=context_str,
            message=email_content[:1000]
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",  # Use stronger model for verification
            messages=[
                {"role": "system", "content": "You are an expert at verifying email intent classifications."},
                {"role": "user", "content": verify_prompt}
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        return self._extract_json_from_response(response.choices[0].message.content)

    def _pattern_based_check(
        self,
        email_content: str,
        context: Dict[str, Any]
    ) -> Optional[Tuple[str, str]]:
        """Quick pattern-based check for obvious cases"""
        email_lower = email_content.lower().strip()

        # Acceptance patterns - only valid if carrier proposed a price
        if context.get('last_carrier_price'):
            acceptance_patterns = [
                r'\b(sounds?\s+good|works?\s+for\s+(me|us)|agreed?|deal|let\'?s?\s+do\s+it|confirmed?)\b',
                r'^(yes|yeah|yep|ok|okay|sure|absolutely|definitely)[\s\.\!]*$',
            ]
            for pattern in acceptance_patterns:
                if re.search(pattern, email_lower, re.IGNORECASE):
                    return ("bid_acceptance", "high")

        # Rejection patterns
        rejection_patterns = [
            r'\b(no\s+thanks?|can\'?t\s+do|not\s+interested|pass|decline|reject)\b',
            r'\b(too\s+(high|low)|found\s+another|going\s+with\s+(someone|another))\b',
        ]
        for pattern in rejection_patterns:
            if re.search(pattern, email_lower, re.IGNORECASE):
                return ("bid_rejection", "high")

        return None

    def _generate_clarification(self, email: str, context: Dict[str, Any]) -> str:
        """Generate contextual clarification"""
        try:
            context_info = ""
            if context.get("negotiation_stage") == "active_negotiation":
                context_info = "We're in active rate negotiations. "
            elif context.get("has_load_details"):
                context_info = "We've exchanged load details. "

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.4,
                max_tokens=100,
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a professional freight carrier.
{context_info}Generate a brief, professional clarification (1-2 sentences max)."""
                    },
                    {
                        "role": "user",
                        "content": f"The broker wrote: \"{email[:300]}\"\nGenerate a clarification request."
                    }
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.warning(f"Clarification generation failed: {e}")
            return "Could you please clarify what you need regarding this load?"

    def classify(
        self,
        email_content: str,
        attachments: Optional[List[EmailAttachment]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> LoadBidEmailClassificationResponse:
        """
        Classify email intent with multi-API approach for accuracy.

        Uses multiple OpenAI calls when confidence is low.
        """
        try:
            # Analyze conversation context
            context = self.context_analyzer.extract_context(conversation_history)
            history_summary = self._format_conversation_history(conversation_history)

            self.logger.info(f"Classification - Info exchanges: {context.get('info_exchanges', 0)}")
            self.logger.info(f"Classification - Negotiation rounds: {context.get('negotiation_rounds', 0)}")
            self.logger.info(f"Classification - Stage: {context.get('negotiation_stage', 'initial')}")

            # Quick pattern check first
            pattern_result = self._pattern_based_check(email_content, context)
            if pattern_result:
                intent, confidence = pattern_result
                self.logger.info(f"Pattern match: {intent} ({confidence})")
                return LoadBidEmailClassificationResponse(
                    intent=intent,
                    details=LoadBidEmailDetails(),
                    confidence=confidence,
                    reasoning="Pattern match with context validation"
                )

            # First classification pass
            initial_result = self._initial_classification(
                email_content=email_content,
                conversation_context=context,
                history_summary=history_summary,
                attachments=attachments
            )

            initial_confidence = initial_result.get('confidence', 'low')
            initial_intent = initial_result.get('intent', 'unclear')

            self.logger.info(f"Initial classification: {initial_intent} ({initial_confidence})")

            # If confidence is low or medium, verify with second API call
            if initial_confidence in ['low', 'medium']:
                self.logger.info("Low/medium confidence - running verification")
                verification = self._verify_classification(
                    email_content=email_content,
                    initial_result=initial_result,
                    conversation_context=context
                )

                if not verification.get('agree', True):
                    self.logger.info(f"Verification disagreed: {verification.get('correct_intent')}")
                    initial_result['intent'] = verification.get('correct_intent', initial_intent)
                    initial_result['confidence'] = verification.get('confidence', 'medium')
                    initial_result['reasoning'] = verification.get('reasoning', '')

            # Ensure required fields
            if "intent" not in initial_result:
                initial_result["intent"] = "unclear"
            if "details" not in initial_result:
                initial_result["details"] = {}
            if "confidence" not in initial_result:
                initial_result["confidence"] = "medium"

            # Generate clarification if unclear
            if initial_result["intent"] == "unclear":
                clarification = self._generate_clarification(email_content, context)
                initial_result["details"]["questions"] = [clarification]

            result = LoadBidEmailClassificationResponse(**initial_result)

            self.logger.info(f"Final classification: {result.intent} ({result.confidence})")
            if result.reasoning:
                self.logger.info(f"Reasoning: {result.reasoning}")

            return result

        except ValidationError as e:
            self.logger.error(f"Validation error: {e}")
            return self._create_fallback_response(email_content, context if 'context' in locals() else {})

        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return self._create_fallback_response(email_content, {})

    def _create_fallback_response(
        self,
        email: str,
        context: Dict[str, Any]
    ) -> LoadBidEmailClassificationResponse:
        """Create fallback response"""
        clarification = self._generate_clarification(email, context)

        return LoadBidEmailClassificationResponse(
            intent="unclear",
            details=LoadBidEmailDetails(questions=[clarification]),
            confidence="low",
            reasoning="Classification failed, requesting clarification",
            suggested_action="Send clarification message to broker"
        )

    # Backwards compatible method
    def classify_load_bid_email_intent(
        self,
        input_email: str,
        attachments: Optional[List] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> LoadBidEmailClassificationResponse:
        """Backwards compatible method name"""
        return self.classify(input_email, attachments, conversation_history)


# =============================================================================
# LEGACY ALIAS
# =============================================================================

EnhancedIntentClassifierService = IntentClassifierService


# =============================================================================
# TEST SUITE
# =============================================================================

def run_tests():
    """Run test cases"""
    test_cases = [
        {
            "email": "Sounds good, let's proceed",
            "history": [
                {"negotiationDirection": "incoming", "negotiationRawEmail": "Load from Dallas to Houston"},
                {"negotiationDirection": "outgoing", "negotiationRawEmail": "I can do $1500 for this"},
            ],
            "expected_intent": "bid_acceptance",
            "description": "Acceptance after price proposal"
        },
        {
            "email": "yes",
            "history": [
                {"negotiationDirection": "outgoing", "negotiationRawEmail": "My rate is $1800"}
            ],
            "expected_intent": "bid_acceptance",
            "description": "Short 'yes' after rate proposal"
        },
        {
            "email": "Can you do $1600 instead?",
            "history": [
                {"negotiationDirection": "outgoing", "negotiationRawEmail": "$1800 for this load"}
            ],
            "expected_intent": "negotiation",
            "description": "Counter-offer"
        },
        {
            "email": "What type of trailer do you have?",
            "expected_intent": "information_seeking",
            "description": "Equipment question"
        },
        {
            "email": "No thanks, going with another carrier",
            "expected_intent": "bid_rejection",
            "description": "Clear rejection"
        },
    ]

    print("=" * 60)
    print("INTENT CLASSIFIER TESTS")
    print("=" * 60)

    service = IntentClassifierService()
    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"  Email: {test['email'][:60]}...")

        result = service.classify(
            email_content=test['email'],
            conversation_history=test.get('history')
        )

        passed = result.intent == test['expected_intent']
        status = "PASS" if passed else "FAIL"

        print(f"  Expected: {test['expected_intent']}")
        print(f"  Got: {result.intent} ({result.confidence})")
        print(f"  {status}")

        results.append(passed)

    print("\n" + "=" * 60)
    print(f"RESULTS: {sum(results)}/{len(results)} passed")
    print("=" * 60)

    return all(results)


if __name__ == "__main__":
    run_tests()
