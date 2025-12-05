"""
Negotiation Orchestrator - Unified State Machine for Load Negotiations
Version 2.0 - With Database Persistence via Metadata

Key Fixes:
1. Persists state in negotiation metadata via supertruck API
2. Retrieves state from previous negotiations
3. Proper round counting with database backing
4. Context-aware intent classification
"""

import logging
import json
import re
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from core.utils import extract_price
from openai import OpenAI
import os

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class ConversationState(str, Enum):
    """State machine states for conversation flow"""
    INITIAL = "initial"
    INFO_GATHERING = "info_gathering"
    READY_TO_NEGOTIATE = "ready_to_negotiate"
    NEGOTIATING = "negotiating"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALLED = "stalled"


class MessageIntent(str, Enum):
    """Non-overlapping intent categories"""
    # Price-related (affects negotiation rounds)
    PRICE_OFFER = "price_offer"
    PRICE_COUNTER = "price_counter"
    PRICE_INQUIRY = "price_inquiry"
    PRICE_ACCEPTANCE = "price_acceptance"
    PRICE_REJECTION = "price_rejection"

    # Info-related (does NOT affect negotiation rounds)
    INFO_REQUEST = "info_request"           # Broker asking for carrier info (MC, DOT, etc.)
    CARRIER_INFO_REQUEST = "carrier_info_request"  # Specifically asking about MC, DOT, equipment
    INFO_RESPONSE = "info_response"
    LOAD_DETAILS = "load_details"

    # Admin
    RATE_CONFIRMATION = "rate_confirmation"
    BOOKING_REQUEST = "booking_request"
    BROKER_SETUP = "broker_setup"

    # Fallback
    UNCLEAR = "unclear"


class NegotiationAction(str, Enum):
    """Actions to take after classification"""
    COUNTER = "counter"
    ACCEPT = "accept"
    REJECT = "reject"
    REQUEST_INFO = "request_info"
    PROVIDE_INFO = "provide_info"
    CLARIFY = "clarify"
    CONFIRM_BOOKING = "confirm_booking"


# =============================================================================
# METADATA SCHEMA - This gets saved to database
# =============================================================================

@dataclass
class NegotiationMetadata:
    """
    Metadata that gets persisted with each negotiation message.
    This is the KEY to proper state tracking across API calls.
    """
    messageType: str = "UNKNOWN"           # PRICE_OFFER, PRICE_COUNTER, INFO_REQUEST, etc.
    negotiationRound: int = 0              # Actual price negotiation round (not total messages)
    infoExchangeCount: int = 0             # Number of info exchanges
    conversationState: str = "initial"     # Current state machine state
    lastCarrierPrice: Optional[float] = None
    lastBrokerPrice: Optional[float] = None
    belowMinCount: int = 0                 # How many times broker offered below min
    extractedPrice: Optional[float] = None # Price extracted from this message
    confidence: str = "medium"             # Classification confidence
    timestamp: str = ""                    # When this was processed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API call"""
        return {
            "messageType": self.messageType,
            "negotiationRound": self.negotiationRound,
            "infoExchangeCount": self.infoExchangeCount,
            "conversationState": self.conversationState,
            "lastCarrierPrice": self.lastCarrierPrice,
            "lastBrokerPrice": self.lastBrokerPrice,
            "belowMinCount": self.belowMinCount,
            "extractedPrice": self.extractedPrice,
            "confidence": self.confidence,
            "timestamp": self.timestamp or datetime.utcnow().isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NegotiationMetadata':
        """Create from dict (from API response)"""
        if not data:
            return cls()
        return cls(
            messageType=data.get("messageType", "UNKNOWN"),
            negotiationRound=data.get("negotiationRound", 0),
            infoExchangeCount=data.get("infoExchangeCount", 0),
            conversationState=data.get("conversationState", "initial"),
            lastCarrierPrice=data.get("lastCarrierPrice"),
            lastBrokerPrice=data.get("lastBrokerPrice"),
            belowMinCount=data.get("belowMinCount", 0),
            extractedPrice=data.get("extractedPrice"),
            confidence=data.get("confidence", "medium"),
            timestamp=data.get("timestamp", "")
        )


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ConversationContext:
    """Tracks conversation state - built from persisted metadata"""
    state: ConversationState = ConversationState.INITIAL
    info_exchanges: int = 0
    negotiation_rounds: int = 0
    last_broker_price: Optional[float] = None
    last_carrier_price: Optional[float] = None
    initial_broker_price: Optional[float] = None
    below_min_count: int = 0
    above_target_count: int = 0
    has_sufficient_load_info: bool = False
    missing_critical_fields: List[str] = field(default_factory=list)
    total_messages: int = 0
    last_message_direction: str = ""
    last_intent: Optional[MessageIntent] = None


@dataclass
class ClassificationResult:
    """Result from intent classification"""
    intent: MessageIntent
    confidence: float
    extracted_price: Optional[float] = None
    extracted_info: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class NegotiationResult:
    """Result from negotiation orchestrator"""
    action: NegotiationAction
    response: str
    proposed_price: Optional[float] = None
    status: str = "negotiating"
    metadata: Optional[NegotiationMetadata] = None
    reasoning: str = ""





# =============================================================================
# CONVERSATION STATE BUILDER - FROM PERSISTED METADATA
# =============================================================================

class ConversationStateBuilder:
    """
    Builds conversation context from persisted negotiation history.

    KEY: Reads metadata from previous negotiations to get accurate counts.
    """

    @staticmethod
    def build_from_history(negotiations: List[Dict]) -> Tuple[ConversationContext, NegotiationMetadata]:
        """
        Build context from negotiation history with persisted metadata.

        Returns:
            Tuple of (ConversationContext, last_metadata)
        """
        context = ConversationContext()
        last_metadata = NegotiationMetadata()

        if not negotiations:
            return context, last_metadata

        # Find the most recent negotiation with metadata
        for neg in reversed(negotiations):
            metadata_dict = neg.get('metadata')
            if metadata_dict:
                last_metadata = NegotiationMetadata.from_dict(metadata_dict)
                break

        # If we found persisted metadata, use it as base
        if last_metadata.negotiationRound > 0 or last_metadata.infoExchangeCount > 0:
            context.negotiation_rounds = last_metadata.negotiationRound
            context.info_exchanges = last_metadata.infoExchangeCount
            context.last_carrier_price = last_metadata.lastCarrierPrice
            context.last_broker_price = last_metadata.lastBrokerPrice
            context.below_min_count = last_metadata.belowMinCount

            try:
                context.state = ConversationState(last_metadata.conversationState)
            except ValueError:
                context.state = ConversationState.INITIAL

            logger.info(f"Loaded persisted state: rounds={context.negotiation_rounds}, info={context.info_exchanges}")
        else:
            # Fallback: Calculate from history if no metadata exists
            logger.info("No persisted metadata found, calculating from history...")
            context, last_metadata = ConversationStateBuilder._calculate_from_history(negotiations)

        context.total_messages = len(negotiations)

        # Get last message direction
        if negotiations:
            last_neg = negotiations[-1]
            context.last_message_direction = last_neg.get('negotiationDirection', '')

        return context, last_metadata

    @staticmethod
    def _calculate_from_history(negotiations: List[Dict]) -> Tuple[ConversationContext, NegotiationMetadata]:
        """
        Fallback: Calculate state from history when no metadata exists.
        Used for backwards compatibility with existing data.
        """
        context = ConversationContext()
        metadata = NegotiationMetadata()

        PRICE_KEYWORDS = [
            r'\$\d+', r'\d+\s*dollars?', r'rate', r'price', r'bid', r'offer',
            r'can you do', r'i need', r'best i can', r'works for me', r'deal'
        ]

        INFO_KEYWORDS = [
            r'equipment', r'trailer', r'van', r'flatbed', r'reefer',
            r'pickup', r'delivery', r'date', r'time', r'weight', r'commodity',
            r'mc number', r'dot', r'insurance', r'location', r'miles', r'\?'
        ]

        for neg in negotiations:
            direction = neg.get('negotiationDirection', '').lower()
            content = neg.get('negotiationRawEmail', '').lower()
            rate = neg.get('rate')

            if not content:
                continue

            is_price_msg = any(re.search(p, content, re.I) for p in PRICE_KEYWORDS)
            is_info_msg = any(re.search(p, content, re.I) for p in INFO_KEYWORDS)
            extracted_price = extract_price(content) or rate

            if direction == 'outgoing':  # Carrier
                if is_price_msg and extracted_price:
                    context.negotiation_rounds += 1
                    context.last_carrier_price = float(extracted_price) if extracted_price else None
                elif is_info_msg and not is_price_msg:
                    context.info_exchanges += 1

            elif direction == 'incoming':  # Broker
                if extracted_price:
                    context.last_broker_price = float(extracted_price) if extracted_price else None
                    if context.initial_broker_price is None:
                        context.initial_broker_price = context.last_broker_price

        # Determine state
        if context.negotiation_rounds >= 1:
            context.state = ConversationState.NEGOTIATING
        elif context.info_exchanges >= 1:
            context.state = ConversationState.INFO_GATHERING

        # Build metadata
        metadata.negotiationRound = context.negotiation_rounds
        metadata.infoExchangeCount = context.info_exchanges
        metadata.conversationState = context.state.value
        metadata.lastCarrierPrice = context.last_carrier_price
        metadata.lastBrokerPrice = context.last_broker_price

        return context, metadata

    @staticmethod
    def format_history_for_llm(negotiations: List[Dict], max_messages: int = 5) -> str:
        """Format conversation history for LLM prompt"""
        if not negotiations:
            return "No previous conversation"

        lines = []
        for neg in negotiations[-max_messages:]:
            direction = neg.get('negotiationDirection', 'unknown')
            content = neg.get('negotiationRawEmail', '')[:150]
            rate = neg.get('rate')
            metadata = neg.get('metadata') or {}

            prefix = "CARRIER" if direction == "outgoing" else "BROKER"
            rate_str = f" [${rate}]" if rate else ""
            msg_type = metadata.get('messageType', '')
            type_str = f" ({msg_type})" if msg_type else ""

            lines.append(f"{prefix}{rate_str}{type_str}: {content}")

        return "\n".join(lines) if lines else "No previous conversation"


# =============================================================================
# CONTEXT-AWARE INTENT CLASSIFIER
# =============================================================================

class ContextAwareClassifier:
    """Classify message intent with conversation context"""

    SYSTEM_PROMPT = """You are an expert at classifying freight broker-carrier email messages.

CRITICAL: Classification depends on CONVERSATION CONTEXT.

## CONVERSATION STATE PROVIDED:
- negotiationRound: Number of PRICE negotiations (not total messages)
- infoExchangeCount: Number of info exchanges (questions about equipment, etc.)
- lastCarrierPrice: Last price the carrier offered
- lastBrokerPrice: Last price the broker offered

## INTENT CATEGORIES (choose ONE):

### PRICE_ACCEPTANCE (highest priority)
Broker explicitly agrees to carrier's price. REQUIRES carrier to have proposed a price.
- "sounds good", "works for me", "deal", "agreed", "let's do it"
- Short "yes", "ok" ONLY if responding to a price offer

### PRICE_REJECTION
Broker explicitly declines.
- "no thanks", "can't do it", "pass", "too high/low"

### PRICE_OFFER / PRICE_COUNTER
Broker provides or counters with a price.
- Explicit dollar amount
- "can you do $X", "best I can do is $X"

### PRICE_INQUIRY
Broker asks for carrier's rate.
- "what's your rate", "what do you need"

### CARRIER_INFO_REQUEST (high priority for questions about carrier)
Broker asks about CARRIER details - these need immediate answers.
- MC number: "what's your mc", "mc number", "what is your mc"
- DOT number: "dot number", "what's your dot"
- Insurance: "insurance", "coverage"
- Equipment/trucks: "what equipment", "truck type"
- Authority: "authority", "carrier authority"
- Driver: "driver experience", "team drivers"
- Availability: "are you available", "when available"

### INFO_REQUEST
Broker asks OTHER non-price questions (not about carrier).
- Load-related questions: "what equipment do you need"

### INFO_RESPONSE / LOAD_DETAILS
Broker provides load information.
- Pickup/delivery locations, dates, weight, commodity

### RATE_CONFIRMATION
Rate con document mentioned.

### BOOKING_REQUEST
Ready to book.

### UNCLEAR
Cannot determine.

## OUTPUT (JSON only):
{
  "intent": "one of the intents above",
  "confidence": 0.0-1.0,
  "extracted_price": number or null,
  "reasoning": "brief explanation"
}"""

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def classify(
        self,
        message: str,
        context: ConversationContext,
        conversation_history: str = ""
    ) -> ClassificationResult:
        """Classify message with full context awareness"""

        context_desc = f"""
CONVERSATION STATE:
- negotiationRound: {context.negotiation_rounds}
- infoExchangeCount: {context.info_exchanges}
- lastCarrierPrice: ${context.last_carrier_price or 'none'}
- lastBrokerPrice: ${context.last_broker_price or 'none'}
- currentState: {context.state.value}
- lastMessageFrom: {context.last_message_direction or 'none'}
"""

        user_prompt = f"""
{context_desc}

RECENT CONVERSATION:
{conversation_history}

CURRENT BROKER MESSAGE:
{message}

Classify this message's intent.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            intent_str = result.get('intent', 'unclear').lower().replace(' ', '_')
            try:
                intent = MessageIntent(intent_str)
            except ValueError:
                intent_map = {
                    'bid_acceptance': MessageIntent.PRICE_ACCEPTANCE,
                    'bid_rejection': MessageIntent.PRICE_REJECTION,
                    'negotiation': MessageIntent.PRICE_COUNTER,
                    'information_seeking': MessageIntent.INFO_REQUEST,
                    'carrier_info': MessageIntent.CARRIER_INFO_REQUEST,
                    'mc_request': MessageIntent.CARRIER_INFO_REQUEST,
                }
                intent = intent_map.get(intent_str, MessageIntent.UNCLEAR)

            return ClassificationResult(
                intent=intent,
                confidence=result.get('confidence', 0.5),
                extracted_price=result.get('extracted_price') or extract_price(message),
                reasoning=result.get('reasoning', '')
            )

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return self._fallback_classify(message, context)

    def _fallback_classify(self, message: str, context: ConversationContext) -> ClassificationResult:
        """Pattern-based fallback classification"""
        message_lower = message.lower().strip()

        # Acceptance patterns - only if carrier proposed price
        if context.last_carrier_price:
            acceptance_patterns = [
                r'\b(sounds?\s+good|works?\s+for\s+(me|us)|agreed?|deal|let\'?s?\s+do\s+it)\b',
                r'^(yes|yeah|yep|ok|okay|sure|absolutely)[\s\.\!]*$'
            ]
            for pattern in acceptance_patterns:
                if re.search(pattern, message_lower):
                    return ClassificationResult(
                        intent=MessageIntent.PRICE_ACCEPTANCE,
                        confidence=0.8,
                        reasoning="Acceptance pattern after price proposal"
                    )

        # Rejection patterns
        rejection_patterns = [
            r'\b(no\s+thanks?|can\'?t\s+do|not\s+interested|pass|decline)\b',
            r'\b(too\s+(high|low)|found\s+another|going\s+with)\b'
        ]
        for pattern in rejection_patterns:
            if re.search(pattern, message_lower):
                return ClassificationResult(
                    intent=MessageIntent.PRICE_REJECTION,
                    confidence=0.8,
                    reasoning="Rejection pattern detected"
                )

        # Price extraction
        price = extract_price(message)
        if price:
            if context.negotiation_rounds >= 1:
                return ClassificationResult(
                    intent=MessageIntent.PRICE_COUNTER,
                    confidence=0.7,
                    extracted_price=price,
                    reasoning="Price in active negotiation"
                )
            else:
                return ClassificationResult(
                    intent=MessageIntent.PRICE_OFFER,
                    confidence=0.7,
                    extracted_price=price,
                    reasoning="Initial price offer"
                )

        # Carrier info patterns - check BEFORE generic info patterns
        carrier_info_patterns = [
            r'\b(mc|m\.c\.|mc\s*number|motor\s*carrier)\b',
            r'\b(dot|d\.o\.t\.|dot\s*number)\b',
            r'\b(insurance|coverage|liability)\b',
            r'\b(authority|carrier\s*authority)\b',
            r'\b(equipment|truck|trailer)\s*(type|size|available)?\b',
            r'\b(available|availability)\b'
        ]
        if '?' in message_lower:
            for pattern in carrier_info_patterns:
                if re.search(pattern, message_lower):
                    return ClassificationResult(
                        intent=MessageIntent.CARRIER_INFO_REQUEST,
                        confidence=0.8,
                        reasoning="Carrier info question detected (MC, DOT, equipment, etc.)"
                    )

        # Generic info patterns
        if '?' in message_lower or any(w in message_lower for w in ['what', 'when', 'where', 'how']):
            return ClassificationResult(
                intent=MessageIntent.INFO_REQUEST,
                confidence=0.6,
                reasoning="Question pattern detected"
            )

        return ClassificationResult(
            intent=MessageIntent.UNCLEAR,
            confidence=0.3,
            reasoning="Could not determine intent"
        )


# =============================================================================
# NEGOTIATION STRATEGY ENGINE
# =============================================================================

class NegotiationStrategy:
    """Decision engine using TRUE negotiation rounds from persisted state"""

    @staticmethod
    def should_accept(
        broker_offer: float,
        min_price: float,
        max_price: float,
        negotiation_rounds: int
    ) -> bool:
        """Determine if we should accept"""
        if broker_offer >= max_price:
            return True
        if negotiation_rounds <= 2:
            return False
        if negotiation_rounds >= 3 and broker_offer >= min_price * 0.98:
            return True
        return False

    @staticmethod
    def should_reject(
        broker_offer: float,
        min_price: float,
        negotiation_rounds: int,
        below_min_count: int
    ) -> bool:
        """Determine if we should reject"""
        if negotiation_rounds <= 1:
            return False
        if broker_offer < min_price * 0.80:
            return True
        if below_min_count >= 3:
            return True
        if negotiation_rounds >= 5 and broker_offer < min_price * 0.95:
            return True
        return False

    @staticmethod
    def calculate_counter(
        min_price: float,
        sweet_spot: float,
        max_price: float,
        negotiation_rounds: int,
        broker_offer: float,
        last_carrier_price: Optional[float] = None
    ) -> float:
        """Calculate strategic counter offer"""
        if negotiation_rounds <= 1:
            return max_price
        if negotiation_rounds == 2:
            return sweet_spot
        buffer = max(50, min_price * 0.02)
        if last_carrier_price and last_carrier_price > min_price + buffer:
            new_price = (broker_offer + last_carrier_price) / 2
            return max(new_price, min_price + buffer)
        return min_price + buffer


# =============================================================================
# RESPONSE GENERATOR
# =============================================================================

class ResponseGenerator:
    """Generate natural, human-like responses"""

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model

    def generate(
        self,
        action: NegotiationAction,
        broker_company: str,
        our_price: Optional[float],
        broker_offer: Optional[float],
        route: str = "",
        min_price: float = 0,
        is_first_message: bool = False
    ) -> str:
        """Generate response based on action"""
        if action == NegotiationAction.ACCEPT:
            return self._acceptance_response(broker_company)
        elif action == NegotiationAction.REJECT:
            return self._rejection_response(broker_company, min_price)
        elif action == NegotiationAction.COUNTER:
            return self._counter_response(broker_company, our_price, broker_offer, is_first_message)
        elif action == NegotiationAction.REQUEST_INFO:
            return "I'm interested but need a few more details to give you a rate. What's the pickup/delivery and dates?"
        elif action == NegotiationAction.CLARIFY:
            return "Hey, could you clarify what you need? Happy to help once I understand the request."
        else:
            return self._counter_response(broker_company, our_price, broker_offer, is_first_message)

    def _acceptance_response(self, broker_company: str) -> str:
        responses = [
            "Perfect! Send rate confirmation to book this load.",
            "Sounds good! Please send RC so we can lock this in.",
            "Deal! Forward the rate confirmation to get this moving.",
            "That works! Send over the rate con and we're good to go."
        ]
        return responses[hash(broker_company) % len(responses)]

    def _rejection_response(self, broker_company: str, min_price: float) -> str:
        responses = [
            f"Sorry, ${min_price:.0f} is my floor for this lane. Maybe next load.",
            f"Appreciate the offer but can't make it work below ${min_price:.0f}. Hit me up on the next one.",
            f"Thanks but I need at least ${min_price:.0f} to run this. Let's try another load.",
            f"Pass on this one - ${min_price:.0f} is my minimum for this route. Next time!"
        ]
        return responses[hash(broker_company) % len(responses)]

    def _counter_response(self, broker_company: str, our_price: Optional[float], broker_offer: Optional[float], is_first_message: bool = False) -> str:
        if not our_price:
            return "Let's discuss pricing. What rate works for you?"

        # First message responses - thank them for the load
        if is_first_message:
            responses = [
                f"Thanks for the load. I can do ${our_price:.0f} for this.",
                f"Appreciate you reaching out. I'd need ${our_price:.0f} for this run.",
                f"Thanks for the opportunity. ${our_price:.0f} would work for me on this.",
                f"Thanks for thinking of us. I can move this for ${our_price:.0f}."
            ]
        else:
            # Regular counter responses
            responses = [
                f"I'd need ${our_price:.0f} to move this. Can you do that?",
                f"Best I can do is ${our_price:.0f} for this lane. Let me know.",
                f"${our_price:.0f} would make this work. What do you think?",
                f"Looking at this run, ${our_price:.0f} is where I need to be."
            ]
        return responses[hash(str(broker_offer)) % len(responses)]


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class NegotiationOrchestrator:
    """
    Main orchestrator with database persistence.

    KEY: Returns metadata that MUST be saved with each negotiation.
    """

    def __init__(self):
        self.classifier = ContextAwareClassifier()
        self.response_generator = ResponseGenerator()
        self.logger = logging.getLogger(__name__)

    def process_message(
        self,
        broker_message: str,
        negotiation_history: List[Dict],
        load_offer: Dict[str, Any],
        pricing: Dict[str, float],
        carrier_info: Optional[Dict] = None
    ) -> NegotiationResult:
        """
        Main entry point for processing broker messages.

        IMPORTANT: The returned metadata MUST be saved via supertruck.negotiate()

        Returns:
            NegotiationResult with metadata that must be persisted
        """
        self.logger.info("="*60)
        self.logger.info("NEGOTIATION ORCHESTRATOR - Processing Message")
        self.logger.info("="*60)

        # Step 1: Build context from persisted history
        context, prev_metadata = ConversationStateBuilder.build_from_history(negotiation_history)
        history_formatted = ConversationStateBuilder.format_history_for_llm(negotiation_history)

        self.logger.info(f"Negotiation History - Total messages: {len(negotiation_history)}")
        self.logger.info(f"Context from DB - State: {context.state.value}")
        self.logger.info(f"Context from DB - Info Exchanges: {context.info_exchanges}")
        self.logger.info(f"Context from DB - Negotiation Rounds: {context.negotiation_rounds}")
        self.logger.info(f"Context from DB - Last Broker Price: ${context.last_broker_price}")
        self.logger.info(f"Context from DB - Last Carrier Price: ${context.last_carrier_price}")
        self.logger.info(f"Context from DB - Below Min Count: {context.below_min_count}")
        if history_formatted and history_formatted != "No previous conversation":
            self.logger.info(f"Formatted History for LLM:\n{history_formatted}")

        # Step 2: Classify intent
        classification = self.classifier.classify(broker_message, context, history_formatted)

        self.logger.info(f"Classification - Intent: {classification.intent.value}")
        self.logger.info(f"Classification - Confidence: {classification.confidence:.2f}")
        self.logger.info(f"Classification - Extracted Price: ${classification.extracted_price}")
        self.logger.info(f"Classification - Reasoning: {classification.reasoning}")

        # Step 3: Extract pricing
        min_price = pricing.get('min_price', 0)
        max_price = pricing.get('max_price', 0)
        sweet_spot = pricing.get('sweet_spot', min_price + (max_price - min_price) * 0.6)

        broker_offer = classification.extracted_price or context.last_broker_price or 0
        broker_company = load_offer.get('brokerCompany', load_offer.get('brokerContactEmail', 'Broker'))
        route = f"{load_offer.get('pickupLocation', 'Origin')} → {load_offer.get('dropoffLocation', 'Dest')}"

        # Step 4: Initialize new metadata from previous state
        new_metadata = NegotiationMetadata(
            negotiationRound=prev_metadata.negotiationRound,
            infoExchangeCount=prev_metadata.infoExchangeCount,
            conversationState=prev_metadata.conversationState,
            lastCarrierPrice=prev_metadata.lastCarrierPrice,
            lastBrokerPrice=broker_offer if broker_offer > 0 else prev_metadata.lastBrokerPrice,
            belowMinCount=prev_metadata.belowMinCount,
            extractedPrice=classification.extracted_price,
            confidence=str(classification.confidence),
            timestamp=datetime.utcnow().isoformat()
        )

        # Update below_min_count if broker offer is below floor
        if broker_offer > 0 and broker_offer < min_price * 0.95:
            new_metadata.belowMinCount += 1

        # Step 5: Handle intent
        result = self._handle_intent(
            intent=classification.intent,
            context=context,
            new_metadata=new_metadata,
            broker_message=broker_message,
            broker_offer=broker_offer,
            min_price=min_price,
            max_price=max_price,
            sweet_spot=sweet_spot,
            broker_company=broker_company,
            route=route,
            carrier_info=carrier_info
        )

        self.logger.info(f"Result - Action: {result.action.value}")
        self.logger.info(f"Result - Status: {result.status}")
        self.logger.info(f"Result - Proposed Price: ${result.proposed_price}")
        self.logger.info(f"Result - New Negotiation Round: {result.metadata.negotiationRound}")
        self.logger.info(f"Result - New Info Exchange Count: {result.metadata.infoExchangeCount}")
        self.logger.info("="*60)

        return result

    def _handle_intent(
        self,
        intent: MessageIntent,
        context: ConversationContext,
        new_metadata: NegotiationMetadata,
        broker_message: str,
        broker_offer: float,
        min_price: float,
        max_price: float,
        sweet_spot: float,
        broker_company: str,
        route: str,
        carrier_info: Optional[Dict]
    ) -> NegotiationResult:
        """Handle classified intent and return result with metadata"""

        # Price Acceptance
        if intent == MessageIntent.PRICE_ACCEPTANCE:
            new_metadata.messageType = "PRICE_ACCEPTANCE"
            new_metadata.conversationState = ConversationState.ACCEPTED.value

            return NegotiationResult(
                action=NegotiationAction.ACCEPT,
                response=self.response_generator.generate(
                    NegotiationAction.ACCEPT, broker_company, None, broker_offer, route
                ),
                proposed_price=None,
                status="accepted",
                metadata=new_metadata,
                reasoning="Broker accepted our price"
            )

        # Price Rejection
        if intent == MessageIntent.PRICE_REJECTION:
            new_metadata.messageType = "PRICE_REJECTION"
            new_metadata.conversationState = ConversationState.REJECTED.value

            return NegotiationResult(
                action=NegotiationAction.REJECT,
                response="Thanks for considering us. Feel free to reach out on future loads.",
                proposed_price=None,
                status="rejected",
                metadata=new_metadata,
                reasoning="Broker rejected our offer"
            )

        # Price-related intents - INCREMENT negotiation round
        if intent in [MessageIntent.PRICE_OFFER, MessageIntent.PRICE_COUNTER, MessageIntent.PRICE_INQUIRY]:
            new_metadata.messageType = intent.value.upper()
            new_metadata.negotiationRound += 1  # INCREMENT HERE
            new_metadata.conversationState = ConversationState.NEGOTIATING.value

            return self._handle_price_negotiation(
                new_metadata=new_metadata,
                broker_offer=broker_offer,
                min_price=min_price,
                max_price=max_price,
                sweet_spot=sweet_spot,
                broker_company=broker_company,
                route=route
            )

        # Carrier Info Request - Broker asking about MC, DOT, equipment, etc.
        if intent == MessageIntent.CARRIER_INFO_REQUEST:
            new_metadata.messageType = "CARRIER_INFO_REQUEST"
            new_metadata.infoExchangeCount += 1
            new_metadata.conversationState = ConversationState.INFO_GATHERING.value

            response = self._generate_info_response(broker_message, carrier_info)

            # If carrier_info is missing, provide a generic response
            if not carrier_info:
                response = "Let me get that information for you. In the meantime, what are the load details?"

            return NegotiationResult(
                action=NegotiationAction.PROVIDE_INFO,
                response=response,
                proposed_price=None,
                status="negotiating",
                metadata=new_metadata,
                reasoning="Providing carrier information (MC, DOT, equipment, etc.)"
            )

        # Info Request - INCREMENT info exchange, NOT negotiation round
        if intent == MessageIntent.INFO_REQUEST:
            new_metadata.messageType = "INFO_REQUEST"
            new_metadata.infoExchangeCount += 1  # INCREMENT HERE
            new_metadata.conversationState = ConversationState.INFO_GATHERING.value

            response = self._generate_info_response(broker_message, carrier_info)

            # If carrier_info is missing, we can't answer, so ask for load details instead.
            if not carrier_info:
                response = "I can answer that, but first, could you please provide the load details, such as pickup/delivery, dates, and weight?"

            return NegotiationResult(
                action=NegotiationAction.PROVIDE_INFO,
                response=response,
                proposed_price=None,
                status="negotiating",
                metadata=new_metadata,
                reasoning="Providing requested information or asking for load details if carrier info is missing."
            )

        # Info Response / Load Details - INCREMENT info exchange
        if intent in [MessageIntent.INFO_RESPONSE, MessageIntent.LOAD_DETAILS]:
            new_metadata.messageType = intent.value.upper()
            new_metadata.infoExchangeCount += 1

            if min_price > 0 and max_price > 0:
                new_metadata.conversationState = ConversationState.READY_TO_NEGOTIATE.value
                new_metadata.negotiationRound += 1
                new_metadata.lastCarrierPrice = max_price

                return NegotiationResult(
                    action=NegotiationAction.COUNTER,
                    response=self.response_generator.generate(
                        NegotiationAction.COUNTER, broker_company, max_price, broker_offer, route, is_first_message=True
                    ),
                    proposed_price=max_price,
                    status="negotiating",
                    metadata=new_metadata,
                    reasoning="Info received, providing initial rate"
                )
            else:
                return NegotiationResult(
                    action=NegotiationAction.REQUEST_INFO,
                    response="Thanks for the info! What rate are you thinking for this load?",
                    proposed_price=None,
                    status="negotiating",
                    metadata=new_metadata,
                    reasoning="Need rate info to proceed"
                )

        # Rate Confirmation
        if intent == MessageIntent.RATE_CONFIRMATION:
            new_metadata.messageType = "RATE_CONFIRMATION"
            new_metadata.conversationState = ConversationState.ACCEPTED.value

            return NegotiationResult(
                action=NegotiationAction.CONFIRM_BOOKING,
                response="Got the rate con. We're booked and ready to roll!",
                proposed_price=None,
                status="accepted",
                metadata=new_metadata,
                reasoning="Rate confirmation received"
            )

        # Booking Request
        if intent == MessageIntent.BOOKING_REQUEST:
            new_metadata.messageType = "BOOKING_REQUEST"
            new_metadata.conversationState = ConversationState.ACCEPTED.value

            return NegotiationResult(
                action=NegotiationAction.ACCEPT,
                response=self.response_generator.generate(
                    NegotiationAction.ACCEPT, broker_company, None, broker_offer, route
                ),
                proposed_price=None,
                status="accepted",
                metadata=new_metadata,
                reasoning="Booking requested"
            )

        # Unclear
        new_metadata.messageType = "UNCLEAR"
        return NegotiationResult(
            action=NegotiationAction.CLARIFY,
            response=self.response_generator.generate(
                NegotiationAction.CLARIFY, broker_company, None, broker_offer, route
            ),
            proposed_price=None,
            status="negotiating",
            metadata=new_metadata,
            reasoning="Intent unclear, requesting clarification"
        )

    def _handle_price_negotiation(
        self,
        new_metadata: NegotiationMetadata,
        broker_offer: float,
        min_price: float,
        max_price: float,
        sweet_spot: float,
        broker_company: str,
        route: str
    ) -> NegotiationResult:
        """Handle price negotiation using persisted round count"""

        negotiation_round = new_metadata.negotiationRound
        below_min_count = new_metadata.belowMinCount
        last_carrier_price = new_metadata.lastCarrierPrice

        self.logger.info(f"Price Negotiation - Round: {negotiation_round}")
        self.logger.info(f"Price Negotiation - Broker Offer: ${broker_offer}")

        # Check for acceptance
        if NegotiationStrategy.should_accept(broker_offer, min_price, max_price, negotiation_round):
            new_metadata.conversationState = ConversationState.ACCEPTED.value
            return NegotiationResult(
                action=NegotiationAction.ACCEPT,
                response=self.response_generator.generate(
                    NegotiationAction.ACCEPT, broker_company, None, broker_offer, route, min_price
                ),
                proposed_price=None,
                status="accepted",
                metadata=new_metadata,
                reasoning=f"Accepting ${broker_offer} in round {negotiation_round}"
            )

        # Check for rejection
        if NegotiationStrategy.should_reject(broker_offer, min_price, negotiation_round, below_min_count):
            new_metadata.conversationState = ConversationState.REJECTED.value
            return NegotiationResult(
                action=NegotiationAction.REJECT,
                response=self.response_generator.generate(
                    NegotiationAction.REJECT, broker_company, None, broker_offer, route, min_price
                ),
                proposed_price=None,
                status="rejected",
                metadata=new_metadata,
                reasoning=f"Rejecting ${broker_offer} after {negotiation_round} rounds"
            )

        # Calculate counter
        counter_price = NegotiationStrategy.calculate_counter(
            min_price=min_price,
            sweet_spot=sweet_spot,
            max_price=max_price,
            negotiation_rounds=negotiation_round,
            broker_offer=broker_offer,
            last_carrier_price=last_carrier_price
        )

        new_metadata.lastCarrierPrice = counter_price

        # First round means this is our initial response to broker's first price offer
        is_first_message = (negotiation_round == 1)

        return NegotiationResult(
            action=NegotiationAction.COUNTER,
            response=self.response_generator.generate(
                NegotiationAction.COUNTER, broker_company, counter_price, broker_offer, route, min_price, is_first_message
            ),
            proposed_price=counter_price,
            status="negotiating",
            metadata=new_metadata,
            reasoning=f"Counter ${counter_price} in round {negotiation_round}"
        )

    def _generate_info_response(self, broker_message: str, carrier_info: Optional[Dict]) -> str:
        """Generate response to info request"""
        if not carrier_info:
            return "Let me check on that and get back to you. What's the load details?"

        try:
            from services.information_seek import InformationSeeker
            seeker = InformationSeeker(data=carrier_info)
            return seeker.ask(question=broker_message)
        except Exception as e:
            self.logger.warning(f"Info seeker failed: {e}")
            return "Let me check on that. In the meantime, what's the rate you're thinking?"
