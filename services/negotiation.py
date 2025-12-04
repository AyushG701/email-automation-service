"""
Load Offer Negotiation Service - Integrated with Orchestrator
File: services/negotiation.py

Key Fixes:
1. Uses NegotiationOrchestrator for proper round counting
2. Separates info exchanges from price negotiations
3. Context-aware response generation
4. Multiple API calls for accuracy when needed
"""
import logging
import json
import re
from typing import Optional, Dict, List, Any
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
import os

from core.openai import get_llm

logger = logging.getLogger(__name__)


# =============================================================================
# PRICE EXTRACTOR - Robust Version
# =============================================================================

class PriceExtractor:
    """Extract prices from text messages with robust error handling"""

    PRICE_PATTERNS = [
        r'\$\s*([\d,]+(?:\.\d{2})?)',        # $1,500 or $1500.00
        r'([\d,]+)\s*(?:dollars?|usd)',       # 1500 dollars
        r'(?:^|\s)([\d]{4,5})(?:\s|$|\.)',    # Standalone 4-5 digit number
        r'([\d]+(?:\.\d+)?)\s*k\b',           # 1.5k
    ]

    @staticmethod
    def extract_price(message: str) -> Optional[float]:
        """Extract first valid price from message with fallbacks"""
        if not message or not isinstance(message, str):
            logger.warning("Invalid message for price extraction")
            return None

        for pattern in PriceExtractor.PRICE_PATTERNS:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                try:
                    clean_price = match.replace(',', '')
                    value = float(clean_price)

                    # Handle 'k' suffix
                    if 'k' in message.lower()[message.lower().find(match):]:
                        value *= 1000

                    # Realistic freight range
                    if 300 <= value <= 50000:
                        return value
                except (ValueError, TypeError):
                    continue

        return None


# =============================================================================
# CHAT HISTORY ANALYZER - FIXED VERSION
# =============================================================================

class ChatHistoryAnalyzer:
    """
    Analyze negotiation history with SEPARATE counters.

    KEY FIX: Counts info exchanges separately from price negotiations.
    """

    # Keywords for identifying message types
    PRICE_KEYWORDS = [
        r'\$\d+', r'\d+\s*dollars?', r'rate', r'price', r'bid', r'offer',
        r'can you do', r'i need', r'best i can', r'works for me', r'deal',
        r'accept', r'agreed', r'confirmed', r'too (high|low)', r'counter'
    ]

    INFO_KEYWORDS = [
        r'equipment', r'trailer', r'van', r'flatbed', r'reefer',
        r'pickup', r'delivery', r'date', r'time', r'weight', r'commodity',
        r'mc number', r'dot', r'insurance', r'location', r'miles',
        r'when', r'where', r'what type', r'how many', r'\?'
    ]

    @staticmethod
    def format_history(negotiations: List[Dict]) -> str:
        """Format negotiation list into natural chat history"""
        if not negotiations:
            return "First message in negotiation"

        lines = []
        for neg in negotiations[-5:]:  # Last 5 messages
            direction = neg.get('negotiationDirection', '').lower()
            message = neg.get('negotiationRawEmail', '').strip()
            rate = neg.get('rate')

            if not message:
                continue

            if direction == 'outgoing':
                prefix = "Me (Carrier)"
            elif direction == 'incoming':
                prefix = "Broker"
            else:
                continue

            if rate and float(rate) > 0:
                lines.append(f"{prefix} [${rate:.0f}]: {message[:100]}")
            else:
                lines.append(f"{prefix}: {message[:100]}")

        return "\n".join(lines) if lines else "First message in negotiation"

    @staticmethod
    def analyze_negotiation(
        chat_history: str,
        broker_message: str,
        min_price: float,
        negotiations: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Analyze negotiation state with SEPARATE counters.

        KEY FIX: info_exchanges vs negotiation_rounds
        """
        result = {
            'info_exchanges': 0,
            'negotiation_rounds': 0,
            'last_carrier_price': None,
            'broker_current_offer': None,
            'below_min_count': 0
        }

        # If we have the raw negotiations list, use it for accurate counting
        if negotiations:
            for neg in negotiations:
                direction = neg.get('negotiationDirection', '').lower()
                content = neg.get('negotiationRawEmail', '').lower()
                rate = neg.get('rate')

                if direction != 'outgoing':  # Only count carrier messages
                    continue

                # Determine if this is a price message or info message
                is_price_msg = any(re.search(p, content, re.I) for p in ChatHistoryAnalyzer.PRICE_KEYWORDS)
                is_info_msg = any(re.search(p, content, re.I) for p in ChatHistoryAnalyzer.INFO_KEYWORDS)
                has_price = PriceExtractor.extract_price(content) or (rate and float(rate) > 0)

                if is_price_msg and has_price:
                    result['negotiation_rounds'] += 1
                    result['last_carrier_price'] = float(rate) if rate else PriceExtractor.extract_price(content)
                elif is_info_msg and not is_price_msg:
                    result['info_exchanges'] += 1

            # Get broker's offers
            for neg in reversed(negotiations):
                if neg.get('negotiationDirection', '').lower() == 'incoming':
                    broker_price = PriceExtractor.extract_price(neg.get('negotiationRawEmail', ''))
                    if broker_price:
                        result['broker_current_offer'] = broker_price
                        if broker_price < min_price * 0.95:
                            result['below_min_count'] += 1

        # Fallback to chat_history string parsing
        else:
            lines = [l.strip() for l in chat_history.split('\n') if l.strip()]
            carrier_msgs = [l for l in lines if l.startswith('Me (Carrier)')]
            broker_msgs = [l for l in lines if l.startswith('Broker')]

            for line in carrier_msgs:
                price = PriceExtractor.extract_price(line)
                if price:
                    result['negotiation_rounds'] += 1
                    result['last_carrier_price'] = price
                else:
                    result['info_exchanges'] += 1

        # Get current broker offer from message
        current_offer = PriceExtractor.extract_price(broker_message)
        if current_offer:
            result['broker_current_offer'] = current_offer
            if current_offer < min_price * 0.95:
                result['below_min_count'] += 1

        # The REAL negotiation round for this response
        # Add 1 because we're about to respond with a price
        result['negotiation_round'] = result['negotiation_rounds'] + 1

        return result


# =============================================================================
# NEGOTIATION STRATEGY - Using TRUE Negotiation Rounds
# =============================================================================

class NegotiationStrategy:
    """
    Real-world negotiation decision engine.

    KEY FIX: Uses negotiation_rounds (price discussions only), NOT total messages.
    """

    @staticmethod
    def should_accept(
        broker_offer: float,
        min_price: float,
        max_price: float,
        negotiation_rounds: int  # TRUE negotiation rounds, not total messages
    ) -> bool:
        """Determine acceptance based on TRUE price negotiation rounds"""

        # Always accept if at or above max price
        if broker_offer >= max_price:
            return True

        # NEVER accept in first 2 NEGOTIATION rounds
        if negotiation_rounds <= 2:
            return False

        # Round 3+ of ACTUAL price discussion - accept if above floor
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
        """Determine rejection based on TRUE negotiation rounds"""

        # NEVER reject in first round - always counter
        if negotiation_rounds <= 1:
            return False

        # Reject very low offers (round 2+)
        if broker_offer < min_price * 0.80:
            return True

        # Persistent low offers
        if below_min_count >= 3:
            return True

        # Stalemate after many ACTUAL price discussions
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
        """Calculate strategic counter based on TRUE negotiation round"""

        # Round 1: Anchor high
        if negotiation_rounds <= 1:
            return max_price

        # Round 2: Move to sweet spot
        if negotiation_rounds == 2:
            return sweet_spot

        # Round 3+: Defend minimum with buffer
        buffer = max(50, min_price * 0.02)

        # If we've been countering, move slightly toward broker
        if last_carrier_price and last_carrier_price > min_price + buffer:
            new_price = (broker_offer + last_carrier_price) / 2
            return max(new_price, min_price + buffer)

        return min_price + buffer


# =============================================================================
# RESPONSE GENERATOR
# =============================================================================

class ResponseGenerator:
    """Generate natural, human-like negotiation responses"""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

    def generate_acceptance(self, broker_company: str = "Broker") -> str:
        """Generate acceptance with RC request"""
        responses = [
            "Perfect! Send rate confirmation to book this load.",
            "Sounds good! Please send RC so we can lock this in.",
            "Deal! Forward the rate confirmation to get this moving.",
            "That works! Send over the rate con and we're good to go."
        ]
        return responses[hash(broker_company) % len(responses)]

    def generate_rejection(self, min_price: float, broker_company: str = "Broker") -> str:
        """Generate polite rejection"""
        responses = [
            f"Sorry, ${min_price:.0f} is my floor for this lane. Maybe next load.",
            f"Appreciate the offer but can't make it work below ${min_price:.0f}. Hit me up on the next one.",
            f"Thanks but I need at least ${min_price:.0f} to run this. Let's try another load.",
            f"Pass on this one - ${min_price:.0f} is my minimum for this route. Next time!"
        ]
        return responses[hash(broker_company) % len(responses)]

    def generate_counter(self, counter_price: float, broker_company: str = "Broker") -> str:
        """Generate counter offer"""
        responses = [
            f"I'd need ${counter_price:.0f} to move this. Can you do that?",
            f"Best I can do is ${counter_price:.0f} for this lane. Let me know.",
            f"${counter_price:.0f} would make this work. What do you think?",
            f"Looking at this run, ${counter_price:.0f} is where I need to be."
        ]
        return responses[hash(str(counter_price)) % len(responses)]

    def generate_llm_response(
        self,
        action: str,
        broker_message: str,
        counter_price: Optional[float],
        min_price: float,
        max_price: float,
        negotiation_round: int,
        broker_company: str,
        load_details: Dict
    ) -> str:
        """Generate response using LLM for complex scenarios"""
        if not self.client:
            # Fallback to templates
            if action == 'accept':
                return self.generate_acceptance(broker_company)
            elif action == 'reject':
                return self.generate_rejection(min_price, broker_company)
            else:
                return self.generate_counter(counter_price or max_price, broker_company)

        prompt = f"""You are a freight carrier negotiator. Generate a SHORT response (1-2 sentences max).

Action: {action}
Broker's message: {broker_message[:100]}
Our price: ${counter_price or 'N/A'}
Round: {negotiation_round}
Route: {load_details.get('pickupLocation', 'Origin')} → {load_details.get('deliveryLocation', 'Dest')}

Rules:
- MAX 2 sentences
- Sound human, casual but professional
- If accepting: ask for rate confirmation
- If countering: be firm but friendly
- If rejecting: be polite, mention minimum ${min_price}

Generate ONLY the response text:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM response failed: {e}")
            if action == 'accept':
                return self.generate_acceptance(broker_company)
            elif action == 'reject':
                return self.generate_rejection(min_price, broker_company)
            else:
                return self.generate_counter(counter_price or max_price, broker_company)


# =============================================================================
# MAIN NEGOTIATION SERVICE - FIXED VERSION
# =============================================================================

class NegotiationService:
    """
    Production-ready negotiation service.

    KEY FIXES:
    1. Separates info exchanges from price negotiations
    2. Uses TRUE negotiation rounds for decisions
    3. Context-aware response generation
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.response_generator = ResponseGenerator()

    def offer_negotiation(
        self,
        broker_message: str,
        min_price: float,
        max_price: float,
        chat_history: List[Dict],
        load_details: Dict
    ) -> Dict:
        """
        Main negotiation handler with fixed round counting.

        Args:
            broker_message: Latest broker message
            min_price: Minimum acceptable price
            max_price: Maximum/anchor price
            chat_history: List of negotiation history
            load_details: Load offer details

        Returns:
            Dict with response, proposed_price, and status
        """
        try:
            self.logger.info("="*60)
            self.logger.info("NEGOTIATION SERVICE - Processing")
            self.logger.info("="*60)

            # Validate inputs
            if min_price <= 0 or max_price <= 0:
                raise ValueError("Invalid price parameters")

            # Calculate sweet spot
            sweet_spot = min_price + (max_price - min_price) * 0.6

            # Format and analyze history with SEPARATE counters
            formatted_history = ChatHistoryAnalyzer.format_history(chat_history)
            analysis = ChatHistoryAnalyzer.analyze_negotiation(
                formatted_history,
                broker_message,
                min_price,
                chat_history  # Pass raw history for accurate counting
            )

            self.logger.info(f"Analysis - Info exchanges: {analysis['info_exchanges']}")
            self.logger.info(f"Analysis - Negotiation rounds: {analysis['negotiation_rounds']}")
            self.logger.info(f"Analysis - Current round: {analysis['negotiation_round']}")
            self.logger.info(f"Analysis - Broker offer: ${analysis['broker_current_offer']}")
            self.logger.info(f"Analysis - Last carrier price: ${analysis['last_carrier_price']}")

            broker_offer = analysis['broker_current_offer'] or 0
            negotiation_round = analysis['negotiation_round']  # TRUE negotiation round
            below_min_count = analysis['below_min_count']
            last_carrier_price = analysis['last_carrier_price']

            broker_company = load_details.get('brokerCompany',
                            load_details.get('brokerContactEmail', 'Broker'))

            # DECISION ENGINE using TRUE negotiation rounds

            # Check for acceptance
            if broker_offer > 0 and NegotiationStrategy.should_accept(
                broker_offer, min_price, max_price, negotiation_round
            ):
                self.logger.info(f"ACCEPTING ${broker_offer} in round {negotiation_round}")
                return {
                    'response': self.response_generator.generate_acceptance(broker_company),
                    'proposed_price': None,
                    'status': 'accepted'
                }

            # Check for rejection
            if broker_offer > 0 and NegotiationStrategy.should_reject(
                broker_offer, min_price, negotiation_round, below_min_count
            ):
                self.logger.info(f"REJECTING ${broker_offer} after {negotiation_round} rounds")
                return {
                    'response': self.response_generator.generate_rejection(min_price, broker_company),
                    'proposed_price': None,
                    'status': 'rejected'
                }

            # Calculate counter offer
            counter_price = NegotiationStrategy.calculate_counter(
                min_price=min_price,
                sweet_spot=sweet_spot,
                max_price=max_price,
                negotiation_rounds=negotiation_round,
                broker_offer=broker_offer,
                last_carrier_price=last_carrier_price
            )

            self.logger.info(f"COUNTERING with ${counter_price} in round {negotiation_round}")

            # Generate response
            response = self.response_generator.generate_llm_response(
                action='counter',
                broker_message=broker_message,
                counter_price=counter_price,
                min_price=min_price,
                max_price=max_price,
                negotiation_round=negotiation_round,
                broker_company=broker_company,
                load_details=load_details
            )

            return {
                'response': response,
                'proposed_price': f"{counter_price:.2f}",
                'status': 'negotiating'
            }

        except Exception as e:
            self.logger.error(f"Negotiation error: {str(e)}", exc_info=True)
            return {
                'response': "Let's discuss pricing. What rate works for you?",
                'proposed_price': None,
                'status': 'negotiating'
            }

    def handle_info_exchange(
        self,
        broker_message: str,
        carrier_info: Dict
    ) -> Dict:
        """
        Handle information exchange (non-price) messages.

        This does NOT increment negotiation rounds.
        """
        try:
            from services.information_seek import InformationSeeker
            seeker = InformationSeeker(data=carrier_info)
            response = seeker.ask(question=broker_message)

            return {
                'response': response,
                'proposed_price': None,
                'status': 'negotiating',
                'type': 'info_exchange'
            }
        except Exception as e:
            self.logger.warning(f"Info exchange failed: {e}")
            return {
                'response': "Let me check on that. What's the load details?",
                'proposed_price': None,
                'status': 'negotiating',
                'type': 'info_exchange'
            }


# =============================================================================
# LEGACY PROMPT (for reference, not used in main flow)
# =============================================================================

NEGOTIATION_SYSTEM_PROMPT = """You are an expert freight carrier negotiator. Your goal is to maximize profit while maintaining professional relationships.

**CRITICAL RULES:**

1. **ROUND COUNTING (IMPORTANT!):**
   - Only count PRICE discussions as negotiation rounds
   - Info questions (equipment, dates, etc.) do NOT count
   - This prevents premature acceptance/rejection

2. **ACCEPTANCE CRITERIA:**
   - Round 1-2: NEVER accept (always counter)
   - Round 3+: Accept if >= ${min_price:.2f}
   - ANY round: Accept immediately if >= ${max_price:.2f}

3. **COUNTER STRATEGY:**
   - Round 1: ${max_price:.2f} (anchor high)
   - Round 2: ${sweet_spot:.2f} (strategic drop)
   - Round 3+: ${min_price:.2f} + buffer
   - Never go below ${min_price:.2f}

4. **AFTER ACCEPTANCE - ALWAYS ASK FOR RC**

**RESPONSE FORMAT (JSON only):**
{{
  "response": "Natural, concise message (max 2 sentences)",
  "proposed_price": 1850.00 or null,
  "status": "negotiating" | "accepted" | "rejected"
}}
"""


def create_negotiation_prompt():
    """Create the negotiation prompt template"""
    return ChatPromptTemplate.from_messages([
        ("system", NEGOTIATION_SYSTEM_PROMPT),
        ("human", "Generate negotiation response")
    ])
