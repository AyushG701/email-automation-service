
"""
Load Offer Negotiation Service - OPTIMIZED VERSION
File: services/negotiation.py

Key Improvements:
- Ultra-concise real-world messaging style
- Aggressive first-round max_price anchoring
- Robust handling of missing fields
- Chat-history-driven decision making
- Natural language flow with broker name integration
- Perfect RC handling after acceptance
- Error-tolerant price extraction
"""

import logging
import json
import re
from typing import Optional, Dict, List
from langchain.prompts import ChatPromptTemplate
from core.openai import get_llm

logger = logging.getLogger(__name__)

# ============================================================================
# NEGOTIATION PROMPT - REAL-WORLD OPTIMIZED
# ============================================================================

NEGOTIATION_SYSTEM_PROMPT = """You are an expert freight carrier negotiator. Your goal is to maximize profit while maintaining professional relationships.

**YOUR PERSONALITY:**
- Confident but friendly
- Professional yet conversational
- Personable - use broker's company name naturally
- Concise communicator - keep messages short (1-2 sentences max)

**CRITICAL NEGOTIATION RULES:**

1. **FIRST ROUND AGGRESSION:**
   - ALWAYS counter at ${max_price:.2f} in first round
   - Never accept first offer unless >= ${max_price:.2f}

2. **ACCEPTANCE CRITERIA:**
   - Round 1-2: NEVER accept (keep negotiating)
   - Round 3+: Accept if >= ${min_price:.2f} AND we've negotiated well
   - ANY round: Accept immediately if >= ${max_price:.2f}
   
3. **AFTER ACCEPTANCE - ALWAYS ASK FOR RC:**
   - "Perfect! Send rate confirmation to book."
   - "Great! Please send RC so we can lock this in."
   - Keep it casual but mandatory

4. **COUNTER STRATEGY:**
   - Round 1: ${max_price:.2f} (anchor high)
   - Round 2: ${sweet_spot:.2f} (strategic drop)
   - Round 3+: ${min_price:.2f} + small buffer
   - Never go below ${min_price:.2f}

5. **REJECTION:**
   - Reject immediately if offer < ${min_price:.2f} * 0.85
   - Reject after 3 rounds if still < ${min_price:.2f}

**CONVERSATION CONTEXT:**

Load Details:
- Broker: {broker_company}
- Route: {pickup_location} → {delivery_location}
- Equipment: {equipment_type} | Weight: {weight} lbs

Pricing Strategy:
- Floor Price: ${min_price:.2f} (never go below)
- Target Price: ${sweet_spot:.2f} (ideal)
- Anchor Price: ${max_price:.2f} (first counter)

Negotiation State:
- Current Round: {negotiation_round}
- Broker's Offer: ${broker_offer:.2f}
- Our Last Offer: ${last_carrier_price}
- Below Floor Count: {below_min_count}

**CHAT HISTORY:**
{chat_history}

**BROKER'S MESSAGE:**
"{broker_message}"

**YOUR RESPONSE RULES:**
- MAX 2 SENTENCES - be concise like real freight negotiations
- Use broker's company name naturally once per message
- Sound human: "Hey {broker_company}" not "Dear Sir/Madam"
- NEVER use markdown, JSON, or special formatting
- If accepting: ALWAYS request RC in same message

**RESPONSE FORMAT (JSON only):**
{{
  "response": "Natural, concise message",
  "proposed_price": 1850.00 or null,
  "status": "negotiating" | "accepted" | "rejected",
  "reasoning": "1-sentence strategy explanation"
}}

**REAL-WORLD EXAMPLES:**

Round 1, Broker offers $1400:
{{
  "response": " I'd need ${max_price:.2f} to move this. What works for you?",
  "proposed_price": {max_price:.2f},
  "status": "negotiating",
  "reasoning": "First round - anchor at max price"
}}

Round 2, Broker offers $1600:
{{
  "response": " I can do ${sweet_spot:.2f} if we book today. Final offer.",
  "proposed_price": {sweet_spot:.2f},
  "status": "negotiating",
  "reasoning": "Second round - move to sweet spot"
}}

Round 3, Broker accepts $1680:
{{
  "response": "Perfect! Send rate confirmation to book this load.",
  "proposed_price": null,
  "status": "accepted",
  "reasoning": "Acceptance with RC request"
}}

Round 3, Broker offers $1450 (below min):
{{
  "response": "Sorry, ${min_price:.2f} is my absolute floor for this lane. Maybe next load.",
  "proposed_price": null,
  "status": "rejected",
  "reasoning": "Below minimum after 3 rounds"
}}
"""


def create_negotiation_prompt():
    """Create the negotiation prompt template"""
    return ChatPromptTemplate.from_messages([
        ("system", NEGOTIATION_SYSTEM_PROMPT),
        ("human", "Generate negotiation response")
    ])


# ============================================================================
# PRICE EXTRACTION - ROBUST VERSION
# ============================================================================

class PriceExtractor:
    """Extract prices from text messages with robust error handling"""

    @staticmethod
    def extract_price(message: str) -> Optional[float]:
        """Extract first valid price from message with fallbacks"""
        if not message or not isinstance(message, str):
            logger.warning("Invalid message for price extraction")
            return None

        # Try primary pattern first (most common in freight)
        primary_match = re.search(
            r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', message)
        if primary_match:
            clean_price = primary_match.group(1).replace(',', '')
            try:
                value = float(clean_price)
                if 300 <= value <= 50000:  # Realistic freight range
                    return value
            except (ValueError, TypeError):
                pass

        # Fallback patterns
        fallback_patterns = [
            r'(\d+)k',          # 1.5k
            r'(\d{4,5})\b',     # 1500
            r'(\d+)\s*k',       # 1.5 k
        ]

        for pattern in fallback_patterns:
            matches = re.findall(pattern, message.lower())
            for match in matches:
                try:
                    if 'k' in str(match):
                        value = float(match.replace('k', '')) * 1000
                    else:
                        value = float(match)

                    if 300 <= value <= 50000:
                        return value
                except (ValueError, TypeError):
                    continue

        logger.warning(f"No valid price found in message: {message[:50]}...")
        return None


# ============================================================================
# CHAT HISTORY ANALYZER - HISTORY-FIRST APPROACH
# ============================================================================

class ChatHistoryAnalyzer:
    """Analyze negotiation history with chat-first priority"""

    @staticmethod
    def format_history(negotiations: List[Dict]) -> str:
        """Format negotiation list into natural chat history"""
        if not negotiations:
            return "First message in negotiation"

        lines = []
        for i, neg in enumerate(negotiations[-3:], 1):  # Only last 3 messages
            direction = neg.get('negotiationDirection', '').lower()
            message = neg.get('negotiationRawEmail', '').strip()
            rate = 0

            if not message:
                continue

            if direction == 'outgoing':
                prefix = "Me (Carrier)"
            elif direction == 'incoming':
                prefix = "Broker"
            else:
                continue

            # Add rate context only if available
            if rate and rate > 0:
                lines.append(f"{prefix} [${rate:.0f}]: {message}")
            else:
                lines.append(f"{prefix}: {message}")

        return "\n".join(lines[-3:])  # Keep it concise

    @staticmethod
    def analyze_negotiation(
        chat_history: str,
        broker_message: str,
        min_price: float,
       
    ) -> Dict:
        """Analyze negotiation state with chat-history priority"""

        # Count negotiation rounds based on messages
        lines = [l.strip() for l in chat_history.split('\n') if l.strip()]
        carrier_msgs = [l for l in lines if l.startswith('Me (Carrier)')]
        broker_msgs = [l for l in lines if l.startswith('Broker')]

        # Determine round number (1-based)
        negotiation_round = len(carrier_msgs) + 1

        # Extract last prices from history
        last_carrier_price = None
        for line in reversed(carrier_msgs):
            price = PriceExtractor.extract_price(line)
            if price:
                last_carrier_price = price
                break

        # Get current broker offer
        broker_current_offer = PriceExtractor.extract_price(broker_message)

        # Fallback to last known broker offer if current missing
        if broker_current_offer is None and broker_msgs:
            for line in reversed(broker_msgs):
                price = PriceExtractor.extract_price(line)
                if price:
                    broker_current_offer = price
                    break
         # Ultimate fallback: use base_rate if all else fails
        if broker_current_offer is None or broker_current_offer == 0.0:
            broker_current_offer = base_rate if base_rate and base_rate > 0 else 0.0
        broker_current_offer = broker_current_offer or 0.0

        # Count below-min offers
        below_min_count = 0
        all_broker_msgs = broker_msgs + [f"Broker: {broker_message}"]
        for line in all_broker_msgs:
            price = PriceExtractor.extract_price(line)
            if price and price < min_price * 0.95:  # 5% buffer
                below_min_count += 1

        return {
            'negotiation_round': negotiation_round,
            'last_carrier_price': last_carrier_price,
            'broker_current_offer': broker_current_offer,
            'below_min_count': below_min_count
        }


# ============================================================================
# NEGOTIATION STRATEGY - REAL-WORLD LOGIC
# ============================================================================

class NegotiationStrategy:
    """Real-world negotiation decision engine"""

    @staticmethod
    def should_accept(
        broker_offer: float,
        min_price: float,
        max_price: float,
        round_num: int
    ) -> bool:
        """Determine acceptance with real-world logic"""

        # Always accept if at max price
        if broker_offer >= max_price:
            return True

        # Never accept first two rounds unless at max price
        if round_num <= 2:
            return False

        # Accept after round 2 if above minimum
        if round_num >= 3 and broker_offer >= min_price * 0.98:  # 2% buffer
            return True

        return False

    
    @staticmethod
    def should_reject(
        broker_offer: float,
        min_price: float,
        round_num: int,
        below_min_count: int
    ) -> bool:
        # NEVER reject in first round - always counter
        if round_num == 1:
            return False

        # Immediate rejection for very low offers (round 2+)
        if broker_offer < min_price * 0.85:
            return True

        # Persistent low offers
        if below_min_count >= 3:
            return True

        # Stalemate after multiple rounds
        if round_num >=6 and broker_offer < min_price * 0.95:
            return True

        return False


    @staticmethod
    def calculate_counter(
        min_price: float,
        sweet_spot: float,
        max_price: float,
        round_num: int,
        broker_offer: float
    ) -> float:
        """Calculate strategic counter with round-based logic"""

        if round_num == 1:
            return max_price  # Anchor high

        if round_num == 2:
            return sweet_spot  # Strategic drop

        # Round 3+ - fight for minimum + buffer
        buffer = max(50, min_price * 0.03)  # Min $50 or 3% buffer
        return min_price + buffer


# ============================================================================
# MAIN NEGOTIATION SERVICE - OPTIMIZED
# ============================================================================

class NegotiationService:
    """Production-ready negotiation service"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def offer_negotiation(
        self,
        broker_message: str,
        min_price: float,
        max_price: float,
        chat_history: List[Dict],
        load_details: Dict
    ) -> Dict:
        """
        Main negotiation handler with real-world optimizations
        """
        try:
            # Validate essential inputs
            if min_price <= 0 or max_price <= 0:
                raise ValueError("Invalid price parameters")

            # Calculate strategic prices
            sweet_spot = min_price + (max_price - min_price) * 0.6

            # Format chat history (concise)
            formatted_history = ChatHistoryAnalyzer.format_history(
                chat_history)
                
            base_rate = load_details.get('requestedRate')
            if base_rate and isinstance(base_rate, str):
                try:
                    base_rate = float(base_rate)
                except (ValueError, TypeError):
                    base_rate = None

            # Analyze negotiation state
            analysis = ChatHistoryAnalyzer.analyze_negotiation(
                formatted_history,
                broker_message,
                min_price
            )

            broker_offer = analysis['broker_current_offer']
            round_num = analysis['negotiation_round']

            # DECISION ENGINE

            # Acceptance check
            if NegotiationStrategy.should_accept(
                broker_offer,
                min_price,
                max_price,
                round_num
            ):
                return {
                    'response': self._generate_acceptance_message(
                        broker_offer,
                        load_details.get('brokerCompany', 'Broker')
                    ),
                    'proposed_price': None,
                    'status': 'accepted'
                }

            # Rejection check
            if NegotiationStrategy.should_reject(
                broker_offer,
                min_price,
                round_num,
                analysis['below_min_count']
            ):
                return {
                    'response': self._generate_rejection_message(
                        min_price,
                        load_details.get('brokerCompany', 'Broker')
                    ),
                    'proposed_price': None,
                    'status': 'rejected'
                }

            # Calculate counter offer
            counter_price = NegotiationStrategy.calculate_counter(
                min_price,
                sweet_spot,
                max_price,
                round_num,
                broker_offer
            )

            # Generate AI response (with fallback)
            try:
                return self._generate_llm_response(
                    broker_message=broker_message,
                    chat_history=formatted_history,
                    broker_offer=broker_offer,
                    counter_price=counter_price,
                    min_price=min_price,
                    sweet_spot=sweet_spot,
                    max_price=max_price,
                    round_num=round_num,
                    last_carrier_price=analysis['last_carrier_price'],
                    below_min_count=analysis['below_min_count'],
                    load_details=load_details
                )
            except Exception as llm_error:
                self.logger.warning(
                    f"LLM fallback triggered: {str(llm_error)}")
                return {
                    'response': self._generate_fallback_counter(
                        counter_price,
                        load_details.get('brokerCompany', 'Broker')
                    ),
                    'proposed_price': f"{counter_price:.2f}",
                    'status': 'negotiating'
                }

        except Exception as e:
            self.logger.error(
                f"Critical negotiation error: {str(e)}", exc_info=True)
            # Safe fallback response
            return {
                'response': f"Let's discuss pricing. What rate works for you?",
                'proposed_price': None,
                'status': 'negotiating'
            }

    def _generate_llm_response(
        self,
        broker_message: str,
        chat_history: str,
        broker_offer: float,
        counter_price: float,
        min_price: float,
        sweet_spot: float,
        max_price: float,
        round_num: int,
        last_carrier_price: Optional[float],
        below_min_count: int,
        load_details: Dict
    ) -> Dict:
        """Generate concise, natural LLM response"""

        # Build context with fallbacks for missing fields
        context = {
            'broker_company': "Broker" [:20],
            #  load_details.get('brokerContactEmail', 'Broker')[:20],
            'pickup_location': load_details.get('pickupLocation', 'Origin')[:30],
            'delivery_location': load_details.get('deliveryLocation', 'Destination')[:30],
            'equipment_type': load_details.get('equipmentType', 'Van')[:20],
            'weight': load_details.get('weightLbs', 40000),
            'pickup_date': load_details.get('pickupDate', 'TBD')[:10],
            'delivery_date': load_details.get('deliveryDate', 'TBD')[:10],
            'min_price': min_price,
            'sweet_spot': sweet_spot,
            'max_price': max_price,
            'broker_offer': broker_offer,
            'negotiation_round': round_num,
            'last_carrier_price': last_carrier_price or 0,
            'below_min_count': below_min_count,
            'chat_history': chat_history,
            'broker_message': broker_message[:100]  # Prevent overflow
        }

        print(context)

        # Generate response
        llm = get_llm(temperature=0.2)  # Lower temp for consistency
        prompt = create_negotiation_prompt()
        response = llm.invoke(prompt.format_messages(**context))

        # Parse response safely
        try:
            content_str = str(response.content) if not isinstance(
                response.content, str) else response.content
            json_str = re.sub(r'^```json\s*|\s*```$', '',
                              content_str, flags=re.MULTILINE)
            result = json.loads(json_str)

            # Validate response structure
            if 'response' not in result:
                raise ValueError("Missing response field")

            # Format price
            proposed_price = result.get('proposed_price')
            if proposed_price and isinstance(proposed_price, (int, float)):
                proposed_price = f"{float(proposed_price):.2f}"

            return {
                'response': self._sanitize_response(result['response']),
                'proposed_price': proposed_price,
                'status': result.get('status', 'negotiating')
            }
        except Exception as e:
            self.logger.error(f"Response parsing failed: {str(e)}")
            raise

    def _sanitize_response(self, response: str) -> str:
        """Clean and truncate response to real-world length"""
        # Remove any JSON artifacts or markdown
        clean = re.sub(r'```json|```|{|}|"', '', response)
        clean = clean.strip()

        # Enforce maximum length (real freight messages are short)
        if len(clean) > 120:
            clean = clean[:117] + "..."

        # Ensure it ends properly
        if not clean.endswith(('.', '!', '?')):
            clean += '.'

        return clean

    def _generate_acceptance_message(self, price: float, broker_company: str) -> str:
        """Ultra-concise acceptance with RC request"""
        messages = [
            f"Perfect! Send rate confirmation to book this load.",
            f"Great! Please send RC so we can lock this in.",
            f"Deal! Forward the rate confirmation to schedule.",
            f"Confirmed! Send RC to get this moving."
        ]
        return messages[hash(broker_company) % len(messages)]

    def _generate_rejection_message(self, min_price: float, broker_company: str) -> str:
        """Polite but firm rejection"""
        messages = [
            f"Sorry, ${min_price:.0f} is my floor for this lane. Maybe next load.",
            f"{broker_company}, I can't move below ${min_price:.0f}. Appreciate your time.",
            f"Thanks but no - ${min_price:.0f} is minimum for this route. Let's try another load.",
            f"Pass on this one {broker_company}. Need at least ${min_price:.0f} to run this."
        ]
        return messages[hash(broker_company) % len(messages)]

    def _generate_fallback_counter(self, counter_price: float, broker_company: str) -> str:
        """Natural fallback counter offer"""
        messages = [
            f" I'd need ${counter_price:.0f} to move this. What works?",
            f" best I can do is ${counter_price:.0f} for this lane.",
            f"Looking at this run, ${counter_price:.0f} would work. Can you approve?",
            f"${counter_price:.0f} would make this work for me. Let me know."
        ]
        return messages[hash(broker_company) % len(messages)]

