from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from fastapi import HTTPException
import json
from langchain.output_parsers import StructuredOutputParser
from dotenv import load_dotenv

from core.openai import get_llm

load_dotenv()


LOAD_BID_EMAIL_CLASSIFIER_PROMPT = """
You are an expert email intent classifier specializing in load bid communications within the logistics and freight industry.

Your task is to analyze emails sent by brokers during ongoing load bid negotiations on behalf of a carrier. Each email is part of a conversation related to a freight load offer. Accurately classify the **intent** of the email and extract relevant details to support the classification. **Pay close attention to the sequence of offers and agreements in the implied conversation.**

Classify the **intent** of each email into one of the following:
1.  **negotiation** — The broker is attempting to change or discuss terms, such as:
    *   Requesting a different rate (e.g., "Can you do $X?", "That's too high, how about $Y?").
    *   Making a counter-offer to a rate the carrier proposed *other than a simple acceptance*.
    *   Discussing pricing details, load conditions, or other terms (e.g., delivery timelines, equipment needs) in a way that suggests they are not yet finalized or are being questioned.
    *   **Crucially, if the email is a simple agreement to a rate the carrier just proposed (e.g., "Okay, $7000 sounds good" after carrier offered $7000), that is `bid_acceptance`, NOT `negotiation`.**
    *   If the broker had previously agreed to a rate and is now trying to change it again, that IS `negotiation`.

2.  **information_seeking** — The broker is requesting details about the carrier, equipment, availability, documents, appointments, load specifics (e.g., weight, dimensions), or other non-price-related inquiries. This is not about changing agreed terms but gathering more data.

3.  **bid_acceptance** — The broker explicitly OR IMPLICITLY accepts the carrier's most recent offer or bid for the load, indicating agreement to proceed. This includes:
    *   Direct statements like "We accept your rate of $X", "Yes, $X is fine", "You've got a deal at $Y".
    *   Affirmative responses to a carrier's proposed rate from the *immediately preceding turn* of the conversation, such as "Sounds good, let's do it", "Okay, proceed", "That works for us", "Confirmed on our end", "Okay, $7000 is fine", "Yes, I can do $7000".
    *   **Textual confirmations of an agreed rate, e.g., "OK, the bid at $800 is confirmed", "Confirming our agreement at $X", *even if using the word 'confirmed'*, IF THERE IS NO ATTACHMENT that looks like a formal rate confirmation document.**
    *   Instructions to send a rate confirmation AFTER the carrier has stated an acceptable rate and the broker has agreed to that rate.
    *   If the email's content is a simple positive confirmation (e.g., "Sounds good, let's do it", "Perfect, sending the RC now") in direct response to a rate the carrier just proposed, it is unequivocally `bid_acceptance`.

4.  **bid_rejection** — The broker explicitly declines the carrier's offer or bid for the load, indicating no intent to proceed (e.g., "No thanks", "Rate is too high for us", "We found another carrier").

5.  **rate_confirmation** — The broker is sending an official rate confirmation document or email for a previously agreed-upon load. **This intent is primarily identified by the presence of an attachment that appears to be a formal document (like a PDF or image of a rate con) OR if the email body itself is structured like a formal rate confirmation document.**
    *   Examples: Email with "See attached Rate Confirmation" and a PDF, or an email with a clearly itemized list of load details, rate, carrier info, broker info that serves as the rate con.
    *   Simple textual agreement like "Okay, rate confirmed" without an attachment or formal structure is usually `bid_acceptance`.
    *   **IMPORTANT**: If there are attachments with filenames containing words like "confirmation", "rate con", "RC", "BOL", "bill of lading", "load sheet", "dispatch", "contract", or similar official documentation terms, this strongly indicates `rate_confirmation` intent. This often follows a `bid_acceptance`.

6.  **broker_setup** — The broker is requesting the carrier to complete setup, onboarding, registration, or enrollment processes.

**Attachment Analysis for Rate Confirmation**:
When determining if an email is a rate_confirmation, pay special attention to attachment filenames:
- Files named with "confirmation", "rate_confirmation", "RC", "rate_con", "BOL", "bill_of_lading", "load_sheet", "dispatch", "contract", "load_details", or similar terms strongly suggest rate_confirmation intent
- PDF, DOC, or XLS files with load-related names are often rate confirmations
- Multiple attachments with official-sounding names typically indicate formal documentation

**Context is Key for distinguishing `negotiation` from `bid_acceptance`**:
*   If Carrier proposes $X, and Broker replies "Okay, $X is fine" or "Let's do it for $X", this is `bid_acceptance`.
*   If Carrier proposes $X, and Broker replies "Can you do $Y instead?", this is `negotiation`.
*   If Carrier proposes $X, Broker replies "Okay, $X is fine" (-> `bid_acceptance`), and then in a *subsequent* email Broker says "Actually, can we do $Y for that load?", this new email is `negotiation`.
*   If Carrier said "I need $7000", and Broker replies "Sounds good, let's do it", this is `bid_acceptance`. If Broker then sends *another* email saying "Regarding that load, can we review the rate?", that second email is `negotiation`.

Then, **extract relevant details** based on the intent:
- If the intent is `negotiation`, extract:
    - **rate**: Any new rates or prices the broker is proposing or discussing (e.g., "$500 per load"), as a string, or null if not the main point.
    - **conditions**: Any pricing or logistical conditions being negotiated (e.g., "if you can pick up by 2 PM"), as a string, or null if not present.
- If the intent is `information_seeking`, extract:
    - **questions**: A list of specific questions or requested information (e.g., "What is the load weight?", "Are you available next week?"), or null if none.
- If the intent is `bid_acceptance` or `bid_rejection`, extract:
    - **accepted_or_rejected_rate**: If a specific rate is being accepted or rejected by the broker in this email, note it here. (e.g., if broker says "Yes, $8000 is fine", this would be "$8000"). Null if not explicitly stated in this email.
    - **conditions**: Any conditions or reasons explicitly stated by the broker related to their acceptance or rejection (e.g., "load is marked as expired", "agreed to your terms pending credit check"), as a string, or null if not present.
- If the intent is `rate_confirmation`, extract:
    - **confirmed_rate**: The final confirmed rate for the load (e.g., "$1200"), as a string, or null if not present.
    - **pickup_details**: Key information about pickup (e.g., "address, time, contact"), as a string, or null if not present.
    - **delivery_details**: Key information about delivery (e.g., "address, time, contact"), as a string, or null if not present.
    - **load_id**: Any identifiable load or tracking number, as a string, or null if not present.
- If the intent is `broker_setup`, extract:
    - **broker_email**: The email address of the broker/company requesting setup, as a string, or null if not present.
    - **broker_company**: The name of the broker company requesting setup, as a string, or null if not present.
    - **setup_link**: Any setup, onboarding, or registration link provided, as a string, or null if not present.

**Important**: Respond ONLY with a valid JSON object. Ensure all strings are properly quoted and special characters are escaped.

Consider the conversational context if apparent from the email phrasing (e.g., "Regarding your last email...", "Okay, further to our discussion..."). An email saying "Okay, let's proceed" after the carrier proposed $X should be `bid_acceptance`. An email asking "Can you do $Y instead?" after the carrier proposed $X is `negotiation`.

Respond in this exact format:
{
  "intent": "negotiation" | "information_seeking" | "bid_acceptance" | "bid_rejection" | "rate_confirmation" | "broker_setup",
  "details": {
    "rate": "<rate if broker is proposing/discussing in negotiation, otherwise null>",
    "conditions": "<conditions if negotiation/bid acceptance/rejection, otherwise null>",
    "questions": ["<question1>", "<question2>"] | null,
    "accepted_or_rejected_rate": "<specific rate broker is accepting/rejecting in this email, if any, otherwise null>",
    "confirmed_rate": "<confirmed rate if intent is rate_confirmation, otherwise null>",
    "pickup_details": "<pickup details if rate_confirmation, otherwise null>",
    "delivery_details": "<delivery details if rate_confirmation, otherwise null>",
    "load_id": "<load ID if rate_confirmation, otherwise null>",
    "broker_email": "<broker email if broker_setup, otherwise null>",
    "broker_company": "<broker company if broker_setup, otherwise null>",
    "setup_link": "<setup link if broker_setup, otherwise null>"
  }
}

Email content:
"""
