# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from typing import Optional, Any
# from memory.memory_manager import MemoryManager
# from agents.negotiation_agent import NegotiationAgent
# from agents.email_classifier_agent import EmailClassifierAgent
# from schemas.bid_email_intent_classifier import LoadBidEmailClassificationRequest, LoadBidEmailClassificationResponseStructured
# from schemas.negotiation import NegotiationRequest, NegotiationResponse
# import re
# from schemas.carrier_data import QueryPayload
# from agents.carrier_data_retrival_agent import fetch_query_data, get_schema_name
# from agents.load_negotiation import LoadNegotiationAgent, LoadNegotiationRequest, LoadNegotiationResponse
# from pydantic import BaseModel
# import psycopg2
# from psycopg2.extras import RealDictCursor
# import os
# from dotenv import load_dotenv
# from models.email_composer import EmailComposerRequest, EmailComposerResponse
# from tools.email_composer import LoadNegotiationComposer
# from common.utils.logger import get_logger
# from schemas.broker_setup_extract import BrokerSetupExtractRequest, BrokerSetupExtractResponse, BrokerSetupExtractDetails
# from services.broker_setup_extractor import BrokerSetupExtractor
# from agents.negotiation_agent import NegotiationAgent, NegotiationRequest
# import json
# from models.llm_setup import get_llm
# from prompts.negotiation_prompt import create_negotiation_prompt
# from memory.negotiation_history_manager import get_formatted_chat_history, add_negotiation_message, create_history_table_if_not_exists

# # Initialize logger
# logger = get_logger(__name__)

# # Call this once at application startup (e.g., in main.py or at the top of api.py if appropriate)
# # For now, let's add a check within the endpoint, or ensure it's called during app setup.
# # A better place might be in your application's startup event handler if using FastAPI's lifespan events.
# create_history_table_if_not_exists()

# load_dotenv()
# DB_CONFIG = {
#     "host": os.getenv("host", "services.supertruck.ai"),
#     "port": os.getenv("port", 5432),
#     "user": os.getenv("user", "st_user"),
#     "password": os.getenv("password", "st_password"),
#     "dbname": os.getenv("dbname", "st_db")
# }


# class NegotiationQuery(BaseModel):
#     load_id: str
#     broker_id: str
#     carrier_id: str


# router = APIRouter(prefix='/api/v1', tags=['chat'])
# # Still initialized for potential use by other agents
# memory_manager = MemoryManager()
# composer = LoadNegotiationComposer()

# # Initialize agents
# email_classifier_agent = EmailClassifierAgent()


# @router.post('/negotiate', response_model=dict)
# async def negotiate(request: NegotiationRequest):
#     """
#     Handles negotiation requests from brokers.
#     Processes load offers and generates carrier responses using prompts directly.
#     """
#     try:
#         # Get LLM instance
#         llm = get_llm(temperature=0.5)
#         prompt = create_negotiation_prompt()

#         # Prepare input data for the prompt
#         chat_history = ""  # In a real implementation, you'd fetch this from database

#         # Format the prompt with the request data
#         formatted_prompt = prompt.format_messages(
#             input=request.message,
#             chat_history=chat_history,
#             min_price=request.min_price,
#             max_price=request.max_price
#         )

#         # Invoke the LLM directly
#         response = llm.invoke(formatted_prompt)

#         # Parse the JSON response
#         json_string = response.content.strip().replace(
#             "```json\n", "").replace("\n```", "")
#         negotiation_result = json.loads(json_string)

#         # Validate response structure
#         if not all(key in negotiation_result for key in ["response", "proposed_price", "status"]):
#             raise ValueError(
#                 "LLM output is missing required keys: 'response', 'proposed_price', 'status'")

#         # Handle proposed_price conversion
#         proposed_price = negotiation_result.get("proposed_price")
#         if proposed_price is not None and proposed_price != "null":
#             try:
#                 proposed_price = float(proposed_price)
#             except (ValueError, TypeError):
#                 # Extract price from response text as fallback
#                 price_match = re.search(
#                     r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', negotiation_result["response"])
#                 if price_match:
#                     proposed_price = float(
#                         price_match.group(1).replace(',', ''))
#                 else:
#                     proposed_price = None
#         else:
#             proposed_price = None

#         return {
#             "proposed_price": proposed_price,
#             "response": negotiation_result["response"],
#             "status": negotiation_result["status"]
#         }

#     except json.JSONDecodeError as e:
#         logger.error(f"Error decoding JSON from LLM: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail="Failed to parse negotiation response"
#         )
#     except Exception as e:
#         logger.error(f"Error in negotiate endpoint: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"An internal server error occurred during negotiation: {str(e)}"
#         )


# @router.post("/load_negotiation", response_model=LoadNegotiationResponse)
# async def load_negotiation(request: LoadNegotiationRequest):
#     """
#     Handle load negotiation requests from brokers using prompts directly.
#     Evaluates the request and responds with a counter-offer or decision.
#     """
#     # --- START: Log incoming request payload ---
#     try:
#         request_payload_dict = request.model_dump(
#             mode='json')  # Or request.dict() if Pydantic v1
#         logger.info(
#             "Received /load_negotiation request",
#             payload=request_payload_dict,
#             carrier_id=request.carrier_id,
#             load_id=request.load_id,
#             broker_id=request.broker_id,
#             serviceProviderLoadId=request.serviceProviderLoadId
#         )
#     except AttributeError as ae:  # More specific exception for this case
#         logger.error(
#             "AttributeError logging request payload - check field names",
#             exc_info=ae,
#             # Optionally log what attributes ARE available if you want to debug further:
#             # available_request_fields=list(request.__fields__.keys()) # Pydantic v1
#             # available_request_fields=list(request.model_fields.keys()) # Pydantic v2
#         )
#     except Exception as log_exc:
#         logger.error("Error logging request payload", exc_info=log_exc)
#     # --- END: Log incoming request payload ---

#     try:
#         # Get LLM instance
#         llm = get_llm(temperature=0.5)

#         # Use the load negotiation prompt
#         from prompts.load_negotation_prompt import create_load_negotiation_prompt
#         prompt_template = create_load_negotiation_prompt()

#         # Extract key identifiers
#         # Ensure these also use the correct attribute names from the request model
#         carrier_id = request.carrier_id
#         load_id = request.load_id
#         serviceProviderLoadId = request.serviceProviderLoadId
#         # This is where you correctly used broker_id later, so it should be consistent:
#         # Using this for clarity, assuming your history manager and other logic might expect 'brokerid'
#         brokerid_for_history_and_logic = request.broker_id

#         logger.info(
#             "Processing /load_negotiation request internally",
#             carrier_id=carrier_id,
#             load_id=load_id,
#             broker_id_from_request=brokerid_for_history_and_logic,  # Log the value being used
#             serviceProviderLoadId=serviceProviderLoadId
#         )

#         # Extract detailed load information
#         origin = getattr(request.load, 'pickupLocation', 'Unknown Origin')
#         destination = getattr(
#             request.load, 'dropoffLocation', 'Unknown Destination')
#         distance = getattr(request.load, 'distance', 0)

#         # Format pickup date
#         pickup_date = "TBD"
#         if request.load.pickupTimeType == "EXACT" and request.load.pickupTimeExact:
#             pickup_date = request.load.pickupTimeExact.strftime(
#                 '%B %d, %Y at %H:%M UTC')
#         elif request.load.pickupTimeType == "RANGE" and request.load.pickupTimeWindowStart:
#             pickup_date = request.load.pickupTimeWindowStart.strftime(
#                 '%B %d, %Y at %H:%M UTC')

#         # Extract load details
#         equipment_type = getattr(
#             request.load, 'equipmentType', 'Unknown Equipment')
#         commodity = getattr(request.load, 'commodity', 'Unknown Commodity')
#         weight = getattr(request.load, 'weight', 'Unknown')
#         weight_unit = getattr(request.load, 'weightUnit', 'lbs')
#         broker_company = getattr(request.load, 'brokerCompany', None)

#         # Create detailed load context
#         load_details_str = f"Load {request.serviceProviderLoadId}: {commodity} ({weight} {weight_unit}) from {origin} to {destination}"

#         # Build input message based on what's provided
#         current_broker_message_text: str = request.message if request.message is not None else ""
#         # The 'input' for the LLM will be this current_broker_message_text.
#         # If it's empty, the new prompt instructions will guide the AI to make a bid.

#         # ---- START HISTORY INTEGRATION ----
#         # Define the key fields for identifying the conversation
#         # These should match the parameters in your LoadNegotiationRequest or be derivable from it
#         carrier_id = request.carrier_id
#         load_id = request.load_id
#         serviceProviderLoadId = request.serviceProviderLoadId
#         brokerid = request.broker_id

#         logger.info(
#             f"Processing request for carrier_id: {carrier_id}, load_id: {load_id}, sp_load_id: {serviceProviderLoadId}, broker_id: {brokerid}")

#         # 1. Retrieve existing formatted chat history
#         retrieved_chat_history = get_formatted_chat_history(
#             carrier_id=carrier_id,
#             load_id=load_id,
#             serviceProviderLoadId=serviceProviderLoadId,
#             brokerid=brokerid
#         )
#         logger.debug(f"Retrieved chat history: {retrieved_chat_history}")

#         # 2. Store the new incoming broker message
#         add_negotiation_message(
#             carrier_id=carrier_id,
#             load_id=load_id,
#             serviceProviderLoadId=serviceProviderLoadId,
#             brokerid=brokerid,
#             sender_role="broker",
#             message_text=current_broker_message_text
#         )
#         # ---- END HISTORY INTEGRATION ----

#         # Store the actual incoming message (which could be empty)
#         add_negotiation_message(
#             carrier_id=carrier_id,
#             load_id=load_id,
#             serviceProviderLoadId=serviceProviderLoadId,
#             brokerid=brokerid,
#             sender_role="broker",
#             message_text=current_broker_message_text
#         )

#         # Prepare the brokers_reference_rate for the prompt
#         brokers_ref_rate_for_prompt = request.load.rate if hasattr(
#             request.load, 'rate') and request.load.rate is not None else "N/A"
#         # Ensure it's a string if numeric
#         if isinstance(brokers_ref_rate_for_prompt, (int, float)):
#             brokers_ref_rate_for_prompt = str(brokers_ref_rate_for_prompt)

#         # Format the prompt with all context
#         formatted_prompt = prompt_template.format_messages(
#             input=current_broker_message_text,  # This will be "" if message was empty
#             chat_history=retrieved_chat_history,
#             min_price=request.min_price,
#             max_price=request.max_price,
#             load_details=load_details_str,
#             pickup_location=origin,
#             delivery_location=destination,
#             distance=distance,
#             equipment_type=equipment_type,
#             commodity=commodity,
#             weight=weight,
#             weight_unit=weight_unit,
#             pickup_date=pickup_date,
#             broker_company=broker_company or "the broker",
#             brokers_reference_rate=brokers_ref_rate_for_prompt  # PASS THE NEW PLACEHOLDER
#         )
#         logger.debug(f"Formatted prompt messages for LLM: {formatted_prompt}")

#         # Invoke the LLM
#         response = llm.invoke(formatted_prompt)
#         logger.debug(f"LLM raw response content: {response.content}")

#         # Parse the JSON response
#         json_string = response.content.strip().replace(
#             "```json\n", "").replace("\n```", "")
#         negotiation_result = json.loads(json_string)

#         # Validate response structure
#         if not all(key in negotiation_result for key in ["response", "proposed_price", "status"]):
#             logger.error(
#                 f"LLM output is missing required keys. Received: {negotiation_result}")
#             raise ValueError("LLM output is missing required keys")

#         ai_response_text = negotiation_result.get("response")
#         ai_proposed_price_str = negotiation_result.get("proposed_price")
#         ai_status = negotiation_result.get("status")

#         ai_proposed_price_numeric: Optional[float] = None
#         if ai_proposed_price_str is not None:
#             try:
#                 ai_proposed_price_numeric = float(ai_proposed_price_str)
#             except ValueError:
#                 logger.warning(
#                     f"Could not convert proposed_price '{ai_proposed_price_str}' to float. Storing as null.")
#                 # Keep ai_proposed_price_numeric as None

#         # ---- STORE AI RESPONSE ----
#         if ai_response_text:
#             add_negotiation_message(
#                 carrier_id=carrier_id,
#                 load_id=load_id,
#                 serviceProviderLoadId=serviceProviderLoadId,
#                 brokerid=brokerid,
#                 sender_role="ai_negotiator",
#                 message_text=ai_response_text,
#                 proposed_price_by_ai=ai_proposed_price_numeric,
#                 status_by_ai=ai_status
#             )
#         # ---- END STORE AI RESPONSE ----

#         # Return the structured response
#         # Ensure proposed_price is a string as per LoadNegotiationResponse model if it's not null
#         return LoadNegotiationResponse(
#             response=ai_response_text,
#             proposed_price=str(
#                 ai_proposed_price_numeric) if ai_proposed_price_numeric is not None else None,
#             status=ai_status
#         )

#     except json.JSONDecodeError as e:
#         logger.error(
#             f"JSON Decode Error from LLM response: {e}. Raw string: '{json_string}'")
#         raise HTTPException(
#             status_code=500, detail="Error decoding LLM response")
#     except ValueError as e:
#         logger.error(f"Value Error processing negotiation: {e}")
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         logger.exception(
#             "An unexpected error occurred during load negotiation.")
#         raise HTTPException(status_code=500, detail="Internal server error")


# # @router.post("/fetch-data/")
# # async def fetch_data(payload: QueryPayload):
# #     """
# #     FastAPI endpoint to fetch query-specific data based on a UUID and query string.

# #     Args:
# #         payload: JSON payload containing 'query' and 'id' fields.

# #     Returns:
# #         A JSON response with the paragraph-format data or an error message.
# #     """
# #     query = payload.query.strip()
# #     uuid_input = payload.id.strip()

# #     uuid_pattern = re.compile(
# #         r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
# #     if not uuid_pattern.match(uuid_input):
# #         logger.warning(f"Invalid UUID format: {uuid_input}")
# #         raise HTTPException(status_code=400, detail="Invalid UUID format.")

# #     schema_name = get_schema_name(uuid_input)
# #     if not schema_name:
# #         logger.warning(
# #             f"No schema found for UUID {uuid_input} in public.tenant")
# #         raise HTTPException(
# #             status_code=404, detail=f"No schema found for UUID {uuid_input} in public.tenant")

# #     # Fetch query-specific data
# #     result = fetch_query_data(schema_name, query)

# #     paragraph_lines = [line for line in result.split("\n") if line.startswith(
# #         "- Query Data in Paragraph Format:") or line.startswith("  For one")]
# #     if paragraph_lines:
# #         paragraph = " ".join(
# #             line.strip() for line in paragraph_lines if line.startswith("  For one"))
# #         return {"result": paragraph}

# #     return {"result": result.strip()}


# # @router.post("/compose-email", response_model=EmailComposerResponse)
# # async def compose_email(request: EmailComposerRequest):
# #     try:
# #         input_data = request.model_dump(exclude_none=True)
# #         result = composer.compose_negotiation_email(input_data)
# #         return {
# #             "subject": result.get("subject", ""),
# #             "body": result.get("body", "")
# #         }
# #     except Exception as err:
# #         logger.error(f"Error in compose_email endpoint: {str(err)}")
# #         raise HTTPException(
# #             status_code=500,
# #             detail="Failed to compose negotiation email"
# #         )


# # @router.post('/get-info')
# # async def get_carrier_info(request: Any):
# #     try:
# #         response_text = "Carrier MC number is MC11333"
# #         return response_text
# #     except Exception as e:
# #         logger.error(f"Failed getting carrier information: {str(e)}")
# #         raise HTTPException(
# #             status_code=500, detail=f"Failed getting carrier information: {str(e)}")


# # @router.post('/classify-load-offer-bid-email', response_model=LoadBidEmailClassificationResponseStructured)
# # async def get_load_offer_bid_email_intent(payload: LoadBidEmailClassificationRequest):
# #     """
# #     Classify the intent of a load offer or bid email and extract structured details.
# #     Now supports attachments for better rate_confirmation detection.
# #     """
# #     try:
# #         # Initialize the agent
# #         email_classifier_agent = EmailClassifierAgent()

# #         # Pass both email content and attachments to the classifier
# #         response = email_classifier_agent.classify_load_bid_email_intent(
# #             payload.email, payload.attachments
# #         )
# #         return response
# #     except Exception as e:
# #         logger.error(f"Failed to classify email: {str(e)}")
# #         raise HTTPException(
# #             status_code=500, detail=f"Failed to classify email: {str(e)}")


# # @router.post('/broker-setup-extract', response_model=BrokerSetupExtractResponse)
# # async def broker_setup_extract(request: BrokerSetupExtractRequest):
# #     """
# #     Extract broker setup information from raw email content.
# #     Focuses on extracting brokerEmail, brokerCompany, and setupLink.
# #     """
# #     try:
# #         # Initialize the broker setup extractor
# #         extractor = BrokerSetupExtractor()

# #         # Determine the email content to process
# #         email_content = ""

# #         if request.email:
# #             # Simple string format
# #             email_content = request.email
# #         else:
# #             # Full email object format - combine relevant fields
# #             parts = []
# #             if request.subject:
# #                 parts.append(f"Subject: {request.subject}")
# #             if request.sender:
# #                 parts.append(f"From: {request.sender}")
# #             if request.to:
# #                 parts.append(f"To: {request.to}")
# #             if request.body:
# #                 parts.append(f"Body: {request.body}")

# #             email_content = "\n".join(parts)

# #         if not email_content.strip():
# #             raise ValueError("No email content provided")

# #         # Extract broker setup information
# #         extracted_info = extractor.extract_broker_setup_info(email_content)

# #         # Create the response with the extracted fields
# #         broker_setup_details = BrokerSetupExtractDetails(
# #             brokerEmail=extracted_info.get("brokerEmail"),
# #             brokerCompany=extracted_info.get("brokerCompany"),
# #             setupLink=extracted_info.get("setupLink")
# #         )

# #         return BrokerSetupExtractResponse(
# #             extracted_details=broker_setup_details
# #         )

# #     except Exception as e:
# #         logger.error(f"Error in broker_setup_extract endpoint: {str(e)}")
# #         raise HTTPException(status_code=500, detail=str(e))
