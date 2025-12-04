"""
Email Automation Service - Main Application
Version 3.0 - Refactored with Orchestrator Pattern

Key Improvements:
1. Uses NegotiationOrchestrator for unified state management
2. Proper round counting (info vs price exchanges)
3. Context-aware intent classification
4. Clean error handling and logging
"""

from typing import Union
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import aio_pika
import asyncio
import json
import httpx
from contextlib import asynccontextmanager

from core.rabbitmq import RabbitMQManager
from core.auth_client import AuthApiClient

from models.email import EmailMessage
from constant.enum import RMQEnum, EMAIL_CLASSIFICATION
from config import AppConfig

from services.email_classifier import EmailClassifierOpenAI

from integration.supertruck import SuperTruck
from integration.load_board import LoadBoard

from actions.negotiation import NegotiationAction

from api.v1.negotiation import NegotiationController
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()


config = AppConfig()
# ------------------------------
# Create RabbitMQ manager
# ------------------------------
rabbit_mq_manager = RabbitMQManager(
    rabbitmq_url=config.RABBITMQ_URL,
    queue_name=RMQEnum.EMAIL_INGESTION_QUEUE.value
)


# ------------------------------
# Lifespan (startup / shutdown)
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Attempting RabbitMQ connection...")
    # Startup
    await rabbit_mq_manager.connect()
    consumer_task = asyncio.create_task(
        rabbit_mq_manager.consume(message_processor)
    )
    app.state.consumer_task = consumer_task
    logger.info("Connected")
    yield

    # Shutdown
    rabbit_mq_manager.should_reconnect = False

    if hasattr(app.state, "consumer_task"):
        app.state.consumer_task.cancel()
        try:
            await app.state.consumer_task
        except asyncio.CancelledError:
            pass

    await rabbit_mq_manager.disconnect()


# ------------------------------
# FastAPI instance (SINGLE)
# ------------------------------
app = FastAPI(
    title="SuperTruck Email Automation",
    description="APIs for email processing and negotiation",
    version="0.1.0",
    root_path="/email-processor",
    lifespan=lifespan
)


# ------------------------------
# Middleware
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------
# Routers
# ------------------------------
negotiation_router = NegotiationController().router
app.include_router(negotiation_router)

received_messages = []


supertruck = SuperTruck()
negotiation = NegotiationAction()
load_board = LoadBoard()


def remove_none_values(data: dict) -> dict:
    return {k: v for k, v in data.items() if v is not None}


# =============================================================================
# MAIN MESSAGE PROCESSOR
# =============================================================================

async def message_processor(message: aio_pika.IncomingMessage):
    """
    Main processor that handles incoming email messages.

    Flow:
    1. If reply (inReplyTo exists) -> Continue existing conversation
    2. Otherwise -> Classify and route to appropriate handler

    The NegotiationAction now uses the NegotiationOrchestrator for:
    - Proper round counting (info vs price exchanges)
    - Context-aware intent classification
    - Unified state management
    """
    logger.info("="*60)
    logger.info("MESSAGE RECEIVED IN PROCESSOR")
    logger.info("="*60)

    async with message.process():
        try:
            # Parse email data
            email_data_decode = message.body.decode()
            email_data = json.loads(email_data_decode)

            # Store for dev/debug purposes
            received_messages.append(json.loads(email_data_decode))

            # Build params for handlers
            params = {
                "tenant_id": email_data["tenantId"],
                "to": email_data["to"],
                "from": email_data["from"],
                "thread_Id": email_data["threadId"],
                "body": email_data["body"],
                "messageId": email_data["messageId"],
                "inReplyTo": email_data.get("inReplyTo"),
                "references": email_data.get("references")
            }

            logger.info(f"From: {email_data.get('from')}")
            logger.info(f"Subject: {email_data.get('subject', 'N/A')}")
            logger.info(f"Thread: {email_data.get('threadId')}")
            logger.info(f"Is Reply: {bool(email_data.get('inReplyTo'))}")

            # FLOW 1: Continue existing conversation (reply)
            if email_data.get("inReplyTo"):
                logger.info("Continuing existing conversation (reply)")
                email_log = await supertruck.find_email_log(
                    tenant_id=email_data["tenantId"],
                    thread_id=email_data["threadId"]
                )
                # Use the new orchestrator-based negotiation
                await negotiation.execute_negotiation(data=params)

            # FLOW 2: New email - classify and route
            else:
                logger.info("New email - classifying")

                if not any(email_data.get(key) for key in ["subject", "body"]):
                    raise ValueError("No email content to classify")

                # Classify the email
                classifier = EmailClassifierOpenAI()
                output = classifier.process_email(email_data)

                classification = output["classification"]
                cleaned_data = remove_none_values(output["extracted_details"])

                logger.info(f"Classification: {classification}")
                logger.info(f"Extracted fields: {list(cleaned_data.keys())}")

                # Route to appropriate handler
                actions = {
                    EMAIL_CLASSIFICATION.LOAD_OFFER_NEGOTIATION.value: lambda: negotiation.execute_negotiation(data=params),
                    EMAIL_CLASSIFICATION.LOAD_OFFER.value: lambda: load_board.create_load_offer(
                        tenant_id=email_data["tenantId"], data=cleaned_data),
                    EMAIL_CLASSIFICATION.LOAD.value: lambda: load_board.create_load(
                        tenant_id=email_data["tenantId"], data=output["extracted_details"]),
                    EMAIL_CLASSIFICATION.BROKER_SETUP.value: lambda: logger.info("Broker setup email - requires manual handling"),
                }

                action = actions.get(
                    classification,
                    lambda: logger.info(f"Unknown classification: {classification}")
                )
                await action()

            logger.info("Message processed successfully")
            logger.info("="*60)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}")
            return e

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return e

        except KeyError as e:
            logger.error(f"Missing field: {e}")
            return e

        except Exception as e:
            logger.exception(f"Unexpected error: {repr(e)}")
            return e


# !NEED TO CHANGE FOR DEVELOPMENT PURPOSE ONLY
@app.get("/email")
async def receive_message():
    return {"messages": received_messages}


@app.post("/send")
async def send_message(
    message: EmailMessage = Body(
        ...,
        example=EmailMessage.Config.json_schema_extra.get("example")
        if hasattr(EmailMessage, 'Config') and hasattr(EmailMessage.Config, 'json_schema_extra')
        else None
    )
):
    """
    Send an email message to the RabbitMQ queue for processing.

    - **message**: EmailMessage object with all email details
    """
    try:
        # Convert Pydantic model to dict
        message_dict = message.dict(by_alias=True)

        # Publish to RabbitMQ
        await rabbit_mq_manager.publish(message_dict)

        logger.info(
            f"✅ Message published successfully: {message_dict.get('subject', 'No subject')}")

        return {
            "status": "success",
            "message": "Email message sent to queue",
            "queue": rabbit_mq_manager.queue_name,
            "data": message_dict
        }

    except Exception as e:
        logger.error(f"❌ Failed to send message: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send message to RabbitMQ: {str(e)}"
        )


@app.get("/")
def read_root():
    return {"Hello": "Developer"}
