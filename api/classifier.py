# import logging
# from fastapi import APIRouter, HTTPException
# from models.classifier import EmailClassificationRequest, EmailClassificationResponse
# from services.email_classifier import EmailClassifierOpenAI

# # ! NOT REQUIRED JUST FOR TESTING PURPOSE CAN BE CALL WHEN RABBITMQ MESSAGE IS CONSUME AND CALL CLASSIFIER SERVICE
# class EmailClassifierController:
#     def __init__(self):
#         # self.message = message
#         self.logger = logging.getLogger(__name__)
#         self.router = APIRouter(prefix='/api/v1')

#         # Register routes
#         self.router.add_api_route(
#             '/classify_email', self.classify_email, methods=["POST"])

#     async def classify_email(self, email_data: EmailClassificationRequest) -> EmailClassificationResponse:
#         """Classify email intent and extract key details."""
#         try:
#             print(f"This is from classifier: {email_data}")

#             classifier = EmailClassifierOpenAI()
#             email_details_dict = email_data.model_dump(
#                 by_alias=True, exclude_none=True)

#             if not any(email_data.get(key) for key in ["Subject"]):
#                 self.logger.warning(
#                     "Classifier endpoint: No relevant content for processing.")
#                 raise HTTPException(
#                     status_code=400,
#                     detail="No relevant email content (Subject, Short Preview, or Main Body (Trimmed)) provided."
#                 )

#             # Use the new method that returns both classification and details
#             processed_data = classifier.process_email(email_details_dict)

#             # Ensure the response matches the Pydantic model structure
#             return EmailClassificationResponse(
#                 classification=processed_data.get("classification", "error"),
#                 extracted_details=processed_data.get("extracted_details", {}),
#                 reason=processed_data.get("reason")
#             )
#             # return {
#             #     "classification": "load-offer",
#             #     "extracted_details": message,
#             #     "reason": "I am on testing mode, Reason will be available soon"
#             # }

#         except HTTPException as http_exc:
#             raise http_exc
#         except Exception as err:
#             self.logger.error(
#                 "Error in classify_email endpoint", exc_info=True)
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Failed to process email: {str(err)}"
#             )
