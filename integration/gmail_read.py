from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

creds = Credentials.from_authorized_user_file("token.json", SCOPES)
service = build("gmail", "v1", credentials=creds)

# Get messages (example: first 5)
results = service.users().messages().list(userId="me", maxResults=5).execute()
messages = results.get("messages", [])

for msg in messages:
    msg_detail = service.users().messages().get(userId="me",
                                                id=msg["id"], format="metadata", metadataHeaders=["In-Reply-To", "Subject", "From"]).execute()
    headers = msg_detail.get("payload", {}).get("headers", [])

    # Extract header values
    in_reply_to = next((h["value"]
                       for h in headers if h["name"] == "In-Reply-To"), None)
    subject = next((h["value"]
                   for h in headers if h["name"] == "Subject"), None)
    sender = next((h["value"] for h in headers if h["name"] == "From"), None)

    thread_id = msg_detail["threadId"]
    thread = service.users().threads().get(userId="me", id=thread_id).execute()
    print(thread)

    print(f"From: {sender}")
    print(f"Subject: {subject}")
    print(f"In-Reply-To: {in_reply_to}\n")
