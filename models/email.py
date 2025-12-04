from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Attachment(BaseModel):
    filename: str
    mimeType: str
    size: int
    attachmentId: str


class EmailMessage(BaseModel):
    id: str
    emailProfileId: str
    tenantId: str
    messageId: str
    threadId: str
    subject: Optional[str] = None
    from_: str = Field(..., alias="from")  # reserved keyword
    to: str
    date: str
    snippet: str
    body: Optional[str] = None
    attachments: Optional[List[Attachment]] = None
    inReplyTo: Optional[str] = None
    references: Optional[str] = None

    # New fields
    isCompiledByAI: bool
    isPublishedToRmq: bool
    maxRetries: int
    retryCount: int
    isEmailDead: bool
    lastSyncedAt: str

    class Config:
        validate_by_name = True
        json_schema_extra = {
            "example": {
                "emailProfileId": "44495ecc-8aa9-4863-bd41-0f5f3ef0e6f2",
                "tenantId": "43202a01-20d5-436e-96eb-a5b82a513226",
                "messageId": "19ac43b7ea0bc048",
                "threadId": "19abfeec157d9407",
                "from": "Basant Rai <basantrai2486@gmail.com>",
                "subject": "Re: Request for load offer",
                "to": "basanta.advikai@gmail.com",
                "date": "2025-11-27T07:33:32.000Z",
                "snippet": "What is your company name?",
                "body": "What is your company name?\r\n",
                "inReplyTo": "<CAEQdmrGfyhRnuZk0kERcnzY24r113ZnimQQ3Uzw9V0SvM1vy1w@mail.gmail.com>",
                "references": "<CAGFO_UtKAHJ_pFT4MxYitf+xTDJdHHWcrqxWGaUwdz+t-nuMqg@mail.gmail.com> <CAEQdmrF-4NgtPUJB3naO7WBt3+hfwdtgx92L814pUpyVQo1MYg@mail.gmail.com> <CAEQdmrE_i44D=QyxUJRZr2UNWKyR+T-gueUFKc3Q=er8N56DfQ@mail.gmail.com> <CAGFO_UtVM4SRjOaPbsFoGTsxY7Ep79zqoq=ACvmT-MKUWbdE_w@mail.gmail.com> <CAGFO_Us7KmaMVwzFT-N4MVtHT_SeT=iYuhshNrpHfQR-FEdO-g@mail.gmail.com> <CAGFO_Utdvt76uB_tNUR9zPNAuwYaB3bCYxxr1TY27a3rUFUaSQ@mail.gmail.com> <CAEQdmrH_1Xptw3B+xLTn7RKzyTXdwhroTA7ARzXJCQkEt=w0-Q@mail.gmail.com> <CAGFO_UtobzbOHwasBCmRPPouotRWTfBLVLkqp3F+nFZ8bUEcpQ@mail.gmail.com> <CAGFO_Uv_6uxSrSnMyvirLUQfYMF_xVAydMOQ_Lz1ZEzhR95AGw@mail.gmail.com> <CAGFO_UtDgMBAM+-Vx0hp3G3za=tVippVER0HiytN0K14xMph3g@mail.gmail.com> <CAGFO_UvGQTjg7serkJ+Sv+6-geyFx7eTQVkaO=qm29pPwyESwg@mail.gmail.com> <CAEQdmrGfyhRnuZk0kERcnzY24r113ZnimQQ3Uzw9V0SvM1vy1w@mail.gmail.com>",
                "lastSyncedAt": "2025-11-27T07:34:31.772Z",
                "aiResponseErrorCode": "None",
                "id": "2c013cbd-8605-4ad4-841e-e51cc39d4d83",
                "createdAt": "2025-11-27T01:49:31.764Z",
                "updatedAt": "2025-11-27T01:49:31.764Z",
                "deletedAt": "None",
                "isCompiledByAI": "false",
                "isPublishedToRmq": "false",
                "maxRetries": 5,
                "retryCount": 0,
                "isEmailDead": "false",
                "compilationStatus": "pending"
            }
        }
