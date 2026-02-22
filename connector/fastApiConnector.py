from fastapi import FastAPI,Depends
from databasemodels import Users,Messages,Documents,Conversations
from database import session
from sqlalchemy.orm import Session
from uuid import uuid4
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from temp import answer_query,generate_pdf_emb
from fastapi import UploadFile, File,Form
import os,shutil,uuid

app = FastAPI()

origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # or ["*"] for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():

    db = session()
    try:
        yield db
    finally:
        db.close()

class get_mail(BaseModel):

    email : str

@app.post("/api")

def user_login(data:get_mail, db: Session = Depends(get_db)):

    user = db.query(Users).filter(Users.email == data.email).first()

    if not user:
        user = Users(
            id = str(uuid4()),
            email = data.email
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    return {"user_id" : user.id, "email": user.email}

#uploading pdf on the website for processing

class DocumentsCreate(BaseModel):

    user_id: str
    file_name:str
    file_path:str

@app.post("/api/documents")

def pdf_upload(
    # data:DocumentsCreate,
    user_id:str = Form(...),
    file: UploadFile = File(...),
    db:Session = Depends(get_db)
):
    
    os.makedirs("uploads",exist_ok=True)

    document_id = str(uuid4())
    
    file_path = os.path.join("uploads", f"{document_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    doc = Documents(
        id =  document_id,
        user_id = user_id,
        file_name = file.filename,
        file_path = file_path
    )

    db.add(doc)
    db.commit()
    db.refresh(doc)

    generate_pdf_emb(file_path,user_id,doc.id)

    return {"document_id" : doc.id}

# ✅ FIXED: Added Pydantic model for conversation
class ConversationCreate(BaseModel):
    user_id: str
    document_id: str

#for creating a conversation that is when a user uploads a pdf file a chat is initiated 
@app.post("/api/conversations")
def create_conversation(
    data: ConversationCreate,  # ✅ CHANGED: Now accepts JSON body
    db: Session = Depends(get_db)
):
    convo = Conversations(
        id=str(uuid4()),
        user_id=data.user_id,  # ✅ CHANGED: Use data.user_id
        document_id=data.document_id  # ✅ CHANGED: Use data.document_id
    )
    db.add(convo)
    db.commit()
    db.refresh(convo)

    return {"conversation_id": convo.id}

# ✅ FIXED: Added Pydantic model for messages
class MessageCreate(BaseModel):
    conversation_id: str
    role: str
    content: str

#for saving a particular question and answer between teh user and bot 
@app.post("/api/messages")

def save_message(
    data: MessageCreate, 
    db: Session = Depends(get_db)
):
    msg = Messages(
        conversation_id=data.conversation_id,  
        role=data.role,  
        content=data.content  
    )
    db.add(msg)
    db.commit()

    return {"status": "saved"}

@app.get("/api/messages/{conversation_id}")

def get_messages(conversation_id: str, db : Session = Depends(get_db)):  

    messages = db.query(Messages).filter(Messages.conversation_id == conversation_id).order_by(Messages.created_at).all() 
    #this asks for messages from database and sort them in increasing order and then returns all the messages 

    return [
        {
            "role": message.role,
            "content": message.content
        }
        for message in messages
    ]

# when a user uploads a pdf

@app.get("/api/documents/{user_id}")
def get_documents(user_id: str, db: Session = Depends(get_db)):
    docs = db.query(Documents).filter(Documents.user_id == user_id).all()
    return [
        {"id": d.id, "file_name": d.file_name}
        for d in docs
    ]

@app.get("/api/conversations/{document_id}")
def get_conversations_by_document(document_id: str, db: Session = Depends(get_db)):  # ✅ CHANGED: Renamed to avoid duplicate
    convos = (
        db.query(Conversations)
        .filter(Conversations.document_id == document_id)
        .order_by(Conversations.created_at.desc())
        .all()
    )
    return [{"id": c.id} for c in convos]

#question and answer from the user 

class askRequest(BaseModel):
    conversation_id:str
    document_id:str
    question:str

@app.post("/api/ask")

def ask_question(data:askRequest, db: Session = Depends(get_db)):

    messages = db.query(Messages).filter(data.conversation_id == Messages.conversation_id).order_by(Messages.created_at).all()

    history = [
        {"role" : m.role, "content" : m.content}
        for m in messages
    ]

    answer = answer_query(data.question, data.document_id)

    user_ques = Messages(
        id = str(uuid4()),
        conversation_id = data.conversation_id,
        role = "user",
        content = data.question
    )

    db.add(user_ques)
    # db.commit()

    assistant_ans = Messages(
        id = str(uuid4()),
        conversation_id = data.conversation_id,
        role = "assistant",
        content = answer
    )

    db.add(assistant_ans)
    db.commit()

    return {"answer" : answer}

@app.get("/api/messages/{conversation_id}")

def get_chats_by_id(conversation_id:str, db: Session = Depends(get_db)):

    messages = db.query(Messages).filter(Messages.conversation_id == conversation_id).order_by(Messages.created_at).all()

    return [
        {"role":m.role, "content":m.content}
        for m in messages
    ]

@app.get("/api/user-data/{user_id}")
def get_user_data(user_id: str, db: Session = Depends(get_db)):

    documents = db.query(Documents).filter(
        Documents.user_id == user_id
    ).all()

    result = []

    for doc in documents:
        conversations = db.query(Conversations).filter(
            Conversations.document_id == doc.id
        ).all()

        conv_list = []

        for conv in conversations:
            messages = db.query(Messages).filter(
                Messages.conversation_id == conv.id
            ).order_by(Messages.created_at.asc()).all()

            conv_list.append({
                "id": conv.id,
                "messages": [
                    {"role": m.role, "content": m.content}
                    for m in messages
                ]
            })

        result.append({
            "id": doc.id,
            "file_name": doc.file_name,
            "conversations": conv_list
        })

    return result


