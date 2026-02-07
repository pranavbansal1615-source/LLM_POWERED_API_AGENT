from fastapi import FastAPI,Depends
from databasemodels import Users,Messages,Documents,Conversations
from database import session
from sqlalchemy.orm import Session
from uuid import uuid4
from datetime import datetime

app = FastAPI()

def get_db():

    db = session()
    try:
        yield db
    finally:
        db.close()

@app.post("/api")

def user_login(email: str, db: Session = Depends(get_db)):

    user = db.query(Users).filter(Users.email == email).first()

    if not user:
        user = Users(
            id = str(uuid4()),
            email = email
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    return {"user_id" : user.id}

#uploading pdf on the website for processing
@app.post("/api/documents")

def pdf_upload(
    file_name:str,
    file_path:str,
    user_id:str,
    db:Session = Depends(get_db)
):
    doc = Documents(
        id =  str(uuid4()),
        user_id = user_id,
        file_name = file_name,
        file_path = file_path
    )

    db.add(doc)
    db.commit()
    db.refresh(doc)

    return

#for creating a conversation that is when a user uploads a pdf file a chat is initiated 
@app.post("/api/conversations")
def create_conversation(
    user_id: str,
    document_id: str,
    db: Session = Depends(get_db)
):
    convo = Conversations(
        id=str(uuid4()),
        user_id=user_id,
        document_id=document_id
    )
    db.add(convo)
    db.commit()
    db.refresh(convo)

    return {"conversation_id": convo.id}

#for saving a particular question and answer between teh user and bot 
@app.post("/api/messages")
def save_message(
    conversation_id: str,
    role: str,
    content: str,
    db: Session = Depends(get_db)
):
    msg = Messages(
        conversation_id=conversation_id,
        role=role,
        content=content
    )
    db.add(msg)
    db.commit()

    return {"status": "saved"}

@app.get("/api/messages/{conversation_id}")

def get_conversations(conversation_id: str, db : Session = Depends(get_db)):

    messages = db.query(Messages).filter(Messages.conversation_id == conversation_id).order_by(Messages.created_at).all() 
    #this asks for messages from database and sort them in increasing order and then returns all the messages 

    return [
        {
            "role": message.role,
            "content": message.content
        }
        for message in messages
    ]

@app.get("/api/documents/{user_id}")
def get_documents(user_id: str, db: Session = Depends(get_db)):
    docs = db.query(Documents).filter(Documents.user_id == user_id).all()
    return [
        {"id": d.id, "file_name": d.file_name}
        for d in docs
    ]

@app.get("/api/conversations/{document_id}")
def get_conversations(document_id: str, db: Session = Depends(get_db)):
    convos = (
        db.query(Conversations)
        .filter(Conversations.document_id == document_id)
        .order_by(Conversations.created_at.desc())
        .all()
    )
    return [{"id": c.id} for c in convos]

