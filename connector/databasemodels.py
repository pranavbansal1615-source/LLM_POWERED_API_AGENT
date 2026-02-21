from sqlalchemy import Column,String, TIMESTAMP, Integer, Text,DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Users(Base):

    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique = True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Documents(Base):

    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False)
    file_name = Column(String, nullable = False)
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Conversations(Base):

    __tablename__ = "conversations"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False)
    document_id = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Messages(Base):

    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    conversation_id = Column(String(36), nullable=False)
    role = Column(String(20), nullable=False)  
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
