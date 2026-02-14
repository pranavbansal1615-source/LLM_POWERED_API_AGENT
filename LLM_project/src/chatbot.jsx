import { useEffect } from "react";
import React, { useState } from "react";
import SideBar from "./side_bar";

function ChatBot() {
  const [pdfs, setPdfs] = useState([]);
  const [selectedPdfId, setSelectedPdfId] = useState(null);

  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [activeChatTitle, setActiveChatTitle] = useState("");

  const [chatMessages, setChatMessages] = useState({});
  const [message, setMessage] = useState("");

  // const [response,setResponse] = useState("");

  // setting the user_id once and it doesnt change when we refresh
  useEffect(() => {

    const userID = localStorage.getItem("user_id");
    if(!userID) return;

    async function load_pdfs() {
      
      const res = await fetch(`http://127.0.0.1:8000/api/documents/${userID}`);

      const data = await res.json();

      //mapping each document with the previous uploaded pdfs
      const formatted_data = data.map(doc => ({
        id:doc.id,
        name:doc.file_name
      }));
      
      setPdfs(formatted_data);

      const savedPdfId = localStorage.getItem("selected_pdf_id");

      if (savedPdfId) {
      handlePdfSelect(savedPdfId);
    }
    }
    
    load_pdfs();
    
  },[])

  async function handlePdfSelect(pdfId) {
    setSelectedPdfId(pdfId);
    setActiveChatId(null);
    setActiveChatTitle("");

    const res = await fetch(`http://127.0.0.1:8000/api/conversations/${pdfId}`);
    const data = await res.json();

    const formatted = data.map((c, i) => ({
      id: c.id,
      title: `Chat ${i + 1}`
    }));

    localStorage.setItem("selected_pdf_id", pdfId);

    setChats(formatted);
  }


  // ---------- PDF ----------
  async function handlePdfUpload(file) {
    const res = await fetch("http://127.0.0.1:8000/api/documents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: localStorage.getItem("user_id"),
        file_name: file.name,
        file_path: `/uploads/${file.name}`
      })
    });

    // console.log(localStorage.getItem("user_id"));
    const data = await res.json();

    const newPdf = {
      id: data.document_id,
      name: file.name
    };

    setPdfs(prev => [...prev, newPdf]);
    setSelectedPdfId(newPdf.id);
    setChats([]);
    setActiveChatId(null);

  }

  // ---------- CHAT ----------
  async function handleNewChat() {
  if (!selectedPdfId) return;

  const res = await fetch("http://127.0.0.1:8000/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: localStorage.getItem("user_id"),
        document_id: selectedPdfId
      })
    });

    const data = await res.json();

    const newChat = {
      id: data.conversation_id,
      title: `Chat ${chats.length + 1}`
    };

    setChats(prev => [...prev, newChat]);
    setActiveChatId(newChat.id);
    setActiveChatTitle(newChat.title);
    setChatMessages(prev => ({ ...prev, [newChat.id]: [] }));
  }


  async function handleChatSelect(chatId, title) {
    setActiveChatId(chatId);
    setActiveChatTitle(title);

    const res = await fetch(`http://127.0.0.1:8000/api/messages/${chatId}`);

    const old_chats = res.json;

    setChatMessages(prev => ({
      ...prev,
      [chatId]: Array.isArray(prev[chatId]) ? prev[chatId] : []
    }));

    setMessage("");
  }

  // ---------- MESSAGES ----------
  async function handleMessageAddition() {
  if (!message.trim() || !activeChatId) return;

  const question = message;
  setMessage("");

  // Always ensure array
  setChatMessages(prev => {
    const currentMessages = Array.isArray(prev[activeChatId])
      ? prev[activeChatId]
      : [];

    return {
      ...prev,
      [activeChatId]: [
        ...currentMessages,
        { role: "user", content: question }
      ]
    };
  });

  const res = await fetch("http://127.0.0.1:8000/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversation_id: activeChatId,
      question: question
    })
  });

  const data = await res.json();

  setChatMessages(prev => {
    const currentMessages = Array.isArray(prev[activeChatId])
      ? prev[activeChatId]
      : [];

    return {
      ...prev,
      [activeChatId]: [
        ...currentMessages,
        { role: "assistant", content: data.answer }
      ]
    };
  });
}


  const inbox = Array.isArray(chatMessages[activeChatId])
  ? chatMessages[activeChatId]
  : [];


  function handleLogout() {
    localStorage.removeItem("user_id");
    window.location.reload();
  }

  
  return (
    <div className="layout">
      <SideBar
        pdfs={pdfs}
        chats={chats}
        selectedPdfId={selectedPdfId}
        onPdfUpload={handlePdfUpload}
        onPdfSelect={handlePdfSelect}
        onChatSelect={handleChatSelect}
        onNewChat={handleNewChat}
      />

      <div className="chat-bot">
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <h2>
            {activeChatId ? activeChatTitle : "Select a chat"} &ensp;
            <button onClick={handleLogout}>Logout</button>
          </h2>
          {selectedPdfId && (
            <button onClick={handleNewChat}>+ New Chat</button>
          )}
        </div>

        <div className="chatting-box">
          <ul>
            {inbox.map((msg, i) => (
              <li key={i}>
                <strong>{msg.role}:</strong>{msg.content}
              </li>
            ))}
          </ul>
        </div>

        {activeChatId && (
          <>
            <input
              className="question-bar"
              value={message}
              onChange={e => setMessage(e.target.value)}
              placeholder="Enter question..."
            />
            <button onClick={handleMessageAddition}>Send</button>
          </>
        )}
      </div>

      <div className="sandbox-terminal"></div>
    </div>
  );
}

export default ChatBot;
