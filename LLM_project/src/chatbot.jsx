import { useEffect } from "react";
import React, { useState } from "react";
import SideBar from "./side_bar";
import Sandbox from "./sandbox";

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
    if (!userID) return;
    
    async function loadAllData() {
      const res = await fetch(
        `http://127.0.0.1:8000/api/user-data/${userID}`
      );
    
      const data = await res.json();
    
      // Store PDFs
      setPdfs(data.map(doc => ({
        id: doc.id,
        name: doc.file_name
      })));
    
      // Build chats map
      let allChats = [];
      let allMessages = {};
    
      data.forEach(doc => {
        if (!Array.isArray(doc.conversations)) return;
            
        doc.conversations.forEach((conv, index) => {
          allChats.push({
            id: conv.id,
            title: `Chat ${index + 1}`,
            document_id: doc.id
          });
        
          allMessages[conv.id] = Array.isArray(conv.messages)
            ? conv.messages
            : [];
        });
      });
    
      setChats(allChats);
      setChatMessages(allMessages);
    }
  
    loadAllData();
  }, []);


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

    // Add user message immediately
    setChatMessages(prev => {
      const current = Array.isArray(prev[activeChatId])
        ? prev[activeChatId]
        : [];

      return {
        ...prev,
        [activeChatId]: [
          ...current,
          { role: "user", content: question }
        ]
      };
    });

    try {
      const res = await fetch("http://127.0.0.1:8000/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: activeChatId,
          question: question
        })
      });

      const data = await res.json();

      // console.log("Backend answer:", data.answer);  // DEBUG

      // Add assistant message ONLY after response arrives
      setChatMessages(prev => {
        const current = Array.isArray(prev[activeChatId])
          ? prev[activeChatId]
          : [];

        return {
          ...prev,
          [activeChatId]: [
            ...current,
            { role: "assistant", content: data.answer }
          ]
        };
      });

    } catch (err) {
      console.error("Error fetching answer:", err);
    }

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

      <div className="sandbox-terminal"><Sandbox/></div>
    </div>
  );
}

export default ChatBot;
