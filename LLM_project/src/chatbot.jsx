import { useEffect, useRef } from "react";
import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import SideBar from "./side_bar";
import Sandbox from "./sandbox";

// ── Markdown renderer for assistant messages ──────────────────────────
function MarkdownMessage({ content }) {
  const [copied, setCopied] = useState(null);

  function handleCopy(code, id) {
    navigator.clipboard.writeText(code);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  }

  return (
    <ReactMarkdown
      components={{
        // ── Code blocks ──
        code({ node, inline, className, children, ...props }) {
          const language = (className || "").replace("language-", "") || "text";
          const codeString = String(children).replace(/\n$/, "");
          const id = `${language}-${codeString.slice(0, 20)}`;

          if (inline) {
            return <code className="inline-code" {...props}>{children}</code>;
          }

          return (
            <div className="code-block">
              <div className="code-header">
                <span className="code-lang">{language}</span>
                <button onClick={() => handleCopy(codeString, id)}>
                  {copied === id ? "✓ Copied" : "Copy"}
                </button>
              </div>
              <SyntaxHighlighter
                style={vscDarkPlus}
                language={language}
                PreTag="div"
                customStyle={{
                  margin: 0,
                  padding: "14px 16px",
                  background: "#060b0e",
                  fontSize: "13.5px",
                  lineHeight: "1.75",
                  borderRadius: 0,
                }}
                {...props}
              >
                {codeString}
              </SyntaxHighlighter>
            </div>
          );
        },

        // ── Other markdown elements ──
        p: ({ children }) => <p className="md-p">{children}</p>,
        h1: ({ children }) => <h1 className="md-h1">{children}</h1>,
        h2: ({ children }) => <h2 className="md-h2">{children}</h2>,
        h3: ({ children }) => <h3 className="md-h3">{children}</h3>,
        ul: ({ children }) => <ul className="md-ul">{children}</ul>,
        ol: ({ children }) => <ol className="md-ol">{children}</ol>,
        li: ({ children }) => <li className="md-li">{children}</li>,
        strong: ({ children }) => <strong className="md-strong">{children}</strong>,
        em: ({ children }) => <em className="md-em">{children}</em>,
        blockquote: ({ children }) => <blockquote className="md-blockquote">{children}</blockquote>,
        hr: () => <hr className="md-hr" />,
        a: ({ href, children }) => (
          <a href={href} className="md-link" target="_blank" rel="noreferrer">
            {children}
          </a>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

// ── Main ChatBot ───────────────────────────────────────────────────────
function ChatBot() {
  const [pdfs, setPdfs] = useState([]);
  const [selectedPdfId, setSelectedPdfId] = useState(null);

  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [activeChatTitle, setActiveChatTitle] = useState("");

  const [chatMessages, setChatMessages] = useState({});
  const [message, setMessage] = useState("");

  const bottomRef = useRef(null);

  // auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, activeChatId]);

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
  const formData = new FormData();

  formData.append("file", file);
  formData.append("user_id", localStorage.getItem("user_id"));

  const res = await fetch("http://127.0.0.1:8000/api/documents", {
    method: "POST",
    body: formData,   // 🚨 DO NOT set headers
  });

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
          document_id: selectedPdfId,
          question: question
        })
      });

      const data = await res.json();

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
          {inbox.map((msg, i) => (
            <div key={i} className={`chat-message ${msg.role}`}>
              {msg.role === "assistant" ? (
                <MarkdownMessage content={msg.content} />
              ) : (
                msg.content
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {activeChatId && (
          <>
            <input
              className="question-bar"
              value={message}
              onChange={e => setMessage(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleMessageAddition()}
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
