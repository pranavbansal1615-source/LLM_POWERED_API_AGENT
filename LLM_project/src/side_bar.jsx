import React from "react";

function SideBar({
  pdfs,
  chats,
  selectedPdfId,
  onPdfUpload,
  onPdfSelect,
  onChatSelect,
  onNewChat,
  isUploading
}) {
  return (
    <div className="uploading-documents">
      {/* Brand */}
      <div className="sidebar-brand">
        <span className="brand-icon">⚡</span>
        <span className="brand-text">API Agent</span>
      </div>

      {/* Upload */}
      <label className={`upload-btn ${isUploading ? "uploading" : ""}`}>
        {isUploading ? (
          <>
            <span className="btn-spinner"></span>
            Processing…
          </>
        ) : (
          <>Upload PDF</>
        )}
        <input
          type="file"
          accept=".pdf"
          hidden
          disabled={isUploading}
          onChange={e => {
            if (e.target.files[0]) onPdfUpload(e.target.files[0]);
          }}
        />
      </label>

      <hr />

      {!selectedPdfId ? (
        <>
          <h4>Your Documents</h4>
          {pdfs.length === 0 ? (
            <div className="sidebar-empty">
              <span>📂</span>
              <p>No documents yet</p>
            </div>
          ) : (
            pdfs.map(pdf => (
              <div
                key={pdf.id}
                className="sidebar-item"
                onClick={() => onPdfSelect(pdf.id)}
              >
                <span className="sidebar-item-icon">📄</span>
                <span className="sidebar-item-text">{pdf.name}</span>
              </div>
            ))
          )}
        </>
      ) : (
        <>
          <button className="back-btn" onClick={() => onPdfSelect(null)}>
            ← Back to Documents
          </button>
          <h4>Conversations</h4>

          {chats.length === 0 ? (
            <div className="sidebar-empty">
              <span>💬</span>
              <p>No chats yet</p>
            </div>
          ) : (
            chats.map(chat => (
              <div
                key={chat.id}
                className="sidebar-item"
                onClick={() => onChatSelect(chat.id, chat.title)}
              >
                <span className="sidebar-item-icon">💬</span>
                <span className="sidebar-item-text">{chat.title}</span>
              </div>
            ))
          )}

          <button className="new-chat-sidebar-btn" onClick={onNewChat}>+ New Chat</button>
        </>
      )}
    </div>
  );
}

export default SideBar;
