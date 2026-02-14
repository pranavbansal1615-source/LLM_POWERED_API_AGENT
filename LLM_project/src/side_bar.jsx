import React from "react";

function SideBar({
  pdfs,
  chats,
  selectedPdfId,
  onPdfUpload,
  onPdfSelect,
  onChatSelect,
  onNewChat
}) {
  return (
    <div className="uploading-documents">
      <label className="upload-btn">
        Upload PDF
        <input
          type="file"
          accept=".pdf"
          hidden
          onChange={e => onPdfUpload(e.target.files[0])}
        />
      </label>

      <hr />

      {!selectedPdfId ? (
        <>
          <h4>Your PDFs</h4>
          {pdfs.map(pdf => (
            <div
              key={pdf.id}
              className="sidebar-item"
              onClick={() => onPdfSelect(pdf.id)}
            >
              📄 {pdf.name}
            </div>
          ))}
        </>
      ) : (
        <>
          <button onClick={() => onPdfSelect(null)}>← Back</button>
          <h4>Chats</h4>

          {chats.map(chat => (
            <div
              key={chat.id}
              className="sidebar-item"
              onClick={() => onChatSelect(chat.id, chat.title)}
            >
              💬 {chat.title}
            </div>
          ))}

          <button onClick={onNewChat}>+ New Chat</button>
        </>
      )}
    </div>
  );
}

export default SideBar;
