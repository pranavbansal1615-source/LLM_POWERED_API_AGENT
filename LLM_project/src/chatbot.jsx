import React, { useState } from "react";

function ChatBot(){

    const [inbox, setInbox] = useState([]);
    const [message, setMessage] = useState("");
    const [botMsg, changeBotMsg] = useState("");
    const [botInbox, botInboxChange] = useState([]);
    
    function handleMessageChange(event){

        if(event.target.value == "") return;

        setMessage(event.target.value);
    }

    function handleMessageAddition(){

        setInbox(inbox => [...inbox, message]);
        setMessage("");
    }

    return(<>
    <div className="layout">
    <div className="uploading-documents"></div>
    <div className = "chat-bot">
        <h2>Simple Chat App</h2>

        <div className = "chatting-box">
            <ul>{inbox.map((message, index) => (
                <li key = {index}>{message}</li>
            ))}</ul>
        </div>

        <input className = "question-bar" type = "text" placeholder="Enter question..." value = {message} onChange={handleMessageChange}></input>
        <button onClick={handleMessageAddition}>Send</button>
    </div>
    <div className="sandbox-terminal"></div>
    </div>
    </>
    );
}

export default ChatBot;