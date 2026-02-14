import { useState, useEffect } from "react";
import Login from "./Login";
import ChatBot from "./chatbot";

function App() {
  const [userId, setUserId] = useState(null);

  useEffect(() => {
    const savedId = localStorage.getItem("user_id");
    if (savedId) setUserId(savedId);
  }, []);

  return (
    <>
      {!userId ? (
        <Login onLogin={setUserId} />
      ) : (
        <ChatBot userId={userId} />
      )}
    </>
  );
}

export default App;
