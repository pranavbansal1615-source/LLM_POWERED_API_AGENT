import React, { useState } from "react";

function Login({ onLogin }) {
  const [email, setEmail] = useState("");

  async function handleLogin(e) {
    e.preventDefault();

    const res = await fetch("http://127.0.0.1:8000/api", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email })
    });

    const data = await res.json();
    
    localStorage.setItem("user_id", data.user_id);
    localStorage.setItem("email", data.email);
    
    onLogin(data.user_id);
  }

  return (
    <div style={{ 
      display: "flex", 
      justifyContent: "center", 
      alignItems: "center", 
      height: "100vh" 
    }}>
      <form onSubmit={handleLogin}>
        <h2>Login</h2>
        <input
          type="email"
          placeholder="Enter your email"
          value={email}
          onChange={e => setEmail(e.target.value)}
        />
        <button type="submit">Login</button>
      </form>
    </div>
  );
}

export default Login;