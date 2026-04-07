import React, { useState } from "react";

function Login({ onLogin }) {
  const [email, setEmail] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleLogin(e) {
    e.preventDefault();
    if (!email.trim()) return;

    setIsLoading(true);
    setError("");

    try {
      const res = await fetch("http://127.0.0.1:8000/api", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email })
      });

      const data = await res.json();
      
      localStorage.setItem("user_id", data.user_id);
      localStorage.setItem("email", data.email);
      
      onLogin(data.user_id);
    } catch (err) {
      setError("Could not connect to the server. Is the backend running?");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="login-wrapper">
      <div className="login-bg-orb orb-1"></div>
      <div className="login-bg-orb orb-2"></div>
      <div className="login-bg-orb orb-3"></div>

      <form className="login-card" onSubmit={handleLogin}>
        <div className="login-brand">
          <span className="login-brand-icon">⚡</span>
          <h1>LLM API Agent</h1>
        </div>
        <h2>Welcome Back</h2>
        <p>Sign in with your email to continue</p>
        
        <input
          type="email"
          placeholder="you@example.com"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
          disabled={isLoading}
        />

        {error && <div className="login-error">{error}</div>}

        <button type="submit" disabled={isLoading || !email.trim()}>
          {isLoading ? (
            <>
              <span className="btn-spinner"></span>
              Signing in…
            </>
          ) : (
            "Continue →"
          )}
        </button>
        
        <p className="login-footer">
          Powered by RAG + LLM • Upload docs & ask questions
        </p>
      </form>
    </div>
  );
}

export default Login;
