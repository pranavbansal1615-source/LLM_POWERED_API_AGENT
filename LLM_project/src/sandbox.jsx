import React, { useEffect, useState } from "react";

function Sandbox() {
  const [pyodide, setPyodide] = useState(null);
  const [code, setCode] = useState("print('Hello from Pyodide')");
  const [output, setOutput] = useState("Loading Python runtime...");

  // Load Pyodide once
  useEffect(() => {
    async function loadPyodideAndPackages() {
      const pyodideInstance = await window.loadPyodide();
      setPyodide(pyodideInstance);
      setOutput("Python Ready ✅");
    }

    loadPyodideAndPackages();
  }, []);

  async function runCode() {
    if (!pyodide) return;

    try {
      // Redirect stdout
      pyodide.runPython(`
import sys
from io import StringIO
sys.stdout = StringIO()
      `);

      pyodide.runPython(code);

      const result = pyodide.runPython("sys.stdout.getvalue()");
      setOutput(result || "Code executed successfully.");
    } catch (err) {
      setOutput("Error:\n" + err.message);
    }
  }

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <h3>Python Sandbox</h3>

      <textarea
        value={code}
        onChange={e => setCode(e.target.value)}
        style={{
          flex: 1,
          background: "#1e1e1e",
          color: "#00ff88",
          fontFamily: "monospace",
          padding: "10px",
          borderRadius: "8px",
          marginBottom: "10px"
        }}
      />

      <button onClick={runCode}>Run Code</button>

      <div
        style={{
          marginTop: "10px",
          background: "black",
          color: "#00ff88",
          padding: "10px",
          borderRadius: "8px",
          minHeight: "100px",
          whiteSpace: "pre-wrap"
        }}
      >
        {output}
      </div>
    </div>
  );
}

export default Sandbox;