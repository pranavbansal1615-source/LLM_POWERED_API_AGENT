import React, { useEffect, useState, useRef, useCallback } from "react";

const DEFAULT_CODE = `import requests

# Example: Fetch data from a public API
response = requests.get("https://jsonplaceholder.typicode.com/posts/1")
print(response.status_code)
print(response.json())`;

const PRE_INSTALLED_PACKAGES = [
  "micropip",
  "pyodide-http",
  "requests",
  "urllib3",
];

function Sandbox() {
  const [pyodide, setPyodide] = useState(null);
  const [code, setCode] = useState(DEFAULT_CODE);
  const [output, setOutput] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState("Initializing Python runtime…");
  const [packagesReady, setPackagesReady] = useState(false);
  const textareaRef = useRef(null);
  const lineNumberRef = useRef(null);

  // Sync scroll between textarea and line numbers
  const handleScroll = useCallback(() => {
    if (lineNumberRef.current && textareaRef.current) {
      lineNumberRef.current.scrollTop = textareaRef.current.scrollTop;
    }
  }, []);

  // Load Pyodide + install API packages
  useEffect(() => {
    async function init() {
      try {
        setLoadingStatus("Loading Python runtime…");
        const pyodideInstance = await window.loadPyodide();

        setLoadingStatus("Installing API packages…");
        await pyodideInstance.loadPackage("micropip");
        const micropip = pyodideInstance.pyimport("micropip");

        // Install packages one by one for better status tracking
        for (const pkg of ["pyodide-http", "requests", "urllib3"]) {
          setLoadingStatus(`Installing ${pkg}…`);
          try {
            await micropip.install(pkg);
          } catch (e) {
            console.warn(`Could not install ${pkg}:`, e.message);
          }
        }

        // Patch requests to work in browser via pyodide-http
        try {
          pyodideInstance.runPython(`
import pyodide_http
pyodide_http.patch_all()
          `);
        } catch (e) {
          console.warn("pyodide-http patch failed:", e.message);
        }

        setPyodide(pyodideInstance);
        setPackagesReady(true);
        setIsLoading(false);
        setOutput("✅ Python ready — requests, urllib3 pre-installed\n$ ");
      } catch (err) {
        setIsLoading(false);
        setOutput("❌ Failed to load Python: " + err.message);
      }
    }

    init();
  }, []);

  async function runCode() {
    if (!pyodide || isRunning) return;
    setIsRunning(true);
    setOutput("");

    try {
      // Redirect stdout & stderr
      pyodide.runPython(`
import sys
from io import StringIO
sys.stdout = StringIO()
sys.stderr = StringIO()
      `);

      await pyodide.runPythonAsync(code);

      const stdout = pyodide.runPython("sys.stdout.getvalue()");
      const stderr = pyodide.runPython("sys.stderr.getvalue()");

      let result = "";
      if (stdout) result += stdout;
      if (stderr) result += "\n⚠️ " + stderr;
      if (!result.trim()) result = "✓ Code executed (no output)";

      setOutput(result);
    } catch (err) {
      setOutput("❌ Error:\n" + err.message);
    } finally {
      setIsRunning(false);
    }
  }

  function clearOutput() {
    setOutput("$ ");
  }

  function handleKeyDown(e) {
    // Tab support in textarea
    if (e.key === "Tab") {
      e.preventDefault();
      const start = e.target.selectionStart;
      const end = e.target.selectionEnd;
      const newCode = code.substring(0, start) + "    " + code.substring(end);
      setCode(newCode);
      // Set cursor position after state update
      setTimeout(() => {
        e.target.selectionStart = e.target.selectionEnd = start + 4;
      }, 0);
    }
    // Ctrl+Enter to run
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      runCode();
    }
  }

  const lineCount = code.split("\n").length;

  return (
    <div className="sandbox-container">
      {/* Header */}
      <div className="sandbox-header">
        <div className="sandbox-title">
          <span className="sandbox-dot red"></span>
          <span className="sandbox-dot yellow"></span>
          <span className="sandbox-dot green"></span>
          <span className="sandbox-title-text">Python Sandbox</span>
        </div>
        <div className="sandbox-status">
          {isLoading ? (
            <span className="status-loading">
              <span className="status-spinner"></span>
              {loadingStatus}
            </span>
          ) : packagesReady ? (
            <span className="status-ready">● Ready</span>
          ) : (
            <span className="status-error">● Error</span>
          )}
        </div>
      </div>

      {/* Code Editor */}
      <div className="sandbox-editor">
        <div className="line-numbers" ref={lineNumberRef}>
          {Array.from({ length: lineCount }, (_, i) => (
            <span key={i + 1}>{i + 1}</span>
          ))}
        </div>
        <textarea
          ref={textareaRef}
          className="sandbox-code"
          value={code}
          onChange={(e) => setCode(e.target.value)}
          onScroll={handleScroll}
          onKeyDown={handleKeyDown}
          spellCheck={false}
          disabled={isLoading}
          placeholder="# Write your Python code here..."
        />
      </div>

      {/* Action Bar */}
      <div className="sandbox-actions">
        <button
          className="sandbox-run-btn"
          onClick={runCode}
          disabled={isLoading || isRunning}
        >
          {isRunning ? (
            <>
              <span className="btn-spinner"></span> Running…
            </>
          ) : (
            <>▶ Run Code</>
          )}
        </button>
        <button className="sandbox-clear-btn" onClick={clearOutput}>
          Clear
        </button>
        <span className="sandbox-hint">Ctrl + Enter to run</span>
      </div>

      {/* Output Terminal */}
      <div className="sandbox-output">
        <div className="output-header">
          <span>Output</span>
        </div>
        <pre className="output-content">
          {isRunning ? (
            <span className="output-running">
              <span className="status-spinner"></span> Executing…
            </span>
          ) : (
            output || "$ "
          )}
        </pre>
      </div>
    </div>
  );
}

export default Sandbox;