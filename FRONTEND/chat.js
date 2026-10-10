const API_BASE = window.location.origin;
const AUTH_KEYS = { email: "company_email", role: "company_role", token: "company_token" };
const roleLabels = { employee: "Employee", finance: "Finance", hr: "HR", engineering: "Engineering", marketing: "Marketing", admin: "Admin" };

const DOCS_BASE = "https://github.com/Irfan-gitt/Enterprise-RAG-System-RBAC-/tree/main/resources";

// Document folders each role can see. Keep in sync with ROLE_PERMISSIONS in rbac.py.
const DOC_FOLDERS = {
  employee: ["general"],
  finance: ["financial", "general"],
  hr: ["hr", "general"],
  engineering: ["engineering", "general"],
  marketing: ["marketing", "general"],
  admin: ["financial", "hr", "engineering", "marketing", "general"],
};

// Clickable test questions per role. "deny" = a question that role should NOT be able to answer (RBAC check).
const EXAMPLES = {
  employee: {
    ask: [
      "What is our leave policy?",
      "Give me the company overview",
      "How many days of sick leave do I get, and when do I need a medical certificate?",
      "What is the maternity and paternity leave entitlement?",
      "What happens if I run out of paid leave?",
      "How do I apply for emergency leave?",
    ],
    deny: "Who is FINEMP1078?",
  },
  hr: {
    ask: [
      "Who is FINEMP1078?",
      "List the DevOps engineers working in Bengaluru",
      "List all HR Managers and their locations",
      "Who joined the company in 2024 in the Sales department?",
      "Which department does Isha Desai work in?",
      "Who is the manager of FINEMP1014, and what is their role?",
    ],
    deny: "What was the Q4 2024 marketing spend?",
  },
  finance: {
    ask: [
      "What was Q1 2024 revenue?",
      "How did revenue grow from Q1 to Q4 2024?",
      "Why did vendor costs go up in Q4?",
      "What risks were flagged in Q3 and how were they mitigated?",
      "What was the 2024 net income and cash flow from operations?",
      "What are the recommendations for 2025?",
    ],
    deny: "What is the salary of FINEMP1078?",
  },
  engineering: {
    ask: [
      "What are the RTO and RPO for disaster recovery?",
      "Which databases does FinSolve use?",
      "What is the code review process before merging?",
      "A test fails randomly then passes, what does the pipeline do?",
      "How do we stop old wrong data from being shown after a change?",
      "How fast must a critical S1 bug be fixed, and who do I contact for a security incident?",
    ],
    deny: "Who is FINEMP1078?",
  },
  marketing: {
    ask: [
      "What was the Q4 2024 customer acquisition target?",
      "How was the Q4 marketing spend allocated?",
      "How did the loyalty program perform?",
      "Did we hit the Q4 conversion rate target?",
      "What did account-based marketing deliver in Q4?",
      "What does the team recommend for Q1 2025?",
    ],
    deny: "What is the salary of FINEMP1078?",
  },
  admin: {
    ask: [
      "Give me the company overview",
      "Who is FINEMP1078?",
      "List the DevOps engineers working in Bengaluru",
      "List all HR Managers and their locations",
      "What was Q3 2024 revenue and net income?",
      "What are the RTO and RPO for disaster recovery?",
      "How did the Q4 conversion rate compare to the target?",
    ],
  },
};
function goToLogin() {
  Object.values(AUTH_KEYS).forEach((key) => sessionStorage.removeItem(key));
  window.location.replace("/");
}

const authToken = sessionStorage.getItem(AUTH_KEYS.token);
const userEmail = sessionStorage.getItem(AUTH_KEYS.email);
const userRole = sessionStorage.getItem(AUTH_KEYS.role);

if (!authToken || !userEmail || !userRole) {
  goToLogin();
} else {
  startChat();
}

// Renders markdown safely. Falls back to plain text if the libraries did not load.
function renderMarkdown(target, text) {
  if (!(window.marked && window.DOMPurify)) {
    target.textContent = text;
    return;
  }
  target.innerHTML = DOMPurify.sanitize(marked.parse(text, { gfm: true, breaks: true }));
  target.querySelectorAll("table").forEach((table) => {
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    table.replaceWith(wrap);
    wrap.appendChild(table);
  });
  target.querySelectorAll("a").forEach((a) => {
    a.target = "_blank";
    a.rel = "noopener noreferrer";
  });
}

function startChat() {
  const chatScreen = document.getElementById("chat-screen");
  const messagesEl = document.getElementById("messages");
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  const loadingIndicator = document.getElementById("loading-indicator");
  const exampleList = document.getElementById("example-list");
  const docLinks = document.getElementById("doc-links");

  let conversationId = null; // memory thread id, issued by the server
  const roleName = roleLabels[userRole] || userRole;

  chatScreen.hidden = false;
  document.getElementById("current-username").textContent = userEmail;
  document.getElementById("current-role-badge").textContent = roleName;

  function addMessage(kind, text) {
    const row = document.createElement("div");
    row.className = `msg-row ${kind}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (kind === "assistant") renderMarkdown(bubble, text);
    else bubble.textContent = text;
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // Persistent test panel: built once per page load, never removed.
  function buildTestPanel() {
    const set = EXAMPLES[userRole];
    if (set) {
      const addChip = (question, isDeny) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "example-q-btn" + (isDeny ? " deny" : "");
        btn.textContent = isDeny ? `Access test: ${question}` : question;
        btn.addEventListener("click", () => {
          if (!sendBtn.disabled) runQuery(question);
        });
        exampleList.appendChild(btn);
      };
      set.ask.forEach((q) => addChip(q, false));
      if (set.deny) addChip(set.deny, true);
    }

    (DOC_FOLDERS[userRole] || []).forEach((folder) => {
      const a = document.createElement("a");
      a.className = "doc-link";
      a.href = `${DOCS_BASE}/${encodeURIComponent(folder)}`;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = `${folder} ↗`;
      docLinks.appendChild(a);
    });
  }

  function resizeInput() {
    chatInput.style.height = "auto";
    chatInput.style.height = `${Math.min(chatInput.scrollHeight, 140)}px`;
  }

  function resetChat() {
    messagesEl.innerHTML = "";
    conversationId = null; // fresh memory thread
    addMessage("assistant", `Welcome. You are signed in as **${roleName}**. Ask about the company information you are authorized to access, or use the test panel above.`);
  }

  async function sendChatMessage(question) {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${authToken}` },
      body: JSON.stringify({ question, conversation_id: conversationId }),
    });
    const data = await response.json().catch(() => ({}));
    if (response.status === 401) { goToLogin(); throw new Error("Session expired. Please sign in again."); }
    if (response.status === 403) throw new Error("You do not have permission to access this information.");
    if (!response.ok) throw new Error(data.detail || "Unable to retrieve an answer.");
    conversationId = data.conversation_id; // keep the same thread for the next message
    return data;
  }

  async function runQuery(question) {
    addMessage("user", question);
    loadingIndicator.hidden = false;
    sendBtn.disabled = true;
    try {
      const result = await sendChatMessage(question);
      addMessage("assistant", result.answer);
    } catch (error) {
      addMessage("assistant", error.message || "Something went wrong.");
    } finally {
      loadingIndicator.hidden = true;
      sendBtn.disabled = false;
      chatInput.focus();
    }
  }

  chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = chatInput.value.trim();
    if (!question || sendBtn.disabled) return;
    chatInput.value = "";
    resizeInput();
    runQuery(question);
  });

  chatInput.addEventListener("input", resizeInput);
  chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      chatForm.requestSubmit();
    }
  });

  document.getElementById("logout-btn").addEventListener("click", goToLogin);
  document.getElementById("clear-chat-btn").addEventListener("click", resetChat);

  buildTestPanel();
  resetChat();
  chatInput.focus();
}