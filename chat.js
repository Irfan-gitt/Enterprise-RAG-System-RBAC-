const API_BASE = window.location.origin;
const AUTH_KEYS = { email: "company_email", role: "company_role", token: "company_token" };
const roleLabels = { employee: "Employee", finance: "Finance", hr: "HR", engineering: "Engineering", marketing: "Marketing", admin: "Admin" };

// Clickable test questions per role. "deny" = a question that role should NOT be able to answer (RBAC check).
const EXAMPLES = {
  employee: {
    ask: ["What is our leave policy?", "Give me the company overview", "How do I apply for leave?"],
    deny: "Who is FINEMP1078?",
  },
  hr: {
    ask: ["Who is FINEMP1078?", "How many sick leaves do employees get?", "What is our leave policy?"],
    deny: "What was the Q4 2024 marketing spend?",
  },
  finance: {
    ask: ["What was Q1 2024 revenue?", "What was last year's company turnover?", "When is the reimbursement deadline?"],
    deny: "What is the salary of FINEMP1078?",
  },
  engineering: {
    ask: ["What are the RTO and RPO for disaster recovery?", "Which databases does FinSolve use?", "What is the minimum unit test coverage?"],
    deny: "Who is FINEMP1078?",
  },
  marketing: {
    ask: ["What was the Q4 2024 customer acquisition target?", "How was the Q4 marketing spend allocated?", "How did the loyalty program perform?"],
    deny: "What is the salary of FINEMP1078?",
  },
  admin: {
    ask: ["Give me the company overview", "Who is FINEMP1078?", "What was the Q4 2024 marketing spend?"],
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
    return bubble;
  }

  function addExamples(bubble) {
    const set = EXAMPLES[userRole];
    if (!set) return;
    const wrap = document.createElement("div");
    wrap.className = "example-questions";

    const addChip = (question, isDeny) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "example-q-btn" + (isDeny ? " deny" : "");
      btn.textContent = isDeny ? `Access test: ${question}` : question;
      btn.addEventListener("click", () => {
        if (!sendBtn.disabled) runQuery(question);
      });
      wrap.appendChild(btn);
    };

    set.ask.forEach((q) => addChip(q, false));
    if (set.deny) addChip(set.deny, true);
    bubble.appendChild(wrap);
  }

  function resizeInput() {
    chatInput.style.height = "auto";
    chatInput.style.height = `${Math.min(chatInput.scrollHeight, 140)}px`;
  }

  function resetChat() {
    messagesEl.innerHTML = "";
    conversationId = null; // fresh memory thread
    const bubble = addMessage(
      "assistant",
      `Welcome. You are signed in as **${roleName}**. Ask about the company information you are authorized to access. Try one of these to get started:`
    );
    addExamples(bubble);
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
    messagesEl.querySelectorAll(".example-questions").forEach((el) => el.remove());
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

  resetChat();
  chatInput.focus();
}