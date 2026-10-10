const API_BASE = window.location.origin;
const AUTH_KEYS = { email: "company_email", role: "company_role", token: "company_token" };

const loginScreen = document.getElementById("login-screen");
const chatScreen = document.getElementById("chat-screen");
const loginForm = document.getElementById("login-form");
const usernameInput = document.getElementById("username");
const passwordInput = document.getElementById("password");
const loginError = document.getElementById("login-error");
const loginBtn = document.getElementById("login-btn");
const loginBtnText = document.getElementById("login-btn-text");
const loginSpinner = document.getElementById("login-spinner");
const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const loadingIndicator = document.getElementById("loading-indicator");

let lastQuestion = null;
let conversationId = null; // memory thread id, issued by the server
const roleLabels = { employee: "Employee", finance: "Finance", hr: "HR", engineering: "Engineering", marketing: "Marketing", admin: "Admin" };

function getCurrentUser() {
  const email = sessionStorage.getItem(AUTH_KEYS.email);
  const role = sessionStorage.getItem(AUTH_KEYS.role);
  return email && role ? { email, role } : null;
}

function getAuthToken() { return sessionStorage.getItem(AUTH_KEYS.token); }

function setLoginLoading(loading) {
  loginBtn.disabled = loading;
  loginSpinner.hidden = !loading;
  loginBtnText.textContent = loading ? "Signing in…" : "Sign In";
}

function showLoginError(message) {
  loginError.textContent = message;
  loginError.hidden = false;
}

function showLoginScreen() {
  loginScreen.hidden = false;
  chatScreen.hidden = true;
  loginForm.reset();
  loginError.hidden = true;
}

async function login(email, password) {
  const response = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "Unable to sign in.");
  sessionStorage.setItem(AUTH_KEYS.token, data.access_token);
  sessionStorage.setItem(AUTH_KEYS.email, data.user.email);
  // This is display-only. The backend extracts the role from the signed JWT.
  sessionStorage.setItem(AUTH_KEYS.role, data.user.role);
}

function logout() {
  Object.values(AUTH_KEYS).forEach((key) => sessionStorage.removeItem(key));
  lastQuestion = null;
  conversationId = null;
  showLoginScreen();
}

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  loginError.hidden = true;
  setLoginLoading(true);
  try {
    await login(usernameInput.value.trim(), passwordInput.value);
    showChatScreen();
  } catch (error) {
    showLoginError(error.message || "Unable to sign in.");
  } finally {
    setLoginLoading(false);
  }
});

document.getElementById("toggle-password").addEventListener("click", () => {
  passwordInput.type = passwordInput.type === "password" ? "text" : "password";
});
document.getElementById("logout-btn").addEventListener("click", logout);
document.getElementById("clear-chat-btn").addEventListener("click", () => showChatScreen());

function showChatScreen() {
  const user = getCurrentUser();
  if (!user) return showLoginScreen();
  loginScreen.hidden = true;
  chatScreen.hidden = false;
  document.getElementById("current-username").textContent = user.email;
  document.getElementById("current-role-badge").textContent = roleLabels[user.role] || user.role;
  messagesEl.innerHTML = "";
  lastQuestion = null;
  conversationId = null; // clearing the chat starts a fresh memory thread
  addMessage("assistant", `Welcome. You are signed in as ${roleLabels[user.role] || user.role}. Ask about the company information you are authorized to access.`);
}

function addMessage(kind, text) {
  const row = document.createElement("div");
  row.className = `msg-row ${kind}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendChatMessage(question) {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${getAuthToken()}` },
    body: JSON.stringify({ question, conversation_id: conversationId }),
  });
  const data = await response.json().catch(() => ({}));
  if (response.status === 401) { logout(); throw new Error("Session expired. Please sign in again."); }
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
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;
  chatInput.value = "";
  lastQuestion = question;
  runQuery(question);
});
chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); chatForm.requestSubmit(); }
});

if (getCurrentUser() && getAuthToken()) showChatScreen(); else showLoginScreen();