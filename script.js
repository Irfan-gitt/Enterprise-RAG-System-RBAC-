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
  addMessage("assistant", `Welcome. You are signed in as ${roleLabels[user.role] || user.role}. Ask about the company information you are authorized to access.`);
}

function addMessage(kind, text, result = null) {
  const row = document.createElement("div");
  row.className = `msg-row ${kind}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  if (result?.sources?.length) {
    const sources = document.createElement("div");
    sources.className = "sources-row";
    result.sources.forEach((source) => {
      const chip = document.createElement("span");
      chip.className = "source-chip";
      chip.textContent = `Source: ${source}`;
      sources.appendChild(chip);
    });
    bubble.appendChild(sources);
  }

  if (result?.pagination) addPagination(bubble, result.pagination);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addPagination(container, pagination) {
  const row = document.createElement("div");
  row.className = "pagination-row";
  row.append(`Page ${pagination.page} of ${pagination.total_pages} — ${pagination.total_matches} total records`);
  for (const [label, nextPage, disabled] of [
    ["Previous", pagination.page - 1, pagination.page <= 1],
    ["Next", pagination.page + 1, pagination.page >= pagination.total_pages],
  ]) {
    const button = document.createElement("button");
    button.textContent = label;
    button.disabled = disabled;
    button.addEventListener("click", () => lastQuestion && runQuery(lastQuestion, nextPage, false));
    row.appendChild(button);
  }
  container.appendChild(row);
}

async function sendChatMessage(question, page) {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${getAuthToken()}` },
    body: JSON.stringify({ question, page, page_size: 10 }),
  });
  const data = await response.json().catch(() => ({}));
  if (response.status === 401) { logout(); throw new Error("Session expired. Please sign in again."); }
  if (response.status === 403) throw new Error("You do not have permission to access this information.");
  if (!response.ok) throw new Error(data.detail || "Unable to retrieve an answer.");
  return data;
}

async function runQuery(question, page = 1, addUser = true) {
  if (addUser) addMessage("user", question);
  loadingIndicator.hidden = false;
  sendBtn.disabled = true;
  try {
    const result = await sendChatMessage(question, page);
    addMessage("assistant", result.answer, result);
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
