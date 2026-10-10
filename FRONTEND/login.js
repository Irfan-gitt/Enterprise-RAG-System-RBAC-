const API_BASE = window.location.origin;
const AUTH_KEYS = { email: "company_email", role: "company_role", token: "company_token" };

// already signed in -> go straight to the chat page
if (sessionStorage.getItem(AUTH_KEYS.token)) window.location.replace("/chat");

const DEMO_PASSWORD = "demo123";
const DEMO_ACCOUNTS = [
  { role: "Employee", email: "employee@company.com", access: "General documents" },
  { role: "HR", email: "hr@company.com", access: "HR + General" },
  { role: "Finance", email: "finance@company.com", access: "Financial + General" },
  { role: "Engineering", email: "engineering@company.com", access: "Engineering + General" },
  { role: "Marketing", email: "marketing@company.com", access: "Marketing + General" },
  { role: "Admin", email: "admin@company.com", access: "All departments" },
];

const loginScreen = document.getElementById("login-screen");
const loginForm = document.getElementById("login-form");
const usernameInput = document.getElementById("username");
const passwordInput = document.getElementById("password");
const loginError = document.getElementById("login-error");
const loginBtn = document.getElementById("login-btn");
const loginBtnText = document.getElementById("login-btn-text");
const loginSpinner = document.getElementById("login-spinner");
const demoList = document.getElementById("demo-list");

loginScreen.hidden = false;

function setLoginLoading(loading) {
  loginBtn.disabled = loading;
  loginSpinner.hidden = !loading;
  loginBtnText.textContent = loading ? "Signing in…" : "Sign In";
  demoList.querySelectorAll("button").forEach((b) => { b.disabled = loading; });
}

function showLoginError(message) {
  loginError.textContent = message;
  loginError.hidden = false;
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
  // Display-only. The backend reads the role from the signed JWT.
  sessionStorage.setItem(AUTH_KEYS.role, data.user.role);
}

// Demo accounts panel: click an account to fill the form and sign in
DEMO_ACCOUNTS.forEach((account) => {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "demo-item";

  const role = document.createElement("span");
  role.className = "demo-role";
  role.textContent = account.role;

  const email = document.createElement("span");
  email.className = "demo-email";
  email.textContent = account.email;

  const access = document.createElement("span");
  access.className = "demo-access";
  access.textContent = `Access: ${account.access}`;

  btn.append(role, email, access);
  btn.addEventListener("click", () => {
    if (loginBtn.disabled) return;
    usernameInput.value = account.email;
    passwordInput.value = DEMO_PASSWORD;
    loginError.hidden = true;
    loginBtn.focus(); // user presses Enter or clicks Sign In
  });
  demoList.appendChild(btn);
});

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  loginError.hidden = true;
  setLoginLoading(true);
  try {
    await login(usernameInput.value.trim(), passwordInput.value);
    window.location.replace("/chat");
  } catch (error) {
    showLoginError(error.message || "Unable to sign in.");
    setLoginLoading(false);
  }
});

document.getElementById("toggle-password").addEventListener("click", () => {
  passwordInput.type = passwordInput.type === "password" ? "text" : "password";
});