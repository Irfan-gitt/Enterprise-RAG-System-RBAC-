const API_BASE = window.location.origin;
const AUTH_KEYS = { email: "company_email", role: "company_role", token: "company_token" };

// already signed in -> go straight to the chat page
if (sessionStorage.getItem(AUTH_KEYS.token)) window.location.replace("/chat");

const loginScreen = document.getElementById("login-screen");
const loginForm = document.getElementById("login-form");
const usernameInput = document.getElementById("username");
const passwordInput = document.getElementById("password");
const loginError = document.getElementById("login-error");
const loginBtn = document.getElementById("login-btn");
const loginBtnText = document.getElementById("login-btn-text");
const loginSpinner = document.getElementById("login-spinner");

loginScreen.hidden = false;

function setLoginLoading(loading) {
  loginBtn.disabled = loading;
  loginSpinner.hidden = !loading;
  loginBtnText.textContent = loading ? "Signing in…" : "Sign In";
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