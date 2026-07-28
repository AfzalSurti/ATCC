const TOKEN_KEY = "atcc_token";
const EMAIL_KEY = "atcc_email";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function getSavedEmail(): string | null {
  return localStorage.getItem(EMAIL_KEY);
}

export function setSession(token: string, email: string): void {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(EMAIL_KEY, email);
}

export function clearSession(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(EMAIL_KEY);
}
