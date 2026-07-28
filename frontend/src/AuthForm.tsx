import { FormEvent, useState } from "react";

interface Props {
  mode: "login" | "signup";
  onSubmit: (email: string, password: string) => Promise<void>;
  onSwitch: () => void;
}

export function AuthForm({ mode, onSubmit, onSwitch }: Props) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const title = mode === "login" ? "Log in" : "Create account";
  const switchLabel = mode === "login" ? "Need an account? Sign up" : "Already have an account? Log in";

  const handle = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await onSubmit(email.trim(), password);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="app auth-page">
      <header className="top">
        <div className="brand">
          <div className="brand-mark">ATCC</div>
          <h1>{title}</h1>
          <p>
            Sign in to start traffic counting. Jobs keep running on the server if you close this tab —
            open your account later to see progress and download Excel.
          </p>
        </div>
      </header>

      <form className="panel auth-panel" onSubmit={(e) => void handle(e)}>
        <label className="field">
          <span>Email</span>
          <input
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
          />
        </label>
        <label className="field">
          <span>Password</span>
          <input
            type="password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            required
            minLength={6}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="At least 6 characters"
          />
        </label>
        {error ? <div className="error-banner">{error}</div> : null}
        <div className="actions">
          <button className="btn btn-primary" type="submit" disabled={busy}>
            {busy ? "Please wait…" : title}
          </button>
          <button className="btn btn-ghost" type="button" onClick={onSwitch} disabled={busy}>
            {switchLabel}
          </button>
        </div>
      </form>
    </div>
  );
}
