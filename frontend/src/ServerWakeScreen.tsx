import { useEffect, useState } from "react";

const MESSAGES = [
  "Waking the server…",
  "Render is starting the API (first load can take up to a minute)…",
  "Almost there — loading the backend…",
  "Still starting — free-tier servers sleep when idle…",
];

interface Props {
  title?: string;
  detail?: string;
  attempt?: number;
}

export function ServerWakeScreen({
  title = "Starting backend",
  detail,
  attempt = 0,
}: Props) {
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const id = window.setInterval(() => setTick((t) => t + 1), 3200);
    return () => window.clearInterval(id);
  }, []);

  const message = detail ?? MESSAGES[Math.min(tick, MESSAGES.length - 1)];

  return (
    <div className="app wake-page" role="status" aria-live="polite">
      <div className="panel wake-card">
        <div className="brand-mark">ATCC</div>
        <h1>{title}</h1>
        <div className="wake-visual" aria-hidden="true">
          <span className="wake-ring" />
          <span className="wake-ring wake-ring-delay" />
          <span className="wake-core" />
        </div>
        <p className="wake-message">{message}</p>
        <p className="wake-hint">
          The API runs on Render and sleeps after idle time. This screen clears once it responds.
          {attempt > 0 ? ` · try ${attempt}` : null}
        </p>
        <div className="wake-bar" aria-hidden="true">
          <i />
        </div>
      </div>
    </div>
  );
}
