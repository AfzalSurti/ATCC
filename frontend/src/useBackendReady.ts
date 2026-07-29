import { useCallback, useEffect, useRef, useState } from "react";
import { waitForBackend } from "./api";

export type BackendState = "checking" | "ready" | "error";

/** Poll /api/health until Render finishes cold start. */
export function useBackendReady() {
  const [state, setState] = useState<BackendState>("checking");
  const [attempt, setAttempt] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const cancelled = useRef(false);

  const run = useCallback(async () => {
    cancelled.current = false;
    setState("checking");
    setError(null);
    setAttempt(0);
    try {
      await waitForBackend({
        onAttempt: (n) => {
          if (!cancelled.current) setAttempt(n);
        },
      });
      if (!cancelled.current) setState("ready");
    } catch (err) {
      if (cancelled.current) return;
      setError(err instanceof Error ? err.message : String(err));
      setState("error");
    }
  }, []);

  useEffect(() => {
    void run();
    return () => {
      cancelled.current = true;
    };
  }, [run]);

  return { state, attempt, error, retry: run };
}
