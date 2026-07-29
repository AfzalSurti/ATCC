import { useCallback, useEffect, useMemo, useState } from "react";
import {
  BatchJob,
  JunctionType,
  JunctionTypeInfo,
  cancelJob,
  downloadReport,
  fetchJob,
  fetchJobs,
  fetchJunctionTypes,
  fetchMe,
  login,
  signup,
  uploadVideos,
} from "./api";
import { AuthForm } from "./AuthForm";
import { UploadZone } from "./UploadZone";
import { JobCard } from "./JobCard";
import { ServerWakeScreen } from "./ServerWakeScreen";
import { clearSession, getSavedEmail, getToken, setSession } from "./authStorage";
import { useBackendReady } from "./useBackendReady";

const FALLBACK_TYPES: JunctionTypeInfo[] = [
  {
    id: "two_way",
    label: "2-way road",
    arms: 1,
    movements: 2,
    description: "Incoming + outgoing (2 ways).",
  },
  {
    id: "three_way",
    label: "3-way / T-junction",
    arms: 3,
    movements: 6,
    description: "Three approaches × IN + OUT (6 ways).",
  },
  {
    id: "four_way",
    label: "4-way crossroads",
    arms: 4,
    movements: 8,
    description: "Four approaches × IN + OUT (8 ways).",
  },
];

type AuthMode = "login" | "signup";

export default function App() {
  const backend = useBackendReady();
  const [token, setToken] = useState<string | null>(() => getToken());
  const [email, setEmail] = useState<string | null>(() => getSavedEmail());
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [authChecking, setAuthChecking] = useState(Boolean(getToken()));

  const [files, setFiles] = useState<File[]>([]);
  const [jobs, setJobs] = useState<BatchJob[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadSlow, setUploadSlow] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeIds, setActiveIds] = useState<string[]>([]);
  const [junctionType, setJunctionType] = useState<JunctionType>("four_way");
  const [junctionTypes, setJunctionTypes] = useState<JunctionTypeInfo[]>(FALLBACK_TYPES);

  const logout = useCallback(() => {
    clearSession();
    setToken(null);
    setEmail(null);
    setJobs([]);
    setActiveIds([]);
    setError(null);
  }, []);

  useEffect(() => {
    if (backend.state !== "ready") return;
    if (!token) {
      setAuthChecking(false);
      return;
    }
    let cancelled = false;
    void fetchMe()
      .then((me) => {
        if (cancelled) return;
        setEmail(me.email);
        setAuthChecking(false);
      })
      .catch(() => {
        if (cancelled) return;
        logout();
        setAuthChecking(false);
      });
    return () => {
      cancelled = true;
    };
  }, [backend.state, token, logout]);

  const refreshList = useCallback(async () => {
    if (!getToken()) return;
    try {
      const list = await fetchJobs();
      setJobs(list);
      setActiveIds(
        list.filter((j) => j.status === "queued" || j.status === "running").map((j) => j.job_id),
      );
    } catch (err) {
      console.warn(err);
      if (err instanceof Error && /not authenticated|invalid|401/i.test(err.message)) {
        logout();
      }
    }
  }, [logout]);

  useEffect(() => {
    if (backend.state !== "ready" || !token || authChecking) return;
    void refreshList();
    void fetchJunctionTypes()
      .then(setJunctionTypes)
      .catch(() => setJunctionTypes(FALLBACK_TYPES));
  }, [backend.state, token, authChecking, refreshList]);

  useEffect(() => {
    if (activeIds.length === 0) return;
    const timer = window.setInterval(async () => {
      try {
        const updated = await Promise.all(activeIds.map((id) => fetchJob(id)));
        setJobs((prev) => {
          const map = new Map(prev.map((j) => [j.job_id, j]));
          for (const job of updated) map.set(job.job_id, job);
          return Array.from(map.values()).sort((a, b) => b.created_at.localeCompare(a.created_at));
        });
        setActiveIds(
          updated.filter((j) => j.status === "queued" || j.status === "running").map((j) => j.job_id),
        );
      } catch (err) {
        console.warn(err);
      }
    }, 700);
    return () => window.clearInterval(timer);
  }, [activeIds]);

  useEffect(() => {
    if (!uploading) {
      setUploadSlow(false);
      return;
    }
    const timer = window.setTimeout(() => setUploadSlow(true), 3000);
    return () => window.clearTimeout(timer);
  }, [uploading]);

  const onAuth = async (authEmail: string, password: string) => {
    const result = authMode === "login" ? await login(authEmail, password) : await signup(authEmail, password);
    setSession(result.access_token, result.email);
    setToken(result.access_token);
    setEmail(result.email);
  };

  const selectedInfo = junctionTypes.find((t) => t.id === junctionType) ?? FALLBACK_TYPES[2];

  const onUpload = async () => {
    if (files.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      const job = await uploadVideos(files, junctionType);
      setFiles([]);
      setJobs((prev) => [job, ...prev.filter((j) => j.job_id !== job.job_id)]);
      setActiveIds((ids) => Array.from(new Set([job.job_id, ...ids])));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  const onCancel = async (jobId: string) => {
    try {
      const job = await cancelJob(jobId);
      setJobs((prev) => prev.map((j) => (j.job_id === jobId ? job : j)));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const onDownload = async (job: BatchJob, videoId: string) => {
    const video = job.videos.find((v) => v.video_id === videoId);
    if (!video) return;
    try {
      await downloadReport(job, video);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const pendingLabel = useMemo(() => {
    if (files.length === 0) return "No videos selected";
    if (files.length === 1) return "1 video ready";
    return `${files.length} videos ready`;
  }, [files.length]);

  if (backend.state === "checking") {
    return <ServerWakeScreen attempt={backend.attempt} />;
  }

  if (backend.state === "error") {
    return (
      <div className="app wake-page">
        <div className="panel wake-card">
          <div className="brand-mark">ATCC</div>
          <h1>Backend not reachable</h1>
          <p className="wake-message">{backend.error}</p>
          <div className="actions" style={{ justifyContent: "center" }}>
            <button className="btn btn-primary" type="button" onClick={() => void backend.retry()}>
              Retry wake-up
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (authChecking) {
    return <ServerWakeScreen title="Checking your session" detail="Confirming login with the API…" />;
  }

  if (!token) {
    return (
      <AuthForm
        mode={authMode}
        onSubmit={onAuth}
        onSwitch={() => setAuthMode((m) => (m === "login" ? "signup" : "login"))}
      />
    );
  }

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <div className="brand-mark">ATCC</div>
          <h1>Traffic counter</h1>
          <p>
            Upload roadside videos and count vehicles on every junction movement. Processing runs on
            the server — you can close this tab and come back to this account anytime.
          </p>
        </div>
        <div className="account-bar">
          <span className="account-email">{email}</span>
          <button className="btn btn-ghost" type="button" onClick={logout}>
            Log out
          </button>
        </div>
      </header>

      <section className="panel">
        <div className="junction-picker">
          <div className="section-title" style={{ marginTop: 0 }}>
            <h2>Junction type</h2>
            <span>{selectedInfo.movements} counting ways</span>
          </div>
          <div className="junction-options">
            {junctionTypes.map((t) => (
              <button
                key={t.id}
                type="button"
                className={`junction-card ${junctionType === t.id ? "selected" : ""}`}
                onClick={() => setJunctionType(t.id)}
              >
                <strong>{t.label}</strong>
                <span>
                  {t.movements} ways · {t.arms} arm(s)
                </span>
                <em>{t.description}</em>
              </button>
            ))}
          </div>
        </div>

        <UploadZone files={files} onChange={setFiles} />
        <div className="actions">
          <button className="btn btn-primary" disabled={files.length === 0 || uploading} onClick={onUpload}>
            {uploading
              ? uploadSlow
                ? "Uploading (server may be busy)…"
                : "Uploading…"
              : `Process ${pendingLabel} (${selectedInfo.movements}-way)`}
          </button>
          <button className="btn btn-ghost" disabled={files.length === 0 || uploading} onClick={() => setFiles([])}>
            Clear selection
          </button>
        </div>
        {uploading && uploadSlow ? (
          <div className="wake-inline" role="status">
            <span className="wake-inline-spin" aria-hidden="true" />
            <span>Large uploads or a waking server can take a while — keep this tab open.</span>
          </div>
        ) : null}
        {error ? <div className="error-banner">{error}</div> : null}
      </section>

      <div className="section-title">
        <h2>Your jobs</h2>
        <span>{jobs.length === 0 ? "Upload to create a job" : `${jobs.length} job(s)`}</span>
      </div>

      <div className="job-list">
        {jobs.length === 0 ? (
          <div className="panel empty">No jobs yet. Choose a junction type, then drop videos.</div>
        ) : (
          jobs.map((job) => (
            <JobCard
              key={job.job_id}
              job={job}
              onCancel={() => void onCancel(job.job_id)}
              onDownload={(videoId) => void onDownload(job, videoId)}
            />
          ))
        )}
      </div>
    </div>
  );
}
