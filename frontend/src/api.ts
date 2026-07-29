export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
export type JunctionType = "two_way" | "three_way" | "four_way";

export interface JunctionTypeInfo {
  id: JunctionType;
  label: string;
  arms: number;
  movements: number;
  description: string;
}

export interface VideoResult {
  video_id: string;
  filename: string;
  status: JobStatus;
  total_events: number;
  report_path: string | null;
  annotated_path: string | null;
  error: string | null;
  lane_counts: Record<string, number>;
  movement_counts: Record<string, number>;
  class_counts: Record<string, number>;
  frames_done: number;
  frames_total: number;
  video_progress: number;
  fps: number;
  message: string;
  report_url: string | null;
}

export interface BatchJob {
  job_id: string;
  created_at: string;
  status: JobStatus;
  progress: number;
  error: string | null;
  junction_type: JunctionType | string;
  expected_movements: number;
  videos: VideoResult[];
}

export interface AuthUser {
  access_token: string;
  email: string;
  user_id: number;
}

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

function authHeaders(): HeadersInit {
  const token = localStorage.getItem("atcc_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

/** Single health check — used to detect Render cold start. */
export async function pingHealth(timeoutMs = 20000): Promise<boolean> {
  const ctrl = new AbortController();
  const timer = window.setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}/api/health`, {
      signal: ctrl.signal,
      cache: "no-store",
    });
    if (!res.ok) return false;
    const data = (await res.json()) as { status?: string };
    return data.status === "ok";
  } catch {
    return false;
  } finally {
    window.clearTimeout(timer);
  }
}

export interface WaitForBackendOptions {
  maxAttempts?: number;
  onAttempt?: (attempt: number) => void;
}

/** Keep pinging until the Render API wakes (or give up). */
export async function waitForBackend(options: WaitForBackendOptions = {}): Promise<void> {
  const maxAttempts = options.maxAttempts ?? 40;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    options.onAttempt?.(attempt);
    const ok = await pingHealth(attempt === 1 ? 25000 : 15000);
    if (ok) return;
    // Brief pause between tries (Render cold start often needs 30–90s)
    await new Promise((r) => window.setTimeout(r, attempt < 3 ? 1500 : 2500));
  }
  throw new Error(
    "Backend did not wake up in time. Open your Render service, confirm it is live, then retry.",
  );
}

export async function signup(email: string, password: string): Promise<AuthUser> {
  const res = await fetch(`${API_BASE}/api/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return parseJson<AuthUser>(res);
}

export async function login(email: string, password: string): Promise<AuthUser> {
  const res = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return parseJson<AuthUser>(res);
}

export async function fetchMe(): Promise<{ user_id: number; email: string }> {
  const res = await fetch(`${API_BASE}/api/auth/me`, { headers: authHeaders() });
  return parseJson(res);
}

export async function fetchJunctionTypes(): Promise<JunctionTypeInfo[]> {
  const res = await fetch(`${API_BASE}/api/junction-types`);
  const data = await parseJson<{ types: JunctionTypeInfo[] }>(res);
  return data.types;
}

export async function uploadVideos(files: File[], junctionType: JunctionType): Promise<BatchJob> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  form.append("junction_type", junctionType);
  const res = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  return parseJson<BatchJob>(res);
}

export async function fetchJob(jobId: string): Promise<BatchJob> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`, { headers: authHeaders() });
  return parseJson<BatchJob>(res);
}

export async function fetchJobs(): Promise<BatchJob[]> {
  const res = await fetch(`${API_BASE}/api/jobs`, { headers: authHeaders() });
  const data = await parseJson<{ jobs: BatchJob[] }>(res);
  return data.jobs;
}

export async function cancelJob(jobId: string): Promise<BatchJob> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/cancel`, {
    method: "POST",
    headers: authHeaders(),
  });
  return parseJson<BatchJob>(res);
}

export async function downloadReport(job: BatchJob, video: VideoResult): Promise<void> {
  if (!video.report_url) return;
  const res = await fetch(`${API_BASE}${video.report_url}`, { headers: authHeaders() });
  if (!res.ok) {
    throw new Error("Failed to download report");
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${video.filename.replace(/\.[^.]+$/, "")}_counts.xlsx`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
