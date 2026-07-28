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

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

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
    body: form,
  });
  return parseJson<BatchJob>(res);
}

export async function fetchJob(jobId: string): Promise<BatchJob> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
  return parseJson<BatchJob>(res);
}

export async function fetchJobs(): Promise<BatchJob[]> {
  const res = await fetch(`${API_BASE}/api/jobs`);
  const data = await parseJson<{ jobs: BatchJob[] }>(res);
  return data.jobs;
}

export async function cancelJob(jobId: string): Promise<BatchJob> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/cancel`, { method: "POST" });
  return parseJson<BatchJob>(res);
}

export function reportDownloadUrl(job: BatchJob, video: VideoResult): string | null {
  if (!video.report_url) return null;
  return `${API_BASE}${video.report_url}`;
}
