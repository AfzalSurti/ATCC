import { useCallback, useEffect, useMemo, useState } from "react";
import {
  BatchJob,
  cancelJob,
  fetchJob,
  fetchJobs,
  reportDownloadUrl,
  uploadVideos,
} from "./api";
import { UploadZone } from "./UploadZone";
import { JobCard } from "./JobCard";

export default function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [jobs, setJobs] = useState<BatchJob[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeIds, setActiveIds] = useState<string[]>([]);

  const refreshList = useCallback(async () => {
    try {
      const list = await fetchJobs();
      setJobs(list);
      setActiveIds(
        list.filter((j) => j.status === "queued" || j.status === "running").map((j) => j.job_id),
      );
    } catch (err) {
      // API may not be up yet on first paint
      console.warn(err);
    }
  }, []);

  useEffect(() => {
    void refreshList();
  }, [refreshList]);

  // Poll active jobs
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
    }, 1500);
    return () => window.clearInterval(timer);
  }, [activeIds]);

  const onUpload = async () => {
    if (files.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      const job = await uploadVideos(files);
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

  const pendingLabel = useMemo(() => {
    if (files.length === 0) return "No videos selected";
    if (files.length === 1) return "1 video ready";
    return `${files.length} videos ready`;
  }, [files.length]);

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <div className="brand-mark">ATCC</div>
          <h1>Traffic counter</h1>
          <p>
            Upload one or more roadside videos. The server detects, tracks, and counts vehicles,
            then returns Excel reports — no copying files into a backend folder.
          </p>
        </div>
      </header>

      <section className="panel">
        <UploadZone files={files} onChange={setFiles} />
        <div className="actions">
          <button className="btn btn-primary" disabled={files.length === 0 || uploading} onClick={onUpload}>
            {uploading ? "Uploading…" : `Process ${pendingLabel}`}
          </button>
          <button className="btn btn-ghost" disabled={files.length === 0 || uploading} onClick={() => setFiles([])}>
            Clear selection
          </button>
        </div>
        {error ? <div className="error-banner">{error}</div> : null}
      </section>

      <div className="section-title">
        <h2>Jobs</h2>
        <span>{jobs.length === 0 ? "Upload to create a job" : `${jobs.length} job(s)`}</span>
      </div>

      <div className="job-list">
        {jobs.length === 0 ? (
          <div className="panel empty">No jobs yet. Drop MP4 / MOV / MKV / WEBM files above.</div>
        ) : (
          jobs.map((job) => (
            <JobCard
              key={job.job_id}
              job={job}
              onCancel={() => void onCancel(job.job_id)}
              reportUrl={(video) => reportDownloadUrl(job, video)}
            />
          ))
        )}
      </div>
    </div>
  );
}
