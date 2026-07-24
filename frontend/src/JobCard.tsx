import { BatchJob, VideoResult } from "./api";

interface Props {
  job: BatchJob;
  onCancel: () => void;
  reportUrl: (video: VideoResult) => string | null;
}

export function JobCard({ job, onCancel, reportUrl }: Props) {
  const running = job.status === "running" || job.status === "queued";

  return (
    <article className="panel job">
      <div className="job-head">
        <div className="meta">
          <strong>Job {job.job_id}</strong>
          <span>{new Date(job.created_at).toLocaleString()} · {job.videos.length} video(s)</span>
        </div>
        <div className="actions" style={{ marginTop: 0 }}>
          <span className={`badge ${job.status}`}>{job.status}</span>
          {running ? (
            <button className="btn btn-ghost" type="button" onClick={onCancel}>
              Cancel
            </button>
          ) : null}
        </div>
      </div>

      <div className="progress" aria-label="Batch progress">
        <i style={{ width: `${Math.round(job.progress * 100)}%` }} />
      </div>

      <div className="video-grid">
        {job.videos.map((video) => {
          const url = reportUrl(video);
          const lanes = Object.entries(video.lane_counts);
          const classes = Object.entries(video.class_counts);
          return (
            <div className="video-row" key={video.video_id}>
              <div className="video-row-top">
                <h3>{video.filename}</h3>
                <span className={`badge ${video.status}`}>{video.status}</span>
              </div>
              <div className="counts">
                <span>
                  Counts <b>{video.total_events}</b>
                </span>
                {video.error ? <span style={{ color: "var(--danger)" }}>{video.error}</span> : null}
              </div>
              {(lanes.length > 0 || classes.length > 0) && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
                  {lanes.length > 0 ? (
                    <table className="mini-table">
                      <thead>
                        <tr>
                          <th>Lane</th>
                          <th>Count</th>
                        </tr>
                      </thead>
                      <tbody>
                        {lanes.map(([k, v]) => (
                          <tr key={k}>
                            <td>{k}</td>
                            <td>{v}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : null}
                  {classes.length > 0 ? (
                    <table className="mini-table">
                      <thead>
                        <tr>
                          <th>Class</th>
                          <th>Count</th>
                        </tr>
                      </thead>
                      <tbody>
                        {classes.map(([k, v]) => (
                          <tr key={k}>
                            <td>{k}</td>
                            <td>{v}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : null}
                </div>
              )}
              {url ? (
                <div>
                  <a className="btn btn-primary" href={url} style={{ display: "inline-block", textDecoration: "none" }}>
                    Download Excel
                  </a>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </article>
  );
}
