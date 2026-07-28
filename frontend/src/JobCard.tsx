import { BatchJob } from "./api";

interface Props {
  job: BatchJob;
  onCancel: () => void;
  onDownload: (videoId: string) => void;
}

export function JobCard({ job, onCancel, onDownload }: Props) {
  const running = job.status === "running" || job.status === "queued";

  return (
    <article className="panel job">
      <div className="job-head">
        <div className="meta">
          <strong>Job {job.job_id}</strong>
          <span>
            {new Date(job.created_at).toLocaleString()} · {job.videos.length} video(s) ·{" "}
            {job.junction_type} ({job.expected_movements} ways)
          </span>
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
        <i style={{ width: `${Math.round((job.progress || 0) * 100)}%` }} />
      </div>
      <div className="counts" style={{ marginTop: "-0.35rem" }}>
        <span>
          Batch <b>{Math.round((job.progress || 0) * 100)}%</b>
        </span>
      </div>

      <div className="video-grid">
        {job.videos.map((video) => {
          const canDownload = Boolean(video.report_url);
          const movements = Object.entries(video.movement_counts || video.lane_counts || {});
          const classes = Object.entries(video.class_counts || {});
          const pct = Math.round((video.video_progress || 0) * 100);
          return (
            <div className="video-row" key={video.video_id}>
              <div className="video-row-top">
                <h3>{video.filename}</h3>
                <span className={`badge ${video.status}`}>{video.status}</span>
              </div>

              <div className="progress" aria-label="Video progress">
                <i style={{ width: `${pct}%` }} />
              </div>

              <div className="counts">
                <span>
                  Frames <b>{video.frames_done || 0}</b>
                  {video.frames_total ? <> / {video.frames_total}</> : null}
                </span>
                <span>
                  Progress <b>{pct}%</b>
                </span>
                <span>
                  Vehicles <b>{video.total_events}</b>
                </span>
                {video.fps ? (
                  <span>
                    FPS <b>{video.fps.toFixed(1)}</b>
                  </span>
                ) : null}
              </div>
              {video.message ? (
                <div className="counts">
                  <span>{video.message}</span>
                </div>
              ) : null}
              {video.error ? <span style={{ color: "var(--danger)" }}>{video.error}</span> : null}

              {(movements.length > 0 || classes.length > 0) && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
                  {movements.length > 0 ? (
                    <table className="mini-table">
                      <thead>
                        <tr>
                          <th>Movement (way)</th>
                          <th>Count</th>
                        </tr>
                      </thead>
                      <tbody>
                        {movements.map(([k, v]) => (
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
                          <th>Vehicle class</th>
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
              {canDownload ? (
                <div>
                  <button
                    className="btn btn-primary"
                    type="button"
                    onClick={() => onDownload(video.video_id)}
                  >
                    Download Excel
                  </button>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </article>
  );
}
