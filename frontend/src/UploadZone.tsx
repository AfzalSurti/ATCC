import { useRef, useState } from "react";

interface Props {
  files: File[];
  onChange: (files: File[]) => void;
}

const ACCEPT = ".mp4,.avi,.mov,.mkv,.webm,.m4v,video/*";

export function UploadZone({ files, onChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [active, setActive] = useState(false);

  const mergeFiles = (incoming: FileList | File[]) => {
    const list = Array.from(incoming);
    const byKey = new Map<string, File>();
    for (const f of files) byKey.set(`${f.name}-${f.size}-${f.lastModified}`, f);
    for (const f of list) byKey.set(`${f.name}-${f.size}-${f.lastModified}`, f);
    onChange(Array.from(byKey.values()));
  };

  return (
    <div>
      <div
        className={`dropzone ${active ? "active" : ""}`}
        onDragEnter={(e) => {
          e.preventDefault();
          setActive(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setActive(true);
        }}
        onDragLeave={() => setActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setActive(false);
          if (e.dataTransfer.files?.length) mergeFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          multiple
          onChange={(e) => {
            if (e.target.files?.length) mergeFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <h2>Drop videos here</h2>
        <p>Single or multiple · MP4, MOV, MKV, WEBM, AVI</p>
      </div>

      {files.length > 0 ? (
        <div className="file-chip-row">
          {files.map((f) => (
            <span className="chip" key={`${f.name}-${f.size}-${f.lastModified}`}>
              {f.name}
              <button
                type="button"
                aria-label={`Remove ${f.name}`}
                onClick={(e) => {
                  e.stopPropagation();
                  onChange(files.filter((x) => x !== f));
                }}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}
