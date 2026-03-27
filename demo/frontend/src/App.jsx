import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export default function App() {
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState("");
  const [loadingVideos, setLoadingVideos] = useState(true);
  const [loadingResult, setLoadingResult] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const selectedVideoUrl = useMemo(() => {
    if (!selectedVideo) {
      return "";
    }
    return `${API_BASE}/videos/${encodeURIComponent(selectedVideo)}`;
  }, [selectedVideo]);

  useEffect(() => {
    const loadVideos = async () => {
      setLoadingVideos(true);
      setError("");
      try {
        const resp = await fetch(`${API_BASE}/api/videos`);
        if (!resp.ok) {
          throw new Error(`加载视频失败: ${resp.status}`);
        }
        const data = await resp.json();
        const names = (data.videos || []).map((v) => v.name);
        setVideos(names);
        if (names.length > 0) {
          setSelectedVideo(names[0]);
        }
      } catch (err) {
        setError(err.message || "加载视频失败");
      } finally {
        setLoadingVideos(false);
      }
    };

    loadVideos();
  }, []);

  const handleSelectVideo = (name) => {
    setSelectedVideo(name);
    setResult(null);
    setError("");
  };

  const handleRecognize = async () => {
    if (!selectedVideo) {
      return;
    }

    setLoadingResult(true);
    setResult(null);
    setError("");

    try {
      const resp = await fetch(`${API_BASE}/api/recognize`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ video_name: selectedVideo })
      });

      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || "识别失败");
      }
      setResult(data);
    } catch (err) {
      setError(err.message || "识别失败");
    } finally {
      setLoadingResult(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div>
          <h1>Kestrel 药瓶识别演示</h1>
        </div>
      </header>

      <div className="content-grid">
        <aside className="panel sidebar">
          <div className="panel-title-row">
            <h2>视频列表</h2>
            <span>{videos.length}</span>
          </div>
          {loadingVideos ? <p className="muted">正在加载视频...</p> : null}
          <div className="video-list">
            {videos.map((name) => (
              <button
                key={name}
                className={name === selectedVideo ? "video-item active" : "video-item"}
                onClick={() => handleSelectVideo(name)}
                type="button"
              >
                {name}
              </button>
            ))}
          </div>
        </aside>

        <section className="panel viewer">
          <div className="panel-title-row">
            <h2>视频预览</h2>
            <button
              className="primary-btn"
              type="button"
              disabled={!selectedVideo || loadingResult}
              onClick={handleRecognize}
            >
              {loadingResult ? "识别中..." : "开始识别"}
            </button>
          </div>

          {selectedVideo ? (
            <video key={selectedVideoUrl} src={selectedVideoUrl} controls className="video-player" />
          ) : (
            <p className="muted">请选择左侧视频</p>
          )}
        </section>

        <aside className="panel result">
          <h2>识别结果</h2>
          {error ? <p className="error">{error}</p> : null}

          {result ? (
            <>
              <div className="meta">
                <p><strong>视频:</strong> {result.video_name}</p>
                <p><strong>耗时:</strong> {result.elapsed}s</p>
                <p><strong>药品名称:</strong> {result.drug_name || "未提取到"}</p>
              </div>
              <pre className="raw-output">{result.raw_result}</pre>
            </>
          ) : (
            <p className="muted">点击“开始识别”后显示结果。</p>
          )}
        </aside>
      </div>
    </div>
  );
}
