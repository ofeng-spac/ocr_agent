import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export default function App() {
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState("");
  const [loadingVideos, setLoadingVideos] = useState(true);
  const [loadingResult, setLoadingResult] = useState(false);
  const [loadingQa, setLoadingQa] = useState(false);
  const [result, setResult] = useState(null);
  const [expectedDrugName, setExpectedDrugName] = useState("");
  const [question, setQuestion] = useState("");
  const [qaResult, setQaResult] = useState(null);
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
    setQaResult(null);
    setQuestion("");
    setError("");
  };

  const postJson = async (path, body, fallbackMessage) => {
    const resp = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(body)
    });

    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || fallbackMessage);
    }
    return data;
  };

  const handleRecognize = async () => {
    if (!selectedVideo) return;

    setLoadingResult(true);
    setQaResult(null);
    setResult(null);
    setError("");

    try {
      const data = await postJson(
        "/api/recognize",
        { video_name: selectedVideo },
        "识别失败"
      );
      setResult(data);
    } catch (err) {
      setError(err.message || "识别失败");
    } finally {
      setLoadingResult(false);
    }
  };

  const handleVerify = async () => {
    if (!selectedVideo || !expectedDrugName.trim()) {
      return;
    }

    setLoadingResult(true);
    setQaResult(null);
    setResult(null);
    setError("");

    try {
      const data = await postJson(
        "/api/verify",
        {
          video_name: selectedVideo,
          expected_drug_name: expectedDrugName.trim()
        },
        "核验失败"
      );
      setResult(data);
    } catch (err) {
      setError(err.message || "核验失败");
    } finally {
      setLoadingResult(false);
    }
  };

  const handleAsk = async () => {
    if (!result?.canonical_name || !question.trim()) {
      return;
    }

    setLoadingQa(true);
    setQaResult(null);
    setError("");

    try {
      const data = await postJson(
        "/api/rag/ask",
        {
          canonical_name: result.canonical_name,
          question: question.trim()
        },
        "问答失败"
      );
      setQaResult(data);
    } catch (err) {
      setError(err.message || "问答失败");
    } finally {
      setLoadingQa(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div>
          <h1>Kestrel 药瓶识别演示</h1>
          <p>视频识别、标准名校验、说明书问答</p>
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
          </div>

          <div className="action-bar">
            <input
              className="text-input"
              type="text"
              value={expectedDrugName}
              onChange={(e) => setExpectedDrugName(e.target.value)}
              placeholder="输入期望药名，例如：注射用头孢噻呋钠"
            />
            <div className="action-buttons">
              <button
                className="primary-btn"
                type="button"
                disabled={!selectedVideo || loadingResult}
                onClick={handleRecognize}
              >
                {loadingResult ? "处理中..." : "开始识别"}
              </button>
              <button
                className="secondary-btn"
                type="button"
                disabled={!selectedVideo || !expectedDrugName.trim() || loadingResult}
                onClick={handleVerify}
              >
                {loadingResult ? "处理中..." : "开始核验"}
              </button>
            </div>
          </div>

          {selectedVideo ? (
            <video key={selectedVideoUrl} src={selectedVideoUrl} controls className="video-player" />
          ) : (
            <p className="muted">请选择左侧视频</p>
          )}
        </section>

        <aside className="panel result">
          <h2>结构化结果</h2>
          {error ? <p className="error">{error}</p> : null}

          {result ? (
            <>
              <div className="result-card">
                <div className="meta-grid">
                  <p><strong>视频</strong><span>{result.video_name}</span></p>
                  <p><strong>耗时</strong><span>{result.elapsed}s</span></p>
                  <p><strong>原始名称</strong><span>{result.raw_name || "未提取到"}</span></p>
                  <p><strong>标准名称</strong><span>{result.canonical_name || "未确认"}</span></p>
                  <p><strong>校验状态</strong><span>{result.verify_status}</span></p>
                  <p><strong>匹配类型</strong><span>{result.verify_match_type}</span></p>
                  <p><strong>不确定性</strong><span>{result.uncertainty_level || "-"}</span></p>
                  <p><strong>候选名称</strong><span>{result.candidate_name || "-"}</span></p>
                </div>
                <div className="note-block">
                  <strong>校验说明</strong>
                  <p>{result.verify_reason}</p>
                </div>
                {result.expected_check ? (
                  <div className="note-block verify-block">
                    <strong>核验结果</strong>
                    <p>{result.expected_check.status}</p>
                    <p>{result.expected_check.reason}</p>
                  </div>
                ) : null}
                <div className="note-block">
                  <strong>证据文本</strong>
                  <p>{result.evidence_text || "无"}</p>
                </div>
                <details className="details-block">
                  <summary>查看原始模型输出</summary>
                  <pre className="raw-output">{result.result}</pre>
                </details>
              </div>

              <div className="qa-block">
                <div className="panel-title-row">
                  <h2>说明书问答</h2>
                </div>
                <div className="qa-controls">
                  <input
                    className="text-input"
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="例如：这个药的适应症是什么"
                  />
                  <button
                    className="secondary-btn"
                    type="button"
                    disabled={!result.canonical_name || !question.trim() || loadingQa}
                    onClick={handleAsk}
                  >
                    {loadingQa ? "问答中..." : "开始问答"}
                  </button>
                </div>

                {qaResult ? (
                  <div className="qa-result">
                    <p><strong>字段</strong> {qaResult.target_field}</p>
                    <p><strong>状态</strong> {qaResult.status}</p>
                    <p><strong>说明</strong> {qaResult.reason}</p>
                    {qaResult.answer ? (
                      <div className="note-block">
                        <strong>回答</strong>
                        <p>{qaResult.answer}</p>
                      </div>
                    ) : null}
                    {qaResult.citations?.length ? (
                      <div className="citations">
                        <strong>引用</strong>
                        {qaResult.citations.map((item, idx) => (
                          <div key={`${item.source_file}-${idx}`} className="citation-item">
                            <p><strong>{item.field_name}</strong></p>
                            <p>{item.field_value}</p>
                            <p className="muted">{item.source_file}</p>
                          </div>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <p className="muted">当前支持规格、适应症、用法用量、注意事项、禁忌等字段问答。</p>
                )}
              </div>
            </>
          ) : (
            <p className="muted">先点击“开始识别”或“开始核验”，再查看结构化结果和说明书问答。</p>
          )}
        </aside>
      </div>
    </div>
  );
}
