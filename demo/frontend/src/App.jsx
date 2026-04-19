import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

function WorkflowTimeline({ title, items }) {
  if (!items?.length) {
    return null;
  }

  return (
    <div className="workflow-block">
      <strong>{title}</strong>
      <div className="workflow-timeline">
        {items.map((item, idx) => (
          <div key={`${item.node}-${idx}`} className="timeline-item">
            <div className="timeline-marker" />
            <div className="timeline-card">
              <div className="timeline-head">
                <span className="timeline-node">{item.node}</span>
                <span className={`timeline-status status-${item.status}`}>{item.status}</span>
              </div>
              <p className="timeline-summary">{item.summary}</p>
              <div className="timeline-meta">
                {item.duration_ms !== undefined ? <span>耗时 {item.duration_ms} ms</span> : null}
                {item.started_at ? <span>开始 {item.started_at}</span> : null}
                {item.finished_at ? <span>结束 {item.finished_at}</span> : null}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState("");
  const [loadingVideos, setLoadingVideos] = useState(true);
  const [loadingResult, setLoadingResult] = useState(false);
  const [loadingQa, setLoadingQa] = useState(false);
  const [loadingAudit, setLoadingAudit] = useState(false);
  const [loadingEval, setLoadingEval] = useState(false);
  const [result, setResult] = useState(null);
  const [expectedDrugName, setExpectedDrugName] = useState("");
  const [question, setQuestion] = useState("");
  const [qaResult, setQaResult] = useState(null);
  const [auditLogs, setAuditLogs] = useState([]);
  const [evalSummary, setEvalSummary] = useState(null);
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

  useEffect(() => {
    const loadMeta = async () => {
      await Promise.all([fetchAuditLogs(), fetchEvalSummary()]);
    };
    loadMeta();
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

  const getJson = async (path, fallbackMessage) => {
    const resp = await fetch(`${API_BASE}${path}`);
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || fallbackMessage);
    }
    return data;
  };

  const fetchAuditLogs = async () => {
    setLoadingAudit(true);
    try {
      const data = await getJson("/api/audit_logs?limit=6", "加载日志失败");
      setAuditLogs(data.logs || []);
    } catch (err) {
      setError((prev) => prev || err.message || "加载日志失败");
    } finally {
      setLoadingAudit(false);
    }
  };

  const fetchEvalSummary = async () => {
    setLoadingEval(true);
    try {
      const data = await getJson("/api/eval/summary", "加载评测摘要失败");
      setEvalSummary(data);
    } catch (err) {
      setError((prev) => prev || err.message || "加载评测摘要失败");
    } finally {
      setLoadingEval(false);
    }
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
      fetchAuditLogs();
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
      fetchAuditLogs();
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
      fetchAuditLogs();
    } catch (err) {
      setError(err.message || "问答失败");
    } finally {
      setLoadingQa(false);
    }
  };

  const defaultModelName = "Qwen3-VL-8B-Instruct-AWQ-4bit";
  const defaultModelBest = evalSummary?.best_by_model?.[defaultModelName];

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
                data-testid={`video-item-${name}`}
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
              data-testid="expected-drug-input"
            />
            <div className="action-buttons">
              <button
                className="primary-btn"
                type="button"
                disabled={!selectedVideo || loadingResult}
                onClick={handleRecognize}
                data-testid="recognize-btn"
              >
                {loadingResult ? "处理中..." : "开始识别"}
              </button>
              <button
                className="secondary-btn"
                type="button"
                disabled={!selectedVideo || !expectedDrugName.trim() || loadingResult}
                onClick={handleVerify}
                data-testid="verify-btn"
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
              <div className="result-card" data-testid="result-card">
                <div className="meta-grid">
                  <p><strong>视频</strong><span data-testid="video-name-value">{result.video_name}</span></p>
                  <p><strong>耗时</strong><span data-testid="elapsed-value">{result.elapsed}s</span></p>
                  <p><strong>原始名称</strong><span data-testid="raw-name-value">{result.raw_name || "未提取到"}</span></p>
                  <p><strong>标准名称</strong><span data-testid="canonical-name-value">{result.canonical_name || "未确认"}</span></p>
                  <p><strong>校验状态</strong><span data-testid="verify-status-value">{result.verify_status}</span></p>
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
                <WorkflowTimeline title="执行轨迹" items={result.workflow_trace} />
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
                    data-testid="qa-question-input"
                  />
                  <button
                    className="secondary-btn"
                    type="button"
                    disabled={!result.canonical_name || !question.trim() || loadingQa}
                    onClick={handleAsk}
                    data-testid="qa-ask-btn"
                  >
                    {loadingQa ? "问答中..." : "开始问答"}
                  </button>
                </div>

                {qaResult ? (
                  <div className="qa-result" data-testid="qa-result">
                    <p><strong>字段</strong> {qaResult.target_field}</p>
                    <p><strong>状态</strong> {qaResult.status}</p>
                    <p><strong>说明</strong> {qaResult.reason}</p>
                    {qaResult.trace_id ? (
                      <p><strong>Trace</strong> {qaResult.trace_id}</p>
                    ) : null}
                    {qaResult.answer ? (
                      <div className="note-block">
                        <strong>回答</strong>
                        <p data-testid="qa-answer">{qaResult.answer}</p>
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
                    <WorkflowTimeline title="问答轨迹" items={qaResult.workflow_trace} />
                  </div>
                ) : (
                  <p className="muted">当前支持规格、适应症、用法用量、注意事项、禁忌等字段问答。</p>
                )}
              </div>
            </>
          ) : (
            <p className="muted">先点击“开始识别”或“开始核验”，再查看结构化结果和说明书问答。</p>
          )}

          <div className="summary-block">
            <div className="panel-title-row">
              <h2>评测摘要</h2>
            </div>
            {loadingEval ? (
              <p className="muted">正在加载评测摘要...</p>
            ) : evalSummary?.available ? (
              <>
                <div className="note-block compact-block">
                  <strong>总体推荐</strong>
                  <p>{evalSummary.recommended_model}</p>
                  <p>
                    配置：{evalSummary.recommended_config?.config}，正确率：
                    {evalSummary.recommended_config?.metrics_pct?.correct}% ，平均耗时：
                    {evalSummary.recommended_config?.avg_time_sec}s
                  </p>
                </div>
                {defaultModelBest ? (
                  <div className="note-block compact-block">
                    <strong>当前默认模型最佳配置</strong>
                    <p>{defaultModelName}</p>
                    <p>
                      配置：{defaultModelBest.config}，正确率：
                      {defaultModelBest.metrics_pct?.correct}% ，平均耗时：
                      {defaultModelBest.avg_time_sec}s
                    </p>
                  </div>
                ) : null}
              </>
            ) : (
              <p className="muted">暂无评测摘要。</p>
            )}
          </div>

          <div className="summary-block">
            <div className="panel-title-row">
              <h2>最近日志</h2>
            </div>
            {loadingAudit ? (
              <p className="muted">正在加载日志...</p>
            ) : auditLogs.length ? (
              <div className="audit-list">
                {auditLogs.map((log) => (
                  <div key={log.trace_id} className="audit-item">
                    <p><strong>{log.event_type}</strong></p>
                    <p className="muted">{log.created_at}</p>
                    <p className="trace-id">{log.trace_id}</p>
                    {log.payload?.canonical_name ? (
                      <p>标准名：{log.payload.canonical_name}</p>
                    ) : null}
                    {log.payload?.question ? (
                      <p>问题：{log.payload.question}</p>
                    ) : null}
                    {log.payload?.status ? (
                      <p>状态：{log.payload.status}</p>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted">暂无日志。</p>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
