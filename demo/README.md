# Kestrel Demo

这个目录是一个最小可运行的演示项目:
- 后端: FastAPI (`demo/backend`)
- 前端: React + Vite (`demo/frontend`)

功能:
- 左侧列出 `C:\Zoo\Kestrel\video` 下的视频
- 中间播放选择的视频
- 点击按钮调用本地 LLM 进行识别并展示结果
- 演示固定使用模型: `qwen3-vl-8b-instruct-awq-4bit`
- 后端内部已复刻 `config.toml` 与 `prompt.md`，不依赖 `src` 目录运行

## 1. 启动后端

```bash
cd demo/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

后端接口:
- `GET /api/health`
- `GET /api/videos`
- `POST /api/recognize`
- `GET /videos/<video_name>`

## 2. 启动前端

```bash
cd demo/frontend
cmd /c npm install
cmd /c npm run dev
```

默认前端地址: `http://localhost:5173`

前端会通过 Vite 代理访问后端 `http://127.0.0.1:8080`。
如果你希望修改地址，可以在前端环境变量中设置 `VITE_API_BASE`。


