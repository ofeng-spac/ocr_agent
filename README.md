# Kestrel

Pharmaceutical Bottle Label Recognition System Powered by Vision-Language Models.

```
kestrel/
├── src/
│   ├── server.py       # core inference (frame sampling, VLM call, prompt loading)
│   ├── prompt.md       # VLM system prompt (knowledge base + recognition guide sections)
│   ├── kb.yaml         # drug knowledge base
│   ├── config.toml     # model list, API credentials, video/image settings
│   ├── main.py         # single-video CLI
│   └── run.py          # batch experiment runner
├── video/
│   ├── video.csv       # dataset index: filename, ground_truth, condition
│   └── *.mp4
└── logs/               # experiment output logs
```