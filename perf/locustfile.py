from __future__ import annotations

import random

from locust import HttpUser, between, tag, task


FIELD_QUESTIONS = [
    {"canonical_name": "双黄连口服液", "question": "规格是什么"},
    {"canonical_name": "盐酸小檗碱片", "question": "禁忌是什么"},
    {"canonical_name": "奥美拉唑肠溶胶囊", "question": "用法用量是什么"},
    {"canonical_name": "贝伐珠单抗注射液", "question": "商品名是什么"},
]

SEMANTIC_QUESTIONS = [
    {"canonical_name": "双黄连口服液", "question": "这个药主要用于哪些场景"},
    {"canonical_name": "贝伐珠单抗注射液", "question": "这个药主要用于哪些场景"},
    {"canonical_name": "注射用头孢曲松钠", "question": "这个药一般治疗哪些感染"},
    {"canonical_name": "蒲地蓝消炎口服液", "question": "这个药主要用于哪些场景"},
]

VERIFY_CASES = [
    {
        "video_name": "video001.mp4",
        "expected_drug_name": "注射用头孢噻呋钠",
        "model": "qwen3-vl-8b-instruct-awq-4bit",
    },
    {
        "video_name": "video011.mp4",
        "expected_drug_name": "注射用头孢曲松钠",
        "model": "qwen3-vl-8b-instruct-awq-4bit",
    },
]


class DrugAgentUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @tag("light")
    @task(1)
    def health(self):
        self.client.get("/api/health", name="GET /api/health")

    @tag("light")
    @task(1)
    def eval_summary(self):
        self.client.get("/api/eval/summary", name="GET /api/eval/summary")

    @tag("light")
    @task(1)
    def audit_logs(self):
        self.client.get("/api/audit_logs?limit=10", name="GET /api/audit_logs")

    @tag("rag", "field")
    @task(4)
    def rag_field(self):
        payload = random.choice(FIELD_QUESTIONS)
        self.client.post("/api/rag/ask", json=payload, name="POST /api/rag/ask [field]")

    @tag("rag", "semantic")
    @task(4)
    def rag_semantic(self):
        payload = random.choice(SEMANTIC_QUESTIONS)
        self.client.post("/api/rag/ask", json=payload, name="POST /api/rag/ask [semantic]")

    @tag("verify")
    @task(5)
    def verify_exact(self):
        payload = random.choice(VERIFY_CASES)
        self.client.post("/api/verify", json=payload, name="POST /api/verify")
