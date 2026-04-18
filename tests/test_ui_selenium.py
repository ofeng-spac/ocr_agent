from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


ROOT = Path(__file__).resolve().parents[1]
BACKEND_URL = "http://127.0.0.1:8090"
FRONTEND_URL = "http://127.0.0.1:4173"
FIREFOX_BINARY = "/snap/firefox/current/usr/lib/firefox/firefox"


def wait_http(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):  # noqa: S310
                return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"timeout waiting for {url}")


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


@pytest.fixture(scope="session")
def live_servers():
    backend_proc = subprocess.Popen(
        [
            "micromamba",
            "run",
            "-n",
            "ocr",
            "uvicorn",
            "app.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8090",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ},
    )

    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "4173"],
        cwd=ROOT / "demo" / "frontend",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "VITE_API_BASE": BACKEND_URL},
    )

    try:
        wait_http(f"{BACKEND_URL}/api/health", timeout=40)
        wait_http(FRONTEND_URL, timeout=40)
        yield
    finally:
        frontend_proc.terminate()
        backend_proc.terminate()
        frontend_proc.wait(timeout=10)
        backend_proc.wait(timeout=10)


@pytest.fixture
def browser(live_servers):
    options = Options()
    if os.getenv("HEADLESS", "1") == "1":
        options.add_argument("--headless")
    options.binary_location = FIREFOX_BINARY
    driver = webdriver.Firefox(options=options)
    try:
        yield driver
    finally:
        driver.quit()


@pytest.mark.e2e
def test_demo_ui_recognize_verify_and_ask(browser):
    wait = WebDriverWait(browser, 40)
    browser.get(FRONTEND_URL)

    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='recognize-btn']")))

    browser.find_element(By.CSS_SELECTOR, "[data-testid='recognize-btn']").click()

    canonical_name = wait.until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "[data-testid='canonical-name-value']"))
    )
    wait.until(lambda d: canonical_name.text.strip() != "")
    assert canonical_name.text == "注射用头孢噻呋钠"

    expected_input = browser.find_element(By.CSS_SELECTOR, "[data-testid='expected-drug-input']")
    expected_input.clear()
    expected_input.send_keys("注射用头孢噻呋钠")
    browser.find_element(By.CSS_SELECTOR, "[data-testid='verify-btn']").click()

    verify_status = wait.until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "[data-testid='verify-status-value']"))
    )
    wait.until(lambda d: verify_status.text.strip() != "")
    assert verify_status.text == "verified_exact"

    question_input = browser.find_element(By.CSS_SELECTOR, "[data-testid='qa-question-input']")
    question_input.clear()
    question_input.send_keys("这个药一般治疗哪些感染")
    browser.find_element(By.CSS_SELECTOR, "[data-testid='qa-ask-btn']").click()

    answer = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "[data-testid='qa-answer']")))
    wait.until(lambda d: answer.text.strip() != "")
    assert "呼吸道" in answer.text
