"""LLM classifier for the educational Cleveland Heart Disease practical."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
LABELS = {0: "No Heart Disease", 1: "Heart Disease"}
CATEGORICAL = {
    "sex": {0, 1}, "cp": {1, 2, 3, 4}, "fbs": {0, 1},
    "restecg": {0, 1, 2}, "exang": {0, 1}, "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3}, "thal": {3, 6, 7},
}
RANGES = {
    "age": (1, 120), "trestbps": (1, 400), "chol": (1, 1000),
    "thalach": (1, 300), "oldpeak": (0, 20),
}
CODEBOOK = """
Dataset codebook: sex 0=female, 1=male; cp 1=typical angina, 2= atypical angina,
3=non-anginal pain, 4=asymptomatic; fbs 0=false (<=120 mg/dl), 1=true;
restecg 0=normal, 1=ST-T wave abnormality, 2=probable/definite LV hypertrophy;
exang 0=no, 1=yes; slope 1=upsloping, 2=flat, 3=downsloping;
ca=number of major vessels (0-3); thal 3=normal, 6=fixed defect, 7=reversible defect.
""".strip()

SYSTEM_PROMPT = f"""You are a structured binary classification component for an educational
machine-learning practical using the Cleveland Heart Disease dataset.

Classify the supplied feature record as 0 = No Heart Disease or 1 = Heart Disease.
Use only supplied features. Do not invent or fill missing values. Do not diagnose
or provide medical advice. Interpret coded categorical fields using this codebook:
{CODEBOOK}
Return valid JSON only, with keys status, prediction, label, reason. For valid input
status must be "ok", prediction integer 0 or 1, label the exact corresponding label,
and reason 1-2 clear sentences in Indonesian (maximum 25 words). Explain the dataset
classification by citing up to two supplied feature values, interpreting category
codes with the codebook. Say "data/record dikelompokkan" rather than diagnosing a
person. Do not invent facts, certainty, or medical advice. For invalid input return
status "invalid_input", prediction null, label null, and identify the field in reason."""

MAX_API_RETRIES = 3
MAX_COMPLETION_TOKENS = 128


class ClassifierError(RuntimeError):
    """API, response parsing, or output validation failed."""

    def __init__(self, message: str, retries: int = 0):
        super().__init__(message)
        self.retries = retries


def validate_patient(patient_data: dict[str, Any]) -> dict[str, float]:
    missing = [name for name in FEATURES if name not in patient_data]
    extra = [name for name in patient_data if name not in FEATURES]
    if missing or extra:
        raise ValueError(f"Kolom tidak sesuai. Hilang: {missing}; tidak dikenal: {extra}")
    result: dict[str, float] = {}
    for name in FEATURES:
        try:
            value = float(patient_data[name])
        except (TypeError, ValueError):
            raise ValueError(f"{name} harus berupa angka") from None
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{name} tidak boleh kosong atau non-finite")
        if name in CATEGORICAL and value not in CATEGORICAL[name]:
            raise ValueError(f"{name} harus salah satu dari {sorted(CATEGORICAL[name])}")
        if name in RANGES and not (RANGES[name][0] <= value <= RANGES[name][1]):
            raise ValueError(f"{name} di luar rentang {RANGES[name]}")
        result[name] = value
    return result


def _settings() -> tuple[str, str, str]:
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not key:
        raise ClassifierError("API key belum diisi pada .env atau Streamlit Secrets")
    if provider == "groq":
        return "https://api.groq.com/openai/v1", key, model or "llama-3.3-70b-versatile"
    if provider == "gemini":
        return "https://generativelanguage.googleapis.com/v1beta/openai/", key, model or "gemini-2.0-flash"
    raise ClassifierError("LLM_PROVIDER harus groq atau gemini")


def _retry_delay_seconds(exc: RateLimitError, attempt: int) -> float:
    """Use provider reset headers/message when present; otherwise back off."""
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    candidates = [headers.get("retry-after"), headers.get("x-ratelimit-reset-tokens")]
    match = re.search(r"try again in\s+([\d.]+)\s*(ms|s|m)?", str(exc), re.I)
    if match:
        candidates.append(match.group(1) + (match.group(2) or "s"))
    for value in candidates:
        if value is None:
            continue
        try:
            match = re.fullmatch(r"\s*([\d.]+)\s*(ms|s|m)?\s*", str(value), re.I)
            if not match:
                seconds = float(value)
            else:
                amount, unit = float(match.group(1)), (match.group(2) or "s").lower()
                seconds = amount / 1000 if unit == "ms" else amount * 60 if unit == "m" else amount
            return min(max(seconds, 0.5), 60)
        except (TypeError, ValueError):
            continue
    return min(2 ** attempt, 60)


def classify_heart_disease(patient_data: dict[str, Any]) -> dict[str, Any]:
    """Validate 13 features, call configured API, and validate JSON result."""
    patient = validate_patient(patient_data)
    base_url, key, model = _settings()
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    client = OpenAI(api_key=key, base_url=base_url, timeout=45, max_retries=0)
    request_options: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"patient_features": patient})},
        ],
    }
    # Groq recommends max_completion_tokens. Gemini's OpenAI-compatible API
    # accepts max_tokens. A short JSON response needs far fewer than 128 tokens.
    request_options["max_completion_tokens" if provider == "groq" else "max_tokens"] = MAX_COMPLETION_TOKENS
    if provider == "groq" and "qwen3.8-27b" in model.lower():
        request_options["reasoning_effort"] = "none"

    response = None
    retry_count = 0
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            response = client.chat.completions.create(**request_options)
            break
        except RateLimitError as exc:
            message = str(exc).lower()
            # This is a permanent per-request output cap error, not a transient
            # rate limit; retrying it unchanged would only waste requests.
            too_large = "request too large" in message or "expected output tokens exceed" in message
            if too_large or attempt >= MAX_API_RETRIES:
                raise ClassifierError(f"Permintaan API gagal: {exc}", retries=retry_count) from exc
            retry_count += 1
            time.sleep(_retry_delay_seconds(exc, attempt))
        except Exception as exc:
            raise ClassifierError(f"Permintaan API gagal: {exc}") from exc
    if response is None:
        raise ClassifierError("Permintaan API gagal setelah retry rate limit")
    content = response.choices[0].message.content or ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Allow providers that ignore JSON mode but return a fenced JSON object.
        match = re.search(r"\{.*\}", content, re.S)
        try:
            parsed = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            parsed = None
    if not isinstance(parsed, dict):
        raise ClassifierError("Output JSON tidak valid")
    if parsed.get("status") != "ok" or parsed.get("prediction") not in (0, 1):
        raise ClassifierError(f"Output classifier tidak valid: {parsed}")
    prediction = int(parsed["prediction"])
    if parsed.get("label") != LABELS[prediction] or not isinstance(parsed.get("reason"), str):
        raise ClassifierError("Label atau reason pada output API tidak valid")
    return {"status": "ok", "prediction": prediction, "label": LABELS[prediction],
            "reason": parsed["reason"], "provider": os.getenv("LLM_PROVIDER", "groq"),
            "model": model, "temperature": 0, "api_retries": retry_count}
