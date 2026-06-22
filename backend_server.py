from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime, timezone
import httpx
import base64
import json
import uuid
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ZAI_API_KEY = os.getenv("ZAI_API_KEY")
ZAI_BASE_URL = os.getenv("ZAI_BASE_URL", "https://api.z.ai/v1")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "dev-token-change-me")

security = HTTPBearer(auto_error=False)

# In-memory score storage (resets on restart — swap for a DB later)
score_store: list[dict] = []


# MARK: - Auth

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None or credentials.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.credentials


# MARK: - Models

class PASIScore(BaseModel):
    erythema: int
    induration: int
    desquamation: int
    area_pct: float
    pasi_total: float


class AnalysisResult(BaseModel):
    id: str
    photo_url: str
    body_region: str
    score: PASIScore
    analyzed_at: str  # ISO8601


class AnalyzePhotoResponse(BaseModel):
    success: bool
    result: AnalysisResult | None = None
    error: str | None = None


class ScoresListResponse(BaseModel):
    scores: list[AnalysisResult]
    count: int


# MARK: - Routes

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/analyze-photo", response_model=AnalyzePhotoResponse)
async def analyze_photo(
    file: UploadFile = File(...),
    body_region: str | None = Form(None),
    _token: str = Depends(verify_token),
):
    if not ZAI_API_KEY:
        raise HTTPException(status_code=500, detail="ZAI_API_KEY not configured")

    image_data = await file.read()
    base64_image = base64.b64encode(image_data).decode("utf-8")

    prompt = """Analyze this image for psoriasis severity using the PASI (Psoriasis Area and Severity Index) scoring system.

Please evaluate:
1. Erythema (redness) - score 0-4
2. Induration (thickness/elevation) - score 0-4
3. Desquamation (scaling) - score 0-4
4. Affected area percentage - estimate percentage of visible skin affected
5. Body region - identify which body part is shown (e.g., elbow, knee, scalp, torso, hand)

Respond ONLY with valid JSON in this exact format:
{
    "erythema": <number 0-4>,
    "induration": <number 0-4>,
    "desquamation": <number 0-4>,
    "area_pct": <number 0-100>,
    "body_region": "<region name>"
}"""

    headers = {
        "x-api-key": ZAI_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 500,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ZAI_BASE_URL}/messages",
                headers=headers,
                json=payload,
            )
    except httpx.RequestError as e:
        return AnalyzePhotoResponse(success=False, error=f"Network error: {e}")

    if response.status_code != 200:
        return AnalyzePhotoResponse(success=False, error=f"AI API error {response.status_code}: {response.text}")

    try:
        data = response.json()
        content_text = data["content"][0]["text"]
        # Strip markdown code fences if present
        if "```" in content_text:
            content_text = content_text.split("```")[1]
            if content_text.startswith("json"):
                content_text = content_text[4:]
        analysis = json.loads(content_text.strip())
    except Exception as e:
        return AnalyzePhotoResponse(success=False, error=f"Failed to parse AI response: {e}")

    erythema = int(analysis.get("erythema", 0))
    induration = int(analysis.get("induration", 0))
    desquamation = int(analysis.get("desquamation", 0))
    area_pct = float(analysis.get("area_pct", 0))
    detected_region = analysis.get("body_region", "unknown")
    final_region = body_region or detected_region

    pasi_total = (erythema + induration + desquamation) * area_pct / 100.0

    result = AnalysisResult(
        id=str(uuid.uuid4()),
        photo_url=f"server://{uuid.uuid4()}",
        body_region=final_region,
        score=PASIScore(
            erythema=erythema,
            induration=induration,
            desquamation=desquamation,
            area_pct=area_pct,
            pasi_total=pasi_total,
        ),
        analyzed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    score_store.append(result.model_dump())

    return AnalyzePhotoResponse(success=True, result=result)


@app.get("/api/v1/scores", response_model=ScoresListResponse)
def get_scores(
    limit: int = 30,
    _token: str = Depends(verify_token),
):
    recent = score_store[-limit:][::-1]  # Most recent first
    results = [AnalysisResult(**s) for s in recent]
    return ScoresListResponse(scores=results, count=len(results))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend_server:app", host="0.0.0.0", port=port)
