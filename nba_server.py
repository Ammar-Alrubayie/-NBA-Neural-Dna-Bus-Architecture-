#!/usr/bin/env python3
"""
NBA API Server
FastAPI wrapper for Neural Bus Architecture

Run:
  py -3.10 nba_server.py

Then open: http://localhost:8000/docs

Endpoints:
  POST /generate        - text prompt
  POST /generate-image  - text + image
  GET  /health          - system status
"""

import os
import sys
import base64
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Import NBA system
from nba_system import NBASystem

app = FastAPI(
    title="NBA - Neural Bus Architecture",
    description="Multi-Agent AI Orchestration API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance
system = None


class PromptRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1024


class GenerateResponse(BaseModel):
    response: str
    strategy: str
    confidence: float
    gen_params: dict


@app.on_event("startup")
def load_system():
    global system
    print("Loading NBA System...")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def find(name):
        for p in [os.path.join(script_dir, name),
                  os.path.join("/workspace", name)]:
            if os.path.exists(p):
                return p
        return os.path.join(script_dir, name)

    system = NBASystem(device="cuda")
    system.load(
        coder_path=find("models/qwen-coder"),
        vl_path=find("models/qwen-vl"),
        dna_coder_path=find("dna_coder_trained.pth"),
        dna_vl_path=find("dna_vl_trained.pth"),
        router_path=find("router_admin_trained.pth"),
    )
    print("NBA System Ready!")


@app.get("/health")
def health():
    return {"status": "ready" if system else "loading"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: PromptRequest):
    result = system.generate(req.prompt, max_new_tokens=req.max_tokens)
    return GenerateResponse(
        response=result["response"],
        strategy=result["strategy"],
        confidence=result["confidence"],
        gen_params=result.get("gen_params", {}),
    )


@app.post("/generate-image", response_model=GenerateResponse)
async def generate_with_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    max_tokens: int = Form(1024),
):
    # Save uploaded image to temp file
    ext = os.path.splitext(image.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = system.generate(prompt, max_new_tokens=max_tokens,
                                 image_path=tmp_path)
        return GenerateResponse(
            response=result["response"],
            strategy=result["strategy"],
            confidence=result["confidence"],
            gen_params=result.get("gen_params", {}),
        )
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)