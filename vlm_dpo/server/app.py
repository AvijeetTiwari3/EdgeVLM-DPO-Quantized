"""
vlm_dpo/server/app.py
=====================
FastAPI Application Entry Point for On-Device Vision-Language Model Service.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer

from vlm_dpo.models.vlm_backbone import EdgeVLM
from vlm_dpo.server.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Initializing EdgeVLM On-Device Microservice...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = EdgeVLM(
        vocab_size=len(tokenizer),
        visual_dim=64,
        vision_hidden_dim=128,
        llm_hidden_dim=256,
        num_layers=3,
        num_heads=4,
        num_visual_patches=16
    )
    model.eval()

    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.device = "cpu"
    print("[OK] EdgeVLM Inference Service Ready.")
    yield
    print("[*] Shutting down EdgeVLM Service...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="EdgeVLM-DPO: On-Device Multimodal AI Service",
        description="Low-latency on-device Vision-Language inference service aligned with Direct Preference Optimization and INT4 Edge Quantization.",
        version="1.0.0",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("vlm_dpo.server.app:app", host="0.0.0.0", port=8002, reload=False)
