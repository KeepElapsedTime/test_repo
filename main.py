from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any, AsyncGenerator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
from datetime import datetime
import json
import asyncio

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic 模型保持不變
class Message(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = True
    options: Optional[Dict[str, Any]] = None
    template: Optional[str] = None
    format: Optional[str] = None

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: Optional[bool] = True
    raw: Optional[bool] = False
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

# 模型配置
MODEL_CONFIGS = {
    "llama-3-sherkala-8b": "inceptionai/Llama-3.1-Sherkala-8B-Chat",
    "llama-3-nanda-10b": "MBZUAI/Llama-3-Nanda-10B-Chat"
}

# 全局變量
loaded_models = {}

def get_current_datetime():
    return datetime.utcnow().isoformat() + "Z"


def load_model(model_name: str):
    """加載模型和tokenizer，使用更快的設置"""
    if model_name not in loaded_models:
        model_path = MODEL_CONFIGS[model_name]
        start_time = time.time()
        
        # 使用快速加載選項
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            legacy=False,  # 使用更快的tokenizer實現
            use_fast=True  # 使用快速tokenizer
        )
        
        # 使用更簡單的chat template
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% elif message['role'] == 'system' %}System: {{ message['content'] }}{% endif %}\n{% endfor %}\nAssistant:"
        
        # 移除 Flash Attention 2 設置
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # 設置為評估模式以提升推理速度
        model.eval()
        
        load_duration = int((time.time() - start_time) * 1e9)
        
        loaded_models[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "load_duration": load_duration
        }
    
    return loaded_models[model_name]

def process_chunks(text: str) -> List[str]:
    """更快的分塊策略"""
    # 對於較短的文本，直接作為一個塊返回
    if len(text) < 50:
        return [text]
    
    # 使用更大的塊大小，減少流式傳輸的次數
    chunk_size = 10  # 每個塊10個詞
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk + " ")
    
    return chunks

async def stream_generate(
    text_chunks: List[str],
    model_name: str,
    stats: Dict[str, Any],
    is_chat: bool = False
) -> AsyncGenerator[str, None]:
    """優化的串流生成器"""
    for i, chunk in enumerate(text_chunks):
        is_last = i == len(text_chunks) - 1
        
        if is_chat:
            response_data = {
                "model": model_name,
                "created_at": get_current_datetime(),
                "message": {
                    "role": "assistant",
                    "content": chunk
                },
                "done": is_last
            }
        else:
            response_data = {
                "model": model_name,
                "created_at": get_current_datetime(),
                "response": chunk,
                "done": is_last,
                "context": [1, 2, 3]
            }
        
        if is_last:
            response_data.update(stats)
        
        yield json.dumps(response_data) + "\n"
        
        # 減少延遲時間
        await asyncio.sleep(0.01)  # 從0.05減少到0.01

async def generate_text(
    model_name: str,
    prompt: str,
    system: Optional[str] = None,
    input_messages: Optional[List[Message]] = None
) -> tuple[str, Dict[str, Any]]:
    """優化的文本生成"""
    start_time = time.time()
    
    model_data = load_model(model_name)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    load_duration = model_data["load_duration"]
    
    # 準備輸入
    encoding_start = time.time()
    if input_messages:
        conversation = [{"role": msg.role, "content": msg.content} 
                       for msg in input_messages]
        input_ids = tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
    else:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            return_attention_mask=True
        ).to(model.device)
        input_ids = inputs.input_ids
    
    prompt_eval_duration = int((time.time() - encoding_start) * 1e9)
    prompt_eval_count = input_ids.shape[1]
    
    # 優化的生成配置
    generation_config = {
        "max_new_tokens": 128,  # 減少最大token數
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,  # 使用KV緩存
    }
    
    # 生成文本
    generation_start = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():  # 使用自動混合精度
        outputs = model.generate(
            input_ids,
            **generation_config
        )
    
    # 解碼輸出
    start_idx = input_ids.shape[1]
    response = tokenizer.decode(outputs[0][start_idx:], skip_special_tokens=True)
    
    # 計算統計信息
    eval_duration = int((time.time() - generation_start) * 1e9)
    eval_count = outputs.shape[1] - start_idx
    total_duration = int((time.time() - start_time) * 1e9)
    
    stats = {
        "total_duration": total_duration,
        "load_duration": load_duration,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_count": eval_count,
        "eval_duration": eval_duration
    }
    
    return response, stats

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """生成端點"""
    try:
        if request.model not in MODEL_CONFIGS:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model} not found. Available models: {list(MODEL_CONFIGS.keys())}"
            )
        
        # 生成文本
        response, stats = await generate_text(
            model_name=request.model,
            prompt=request.prompt,
            system=request.system
        )
        
        # 根據 stream 參數決定返回方式
        if request.stream:
            chunks = process_chunks(response)
            return StreamingResponse(
                stream_generate(
                    text_chunks=chunks,
                    model_name=request.model,
                    stats=stats
                ),
                media_type="text/event-stream"
            )
        else:
            return JSONResponse({
                "model": request.model,
                "created_at": get_current_datetime(),
                "response": response,
                "done": True,
                "context": [1, 2, 3],
                **stats
            })
            
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """聊天端點"""
    try:
        if request.model not in MODEL_CONFIGS:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model} not found. Available models: {list(MODEL_CONFIGS.keys())}"
            )
        
        # 生成回應
        response, stats = await generate_text(
            model_name=request.model,
            prompt="",
            input_messages=request.messages
        )
        
        # 根據 stream 參數決定返回方式
        if request.stream:
            chunks = process_chunks(response)
            return StreamingResponse(
                stream_generate(
                    text_chunks=chunks,
                    model_name=request.model,
                    stats=stats,
                    is_chat=True
                ),
                media_type="text/event-stream"
            )
        else:
            return JSONResponse({
                "model": request.model,
                "created_at": get_current_datetime(),
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "done": True,
                **stats
            })
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
