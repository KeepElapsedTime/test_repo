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

# Pydantic models (保持不變)
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

class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: Message
    done: bool = False
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool = False
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

# 模型配置
MODEL_CONFIGS = {
    "llama-3-sherkala-8b": "inceptionai/Llama-3.1-Sherkala-8B-Chat",
    "llama-3-nanda-10b": "MBZUAI/Llama-3-Nanda-10B-Chat"
}

loaded_models = {}

# Helper functions (保持不變)
def get_current_datetime():
    return datetime.utcnow().isoformat() + "Z"

def load_model(model_name: str):
    """加載模型和tokenizer"""
    if model_name not in loaded_models:
        model_path = MODEL_CONFIGS[model_name]
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role']+'<|end_header_id|>\n\n'+ "
            "message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}"
            "{% set content = bos_token + content %} "
            "{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        load_duration = int((time.time() - start_time) * 1e9)
        
        loaded_models[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "load_duration": load_duration
        }
    
    return loaded_models[model_name]

async def generate_stream(model_name: str, prompt: str, system: Optional[str] = None) -> AsyncGenerator[str, None]:
    """串流生成回應"""
    start_time = time.time()
    
    model_data = load_model(model_name)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    load_duration = model_data["load_duration"]
    
    if system:
        full_prompt = f"{system}\n\n{prompt}"
    else:
        full_prompt = prompt
    
    encoding_start = time.time()
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        return_attention_mask=True
    ).to(model.device)
    
    prompt_eval_duration = int((time.time() - encoding_start) * 1e9)
    prompt_eval_count = inputs.input_ids.shape[1]
    
    generation_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    response = tokenizer.decode(outputs.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    eval_duration = int((time.time() - generation_start) * 1e9)
    eval_count = outputs.sequences.shape[1] - inputs.input_ids.shape[1]
    
    total_duration = int((time.time() - start_time) * 1e9)
    
    # 串流輸出
    for i in range(0, len(response), 10):
        chunk = response[i:i+10]
        is_done = i + 10 >= len(response)
        
        response_data = {
            "model": model_name,
            "created_at": get_current_datetime(),
            "response": chunk,
            "done": is_done,
            "context": [1, 2, 3]
        }
        
        if is_done:
            response_data.update({
                "total_duration": total_duration,
                "load_duration": load_duration,
                "prompt_eval_count": prompt_eval_count,
                "prompt_eval_duration": prompt_eval_duration,
                "eval_count": eval_count,
                "eval_duration": eval_duration
            })
        
        yield json.dumps(response_data) + "\n"
        await asyncio.sleep(0.01)  # 小延遲以模擬真實串流

async def chat_stream(model_name: str, messages: List[Message]) -> AsyncGenerator[str, None]:
    """串流聊天回應"""
    start_time = time.time()
    
    model_data = load_model(model_name)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    load_duration = model_data["load_duration"]
    
    conversation = [{"role": msg.role, "content": msg.content} for msg in messages]
    
    encoding_start = time.time()
    input_ids = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    prompt_eval_duration = int((time.time() - encoding_start) * 1e9)
    prompt_eval_count = input_ids.shape[1]
    
    generation_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    response = tokenizer.decode(outputs.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    eval_duration = int((time.time() - generation_start) * 1e9)
    eval_count = outputs.sequences.shape[1] - input_ids.shape[1]
    
    total_duration = int((time.time() - start_time) * 1e9)
    
    # 串流輸出
    for i in range(0, len(response), 10):
        chunk = response[i:i+10]
        is_done = i + 10 >= len(response)
        
        response_data = {
            "model": model_name,
            "created_at": get_current_datetime(),
            "message": {
                "role": "assistant",
                "content": chunk
            },
            "done": is_done
        }
        
        if is_done:
            response_data.update({
                "total_duration": total_duration,
                "load_duration": load_duration,
                "prompt_eval_count": prompt_eval_count,
                "prompt_eval_duration": prompt_eval_duration,
                "eval_count": eval_count,
                "eval_duration": eval_duration
            })
        
        yield json.dumps(response_data) + "\n"
        await asyncio.sleep(0.01)  # 小延遲以模擬真實串流

def generate_response(model_name: str, prompt: str, system: Optional[str] = None) -> Dict:
    """生成單一回應"""
    start_time = time.time()
    
    model_data = load_model(model_name)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    load_duration = model_data["load_duration"]
    
    if system:
        full_prompt = f"{system}\n\n{prompt}"
    else:
        full_prompt = prompt
    
    encoding_start = time.time()
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        return_attention_mask=True
    ).to(model.device)
    
    prompt_eval_duration = int((time.time() - encoding_start) * 1e9)
    prompt_eval_count = inputs.input_ids.shape[1]
    
    generation_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    response = tokenizer.decode(outputs.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    eval_duration = int((time.time() - generation_start) * 1e9)
    eval_count = outputs.sequences.shape[1] - inputs.input_ids.shape[1]
    
    total_duration = int((time.time() - start_time) * 1e9)
    
    return {
        "model": model_name,
        "created_at": get_current_datetime(),
        "response": response,
        "done": True,
        "context": [1, 2, 3],
        "total_duration": total_duration,
        "load_duration": load_duration,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_count": eval_count,
        "eval_duration": eval_duration
    }

def chat_response(model_name: str, messages: List[Message]) -> Dict:
    """生成單一聊天回應"""
    start_time = time.time()
    
    model_data = load_model(model_name)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    load_duration = model_data["load_duration"]
    
    conversation = [{"role": msg.role, "content": msg.content} for msg in messages]
    
    encoding_start = time.time()
    input_ids = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    prompt_eval_duration = int((time.time() - encoding_start) * 1e9)
    prompt_eval_count = input_ids.shape[1]
    
    generation_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    response = tokenizer.decode(outputs.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    eval_duration = int((time.time() - generation_start) * 1e9)
    eval_count = outputs.sequences.shape[1] - input_ids.shape[1]
    
    total_duration = int((time.time() - start_time) * 1e9)
    
    return {
        "model": model_name,
        "created_at": get_current_datetime(),
        "message": {
            "role": "assistant",
            "content": response
        },
        "done": True,
        "total_duration": total_duration,
        "load_duration": load_duration,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_count": eval_count,
        "eval_duration": eval_duration
    }

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """生成端點"""
    try:
        if request.model not in MODEL_CONFIGS:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model} not found. Available models: {list(MODEL_CONFIGS.keys())}"
            )
        
        # 根據 stream 參數決定返回方式
        if request.stream:
            return StreamingResponse(
                generate_stream(
                    model_name=request.model,
                    prompt=request.prompt,
                    system=request.system
                ),
                media_type="text/event-stream"
            )
        else:
            return JSONResponse(
                generate_response(
                    model_name=request.model,
                    prompt=request.prompt,
                    system=request.system
                )
            )
            
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
        
        # 根據 stream 參數決定返回方式
        if request.stream:
            return StreamingResponse(
                chat_stream(
                    model_name=request.model,
                    messages=request.messages
                ),
                media_type="text/event-stream"
            )
        else:
            return JSONResponse(
                chat_response(
                    model_name=request.model,
                    messages=request.messages
                )
            )
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
