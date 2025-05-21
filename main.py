from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging

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

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    context: Optional[List[str]] = None
    options: Optional[Dict] = None
    max_tokens: Optional[int] = 500

class GenerateResponse(BaseModel):
    model: str
    response: str
    done: bool

MODEL_CONFIGS = {
    "llama-3-sherkala-8b": "inceptionai/Llama-3.1-Sherkala-8B-Chat",
    "llama-3-nanda-10b": "MBZUAI/Llama-3-Nanda-10B-Chat"
}

loaded_models = {}

def load_model(model_name: str):
    """加載模型和tokenizer"""
    if model_name not in loaded_models:
        model_path = MODEL_CONFIGS[model_name]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading model {model_name} from {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 設置chat template
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
        
        loaded_models[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device
        }
    
    return loaded_models[model_name]

def get_response(text: str, model_name: str, system: Optional[str] = None, max_tokens: int = 500):
    """生成回應"""
    model_data = load_model(model_name)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    device = model_data["device"]
    
    # 準備對話內容
    conversation = []
    if system:
        conversation.append({"role": "system", "content": system})
    conversation.append({"role": "user", "content": text})
    
    # 使用chat template生成輸入
    input_ids = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    
    # 生成回應
    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        stop_strings=["<|eot_id|>"],
        tokenizer=tokenizer,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    # 解碼生成的文本
    gen_text = tokenizer.decode(
        gen_tokens[0][len(input_ids[0]): -1],
        skip_special_tokens=True
    )
    
    return gen_text.strip()

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """處理生成請求的端點"""
    try:
        if request.model not in MODEL_CONFIGS:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model} not found. Available models: {list(MODEL_CONFIGS.keys())}"
            )
        
        max_tokens = request.max_tokens or 500
        response = get_response(
            text=request.prompt,
            model_name=request.model,
            system=request.system,
            max_tokens=max_tokens
        )
        
        return GenerateResponse(
            model=request.model,
            response=response,
            done=True
        )
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
