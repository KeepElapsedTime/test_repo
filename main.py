from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定義與 Ollama API 相容的請求模型
class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[str]] = None
    options: Optional[Dict] = None

class GenerateResponse(BaseModel):
    model: str
    response: str
    done: bool

# 模型配置
MODEL_CONFIGS = {
    "jais-family-30b-16k-chat": "inceptionai/jais-family-30b-16k-chat",
    "jais-family-30b-8k-chat": "inceptionai/jais-family-30b-8k-chat",
    "jais-family-13b-chat": "inceptionai/jais-family-13b-chat",
    "jais-adapted-70b-chat": "inceptionai/jais-adapted-70b-chat",
    "jais-adapted-13b-chat": "inceptionai/jais-adapted-13b-chat",
    "llama-3-sherkala-8b": "inceptionai/Llama-3.1-Sherkala-8B-Chat",
    "llama-3-nanda-10b": "MBZUAI/Llama-3-Nanda-10B-Chat"
}

# 加載模型的緩存
loaded_models = {}

def load_model(model_name: str):
    """加載模型和tokenizer"""
    if model_name not in loaded_models:
        model_path = MODEL_CONFIGS[model_name]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model {model_name} from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        
        loaded_models[model_name] = {
            "model": model,
            "tokenizer": tokenizer
        }
    
    return loaded_models[model_name]

def get_response(text: str, model_name: str):
    """生成回應"""
    model_data = load_model(model_name)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    inputs = input_ids.to(device)
    input_len = inputs.shape[-1]
    
    generate_ids = model.generate(
        inputs,
        top_p=0.9,
        temperature=0.3,
        max_length=2048,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
    )
    
    response = tokenizer.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]
    
    return response

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """處理生成請求的端點"""
    try:
        if request.model not in MODEL_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
        
        # 組合提示詞
        prompt = request.prompt
        if request.system:
            prompt = f"{request.system}\n\n{prompt}"
            
        response = get_response(prompt, request.model)
        
        return GenerateResponse(
            model=request.model,
            response=response,
            done=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
