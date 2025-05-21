from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging
import traceback

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
    template: Optional[str] = None
    context: Optional[List[str]] = None
    options: Optional[Dict] = None

class GenerateResponse(BaseModel):
    model: str
    response: str
    done: bool

MODEL_CONFIGS = {
    "jais-family-30b-16k-chat": "inceptionai/jais-family-30b-16k-chat",
    "jais-family-30b-8k-chat": "inceptionai/jais-family-30b-8k-chat",
    "jais-family-13b-chat": "inceptionai/jais-family-13b-chat",
    "jais-adapted-70b-chat": "inceptionai/jais-adapted-70b-chat",
    "jais-adapted-13b-chat": "inceptionai/jais-adapted-13b-chat",
    "llama-3-sherkala-8b": "inceptionai/Llama-3.1-Sherkala-8B-Chat",
    "llama-3-nanda-10b": "MBZUAI/Llama-3-Nanda-10B-Chat"
}

loaded_models = {}

def load_model(model_name: str):
    """加載模型和tokenizer"""
    try:
        if model_name not in loaded_models:
            model_path = MODEL_CONFIGS[model_name]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading model {model_name} from {model_path}")
            logger.info(f"Using device: {device}")
            
            # 設置 tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"  # 確保padding在左側
            )
            
            # 確保設置特殊token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = tokenizer.eos_token = "</s>"
                    
            logger.info("Tokenizer loaded successfully")
            
            # 加載模型
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                pad_token_id=tokenizer.pad_token_id
            )
            
            logger.info("Model loaded successfully")
            
            loaded_models[model_name] = {
                "model": model,
                "tokenizer": tokenizer
            }
        
        return loaded_models[model_name]
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_response(text: str, model_name: str):
    """生成回應"""
    try:
        model_data = load_model(model_name)
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Generating response using device: {device}")
        
        # 正確設置 tokenizer 輸入
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            return_attention_mask=True
        )
        
        # 將所有輸入移到正確的設備上
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        input_len = input_ids.shape[-1]
        logger.info(f"Input length: {input_len}")
        
        # 生成回應
        generate_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            top_p=0.9,
            temperature=0.3,
            max_length=2048,
            min_length=input_len + 4,
            repetition_penalty=1.2,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # 解碼回應
        response = tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        # 移除輸入文本
        response = response[len(text):].strip()
        
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": traceback.format_exc()}
    )

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """處理生成請求的端點"""
    try:
        logger.info(f"Received request for model: {request.model}")
        logger.info(f"Prompt: {request.prompt[:100]}...")
        
        if request.model not in MODEL_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
        
        # 組合提示詞
        prompt = request.prompt
        if request.system:
            prompt = f"{request.system}\n\n{prompt}"
            
        response = get_response(prompt, request.model)
        logger.info("Response generated successfully")
        
        return GenerateResponse(
            model=request.model,
            response=response,
            done=True
        )
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
