import subprocess
import time
import httpx
import uvicorn
import os
import argparse
import json
import datetime
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

# ============================================================================
# CONFIGURATION
# ============================================================================
CENTRAL_LOG_API = "https://ai.nomineelife.com/api/logs/internal"
LLAMA_BINARY_PATH = "./build/bin/llama-server" 
LLAMA_PORT = 8081    # Internal port (Hidden)
PUBLIC_PORT = 8080   # Exposed port (Tailscale)

# Default model if none is provided
DEFAULT_MODEL = "default-model.gguf"
CURRENT_MODEL_PATH = "" 

app = FastAPI()
llama_process = None

# ============================================================================
# LIFECYCLE MANAGEMENT
# ============================================================================

@app.on_event("startup")
def startup_event():
    global llama_process, CURRENT_MODEL_PATH
    
    if not CURRENT_MODEL_PATH: 
        CURRENT_MODEL_PATH = f"./models/{DEFAULT_MODEL}"

    print(f"🚀 Starting Llama.cpp with model: {CURRENT_MODEL_PATH}")

    if not os.path.exists(CURRENT_MODEL_PATH):
        print(f"❌ Error: Model not found at {CURRENT_MODEL_PATH}")
        return

    # Start the subprocess
    llama_process = subprocess.Popen([
        LLAMA_BINARY_PATH,
        "-m", CURRENT_MODEL_PATH,
        "--port", str(LLAMA_PORT),
        "--ctx-size", "4096",
        "--host", "127.0.0.1" 
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    time.sleep(2)
    print("✅ Llama.cpp is running.")

@app.on_event("shutdown")
def shutdown_event():
    global llama_process
    if llama_process:
        print("🛑 Terminating Llama.cpp...")
        llama_process.terminate()

# ============================================================================
# LOGGING LOGIC
# ============================================================================

async def upload_logs(log_payload: dict):
    """
    Writes logs to a local file for testing.
    """
    try:
        log_file = "worker_logs.jsonl"
        log_entry = json.dumps(log_payload)
        with open(log_file, "a") as f:
            f.write(log_entry + "\n")
        print(f"📝 Log saved locally to {log_file}")
    except Exception as e:
        print(f"❌ Failed to write local log: {e}")

# ============================================================================
# NEW: HEALTH & STATUS ENDPOINTS (Fixes 404s)
# ============================================================================

@app.get("/health")
async def health_check():
    """Proxies the health check to the internal Llama server."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=2.0)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except httpx.RequestError:
            return JSONResponse(content={"status": "error", "message": "Llama server unreachable"}, status_code=503)

@app.get("/v1/models")
async def get_models():
    """Proxies the model list to the internal Llama server."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"http://127.0.0.1:{LLAMA_PORT}/v1/models", timeout=2.0)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except httpx.RequestError:
            return JSONResponse(content={"error": "Llama server unreachable"}, status_code=503)

# ============================================================================
# PROXY ENDPOINT WITH CLEAN LOGGING
# ============================================================================

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    start_time = time.time()
    
    # 1. EXTRACT METADATA (Request Side)
    user_id = body.get("user", "unknown")
    client_ip = request.client.host if request.client else "unknown"
    
    # Extract the user prompt (Last message content)
    messages = body.get("messages", [])
    user_prompt = "N/A"
    if messages and isinstance(messages, list):
        # Get the last message from the user
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")
                break

    async def response_generator():
        # Variables to reconstruct the clean response
        model_response = ""
        token_count = 0
        tokens_per_sec = 0.0
        buffer = "" # To handle split JSON chunks

        async with httpx.AsyncClient() as client:
            try:
                req = client.build_request(
                    "POST", 
                    f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions", 
                    json=body, 
                    timeout=None
                )
                r = await client.send(req, stream=True)
                
                async for chunk in r.aiter_bytes():
                    yield chunk # Send raw chunk to user immediately (Low Latency)
                    
                    # --- PARSING LOGIC ---
                    try:
                        # Decode chunk and add to buffer
                        text_chunk = chunk.decode("utf-8")
                        buffer += text_chunk
                        
                        # Process complete lines from buffer
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            
                            # Parse "data: {...}" lines
                            if line.startswith("data: ") and line != "data: [DONE]":
                                json_str = line[6:] # Remove "data: " prefix
                                try:
                                    data = json.loads(json_str)
                                    
                                    # 1. Extract Content (Text Delta)
                                    choices = data.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            model_response += content
                                    
                                    # 2. Extract Stats (Llama.cpp specific)
                                    # We look for 'timings' in every chunk, usually present in the last few
                                    timings = data.get("timings", {})
                                    if timings:
                                        # Update stats with the latest values found
                                        if "predicted_per_second" in timings:
                                            tokens_per_sec = timings["predicted_per_second"]
                                        if "predicted_n" in timings:
                                            token_count = timings["predicted_n"]

                                except json.JSONDecodeError:
                                    continue # Skip incomplete JSON lines

                    except Exception:
                        pass # Don't crash the stream if logging fails temporarily
            
            except Exception as e:
                error_json = json.dumps({"error": str(e)}).encode()
                yield error_json

        # --- FINAL LOG CONSTRUCTION ---
        # Calculate tokens/sec manually if Llama didn't provide it
        duration_sec = time.time() - start_time
        if tokens_per_sec == 0 and duration_sec > 0 and token_count > 0:
            tokens_per_sec = token_count / duration_sec

        log_data = {
            "date_time": datetime.datetime.now().isoformat(),
            "user_id": user_id,
            "ip_address": client_ip,
            "model": CURRENT_MODEL_PATH,
            "user_prompt": user_prompt,
            "model_response": model_response, # Now a clean string
            "number_of_tokens": token_count,
            "tokens_per_sec": round(tokens_per_sec, 2)
        }
        
        background_tasks.add_task(upload_logs, log_data)

    return StreamingResponse(response_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama Edge Worker")
    parser.add_argument("--model", type=str, default="default-model.gguf", help="Model filename")
    args = parser.parse_args()
    
    CURRENT_MODEL_PATH = f"./models/{args.model}"
    uvicorn.run(app, host="0.0.0.0", port=PUBLIC_PORT)
