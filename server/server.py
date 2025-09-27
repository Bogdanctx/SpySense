import tempfile
import uvicorn

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from src.council import Council

app = FastAPI()
council = Council()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)


@app.post("/scan")
async def scan_file(file: UploadFile = File(...)):
    content = await file.read()

    with tempfile.NamedTemporaryFile(delete = False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name

    try:
        result = council.judge(tmp_path)
        result["verdict"] = int(result["verdict"])
        result["details"] = {k: float(v) for k, v in result["details"].items()}
        
        return result
    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)