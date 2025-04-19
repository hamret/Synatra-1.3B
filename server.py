
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from synatra import generate_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate_response(request: Request):
    data = await request.json()
    question = data.get("question")

    if not question:
        return {"answer": "질문이 비어 있습니다."}


    answer = generate_text(question)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7001)
