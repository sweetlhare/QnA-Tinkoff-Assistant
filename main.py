from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import List, Union
from rag import ChatBotRAG
import uvicorn

# Example usage
data_path = 'train_Tinkoff/dataset.json'
vector_db_path = './local_qdrant'
embeddings_model = 'aya:8b'
llm_name = 'aya:8b'
light_llm_name = 'aya:8b'

chatbot = ChatBotRAG(data_path, vector_db_path, embeddings_model, llm_name, light_llm_name)
chatbot.load_data()
docs_processed = chatbot.split_documents()
chatbot.create_vector_database(docs_processed)

app = FastAPI(title="Assistant API", version="0.1.0")

class ValidationErrorDetail(BaseModel):
    loc: List[Union[str, int]]
    msg: str
    type: str

class HTTPValidationError(BaseModel):
    detail: List[ValidationErrorDetail]

class Request(BaseModel):
    query: str = Field(..., title="Query")

class Response(BaseModel):
    text: str = Field(..., title="Text")
    links: List[str] = Field(..., title="Links")

@app.post("/assist", response_model=Response, responses={422: {"model": HTTPValidationError}})
def assist(request: Request):
    try:
        # Пример логики обработки запроса
        question = request.query
        # Здесь можно добавить логику обработки запроса
        # Например, вызов сторонних API, обработка текста и т.д.

        response_text, response_links = chatbot.answer_with_rag(question, num_docs_final=6)
        
        print("==================================Answer==================================")
        print(f"{response_text}")
        print("\n==================================Source docs==================================")
        for i, url in enumerate(response_links):
            print(f"{i}) {url}\n")

        # response_text = 'test'
        # response_links = ['url', 'url2']
        
        return Response(text=response_text, links=response_links)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8889)
