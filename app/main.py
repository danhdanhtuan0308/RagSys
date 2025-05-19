from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path
import uvicorn

from routers import metadata, interaction
from services import api_router, get_data_manager

#call API app 
app = FastAPI(
    title="Amazon Product Review API",
    description="FastAPI service for recommendation application",
    version="0.0.1",
)

# point Jinja2 to ./templates
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.on_event("startup")
def startup_event():
    # preload & cache DataManager
    get_data_manager()

# include your existing routers (if any)
app.include_router(metadata.router)
app.include_router(interaction.router)

# JSON API under /recommend
app.include_router(api_router)

# Show form
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def form_get(request: Request):
    return templates.TemplateResponse(
        "recommend_form.html",
        {"request": request, "recs": None, "error": None, "user_id": None},
    )

# Handle form submit
@app.post("/", response_class=HTMLResponse, include_in_schema=False)
def form_post(
    request: Request,
    user_id: str = Form(...),
    top_k: int = Form(10),
    dm = Depends(get_data_manager),
):
    try:
        candidates = dm.retrieve(user_id, top_k)
        ranked     = dm.rank(user_id, candidates)
        recs = [{"item_id": dm.item_ids[idx], "score": score} for idx, score in ranked]
        return templates.TemplateResponse(
            "recommend_form.html",
            {"request": request, "recs": recs, "error": None, "user_id": user_id},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "recommend_form.html",
            {"request": request, "recs": None, "error": str(e), "user_id": user_id},
        )

if __name__ == "__main__":
    # Use import string so reload works
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
