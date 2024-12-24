from fastapi import APIRouter


base_router = APIRouter()

@base_router.get("/")
def health_check():
    return {"ok": True}
