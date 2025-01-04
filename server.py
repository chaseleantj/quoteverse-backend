import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from run_migrations import run_migrations
from api.routers import base, quotes
from api.services.initialization.quote_initializer import init_quotes_and_processor
from api.settings import settings


app = FastAPI()

origins = [
    "http://127.0.0.1:5500",
    settings.ALLOWED_ORIGINS,   # Allow configurable frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(base.base_router)
app.include_router(quotes.quotes_router)

@app.on_event("startup")
async def startup_event():
    run_migrations()
    processor = init_quotes_and_processor()
    app.state.processor = processor
    print("Processor initialized")