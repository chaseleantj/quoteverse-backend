from fastapi import FastAPI
from api.routers import base, quotes
from api.services.initialization.quote_initializer import init_quotes_and_processor


app = FastAPI()
app.include_router(base.base_router)
app.include_router(quotes.quotes_router)

@app.on_event("startup")
async def startup_event():
    """Run initialization on startup"""
    processor = init_quotes_and_processor()
    app.state.processor = processor
    print("Processor initialized")