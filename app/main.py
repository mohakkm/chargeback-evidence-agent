"""
FastAPI entrypoint for the Chargeback Evidence Responder.
"""

from fastapi import FastAPI

app = FastAPI(title="Chargeback Evidence Responder")

@app.get("/")
def read_root():
    return {"status": "scaffolded"}
