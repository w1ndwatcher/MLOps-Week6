# main.py
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging
import json
import time
# --- OpenTelemetry Setup (from demo_log.py) ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

try:
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
    trace.get_tracer_provider().add_span_processor(span_processor)
    print("OpenTelemetry setup successful.")
except Exception as e:
    print(f"Failed to set up OpenTelemetry: {e}")

# Structured Logging Setup
logger = logging.getLogger("iris-api-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

app = FastAPI(title="🌸 Iris Classifier FastAPI")

# Load model
model = None
try:
    model_name = "model.joblib"
    model_uri = f"model/{model_name}"
    
    logger.info(f"Attempting to load model '{model_name}' from URI: {model_uri}")
    model = joblib.load(model_uri)
    logger.info("Successfully loaded model.")
except Exception as e:
    logger.error(f"Failed to load model on startup. Error: {e}")


# App State & Probes
@app.on_event("startup")
async def startup_event():
    # This check ensures the app reports 'ready' only if the model loaded.
    if model is None:
        logger.warning("Model is not loaded. Readiness probe will fail.")
        app_state["is_ready"] = False
    else:
        logger.info("Model is loaded. Application is ready.")
        app_state["is_ready"] = True

app_state = {"is_ready": False, "is_alive": True}

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=500)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"] and model is not None:
        return {"status": "ready"}
    return Response(status_code=503) # Service Unavailable
# ----

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
# Exception Handler
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}

# @app.post("/predict/")
# def predict_species(data: IrisInput):
#     input_df = pd.DataFrame([data.dict()])
#     prediction = model.predict(input_df)[0]
#     return {
#         "predicted_class": prediction
#     }
    
# Predict Endpoint
@app.post("/predict/")
async def predict(data: IrisInput):
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")
        try:
            input_df = pd.DataFrame([data.dict()])
            prediction = model.predict(input_df)
            result = {"prediction": prediction[0]}
            latency = round((time.time() - start_time) * 1000, 2)

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": data.dict(),
                "predicted_class": result,
                "latency_ms": latency,
                "status": "success"
            }))
            return result
        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")