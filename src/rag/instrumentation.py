# telemetry_config.py
import os

import phoenix as px
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from phoenix.otel import register


def setup_telemetry():
    # Phoenix setup
    px.launch_app()
    phoenix_tracer = register(  # noqa: F841
        project_name="langgraph-rag-service", endpoint="http://localhost:6006/v1/traces"
    )

    # SigNoz OTLP setup
    signoz_span_exporter = OTLPSpanExporter(
        endpoint=os.getenv("SIGNOZ_ENDPOINT", "http://localhost:4317"),
        headers={"signoz-access-token": os.getenv("SIGNOZ_TOKEN", "")},
    )

    signoz_metric_exporter = OTLPMetricExporter(
        endpoint=os.getenv("SIGNOZ_ENDPOINT", "http://localhost:4317"),
        headers={"signoz-access-token": os.getenv("SIGNOZ_TOKEN", "")},
    )

    # Configure tracing
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(BatchSpanProcessor(signoz_span_exporter))
    trace.set_tracer_provider(tracer_provider)

    # Configure metrics
    metric_reader = PeriodicExportingMetricReader(
        exporter=signoz_metric_exporter, export_interval_millis=5000
    )
    metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

    return trace.get_tracer(__name__), metrics.get_meter(__name__)


def instrument_app(app):
    """Instrument FastAPI app"""
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    LoggingInstrumentor().instrument(set_logging_format=True)
