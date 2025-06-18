import logging
import os
from typing import Optional

import loguru
from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPOTLPSpanExporter,
)
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.attributes.http_attributes import (
    HTTP_REQUEST_METHOD,
    HTTP_RESPONSE_STATUS_CODE,
    HTTP_ROUTE,
)
from phoenix.otel import register

# Configuration constants
SERVICE_NAME_DEFAULT = "langgraph-rag-service"
PHOENIX_ENDPOINT_DEFAULT = "http://localhost:6006/v1/traces"
SIGNOZ_ENDPOINT_DEFAULT = "http://localhost:4317"
METRIC_EXPORT_INTERVAL_MS = 5000

# TODO: Move these to settings


def setup_telemetry(
    service_name: Optional[str] = None,
    phoenix_endpoint: Optional[str] = None,
    signoz_endpoint: Optional[str] = None,
    signoz_token: Optional[str] = None,
) -> tuple[trace.Tracer, metrics.Meter]:
    """
    Setup OpenTelemetry instrumentation with Phoenix and SigNoz exporters.

    Args:
        service_name: Name of the service for telemetry
        phoenix_endpoint: Phoenix OTLP endpoint URL
        signoz_endpoint: SigNoz OTLP endpoint URL
        signoz_token: SigNoz access token

    Returns:
        Tuple of (tracer, meter) instances
    """
    # Get configuration from environment or use defaults
    service_name = service_name or os.getenv("SERVICE_NAME", SERVICE_NAME_DEFAULT)
    phoenix_endpoint = phoenix_endpoint or os.getenv(
        "PHOENIX_ENDPOINT", PHOENIX_ENDPOINT_DEFAULT
    )
    signoz_endpoint = signoz_endpoint or os.getenv(
        "SIGNOZ_ENDPOINT", SIGNOZ_ENDPOINT_DEFAULT
    )
    signoz_token = signoz_token or os.getenv("SIGNOZ_TOKEN", "<SIGNOZ_TOKEN>")

    try:
        # Phoenix setup - prevent it from setting global tracer provider
        phoenix_tracer = register(  # noqa: F841
            project_name=service_name,
            endpoint=phoenix_endpoint,
            set_global_tracer_provider=False,
        )

        # Setup exporters
        phoenix_span_exporter = HTTPOTLPSpanExporter(endpoint=phoenix_endpoint)
        signoz_headers = {"signoz-access-token": signoz_token}

        signoz_span_exporter = OTLPSpanExporter(
            endpoint=signoz_endpoint,
            headers=signoz_headers,
        )
        signoz_metric_exporter = OTLPMetricExporter(
            endpoint=signoz_endpoint,
            headers=signoz_headers,
        )
        signoz_log_exporter = OTLPLogExporter(
            endpoint=signoz_endpoint,
            headers=signoz_headers,
        )

        # Configure tracing with both exporters
        resource = Resource.create({SERVICE_NAME: service_name})
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(phoenix_span_exporter))
        tracer_provider.add_span_processor(BatchSpanProcessor(signoz_span_exporter))
        trace.set_tracer_provider(tracer_provider)

        # Configure metrics
        metric_reader = PeriodicExportingMetricReader(
            exporter=signoz_metric_exporter,
            export_interval_millis=METRIC_EXPORT_INTERVAL_MS,
        )
        metrics.set_meter_provider(
            MeterProvider(metric_readers=[metric_reader], resource=resource)
        )

        # Configure logging
        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(signoz_log_exporter)
        )
        set_logger_provider(logger_provider)

        # Setup logging handlers
        handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.DEBUG)

        # Setup loguru integration
        setup_loguru_integration(handler)

        return trace.get_tracer(__name__), metrics.get_meter(__name__)

    except Exception as e:
        logging.error(f"Failed to setup telemetry: {e}")
        # Return no-op tracer and meter if setup fails
        return trace.NoOpTracer(), metrics.NoOpMeter(__name__)


def setup_loguru_integration(otel_handler: LoggingHandler) -> None:
    """
    Setup loguru to send logs to OpenTelemetry.

    Args:
        otel_handler: OpenTelemetry logging handler
    """

    def otel_sink(message):
        """Custom sink that forwards loguru messages to OTEL handler"""
        record = message.record
        level_name = record["level"].name
        log_message = record["message"]

        # Map loguru levels to standard logging levels
        level_mapping = {
            "TRACE": logging.DEBUG,
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "SUCCESS": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        # Create a standard library LogRecord
        level = level_mapping.get(level_name, logging.INFO)
        log_record = logging.LogRecord(
            name=record["name"],
            level=level,
            pathname=record["file"].path,
            lineno=record["line"],
            msg=log_message,
            args=(),
            exc_info=record["exception"],
        )

        # Add trace context for correlation
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            span_context = current_span.get_span_context()
            log_record.otelTraceID = format(span_context.trace_id, "032x")
            log_record.otelSpanID = format(span_context.span_id, "016x")

        otel_handler.emit(log_record)

    # Add the OTEL sink to loguru
    loguru.logger.add(otel_sink, level="TRACE", serialize=False)


def instrument_app(app):
    """
    Instrument FastAPI app with OpenTelemetry with enhanced trace naming and metadata.

    Usage:
        app = FastAPI()
        instrument_app(app)

    Args:
        app: FastAPI application instance

    Returns:
        Instrumented FastAPI app
    """
    try:
        # Custom request hook to add better span names and metadata
        def fastapi_request_hook(span, scope):
            """Add enhanced metadata to FastAPI request spans"""
            if span and span.is_recording():
                # Get route info
                route = scope.get("route")
                method = scope.get("method", "UNKNOWN")
                path = scope.get("path", "/")

                # Set operation type
                span.set_attribute("operation.type", "http.request")
                span.set_attribute("service.component", "api")

                # Enhanced span name with method and route
                if route and hasattr(route, "path"):
                    route_path = route.path
                    span.update_name(f"{method} {route_path}")
                    span.set_attribute(HTTP_ROUTE, route_path)
                else:
                    span.update_name(f"{method} {path}")

                # Add request metadata using semantic conventions
                span.set_attribute(HTTP_REQUEST_METHOD, method)
                span.set_attribute("http.target", path)
                span.set_attribute("http.scheme", scope.get("scheme", "http"))

                # Add client info if available
                client = scope.get("client")
                if client:
                    span.set_attribute("http.client_ip", client[0])

        def fastapi_response_hook(span, message):
            """Add response metadata to FastAPI spans"""
            if span and span.is_recording():
                if message.get("type") == "http.response.start":
                    status_code = message.get("status", 0)
                    span.set_attribute(HTTP_RESPONSE_STATUS_CODE, status_code)

                    # Set span status based on HTTP status
                    if status_code >= 400:
                        span.set_status(trace.Status(trace.StatusCode.ERROR))
                    else:
                        span.set_status(trace.Status(trace.StatusCode.OK))

        # Custom HTTP client hook for outbound requests
        def httpx_request_hook(span, request):
            """Add enhanced metadata to HTTPX client spans"""
            if span and span.is_recording():
                # Set operation type for outbound requests
                span.set_attribute("operation.type", "http.client")
                span.set_attribute("service.component", "http_client")

                # Enhanced span name for external calls
                url = str(request.url)
                method = request.method
                host = request.url.host

                span.update_name(f"HTTP {method} {host}")
                span.set_attribute(HTTP_REQUEST_METHOD, method)
                span.set_attribute("http.url", url)
                span.set_attribute("http.target_host", host)
                span.set_attribute("external.service", host)

        def httpx_response_hook(span, request, response):
            """Add response metadata to HTTPX client spans"""
            if span and span.is_recording():
                span.set_attribute(HTTP_RESPONSE_STATUS_CODE, response.status_code)

                # Set span status based on HTTP status
                if response.status_code >= 400:
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                else:
                    span.set_status(trace.Status(trace.StatusCode.OK))

        # Instrument FastAPI with custom hooks
        FastAPIInstrumentor.instrument_app(
            app,
            server_request_hook=fastapi_request_hook,
            client_request_hook=fastapi_response_hook,
        )

        # Instrument HTTP client with custom hooks
        HTTPXClientInstrumentor().instrument(
            request_hook=httpx_request_hook,
            response_hook=httpx_response_hook,
        )

        # Enhanced logging instrumentation
        def logging_hook(span, record):
            """Add enhanced metadata to logging spans"""
            if span and span.is_recording():
                span.set_attribute("operation.type", "log")
                span.set_attribute("service.component", "logger")
                span.set_attribute("log.message", record.getMessage())
                span.set_attribute("log.level", record.levelname)
                span.set_attribute("log.logger", record.name)

                # Add source location
                if record.pathname:
                    span.set_attribute("code.filepath", record.pathname)
                    span.set_attribute("code.lineno", record.lineno)
                    if record.funcName:
                        span.set_attribute("code.function", record.funcName)

        # Configure logging instrumentation
        LoggingInstrumentor().instrument(
            set_logging_format=True,
            log_hook=logging_hook,
        )

        logging.info(f"Successfully instrumented FastAPI app: {app}")

    except Exception as e:
        logging.error(f"Failed to instrument FastAPI app: {e}")

    return app


def create_custom_span(tracer, operation_name: str, operation_type: str, **attributes):
    """
    Create a custom span with standardized naming and metadata.

    Args:
        tracer: OpenTelemetry tracer instance
        operation_name: Name of the operation (e.g., "database_query", "llm_call")
        operation_type: Type of operation (e.g., "database", "ai.llm", "cache")
        **attributes: Additional span attributes

    Returns:
        Span context manager
    """
    span = tracer.start_span(operation_name)

    # Set standard attributes
    span.set_attribute("operation.type", operation_type)
    span.set_attribute("service.name", SERVICE_NAME_DEFAULT)

    # Set custom attributes
    for key, value in attributes.items():
        span.set_attribute(key, value)

    return span
