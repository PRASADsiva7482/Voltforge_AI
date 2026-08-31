# VFAI-031 observability contract

VoltForge AI exposes process-local, content-free telemetry for the Python
service and the Spring gateway. It is operational state, not conversation
storage.

The private Python endpoint
`/voltForge-ai/api/v1/model/metrics` returns the `vfai031-observability-v1`
snapshot. The existing health and system-health responses expose the smaller
observability health view. Every model route, including metrics, remains behind
the service-token boundary in production. The Spring adapter publishes its
bounded view through the `voltforgeAiTelemetry` Actuator health contributor.

The snapshot records request outcomes, route-template counts, bounded latency
windows, explicit zero queue time for non-blocking local admission, stage
latency, stream event/byte counts, first-delta latency, output character
counts, observed token counts, generation path, fallback reason, artifact
identity, retrieval status/cache/failure counts, model-load failures, and
traced Python heap samples.

Allowed logs contain only a safe request ID, static route template, enum-like
outcome, numeric status, bounded duration, and internal error type/code. Raw
prompts, project/canvas/code contents, memory text, JWT subjects, response
text, hidden reasoning, and arbitrary user labels are not accepted by the
telemetry registry and are not written to its report.

Alerts are exposed as state transitions and health data for model-load
failure, latency saturation, active-request saturation, repeated fallback
after a neural attempt, and repeated retrieval errors. Thresholds are bounded
by the service environment variables documented in `.env.example`. No remote
metrics provider or hosted model is required.
