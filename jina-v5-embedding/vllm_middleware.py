"""Compatibility patch for vLLM pooling requests selecting static LoRAs."""

from types import MethodType


def _patch_handler(handler) -> None:
    if handler is None or getattr(handler, "_jina_lora_patched", False):
        return

    original = handler._maybe_get_adapters

    def patched(this, ctx, supports_default_mm_loras=False):
        raw_name = ctx.request.model
        model_name = (
            raw_name
            if isinstance(raw_name, str)
            else getattr(raw_name, "value", str(raw_name))
        )
        if model_name in this.models.lora_requests:
            ctx.lora_request = this.models.lora_requests[model_name]
            return None
        return original(ctx, supports_default_mm_loras)

    handler._maybe_get_adapters = MethodType(patched, handler)
    handler._jina_lora_patched = True


async def patch_pooling_lora(request, call_next):
    state = request.app.state
    _patch_handler(getattr(state, "serving_embedding", None))
    _patch_handler(getattr(state, "serving_pooling", None))
    return await call_next(request)
