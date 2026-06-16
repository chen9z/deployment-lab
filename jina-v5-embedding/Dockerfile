ARG VLLM_IMAGE=vllm/vllm-openai:nightly-rollback-20260516
FROM ${VLLM_IMAGE}

RUN python3 -m pip install --no-cache-dir \
    orjson==3.11.7

WORKDIR /app
COPY gateway.py /app/gateway.py
COPY vllm_middleware.py /app/vllm_middleware.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod 0755 /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
