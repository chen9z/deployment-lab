from models.factory import ModelFactory


def test_supported_models():
    models = ModelFactory.get_supported_models()
    assert models["jinaai/jina-embeddings-v4"] == "embedding"
    assert models["jinaai/jina-code-embeddings-1.5b"] == "embedding"
    assert models["Qwen/Qwen3-Embedding-4B"] == "embedding"
    assert models["jinaai/jina-reranker-m0"] == "rerank"
    assert models["jinaai/jina-reranker-v2-base-multilingual"] == "rerank"


def test_create_model_types():
    assert ModelFactory.create_model("jinaai/jina-embeddings-v4").get_type() == "embedding"
    assert (
        ModelFactory.create_model("jinaai/jina-code-embeddings-1.5b").get_type()
        == "embedding"
    )
    assert (
        ModelFactory.create_model("Qwen/Qwen3-Embedding-4B").get_type()
        == "embedding"
    )
    assert ModelFactory.create_model("jinaai/jina-reranker-m0").get_type() == "rerank"
    assert (
        ModelFactory.create_model("jinaai/jina-reranker-v2-base-multilingual").get_type()
        == "rerank"
    )


def test_unknown_model():
    try:
        ModelFactory.create_model("unknown")
    except ValueError as exc:
        assert "Unsupported model" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown model")
