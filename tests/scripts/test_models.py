#!/usr/bin/env python3
"""
Test script for individual model implementations
"""

if __name__ != "__main__":
    import pytest
    pytest.skip("Manual script; run directly with python", allow_module_level=True)

import asyncio
import time
import logging
from typing import List

from models.factory import ModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test data
TEST_TEXTS = [
    "def hello_world():\n    print('Hello, World!')",
    "function greet(name) {\n    console.log(`Hello, ${name}!`);\n}",
    "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}",
    "print('Hello, World!')",
    "echo 'Hello, World!'"
]

TEST_QUERY = "How to optimize database queries for better performance?"
TEST_DOCUMENTS = [
    "Database indexing is crucial for query performance. Create indexes on frequently queried columns.",
    "Use EXPLAIN PLAN to analyze query execution and identify bottlenecks in your SQL queries.",
    "Normalization reduces data redundancy but can impact query performance due to joins.",
    "Connection pooling helps manage database connections efficiently in web applications.",
    "Caching frequently accessed data can significantly reduce database load and improve response times.",
    "Query optimization involves rewriting SQL queries to use more efficient execution plans.",
    "Partitioning large tables can improve query performance by reducing the amount of data scanned.",
    "Regular database maintenance like updating statistics helps the query optimizer make better decisions.",
    "Cooking pasta requires boiling water and adding salt for better flavor.",
    "Machine learning models need proper feature engineering for optimal performance."
]


async def test_embedding_model(model_name: str):
    """Test an embedding model"""
    print(f"\n🔍 Testing Embedding Model: {model_name}")
    print("=" * 60)
    
    try:
        # Create and load model
        model = ModelFactory.create_model(model_name)
        print(f"✅ Model created: {model.__class__.__name__}")
        
        start_time = time.time()
        await model.load()
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Test encoding
        start_time = time.time()
        embeddings = await model.encode(TEST_TEXTS)
        encode_time = time.time() - start_time
        
        print(f"✅ Encoded {len(TEST_TEXTS)} texts in {encode_time:.2f}s")
        print(f"📊 Embedding shape: {embeddings.shape}")
        print(f"📏 Embedding dimension: {embeddings.shape[1]}")
        
        # Test similarity
        if hasattr(model, 'compute_similarity'):
            similarity = model.compute_similarity(embeddings[0], embeddings[1])
            print(f"🔗 Python-JavaScript similarity: {similarity:.4f}")
        
        # Performance metrics
        avg_time = encode_time / len(TEST_TEXTS) * 1000
        print(f"⚡ Average encoding time: {avg_time:.2f}ms per text")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {str(e)}")
        return False


async def test_rerank_model(model_name: str):
    """Test a rerank model"""
    print(f"\n🎯 Testing Rerank Model: {model_name}")
    print("=" * 60)
    
    try:
        # Create and load model
        model = ModelFactory.create_model(model_name)
        print(f"✅ Model created: {model.__class__.__name__}")
        
        start_time = time.time()
        await model.load()
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Test reranking
        start_time = time.time()
        scores = await model.compute_scores(TEST_QUERY, TEST_DOCUMENTS)
        rerank_time = time.time() - start_time
        
        print(f"✅ Reranked {len(TEST_DOCUMENTS)} documents in {rerank_time:.2f}s")
        print(f"📊 Scores range: {min(scores):.4f} - {max(scores):.4f}")
        
        # Show top results
        results_with_scores = list(zip(scores, TEST_DOCUMENTS, range(len(TEST_DOCUMENTS))))
        results_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        print(f"\n🏆 Top 3 results:")
        for i, (score, doc, idx) in enumerate(results_with_scores[:3]):
            print(f"  {i+1}. Score: {score:.4f} (Index: {idx})")
            print(f"     {doc[:80]}{'...' if len(doc) > 80 else ''}")
        
        # Performance metrics
        avg_time = rerank_time / len(TEST_DOCUMENTS) * 1000
        print(f"⚡ Average rerank time: {avg_time:.2f}ms per document")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {str(e)}")
        return False


async def test_all_models():
    """Test all available models"""
    print("🚀 TESTING ALL MODELS")
    print("=" * 80)
    
    # Get all models
    all_models = ModelFactory.list_all_models()
    
    print(f"📋 Found {len(all_models)} models:")
    for model_name, metadata in all_models.items():
        print(f"  • {model_name} ({metadata['type']})")
    
    # Test embedding models
    embedding_models = ModelFactory.get_models_by_type("embedding")
    rerank_models = ModelFactory.get_models_by_type("rerank")
    
    print(f"\n📊 Model breakdown:")
    print(f"  • Embedding models: {len(embedding_models)}")
    print(f"  • Rerank models: {len(rerank_models)}")
    
    results = {"embedding": {}, "rerank": {}}
    
    # Test embedding models
    for model_name in embedding_models:
        success = await test_embedding_model(model_name)
        results["embedding"][model_name] = success
    
    # Test rerank models
    for model_name in rerank_models:
        success = await test_rerank_model(model_name)
        results["rerank"][model_name] = success
    
    # Summary
    print("\n\n📈 TEST SUMMARY")
    print("=" * 80)
    
    for model_type, model_results in results.items():
        print(f"\n{model_type.upper()} MODELS:")
        for model_name, success in model_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {status} {model_name}")
    
    # Overall stats
    total_tests = sum(len(models) for models in results.values())
    passed_tests = sum(sum(models.values()) for models in results.values())
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} models passed")
    
    if passed_tests == total_tests:
        print("🎉 All models working correctly!")
    else:
        print("⚠️  Some models failed - check logs for details")


async def test_specific_model(model_name: str):
    """Test a specific model"""
    try:
        metadata = ModelFactory.get_model_metadata(model_name)
        model_type = metadata["type"]
        
        if model_type == "embedding":
            await test_embedding_model(model_name)
        elif model_type == "rerank":
            await test_rerank_model(model_name)
        else:
            print(f"❌ Unknown model type: {model_type}")
            
    except ValueError as e:
        print(f"❌ Error: {e}")


async def main():
    """Main test function"""
    import sys
    
    if len(sys.argv) > 1:
        # Test specific model
        model_name = sys.argv[1]
        print(f"Testing specific model: {model_name}")
        await test_specific_model(model_name)
    else:
        # Test all models
        await test_all_models()


if __name__ == "__main__":
    asyncio.run(main())
