#!/usr/bin/env python3
"""
Test script for the Multi-Model API Server
"""

if __name__ != "__main__":
    import pytest
    pytest.skip("Manual script; run directly with python", allow_module_level=True)

import asyncio
import aiohttp
import json
import time

# API server configuration
API_BASE_URL = "http://localhost:8000"

async def test_api():
    """Test the API endpoints"""
    async with aiohttp.ClientSession() as session:
        
        # Test root endpoint
        print("=== Testing Root Endpoint ===")
        async with session.get(f"{API_BASE_URL}/") as response:
            data = await response.json()
            print(f"Status: {response.status}")
            print(f"Response: {json.dumps(data, indent=2)}")
        
        # Test models endpoint
        print("\n=== Testing Models Endpoint ===")
        async with session.get(f"{API_BASE_URL}/v1/models") as response:
            data = await response.json()
            print(f"Status: {response.status}")
            print(f"Available models: {[model['id'] for model in data['data']]}")
        
        # Test embedding endpoint
        print("\n=== Testing Embedding Endpoint ===")
        embedding_request = {
            "input": ["Hello world", "How are you?"],
            "model": "jinaai/jina-embeddings-v2-base-code"
        }
        
        async with session.post(
            f"{API_BASE_URL}/v1/embeddings",
            json=embedding_request
        ) as response:
            if response.status == 200:
                data = await response.json()
                print(f"Status: {response.status}")
                print(f"Model: {data['model']}")
                print(f"Number of embeddings: {len(data['data'])}")
                print(f"Embedding dimension: {len(data['data'][0]['embedding'])}")
                print(f"Usage: {data['usage']}")
            else:
                error_data = await response.text()
                print(f"Error Status: {response.status}")
                print(f"Error: {error_data}")
        
        # Test rerank endpoint
        print("\n=== Testing Rerank Endpoint ===")
        rerank_request = {
            "model": "jinaai/jina-reranker-m0",
            "query": "What is machine learning?",
            "documents": [
                "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                "Cooking is the art of preparing food using various techniques.",
                "Deep learning uses neural networks with multiple layers.",
                "Sports are physical activities involving skill and competition."
            ],
            "top_k": 3,
            "return_documents": True
        }
        
        async with session.post(
            f"{API_BASE_URL}/v1/rerank",
            json=rerank_request
        ) as response:
            if response.status == 200:
                data = await response.json()
                print(f"Status: {response.status}")
                print(f"Model: {data['model']}")
                print(f"Number of results: {len(data['results'])}")
                print("Results:")
                for i, result in enumerate(data['results']):
                    print(f"  {i+1}. Score: {result['relevance_score']:.4f}, Index: {result['index']}")
                    if result.get('document'):
                        print(f"     Document: {result['document'][:100]}...")
                print(f"Usage: {data['usage']}")
            else:
                error_data = await response.text()
                print(f"Error Status: {response.status}")
                print(f"Error: {error_data}")

async def benchmark_embedding():
    """Benchmark embedding performance"""
    print("\n=== Embedding Performance Benchmark ===")
    
    texts = [
        "This is a sample text for embedding.",
        "Machine learning is transforming various industries.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of data for training.",
        "Artificial intelligence is becoming increasingly important in modern technology."
    ] * 10  # 50 texts total
    
    embedding_request = {
        "input": texts,
        "model": "jinaai/jina-embeddings-v2-base-code"
    }
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        async with session.post(
            f"{API_BASE_URL}/v1/embeddings",
            json=embedding_request
        ) as response:
            if response.status == 200:
                data = await response.json()
                end_time = time.time()
                
                print(f"Processed {len(texts)} texts in {end_time - start_time:.2f} seconds")
                print(f"Average time per text: {(end_time - start_time) / len(texts) * 1000:.2f} ms")
                print(f"Embedding dimension: {len(data['data'][0]['embedding'])}")
            else:
                error_data = await response.text()
                print(f"Error: {error_data}")

async def benchmark_rerank():
    """Benchmark rerank performance"""
    print("\n=== Rerank Performance Benchmark ===")
    
    query = "What are the benefits of renewable energy?"
    documents = [
        "Solar energy is a clean and renewable source of power that reduces carbon emissions.",
        "Wind power generates electricity without producing harmful pollutants.",
        "Fossil fuels are non-renewable resources that contribute to climate change.",
        "Hydroelectric power harnesses the energy of flowing water to generate electricity.",
        "Nuclear energy produces large amounts of electricity but generates radioactive waste.",
        "Geothermal energy uses heat from the Earth's core to generate power.",
        "Biomass energy comes from organic materials and is considered renewable.",
        "Coal is a fossil fuel that releases significant amounts of carbon dioxide.",
        "Natural gas burns cleaner than coal but is still a fossil fuel.",
        "Energy storage systems help manage the intermittent nature of renewable sources."
    ] * 5  # 50 documents total
    
    rerank_request = {
        "model": "jinaai/jina-reranker-m0",
        "query": query,
        "documents": documents,
        "top_k": 10,
        "return_documents": False
    }
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        async with session.post(
            f"{API_BASE_URL}/v1/rerank",
            json=rerank_request
        ) as response:
            if response.status == 200:
                data = await response.json()
                end_time = time.time()
                
                print(f"Reranked {len(documents)} documents in {end_time - start_time:.2f} seconds")
                print(f"Average time per document: {(end_time - start_time) / len(documents) * 1000:.2f} ms")
                print(f"Top 5 scores: {[f\"{r['relevance_score']:.4f}\" for r in data['results'][:5]]}")
            else:
                error_data = await response.text()
                print(f"Error: {error_data}")

async def main():
    """Main test function"""
    print("Starting API tests...")
    print("Make sure the API server is running on http://localhost:8000")
    print("You can start it with: python server.py")
    
    try:
        await test_api()
        await benchmark_embedding()
        await benchmark_rerank()
    except aiohttp.ClientConnectorError:
        print("\nError: Could not connect to API server.")
        print("Please make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
