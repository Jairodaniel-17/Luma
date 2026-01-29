from luma import Client, CollectionConfig, Distance, VectorRecord

# 1. Initialize the client
# Connects to localhost:9917 by default
client = Client()

collection_name = "test_collection"

# 2. Create a collection
try:
    client.create_collection(
        name=collection_name,
        config=CollectionConfig(dim=4, metric=Distance.DOT),
    )
    print(f"Collection '{collection_name}' created.")
except Exception as e:
    print(f"Collection might already exist: {e}")

# 3. Add vectors
operation_info = client.upsert(
    collection=collection_name,
    records=[
        VectorRecord(id=1, vector=[0.05, 0.61, 0.76, 0.74], metadata={"city": "Berlin"}),
        VectorRecord(id=2, vector=[0.19, 0.81, 0.75, 0.11], metadata={"city": "London"}),
        VectorRecord(id=3, vector=[0.36, 0.55, 0.47, 0.94], metadata={"city": "Moscow"}),
        VectorRecord(id=4, vector=[0.18, 0.01, 0.85, 0.80], metadata={"city": "New York"}),
        VectorRecord(id=5, vector=[0.24, 0.18, 0.22, 0.44], metadata={"city": "Beijing"}),
        VectorRecord(id=6, vector=[0.35, 0.08, 0.11, 0.44], metadata={"city": "Mumbai"}),
    ],
)

print(f"Upsert status: {operation_info.status}")

# 4. Run a query
search_results = client.search(
    collection=collection_name,
    vector=[0.2, 0.1, 0.9, 0.7],
    include_metadata=True,
    k=3
)

print("\nSearch Results:")
for hit in search_results:
    print(f"ID: {hit.id}, Score: {hit.score}, City: {hit.metadata.get('city')}")