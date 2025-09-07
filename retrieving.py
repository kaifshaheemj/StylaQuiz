from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
load_dotenv()
import os
qdrant_client = QdrantClient(
    url=os.getenv("URL"),
    api_key=os.getenv("API"),
)

print(qdrant_client.get_collections())

def reranking_search_batch(query_batch,
                           collection_name,
                           FilterList,
                           FilterColor=None,
                           search_limit=5,
                           prefetch_limit=200
                           ):
    filter_ = None

    print("ON Searching:...")
    print("Query batch:", query_batch)
    print("Collection name:", collection_name)
    print("FiltersList:", FilterList)
    print("FilterColor:",FilterColor)
    if FilterList:
        conditions = [
            models.FieldCondition(
                key="subcategory",
                match=models.MatchAny(any=FilterList),
            )
        ]
        
        if FilterColor:
            conditions.append(
                models.FieldCondition(
                    key="color",
                    match=models.MatchAny(any=FilterColor),
                )
            )
        
        filter_ = models.Filter(should=conditions)

    search_queries = [
        models.QueryRequest(
            query=query,
            prefetch=[
                models.Prefetch(
                    query=query,
                    limit=prefetch_limit,
                    using="mean_pooling_columns"
                ),
                models.Prefetch(
                    query=query,
                    limit=prefetch_limit,
                    using="mean_pooling_rows"
                ),
            ],
            filter=filter_,
            limit=search_limit,
            with_payload=True,
            with_vector=False,
            using="original"
        ) for query in query_batch
    ]

    response = qdrant_client.query_batch_points(
        collection_name=collection_name,
        requests=search_queries
    )

    if all(not res.points for res in response):
        search_queries = [
            models.QueryRequest(
                query=query,
                prefetch=[
                    models.Prefetch(
                        query=query,
                        limit=prefetch_limit,
                        using="mean_pooling_columns"
                    ),
                    models.Prefetch(
                        query=query,
                        limit=prefetch_limit,
                        using="mean_pooling_rows"
                    ),
                ],
                filter=None,
                limit=search_limit,
                with_payload=True,
                with_vector=False,
                using="original"
            ) for query in query_batch
        ]
        response = qdrant_client.query_batch_points(
            collection_name=collection_name,
            requests=search_queries
        )

    return response
