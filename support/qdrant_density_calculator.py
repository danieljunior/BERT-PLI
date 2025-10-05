"""
Script to calculate density for documents in Qdrant based on k-nearest neighbors.
"""
import logging
from typing import List, Dict, Optional, Union
import numpy as np
from tqdm import tqdm
import torch

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, ScrollRequest
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QdrantDensityCalculator:
    """
    A class for calculating density for documents in Qdrant based on k-nearest neighbors.
    """
    
    def __init__(
        self, 
        collection_name: str, 
        host: str = "localhost", 
        port: int = 6333, 
        grpc_port: int = 6334,
        prefer_grpc: bool = True
    ):
        """
        Initialize the QdrantDensityCalculator.
        
        Args:
            collection_name: Name of the Qdrant collection
            host: Qdrant host address
            port: Qdrant HTTP port
            grpc_port: Qdrant gRPC port
            prefer_grpc: Whether to prefer gRPC over HTTP
        """
        self.collection_name = collection_name
        self.client = QdrantClient(
            host=host, 
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc
        )
    
    def get_document_points(self, document_name: str) -> List[Dict]:
        """
        Retrieve all points for a specific document from the Qdrant collection.
        
        Args:
            document_name: Name of the document to retrieve points for
            
        Returns:
            List of points with their IDs, vectors, and payloads
        """
        logger.info(f"Retrieving points for document: {document_name}")
        
        points = []
        offset = None
        limit = 100  # Batch size for scrolling
        
        while True:
            batch_points, next_page_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_name",
                            match=MatchValue(value=document_name)
                        ),
                       models.IsEmptyCondition(is_empty=models.PayloadField(key="density"))
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=True,
                offset=offset
            )
            
            points.extend([
                {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                } for point in batch_points
            ])
            
            if next_page_offset is None:
                break  # No more pages
            
            if len(batch_points) < limit:
                break

            offset = next_page_offset
        
        logger.info(f"Retrieved {len(points)} points for document: {document_name}")
        return points
    
    def calculate_density(
        self, 
        document_name: str, 
        k: int = 5, 
        batch_size: int = 100
    ) -> List[Dict]:
        """
        Calculate density for all points in a document based on k-nearest neighbors.
        
        Args:
            document_name: Name of the document
            k: Number of nearest neighbors to consider for density calculation
            batch_size: Size of batches for Qdrant search operations
            
        Returns:
            List of point IDs with their calculated densities
        """
        logger.info(f"Calculating density for document: {document_name} with k={k}")
        
        # Get all points for the document
        points = self.get_document_points(document_name)
        
        if not points:
            logger.warning(f"No points found for document: {document_name}")
            return []
        
        # Extract vectors and IDs
        vectors = [point["vector"] for point in points]
        ids = [point["id"] for point in points]
        payloads= [point["payload"] for point in points]
        
        # Calculate densities in batches
        densities = []
        
        for i in tqdm(range(0, len(vectors), batch_size), desc="Processing batches", leave=False):
            batch_vectors = vectors[i:i+batch_size]
            search_queries = [
                models.QueryRequest(query=vector, limit=k+1)
                for vector in batch_vectors
            ]
            search_results = self.client.query_batch_points(collection_name=self.collection_name, 
                                           requests=search_queries,)

            # Calculate density for each point
            for j, results in enumerate(search_results):
                # Skip the first result (the point itself)
                distances = [result.score for result in results.points[1:k+1]]
                
                # If we have fewer than k neighbors, we adjust
                if len(distances) < k:
                    logger.warning(f"Point {ids[i+j]} has fewer than {k} neighbors: {len(distances)}")
                    if not distances:
                        densities.append(0.0)  # Default density for isolated points
                        continue
                
                # Calculate density (average similarity, where similarity = 1 - cosine distance)
                # For Qdrant, scores are already similarities if using cosine distance
                avg_similarity = sum(distances) / len(distances)
                densities.append(avg_similarity)
        
        # Create result list with point IDs and densities
        result = [{"id": point_id, "density": density, "payload": payload} 
                  for point_id, density, payload in zip(ids, densities, payloads)]
        
        logger.info(f"Calculated densities for {len(result)} points")
        return result
    
    def update_points_with_density(
        self, 
        density_data: List[Dict]
    ) -> int:
        """
        Update points in Qdrant with calculated density values.
        
        Args:
            density_data: List of dictionaries with point IDs and densities
            
        Returns:
            Number of points updated
        """
        logger.info(f"Updating {len(density_data)} points with density information")
        
        # Update in batches to avoid overwhelming the server
        batch_size = 100
        total_updated = 0
        
        for i in tqdm(range(0, len(density_data), batch_size), desc="Updating points"):
            batch = density_data[i:i+batch_size]
            payloads = [item["payload"] for item in batch]
            for p, d in zip(payloads, batch):
                p['density'] = d['density'] 
            # Create payload updates
            payload_updates = [
                {
                    "id": item["id"],
                    "payload": item["payload"]
                }
                for item in batch
            ]
            
            # Update points
            if payload_updates:
                payload_operations = [ 
                    models.SetPayloadOperation(
                        set_payload=models.SetPayload(
                            payload=payload["payload"],
                            points=[payload["id"]],
                        )
                    )
                    for payload in payload_updates
                ]
                
                self.client.batch_update_points(
                    collection_name=self.collection_name,
                    update_operations=payload_operations
                )

                total_updated += len(payload_updates)
        
        logger.info(f"Successfully updated {total_updated} points with density information")
        return total_updated
    
    def process_document(
        self, 
        document_name: str, 
        k: int = 5, 
        batch_size: int = 100
    ) -> int:
        """
        Process a document by calculating densities for its points and updating them.
        
        Args:
            point: Qdrant point representing the document
            k: Number of nearest neighbors to consider for density calculation
            batch_size: Size of batches for Qdrant operations
            
        Returns:
            Number of points updated
        """
        try:
            # Calculate densities
            density_data = self.calculate_density(
                document_name=document_name,
                k=k,
                batch_size=batch_size
            )
            
            if not density_data:
                return 0
            
            # Update points with calculated densities
            return self.update_points_with_density(density_data)
            
        except Exception as e:
            logger.error(f"Error processing document {document_name}: {e}")
            raise


def main():
    """
    Main function to run the density calculator from command line.
    """
    parser = argparse.ArgumentParser(
        description="Calculate and update density for documents in Qdrant"
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="coliee-train",
        help="Name of the Qdrant collection (default: case_pairs)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="qdrant",
        help="Qdrant host (default: qdrant)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=6333,
        help="Qdrant HTTP port (default: 6333)"
    )
    parser.add_argument(
        "--grpc-port", 
        type=int, 
        default=6334,
        help="Qdrant gRPC port (default: 6334)"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=5,
        help="Number of nearest neighbors for density calculation (default: 5)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=10,
        help="Batch size for Qdrant operations (default: 10)"
    )
    
    args = parser.parse_args()
    
    calculator = QdrantDensityCalculator(
        collection_name=args.collection,
        host=args.host,
        port=args.port,
        grpc_port=args.grpc_port
    )
    client = QdrantClient(
            host=args.host, 
            port=args.port,
            grpc_port=args.grpc_port,
            prefer_grpc=True
    )
    
    offset_ = None
    documents_names = []

    while True:
        scroll_result, next_page_offset = client.scroll(
                collection_name=args.collection,
                offset=offset_,
                limit=args.batch_size,
                with_payload=True,
                with_vectors=False)
        documents_names.extend([p.payload['document_name'] for p in scroll_result])
        if next_page_offset is None:
            break  # No more pages
        offset_ = next_page_offset

    documents_names = list(set(documents_names))
    print(f"Found {len(documents_names)} unique documents to process.")
    for document_name in tqdm(documents_names, desc="Processing documents"):
        total_updated = calculator.process_document(
            document_name=document_name,
            k=args.k,
            batch_size=args.batch_size
        )
    
    print(f"Successfully updated points for collection: {args.collection}")


if __name__ == "__main__":
    main()
