import json
import os
from typing import Dict, List, Tuple, Iterator
import hashlib
import logging

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ray
import torch

logger = logging.getLogger(__name__)
client = QdrantClient(host='qdrant', port=6333)
scroll_result, next_page_offset = client.scroll(
            collection_name='coliee-test',
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="document_name", 
                                          match=models.MatchValue(value='002145.txt')),
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
            offset=None
        )
import pdb; pdb.set_trace()