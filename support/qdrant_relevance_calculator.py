"""
Script para identificar e marcar documentos relevantes com base em quintis de densidade no Qdrant.
"""
import logging
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, ScrollRequest
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QdrantRelevanceCalculator:
    """
    Classe para identificar documentos relevantes com base em quintis de densidade no Qdrant.
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
        Inicializa o QdrantRelevanceCalculator.
        
        Args:
            collection_name: Nome da coleção no Qdrant
            host: Endereço do host do Qdrant
            port: Porta HTTP do Qdrant
            grpc_port: Porta gRPC do Qdrant
            prefer_grpc: Se deve preferir gRPC em vez de HTTP
        """
        self.collection_name = collection_name
        self.client = QdrantClient(
            host=host, 
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc
        )
    
    def get_all_density_points(self, batch_size: int = 100) -> List[Dict]:
        """
        Recupera todos os pontos com valores de densidade do Qdrant.
        
        Args:
            batch_size: Tamanho do lote para operações de rolagem
            
        Returns:
            Lista de pontos com seus IDs, valores de densidade e payloads
        """
        logger.info(f"Recuperando todos os pontos com valores de densidade")
        
        points = []
        offset = None
        
        while True:
            batch_points, next_page_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                with_payload=True,
                with_vectors=False,
                offset=offset
            )
            
            points.extend(batch_points)
            
            if next_page_offset is None:
                break  # Não há mais páginas
            
            if len(batch_points) < batch_size:
                break
            
            offset = next_page_offset
        
        logger.info(f"Recuperados {len(points)} pontos com valores de densidade")
        return points
    
    def calculate_density_quintiles(self, points: List[Dict]) -> Tuple[float, float]:
        """
        Calcula os quintis de densidade dos pontos.
        
        Args:
            points: Lista de pontos com valores de densidade
            
        Returns:
            Tuple contendo o primeiro e o último quintil (0.2 e 0.8)
        """
        if not points:
            logger.warning("Nenhum ponto para calcular quintis")
            return (0.0, 0.0)
        
        densities = [point.payload["density"] for point in points]
        first_quintile = np.quantile(densities, 0.2)
        last_quintile = np.quantile(densities, 0.8)
        
        logger.info(f"Primeiro quintil: {first_quintile}, Último quintil: {last_quintile}")
        return (first_quintile, last_quintile)
    
    def filter_points_by_quintiles(
        self, 
        points: List[Dict], 
        first_quintile: float, 
        last_quintile: float,
        between: bool = True
    ) -> List[Dict]:
        """
        Filtra pontos com densidade entre o primeiro e o último quintil.
        
        Args:
            points: Lista de pontos com valores de densidade
            first_quintile: Valor do primeiro quintil
            last_quintile: Valor do último quintil
            
        Returns:
            Lista de pontos filtrados
        """
        if between:
            filtered_points = [
                point for point in points 
                if first_quintile <= point.payload["density"] <= last_quintile
            ]
            logger.info(f"Filtrados {len(filtered_points)} pontos entre os quintis")
        else:
            filtered_points = [
                point for point in points 
                if point.payload["density"] < first_quintile or point.payload["density"] > last_quintile
            ]
            logger.info(f"Filtrados {len(filtered_points)} pontos fora dos quintis")

        return filtered_points
    
    def update_points_relevance(
        self, 
        points: List[Dict],
        relevance_value: bool = True
    ) -> int:
        """
        Atualiza os pontos no Qdrant com a informação de relevância.
        
        Args:
            points: Lista de dicionários com IDs de pontos e payloads
            relevance_value: Valor de relevância a ser definido
            
        Returns:
            Número de pontos atualizados
        """
        logger.info(f"Atualizando {len(points)} pontos com informação de relevância")
        
        # Atualizar em lotes para evitar sobrecarregar o servidor
        batch_size = 100
        total_updated = 0
        
        for i in tqdm(range(0, len(points), batch_size), desc="Atualizando pontos"):
            batch = points[i:i+batch_size]
            
            # Atualizar payloads
            for point in batch:
                point.payload["relevance"] = relevance_value
            
            # Criar atualizações de payload
            payload_updates = [
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload=point.payload,
                        points=[point.id],
                    )
                )
                for point in batch
            ]
            
            # Atualizar pontos
            if payload_updates:
                self.client.batch_update_points(
                    collection_name=self.collection_name,
                    update_operations=payload_updates
                )
                total_updated += len(payload_updates)
        
        logger.info(f"Atualizados com sucesso {total_updated} pontos com informação de relevância")
        return total_updated
    
    def process_density_quintiles(
        self, 
        batch_size: int = 100,
    ) -> int:
        """
        Processa quintis de densidade e atualiza pontos relevantes.
        
        Args:
            batch_size: Tamanho do lote para operações do Qdrant
            relevance_value: Valor de relevância a definir
            
        Returns:
            Número de pontos atualizados
        """
        try:
            # Obter todos os pontos com densidades
            points = self.get_all_density_points(batch_size=batch_size)
            
            if not points:
                logger.warning("Nenhum ponto encontrado")
                return 0
            
            # Calcular quintis
            first_quintile, last_quintile = self.calculate_density_quintiles(points)
            
            # Filtrar pontos entre os quintis (relevantes)
            relevant_points = self.filter_points_by_quintiles(
                points, first_quintile, last_quintile, between=True
            )
            
            # Filtrar pontos fora dos quintis (não relevantes)
            non_relevant_points = self.filter_points_by_quintiles(
                points, first_quintile, last_quintile, between=False
            )
            
            total_updated = 0
            
            if relevant_points:
                total_updated += self.update_points_relevance(relevant_points, True)
            
            if non_relevant_points:
                total_updated += self.update_points_relevance(non_relevant_points, False)
            
            return total_updated
            
        except Exception as e:
            logger.error(f"Erro ao processar quintis de densidade: {e}")
            raise


def main():
    """
    Função principal para executar o calculador de relevância a partir da linha de comando.
    """
    parser = argparse.ArgumentParser(
        description="Calcular e atualizar relevância para documentos no Qdrant com base em quintis de densidade"
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="coliee-train",
        help="Nome da coleção no Qdrant (padrão: coliee-train)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="qdrant",
        help="Host do Qdrant (padrão: qdrant)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=6333,
        help="Porta HTTP do Qdrant (padrão: 6333)"
    )
    parser.add_argument(
        "--grpc-port", 
        type=int, 
        default=6334,
        help="Porta gRPC do Qdrant (padrão: 6334)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=100,
        help="Tamanho do lote para operações do Qdrant (padrão: 100)"
    )
    
    args = parser.parse_args()
    
    calculator = QdrantRelevanceCalculator(
        collection_name=args.collection,
        host=args.host,
        port=args.port,
        grpc_port=args.grpc_port
    )
    
    total_updated = calculator.process_density_quintiles(
        batch_size=args.batch_size,
    )
    
    print(f"Atualizados com sucesso {total_updated} pontos na coleção: {args.collection}")


if __name__ == "__main__":
    main()
