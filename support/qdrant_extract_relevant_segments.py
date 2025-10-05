"""
Script para extrair segmentos relevantes do Qdrant e gerar um arquivo JSON
com a estrutura especificada para avaliação de tarefas de recuperação legal.
"""
import random
import argparse
import json
import logging
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, ScrollRequest

random.seed(42)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QdrantSegmentExtractor:
    """
    Classe para extrair segmentos relevantes do Qdrant e gerar um arquivo JSON
    estruturado para avaliação de tarefas de recuperação legal.
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
        Inicializa o QdrantSegmentExtractor.
        
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
    
    def get_relevant_segments(self, document_name: str, batch_size: int = 100) -> List[Dict]:
        """
        Recupera todos os segmentos relevantes para um documento específico.
        
        Args:
            document_name: Nome do documento para recuperar segmentos
            batch_size: Tamanho do lote para operações de scroll
            
        Returns:
            Lista de segmentos relevantes
        """
        logger.debug(f"Recuperando segmentos relevantes para o documento: {document_name}.txt")
        
        # Criar filtro para buscar segmentos relevantes do documento específico
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="document_name",
                    match=MatchValue(value=f"{document_name}.txt")
                ),
                FieldCondition(
                    key="relevance",
                    match=MatchValue(value=True)
                )
            ]
        )
        
        segments = []
        offset = None
        
        # Usar scroll para buscar todos os resultados em lotes
        while True:
            batch_results, next_page_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=batch_size,
                with_payload=True,
                with_vectors=False,
                offset=offset
            )
            
            segments.extend(batch_results)

            if not batch_results:
                break
                        
            if next_page_offset is None:
                break
            
            offset = next_page_offset
        
        # Ordenar segmentos por posição no documento
        segments_sorted = sorted(segments, key=lambda x: x.payload.get("position", 0))
        
        logger.debug(f"Recuperados {len(segments_sorted)} segmentos para {document_name}")
        return segments_sorted
    
    def extract_texts_from_segments(self, segments: List[Dict]) -> List[str]:
        """
        Extrai o texto de uma lista de segmentos.
        
        Args:
            segments: Lista de segmentos do Qdrant
            
        Returns:
            Lista de textos dos segmentos
        """
        return [segment.payload.get("text", "") for segment in segments]
    
    def generate_data_for_label_pairs(self, labels_file: str, output_file: str, negative_samples_file: str = None) -> None:
        """
        Gera um arquivo JSON com segmentos relevantes para cada par query-candidato no arquivo de labels.
        
        Args:
            labels_file: Caminho para o arquivo de labels (formato task1_test_labels_2024.json)
            output_file: Caminho para o arquivo de saída
            negative_samples_file: Caminho opcional para um arquivo com amostras negativas (label=0)
        """
        logger.info(f"Gerando dados para pares de labels do arquivo: {labels_file}")
        
        # Carregar arquivo de labels
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
        
        # Criar estrutura para os dados processados
        processed_data = []
        
        # Processar cada par query-candidato
        for q_doc, c_docs in tqdm(labels_data.items(), desc="Processando pares query-candidato"):
            q_name = q_doc.replace(".txt", "")
            
            # Obter segmentos relevantes para o documento query
            q_segments = self.get_relevant_segments(q_name)
            q_texts = self.extract_texts_from_segments(q_segments)
            
            if not q_texts:
                logger.warning(f"Nenhum segmento relevante encontrado para query: {q_name}")
                continue
            
            # Processar cada documento candidato
            for c_doc in c_docs:
                c_name = c_doc.replace(".txt", "")
                
                # Obter segmentos relevantes para o documento candidato
                c_segments = self.get_relevant_segments(c_name)
                c_texts = self.extract_texts_from_segments(c_segments)
                
                if not c_texts:
                    logger.warning(f"Nenhum segmento relevante encontrado para candidato: {c_name}")
                    continue
                
                # Criar entrada no formato desejado
                entry = {
                    "guid": f"{q_name}_{c_name}",
                    "q_paras": q_texts,
                    "c_paras": c_texts,
                    "label": 1  # Todos os pares no arquivo são relevantes
                }
                
                processed_data.append(entry)
        
        # Adicionar amostras negativas de outro arquivo, se fornecido
        if negative_samples_file:
            logger.info(f"Carregando amostras negativas do arquivo: {negative_samples_file}")
            negative_samples = []
            
            try:
                with open(negative_samples_file, 'r') as f:
                    for line in tqdm(f):
                        try:
                            sample = json.loads(line)
                            # Verificar se o arquivo já contém o campo "label"
                            if "label" in sample:
                                # Verificar se o label é 0 (negativo)
                                if sample["label"] == 0:
                                    q_name, c_name = sample["guid"].split("_")
                                    c_segments = self.get_relevant_segments(c_name)
                                    c_texts = self.extract_texts_from_segments(c_segments)
                                    q_segments = self.get_relevant_segments(q_name)
                                    q_texts = self.extract_texts_from_segments(q_segments)
                                    sample["q_paras"] = q_texts
                                    sample["c_paras"] = c_texts
                                    negative_samples.append(sample)
                            else:
                                # Se não houver campo label, assumir que é um exemplo negativo 
                                # e adicionar o campo label=0
                                sample["label"] = 0
                                negative_samples.append(sample)
                        except json.JSONDecodeError:
                            logger.warning(f"Erro ao decodificar linha JSON: {line}")
                            continue
                
                logger.info(f"Carregadas {len(negative_samples)} amostras negativas")
                processed_data.extend(negative_samples)
            except FileNotFoundError:
                logger.error(f"Arquivo de amostras negativas não encontrado: {negative_samples_file}")
        
        # Embaralhar os dados para melhorar a distribuição de amostras positivas e negativas
        random.shuffle(processed_data)
        
        # Salvar resultado em formato JSONL
        with open(output_file, 'w') as f:
            for item in processed_data:
                f.write(json.dumps(item) + "\n")  # JSONL format
        
        # Resumo de estatísticas
        pos_count = sum(1 for item in processed_data if item["label"] == 1)
        neg_count = sum(1 for item in processed_data if item["label"] == 0)
        
        logger.info(f"Dados gerados com sucesso no arquivo: {output_file}")
        logger.info(f"Total de pares processados: {len(processed_data)}")
        logger.info(f"Pares positivos (label=1): {pos_count}")
        logger.info(f"Pares negativos (label=0): {neg_count}")


def main():
    """
    Função principal para executar o extrator a partir da linha de comando.
    """
    parser = argparse.ArgumentParser(
        description="Extrai segmentos relevantes do Qdrant e gera um arquivo JSON estruturado"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="coliee-test",
        help="Nome da coleção no Qdrant (padrão: coliee-test)"
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
        "--labels-file",
        type=str,
        required=True,
        help="Caminho para o arquivo de labels (ex: task1_test_labels_2024.json)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Caminho para o arquivo de saída"
    )
    parser.add_argument(
        "--negative-samples-file",
        type=str,
        help="Caminho opcional para um arquivo com amostras negativas (label=0)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Tamanho do lote para operações do Qdrant (padrão: 100)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ativar logs de debug"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    extractor = QdrantSegmentExtractor(
        collection_name=args.collection,
        host=args.host,
        port=args.port,
        grpc_port=args.grpc_port
    )
    
    extractor.generate_data_for_label_pairs(
        labels_file=args.labels_file,
        output_file=args.output_file,
        negative_samples_file=args.negative_samples_file
    )
    
    print(f"Processamento concluído. Arquivo gerado: {args.output_file}")


if __name__ == "__main__":
    main()
