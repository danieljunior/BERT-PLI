import torch
import torch.nn as nn
from model.nlp.MultiHeadAttention import MultiHeadAttention
from transformers import BertTokenizer, BertModel

class LearnableSequenceSelector(nn.Module):
    """
    Uma camada que aprende a selecionar um subconjunto de `num_to_select` sequências
    de um tensor de entrada de forma dinâmica e diferenciável.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_to_select: int,
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim deve ser divisível por num_heads")

        self.num_to_select = num_to_select

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()  # Coloca o modelo BERT em modo de avaliação
        # Mecanismo de pontuação baseado em auto-atenção para capturar relações contextuais
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout=dropout, bias=bias)

        # Camada linear para projetar a saída da atenção em um score escalar para cada sequência
        self.score_projector = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        Realiza a passagem para a frente, gerando scores e selecionando as k melhores sequências.

        Args:
            x (torch.Tensor): Tensor de entrada com shape [batch_size, num_sequences, embed_dim].

        Returns:
            tuple:
            - selected_sequences (torch.Tensor): Tensor com as sequências selecionadas.
            - selection_indices (torch.Tensor): Índices das sequências selecionadas.
            - scores (torch.Tensor): Scores brutos gerados para cada sequência (usado na loss).
        """
        x = self.process_data(x)
        # 1. Gerar scores usando auto-atenção
        attn_output, _ = self.attention(x, x, x)
        # attn_output = self.attention(x, x, x)

        # 2. Projetar a saída da atenção para obter um score para cada sequência
        scores = self.score_projector(attn_output).squeeze(-1) # Shape: [batch_size, num_sequences]

        # 3. Aplicar o truque Gumbel-Top-K para seleção estocástica e diferenciável
        # Adiciona ruído Gumbel aos scores para permitir a exploração durante o treinamento.
        # A amostragem Gumbel é uma forma de extrair amostras de uma distribuição categórica.
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-20) + 1e-20)
        perturbed_scores = scores + gumbel_noise

        # 4. Selecionar os k melhores índices
        # A operação topk em si não é diferenciável em relação aos índices, mas o gradiente
        # fluirá de volta para os scores que produziram esses índices (comportamento de gradiente surrogato).
        _, selection_indices = torch.topk(perturbed_scores, self.num_to_select, dim=1)

        # 5. Coletar as sequências selecionadas usando os índices
        # A função `gather` é diferenciável em relação aos valores de `x`.
        batch_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1)
        selected_sequences = x[batch_indices, selection_indices]

        return selected_sequences, selection_indices, scores
    
    def process_data(self, data):
        """
        Processa os dados de entrada para garantir que estejam no formato correto.

        Args:
            data (torch.Tensor): Tensor de entrada com shape [batch_size, num_sequences, embed_dim].

        Returns:
            torch.Tensor: Tensor processado.
        """
        res = []
        for example in data:
            tokens_ids, segments_ids = self.tokenize_paras(example)
            embs = self.get_embeddings(tokens_ids, segments_ids)
            res.append(embs)
        return torch.stack(res).to('cuda')  # Shape: [batch_size, num_paras, embed_dim]
    
    def tokenize_paras(self, paras):
        c_tokens = []
        c_segments = []
        for segment in paras:
            tokenized = self.tokenizer.tokenize(segment)
            tokenized = ['[CLS]'] + self.tokenizer.tokenize(segment)
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokenized)[:512]
            pad_len = 512 - len(tokens_ids)
            if pad_len > 0:
                tokens_ids += [0] * pad_len
            segments_ids = [0] * 512
            c_tokens.append(tokens_ids)
            c_segments.append(segments_ids)

        if len(c_tokens) > 200:
            c_tokens = c_tokens[:200]
            c_segments = c_segments[:200]
        elif len(c_tokens) < 200:
            pad_len = 200 - len(c_tokens)
            zero_tensor = [0.0] * 512
            for _ in range(pad_len):
                c_tokens.append(zero_tensor)
                c_segments.append(zero_tensor)
        return torch.tensor(c_tokens), torch.tensor(c_segments)
    
    def get_embeddings(self, tokens, segments):
        res = []
        for tokens_tensor, segments_tensors in zip(tokens, segments):
            with torch.no_grad():
                _, _, hidden_states = self.bert_model(
                    tokens_tensor.unsqueeze(0).long().to('cuda'),
                    token_type_ids=segments_tensors.unsqueeze(0).long().to('cuda'),
                    output_hidden_states=True,
                    return_dict=False,
                )
            # Sum the CLS token (first token) from the last 4 layers into a single vector.
            last_four = hidden_states[-4:]
            if last_four[0].dim() == 3:
                # shape (batch, seq_len, hidden) -> take batch 0, token 0
                cls_vectors = [layer[0, 0, :] for layer in last_four]
            elif last_four[0].dim() == 2:
                # shape (seq_len, hidden) -> take token 0
                cls_vectors = [layer[0, :] for layer in last_four]
            else:
                raise RuntimeError(f"Unexpected encoded_layer shape: {last_four[0].shape}")

            # Sum across the four layers and keep a batch-like dim so .squeeze(0) works later
            embeddings = torch.stack(cls_vectors, dim=0).sum(dim=0).unsqueeze(0)
            res.append(embeddings.detach().cpu())
        
        return torch.stack(res).squeeze(1)  # Shape: [num_paras, embed_dim]
