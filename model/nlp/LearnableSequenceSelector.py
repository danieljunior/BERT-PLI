import torch
import torch.nn as nn


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

        # Mecanismo de pontuação baseado em auto-atenção para capturar relações contextuais
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # self.attention = CustomMultiHeadAttention(embed_dim, embed_dim, embed_dim,
                                                  # embed_dim, num_heads,
                                                  # dropout=dropout, bias=bias)

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