from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk
nltk.download('punkt')


class SumySelection:
    LANGUAGE = "english"

    def __init__(self, percentual=0.5):
        self.percentual = percentual
        stemmer = Stemmer(SumySelection.LANGUAGE)
        self.summarizer = Summarizer(stemmer)
        self.summarizer.stop_words = get_stop_words(SumySelection.LANGUAGE)

    def forward(self, data):
        for row in data:
            c_indices, c_paras = self.select_paras(row["c_paras"])
            q_indices, q_paras = self.select_paras(row["q_paras"])
            row["c_paras"] = c_paras
            row["q_paras"] = q_paras
            row["c_selected_indices"] = c_indices
            row["q_selected_indices"] = q_indices

        return data

    def select_paras(self, paras):
        parser = PlaintextParser.from_string(
            " ".join(paras), Tokenizer(SumySelection.LANGUAGE)
        )
        num_sentences = max(1, int(len(paras) * self.percentual))
        summary = self.summarizer(parser.document, num_sentences)
        selected_paras = [str(sentence) for sentence in summary]
        indices = [paras.index(para) for para in selected_paras]
        return indices, selected_paras
