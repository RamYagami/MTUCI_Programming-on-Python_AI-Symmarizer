import logging
from collections import Counter
from typing import List

import nltk
import numpy as np
from langdetect import detect
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.sum_basic import SumBasicSummarizer

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

logger = logging.getLogger(__name__)


class TextSummarizer:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.stop_words = set(stopwords.words("english"))
        self._model = SentenceTransformer(
            "./models/sentence-transformers/all-MiniLM-L6-v2"
        )

    def detect_language(self, text: str) -> str:
        try:
            detected_lang = detect(text)
            return "en" if detected_lang == "en" else "ru"
        except Exception:
            return "en"

    def extractive_summarize(self, text: str, lang: str, length: str) -> str:
        try:
            mapping = {"short": 2, "medium": 4, "long": 8}
            sentence_count = mapping.get(length, 4)

            parser = PlaintextParser.from_string(text, Tokenizer("english"))

            summarizer = SumBasicSummarizer()

            summary = summarizer(parser.document, sentence_count)

            return " ".join([str(sent) for sent in summary])

        except Exception as e:
            logger.error(f"Extractive summarization failed: {e}")
            return f"Extractive summarization failed: {str(e)}"

    def extractive_hierarchical_summarize(
        self, text: str, lang: str, length: str
    ) -> str:
        """Return concise, thematic, non-redundant bullet-journal summary."""
        try:
            blocks = self._split_into_thematic_blocks(text, length=length)
            if not blocks:
                return "- Summary\n  - (empty)"

            # Map length to desired number of TOPICS (not blocks!)
            max_topics = {"short": 2, "medium": 4, "long": 8}
            target_n = max_topics.get(length, 4)

            if len(blocks) <= target_n:
                # Use all blocks
                selected_blocks = blocks
            else:
                # Step 1: Encode each block (as avg of sentence embeddings)
                block_embs = []
                for block in blocks:
                    sents = sent_tokenize(block)
                    if not sents:
                        block_embs.append(np.zeros(384))
                        continue
                    sent_embs = self._model.encode(sents, convert_to_numpy=True)
                    block_embs.append(np.mean(sent_embs, axis=0))
                block_embs = np.array(block_embs)  # shape: (n_blocks, 384)

                # Step 2: Compute global centroid of entire text
                global_centroid = np.mean(block_embs, axis=0)

                # Step 3: Score each block by:
                #   - similarity to global centroid (importance)
                #   - length (longer = more content)
                scores = []
                for i, block in enumerate(blocks):
                    sim_to_global = cosine_similarity(
                        block_embs[i].reshape(1, -1), global_centroid.reshape(1, -1)
                    )[0][0]
                    length_score = min(len(block) / 1000.0, 1.0)  # normalize
                    score = 0.7 * sim_to_global + 0.3 * length_score
                    scores.append((score, i))

                # Step 4: Select top-N most important blocks
                scores.sort(reverse=True)
                selected_indices = sorted([idx for _, idx in scores[:target_n]])
                selected_blocks = [blocks[i] for i in selected_indices]

            # Step 5: Build bullet-journal output
            lines = []
            for block in selected_blocks:
                key_sent = self._extract_key_sentence(block)
                topic = self._extract_key_phrase(block)
                lines.append(f"- {topic}\n  - {key_sent}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Hierarchical summarization failed: {e}")
            # Fallback: global key sentence
            key_sent = self._extract_key_sentence(text)
            topic = self._extract_key_phrase(text)
            return f"- {topic}\n  - {key_sent}"

    def abstractive_summarize(self, text: str, lang: str, length: str) -> str:
        return "Abstractive summarization not available yet"

    def _extract_key_sentence(self, text: str) -> str:
        """Return the most central (representative) sentence using embedding centrality."""
        sentences = sent_tokenize(text.strip())
        if len(sentences) == 1:
            return sentences[0]

        try:
            embeddings = self._model.encode(sentences, convert_to_numpy=True)
            # Compute centroid of the block
            centroid = np.mean(embeddings, axis=0)
            # Find sentence closest to centroid
            similarities = cosine_similarity(
                embeddings, centroid.reshape(1, -1)
            ).flatten()
            best_idx = np.argmax(similarities)
            return sentences[best_idx]
        except Exception as e:
            logger.warning(f"Key sentence extraction failed: {e}")
            return sentences[0]  # fallback to first

    def _extract_key_phrase(self, text: str, max_words: int = 3) -> str:
        """Extract concise topic from text using meaningful words (nouns, verbs, adjectives, adverbs)."""
        try:
            tokens = word_tokenize(text.lower())

            tokens = [w for w in tokens if w.isalpha() and len(w) > 2]
            if not tokens:
                return "Main point"

            tagged = pos_tag(tokens)

            meaningful_pos = {
                "NN",
                "NNS",
                "NNP",
                "NNPS",
                "VB",
                "VBD",
                "VBG",
                "VBN",
                "VBP",
                "VBZ",
                "JJ",
                "JJR",
                "JJS",
                "RB",
                "RBR",
                "RBS",
            }

            meaningful_words = [
                word
                for word, pos in tagged
                if pos in meaningful_pos and word not in self.stop_words
            ]

            if not meaningful_words:
                nouns = [
                    word
                    for word, pos in tagged
                    if pos.startswith("NN") and word not in self.stop_words
                ]

                if nouns:
                    meaningful_words = nouns
                else:
                    meaningful_words = [
                        w for w in tokens if w not in self.stop_words
                    ] or tokens

            freq = Counter(meaningful_words)

            phrase = " ".join([w for w, _ in freq.most_common(max_words)])

            return phrase.capitalize()

        except BaseException:
            return "Main point"

    def _split_into_thematic_blocks(
        self, text: str, length: str = "medium"
    ) -> List[str]:
        cfg = {
            "short": {"window_size": 5, "min_similarity": 0.6},
            "medium": {"window_size": 3, "min_similarity": 0.7},
            "long": {"window_size": 1, "min_similarity": 0.8},
        }

        params = cfg[length]

        return self._semantic_split(text, **params)

    def _semantic_split(
        self, text: str, window_size: int = 5, min_similarity: float = 0.7
    ) -> List[str]:
        """
        Split text into thematic blocks using window-based semantic coherence.

        Args:
            text: Input plain text.
            window_size: Number of sentences in left/right context (default=3).
            min_similarity: Threshold below which a topic boundary is detected.

        Returns:
            List of thematic blocks (strings).
        """
        sentences = sent_tokenize(text.strip())
        if len(sentences) <= window_size:
            return [text.strip()]

        # Encode all sentences
        embeddings = self._model.encode(sentences, convert_to_numpy=True)

        # Candidate boundary positions (after sentence i)
        boundaries = []

        # We need at least `window_size` on both sides
        for i in range(window_size - 1, len(sentences) - window_size):
            # Left context: [i - window_size + 1 ... i]
            left_start = i - window_size + 1
            left_emb = np.mean(embeddings[left_start : i + 1], axis=0)

            # Right context: [i+1 ... i + window_size]
            right_end = i + window_size + 1
            right_emb = np.mean(embeddings[i + 1 : right_end], axis=0)

            sim = cosine_similarity(
                left_emb.reshape(1, -1),
                right_emb.reshape(1, -1),
            )[0][0]

            if sim < min_similarity:
                boundaries.append(i)

        # Merge nearby boundaries (avoid fragmentation)
        if boundaries:
            filtered = [boundaries[0]]
            for b in boundaries[1:]:
                if (
                    b - filtered[-1] > window_size // 2
                ):  # min distance between boundaries
                    filtered.append(b)
            boundaries = filtered

        # Build blocks
        blocks = []
        start = 0
        for b in boundaries:
            blocks.append(" ".join(sentences[start : b + 1]))
            start = b + 1
        blocks.append(" ".join(sentences[start:]))

        # Post-process: merge tiny blocks (< 2 sentences) into neighbors
        final_blocks = []
        i = 0
        while i < len(blocks):
            current = blocks[i]
            if len(sent_tokenize(current)) < 2 and final_blocks:
                # Merge with previous
                final_blocks[-1] += " " + current
            else:
                final_blocks.append(current)
            i += 1

        return [b.strip() for b in final_blocks if b.strip()]
