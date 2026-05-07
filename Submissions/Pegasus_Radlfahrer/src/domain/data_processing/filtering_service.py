import hashlib
import re
import shutil
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional, List

from datasets import Dataset, concatenate_datasets
from langdetect import DetectorFactory, detect
from tqdm import tqdm
from transformers import GPT2TokenizerFast

# Optional dependencies for high-quality pipelines
try:
    import ftfy
except ImportError:
    ftfy = None

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    MinHash = None
    MinHashLSH = None

DetectorFactory.seed = 42


class DatasetFilteringService:
    """
    A robust, modular data filtering pipeline with checkpointing and progress tracking.
    Updated to match PikoGPT ultra-picky pipeline requirements.
    """

    def __init__(
        self,
        target_lang: str = "en",
        min_coherence_score: float = 0.60,
        min_ttr: float = 0.45,
        max_ttr: float = 0.90,
        min_alpha_ratio: float = 0.75,
        max_symbol_ratio: float = 0.35,
        min_avg_line_length: int = 30,
        min_tokenizer_efficiency: float = 3.0,
        enable_near_dedup: bool = True,
        near_dedup_threshold_3gram: float = 0.80,
        near_dedup_threshold_5gram: float = 0.90,
        near_dedup_num_perm: int = 128,
    ):
        self.target_lang = target_lang
        self.min_coherence_score = min_coherence_score
        self.min_ttr = min_ttr
        self.max_ttr = max_ttr
        self.min_alpha_ratio = min_alpha_ratio
        self.max_symbol_ratio = max_symbol_ratio
        self.min_avg_line_length = min_avg_line_length
        self.min_tokenizer_efficiency = min_tokenizer_efficiency
        
        self.enable_near_dedup = enable_near_dedup
        self.near_dedup_threshold_3gram = near_dedup_threshold_3gram
        self.near_dedup_threshold_5gram = near_dedup_threshold_5gram
        self.near_dedup_num_perm = near_dedup_num_perm

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        self.artifact_keywords = [
            'accept cookies', 'privacy policy', 'terms of service', 'consent', 
            'all rights reserved', 'javascript is disabled', 'please enable javascript',
            'subscribe to our newsletter', 'read more at', 'click here',
            'manage cookie settings', 'we use cookies', 'sign in to continue', 
            'forgot password', 'skip to main content', 'read more', 
            'related articles', 'share on facebook', 'share on twitter', 'gdpr'
        ]
        self.code_keywords = ['def ', 'public static void', 'console.log', '<div>', 'select * from']
        
        self.word_re = re.compile(r"[a-z0-9']+")

    def normalize_formatting(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        if ftfy:
            text = ftfy.fix_text(text)
        # Normalize unicode characters to standard ASCII equivalents where possible
        text = unicodedata.normalize("NFKC", text)
        # Replace curly quotes and apostrophes with standard ones
        text = text.replace('‘', "'").replace('’', "'").replace('“', '"').replace('”', '"')
        # Collapse multiple spaces and newlines (keep at most \n\n)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def check_structural_coherence(self, text: str) -> float:
        """Scores text based on natural prose indicators as tested in EDA."""
        if not text or len(text.strip()) < 200:
            return 0.0
            
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if not sentences:
            return 0.0
            
        valid_sentences = 0
        for s in sentences:
            # Natural sentences: start with capital, end with punct, word count [4, 40]
            words = s.split()
            if len(s) > 0 and s[0].isupper() and s[-1] in '.!?' and 4 <= len(words) <= 40:
                valid_sentences += 1
                
        return valid_sentences / len(sentences)

    def get_ttr(self, text: str) -> float:
        """Calculates Type-Token Ratio for lexical richness."""
        words = text.lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def has_repetitive_spam(self, text: str, n: int = 3, max_ratio: float = 0.20) -> bool:
        """Rejects docs where the most common n-gram makes up > 20% of the document."""
        words = text.lower().split()
        if len(words) < n * 2: 
            return False
        
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
        if not ngrams: 
            return False
        
        _, count = Counter(ngrams).most_common(1)[0]
        if (count * n) / len(words) > max_ratio:
            return True
        return False

    def passes_heuristics(self, text: str) -> bool:
        """
        Comprehensive 'Picky Filter' from the notebook.
        Combines structural, statistical, and noise-reduction heuristics.
        """
        if not text:
            return False

        # 1. Structural Coherence
        if self.check_structural_coherence(text) < self.min_coherence_score:
            return False

        # 2. Lexical Richness (TTR)
        ttr = self.get_ttr(text)
        if not (self.min_ttr <= ttr <= self.max_ttr):
            return False

        text_lower = text.lower()

        # 3. N-Gram Repetition (SEO Spam)
        if self.has_repetitive_spam(text):
            return False

        # 4. Boilerplate / Smog Filter
        if sum(1 for kw in self.artifact_keywords if kw in text_lower) >= 2:
            return False

        # 5. Aggressive Code Snippet Removal
        if sum(text_lower.count(kw) for kw in self.code_keywords) >= 2:
            return False

        # 6. HTML Tag Filter
        if bool(re.search(r'<[^>]+>', text)):
            return False

        # 7. Alpha Ratio (OCR/Gibberish)
        if len(text) > 0 and (sum(c.isalpha() for c in text) / len(text)) < self.min_alpha_ratio:
            return False

        # 8. Long Words Check
        words = text.split()
        if any(len(word) > 30 for word in words):
            return False

        # 9. Symbol Ratio
        symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if words and (symbols / len(words)) > self.max_symbol_ratio:
            return False

        # 10. Average Line Length
        lines = text.split('\n')
        valid_lines = [line.strip() for line in lines if len(line.strip()) > 0]
        if valid_lines:
            avg_line_length = sum(len(line) for line in valid_lines) / len(valid_lines)
            if avg_line_length < self.min_avg_line_length:
                return False

        # 11. Tokenizer Efficiency Check
        sample_text = text[:2000]
        tokens = self.tokenizer.encode(sample_text)
        if tokens and (len(sample_text) / len(tokens)) < self.min_tokenizer_efficiency:
            return False

        return True

    def is_target_language(self, text: str) -> bool:
        """Simplified language detection matching the EDA notebook."""
        if not text:
            return False
        try:
            return detect(text) == self.target_lang
        except Exception:
            return False

    def _build_minhash(self, text: str, ngram: int) -> Optional[object]:
        if MinHash is None:
            return None

        tokens = text.lower().split()
        if len(tokens) < ngram:
            return None

        mh = MinHash(num_perm=self.near_dedup_num_perm)
        for i in range(len(tokens) - ngram + 1):
            shingle = " ".join(tokens[i : i + ngram])
            mh.update(shingle.encode("utf-8"))
        return mh

    def run_deduplication(self, dataset: Dataset, num_proc: int) -> Dataset:
        def get_hash(example):
            return {"hash": hashlib.sha256(example["text"].encode("utf-8")).hexdigest()}

        dataset = dataset.map(get_hash, num_proc=num_proc, desc="Computing SHA-256 hashes")
        unique_hashes = set()
        
        def filter_unique(example):
            if example["hash"] in unique_hashes:
                return False
            unique_hashes.add(example["hash"])
            return True

        return dataset.filter(filter_unique, desc="Exact deduplication")

    def run_near_deduplication(self, dataset: Dataset) -> Dataset:
        """Dual-layer MinHash LSH deduplication (3-gram and 5-gram)."""
        if MinHash is None or MinHashLSH is None:
            raise ImportError("datasketch is not installed.")

        lsh_3gram = MinHashLSH(threshold=self.near_dedup_threshold_3gram, num_perm=self.near_dedup_num_perm)
        lsh_5gram = MinHashLSH(threshold=self.near_dedup_threshold_5gram, num_perm=self.near_dedup_num_perm)
        
        kept_indices = []

        for idx, row in enumerate(tqdm(dataset, total=dataset.num_rows, desc="Dual-layer Near-deduplication")):
            text = row.get("text", "")
            
            m3 = self._build_minhash(text, ngram=3)
            m5 = self._build_minhash(text, ngram=5)

            if m3 is None or m5 is None:
                kept_indices.append(idx)
                continue

            # Query both layers
            if lsh_3gram.query(m3) or lsh_5gram.query(m5):
                continue

            lsh_3gram.insert(f"3g_{idx}", m3)
            lsh_5gram.insert(f"5g_{idx}", m5)
            kept_indices.append(idx)

        return dataset.select(kept_indices)

    def _process_shard(self, shard: Dataset, num_proc: int) -> Dataset:
        """Processes a single shard through the pipeline."""
        # 1. Formatting
        shard = shard.map(
            lambda x: {"text": self.normalize_formatting(x["text"])},
            num_proc=num_proc,
            desc="Normalizing text"
        )

        # 2. Picky Filtering
        shard = shard.filter(
            lambda x: self.is_target_language(x["text"]) and self.passes_heuristics(x["text"]),
            num_proc=num_proc,
            desc="Applying picky heuristic and language filters"
        )
        
        return shard

    def preprocess(
        self, 
        dataset: Dataset, 
        num_proc: int = 8, 
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval_min: int = 30
    ) -> Dataset:
        """
        Integrates all pillars with time-based checkpointing and progress tracking.
        """
        # Step 0: Initial Deduplication (often fast enough to do once)
        dataset = self.run_deduplication(dataset, num_proc)
        if self.enable_near_dedup:
            dataset = self.run_near_deduplication(dataset)

        if not checkpoint_dir:
            return self._process_shard(dataset, num_proc)

        # Checkpoint Setup
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        
        # Determine how many shards (aim for 10-20% chunks of data per shard for granular checkpoints)
        num_shards = max(10, dataset.num_rows // 10000) 
        processed_shards = []
        
        last_checkpoint_time = time.time()
        checkpoint_interval_sec = checkpoint_interval_min * 60
        
        print(f"Starting pipeline with {num_shards} shards. Checkpoints every {checkpoint_interval_min} mins.")
        
        for i in tqdm(range(num_shards), desc="Overall Progress (Shards)"):
            shard = dataset.shard(num_shards=num_shards, index=i, contiguous=True)
            processed_shard = self._process_shard(shard, num_proc)
            processed_shards.append(processed_shard)
            
            # Check if it's time to save a checkpoint
            current_time = time.time()
            if current_time - last_checkpoint_time > checkpoint_interval_sec:
                # Save current progress
                temp_combined = concatenate_datasets(processed_shards)
                
                # Use a rotating checkpoint strategy (save new, then delete old)
                new_ckpt = ckpt_path / f"checkpoint_shard_{i}"
                old_ckpts = list(ckpt_path.glob("checkpoint_shard_*"))
                
                temp_combined.save_to_disk(str(new_ckpt))
                print(f"\n[Checkpoint] Saved shard {i} progress to {new_ckpt}")
                
                # Cleanup older checkpoints
                for old in old_ckpts:
                    if old != new_ckpt:
                        shutil.rmtree(old, ignore_errors=True)
                
                last_checkpoint_time = current_time

        return concatenate_datasets(processed_shards)
