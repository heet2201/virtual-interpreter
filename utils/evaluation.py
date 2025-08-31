"""
Evaluation utilities for the Virtual Interpreter components
"""
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
import re
from collections import Counter
import editdistance

try:
    from sacrebleu import corpus_bleu, sentence_bleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False

logger = logging.getLogger(__name__)


class ASREvaluator:
    """Evaluator for ASR component"""
    
    def __init__(self):
        self.metrics = {}
    
    def word_error_rate(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate (WER)
        
        Args:
            reference: Ground truth text
            hypothesis: ASR output text
            
        Returns:
            WER score (0 = perfect, higher = worse)
        """
        # Normalize texts
        ref_words = self._normalize_text(reference).split()
        hyp_words = self._normalize_text(hypothesis).split()
        
        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0
        
        # Calculate edit distance
        edit_dist = editdistance.eval(ref_words, hyp_words)
        wer = edit_dist / len(ref_words)
        
        return wer
    
    def character_error_rate(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate (CER)
        
        Args:
            reference: Ground truth text
            hypothesis: ASR output text
            
        Returns:
            CER score (0 = perfect, higher = worse)
        """
        ref_chars = self._normalize_text(reference).replace(' ', '')
        hyp_chars = self._normalize_text(hypothesis).replace(' ', '')
        
        if len(ref_chars) == 0:
            return 1.0 if len(hyp_chars) > 0 else 0.0
        
        edit_dist = editdistance.eval(ref_chars, hyp_chars)
        cer = edit_dist / len(ref_chars)
        
        return cer
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for evaluation"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def evaluate_batch(
        self, 
        references: List[str], 
        hypotheses: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a batch of ASR results
        
        Args:
            references: List of ground truth texts
            hypotheses: List of ASR outputs
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses must have same length")
        
        wers = []
        cers = []
        
        for ref, hyp in zip(references, hypotheses):
            wers.append(self.word_error_rate(ref, hyp))
            cers.append(self.character_error_rate(ref, hyp))
        
        return {
            "wer_mean": np.mean(wers),
            "wer_std": np.std(wers),
            "cer_mean": np.mean(cers),
            "cer_std": np.std(cers),
            "num_samples": len(references)
        }


class TranslationEvaluator:
    """Evaluator for Machine Translation component"""
    
    def __init__(self):
        self.metrics = {}
    
    def bleu_score(
        self, 
        reference: str, 
        hypothesis: str, 
        use_sacrebleu: bool = True
    ) -> float:
        """
        Calculate BLEU score
        
        Args:
            reference: Ground truth translation
            hypothesis: Model translation
            use_sacrebleu: Whether to use SacreBLEU (if available)
            
        Returns:
            BLEU score (0-1, higher = better)
        """
        if use_sacrebleu and SACREBLEU_AVAILABLE:
            try:
                # SacreBLEU expects lists
                refs = [[reference]]
                hyps = [hypothesis]
                bleu = corpus_bleu(hyps, refs)
                return bleu.score / 100.0  # Convert to 0-1 scale
            except Exception as e:
                logger.warning(f"SacreBLEU failed, using simple BLEU: {e}")
        
        # Simple BLEU implementation
        return self._simple_bleu(reference, hypothesis)
    
    def _simple_bleu(self, reference: str, hypothesis: str, max_n: int = 4) -> float:
        """Simple BLEU score implementation"""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        # Calculate n-gram precisions
        precisions = []
        
        for n in range(1, max_n + 1):
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            hyp_ngrams = self._get_ngrams(hyp_tokens, n)
            
            if len(hyp_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            # Count matches
            matches = 0
            for ngram in hyp_ngrams:
                if ngram in ref_ngrams:
                    matches += 1
            
            precision = matches / len(hyp_ngrams)
            precisions.append(precision)
        
        if all(p == 0 for p in precisions):
            return 0.0
        
        # Brevity penalty
        bp = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))
        
        # Geometric mean of precisions
        bleu = bp * np.exp(np.mean([np.log(p) if p > 0 else -np.inf for p in precisions]))
        
        return max(0.0, bleu)
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple]:
        """Get n-grams from token list"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def meteor_score(self, reference: str, hypothesis: str) -> float:
        """
        Simple METEOR-like score based on alignment
        
        Args:
            reference: Ground truth translation
            hypothesis: Model translation
            
        Returns:
            METEOR-like score (0-1, higher = better)
        """
        ref_tokens = set(reference.lower().split())
        hyp_tokens = set(hypothesis.lower().split())
        
        if len(hyp_tokens) == 0:
            return 0.0 if len(ref_tokens) > 0 else 1.0
        
        # Calculate precision, recall, F1
        matches = len(ref_tokens.intersection(hyp_tokens))
        precision = matches / len(hyp_tokens)
        recall = matches / max(len(ref_tokens), 1)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def semantic_similarity(self, reference: str, hypothesis: str) -> float:
        """
        Simple semantic similarity based on word overlap
        (In practice, would use embeddings like SentenceTransformers)
        
        Args:
            reference: Ground truth translation
            hypothesis: Model translation
            
        Returns:
            Similarity score (0-1, higher = better)
        """
        # This is a simplified implementation
        # In production, use sentence embeddings
        
        ref_words = Counter(reference.lower().split())
        hyp_words = Counter(hypothesis.lower().split())
        
        # Cosine similarity of word counts
        intersection = sum((ref_words & hyp_words).values())
        norm_ref = sum(ref_words.values())
        norm_hyp = sum(hyp_words.values())
        
        if norm_ref == 0 or norm_hyp == 0:
            return 0.0
        
        similarity = intersection / np.sqrt(norm_ref * norm_hyp)
        return similarity
    
    def evaluate_batch(
        self, 
        references: List[str], 
        hypotheses: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a batch of translation results
        
        Args:
            references: List of ground truth translations
            hypotheses: List of model translations
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses must have same length")
        
        bleu_scores = []
        meteor_scores = []
        semantic_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            bleu_scores.append(self.bleu_score(ref, hyp))
            meteor_scores.append(self.meteor_score(ref, hyp))
            semantic_scores.append(self.semantic_similarity(ref, hyp))
        
        return {
            "bleu_mean": np.mean(bleu_scores),
            "bleu_std": np.std(bleu_scores),
            "meteor_mean": np.mean(meteor_scores),
            "meteor_std": np.std(meteor_scores),
            "semantic_mean": np.mean(semantic_scores),
            "semantic_std": np.std(semantic_scores),
            "num_samples": len(references)
        }


class TTSEvaluator:
    """Evaluator for TTS component"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_naturalness(self, audio_file: str) -> Dict[str, float]:
        """
        Evaluate TTS naturalness (placeholder implementation)
        
        Args:
            audio_file: Path to generated audio
            
        Returns:
            Dictionary with naturalness metrics
        """
        # This would typically involve:
        # 1. Prosody analysis
        # 2. Spectral quality metrics
        # 3. Human evaluation scores
        
        # Placeholder implementation
        return {
            "naturalness_score": 0.75,  # Would compute actual metrics
            "prosody_score": 0.7,
            "spectral_quality": 0.8
        }
    
    def evaluate_intelligibility(self, audio_file: str, reference_text: str) -> float:
        """
        Evaluate TTS intelligibility using ASR
        
        Args:
            audio_file: Path to generated audio
            reference_text: Original text
            
        Returns:
            Intelligibility score based on ASR accuracy
        """
        # This would involve:
        # 1. Running ASR on generated audio
        # 2. Comparing with reference text
        # 3. Computing accuracy metrics
        
        # Placeholder implementation
        return 0.85


class PipelineEvaluator:
    """End-to-end pipeline evaluator"""
    
    def __init__(self):
        self.asr_evaluator = ASREvaluator()
        self.mt_evaluator = TranslationEvaluator()
        self.tts_evaluator = TTSEvaluator()
    
    def evaluate_end_to_end(
        self, 
        input_audio: str,
        reference_transcript: str,
        reference_translation: str,
        pipeline_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate complete pipeline performance
        
        Args:
            input_audio: Input audio file
            reference_transcript: Ground truth transcript
            reference_translation: Ground truth translation
            pipeline_result: Pipeline output
            
        Returns:
            Complete evaluation metrics
        """
        evaluation = {}
        
        try:
            # ASR evaluation
            if "transcription" in pipeline_result:
                asr_wer = self.asr_evaluator.word_error_rate(
                    reference_transcript, 
                    pipeline_result["transcription"]
                )
                asr_cer = self.asr_evaluator.character_error_rate(
                    reference_transcript, 
                    pipeline_result["transcription"]
                )
                evaluation["asr"] = {
                    "wer": asr_wer,
                    "cer": asr_cer
                }
            
            # Translation evaluation
            if "translation" in pipeline_result:
                mt_bleu = self.mt_evaluator.bleu_score(
                    reference_translation, 
                    pipeline_result["translation"]
                )
                mt_meteor = self.mt_evaluator.meteor_score(
                    reference_translation, 
                    pipeline_result["translation"]
                )
                evaluation["translation"] = {
                    "bleu": mt_bleu,
                    "meteor": mt_meteor
                }
            
            # TTS evaluation
            if "output_audio_file" in pipeline_result:
                tts_metrics = self.tts_evaluator.evaluate_naturalness(
                    pipeline_result["output_audio_file"]
                )
                evaluation["tts"] = tts_metrics
            
            # Overall quality score
            evaluation["overall_score"] = self._calculate_overall_score(evaluation)
            
        except Exception as e:
            logger.error(f"Error in pipeline evaluation: {e}")
            evaluation["error"] = str(e)
        
        return evaluation
    
    def _calculate_overall_score(self, evaluation: Dict[str, Any]) -> float:
        """Calculate weighted overall quality score"""
        scores = []
        
        if "asr" in evaluation:
            # Lower WER/CER is better, convert to 0-1 scale
            asr_score = 1.0 - min(1.0, (evaluation["asr"]["wer"] + evaluation["asr"]["cer"]) / 2)
            scores.append(("asr", asr_score, 0.3))
        
        if "translation" in evaluation:
            mt_score = (evaluation["translation"]["bleu"] + evaluation["translation"]["meteor"]) / 2
            scores.append(("mt", mt_score, 0.4))
        
        if "tts" in evaluation:
            tts_score = evaluation["tts"]["naturalness_score"]
            scores.append(("tts", tts_score, 0.3))
        
        if not scores:
            return 0.0
        
        # Weighted average
        total_weight = sum(weight for _, _, weight in scores)
        weighted_score = sum(score * weight for _, score, weight in scores) / total_weight
        
        return weighted_score
    
    def generate_report(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate evaluation report from multiple evaluations"""
        if not evaluations:
            return {"error": "No evaluations provided"}
        
        # Aggregate metrics
        asr_wers = []
        asr_cers = []
        mt_bleus = []
        mt_meteors = []
        tts_scores = []
        overall_scores = []
        
        for eval_result in evaluations:
            if "asr" in eval_result:
                asr_wers.append(eval_result["asr"]["wer"])
                asr_cers.append(eval_result["asr"]["cer"])
            
            if "translation" in eval_result:
                mt_bleus.append(eval_result["translation"]["bleu"])
                mt_meteors.append(eval_result["translation"]["meteor"])
            
            if "tts" in eval_result:
                tts_scores.append(eval_result["tts"]["naturalness_score"])
            
            if "overall_score" in eval_result:
                overall_scores.append(eval_result["overall_score"])
        
        report = {
            "num_evaluations": len(evaluations),
            "summary": {}
        }
        
        if asr_wers:
            report["summary"]["asr"] = {
                "wer_mean": np.mean(asr_wers),
                "wer_std": np.std(asr_wers),
                "cer_mean": np.mean(asr_cers),
                "cer_std": np.std(asr_cers)
            }
        
        if mt_bleus:
            report["summary"]["translation"] = {
                "bleu_mean": np.mean(mt_bleus),
                "bleu_std": np.std(mt_bleus),
                "meteor_mean": np.mean(mt_meteors),
                "meteor_std": np.std(mt_meteors)
            }
        
        if tts_scores:
            report["summary"]["tts"] = {
                "naturalness_mean": np.mean(tts_scores),
                "naturalness_std": np.std(tts_scores)
            }
        
        if overall_scores:
            report["summary"]["overall"] = {
                "score_mean": np.mean(overall_scores),
                "score_std": np.std(overall_scores)
            }
        
        return report
