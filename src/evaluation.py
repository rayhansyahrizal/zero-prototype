"""
Evaluation module for caption quality assessment.
Implements BLEU, METEOR, and semantic similarity metrics.
"""

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import pandas as pd
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def download_nltk_resources():
    """Download required NLTK resources."""
    resources = ['punkt', 'wordnet', 'omw-1.4']
    
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                try:
                    logger.info(f"Downloading NLTK resource: {resource}")
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    logger.warning(f"Failed to download {resource}: {e}")


# Download resources on import
try:
    download_nltk_resources()
except Exception as e:
    logger.warning(f"NLTK resource download issue: {e}")


class CaptionEvaluator:
    """Evaluate generated captions using standard metrics (BLEU, METEOR, semantic similarity)."""
    
    def __init__(self, text_embedding_model=None, use_semantic_similarity: bool = True):
        """
        Initialize evaluator.
        
        Args:
            text_embedding_model: MedImageInsight text encoder for semantic similarity
            use_semantic_similarity: Whether to compute semantic similarity
        """
        self.smoothing = SmoothingFunction()
        self.text_embedding_model = text_embedding_model
        self.use_semantic_similarity = use_semantic_similarity and (text_embedding_model is not None)
        
        if use_semantic_similarity and text_embedding_model is None:
            logger.warning(
                "Semantic similarity requested but no text embedding model provided. "
                "Pass MedImageInsight text encoder to enable semantic similarity."
            )
        
        logger.info(
            f"CaptionEvaluator initialized "
            f"(semantic_similarity={'enabled' if self.use_semantic_similarity else 'disabled'})"
        )
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        try:
            tokens = nltk.word_tokenize(text.lower())
        except:
            # Fallback to simple split
            tokens = text.lower().split()
        return tokens
    
    def compute_bleu(
        self,
        reference: str,
        hypothesis: str,
        max_n: int = 4
    ) -> Dict[str, float]:
        """
        Compute BLEU scores.
        
        Args:
            reference: Reference caption
            hypothesis: Generated caption
            max_n: Maximum n-gram order
            
        Returns:
            Dictionary with BLEU-1 to BLEU-4 scores
        """
        ref_tokens = self.tokenize(reference)
        hyp_tokens = self.tokenize(hypothesis)
        
        scores = {}
        
        for n in range(1, max_n + 1):
            weights = tuple([1.0/n] * n + [0.0] * (max_n - n))
            
            try:
                score = sentence_bleu(
                    [ref_tokens],
                    hyp_tokens,
                    weights=weights,
                    smoothing_function=self.smoothing.method1
                )
            except:
                score = 0.0
            
            scores[f'bleu_{n}'] = score
        
        return scores
    
    def compute_meteor(
        self,
        reference: str,
        hypothesis: str
    ) -> float:
        """
        Compute METEOR score.
        
        Args:
            reference: Reference caption
            hypothesis: Generated caption
            
        Returns:
            METEOR score
        """
        ref_tokens = self.tokenize(reference)
        hyp_tokens = self.tokenize(hypothesis)
        
        try:
            score = meteor_score([ref_tokens], hyp_tokens)
        except Exception as e:
            logger.warning(f"METEOR computation failed: {e}")
            score = 0.0
        
        return score
    
    def compute_semantic_similarity(
        self,
        reference: str,
        hypothesis: str
    ) -> float:
        """
        Compute semantic similarity using MedImageInsight text embeddings.
        
        This uses the same medical domain-specific text encoder that was used
        for generating embeddings, ensuring consistency and domain relevance.
        
        Args:
            reference: Ground truth caption
            hypothesis: Generated caption
            
        Returns:
            Cosine similarity score (0-1)
        """
        if not self.use_semantic_similarity or self.text_embedding_model is None:
            return 0.0
        
        try:
            # Encode both captions using MedImageInsight text encoder
            ref_output = self.text_embedding_model.encode(texts=[reference])
            hyp_output = self.text_embedding_model.encode(texts=[hypothesis])
            
            ref_embedding = ref_output['text_embeddings'][0]
            hyp_embedding = hyp_output['text_embeddings'][0]
            
            # Compute cosine similarity
            ref_norm = ref_embedding / (np.linalg.norm(ref_embedding) + 1e-8)
            hyp_norm = hyp_embedding / (np.linalg.norm(hyp_embedding) + 1e-8)
            
            similarity = float(np.dot(ref_norm, hyp_norm))
            
            # Clip to [0, 1] range (should already be, but just in case)
            similarity = max(0.0, min(1.0, similarity))
            
            return similarity
            
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            return 0.0
    
    def compute_semantic_similarity(
        self,
        reference: str,
        hypothesis: str
    ) -> float:
        """
        Compute semantic similarity using sentence embeddings.
        
        Uses SentenceTransformer ('all-MiniLM-L6-v2') to encode both texts
        and computes cosine similarity between embeddings.
        
        Args:
            reference: Reference caption
            hypothesis: Generated caption
            
        Returns:
            Cosine similarity score (0-1, higher is better)
        """
        if not self.use_semantic_similarity or self.text_embedding_model is None:
            return 0.0
        
        try:
            # Encode both texts using MedImageInsight text encoder
            result = self.text_embedding_model.encode(texts=[reference, hypothesis])
            
            # Extract text embeddings from the returned dictionary
            embeddings = result["text_embeddings"]
            
            # Compute cosine similarity
            ref_emb = embeddings[0]
            hyp_emb = embeddings[1]
            
            # Normalize
            ref_norm = ref_emb / (np.linalg.norm(ref_emb) + 1e-8)
            hyp_norm = hyp_emb / (np.linalg.norm(hyp_emb) + 1e-8)
            
            # Cosine similarity
            similarity = float(np.dot(ref_norm, hyp_norm))
            
            return similarity
            
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            return 0.0
    
    def evaluate_single(
        self,
        reference: str,
        hypothesis: str
    ) -> Dict[str, float]:
        """
        Evaluate a single caption.
        
        Args:
            reference: Ground truth caption
            hypothesis: Generated caption
            
        Returns:
            Dictionary with all metric scores
        """
        scores = {}
        
        # BLEU scores
        bleu_scores = self.compute_bleu(reference, hypothesis)
        scores.update(bleu_scores)
        
        # METEOR score
        scores['meteor'] = self.compute_meteor(reference, hypothesis)
        
        # Semantic similarity
        if self.use_semantic_similarity:
            scores['semantic_similarity'] = self.compute_semantic_similarity(
                reference, hypothesis
            )
        
        return scores
    
    def evaluate_batch(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Evaluate a batch of captions.
        
        Args:
            references: List of ground truth captions
            hypotheses: List of generated captions
            
        Returns:
            Tuple of (average_scores, individual_scores)
        """
        if len(references) != len(hypotheses):
            raise ValueError("Number of references and hypotheses must match")
        
        individual_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            scores = self.evaluate_single(ref, hyp)
            individual_scores.append(scores)
        
        # Compute averages
        avg_scores = {}
        if individual_scores:
            for key in individual_scores[0].keys():
                avg_scores[key] = np.mean([s[key] for s in individual_scores])
        
        return avg_scores, individual_scores
    
    def evaluate_results(
        self,
        results: Dict[str, List[Dict[str, str]]],
        ground_truths: Dict[str, str],
        modalities: Optional[Dict[str, str]] = None
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[Dict]]]:
        """
        Evaluate results from multiple generation modes with detailed per-sample logging.
        
        Args:
            results: Dictionary mapping method to list of generation results
            ground_truths: Dictionary mapping image_id to ground truth caption
            modalities: Optional dictionary mapping image_id to modality
            
        Returns:
            Tuple of (average_scores, detailed_per_sample_scores)
        """
        all_scores = {}
        all_detailed_scores = {}
        
        for method, method_results in results.items():
            logger.info(f"Evaluating {method} results...")
            
            # Extract references and hypotheses with metadata
            references = []
            hypotheses = []
            sample_ids = []
            sample_modalities = []
            retrieved_contexts = []
            
            for result in method_results:
                image_id = result['image_id']
                if image_id in ground_truths:
                    references.append(ground_truths[image_id])
                    hypotheses.append(result['caption'])
                    sample_ids.append(image_id)
                    
                    # Get modality
                    if modalities and image_id in modalities:
                        sample_modalities.append(modalities[image_id])
                    else:
                        sample_modalities.append('UNKNOWN')
                    
                    # Get retrieved context
                    retrieved_contexts.append(result.get('retrieved_context', ''))
            
            if not references:
                logger.warning(f"No valid references found for {method}")
                continue
            
            # Evaluate batch
            avg_scores, individual_scores = self.evaluate_batch(references, hypotheses)
            all_scores[method] = avg_scores
            
            # Build detailed per-sample results
            detailed_samples = []
            for i, (img_id, ref, hyp, scores) in enumerate(zip(
                sample_ids, references, hypotheses, individual_scores
            )):
                sample_detail = {
                    'image_id': img_id,
                    'modality': sample_modalities[i],
                    'ground_truth': ref,
                    'generated_caption': hyp,
                    'retrieved_context': retrieved_contexts[i],
                    **scores  # Include all metric scores
                }
                detailed_samples.append(sample_detail)
                
                # Log first few samples for inspection
                if i < 3:
                    logger.info(f"\n  Sample {i+1} ({img_id}, {sample_modalities[i]}):")
                    logger.info(f"    Ground truth: {ref[:100]}...")
                    logger.info(f"    Generated: {hyp[:100]}...")
                    logger.info(f"    BLEU-4: {scores.get('bleu_4', 0):.4f}, "
                              f"METEOR: {scores.get('meteor', 0):.4f}, "
                              f"Semantic: {scores.get('semantic_similarity', 0):.4f}")
            
            all_detailed_scores[method] = detailed_samples
            
            # Log average scores
            logger.info(f"\n{method.upper()} average scores:")
            for metric, score in avg_scores.items():
                logger.info(f"  {metric}: {score:.4f}")
        
        return all_scores, all_detailed_scores


def save_results(
    results: Dict[str, List[Dict[str, str]]],
    scores: Dict[str, Dict[str, float]],
    config: dict,
    timestamp: str = None,
    detailed_scores: Optional[Dict[str, List[Dict]]] = None
):
    """
    Save generation results and evaluation scores with timestamp.
    
    Args:
        results: Generation results for all methods
        scores: Evaluation scores for all methods
        config: Configuration dictionary
        timestamp: Optional timestamp string (will be generated if not provided)
        detailed_scores: Optional detailed per-sample scores
    """
    from datetime import datetime
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save captions with timestamp
    captions_file = results_dir / f"captions_{timestamp}.json"
    logger.info(f"Saving captions to {captions_file}")
    
    with open(captions_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save to default location (latest run)
    captions_latest = Path(config['output']['captions_file'])
    with open(captions_latest, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metrics with timestamp
    metrics_file = results_dir / f"metrics_{timestamp}.csv"
    logger.info(f"Saving metrics to {metrics_file}")
    
    # Convert to DataFrame for nice CSV format
    metrics_data = []
    for method, method_scores in scores.items():
        row = {'method': method, 'timestamp': timestamp}
        row.update(method_scores)
        metrics_data.append(row)
    
    df = pd.DataFrame(metrics_data)
    df.to_csv(metrics_file, index=False, float_format='%.4f')
    
    # Also save to default location (latest run)
    metrics_latest = Path(config['output']['metrics_file'])
    df.to_csv(metrics_latest, index=False, float_format='%.4f')
    
    # Save detailed per-sample results if provided
    if detailed_scores:
        detailed_file = results_dir / f"detailed_results_{timestamp}.json"
        logger.info(f"Saving detailed per-sample results to {detailed_file}")
        
        with open(detailed_file, 'w') as f:
            json.dump(detailed_scores, f, indent=2)
        
        # Also save as CSV for easy analysis
        for method, samples in detailed_scores.items():
            if samples:
                detailed_csv = results_dir / f"detailed_{method}_{timestamp}.csv"
                df_detailed = pd.DataFrame(samples)
                df_detailed.to_csv(detailed_csv, index=False, float_format='%.4f')
                logger.info(f"Saved detailed {method} results to {detailed_csv}")
    
    logger.info(f"Results saved successfully (timestamp: {timestamp})")


def compare_methods(scores: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create comparison table of different methods.
    
    Args:
        scores: Dictionary of scores for each method
        
    Returns:
        DataFrame with comparison results
    """
    df = pd.DataFrame(scores).T
    
    # Add rank for each metric
    for col in df.columns:
        df[f'{col}_rank'] = df[col].rank(ascending=False)
    
    return df


if __name__ == "__main__":
    # Test evaluation
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Test with sample captions
    evaluator = CaptionEvaluator()
    
    # Example evaluation
    reference = "Chest X-ray showing bilateral infiltrates consistent with pneumonia"
    hypothesis1 = "Chest X-ray with bilateral infiltrates suggesting pneumonia"
    hypothesis2 = "Normal chest radiograph"
    hypothesis3 = "X-ray of the chest"
    
    print("\n=== Single Caption Evaluation ===")
    print(f"Reference: {reference}\n")
    
    for i, hyp in enumerate([hypothesis1, hypothesis2, hypothesis3], 1):
        print(f"Hypothesis {i}: {hyp}")
        scores = evaluator.evaluate_single(reference, hyp)
        for metric, score in scores.items():
            print(f"  {metric}: {score:.4f}")
        print()
    
    # Test batch evaluation
    references = [
        "Chest X-ray showing bilateral infiltrates",
        "CT scan demonstrates large liver mass",
        "MRI reveals herniated disc at L4-L5"
    ]
    
    hypotheses = [
        "Chest X-ray with bilateral infiltrates",
        "CT shows liver mass",
        "MRI shows disc herniation"
    ]
    
    print("\n=== Batch Evaluation ===")
    avg_scores, individual_scores = evaluator.evaluate_batch(references, hypotheses)
    
    print("Average scores:")
    for metric, score in avg_scores.items():
        print(f"  {metric}: {score:.4f}")
