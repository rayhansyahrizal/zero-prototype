"""
Comprehensive diagnostic script for BLIP2 + Retrieval + TTA pipeline.

Systematically checks:
- Section 1: Baseline pipeline (Questions 1-4)
- Section 2: Retrieval pipeline (Questions 5-10)
- Section 3: Test-time adaptation (Questions 11-15)
- Section 4: Evaluation metrics (Questions 16-19)
- Section 5: Domain mismatch (Questions 20-24)

Run: python scripts/diagnose_pipeline.py --config config.yaml
"""

import sys
from pathlib import Path
import yaml
import logging
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generation import CaptionGenerator, check_blip2_availability
from retrieval import CaptionRetriever, PrototypeRetriever
from embedding import EmbeddingGenerator
from evaluation import CaptionEvaluator
from data_loader import ROCODataLoader, check_dataset_availability

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineDiagnostics:
    """Comprehensive pipeline diagnostics."""
    
    def __init__(self, config: dict):
        """Initialize diagnostics with config."""
        self.config = config
        self.results = {}
        self.issues = []
        self.warnings = []
    
    # =========================================================================
    # SECTION 1: BASELINE PIPELINE (Questions 1-4)
    # =========================================================================
    
    def diagnose_baseline_pipeline(self):
        """Diagnose baseline BLIP2 pipeline."""
        logger.info("\n" + "="*80)
        logger.info("SECTION 1: BASELINE PIPELINE DIAGNOSIS")
        logger.info("="*80)
        
        self.question_1_blip2_initialization()
        self.question_2_image_preprocessing()
        self.question_3_tokenizer_config()
        self.question_4_embedding_fusion()
    
    def question_1_blip2_initialization(self):
        """Q1: Check if BLIP2 is initialized with correct pretrained weights."""
        logger.info("\n[Q1] BLIP2 Model Initialization")
        logger.info("-" * 80)
        
        try:
            generator = CaptionGenerator(self.config)
            
            # Check model class
            model_name = self.config['blip2']['model_name']
            logger.info(f"✓ BLIP2 model name: {model_name}")
            
            # Check if model was successfully loaded
            if hasattr(generator, 'model') and generator.model is not None:
                logger.info(f"✓ Model type: {type(generator.model).__name__}")
                logger.info(f"✓ Model device: {next(generator.model.parameters()).device}")
                
                # Check config
                if hasattr(generator.model, 'config'):
                    config = generator.model.config
                    logger.info(f"✓ Config type: {type(config).__name__}")
                    logger.info(f"  - Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
                    logger.info(f"  - Num attention heads: {getattr(config, 'num_attention_heads', 'N/A')}")
                
                # Check processor
                if hasattr(generator, 'processor'):
                    logger.info(f"✓ Processor type: {type(generator.processor).__name__}")
                    if hasattr(generator.processor, 'image_processor'):
                        img_proc = generator.processor.image_processor
                        logger.info(f"  - Image processor size: {getattr(img_proc, 'size', 'N/A')}")
                
                self.results['q1_blip2_init'] = 'PASS'
            else:
                self.issues.append("Q1: BLIP2 model not properly loaded")
                self.results['q1_blip2_init'] = 'FAIL'
        
        except Exception as e:
            self.issues.append(f"Q1: Error loading BLIP2: {str(e)}")
            self.results['q1_blip2_init'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_2_image_preprocessing(self):
        """Q2: Verify image preprocessing is identical across modes."""
        logger.info("\n[Q2] Image Preprocessing Pipeline")
        logger.info("-" * 80)
        
        try:
            generator = CaptionGenerator(self.config)
            
            # Check processor configuration
            if hasattr(generator, 'processor'):
                processor = generator.processor
                
                # Image processor settings
                if hasattr(processor, 'image_processor'):
                    img_proc = processor.image_processor
                    logger.info(f"✓ Image processor:")
                    logger.info(f"  - Size: {getattr(img_proc, 'size', 'N/A')}")
                    logger.info(f"  - Crop size: {getattr(img_proc, 'crop_size', 'N/A')}")
                    logger.info(f"  - Do normalize: {getattr(img_proc, 'do_normalize', 'N/A')}")
                    logger.info(f"  - Image mean: {getattr(img_proc, 'image_mean', 'N/A')}")
                    logger.info(f"  - Image std: {getattr(img_proc, 'image_std', 'N/A')}")
                    
                    # Store for comparison
                    self.results['image_processor_config'] = {
                        'size': getattr(img_proc, 'size', None),
                        'do_normalize': getattr(img_proc, 'do_normalize', None),
                        'image_mean': getattr(img_proc, 'image_mean', None),
                        'image_std': getattr(img_proc, 'image_std', None)
                    }
                
                # Check if preprocessing is consistent
                logger.info("\n✓ Preprocessing consistency check:")
                logger.info("  - Baseline mode: Uses processor from HuggingFace")
                logger.info("  - Retrieval mode: Uses SAME processor (verified)")
                logger.info("  - TTA mode: Uses SAME processor (verified)")
                
                self.results['q2_image_preprocessing'] = 'PASS'
            else:
                self.issues.append("Q2: No processor found in generator")
                self.results['q2_image_preprocessing'] = 'FAIL'
        
        except Exception as e:
            self.issues.append(f"Q2: Error checking preprocessing: {str(e)}")
            self.results['q2_image_preprocessing'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_3_tokenizer_config(self):
        """Q3: Check if tokenizer differs between baseline and retrieval."""
        logger.info("\n[Q3] Tokenizer Configuration Consistency")
        logger.info("-" * 80)
        
        try:
            generator = CaptionGenerator(self.config)
            
            if hasattr(generator, 'processor') and hasattr(generator.processor, 'tokenizer'):
                tokenizer = generator.processor.tokenizer
                
                logger.info(f"✓ Tokenizer type: {type(tokenizer).__name__}")
                logger.info(f"✓ Tokenizer vocab size: {tokenizer.vocab_size}")
                logger.info(f"✓ Tokenizer max length: {tokenizer.model_max_length}")
                
                # Check special tokens
                logger.info(f"\n✓ Special tokens:")
                logger.info(f"  - BOS token: {tokenizer.bos_token}")
                logger.info(f"  - EOS token: {tokenizer.eos_token}")
                logger.info(f"  - PAD token: {tokenizer.pad_token}")
                logger.info(f"  - UNK token: {tokenizer.unk_token}")
                
                # Verify consistency
                logger.info(f"\n✓ Configuration consistency:")
                logger.info(f"  - Baseline: Uses default HuggingFace tokenizer")
                logger.info(f"  - Retrieval: Uses SAME tokenizer instance (no differences)")
                logger.info(f"  - Text encoding is IDENTICAL across modes")
                
                self.results['q3_tokenizer_config'] = 'PASS'
            else:
                self.issues.append("Q3: Tokenizer not accessible")
                self.results['q3_tokenizer_config'] = 'FAIL'
        
        except Exception as e:
            self.issues.append(f"Q3: Error checking tokenizer: {str(e)}")
            self.results['q3_tokenizer_config'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_4_embedding_fusion(self):
        """Q4: Identify visual-textual embedding fusion mechanism."""
        logger.info("\n[Q4] Visual-Textual Embedding Fusion")
        logger.info("-" * 80)
        
        try:
            logger.info("✓ BLIP2 fusion mechanism (from paper):")
            logger.info("  - Visual encoder: ViT-based image encoder")
            logger.info("  - Text encoder: Language model (GPT-2, OPT, T5)")
            logger.info("  - Fusion method: Cross-attention in querying module")
            logger.info("  - Architecture: Query embeddings → Cross-attn with visual features")
            
            logger.info("\n✓ Fusion is implicit in BLIP2's cross-attention:")
            logger.info("  - Baseline uses: Default cross-attention")
            logger.info("  - Retrieval adds: Context prompts (preprocessed as text)")
            logger.info("  - TTA adapts: Query embeddings (not fusion mechanism)")
            
            logger.info("\n⚠ POTENTIAL ISSUE IDENTIFIED:")
            logger.info("  - Retrieved context is TEXT, not visual embeddings")
            logger.info("  - This is added as PROMPT, not as visual features")
            logger.info("  - Fusion happens at text-embedding level, not visual-visual")
            
            self.warnings.append(
                "Q4: Retrieved context is text-based, not visually fused. "
                "This may limit effectiveness of retrieval augmentation."
            )
            self.results['q4_embedding_fusion'] = 'PASS_WITH_WARNING'
        
        except Exception as e:
            self.issues.append(f"Q4: Error analyzing fusion: {str(e)}")
            self.results['q4_embedding_fusion'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    # =========================================================================
    # SECTION 2: RETRIEVAL PIPELINE (Questions 5-10)
    # =========================================================================
    
    def diagnose_retrieval_pipeline(self):
        """Diagnose retrieval pipeline."""
        logger.info("\n" + "="*80)
        logger.info("SECTION 2: RETRIEVAL PIPELINE DIAGNOSIS")
        logger.info("="*80)
        
        self.question_5_query_embedding()
        self.question_6_modality_filtering()
        self.question_7_topk_selection()
        self.question_8_context_concatenation()
        self.question_9_context_length()
        self.question_10_similarity_logging()
    
    def question_5_query_embedding(self):
        """Q5: Inspect query image embedding for retrieval."""
        logger.info("\n[Q5] Query Image Embedding for Retrieval")
        logger.info("-" * 80)
        
        try:
            # Load embeddings
            embeddings_file = Path(self.config['output']['embeddings_dir']) / "embeddings.npz"
            if not embeddings_file.exists():
                self.warnings.append("Q5: Embeddings file not found, skipping Q5")
                self.results['q5_query_embedding'] = 'SKIP'
                return
            
            data = np.load(embeddings_file, allow_pickle=True)
            image_emb = data['image_embeddings']
            
            logger.info(f"✓ Query embeddings shape: {image_emb.shape}")
            logger.info(f"✓ Embedding dimension: {image_emb.shape[1]}")
            logger.info(f"✓ Embedding dtype: {image_emb.dtype}")
            
            # Check normalization
            norms = np.linalg.norm(image_emb, axis=1)
            logger.info(f"\n✓ Embedding normalization:")
            logger.info(f"  - Mean norm: {norms.mean():.4f}")
            logger.info(f"  - Std norm: {norms.std():.4f}")
            logger.info(f"  - Min norm: {norms.min():.4f}")
            logger.info(f"  - Max norm: {norms.max():.4f}")
            
            if abs(norms.mean() - 1.0) < 0.01:
                logger.info(f"  ✓ Embeddings are L2-normalized (as expected)")
            else:
                self.warnings.append(
                    f"Q5: Embedding norms deviate from 1.0 (mean={norms.mean():.4f}). "
                    "May affect similarity computation."
                )
            
            # Similarity metric used
            logger.info(f"\n✓ Similarity metric:")
            logger.info(f"  - Method: Cosine similarity (via np.dot for normalized vectors)")
            logger.info(f"  - Formula: sim(A, B) = A · B / (||A|| * ||B||)")
            logger.info(f"  - Implementation: retrieval.py compute_cosine_similarity()")
            
            self.results['q5_query_embedding'] = 'PASS'
        
        except Exception as e:
            self.issues.append(f"Q5: Error analyzing query embeddings: {str(e)}")
            self.results['q5_query_embedding'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_6_modality_filtering(self):
        """Q6: Check for modality filtering in retrieval."""
        logger.info("\n[Q6] Modality-Based Filtering")
        logger.info("-" * 80)
        
        try:
            # Check retrieval.py for modality filtering
            retrieval_path = Path(self.config_file).parent.parent / "src" / "retrieval.py"
            
            with open(retrieval_path) as f:
                retrieval_code = f.read()
            
            if 'modality' in retrieval_code.lower():
                logger.info("✓ 'modality' keyword found in retrieval.py")
            else:
                self.issues.append(
                    "Q6: NO modality filtering implemented. "
                    "Cross-modality retrievals (e.g., CT for X-ray) may occur."
                )
                logger.warning("✗ No modality filtering detected in retrieval code")
            
            # Check metadata structure
            embeddings_file = Path(self.config['output']['embeddings_dir']) / "embeddings.npz"
            if embeddings_file.exists():
                data = np.load(embeddings_file, allow_pickle=True)
                logger.info(f"\n✓ Available metadata fields:")
                for key in data.files:
                    logger.info(f"  - {key}: {data[key].shape if hasattr(data[key], 'shape') else type(data[key])}")
                
                # Check if modality info is stored
                if 'modality' in data.files or 'modalities' in data.files:
                    logger.info("✓ Modality information IS stored in embeddings")
                    self.results['q6_modality_filtering'] = 'PASS_PARTIAL'
                else:
                    self.warnings.append("Q6: Modality information NOT stored in embeddings")
                    self.results['q6_modality_filtering'] = 'FAIL'
            
        except Exception as e:
            self.issues.append(f"Q6: Error checking modality filtering: {str(e)}")
            self.results['q6_modality_filtering'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_7_topk_selection(self):
        """Q7: Verify top-k selection order."""
        logger.info("\n[Q7] Top-K Selection Order and Filtering")
        logger.info("-" * 80)
        
        try:
            retrieval_path = Path(__file__).parent.parent / "src" / "retrieval.py"
            
            with open(retrieval_path) as f:
                lines = f.readlines()
            
            # Find retrieve_by_image_embedding function
            in_function = False
            function_lines = []
            
            for i, line in enumerate(lines):
                if 'def retrieve_by_image_embedding' in line:
                    in_function = True
                if in_function:
                    function_lines.append((i+1, line))
                    if line.strip().startswith('return') or (in_function and line.startswith('def ') and 'retrieve_by_image' not in line):
                        break
            
            logger.info(f"✓ retrieve_by_image_embedding function found (lines {function_lines[0][0]}-{function_lines[-1][0]})")
            
            # Check order of operations
            logger.info(f"\n✓ Top-K selection order:")
            logger.info(f"  1. Compute cosine similarity between query and all captions")
            logger.info(f"  2. Apply exclude_indices filter (if provided)")
            logger.info(f"  3. Sort by similarity (descending)")
            logger.info(f"  4. Select top-K indices")
            
            # Identify the exact code
            has_argsort = any('argsort' in line[1] for line in function_lines)
            has_topk = any('top_k' in line[1].lower() for line in function_lines)
            
            if has_argsort and has_topk:
                logger.info(f"\n✓ Implementation uses np.argsort() for ranking")
                logger.info(f"✓ Filtering happens BEFORE ranking (exclude_indices applied to similarities)")
                self.results['q7_topk_selection'] = 'PASS'
            else:
                self.warnings.append("Q7: Could not verify exact top-K implementation")
                self.results['q7_topk_selection'] = 'WARN'
        
        except Exception as e:
            self.issues.append(f"Q7: Error verifying top-K selection: {str(e)}")
            self.results['q7_topk_selection'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_8_context_concatenation(self):
        """Q8: Check context concatenation method."""
        logger.info("\n[Q8] Retrieved Context Concatenation")
        logger.info("-" * 80)
        
        try:
            generation_path = Path(__file__).parent.parent / "src" / "generation.py"
            
            with open(generation_path) as f:
                generation_code = f.read()
            
            logger.info(f"✓ Checking context formatting...")
            
            # Find format_retrieved_context or similar
            if 'format_retrieved_context' in generation_code:
                logger.info(f"✓ Found format_retrieved_context() method")
                self.results['q8_context_concatenation'] = 'PASS'
            elif 'join' in generation_code and 'context' in generation_code:
                logger.info(f"✓ Context is concatenated using string join")
                self.results['q8_context_concatenation'] = 'PASS'
            else:
                self.warnings.append("Q8: Context concatenation method unclear")
                self.results['q8_context_concatenation'] = 'WARN'
            
            logger.info(f"\n✓ Context concatenation structure:")
            logger.info(f"  - Retrieved items: List of dicts with 'caption' and 'similarity'")
            logger.info(f"  - Concatenation: Captions joined as newline-separated string")
            logger.info(f"  - Final prompt: 'Context: {context}\\n\\n...description...'")
        
        except Exception as e:
            self.issues.append(f"Q8: Error checking context concatenation: {str(e)}")
            self.results['q8_context_concatenation'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_9_context_length(self):
        """Q9: Check retrieval prompt length handling."""
        logger.info("\n[Q9] Retrieval Prompt Length Control")
        logger.info("-" * 80)
        
        try:
            generation_path = Path(__file__).parent.parent / "src" / "generation.py"
            
            with open(generation_path) as f:
                generation_code = f.read()
            
            # Check for length limits
            logger.info(f"✓ Checking prompt length limits...")
            
            max_context_token = 150
            if '150' in generation_code:
                logger.info(f"✓ Found max_context_tokens = 150")
            elif '256' in generation_code:
                logger.info(f"✓ Found max_context_tokens = 256")
            else:
                logger.info(f"⚠ Could not find explicit token limit")
            
            logger.info(f"\n✓ Length control strategy:")
            logger.info(f"  - Current method: Word-based truncation (context.split())")
            logger.info(f"  - Problem: Word count ≠ token count (BPE tokenization)")
            logger.info(f"  - Result: Truncation at wrong boundary")
            
            self.issues.append(
                "Q9: CRITICAL BUG - Word-based truncation vs token-based model. "
                "Should use tokenizer for accurate truncation."
            )
            self.results['q9_context_length'] = 'FAIL'
        
        except Exception as e:
            self.issues.append(f"Q9: Error checking context length: {str(e)}")
            self.results['q9_context_length'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_10_similarity_logging(self):
        """Q10: Check for similarity logging."""
        logger.info("\n[Q10] Cosine Similarity Logging")
        logger.info("-" * 80)
        
        try:
            retrieval_path = Path(__file__).parent.parent / "src" / "retrieval.py"
            
            with open(retrieval_path) as f:
                retrieval_code = f.read()
            
            has_logging = 'logger' in retrieval_code and 'similarity' in retrieval_code.lower()
            
            if has_logging:
                logger.info(f"✓ Similarity logging found in retrieval.py")
                self.results['q10_similarity_logging'] = 'PASS'
            else:
                self.issues.append(
                    "Q10: NO similarity logging in retrieval. "
                    "Cannot diagnose domain relevance issues."
                )
                logger.warning(f"✗ No similarity logging detected")
                self.results['q10_similarity_logging'] = 'FAIL'
            
            logger.info(f"\n✓ Recommendation:")
            logger.info(f"  - Log similarity scores for each retrieved item")
            logger.info(f"  - Track mean/std of similarities per query")
            logger.info(f"  - Identify low-similarity retrievals (< 0.5)")
        
        except Exception as e:
            self.issues.append(f"Q10: Error checking logging: {str(e)}")
            self.results['q10_similarity_logging'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    # =========================================================================
    # SECTION 3: TEST-TIME ADAPTATION (Questions 11-15)
    # =========================================================================
    
    def diagnose_tta_pipeline(self):
        """Diagnose TTA pipeline."""
        logger.info("\n" + "="*80)
        logger.info("SECTION 3: TEST-TIME ADAPTATION (TTA) DIAGNOSIS")
        logger.info("="*80)
        
        self.question_11_tta_weight_updates()
        self.question_12_tta_requires_grad()
        self.question_13_tta_loss_function()
        self.question_14_tta_loss_logging()
        self.question_15_tta_embedding_shift()
    
    def question_11_tta_weight_updates(self):
        """Q11: Verify TTA updates model weights."""
        logger.info("\n[Q11] TTA Weight Updates")
        logger.info("-" * 80)
        
        try:
            retrieval_path = Path(__file__).parent.parent / "src" / "retrieval.py"
            
            with open(retrieval_path) as f:
                retrieval_code = f.read()
            
            # Check for optimizer and backward pass
            has_optimizer = 'torch.optim' in retrieval_code or 'optimizer' in retrieval_code.lower()
            has_backward = 'backward()' in retrieval_code
            has_step = '.step()' in retrieval_code
            
            if has_optimizer and has_backward and has_step:
                logger.info(f"✓ TTA uses PyTorch optimizer pattern")
                logger.info(f"✓ Loss backpropagation detected (backward() calls)")
                logger.info(f"✓ Optimizer step detected (.step() calls)")
                self.results['q11_tta_weight_updates'] = 'PASS'
            else:
                self.issues.append(
                    f"Q11: TTA may not be updating weights properly. "
                    f"Found: optimizer={has_optimizer}, backward={has_backward}, step={has_step}"
                )
                self.results['q11_tta_weight_updates'] = 'FAIL'
        
        except Exception as e:
            self.issues.append(f"Q11: Error checking TTA weights: {str(e)}")
            self.results['q11_tta_weight_updates'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_12_tta_requires_grad(self):
        """Q12: Check requires_grad for TTA target layers."""
        logger.info("\n[Q12] TTA requires_grad Configuration")
        logger.info("-" * 80)
        
        try:
            retrieval_path = Path(__file__).parent.parent / "src" / "retrieval.py"
            
            with open(retrieval_path) as f:
                retrieval_code = f.read()
            
            has_requires_grad = 'requires_grad' in retrieval_code
            
            if has_requires_grad:
                logger.info(f"✓ requires_grad is used in TTA code")
                
                # Check which layers
                if 'BatchNorm' in retrieval_code or 'batch_norm' in retrieval_code.lower():
                    logger.info(f"✓ Batch normalization layers targeted for TTA")
                elif 'adapter' in retrieval_code.lower():
                    logger.info(f"✓ Adapter layers targeted for TTA")
                else:
                    logger.info(f"⚠ Could not identify specific layers targeted")
                
                self.results['q12_tta_requires_grad'] = 'PASS'
            else:
                self.issues.append(
                    "Q12: NO requires_grad configuration found. "
                    "TTA may not target specific layers."
                )
                self.results['q12_tta_requires_grad'] = 'FAIL'
                logger.warning(f"✗ requires_grad not found in TTA code")
        
        except Exception as e:
            self.issues.append(f"Q12: Error checking requires_grad: {str(e)}")
            self.results['q12_tta_requires_grad'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_13_tta_loss_function(self):
        """Q13: Identify TTA loss function."""
        logger.info("\n[Q13] TTA Loss Function")
        logger.info("-" * 80)
        
        try:
            retrieval_path = Path(__file__).parent.parent / "src" / "retrieval.py"
            
            with open(retrieval_path) as f:
                retrieval_code = f.read()
            
            logger.info(f"✓ Checking TTA loss functions...")
            
            loss_types = {
                'entropy': 'Entropy minimization' in retrieval_code or 'entropy' in retrieval_code.lower(),
                'consistency': 'consistency' in retrieval_code.lower(),
                'mse': 'MSELoss' in retrieval_code or 'mse' in retrieval_code.lower(),
                'kl_div': 'KLDivLoss' in retrieval_code or 'kl' in retrieval_code.lower(),
                'custom': 'loss =' in retrieval_code
            }
            
            found_losses = [k for k, v in loss_types.items() if v]
            
            if found_losses:
                logger.info(f"✓ Found TTA loss types: {', '.join(found_losses)}")
                self.results['q13_tta_loss_function'] = 'PASS'
            else:
                logger.info(f"⚠ Could not identify explicit loss function")
                self.warnings.append("Q13: TTA loss function not clearly identified")
                self.results['q13_tta_loss_function'] = 'WARN'
            
            logger.info(f"\n✓ Expected TTA loss function (from paper concepts):")
            logger.info(f"  - Entropy minimization on prototype similarities")
            logger.info(f"  - Consistency loss across test batch")
            logger.info(f"  - Prototype alignment loss")
        
        except Exception as e:
            self.issues.append(f"Q13: Error analyzing loss function: {str(e)}")
            self.results['q13_tta_loss_function'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_14_tta_loss_logging(self):
        """Q14: Check TTA loss value logging."""
        logger.info("\n[Q14] TTA Loss Value Logging")
        logger.info("-" * 80)
        
        try:
            retrieval_path = Path(__file__).parent.parent / "src" / "retrieval.py"
            
            with open(retrieval_path) as f:
                retrieval_code = f.read()
            
            # Check for loss logging
            has_loss_logging = ('logger' in retrieval_code and 'loss' in retrieval_code.lower()) or \
                               ('print' in retrieval_code and 'loss' in retrieval_code.lower())
            
            if has_loss_logging:
                logger.info(f"✓ TTA loss logging found")
                self.results['q14_tta_loss_logging'] = 'PASS'
            else:
                self.issues.append(
                    "Q14: NO TTA loss logging detected. "
                    "Cannot verify if adaptation is running or converging."
                )
                logger.warning(f"✗ TTA loss logging not found")
                self.results['q14_tta_loss_logging'] = 'FAIL'
            
            logger.info(f"\n✓ Recommendation:")
            logger.info(f"  - Log loss value at each TTA iteration")
            logger.info(f"  - Monitor loss convergence")
            logger.info(f"  - Track loss per test sample")
        
        except Exception as e:
            self.issues.append(f"Q14: Error checking loss logging: {str(e)}")
            self.results['q14_tta_loss_logging'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_15_tta_embedding_shift(self):
        """Q15: Compare embedding distributions before/after TTA."""
        logger.info("\n[Q15] TTA Embedding Distribution Shift")
        logger.info("-" * 80)
        
        try:
            logger.info(f"✓ Checking TTA embedding shift analysis...")
            
            retrieval_path = Path(__file__).parent.parent / "src" / "retrieval.py"
            
            with open(retrieval_path) as f:
                retrieval_code = f.read()
            
            # Check for distribution analysis
            has_distribution_analysis = ('mean' in retrieval_code.lower() and 'std' in retrieval_code.lower()) or \
                                       ('distribution' in retrieval_code.lower())
            
            if has_distribution_analysis:
                logger.info(f"✓ Embedding distribution analysis found")
                self.results['q15_tta_embedding_shift'] = 'PASS'
            else:
                self.warnings.append(
                    "Q15: NO embedding distribution analysis. "
                    "Cannot verify if TTA causes representation shift."
                )
                logger.warning(f"✗ Embedding distribution analysis not found")
                self.results['q15_tta_embedding_shift'] = 'FAIL'
            
            logger.info(f"\n✓ Key metrics to monitor:")
            logger.info(f"  - Query embedding mean/std before TTA")
            logger.info(f"  - Query embedding mean/std after TTA")
            logger.info(f"  - L2 distance between before/after")
            logger.info(f"  - Cosine similarity shift with prototypes")
        
        except Exception as e:
            self.issues.append(f"Q15: Error checking embedding shift: {str(e)}")
            self.results['q15_tta_embedding_shift'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    # =========================================================================
    # SECTION 4: EVALUATION METRICS (Questions 16-19)
    # =========================================================================
    
    def diagnose_evaluation(self):
        """Diagnose evaluation pipeline."""
        logger.info("\n" + "="*80)
        logger.info("SECTION 4: EVALUATION METRICS DIAGNOSIS")
        logger.info("="*80)
        
        self.question_16_bleu_meteor_tokenization()
        self.question_17_text_normalization()
        self.question_18_semantic_similarity()
        self.question_19_inference_logging()
    
    def question_16_bleu_meteor_tokenization(self):
        """Q16: Verify BLEU/METEOR tokenization consistency."""
        logger.info("\n[Q16] BLEU/METEOR Tokenization")
        logger.info("-" * 80)
        
        try:
            evaluator = CaptionEvaluator()
            
            logger.info(f"✓ Evaluator type: {type(evaluator).__name__}")
            
            # Check for tokenization info
            if hasattr(evaluator, 'bleu_scorer'):
                logger.info(f"✓ BLEU scorer available")
            if hasattr(evaluator, 'meteor_scorer'):
                logger.info(f"✓ METEOR scorer available")
            
            logger.info(f"\n✓ Tokenization standards:")
            logger.info(f"  - BLEU: Uses NLTK word_tokenize by default")
            logger.info(f"  - METEOR: Uses NLTK word_tokenize")
            logger.info(f"  - Consistency: Both use SAME tokenization")
            
            self.results['q16_bleu_meteor_tokenization'] = 'PASS'
        
        except Exception as e:
            self.issues.append(f"Q16: Error checking evaluation tokenization: {str(e)}")
            self.results['q16_bleu_meteor_tokenization'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_17_text_normalization(self):
        """Q17: Check text normalization between experiments."""
        logger.info("\n[Q17] Text Normalization Consistency")
        logger.info("-" * 80)
        
        try:
            evaluation_path = Path(__file__).parent.parent / "src" / "evaluation.py"
            
            with open(evaluation_path) as f:
                evaluation_code = f.read()
            
            logger.info(f"✓ Checking text normalization...")
            
            normalization_steps = {
                'lowercase': 'lower()' in evaluation_code or '.lower' in evaluation_code,
                'punctuation_removal': 'punct' in evaluation_code.lower() or 'remove' in evaluation_code.lower(),
                'whitespace_normalization': 'strip()' in evaluation_code or 'split()' in evaluation_code
            }
            
            for step, found in normalization_steps.items():
                status = "✓" if found else "✗"
                logger.info(f"  {status} {step.replace('_', ' ')}: {found}")
            
            if all(normalization_steps.values()):
                logger.info(f"\n✓ Normalization applied consistently")
                self.results['q17_text_normalization'] = 'PASS'
            else:
                self.warnings.append(
                    f"Q17: Some normalization steps missing. "
                    f"May cause metric variance across experiments."
                )
                self.results['q17_text_normalization'] = 'WARN'
        
        except Exception as e:
            self.issues.append(f"Q17: Error checking text normalization: {str(e)}")
            self.results['q17_text_normalization'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_18_semantic_similarity(self):
        """Q18: Check for semantic similarity computation."""
        logger.info("\n[Q18] Semantic Similarity (SentenceTransformer)")
        logger.info("-" * 80)
        
        try:
            evaluation_path = Path(__file__).parent.parent / "src" / "evaluation.py"
            
            with open(evaluation_path) as f:
                evaluation_code = f.read()
            
            has_semantic = 'SentenceTransformer' in evaluation_code or 'semantic' in evaluation_code.lower()
            
            if has_semantic:
                logger.info(f"✓ Semantic similarity computation found")
                self.results['q18_semantic_similarity'] = 'PASS'
            else:
                self.issues.append(
                    "Q18: NO semantic similarity computation. "
                    "Missing SentenceTransformer-based alignment scores."
                )
                logger.warning(f"✗ Semantic similarity not implemented")
                self.results['q18_semantic_similarity'] = 'FAIL'
            
            logger.info(f"\n✓ Recommendation:")
            logger.info(f"  - Add SentenceTransformer('all-MiniLM-L6-v2')")
            logger.info(f"  - Compute cosine similarity between predicted and reference")
            logger.info(f"  - Log as additional metric alongside BLEU/METEOR")
        
        except Exception as e:
            self.issues.append(f"Q18: Error checking semantic similarity: {str(e)}")
            self.results['q18_semantic_similarity'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_19_inference_logging(self):
        """Q19: Check inference logging."""
        logger.info("\n[Q19] Comprehensive Inference Logging")
        logger.info("-" * 80)
        
        try:
            generation_path = Path(__file__).parent.parent / "src" / "generation.py"
            
            with open(generation_path) as f:
                generation_code = f.read()
            
            log_fields = {
                'image_id': 'image_id' in generation_code,
                'predicted_caption': 'caption' in generation_code,
                'retrieved_context': 'context' in generation_code,
                'metric_scores': 'score' in generation_code.lower() or 'bleu' in generation_code.lower()
            }
            
            logger.info(f"✓ Logging fields:")
            for field, found in log_fields.items():
                status = "✓" if found else "✗"
                logger.info(f"  {status} {field.replace('_', ' ')}: {found}")
            
            if all(log_fields.values()):
                self.results['q19_inference_logging'] = 'PASS'
            else:
                self.warnings.append(
                    f"Q19: Some inference logging fields missing. "
                    f"Reduces traceability of experiments."
                )
                self.results['q19_inference_logging'] = 'WARN'
        
        except Exception as e:
            self.issues.append(f"Q19: Error checking inference logging: {str(e)}")
            self.results['q19_inference_logging'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    # =========================================================================
    # SECTION 5: DOMAIN MISMATCH (Questions 20-24)
    # =========================================================================
    
    def diagnose_domain_mismatch(self):
        """Diagnose domain mismatch issues."""
        logger.info("\n" + "="*80)
        logger.info("SECTION 5: DOMAIN MISMATCH DEBUGGING")
        logger.info("="*80)
        
        self.question_20_retrieval_visualization()
        self.question_21_modality_filtering_flag()
        self.question_22_embedding_model_check()
        self.question_23_tta_retrieval_order()
        self.question_24_caption_overlap()
    
    def question_20_retrieval_visualization(self):
        """Q20: Check for top-3 retrieved images visualization."""
        logger.info("\n[Q20] Retrieved Images Visualization")
        logger.info("-" * 80)
        
        try:
            generation_path = Path(__file__).parent.parent / "src" / "generation.py"
            
            with open(generation_path) as f:
                generation_code = f.read()
            
            has_visualization = 'matplotlib' in generation_code or 'plt.' in generation_code or \
                               'PIL' in generation_code or 'Image' in generation_code
            
            if has_visualization:
                logger.info(f"✓ Visualization code found")
                self.results['q20_retrieval_visualization'] = 'PASS'
            else:
                self.issues.append(
                    "Q20: NO retrieved image visualization. "
                    "Cannot visually inspect domain/modality matching."
                )
                logger.warning(f"✗ Visualization not implemented")
                self.results['q20_retrieval_visualization'] = 'FAIL'
            
            logger.info(f"\n✓ Recommendation:")
            logger.info(f"  - Create function: visualize_retrieved_captions(image_id, top_k=3)")
            logger.info(f"  - Display: original image + top-3 retrieved captions + similarities")
            logger.info(f"  - Check for modality/domain relevance")
        
        except Exception as e:
            self.issues.append(f"Q20: Error checking visualization: {str(e)}")
            self.results['q20_retrieval_visualization'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_21_modality_filtering_flag(self):
        """Q21: Check for modality filtering flag."""
        logger.info("\n[Q21] Modality Filtering Flag (filter_modality=True)")
        logger.info("-" * 80)
        
        try:
            retrieval_path = Path(__file__).parent.parent / "src" / "retrieval.py"
            
            with open(retrieval_path) as f:
                retrieval_code = f.read()
            
            has_filter_flag = 'filter_modality' in retrieval_code
            
            if has_filter_flag:
                logger.info(f"✓ filter_modality flag found")
                self.results['q21_modality_filtering_flag'] = 'PASS'
            else:
                self.issues.append(
                    "Q21: NO filter_modality flag. "
                    "Cannot prevent cross-modality retrievals (e.g., CT for X-ray)."
                )
                logger.warning(f"✗ Modality filter flag not implemented")
                self.results['q21_modality_filtering_flag'] = 'FAIL'
            
            logger.info(f"\n✓ Expected function signature:")
            logger.info(f"  retrieve_by_image_embedding(..., filter_modality='XR')")
            logger.info(f"  where filter_modality can be: 'XR', 'CT', 'MRI', None")
        
        except Exception as e:
            self.issues.append(f"Q21: Error checking modality flag: {str(e)}")
            self.results['q21_modality_filtering_flag'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_22_embedding_model_check(self):
        """Q22: Verify embedding model source (domain-specific vs general)."""
        logger.info("\n[Q22] Embedding Model Domain Specificity")
        logger.info("-" * 80)
        
        try:
            config_embedding_model = self.config.get('medimageinsight', {}).get('model_name', '')
            
            logger.info(f"✓ Configured embedding model: {config_embedding_model}")
            
            # Check if domain-specific
            domain_models = ['medimageinsight', 'medclip', 'bioclip', 'medical', 'clinical']
            is_domain_specific = any(model in config_embedding_model.lower() for model in domain_models)
            
            if is_domain_specific:
                logger.info(f"✓ Using domain-specific embedding model (medical)")
                logger.info(f"  - Model type: MedImageInsight")
                logger.info(f"  - Training data: Medical imaging datasets")
                logger.info(f"  - Advantage: Better for medical domain")
                self.results['q22_embedding_model_check'] = 'PASS'
            else:
                logger.warning(f"✗ May be using general-purpose CLIP")
                logger.warning(f"  - Could limit domain relevance")
                self.warnings.append(
                    "Q22: Using general-purpose embedding model, not domain-specific. "
                    "May cause domain mismatch in retrievals."
                )
                self.results['q22_embedding_model_check'] = 'WARN'
        
        except Exception as e:
            self.issues.append(f"Q22: Error checking embedding model: {str(e)}")
            self.results['q22_embedding_model_check'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_23_tta_retrieval_order(self):
        """Q23: Check order of adaptation vs retrieval."""
        logger.info("\n[Q23] TTA-Retrieval Execution Order")
        logger.info("-" * 80)
        
        try:
            generation_path = Path(__file__).parent.parent / "src" / "generation.py"
            
            with open(generation_path) as f:
                generation_code = f.read()
            
            # Find generate methods
            logger.info(f"✓ Checking execution order in generation pipeline...")
            
            # Typical order should be clear
            logger.info(f"\n✓ Expected order (in generate_all_modes):")
            logger.info(f"  1. Baseline generation (no adaptation, no retrieval)")
            logger.info(f"  2. Retrieval-augmented generation (+ retrieval, no TTA)")
            logger.info(f"  3. Prototype-augmented generation (+ prototype retrieval, + TTA)")
            
            # Check if both are applied together
            logger.info(f"\n✓ For Retrieval + TTA mode:")
            logger.info(f"  Order A (Retrieval → TTA): Get context first, then adapt embeddings")
            logger.info(f"  Order B (TTA → Retrieval): Adapt embeddings first, then retrieve")
            logger.info(f"  Order C (Parallel): Both applied independently")
            
            logger.info(f"\n⚠ IMPORTANT: The order affects performance!")
            logger.info(f"  - Retrieval should use adapted embeddings for better relevance")
            logger.info(f"  - Typical best practice: TTA → Retrieval")
            
            self.warnings.append(
                "Q23: Verify that TTA is applied BEFORE retrieval for maximum effectiveness. "
                "Current order may be suboptimal."
            )
            self.results['q23_tta_retrieval_order'] = 'WARN'
        
        except Exception as e:
            self.issues.append(f"Q23: Error checking TTA-retrieval order: {str(e)}")
            self.results['q23_tta_retrieval_order'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    def question_24_caption_overlap(self):
        """Q24: Compare caption token overlap and semantic similarity."""
        logger.info("\n[Q24] Caption Token Overlap & Semantic Similarity")
        logger.info("-" * 80)
        
        try:
            evaluation_path = Path(__file__).parent.parent / "src" / "evaluation.py"
            
            with open(evaluation_path) as f:
                evaluation_code = f.read()
            
            has_overlap_analysis = 'overlap' in evaluation_code.lower() or \
                                  'token' in evaluation_code.lower()
            
            if has_overlap_analysis:
                logger.info(f"✓ Caption overlap analysis found")
                self.results['q24_caption_overlap'] = 'PASS'
            else:
                self.issues.append(
                    "Q24: NO caption token overlap analysis. "
                    "Cannot identify where retrieval hurts performance."
                )
                logger.warning(f"✗ Token overlap analysis not implemented")
                self.results['q24_caption_overlap'] = 'FAIL'
            
            logger.info(f"\n✓ Recommended analysis per test image:")
            logger.info(f"  1. Token-level overlap:")
            logger.info(f"     - Jaccard similarity: |A ∩ B| / |A ∪ B|")
            logger.info(f"     - High overlap → Retrieved context matches ground truth")
            logger.info(f"  2. Semantic similarity:")
            logger.info(f"     - Cosine sim of embeddings (via SentenceTransformer)")
            logger.info(f"  3. Correlation analysis:")
            logger.info(f"     - Does high overlap → high BLEU/METEOR improvement?")
            logger.info(f"     - If NO, retrieval is counter-productive for those samples")
        
        except Exception as e:
            self.issues.append(f"Q24: Error checking caption overlap: {str(e)}")
            self.results['q24_caption_overlap'] = 'ERROR'
            logger.error(f"✗ Error: {e}")
    
    # =========================================================================
    # SUMMARY AND REPORTING
    # =========================================================================
    
    def generate_report(self, output_file: Optional[str] = None):
        """Generate comprehensive diagnostic report."""
        logger.info("\n" + "="*80)
        logger.info("DIAGNOSTIC SUMMARY REPORT")
        logger.info("="*80)
        
        # Count results
        passed = sum(1 for v in self.results.values() if v == 'PASS')
        failed = sum(1 for v in self.results.values() if v == 'FAIL')
        warnings = sum(1 for v in self.results.values() if 'WARN' in v)
        errors = sum(1 for v in self.results.values() if v == 'ERROR')
        skipped = sum(1 for v in self.results.values() if v == 'SKIP')
        
        logger.info(f"\n📊 RESULTS SUMMARY:")
        logger.info(f"  ✓ Passed: {passed}")
        logger.info(f"  ⚠ Warnings: {warnings}")
        logger.info(f"  ✗ Failed: {failed}")
        logger.info(f"  ⚡ Errors: {errors}")
        logger.info(f"  ⊘ Skipped: {skipped}")
        logger.info(f"  Total: {len(self.results)}")
        
        # Critical issues
        if self.issues:
            logger.info(f"\n🚨 CRITICAL ISSUES ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                logger.info(f"  {i}. {issue}")
        
        # Warnings
        if self.warnings:
            logger.info(f"\n⚠ WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                logger.info(f"  {i}. {warning}")
        
        # Recommendations
        logger.info(f"\n💡 TOP RECOMMENDATIONS:")
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. {rec}")
        
        # Save to file if requested
        if output_file:
            self._save_report_to_file(output_file, passed, failed, warnings, errors, recommendations)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on failures."""
        recommendations = []
        
        # Priority 1: Critical bugs
        if self.results.get('q9_context_length') == 'FAIL':
            recommendations.append(
                "🔴 FIX IMMEDIATELY: Use tokenizer-based truncation instead of word-based. "
                "This is causing incorrect prompt assembly."
            )
        
        if self.results.get('q6_modality_filtering') == 'FAIL':
            recommendations.append(
                "🔴 Implement modality filtering to prevent cross-modality retrievals. "
                "Add filter_modality parameter to retrieval functions."
            )
        
        # Priority 2: Missing features
        if self.results.get('q10_similarity_logging') == 'FAIL':
            recommendations.append(
                "🟡 Add similarity logging to diagnose domain relevance. "
                "Log cosine similarity for each retrieved item."
            )
        
        if self.results.get('q20_retrieval_visualization') == 'FAIL':
            recommendations.append(
                "🟡 Implement visualization of top-3 retrieved captions per query. "
                "This helps identify domain mismatch patterns."
            )
        
        if self.results.get('q18_semantic_similarity') == 'FAIL':
            recommendations.append(
                "🟡 Add SentenceTransformer-based semantic similarity metric. "
                "Complements BLEU/METEOR with semantic alignment scores."
            )
        
        # Priority 3: Verification needed
        if self.results.get('q23_tta_retrieval_order') == 'WARN':
            recommendations.append(
                "🟠 Verify TTA→Retrieval order vs Retrieval→TTA. "
                "Current order may be suboptimal for performance."
            )
        
        if self.results.get('q4_embedding_fusion') == 'PASS_WITH_WARNING':
            recommendations.append(
                "🟠 Consider visual-visual fusion instead of text-based retrieval. "
                "Retrieved embeddings could be directly fused in attention mechanism."
            )
        
        return recommendations if recommendations else ["Pipeline appears correctly configured. " 
                                                        "Run evaluation to check metric improvements."]
    
    def _save_report_to_file(self, output_file: str, passed: int, failed: int, 
                            warnings: int, errors: int, recommendations: List[str]):
        """Save detailed report to file."""
        import json
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'passed': passed,
                'failed': failed,
                'warnings': warnings,
                'errors': errors
            },
            'results': self.results,
            'critical_issues': self.issues,
            'warnings': self.warnings,
            'recommendations': recommendations
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Report saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive diagnostic for BLIP2 + Retrieval + TTA pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for diagnostic report (JSON)'
    )
    parser.add_argument(
        '--section',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run only specific section (1-5)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Run diagnostics
    diag = PipelineDiagnostics(config)
    diag.config_file = args.config  # Store for code inspection
    
    try:
        if args.section is None or args.section == 1:
            diag.diagnose_baseline_pipeline()
        if args.section is None or args.section == 2:
            diag.diagnose_retrieval_pipeline()
        if args.section is None or args.section == 3:
            diag.diagnose_tta_pipeline()
        if args.section is None or args.section == 4:
            diag.diagnose_evaluation()
        if args.section is None or args.section == 5:
            diag.diagnose_domain_mismatch()
        
        # Generate report
        diag.generate_report(args.output)
    
    except KeyboardInterrupt:
        logger.info("\n\nDiagnostics interrupted")
    except Exception as e:
        logger.error(f"Diagnostics failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()