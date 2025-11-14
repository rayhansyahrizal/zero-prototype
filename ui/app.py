"""
Gradio UI for testing zero-shot medical image captioning.
Quick interface for comparing different retrieval methods.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import gradio as gr
from PIL import Image
import logging

# Add src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedding import EmbeddingGenerator
from retrieval import CaptionRetriever, PrototypeRetriever
from generation import CaptionGenerator
from data_loader import ROCODataLoader, extract_modality
from huggingface_loader import HuggingFaceDataLoader
from evaluation import CaptionEvaluator

logger = logging.getLogger(__name__)

# Global state
state = {
    'config': None,
    'embedding_gen': None,
    'caption_gen': None,
    'retriever': None,
    'prototype_retriever': None,
    'evaluator': None,
    'data_loader': None,
    'embeddings': None,
    'prototypes': None,
    'loaded': False
}


def load_config(config_name):
    """Load specific config file."""
    config_map = {
        'Default': 'config.yaml',
        'Conservative TTA': 'config_tta_conservative.yaml',
        'Moderate TTA': 'config_tta_moderate.yaml',
        'No TTA': 'config_prototype_no_tta.yaml',
        'More Prototypes': 'config_more_prototypes.yaml'
    }

    config_file = config_map.get(config_name, 'config.yaml')
    config_path = Path(__file__).parent.parent / config_file

    if not config_path.exists():
        return None, f"Config {config_file} not found"

    with open(config_path) as f:
        return yaml.safe_load(f), config_file


def load_models(config_name, progress=gr.Progress()):
    """Load models with selected config."""
    if state['loaded']:
        return "Already loaded. Restart to change config."

    config, config_file = load_config(config_name)
    if not config:
        return f"Failed to load {config_name}"

    state['config'] = config
    status = f"Using config: {config_file}\n"

    try:
        # Load embeddings
        progress(0.1, desc="Loading embeddings...")
        status += "Loading embeddings... "
        emb_file = Path(config['output']['embeddings_dir']) / "embeddings.npz"
        if not emb_file.exists():
            return status + "\nEmbeddings not found. Run pipeline first."

        data = np.load(emb_file, allow_pickle=True)
        state['embeddings'] = {
            'img': data['image_embeddings'],
            'txt': data['text_embeddings'],
            'ids': data['image_ids'].tolist(),
            'caps': data['captions'].tolist(),
            'mods': data.get('modalities', ['UNK'] * len(data['image_ids'])).tolist()
        }
        status += f"OK ({len(state['embeddings']['ids'])} samples)\n"
        progress(0.2, desc="Embeddings loaded")

        # Load prototypes
        status += "Loading prototypes... "
        proto_file = Path(config['output']['prototypes_path'])
        if proto_file.exists():
            state['prototypes'] = np.load(proto_file)
            status += f"OK ({len(state['prototypes'])} prototypes)\n"
        else:
            state['prototypes'] = None
            status += "Not found (will use all embeddings)\n"

        # Setup retrievers
        status += "Setting up retrievers... "

        # Regular retriever
        state['retriever'] = CaptionRetriever(
            image_embeddings=state['embeddings']['img'],
            text_embeddings=state['embeddings']['txt'],
            captions=state['embeddings']['caps'],
            image_ids=state['embeddings']['ids'],
            modalities=state['embeddings']['mods'],
            top_k=config['retrieval']['top_k']
        )

        # Prototype retriever (supports TTA)
        if state['prototypes'] is not None:
            state['prototype_retriever'] = PrototypeRetriever(
                image_embeddings=state['embeddings']['img'],
                text_embeddings=state['embeddings']['txt'],
                captions=state['embeddings']['caps'],
                image_ids=state['embeddings']['ids'],
                modalities=state['embeddings']['mods'],
                prototype_indices=state['prototypes'],
                top_k=config['retrieval']['top_k']
            )
            status += "OK (regular + prototype)\n"
        else:
            state['prototype_retriever'] = None
            status += "OK (regular only)\n"

        # Load MedImageInsight
        progress(0.3, desc="Loading MedImageInsight...")
        status += "Loading MedImageInsight... "
        try:
            state['embedding_gen'] = EmbeddingGenerator(config)
            state['embedding_gen'].load_model()
            status += "OK\n"
        except Exception as e:
            status += f"ERROR: {e}\n"
            return status + "\nFailed to load MedImageInsight. Check model path in config."

        # Load BLIP2
        progress(0.5, desc="Loading BLIP2...")
        status += "Loading BLIP2... "

        # Check if BLIP2 is cached
        import os
        from transformers import Blip2Processor
        model_name = config['blip2']['model_name']
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

        # Estimate if model needs download
        try:
            # Try to load processor with local_files_only
            Blip2Processor.from_pretrained(model_name, local_files_only=True)
            status += "(using cached model) "
        except:
            status += "\n   ⚠️  BLIP2 not cached, downloading (~5GB, may take 5-10 min)...\n   "
            progress(0.5, desc="Downloading BLIP2 model (this may take a while)...")

        try:
            state['caption_gen'] = CaptionGenerator(config)
            state['caption_gen'].load_model()
            status += "OK\n"
        except Exception as e:
            status += f"ERROR: {e}\n"
            return status + "\nFailed to load BLIP2. Check internet connection or model name."

        # Setup evaluator
        progress(0.7, desc="Setting up evaluator...")
        status += "Setting up evaluator... "
        state['evaluator'] = CaptionEvaluator(
            text_embedding_model=state['embedding_gen'].model,
            use_semantic_similarity=True
        )
        status += "OK\n"

        # Load dataset metadata only (lazy loading for speed)
        progress(0.8, desc="Setting up dataset...")
        status += "Setting up dataset... "
        ds_source = config['dataset'].get('source', 'local')

        if ds_source == 'huggingface':
            ds_name = config['dataset'].get('name', 'eltorio/ROCOv2-radiology')

            # Don't load images yet - too slow!
            # Just store HF loader for lazy loading
            hf_loader = HuggingFaceDataLoader(
                dataset_name=ds_name,
                split=config['dataset']['split'],
                max_samples=min(100, config['dataset'].get('max_samples', 100)),  # Limit for UI
                stream=False
            )

            # Load metadata only (no images!)
            samples_meta = []
            for sample in hf_loader.iterate(max_samples=min(100, config['dataset'].get('max_samples', 100))):
                norm = hf_loader.normalize_sample(sample)
                samples_meta.append({
                    'image_id': norm['image_id'],
                    'image_data': norm['image'],  # Store image data, not loaded image
                    'caption': norm['caption'],
                    'hf_loader': hf_loader
                })

            class DataWrapper:
                def __init__(self, samples):
                    self.samples = samples
                def __len__(self):
                    return len(self.samples)
                def __getitem__(self, idx):
                    return {
                        'image_id': self.samples[idx]['image_id'],
                        'image_path': None,
                        'caption': self.samples[idx]['caption']
                    }
                def get_all_samples(self):
                    return [self[i] for i in range(len(self))]
                def load_image(self, image_path=None, sample_idx=None):
                    # Lazy load image only when requested
                    if sample_idx is not None:
                        s = self.samples[sample_idx]
                        return s['hf_loader'].load_image(s['image_data'])
                    return None

            state['data_loader'] = DataWrapper(samples_meta)
        else:
            state['data_loader'] = ROCODataLoader(
                root_dir=config['dataset']['root_dir'],
                split=config['dataset']['split'],
                modality=config['dataset']['modality'],
                max_samples=min(100, config['dataset'].get('max_samples', 100))  # Limit for UI
            )

        status += f"OK ({len(state['data_loader'])} samples available)\n"
        progress(0.9, desc="Almost done...")

        progress(1.0, desc="Complete!")
        state['loaded'] = True
        status += "\nAll models loaded!"
        return status

    except Exception as e:
        logger.error(f"Error loading: {e}", exc_info=True)
        return f"{status}\nError: {e}"


def generate(image, gt_caption, method, use_tta, sim_threshold, lr, steps):
    """Generate caption with specified method."""
    if not state['loaded']:
        return "Load models first", "", "", ""

    if image is None:
        return "Upload an image first", "", "", ""

    try:
        # Encode image
        img_b64 = state['embedding_gen'].image_to_base64(image)
        output = state['embedding_gen'].model.encode(images=[img_b64])
        img_emb = output['image_embeddings'][0]
        img_emb = img_emb / np.linalg.norm(img_emb)

        caption = ""
        context = ""
        eval_text = ""

        # Generate based on method
        if method == "Baseline":
            caption = state['caption_gen'].generate_caption(image)
            context = "No retrieval used"

        elif method == "Retrieval":
            retrieved = state['retriever'].retrieve_by_image_embedding(
                img_emb,
                filter_modality=None
            )

            # Apply similarity threshold
            if sim_threshold > 0:
                retrieved = [r for r in retrieved if r['similarity'] >= sim_threshold]

            if retrieved:
                caption = state['caption_gen'].generate_caption(
                    image,
                    retrieved_captions=retrieved,
                    similarity_threshold=sim_threshold if sim_threshold > 0 else None
                )
                context = "\n".join([
                    f"{i+1}. [{r['similarity']:.3f}] {r['caption'][:100]}"
                    for i, r in enumerate(retrieved[:5])
                ])
            else:
                caption = state['caption_gen'].generate_caption(image)
                context = "All retrievals filtered by threshold"

        elif method == "Prototype":
            if state['prototype_retriever'] is None:
                return "Prototype retriever not available", "", "", ""

            retrieved = state['prototype_retriever'].retrieve_by_image_embedding(
                img_emb,
                use_adaptation=False,
                filter_modality=None
            )

            # Apply similarity threshold
            if sim_threshold > 0:
                retrieved = [r for r in retrieved if r['similarity'] >= sim_threshold]

            if retrieved:
                caption = state['caption_gen'].generate_caption(
                    image,
                    retrieved_captions=retrieved,
                    similarity_threshold=sim_threshold if sim_threshold > 0 else None
                )
                context = "\n".join([
                    f"{i+1}. [{r['similarity']:.3f}] {r['caption'][:100]}"
                    for i, r in enumerate(retrieved[:5])
                ])
            else:
                caption = state['caption_gen'].generate_caption(image)
                context = "All retrievals filtered by threshold"

        elif method == "Prototype+TTA":
            if state['prototype_retriever'] is None:
                return "Prototype retriever not available", "", "", ""

            # Check if TTA is enabled
            if not use_tta:
                return "Enable TTA checkbox first", "", "", ""

            tta_params = {
                'learning_rate': lr,
                'num_steps': int(steps),
                'top_k_for_loss': state['config'].get('tta', {}).get('top_k_for_loss', 5),
                'weight_variance': state['config'].get('tta', {}).get('weight_variance', 0.1),
                'weight_entropy': state['config'].get('tta', {}).get('weight_entropy', 0.01)
            }

            retrieved = state['prototype_retriever'].retrieve_by_image_embedding(
                img_emb,
                use_adaptation=True,
                adaptation_params=tta_params,
                filter_modality=None
            )

            # Apply similarity threshold
            if sim_threshold > 0:
                retrieved = [r for r in retrieved if r['similarity'] >= sim_threshold]

            if retrieved:
                caption = state['caption_gen'].generate_caption(
                    image,
                    retrieved_captions=retrieved,
                    similarity_threshold=sim_threshold if sim_threshold > 0 else None
                )
                context = f"TTA: LR={lr}, Steps={int(steps)}\n\n"
                context += "\n".join([
                    f"{i+1}. [{r['similarity']:.3f}] {r['caption'][:100]}"
                    for i, r in enumerate(retrieved[:5])
                ])
            else:
                caption = state['caption_gen'].generate_caption(image)
                context = "All retrievals filtered by threshold"

        # Evaluate if ground truth provided
        if gt_caption.strip() and caption:
            scores = state['evaluator'].evaluate_single(gt_caption, caption)
            eval_text = (
                f"BLEU-4: {scores['bleu_4']:.4f}\n"
                f"METEOR: {scores['meteor']:.4f}\n"
            )
            if 'semantic_similarity' in scores:
                eval_text += f"Semantic: {scores['semantic_similarity']:.4f}"

        # Info
        info = f"Method: {method}"
        if method != "Baseline":
            info += f"\nSimilarity threshold: {sim_threshold if sim_threshold > 0 else 'None'}"
        if method == "Prototype+TTA" and use_tta:
            info += f"\nTTA enabled (LR={lr}, Steps={int(steps)})"

        return caption, context, eval_text, info

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return f"Error: {e}", "", "", ""


def compare_all(image, gt_caption, sim_threshold, lr, steps):
    """Generate captions using all methods for comparison."""
    if not state['loaded']:
        return ["Load models first"] * 8 + [""]

    if image is None:
        return ["Upload an image first"] * 8 + [""]

    try:
        # Encode image once
        img_b64 = state['embedding_gen'].image_to_base64(image)
        output = state['embedding_gen'].model.encode(images=[img_b64])
        img_emb = output['image_embeddings'][0]
        img_emb = img_emb / np.linalg.norm(img_emb)

        results = {}

        # 1. Baseline
        results['baseline'] = {
            'caption': state['caption_gen'].generate_caption(image),
            'context': 'No retrieval'
        }

        # 2. Retrieval
        retrieved = state['retriever'].retrieve_by_image_embedding(img_emb, filter_modality=None)
        if sim_threshold > 0:
            retrieved = [r for r in retrieved if r['similarity'] >= sim_threshold]

        if retrieved:
            results['retrieval'] = {
                'caption': state['caption_gen'].generate_caption(
                    image,
                    retrieved_captions=retrieved,
                    similarity_threshold=sim_threshold if sim_threshold > 0 else None
                ),
                'context': "\n".join([
                    f"{i+1}. [{r['similarity']:.3f}] {r['caption'][:80]}"
                    for i, r in enumerate(retrieved[:3])
                ])
            }
        else:
            results['retrieval'] = {
                'caption': state['caption_gen'].generate_caption(image),
                'context': 'Filtered by threshold'
            }

        # 3. Prototype (if available)
        if state['prototype_retriever'] is not None:
            retrieved = state['prototype_retriever'].retrieve_by_image_embedding(
                img_emb,
                use_adaptation=False,
                filter_modality=None
            )
            if sim_threshold > 0:
                retrieved = [r for r in retrieved if r['similarity'] >= sim_threshold]

            if retrieved:
                results['prototype'] = {
                    'caption': state['caption_gen'].generate_caption(
                        image,
                        retrieved_captions=retrieved,
                        similarity_threshold=sim_threshold if sim_threshold > 0 else None
                    ),
                    'context': "\n".join([
                        f"{i+1}. [{r['similarity']:.3f}] {r['caption'][:80]}"
                        for i, r in enumerate(retrieved[:3])
                    ])
                }
            else:
                results['prototype'] = {
                    'caption': state['caption_gen'].generate_caption(image),
                    'context': 'Filtered by threshold'
                }

            # 4. Prototype+TTA
            tta_params = {
                'learning_rate': lr,
                'num_steps': int(steps),
                'top_k_for_loss': state['config'].get('tta', {}).get('top_k_for_loss', 5),
                'weight_variance': state['config'].get('tta', {}).get('weight_variance', 0.1),
                'weight_entropy': state['config'].get('tta', {}).get('weight_entropy', 0.01)
            }

            retrieved = state['prototype_retriever'].retrieve_by_image_embedding(
                img_emb,
                use_adaptation=True,
                adaptation_params=tta_params,
                filter_modality=None
            )
            if sim_threshold > 0:
                retrieved = [r for r in retrieved if r['similarity'] >= sim_threshold]

            if retrieved:
                results['tta'] = {
                    'caption': state['caption_gen'].generate_caption(
                        image,
                        retrieved_captions=retrieved,
                        similarity_threshold=sim_threshold if sim_threshold > 0 else None
                    ),
                    'context': f"TTA: LR={lr}, Steps={int(steps)}\n" + "\n".join([
                        f"{i+1}. [{r['similarity']:.3f}] {r['caption'][:80]}"
                        for i, r in enumerate(retrieved[:3])
                    ])
                }
            else:
                results['tta'] = {
                    'caption': state['caption_gen'].generate_caption(image),
                    'context': 'Filtered by threshold'
                }
        else:
            results['prototype'] = {'caption': 'Not available', 'context': 'No prototypes'}
            results['tta'] = {'caption': 'Not available', 'context': 'No prototypes'}

        # Evaluate all if ground truth provided
        eval_table = ""
        if gt_caption.strip():
            eval_lines = ["Method | BLEU-4 | METEOR | Semantic"]
            eval_lines.append("-" * 45)

            for method_name, method_key in [
                ('Baseline', 'baseline'),
                ('Retrieval', 'retrieval'),
                ('Prototype', 'prototype'),
                ('Prototype+TTA', 'tta')
            ]:
                if method_key in results and results[method_key]['caption'] not in ['Not available', 'Load models first', 'Upload an image first']:
                    scores = state['evaluator'].evaluate_single(gt_caption, results[method_key]['caption'])
                    eval_lines.append(
                        f"{method_name:13s} | {scores['bleu_4']:.4f} | {scores['meteor']:.4f} | "
                        f"{scores.get('semantic_similarity', 0):.4f}"
                    )
                else:
                    eval_lines.append(f"{method_name:13s} | N/A    | N/A    | N/A")

            eval_table = "\n".join(eval_lines)

        return (
            results.get('baseline', {}).get('caption', ''),
            results.get('baseline', {}).get('context', ''),
            results.get('retrieval', {}).get('caption', ''),
            results.get('retrieval', {}).get('context', ''),
            results.get('prototype', {}).get('caption', ''),
            results.get('prototype', {}).get('context', ''),
            results.get('tta', {}).get('caption', ''),
            results.get('tta', {}).get('context', ''),
            eval_table
        )

    except Exception as e:
        logger.error(f"Error comparing: {e}", exc_info=True)
        err = f"Error: {e}"
        return [err] * 8 + [""]


def load_test_img(idx):
    """Load test image from dataset."""
    if not state['loaded']:
        return None, ""

    try:
        if idx >= len(state['data_loader']):
            return None, "Invalid index"

        sample = state['data_loader'][idx]

        # Handle HF vs local
        ds_source = state['config']['dataset'].get('source', 'local')
        if ds_source == 'huggingface':
            img = state['data_loader'].load_image(sample_idx=idx)
        else:
            img = state['data_loader'].load_image(sample['image_path'])

        return img, sample['caption']
    except Exception as e:
        return None, f"Error: {e}"


def create_ui():
    """Build Gradio interface."""

    with gr.Blocks(title="Medical Image Captioning Demo") as app:
        gr.Markdown("# Medical Image Captioning")
        gr.Markdown("Test single method or compare all methods side-by-side")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Setup")

                config_choice = gr.Radio(
                    choices=[
                        'Default',
                        'Conservative TTA',
                        'Moderate TTA',
                        'No TTA',
                        'More Prototypes'
                    ],
                    value='Default',
                    label="Config"
                )

                load_btn = gr.Button("Load Models", variant="primary")
                status_box = gr.Textbox(
                    label="Status",
                    lines=12,
                    placeholder="Click Load Models to start"
                )

            with gr.Column(scale=1):
                gr.Markdown("### Image Input")

                with gr.Tabs():
                    with gr.Tab("Upload"):
                        img_input = gr.Image(type="pil", label="Image")
                    with gr.Tab("Dataset"):
                        test_idx = gr.Slider(0, 100, 0, step=1, label="Index")
                        test_load_btn = gr.Button("Load")

                gt_box = gr.Textbox(
                    label="Ground Truth (optional)",
                    lines=2,
                    placeholder="Reference caption for evaluation"
                )

        gr.Markdown("---")

        # Settings (shared between tabs)
        gr.Markdown("### Settings")
        with gr.Row():
            tta_lr = gr.Slider(0.00001, 0.01, 0.001, step=0.00001, label="TTA Learning Rate")
            tta_steps = gr.Slider(1, 50, 10, step=1, label="TTA Steps")
            sim_thresh = gr.Slider(0, 1, 0, step=0.05, label="Similarity Threshold (0=off)")

        gr.Markdown("---")

        # Mode tabs: Single Method vs Compare All
        with gr.Tabs():
            # Tab 1: Single Method
            with gr.Tab("Single Method"):
                with gr.Row():
                    with gr.Column():
                        method_choice = gr.Radio(
                            choices=['Baseline', 'Retrieval', 'Prototype', 'Prototype+TTA'],
                            value='Baseline',
                            label="Select Method"
                        )
                        tta_enable = gr.Checkbox(label="Enable TTA", value=False)
                        gen_btn = gr.Button("Generate Caption", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        info_box = gr.Textbox(label="Info", lines=2)
                        caption_box = gr.Textbox(label="Generated Caption", lines=3)

                        with gr.Row():
                            context_box = gr.Textbox(label="Retrieved Context", lines=8)
                            eval_box = gr.Textbox(label="Evaluation", lines=8)

            # Tab 2: Compare All
            with gr.Tab("Compare All Methods"):
                compare_btn = gr.Button("Generate All Methods", variant="primary", size="lg")

                gr.Markdown("### Results Comparison")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Baseline**")
                        baseline_cap = gr.Textbox(label="Caption", lines=2)
                        baseline_ctx = gr.Textbox(label="Context", lines=4)

                    with gr.Column():
                        gr.Markdown("**Retrieval**")
                        retrieval_cap = gr.Textbox(label="Caption", lines=2)
                        retrieval_ctx = gr.Textbox(label="Context", lines=4)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Prototype**")
                        prototype_cap = gr.Textbox(label="Caption", lines=2)
                        prototype_ctx = gr.Textbox(label="Context", lines=4)

                    with gr.Column():
                        gr.Markdown("**Prototype+TTA**")
                        tta_cap = gr.Textbox(label="Caption", lines=2)
                        tta_ctx = gr.Textbox(label="Context", lines=4)

                gr.Markdown("### Evaluation Scores")
                eval_table = gr.Textbox(label="Score Comparison", lines=8)

        gr.Markdown("---")
        gr.Markdown("""
        **Methods:**
        - Baseline: BLIP2 only
        - Retrieval: Full dataset retrieval
        - Prototype: FPS-sampled prototypes
        - Prototype+TTA: Prototypes with test-time adaptation

        **Tips:**
        - Use "Compare All" to see all methods side-by-side
        - Adjust TTA settings and threshold before generating
        - Ground truth enables automatic evaluation
        """)

        # Wire up events
        load_btn.click(
            load_models,
            inputs=[config_choice],
            outputs=[status_box]
        )

        test_load_btn.click(
            load_test_img,
            inputs=[test_idx],
            outputs=[img_input, gt_box]
        )

        gen_btn.click(
            generate,
            inputs=[
                img_input,
                gt_box,
                method_choice,
                tta_enable,
                sim_thresh,
                tta_lr,
                tta_steps
            ],
            outputs=[caption_box, context_box, eval_box, info_box]
        )

        compare_btn.click(
            compare_all,
            inputs=[
                img_input,
                gt_box,
                sim_thresh,
                tta_lr,
                tta_steps
            ],
            outputs=[
                baseline_cap,
                baseline_ctx,
                retrieval_cap,
                retrieval_ctx,
                prototype_cap,
                prototype_ctx,
                tta_cap,
                tta_ctx,
                eval_table
            ]
        )

    return app


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()

    app = create_ui()

    logger.info(f"Starting UI on port {args.port}")
    logger.info("Note: First-time model loading will take 2-3 minutes")

    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        max_threads=10  # Allow multiple concurrent requests
    )


if __name__ == "__main__":
    main()
