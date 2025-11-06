"""
Gradio UI for interactive testing of zero-shot medical image captioning.

Features:
- Upload or select test images
- Generate captions using three methods (baseline, retrieval, prototype)
- Display retrieved captions and images
- Compare results side-by-side
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import gradio as gr
from PIL import Image
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedding import EmbeddingGenerator
from retrieval import CaptionRetriever, PrototypeRetriever
from generation import CaptionGenerator
from data_loader import ROCODataLoader
from huggingface_loader import HuggingFaceDataLoader
from evaluation import CaptionEvaluator

# Global variables to store loaded models and data
state = {
    'config': None,
    'embedding_generator': None,
    'caption_generator': None,
    'retriever': None,
    'prototype_retriever': None,
    'evaluator': None,
    'data_loader': None,
    'embeddings_data': None,
    'prototype_indices': None,
    'loaded': False
}

logger = logging.getLogger(__name__)


def load_models_and_data():
    """Load all models and data."""
    if state['loaded']:
        return "✓ Models already loaded"
    
    try:
        # Load config
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            state['config'] = yaml.safe_load(f)
        
        status = "Loading resources...\n"
        yield status
        
        # Load embeddings
        status += "\n1. Loading embeddings..."
        yield status
        
        embeddings_file = Path(state['config']['output']['embeddings_dir']) / "embeddings.npz"
        if not embeddings_file.exists():
            status += "\n   ✗ Embeddings not found. Please run main pipeline first."
            yield status
            return
        
        data = np.load(embeddings_file, allow_pickle=True)
        state['embeddings_data'] = {
            'image_embeddings': data['image_embeddings'],
            'text_embeddings': data['text_embeddings'],
            'image_ids': data['image_ids'].tolist(),
            'captions': data['captions'].tolist()
        }
        status += f"\n   ✓ Loaded {len(state['embeddings_data']['image_ids'])} embeddings"
        yield status
        
        # Load prototypes
        status += "\n2. Loading prototypes..."
        yield status
        
        prototypes_path = Path(state['config']['output']['prototypes_path'])
        if prototypes_path.exists():
            state['prototype_indices'] = np.load(prototypes_path)
            status += f"\n   ✓ Loaded {len(state['prototype_indices'])} prototypes"
        else:
            status += "\n   ⚠ Prototypes not found, will skip prototype mode"
        yield status
        
        # Setup retrievers
        status += "\n3. Setting up retrievers..."
        yield status
        
        state['retriever'] = CaptionRetriever(
            image_embeddings=state['embeddings_data']['image_embeddings'],
            text_embeddings=state['embeddings_data']['text_embeddings'],
            captions=state['embeddings_data']['captions'],
            image_ids=state['embeddings_data']['image_ids'],
            top_k=state['config']['retrieval']['top_k']
        )
        
        if state['prototype_indices'] is not None:
            state['prototype_retriever'] = PrototypeRetriever(
                image_embeddings=state['embeddings_data']['image_embeddings'],
                text_embeddings=state['embeddings_data']['text_embeddings'],
                captions=state['embeddings_data']['captions'],
                image_ids=state['embeddings_data']['image_ids'],
                prototype_indices=state['prototype_indices'],
                top_k=state['config']['retrieval']['top_k']
            )
        
        status += "\n   ✓ Retrievers ready"
        yield status
        
        # Load MedImageInsight
        status += "\n4. Loading MedImageInsight..."
        yield status
        
        state['embedding_generator'] = EmbeddingGenerator(state['config'])
        state['embedding_generator'].load_model()
        
        status += "\n   ✓ MedImageInsight loaded"
        yield status
        
        # Load BLIP2
        status += "\n5. Loading BLIP2 (this may take a while)..."
        yield status
        
        state['caption_generator'] = CaptionGenerator(state['config'])
        state['caption_generator'].load_model()
        
        status += "\n   ✓ BLIP2 loaded"
        yield status
        
        # Load evaluator
        state['evaluator'] = CaptionEvaluator()
        
        # Load data loader for test images
        status += "\n6. Loading dataset..."
        yield status
        
        dataset_source = state['config']['dataset'].get('source', 'local')
        
        if dataset_source == 'huggingface':
            dataset_name = state['config']['dataset'].get('name', 'eltorio/ROCOv2-radiology')
            status += f"\n   Using HF dataset: {dataset_name}"
            yield status
            
            hf_loader = HuggingFaceDataLoader(
                dataset_name=dataset_name,
                split=state['config']['dataset']['split'],
                max_samples=state['config']['dataset'].get('max_samples'),
                stream=False
            )
            
            # Convert to standard format
            samples = []
            for sample in hf_loader.iterate(max_samples=state['config']['dataset'].get('max_samples')):
                normalized = hf_loader.normalize_sample(sample)
                samples.append({
                    'image_id': normalized['image_id'],
                    'image': hf_loader.load_image(normalized['image']),
                    'caption': normalized['caption']
                })
            
            # Create wrapper
            class HFDataWrapper:
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
                    if sample_idx is not None:
                        return self.samples[sample_idx]['image']
                    return self.samples[0]['image'] if self.samples else None
            
            state['data_loader'] = HFDataWrapper(samples)
            state['dataset_source'] = 'huggingface'
        else:
            state['data_loader'] = ROCODataLoader(
                root_dir=state['config']['dataset']['root_dir'],
                split=state['config']['dataset']['split'],
                modality=state['config']['dataset']['modality'],
                max_samples=state['config']['dataset'].get('max_samples')
            )
            state['dataset_source'] = 'local'
        
        status += f"\n   ✓ Loaded {len(state['data_loader'])} samples"
        yield status
        
        state['loaded'] = True
        status += "\n\n✓ All models loaded successfully!"
        yield status
        
    except Exception as e:
        yield f"\n✗ Error loading models: {e}"


def generate_captions(image, ground_truth=""):
    """Generate captions for uploaded image using all three methods."""
    if not state['loaded']:
        return "⚠ Please load models first", "", "", "", "", ""
    
    try:
        # Encode image
        image_base64 = state['embedding_generator'].image_to_base64(image)
        output = state['embedding_generator'].model.encode(images=[image_base64])
        image_embedding = output['image_embeddings'][0]
        
        # Normalize
        image_embedding = image_embedding / np.linalg.norm(image_embedding)
        
        # 1. Baseline generation
        baseline_caption = state['caption_generator'].generate_caption(image)
        
        # 2. Retrieval-augmented generation
        retrieved = state['retriever'].retrieve_by_image_embedding(image_embedding)
        retrieval_context = state['retriever'].format_retrieved_context(retrieved)
        retrieval_caption = state['caption_generator'].generate_caption(
            image, context=retrieval_context
        )
        
        # 3. Prototype-augmented generation
        if state['prototype_retriever'] is not None:
            proto_retrieved = state['prototype_retriever'].retrieve_by_image_embedding(
                image_embedding
            )
            proto_context = state['prototype_retriever'].format_retrieved_context(
                proto_retrieved
            )
            prototype_caption = state['caption_generator'].generate_caption(
                image, context=proto_context
            )
        else:
            prototype_caption = "⚠ Prototype mode not available"
            proto_context = ""
        
        # Evaluate if ground truth provided
        evaluation_text = ""
        if ground_truth.strip():
            eval_results = []
            
            for method, caption in [
                ("Baseline", baseline_caption),
                ("Retrieval", retrieval_caption),
                ("Prototype", prototype_caption)
            ]:
                if caption and not caption.startswith("⚠"):
                    scores = state['evaluator'].evaluate_single(ground_truth, caption)
                    eval_results.append(f"{method}:")
                    eval_results.append(f"  BLEU-4: {scores['bleu_4']:.4f}")
                    eval_results.append(f"  METEOR: {scores['meteor']:.4f}")
                    eval_results.append("")
            
            evaluation_text = "\n".join(eval_results)
        
        return (
            baseline_caption,
            retrieval_caption,
            prototype_caption,
            retrieval_context,
            proto_context,
            evaluation_text
        )
        
    except Exception as e:
        return f"✗ Error: {e}", "", "", "", "", ""


def load_test_image(image_idx):
    """Load a test image from the dataset."""
    if not state['loaded'] or state['data_loader'] is None:
        return None, ""
    
    try:
        if image_idx >= len(state['data_loader']):
            return None, "Invalid image index"
        
        sample = state['data_loader'][image_idx]
        
        # Handle both local and HF datasets
        if state.get('dataset_source') == 'huggingface':
            image = state['data_loader'].load_image(sample_idx=image_idx)
        else:
            image = state['data_loader'].load_image(sample['image_path'])
        
        caption = sample['caption']
        
        return image, caption
    except Exception as e:
        return None, f"Error loading image: {e}"


def get_retrieved_images(retrieval_context):
    """Get images corresponding to retrieved captions."""
    if not state['loaded'] or not retrieval_context:
        return []
    
    try:
        # Parse image IDs from context
        lines = retrieval_context.split('\n')
        image_ids = []
        
        for line in lines:
            # Extract image_id from line (assuming format: "N. caption")
            if '. ' in line:
                # Get the caption part
                caption = line.split('. ', 1)[1].split(' (sim:')[0]
                # Find matching caption in embeddings data
                if caption in state['embeddings_data']['captions']:
                    idx = state['embeddings_data']['captions'].index(caption)
                    image_ids.append(state['embeddings_data']['image_ids'][idx])
        
        # Load images
        images = []
        for img_id in image_ids[:3]:  # Show top 3
            for idx, sample in enumerate(state['data_loader'].get_all_samples()):
                if sample['image_id'] == img_id:
                    try:
                        if state.get('dataset_source') == 'huggingface':
                            img = state['data_loader'].load_image(sample_idx=idx)
                        else:
                            img = state['data_loader'].load_image(sample['image_path'])
                        images.append(img)
                    except:
                        pass
                    break
        
        return images
    except Exception as e:
        logger.error(f"Error getting retrieved images: {e}")
        return []


def create_ui():
    """Create Gradio interface."""
    
    with gr.Blocks(title="Zero-Shot Medical Image Captioning", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🏥 Zero-Shot Medical Image Captioning
            
            Generate medical image captions using three methods:
            - **Baseline**: BLIP2 only
            - **Retrieval**: BLIP2 + retrieved similar captions
            - **Prototype**: BLIP2 + prototype sampling
            """
        )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1️⃣ Load Models")
                load_btn = gr.Button("Load Models & Data", variant="primary")
                load_status = gr.Textbox(
                    label="Status",
                    lines=15,
                    placeholder="Click 'Load Models & Data' to start..."
                )
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 2️⃣ Input Image")
                
                with gr.Tabs():
                    with gr.Tab("Upload"):
                        input_image = gr.Image(
                            type="pil",
                            label="Medical Image"
                        )
                    
                    with gr.Tab("Test Images"):
                        test_idx = gr.Slider(
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0,
                            label="Test Image Index"
                        )
                        load_test_btn = gr.Button("Load Test Image")
                
                ground_truth = gr.Textbox(
                    label="Ground Truth Caption (optional, for evaluation)",
                    lines=3,
                    placeholder="Enter the reference caption if available..."
                )
                
                generate_btn = gr.Button("🚀 Generate Captions", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### 3️⃣ Generated Captions")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Baseline**")
                        baseline_output = gr.Textbox(
                            label="BLIP2 Only",
                            lines=3
                        )
                    
                    with gr.Column():
                        gr.Markdown("**Retrieval**")
                        retrieval_output = gr.Textbox(
                            label="BLIP2 + Retrieval",
                            lines=3
                        )
                    
                    with gr.Column():
                        gr.Markdown("**Prototype**")
                        prototype_output = gr.Textbox(
                            label="BLIP2 + Prototypes",
                            lines=3
                        )
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Evaluation Metrics")
                evaluation_output = gr.Textbox(
                    label="Scores (if ground truth provided)",
                    lines=8
                )
            
            with gr.Column():
                gr.Markdown("### 🔍 Retrieved Context (Retrieval)")
                retrieval_context = gr.Textbox(
                    label="Retrieved Captions",
                    lines=8
                )
            
            with gr.Column():
                gr.Markdown("### 🎯 Retrieved Context (Prototype)")
                prototype_context = gr.Textbox(
                    label="Retrieved Prototype Captions",
                    lines=8
                )
        
        gr.Markdown(
            """
            ---
            ### ℹ️ Instructions
            1. Click **Load Models & Data** and wait for all models to load
            2. Upload an image or load a test image from the dataset
            3. Optionally provide ground truth caption for evaluation
            4. Click **Generate Captions** to see results from all three methods
            5. Compare the generated captions and evaluation metrics
            
            **Note**: First-time loading may take several minutes as models are initialized.
            """
        )
        
        # Event handlers
        load_btn.click(
            fn=load_models_and_data,
            inputs=[],
            outputs=[load_status]
        )
        
        load_test_btn.click(
            fn=load_test_image,
            inputs=[test_idx],
            outputs=[input_image, ground_truth]
        )
        
        generate_btn.click(
            fn=generate_captions,
            inputs=[input_image, ground_truth],
            outputs=[
                baseline_output,
                retrieval_output,
                prototype_output,
                retrieval_context,
                prototype_context,
                evaluation_output
            ]
        )
    
    return demo


def main():
    """Launch Gradio app."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Launch Gradio UI")
    parser.add_argument('--share', action='store_true', help='Create shareable link')
    parser.add_argument('--port', type=int, default=7860, help='Port to run on')
    
    args = parser.parse_args()
    
    demo = create_ui()
    
    logger.info("Launching Gradio interface...")
    logger.info(f"Access the UI at: http://localhost:{args.port}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
