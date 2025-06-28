import gradio as gr
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
import gc
import warnings
warnings.filterwarnings("ignore")

# Use reliable and well-tested models
MODEL_NAMES = {
    "GPT-2": "gpt2",
    "DistilGPT2": "distilgpt2",
    "T5-Small": "t5-small"
}

model_pipes = {}
tokenizers = {}

def clear_memory():
    """Clear GPU and CPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(model_name):
    """Load model with comprehensive error handling"""
    if model_name in model_pipes:
        return model_pipes[model_name]
    
    print(f"🔄 Loading {model_name}...")
    
    try:
        # Clear memory before loading
        clear_memory()
        
        if model_name == "T5-Small":
            # T5 models are very reliable
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[model_name])
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAMES[model_name],
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            pipe = pipeline(
                "text2text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                device=-1
            )
        else:
            # GPT-2 models
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[model_name])
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAMES[model_name],
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            pipe = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                device=-1
            )
        
        # Store both pipe and tokenizer
        model_pipes[model_name] = pipe
        tokenizers[model_name] = tokenizer
        
        print(f"✅ Successfully loaded {model_name}")
        return pipe
        
    except Exception as e:
        print(f"❌ Error loading {model_name}: {str(e)}")
        clear_memory()
        return None

def generate_article(prompt, model_choice):
    """Generate article with robust error handling"""
    pipe = load_model(model_choice)
    
    if pipe is None:
        return f"❌ Error: Could not load {model_choice} model. Please try a different model."
    
    try:
        if model_choice == "T5-Small":
            # T5 works best with clear instructions
            formatted_prompt = f"Write a short article about: {prompt}"
            result = pipe(
                formatted_prompt, 
                max_new_tokens=80,
                do_sample=True, 
                temperature=0.8,
                num_return_sequences=1,
                early_stopping=True
            )
            generated_text = result[0]['generated_text']
            
            # Clean up the output
            if generated_text.startswith("Write a short article about:"):
                generated_text = generated_text.replace("Write a short article about:", "").strip()
            
            # Ensure we have meaningful content
            if len(generated_text.strip()) < 10:
                return "Generated content was too short. Please try again with a different topic."
                
            return generated_text
            
        else:
            # GPT-2 models work well with simple prompts
            formatted_prompt = f"Article about {prompt}:\n\n"
            
            # Get the tokenizer for this model
            tokenizer = tokenizers.get(model_choice)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[model_choice])
                tokenizers[model_choice] = tokenizer
            
            result = pipe(
                formatted_prompt, 
                max_new_tokens=80,
                do_sample=True, 
                temperature=0.8, 
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
            generated_text = result[0]['generated_text']
            
            # Clean up the output
            if generated_text.startswith(formatted_prompt):
                generated_text = generated_text[len(formatted_prompt):].strip()
            
            # Ensure we have meaningful content
            if len(generated_text.strip()) < 10:
                return "Generated content was too short. Please try again with a different topic."
                
            return generated_text
            
    except Exception as e:
        print(f"❌ Error generating text with {model_choice}: {str(e)}")
        clear_memory()
        return f"❌ Error generating text with {model_choice}: {str(e)}"

def article_generator(prompt, model_choice):
    """Main function with comprehensive fallback mechanisms"""
    if not prompt.strip():
        return "⚠️ Please enter a topic or prompt."
    
    # Try the selected model first
    result = generate_article(prompt, model_choice)
    
    # If the selected model fails, try fallback models
    if result.startswith("❌ Error:"):
        print(f"🔄 Primary model failed, trying fallback...")
        
        # Try all available models as fallbacks
        for fallback_model in ["DistilGPT2", "GPT-2", "T5-Small"]:
            if fallback_model != model_choice:
                print(f"🔄 Trying {fallback_model} as fallback...")
                result = generate_article(prompt, fallback_model)
                if not result.startswith("❌ Error:"):
                    return f"✅ Generated with {fallback_model} (fallback):\n\n{result}"
        
        # If all models fail, return helpful message
        return "❌ All models failed. Please check your internet connection and try again."
    
    return result

# Create the Gradio interface
with gr.Blocks(title="Article Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📝 Article Generator
    
    Choose a model and enter a topic to generate an article. All models are open-source and run locally.
    
    **Available Models:**
    - **GPT-2**: Standard GPT-2 model for reliable text generation
    - **DistilGPT2**: Smaller, faster version of GPT-2 (recommended)
    - **T5-Small**: Efficient text-to-text model for structured content
    
    **Note:** Models will be downloaded automatically on first use.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Article Topic", 
                lines=3, 
                placeholder="e.g., The future of artificial intelligence in healthcare, Climate change solutions, Benefits of renewable energy"
            )
        with gr.Column(scale=1):
            model_choice = gr.Radio(
                list(MODEL_NAMES.keys()), 
                label="Choose Model", 
                value="DistilGPT2",
                info="DistilGPT2 is recommended for best performance"
            )
    
    with gr.Row():
        generate_btn = gr.Button("🚀 Generate Article", variant="primary", size="lg")
        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
    
    with gr.Row():
        output = gr.Textbox(
            label="Generated Article", 
            lines=15,
            placeholder="Your generated article will appear here..."
        )
    
    # Event handlers
    generate_btn.click(
        article_generator, 
        inputs=[prompt, model_choice], 
        outputs=output
    )
    
    clear_btn.click(
        lambda: ("", ""), 
        outputs=[prompt, output]
    )
    
    # Add some helpful examples
    gr.Examples(
        examples=[
            ["The benefits of renewable energy sources", "DistilGPT2"],
            ["How artificial intelligence is transforming healthcare", "T5-Small"],
            ["The impact of climate change on global ecosystems", "GPT-2"],
            ["The future of remote work and digital transformation", "DistilGPT2"],
            ["Sustainable living practices for modern families", "T5-Small"]
        ],
        inputs=[prompt, model_choice]
    )

if __name__ == "__main__":
    print("🚀 Starting Article Generator...")
    print("📋 Available models:", list(MODEL_NAMES.keys()))
    print("💡 Tip: DistilGPT2 is recommended for best performance")
    demo.launch(share=False, show_error=True)
