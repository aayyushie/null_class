#!/usr/bin/env python3
"""
Test script to verify all models work correctly
"""

import sys
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
import gc

# Model configurations
MODEL_NAMES = {
    "GPT-2": "gpt2",
    "DistilGPT2": "distilgpt2",
    "T5-Small": "t5-small"
}

def clear_memory():
    """Clear memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def test_model(model_name):
    """Test if a model can be loaded and generate text"""
    print(f"\n🧪 Testing {model_name}...")
    
    try:
        # Load model
        print(f"  📥 Loading {model_name}...")
        
        if model_name == "T5-Small":
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[model_name])
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAMES[model_name])
            pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
        else:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[model_name])
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES[model_name])
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
        
        print(f"  ✅ {model_name} loaded successfully")
        
        # Test generation
        print(f"  🚀 Testing text generation...")
        
        if model_name == "T5-Small":
            prompt = "Write a short article about: artificial intelligence"
            result = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.8)
            generated_text = result[0]['generated_text']
            if generated_text.startswith("Write a short article about:"):
                generated_text = generated_text.replace("Write a short article about:", "").strip()
        else:
            prompt = "Article about artificial intelligence:\n\n"
            result = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
            generated_text = result[0]['generated_text']
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
        
        print(f"  ✅ {model_name} generated text successfully")
        print(f"  📝 Generated text: {generated_text[:100]}...")
        
        # Clean up
        del pipe, model, tokenizer
        clear_memory()
        
        return True
        
    except Exception as e:
        print(f"  ❌ {model_name} failed: {str(e)}")
        clear_memory()
        return False

def main():
    """Test all models"""
    print("🚀 Starting model tests...")
    print("=" * 50)
    
    results = {}
    
    for model_name in MODEL_NAMES.keys():
        results[model_name] = test_model(model_name)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print("=" * 50)
    
    all_passed = True
    for model_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{model_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All models passed! Your app.py should work correctly.")
    else:
        print("⚠️ Some models failed. Check your internet connection and try again.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
