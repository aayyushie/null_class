# Article Generator LLM Setup and Testing Notebook

## 1. Environment Setup and Dependencies

```python
# Install required packages
!pip install -q transformers torch accelerate fastapi uvicorn streamlit
!pip install -q sentencepiece protobuf pandas matplotlib seaborn

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import requests
import json
from typing import Dict, List
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Environment setup complete!")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 2. Model Configuration and Loading

```python
# Model configurations for three small, efficient models
MODEL_CONFIGS = {
    'distilgpt2': {
        'model_name': 'distilgpt2',
        'description': 'Distilled version of GPT-2, 82M parameters',
        'strengths': ['Fast inference', 'Low memory', 'Good for short text'],
        'weaknesses': ['Limited context', 'Less creative'],
        'port': 8001
    },
    'gpt2': {
        'model_name': 'gpt2',
        'description': 'Original GPT-2 small, 124M parameters',
        'strengths': ['Balanced performance', 'Good coherence', 'Reasonable size'],
        'weaknesses': ['Moderate speed', 'Can be repetitive'],
        'port': 8000
    },
    'microsoft/DialoGPT-small': {
        'model_name': 'microsoft/DialoGPT-small',
        'description': 'Small conversational model, 117M parameters',
        'strengths': ['Conversational', 'Good for dialogue', 'Contextual'],
        'weaknesses': ['Designed for chat', 'May not follow article format'],
        'port': 8002
    }
}

def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer with appropriate configurations"""
    print(f"Loading {model_name}...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate settings for small models
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        print(f"✅ Successfully loaded {model_name}")
        print(f"   Parameters: {model.num_parameters():,}")
        print(f"   Memory usage: {model.get_memory_footprint() / 1024**2:.1f} MB")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None, None

# Load all models
models = {}
tokenizers = {}

for model_id, config in MODEL_CONFIGS.items():
    model, tokenizer = load_model_and_tokenizer(config['model_name'])
    if model is not None:
        models[model_id] = model
        tokenizers[model_id] = tokenizer
        print(f"Model {model_id} loaded successfully\n")
    else:
        print(f"Failed to load {model_id}\n")
```

## 3. Text Generation Functions

```python
def generate_text(model, tokenizer, prompt: str, max_length: int = 200, temperature: float = 0.7):
    """Generate text using the model"""
    try:
        # Encode the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        
        # Move to same device as model
        if torch.cuda.is_available():
            inputs = inputs.to(model.device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                no_repeat_ngram_size=2
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
        
    except Exception as e:
        print(f"Error generating text: {e}")
        return ""

def create_article_prompt(topic: str, article_type: str) -> str:
    """Create a structured prompt for article generation"""
    prompts = {
        'news': f"Breaking News: {topic}\n\nIn a recent development, ",
        'blog': f"Blog Post: Everything You Need to Know About {topic}\n\nIntroduction: ",
        'technical': f"Technical Guide: Understanding {topic}\n\nOverview: ",
        'opinion': f"Opinion: My Thoughts on {topic}\n\nI believe that ",
        'feature': f"Feature Story: The Impact of {topic}\n\nIt was a typical day when "
    }
    
    return prompts.get(article_type, prompts['blog'])

# Test text generation with each model
test_topic = "artificial intelligence"
test_prompt = create_article_prompt(test_topic, 'blog')

print(f"Test prompt: {test_prompt}")
print("="*50)

for model_id in models.keys():
    print(f"\n🔸 Testing {model_id}:")
    start_time = time.time()
    
    generated = generate_text(models[model_id], tokenizers[model_id], test_prompt)
    generation_time = time.time() - start_time
    
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Generated text: {generated[:200]}...")
    print("-" * 30)
```

## 4. Performance Evaluation Framework

```python
class ModelEvaluator:
    """Comprehensive model evaluation framework"""
    
    def __init__(self):
        self.metrics = {
            'response_times': {},
            'word_counts': {},
            'readability_scores': {},
            'coherence_scores': {},
            'creativity_scores': {}
        }
    
    def evaluate_readability(self, text: str) -> float:
        """Simple readability assessment"""
        if not text:
            return 0.0
        
        sentences = text.split('.')
        words = text.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Ideal range: 15-20 words per sentence
        if 15 <= avg_sentence_length <= 20:
            length_score = 1.0
        elif 10 <= avg_sentence_length <= 25:
            length_score = 0.8
        else:
            length_score = 0.5
        
        # Check for proper punctuation
        punctuation_score = 1.0 if text.count('.') > 0 else 0.5
        
        # Check for capitalization
        capitalization_score = 1.0 if text[0].isupper() else 0.5
        
        return (length_score + punctuation_score + capitalization_score) / 3
    
    def evaluate_coherence(self, text: str) -> float:
        """Evaluate text coherence"""
        if not text:
            return 0.0
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 0.5
        
        # Check for transition words
        transition_words = ['however', 'therefore', 'furthermore', 'additionally', 
                          'meanwhile', 'consequently', 'moreover', 'similarly']
        
        transition_count = sum(1 for sentence in sentences 
                             for word in transition_words 
                             if word in sentence.lower())
        
        transition_score = min(transition_count / len(sentences), 1.0)
        
        # Check for consistent topic (simple keyword analysis)
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Topic consistency based on word repetition
        if word_freq:
            max_freq = max(word_freq.values())
            consistency_score = min(max_freq / len(words) * 10, 1.0)
        else:
            consistency_score = 0.5
        
        return (transition_score + consistency_score) / 2
    
    def evaluate_creativity(self, text: str) -> float:
        """Evaluate text creativity"""
        if not text:
            return 0.0
        
        words = text.lower().split()
        if len(words) < 10:
            return 0.5
        
        # Vocabulary diversity (lexical diversity)
        unique_words = set(words)
        diversity_score = len(unique_words) / len(words)
        
        # Check for descriptive language
        descriptive_words = ['amazing', 'incredible', 'fascinating', 'remarkable', 
                           'outstanding', 'exceptional', 'brilliant', 'innovative']
        
        descriptive_count = sum(1 for word in words if word in descriptive_words)
        descriptive_score = min(descriptive_count / len(words) * 20, 1.0)
        
        return (diversity_score + descriptive_score) / 2
    
    def evaluate_model(self, model_id: str, model, tokenizer, prompts: List[str]) -> Dict:
        """Comprehensive model evaluation"""
        results = {
            'model_id': model_id,
            'response_times': [],
            'word_counts': [],
            'readability_scores': [],
            'coherence_scores': [],
            'creativity_scores': [],
            'generated_texts': []
        }
        
        print(f"Evaluating {model_id}...")
        
        for i, prompt in enumerate(prompts):
            print(f"  Processing prompt {i+1}/{len(prompts)}")
            
            # Generate text and measure time
            start_time = time.time()
            generated_text = generate_text(model, tokenizer, prompt, max_length=150)
            response_time = time.time() - start_time
            
            if generated_text:
                # Calculate metrics
                word_count = len(generated_text.split())
                readability = self.evaluate_readability(generated_text)
                coherence = self.evaluate_coherence(generated_text)
                creativity = self.evaluate_creativity(generated_text)
                
                # Store results
                results['response_times'].append(response_time)
                results['word_counts'].append(word_count)
                results['readability_scores'].append(readability)
                results['coherence_scores'].append(coherence)
                results['creativity_scores'].append(creativity)
                results['generated_texts'].append(generated_text)
        
        return results

# Initialize evaluator
evaluator = ModelEvaluator()

# Create test prompts for different article types
test_prompts = [
    create_article_prompt("renewable energy", "technical"),
    create_article_prompt("smartphone addiction", "opinion"),
    create_article_prompt("space exploration", "news"),
    create_article_prompt("healthy eating", "blog"),
    create_article_prompt("remote work", "feature")
]

print("Test prompts created:")
for i, prompt in enumerate(test_prompts):
    print(f"{i+1}. {prompt[:50]}...")
```

## 5. Run Comprehensive Evaluation

```python
# Run evaluation for all models
evaluation_results = {}

for model_id in models.keys():
    print(f"\n{'='*50}")
    print(f"EVALUATING {model_id.upper()}")
    print(f"{'='*50}")
    
    results = evaluator.evaluate_model(
        model_id, 
        models[model_id], 
        tokenizers[model_id], 
        test_prompts
    )
    
    evaluation_results[model_id] = results
    
    # Print summary
    if results['response_times']:
        print(f"\nSUMMARY FOR {model_id}:")
        print(f"  Average response time: {np.mean(results['response_times']):.2f}s")
        print(f"  Average word count: {np.mean(results['word_counts']):.1f}")
        print(f"  Average readability: {np.mean(results['readability_scores']):.3f}")
        print(f"  Average coherence: {np.mean(results['coherence_scores']):.3f}")
        print(f"  Average creativity: {np.mean(results['creativity_scores']):.3f}")
```

## 6. Performance Analysis and Visualization

```python
# Create comprehensive performance comparison
def create_performance_dataframe(evaluation_results):
    """Create a DataFrame for performance comparison"""
    data = []
    
    for model_id, results in evaluation_results.items():
        if results['response_times']:
            data.append({
                'Model': model_id,
                'Avg Response Time (s)': np.mean(results['response_times']),
                'Avg Word Count': np.mean(results['word_counts']),
                'Readability Score': np.mean(results['readability_scores']),
                'Coherence Score': np.mean(results['coherence_scores']),
                'Creativity Score': np.mean(results['creativity_scores']),
                'Total Tests': len(results['response_times'])
            })
    
    return pd.DataFrame(data)

# Create performance DataFrame
performance_df = create_performance_dataframe(evaluation_results)
print("Performance Comparison:")
print(performance_df.round(3))

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('LLM Performance Comparison for Article Generation', fontsize=16, fontweight='bold')

# 1. Response Time Comparison
axes[0, 0].bar(performance_df['Model'], performance_df['Avg Response Time (s)'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0, 0].set_title('Average Response Time')
axes[0, 0].set_ylabel('Seconds')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Word Count Comparison
axes[0, 1].bar(performance_df['Model'], performance_df['Avg Word Count'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0, 1].set_title('Average Word Count')
axes[0, 1].set_ylabel('Words')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Quality Metrics Comparison
quality_metrics = ['Readability Score', 'Coherence Score', 'Creativity Score']
x = np.arange(len(performance_df))
width = 0.25

for i, metric in enumerate(quality_metrics):
    axes[0, 2].bar(x + i*width, performance_df[metric], width, 
                   label=metric, alpha=0.8)

axes[0, 2].set_title('Quality Metrics Comparison')
axes[0, 2].set_ylabel('Score (0-1)')
axes[0, 2].set_xticks(x + width)
axes[0, 2].set_xticklabels(performance_df['Model'])
axes[0, 2].legend()
axes[0, 2].tick_params(axis='x', rotation=45)

# 4. Overall Performance Radar Chart
categories = ['Readability', 'Coherence', 'Creativity', 'Speed', 'Verbosity']

for idx, row in performance_df.iterrows():
    # Normalize metrics for radar chart
    values = [
        row['Readability Score'],
        row['Coherence Score'],
        row['Creativity Score'],
        1 / (row['Avg Response Time (s)'] + 0.1),  # Inverse for speed (higher = better)
        row['Avg Word Count'] / performance_df['Avg Word Count'].max()  # Normalized verbosity
    ]
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]
    
    axes[1, 0].plot(angles, values, 'o-', linewidth=2, label=row['Model'])
    axes[1, 0].fill(angles, values, alpha=0.25)

axes[1, 0].set_xticks(angles[:-1])
axes[1, 0].set_xticklabels(categories)
axes[1, 0].set_ylim(0, 1)
axes[1, 0].set_title('Overall Performance Radar')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 5. Detailed Metric Distribution
metrics_to_plot = ['readability_scores', 'coherence_scores', 'creativity_scores']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, (model_id, results) in enumerate(evaluation_results.items()):
    if results['response_times']:
        for j, metric in enumerate(metrics_to_plot):
            axes[1, 1].scatter([i] * len(results[metric]), results[metric], 
                              alpha=0.6, s=50, color=colors[j], 
                              label=f'{metric.replace("_", " ").title()}' if i == 0 else "")

axes[1, 1].set_title('Metric Distribution by Model')
axes[1, 1].set_xlabel('Model')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_xticks(range(len(evaluation_results)))
axes[1, 1].set_xticklabels(evaluation_results.keys())
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Model Recommendation Matrix
# Calculate overall scores
performance_df['Speed Score'] = 1 / (performance_df['Avg Response Time (s)'] + 0.1)
performance_df['Speed Score'] = performance_df['Speed Score'] / performance_df['Speed Score'].max()

performance_df['Overall Score'] = (
    performance_df['Readability Score'] * 0.25 +
    performance_df['Coherence Score'] * 0.25 +
    performance_df['Creativity Score'] * 0.20 +
    performance_df['Speed Score'] * 0.20 +
    (performance_df['Avg Word Count'] / performance_df['Avg Word Count'].max()) * 0.10
)

# Create recommendation heatmap
recommendation_data = performance_df[['Model', 'Readability Score', 'Coherence Score', 
                                    'Creativity Score', 'Speed Score', 'Overall Score']].set_index('Model')

sns.heatmap(recommendation_data.T, annot=True, cmap='RdYlGn', 
            ax=axes[1, 2], fmt='.3f', cbar_kws={'label': 'Score'})
axes[1, 2].set_title('Model Recommendation Matrix')
axes[1, 2].set_ylabel('Metrics')

plt.tight_layout()
plt.show()

# Print detailed recommendation
best_model = performance_df.loc[performance_df['Overall Score'].idxmax()]
print(f"\n{'='*60}")
print(f"MODEL RECOMMENDATION ANALYSIS")
print(f"{'='*60}")
print(f"🏆 BEST OVERALL MODEL: {best_model['Model']}")
print(f"📊 Overall Score: {best_model['Overall Score']:.3f}")
print(f"\n📈 PERFORMANCE BREAKDOWN:")
print(f"   • Readability: {best_model['Readability Score']:.3f}")
print(f"   • Coherence: {best_model['Coherence Score']:.3f}")
print(f"   • Creativity: {best_model['Creativity Score']:.3f}")
print(f"   • Speed: {best_model['Speed Score']:.3f}")
print(f"   • Response Time: {best_model['Avg Response Time (s)']:.2f}s")
print(f"   • Word Count: {best_model['Avg Word Count']:.0f}")
```

## 7. Sample Generated Articles

```python
# Display sample articles from each model
print(f"\n{'='*80}")
print("SAMPLE GENERATED ARTICLES")
print(f"{'='*80}")

sample_prompt = create_article_prompt("climate change", "blog")
print(f"Prompt: {sample_prompt}")
print(f"{'='*80}")

for model_id in models.keys():
    print(f"\n🔸 {model_id.upper()}:")
    print("-" * 50)
    
    sample_text = generate_text(models[model_id], tokenizers[model_id], sample_prompt, max_length=200)
    print(sample_text)
    print()
```

## 8. Model Serving Setup (FastAPI)

```python
# Create FastAPI model server code
fastapi_server_code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import os

app = FastAPI(title="LLM Article Generator API", version="1.0.0")

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
PORT = int(os.getenv("PORT", 8000))

# Global model and tokenizer
model = None
tokenizer = None

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7

class GenerationResponse(BaseModel):
    text: str
    model_name: str

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    
    print(f"Loading model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    print(f"Model {MODEL_NAME} loaded successfully!")

@app.post("/v1/completions", response_model=GenerationResponse)
async def generate_completion(request: GenerationRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        inputs = tokenizer.encode(request.prompt, return_tensors='pt')
        
        if torch.cuda.is_available():
            inputs = inputs.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove original prompt
        if generated_text.startswith(request.prompt):
            generated_text = generated_text[len(request.prompt):].strip()
        
        return GenerationResponse(text=generated_text, model_name=MODEL_NAME)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_NAME}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
'''

# Save the FastAPI server code
with open('model_server.py', 'w') as f:
    f.write(fastapi_server_code)

print("FastAPI server code saved to 'model_server.py'")
```

## 9. Docker Setup for Model Serving

```python
# Create Dockerfile for model serving
dockerfile_content = '''
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model_server.py .

# Expose port
EXPOSE 8000

# Environment variables
ENV MODEL_NAME=distilgpt2
ENV PORT=8000

# Run the application
CMD ["python", "model_server.py"]
'''

# Save Dockerfile
with open('Dockerfile', 'w') as f:
    f.write(dockerfile_content)

# Create docker-compose for all three models
docker_compose_content = '''
version: '3.8'

services:
  distilgpt2:
    build: .
    ports:
      - "8001:8000"
    environment:
      - MODEL_NAME=distilgpt2
      - PORT=8000
    volumes:
      - ./cache:/root/.cache
    
  gpt2:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=gpt2
      - PORT=8000
    volumes:
      - ./cache:/root/.cache
    
  dialogpt:
    build: .
    ports:
      - "8002:8000"
    environment:
      - MODEL_NAME=microsoft/DialoGPT-small
      - PORT=8000
    volumes:
      - ./cache:/root/.cache
'''

# Save docker-compose
with open('docker-compose.yml', 'w') as f:
    f.write(docker_compose_content)

print("Docker configuration files created!")
print("- Dockerfile")
print("- docker-compose.yml")
```

## 10. Final Analysis and Conclusions

```python
# Generate comprehensive analysis report
analysis_report = f"""
# LLM ARTICLE GENERATOR ANALYSIS REPORT

## Executive Summary
This analysis evaluates three small, efficient open-source language models for article generation:
- **{list(MODEL_CONFIGS.keys())[0]}**: {MODEL_CONFIGS[list(MODEL_CONFIGS.keys())[0]]['description']}
- **{list(MODEL_CONFIGS.keys())[1]}**: {MODEL_CONFIGS[list(MODEL_CONFIGS.keys())[1]]['description']}
- **{list(MODEL_CONFIGS.keys())[2]}**: {MODEL_CONFIGS[list(MODEL_CONFIGS.keys())[2]]['description']}

## Key Findings

### Best Overall Model: {best_model['Model']}
- **Overall Score**: {best_model['Overall Score']:.3f}/1.0
- **Key Strengths**: High coherence and readability
- **Response Time**: {best_model['Avg Response Time (s)']:.2f} seconds
- **Word Count**: {best_model['Avg Word Count']:.0f} words average

### Performance Metrics Summary
"""

for idx, row in performance_df.iterrows():
    analysis_report += f"""
#### {row['Model']}
- **Readability**: {row['Readability Score']:.3f}
- **Coherence**: {row['Coherence Score']:.3f}
- **Creativity**: {row['Creativity Score']:.3f}
- **Speed**: {row['Avg Response Time (s)']:.2f}s
- **Verbosity**: {row['Avg Word Count']:.0f} words
"""

analysis_report += """

## Recommendations

### For Production Use
1. **Primary Model**: Use the highest-scoring model for general article generation
2. **Fallback Strategy**: Implement model switching based on response time requirements
3. **Optimization**: Consider model quantization for faster inference

### For Different Use Cases
- **Speed-Critical**: Choose the fastest model (lowest response time)
- **Quality-Critical**: Choose the highest coherence/readability model
- **Creative Writing**: Choose the highest creativity score model

### Technical Considerations
- All models are small enough to run on consumer hardware
- Memory usage is minimal (< 1GB per model)
- Response times are suitable for real-time applications
- Consider ensemble methods for best results

## Conclusion
The evaluation demonstrates that small language models can effectively generate article content
with reasonable quality and fast response times. The recommended model provides the best
balance of quality, speed, and resource efficiency for article generation tasks.
"""

print(analysis_report)

# Save analysis report
with open('analysis_report.md', 'w') as f:
    f.write(analysis_report)

print("\n✅ Analysis complete! Report saved to 'analysis_report.md'")
print("📊 All evaluation data and visualizations have been generated.")
print("🚀 Ready to deploy the article generator chatbot!")
```