import streamlit as st
import requests
import json
import time
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    """Base class for LLM clients"""
    
    def __init__(self, model_name: str, api_url: str):
        self.model_name = model_name
        self.api_url = api_url
        self.performance_metrics = {
            'response_times': [],
            'word_counts': [],
            'quality_scores': [],
            'coherence_scores': [],
            'creativity_scores': []
        }
    
    def generate_article(self, prompt: str, max_tokens: int = 500) -> Dict:
        """Generate article using the LLM"""
        start_time = time.time()
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False
            }
            
            response = requests.post(
                f"{self.api_url}/v1/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['text'].strip()
                
                # Calculate metrics
                word_count = len(generated_text.split())
                quality_score = self._assess_quality(generated_text)
                coherence_score = self._assess_coherence(generated_text)
                creativity_score = self._assess_creativity(generated_text)
                
                # Store metrics
                self.performance_metrics['response_times'].append(response_time)
                self.performance_metrics['word_counts'].append(word_count)
                self.performance_metrics['quality_scores'].append(quality_score)
                self.performance_metrics['coherence_scores'].append(coherence_score)
                self.performance_metrics['creativity_scores'].append(creativity_score)
                
                return {
                    'success': True,
                    'text': generated_text,
                    'response_time': response_time,
                    'word_count': word_count,
                    'quality_score': quality_score,
                    'coherence_score': coherence_score,
                    'creativity_score': creativity_score
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'response_time': response_time
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def _assess_quality(self, text: str) -> float:
        """Simple quality assessment based on text characteristics"""
        if not text:
            return 0.0
        
        # Basic quality metrics
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Penalize very short or very long sentences
        if avg_sentence_length < 5 or avg_sentence_length > 30:
            sentence_score = 0.5
        else:
            sentence_score = 1.0
        
        # Check for proper capitalization
        capitalization_score = 1.0 if text[0].isupper() else 0.5
        
        # Check for variety in sentence starters
        sentence_starters = [s.strip()[:10] for s in sentences if s.strip()]
        variety_score = len(set(sentence_starters)) / len(sentence_starters) if sentence_starters else 0
        
        return (sentence_score + capitalization_score + variety_score) / 3
    
    def _assess_coherence(self, text: str) -> float:
        """Assess text coherence based on logical flow"""
        if not text:
            return 0.0
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.5
        
        # Check for transition words
        transition_words = ['however', 'therefore', 'furthermore', 'moreover', 'additionally', 
                          'consequently', 'meanwhile', 'similarly', 'in contrast', 'on the other hand']
        
        transition_count = sum(1 for sentence in sentences 
                             for word in transition_words 
                             if word in sentence.lower())
        
        transition_score = min(transition_count / len(sentences), 1.0)
        
        # Check for repetitive patterns
        word_frequency = {}
        words = text.lower().split()
        for word in words:
            if len(word) > 4:  # Only count longer words
                word_frequency[word] = word_frequency.get(word, 0) + 1
        
        repetition_penalty = sum(1 for count in word_frequency.values() if count > 3) / len(words)
        repetition_score = max(0, 1 - repetition_penalty)
        
        return (transition_score + repetition_score) / 2
    
    def _assess_creativity(self, text: str) -> float:
        """Assess creativity based on word variety and uniqueness"""
        if not text:
            return 0.0
        
        words = text.lower().split()
        if len(words) < 10:
            return 0.5
        
        # Vocabulary diversity
        unique_words = set(words)
        vocabulary_diversity = len(unique_words) / len(words)
        
        # Check for descriptive adjectives and adverbs
        descriptive_words = ['amazing', 'incredible', 'fascinating', 'remarkable', 'extraordinary',
                           'brilliantly', 'elegantly', 'creatively', 'innovatively', 'uniquely']
        
        descriptive_count = sum(1 for word in words if word in descriptive_words)
        descriptive_score = min(descriptive_count / len(words) * 10, 1.0)
        
        return (vocabulary_diversity + descriptive_score) / 2
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        metrics = self.performance_metrics
        
        if not metrics['response_times']:
            return {'error': 'No performance data available'}
        
        return {
            'avg_response_time': sum(metrics['response_times']) / len(metrics['response_times']),
            'avg_word_count': sum(metrics['word_counts']) / len(metrics['word_counts']),
            'avg_quality_score': sum(metrics['quality_scores']) / len(metrics['quality_scores']),
            'avg_coherence_score': sum(metrics['coherence_scores']) / len(metrics['coherence_scores']),
            'avg_creativity_score': sum(metrics['creativity_scores']) / len(metrics['creativity_scores']),
            'total_requests': len(metrics['response_times'])
        }

class ArticleGeneratorChatbot:
    """Main chatbot class managing multiple LLMs"""
    
    def __init__(self):
        self.llm_clients = {
            'GPT-J-6B': LLMClient('gpt-j-6b', 'http://localhost:8000'),
            'DistilGPT-2': LLMClient('distilgpt2', 'http://localhost:8001'),
            'TinyLlama': LLMClient('tinyllama', 'http://localhost:8002')
        }
        
        self.article_templates = {
            'News Article': "Write a professional news article about {topic}. Include a compelling headline, lead paragraph, and supporting details.",
            'Blog Post': "Create an engaging blog post about {topic}. Make it conversational and informative with a clear structure.",
            'Technical Article': "Write a technical article explaining {topic}. Use clear explanations and examples.",
            'Opinion Piece': "Write an opinion piece about {topic}. Present arguments and supporting evidence.",
            'Feature Story': "Create a feature story about {topic}. Include narrative elements and human interest."
        }
    
    def generate_article_with_all_models(self, topic: str, article_type: str) -> Dict:
        """Generate article using all available LLMs"""
        template = self.article_templates.get(article_type, self.article_templates['Blog Post'])
        prompt = template.format(topic=topic)
        
        results = {}
        for model_name, client in self.llm_clients.items():
            st.write(f"Generating with {model_name}...")
            result = client.generate_article(prompt)
            results[model_name] = result
            
            if result['success']:
                st.success(f"✅ {model_name} completed in {result['response_time']:.2f}s")
            else:
                st.error(f"❌ {model_name} failed: {result['error']}")
        
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """Compare performance metrics across all models"""
        comparison_data = []
        
        for model_name, client in self.llm_clients.items():
            summary = client.get_performance_summary()
            if 'error' not in summary:
                comparison_data.append({
                    'Model': model_name,
                    'Avg Response Time (s)': summary['avg_response_time'],
                    'Avg Word Count': summary['avg_word_count'],
                    'Quality Score': summary['avg_quality_score'],
                    'Coherence Score': summary['avg_coherence_score'],
                    'Creativity Score': summary['avg_creativity_score'],
                    'Total Requests': summary['total_requests']
                })
        
        return pd.DataFrame(comparison_data)
    
    def get_model_recommendation(self) -> str:
        """Get recommendation for best model based on performance"""
        df = self.compare_models()
        
        if df.empty:
            return "No performance data available for recommendation."
        
        # Calculate overall score (weighted average)
        df['Overall Score'] = (
            df['Quality Score'] * 0.3 +
            df['Coherence Score'] * 0.3 +
            df['Creativity Score'] * 0.2 +
            (1 / df['Avg Response Time (s)']) * 0.1 +  # Faster is better
            (df['Avg Word Count'] / df['Avg Word Count'].max()) * 0.1
        )
        
        best_model = df.loc[df['Overall Score'].idxmax()]
        
        return f"""
        **Recommended Model: {best_model['Model']}**
        
        **Performance Summary:**
        - Overall Score: {best_model['Overall Score']:.3f}
        - Quality Score: {best_model['Quality Score']:.3f}
        - Coherence Score: {best_model['Coherence Score']:.3f}
        - Creativity Score: {best_model['Creativity Score']:.3f}
        - Avg Response Time: {best_model['Avg Response Time (s)']:.2f}s
        - Avg Word Count: {best_model['Avg Word Count']:.0f}
        
        **Why this model is recommended:**
        This model provides the best balance of quality, coherence, and creativity while maintaining reasonable response times.
        """

def main():
    st.set_page_config(
        page_title="Article Generator Chatbot",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Multi-LLM Article Generator Chatbot")
    st.markdown("Generate articles using three different open-source LLMs and compare their performance!")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ArticleGeneratorChatbot()
    
    # Sidebar for model information
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        **Available Models:**
        - **GPT-J-6B**: 6B parameter model, good for creative writing
        - **DistilGPT-2**: Lightweight distilled model, fast inference
        - **TinyLlama**: Ultra-small model, very fast but simpler output
        
        **Evaluation Metrics:**
        - **Quality**: Grammar, structure, readability
        - **Coherence**: Logical flow and consistency
        - **Creativity**: Word variety and uniqueness
        """)
        
        if st.button("🔄 Reset Performance Data"):
            for client in st.session_state.chatbot.llm_clients.values():
                client.performance_metrics = {
                    'response_times': [],
                    'word_counts': [],
                    'quality_scores': [],
                    'coherence_scores': [],
                    'creativity_scores': []
                }
            st.success("Performance data reset!")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Generate Article")
        
        # Input fields
        topic = st.text_input("Enter article topic:", placeholder="e.g., Artificial Intelligence in Healthcare")
        article_type = st.selectbox(
            "Select article type:",
            list(st.session_state.chatbot.article_templates.keys())
        )
        
        if st.button("🚀 Generate Articles", type="primary"):
            if topic:
                with st.spinner("Generating articles with all models..."):
                    results = st.session_state.chatbot.generate_article_with_all_models(topic, article_type)
                    
                    st.header("📄 Generated Articles")
                    
                    for model_name, result in results.items():
                        with st.expander(f"{model_name} - {article_type}"):
                            if result['success']:
                                st.write(f"**Response Time:** {result['response_time']:.2f}s")
                                st.write(f"**Word Count:** {result['word_count']}")
                                st.write(f"**Quality Score:** {result['quality_score']:.3f}")
                                st.write(f"**Coherence Score:** {result['coherence_score']:.3f}")
                                st.write(f"**Creativity Score:** {result['creativity_score']:.3f}")
                                st.write("---")
                                st.write(result['text'])
                            else:
                                st.error(f"Generation failed: {result['error']}")
            else:
                st.warning("Please enter a topic!")
    
    with col2:
        st.header("📈 Performance Analytics")
        
        # Performance comparison
        if st.button("📊 Compare Models"):
            df = st.session_state.chatbot.compare_models()
            
            if not df.empty:
                st.subheader("Model Comparison")
                st.dataframe(df)
                
                # Visualization
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Response time comparison
                axes[0, 0].bar(df['Model'], df['Avg Response Time (s)'])
                axes[0, 0].set_title('Average Response Time')
                axes[0, 0].set_ylabel('Seconds')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Quality metrics
                metrics = ['Quality Score', 'Coherence Score', 'Creativity Score']
                x = range(len(df))
                width = 0.25
                
                for i, metric in enumerate(metrics):
                    axes[0, 1].bar([xi + i*width for xi in x], df[metric], width, label=metric)
                
                axes[0, 1].set_title('Quality Metrics Comparison')
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].set_xticks([xi + width for xi in x])
                axes[0, 1].set_xticklabels(df['Model'])
                axes[0, 1].legend()
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # Word count comparison
                axes[1, 0].bar(df['Model'], df['Avg Word Count'])
                axes[1, 0].set_title('Average Word Count')
                axes[1, 0].set_ylabel('Words')
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # Overall performance radar chart
                categories = ['Quality', 'Coherence', 'Creativity']
                
                # Create radar chart for best performing model
                if len(df) > 0:
                    best_idx = df['Quality Score'].idxmax()
                    values = [df.iloc[best_idx]['Quality Score'], 
                             df.iloc[best_idx]['Coherence Score'], 
                             df.iloc[best_idx]['Creativity Score']]
                    
                    angles = [n / float(len(categories)) * 2 * 3.14159 for n in range(len(categories))]
                    angles += angles[:1]
                    values += values[:1]
                    
                    axes[1, 1].plot(angles, values, 'o-', linewidth=2)
                    axes[1, 1].fill(angles, values, alpha=0.25)
                    axes[1, 1].set_xticks(angles[:-1])
                    axes[1, 1].set_xticklabels(categories)
                    axes[1, 1].set_ylim(0, 1)
                    axes[1, 1].set_title(f'Best Model: {df.iloc[best_idx]["Model"]}')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Model recommendation
                st.subheader("🎯 Model Recommendation")
                recommendation = st.session_state.chatbot.get_model_recommendation()
                st.markdown(recommendation)
                
            else:
                st.info("No performance data available. Generate some articles first!")

if __name__ == "__main__":
    main()