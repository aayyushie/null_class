#!/usr/bin/env python3
"""
Multi-LLM Article Generator Chatbot
Uses three different open-source LLMs to generate articles and compare performance
"""

import requests
import json
import time
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import statistics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for each LLM"""
    name: str
    api_url: str
    model_name: str
    max_tokens: int
    temperature: float
    headers: Dict[str, str]

@dataclass
class ArticleMetrics:
    """Metrics for evaluating article quality"""
    word_count: int
    sentence_count: int
    paragraph_count: int
    readability_score: float
    coherence_score: float
    generation_time: float
    relevance_score: float

class ArticleEvaluator:
    """Evaluates generated articles based on various metrics"""
    
    @staticmethod
    def calculate_readability(text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        syllables = sum([ArticleEvaluator._count_syllables(word) for word in text.split()])
        
        if sentences == 0 or words == 0:
            return 0
        
        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0, min(100, score))
    
    @staticmethod
    def _count_syllables(word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_char_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_char_was_vowel:
                    syllable_count += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    @staticmethod
    def calculate_coherence(text: str) -> float:
        """Simple coherence score based on transition words and structure"""
        transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'additionally', 'meanwhile', 'nevertheless', 'thus', 'hence',
            'finally', 'firstly', 'secondly', 'in conclusion', 'as a result'
        ]
        
        sentences = re.split(r'[.!?]+', text.lower())
        transition_count = 0
        
        for sentence in sentences:
            for transition in transition_words:
                if transition in sentence:
                    transition_count += 1
                    break
        
        if len(sentences) == 0:
            return 0
        
        return min(100, (transition_count / len(sentences)) * 100 * 3)
    
    @staticmethod
    def calculate_relevance(text: str, topic: str) -> float:
        """Calculate topic relevance score"""
        topic_words = topic.lower().split()
        text_words = text.lower().split()
        
        relevance_count = 0
        for topic_word in topic_words:
            relevance_count += text_words.count(topic_word)
        
        if len(text_words) == 0:
            return 0
        
        return min(100, (relevance_count / len(text_words)) * 100 * 50)
    
    @staticmethod
    def evaluate_article(text: str, topic: str, generation_time: float) -> ArticleMetrics:
        """Comprehensive article evaluation"""
        words = text.split()
        sentences = re.findall(r'[.!?]+', text)
        paragraphs = text.split('\n\n')
        
        return ArticleMetrics(
            word_count=len(words),
            sentence_count=len(sentences),
            paragraph_count=len([p for p in paragraphs if p.strip()]),
            readability_score=ArticleEvaluator.calculate_readability(text),
            coherence_score=ArticleEvaluator.calculate_coherence(text),
            generation_time=generation_time,
            relevance_score=ArticleEvaluator.calculate_relevance(text, topic)
        )

class LLMClient:
    """Base class for LLM clients"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def generate_article(self, prompt: str) -> Tuple[str, float]:
        """Generate article and return content with generation time"""
        raise NotImplementedError

class OllamaClient(LLMClient):
    """Client for Ollama-hosted models"""
    
    def generate_article(self, prompt: str) -> Tuple[str, float]:
        start_time = time.time()
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "options": {
                "num_predict": self.config.max_tokens,
                "temperature": self.config.temperature
            },
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.config.api_url}/api/generate",
                json=payload,
                headers=self.config.headers,
                timeout=120
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', ''), generation_time
            else:
                logger.error(f"Error from {self.config.name}: {response.status_code}")
                return f"Error generating article with {self.config.name}", generation_time
                
        except requests.exceptions.RequestException as e:
            generation_time = time.time() - start_time
            logger.error(f"Request failed for {self.config.name}: {e}")
            return f"Connection error with {self.config.name}", generation_time

class LocalLLMClient(LLMClient):
    """Client for local API-compatible models"""
    
    def generate_article(self, prompt: str) -> Tuple[str, float]:
        start_time = time.time()
        
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        try:
            response = requests.post(
                self.config.api_url,
                json=payload,
                headers=self.config.headers,
                timeout=120
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                return content, generation_time
            else:
                logger.error(f"Error from {self.config.name}: {response.status_code}")
                return f"Error generating article with {self.config.name}", generation_time
                
        except requests.exceptions.RequestException as e:
            generation_time = time.time() - start_time
            logger.error(f"Request failed for {self.config.name}: {e}")
            return f"Connection error with {self.config.name}", generation_time

class ArticleGeneratorChatbot:
    """Main chatbot class that orchestrates multiple LLMs"""
    
    def __init__(self):
        self.llm_configs = self._setup_llm_configs()
        self.llm_clients = self._create_clients()
        self.evaluator = ArticleEvaluator()
        self.performance_data = {config.name: [] for config in self.llm_configs}
    
    def _setup_llm_configs(self) -> List[LLMConfig]:
        """Configure the three LLMs"""
        return [
            # Llama 2 via Ollama
            LLMConfig(
                name="Llama-2-7B",
                api_url="http://localhost:11434",
                model_name="llama2:7b",
                max_tokens=1500,
                temperature=0.7,
                headers={"Content-Type": "application/json"}
            ),
            
            # Mistral via Ollama
            LLMConfig(
                name="Mistral-7B",
                api_url="http://localhost:11434",
                model_name="mistral:7b",
                max_tokens=1500,
                temperature=0.7,
                headers={"Content-Type": "application/json"}
            ),
            
            # CodeLlama via Ollama (repurposed for general text)
            LLMConfig(
                name="CodeLlama-7B",
                api_url="http://localhost:11434",
                model_name="codellama:7b",
                max_tokens=1500,
                temperature=0.8,
                headers={"Content-Type": "application/json"}
            )
        ]
    
    def _create_clients(self) -> Dict[str, LLMClient]:
        """Create LLM clients"""
        clients = {}
        for config in self.llm_configs:
            clients[config.name] = OllamaClient(config)
        return clients
    
    def generate_article_prompt(self, topic: str, length: str = "medium") -> str:
        """Create a comprehensive prompt for article generation"""
        length_guide = {
            "short": "300-500 words",
            "medium": "800-1200 words", 
            "long": "1500-2000 words"
        }
        
        prompt = f"""Write a comprehensive, well-structured article about "{topic}".

Requirements:
- Length: {length_guide.get(length, "800-1200 words")}
- Include an engaging introduction
- Use clear headings and subheadings
- Provide detailed explanations with examples
- Include a strong conclusion
- Write in a professional, informative tone
- Ensure the content is accurate and up-to-date

Topic: {topic}

Please generate the article now:"""
        
        return prompt
    
    def generate_articles(self, topic: str, length: str = "medium") -> Dict[str, Tuple[str, ArticleMetrics]]:
        """Generate articles using all three LLMs"""
        prompt = self.generate_article_prompt(topic, length)
        results = {}
        
        print(f"\n🚀 Generating articles about '{topic}'...")
        print("=" * 60)
        
        for i, (name, client) in enumerate(self.llm_clients.items(), 1):
            print(f"\n[{i}/3] Generating with {name}...")
            
            try:
                article, gen_time = client.generate_article(prompt)
                metrics = self.evaluator.evaluate_article(article, topic, gen_time)
                results[name] = (article, metrics)
                
                # Store performance data
                self.performance_data[name].append(metrics)
                
                print(f"✅ {name} completed in {gen_time:.2f}s")
                print(f"   📝 {metrics.word_count} words, {metrics.sentence_count} sentences")
                
            except Exception as e:
                logger.error(f"Failed to generate with {name}: {e}")
                results[name] = (f"Error: {str(e)}", None)
        
        return results
    
    def compare_performance(self, results: Dict[str, Tuple[str, ArticleMetrics]]) -> str:
        """Generate detailed performance comparison"""
        comparison = "\n" + "="*80 + "\n"
        comparison += "🔍 PERFORMANCE ANALYSIS\n"
        comparison += "="*80 + "\n"
        
        valid_results = {k: v for k, v in results.items() if v[1] is not None}
        
        if not valid_results:
            return comparison + "❌ No valid results to compare.\n"
        
        # Individual LLM Analysis
        for name, (article, metrics) in valid_results.items():
            comparison += f"\n📊 {name} Analysis:\n"
            comparison += f"   Word Count: {metrics.word_count}\n"
            comparison += f"   Sentences: {metrics.sentence_count}\n"
            comparison += f"   Paragraphs: {metrics.paragraph_count}\n"
            comparison += f"   Readability Score: {metrics.readability_score:.1f}/100\n"
            comparison += f"   Coherence Score: {metrics.coherence_score:.1f}/100\n"
            comparison += f"   Topic Relevance: {metrics.relevance_score:.1f}/100\n"
            comparison += f"   Generation Time: {metrics.generation_time:.2f}s\n"
        
        # Comparative Rankings
        comparison += "\n🏆 RANKINGS:\n"
        
        # Speed ranking
        speed_ranking = sorted(valid_results.items(), key=lambda x: x[1][1].generation_time)
        comparison += "\n⚡ Speed (fastest to slowest):\n"
        for i, (name, (_, metrics)) in enumerate(speed_ranking, 1):
            comparison += f"   {i}. {name}: {metrics.generation_time:.2f}s\n"
        
        # Content quality ranking (composite score)
        quality_ranking = sorted(valid_results.items(), 
                                key=lambda x: (x[1][1].readability_score + 
                                             x[1][1].coherence_score + 
                                             x[1][1].relevance_score) / 3, 
                                reverse=True)
        comparison += "\n📝 Content Quality (best to worst):\n"
        for i, (name, (_, metrics)) in enumerate(quality_ranking, 1):
            avg_score = (metrics.readability_score + metrics.coherence_score + 
                        metrics.relevance_score) / 3
            comparison += f"   {i}. {name}: {avg_score:.1f}/100\n"
        
        # Word count ranking
        length_ranking = sorted(valid_results.items(), key=lambda x: x[1][1].word_count, reverse=True)
        comparison += "\n📏 Article Length (longest to shortest):\n"
        for i, (name, (_, metrics)) in enumerate(length_ranking, 1):
            comparison += f"   {i}. {name}: {metrics.word_count} words\n"
        
        # Overall recommendation
        comparison += "\n🎯 RECOMMENDATION:\n"
        
        best_overall = quality_ranking[0][0]
        fastest = speed_ranking[0][0]
        
        comparison += f"\n✨ Best Overall Quality: {best_overall}\n"
        comparison += f"⚡ Fastest Generation: {fastest}\n"
        
        if best_overall == fastest:
            comparison += f"\n🏆 {best_overall} is the clear winner - best quality AND fastest!\n"
        else:
            comparison += f"\n⚖️  Trade-off: Choose {best_overall} for quality or {fastest} for speed.\n"
        
        return comparison
    
    def get_llm_information(self) -> str:
        """Return detailed information about the three LLMs"""
        info = "\n" + "="*80 + "\n"
        info += "🤖 LLM INFORMATION & SPECIFICATIONS\n"
        info += "="*80 + "\n"
        
        llm_details = {
            "Llama-2-7B": {
                "Developer": "Meta (Facebook)",
                "Parameters": "7 billion",
                "Context Length": "4,096 tokens",
                "Training Data": "Publicly available online data (2 trillion tokens)",
                "Strengths": "General-purpose, good reasoning, conversational",
                "Best For": "Balanced article writing, Q&A, general content",
                "License": "Custom license (commercial use allowed)"
            },
            "Mistral-7B": {
                "Developer": "Mistral AI",
                "Parameters": "7 billion", 
                "Context Length": "8,192 tokens",
                "Training Data": "Web data, optimized for efficiency",
                "Strengths": "Fast inference, instruction following, multilingual",
                "Best For": "Quick content generation, technical writing",
                "License": "Apache 2.0"
            },
            "CodeLlama-7B": {
                "Developer": "Meta (based on Llama 2)",
                "Parameters": "7 billion",
                "Context Length": "16,384 tokens",
                "Training Data": "Code repositories + Llama 2 base data",
                "Strengths": "Code generation, technical explanations, structured text",
                "Best For": "Technical articles, how-to guides, structured content",
                "License": "Custom license (commercial use allowed)"
            }
        }
        
        for llm_name, details in llm_details.items():
            info += f"\n🔹 {llm_name}:\n"
            for key, value in details.items():
                info += f"   {key}: {value}\n"
        
        return info
    
    def run_interactive_session(self):
        """Run interactive chatbot session"""
        print("🤖 Multi-LLM Article Generator Chatbot")
        print("=====================================")
        print("Generate articles using three different open-source LLMs!")
        print(self.get_llm_information())
        print("\nCommands:")
        print("- 'generate <topic>' - Generate articles about a topic")
        print("- 'info' - Show LLM information")
        print("- 'stats' - Show performance statistics")
        print("- 'quit' - Exit the chatbot")
        
        while True:
            try:
                user_input = input("\n💬 Enter command: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'info':
                    print(self.get_llm_information())
                elif user_input.lower() == 'stats':
                    self.show_performance_stats()
                elif user_input.lower().startswith('generate '):
                    topic = user_input[9:].strip()
                    if topic:
                        results = self.generate_articles(topic)
                        
                        # Show articles
                        print("\n📄 GENERATED ARTICLES:")
                        print("="*60)
                        for name, (article, metrics) in results.items():
                            if metrics:
                                print(f"\n🤖 {name}:")
                                print("-" * 40)
                                print(article[:500] + "..." if len(article) > 500 else article)
                        
                        # Show comparison
                        print(self.compare_performance(results))
                    else:
                        print("❌ Please provide a topic after 'generate'")
                else:
                    print("❌ Unknown command. Try 'generate <topic>', 'info', 'stats', or 'quit'")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_performance_stats(self):
        """Show accumulated performance statistics"""
        print("\n📊 ACCUMULATED PERFORMANCE STATISTICS")
        print("="*50)
        
        for name, metrics_list in self.performance_data.items():
            if metrics_list:
                avg_time = statistics.mean([m.generation_time for m in metrics_list])
                avg_words = statistics.mean([m.word_count for m in metrics_list])
                avg_quality = statistics.mean([(m.readability_score + m.coherence_score + m.relevance_score)/3 for m in metrics_list])
                
                print(f"\n🤖 {name} (from {len(metrics_list)} generations):")
                print(f"   Average Generation Time: {avg_time:.2f}s")
                print(f"   Average Word Count: {avg_words:.0f}")
                print(f"   Average Quality Score: {avg_quality:.1f}/100")
            else:
                print(f"\n🤖 {name}: No data yet")

def main():
    """Main function to run the chatbot"""
    print("🔧 Setting up Multi-LLM Article Generator...")
    print("\n⚠️  Prerequisites:")
    print("1. Install Ollama: https://ollama.ai")
    print("2. Pull required models:")
    print("   ollama pull llama2:7b")
    print("   ollama pull mistral:7b") 
    print("   ollama pull codellama:7b")
    print("3. Ensure Ollama is running on localhost:11434")
    
    try:
        chatbot = ArticleGeneratorChatbot()
        chatbot.run_interactive_session()
    except Exception as e:
        print(f"❌ Failed to start chatbot: {e}")
        print("Please ensure Ollama is installed and running with the required models.")

if __name__ == "__main__":
    main()
