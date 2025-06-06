#!/usr/bin/env python3
"""
Multi-LLM Article Generator Chatbot
Uses three different open-source LLMs to generate articles and compare performance
"""

import requests
import json
import time
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import statistics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        if not text or not text.strip():
            return 0.0
            
        # More robust sentence detection
        sentences = re.findall(r'[.!?]+', text)
        words = text.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        syllables = sum([ArticleEvaluator._count_syllables(word) for word in words])
        
        if syllables == 0:
            return 0.0
        
        # Flesch Reading Ease formula
        try:
            score = 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
            return max(0, min(100, score))
        except ZeroDivisionError:
            return 0.0
    
    @staticmethod
    def _count_syllables(word: str) -> int:
        """Count syllables in a word"""
        if not word:
            return 1
            
        word = re.sub(r'[^a-zA-Z]', '', word.lower())
        if not word:
            return 1
            
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
        if not text or not text.strip():
            return 0.0
            
        transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'additionally', 'meanwhile', 'nevertheless', 'thus', 'hence',
            'finally', 'firstly', 'secondly', 'in conclusion', 'as a result',
            'for example', 'in addition', 'on the other hand'
        ]
        
        sentences = re.split(r'[.!?]+', text.lower())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
            
        transition_count = 0
        
        for sentence in sentences:
            for transition in transition_words:
                if transition in sentence:
                    transition_count += 1
                    break
        
        return min(100, (transition_count / len(sentences)) * 100 * 2)
    
    @staticmethod
    def calculate_relevance(text: str, topic: str) -> float:
        """Calculate topic relevance score"""
        if not text or not topic:
            return 0.0
            
        topic_words = [word.lower() for word in topic.split() if len(word) > 2]
        text_words = [word.lower() for word in re.findall(r'\b\w+\b', text)]
        
        if not text_words or not topic_words:
            return 0.0
        
        relevance_count = 0
        for topic_word in topic_words:
            relevance_count += text_words.count(topic_word)
        
        # Also check for partial matches and related terms
        partial_matches = 0
        for topic_word in topic_words:
            for text_word in text_words:
                if topic_word in text_word or text_word in topic_word:
                    partial_matches += 1
        
        total_relevance = relevance_count + (partial_matches * 0.5)
        return min(100, (total_relevance / len(text_words)) * 100 * 20)
    
    @staticmethod
    def evaluate_article(text: str, topic: str, generation_time: float) -> ArticleMetrics:
        """Comprehensive article evaluation"""
        if not text:
            return ArticleMetrics(0, 0, 0, 0.0, 0.0, generation_time, 0.0)
            
        words = text.split()
        sentences = re.findall(r'[.!?]+', text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        return ArticleMetrics(
            word_count=len(words),
            sentence_count=len(sentences),
            paragraph_count=len(paragraphs),
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
    
    def test_connection(self) -> bool:
        """Test if the LLM endpoint is available"""
        try:
            response = requests.get(f"{self.config.api_url.rstrip('/')}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

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
                f"{self.config.api_url.rstrip('/')}/api/generate",
                json=payload,
                headers=self.config.headers,
                timeout=180  # Increased timeout
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    content = result.get('response', '')
                    if not content:
                        return f"No content generated by {self.config.name}", generation_time
                    return content, generation_time
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON response from {self.config.name}")
                    return f"Invalid response from {self.config.name}", generation_time
            else:
                logger.error(f"Error from {self.config.name}: {response.status_code} - {response.text}")
                return f"Error {response.status_code} from {self.config.name}", generation_time
                
        except requests.exceptions.Timeout:
            generation_time = time.time() - start_time
            logger.error(f"Timeout for {self.config.name}")
            return f"Timeout error with {self.config.name}", generation_time
        except requests.exceptions.RequestException as e:
            generation_time = time.time() - start_time
            logger.error(f"Request failed for {self.config.name}: {e}")
            return f"Connection error with {self.config.name}: {str(e)}", generation_time

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
    
    def check_system_status(self) -> Dict[str, bool]:
        """Check which LLMs are available"""
        status = {}
        print("\n🔍 Checking system status...")
        
        for name, client in self.llm_clients.items():
            is_available = client.test_connection()
            status[name] = is_available
            status_icon = "✅" if is_available else "❌"
            print(f"   {status_icon} {name}: {'Available' if is_available else 'Unavailable'}")
        
        return status
    
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
- Include an engaging introduction that hooks the reader
- Use clear headings and subheadings to organize content
- Provide detailed explanations with specific examples
- Include relevant facts and information
- Write in a professional, informative tone
- End with a strong conclusion that summarizes key points
- Ensure the content is accurate and informative

Topic: {topic}

Please write the complete article now:"""
        
        return prompt
    
    def generate_articles(self, topic: str, length: str = "medium") -> Dict[str, Tuple[str, Optional[ArticleMetrics]]]:
        """Generate articles using all available LLMs"""
        prompt = self.generate_article_prompt(topic, length)
        results = {}
        
        # Check system status first
        status = self.check_system_status()
        available_clients = {name: client for name, client in self.llm_clients.items() if status.get(name, False)}
        
        if not available_clients:
            print("\n❌ No LLMs are available. Please check your Ollama installation.")
            return results
        
        print(f"\n🚀 Generating articles about '{topic}' using {len(available_clients)} available LLMs...")
        print("=" * 80)
        
        for i, (name, client) in enumerate(available_clients.items(), 1):
            print(f"\n[{i}/{len(available_clients)}] Generating with {name}...")
            
            try:
                article, gen_time = client.generate_article(prompt)
                
                # Check if generation was successful
                if article.startswith("Error") or article.startswith("Connection error") or article.startswith("Timeout"):
                    print(f"❌ {name} failed: {article}")
                    results[name] = (article, None)
                    continue
                
                metrics = self.evaluator.evaluate_article(article, topic, gen_time)
                results[name] = (article, metrics)
                
                # Store performance data
                self.performance_data[name].append(metrics)
                
                print(f"✅ {name} completed in {gen_time:.2f}s")
                print(f"   📝 {metrics.word_count} words, {metrics.sentence_count} sentences")
                print(f"   📊 Quality scores - Readability: {metrics.readability_score:.1f}, Coherence: {metrics.coherence_score:.1f}, Relevance: {metrics.relevance_score:.1f}")
                
            except Exception as e:
                logger.error(f"Failed to generate with {name}: {e}")
                results[name] = (f"Unexpected error: {str(e)}", None)
                print(f"❌ {name} encountered an error: {str(e)}")
        
        return results
    
    def compare_performance(self, results: Dict[str, Tuple[str, Optional[ArticleMetrics]]]) -> str:
        """Generate detailed performance comparison"""
        comparison = "\n" + "="*80 + "\n"
        comparison += "🔍 PERFORMANCE ANALYSIS\n"
        comparison += "="*80 + "\n"
        
        valid_results = {k: v for k, v in results.items() if v[1] is not None}
        
        if not valid_results:
            comparison += "❌ No valid results to compare.\n"
            comparison += "\nFailed generations:\n"
            for name, (error_msg, _) in results.items():
                comparison += f"   • {name}: {error_msg}\n"
            return comparison
        
        # Individual LLM Analysis
        comparison += f"\n📊 Individual Analysis ({len(valid_results)} successful generations):\n"
        for name, (article, metrics) in valid_results.items():
            comparison += f"\n🤖 {name}:\n"
            comparison += f"   Word Count: {metrics.word_count}\n"
            comparison += f"   Sentences: {metrics.sentence_count}\n"
            comparison += f"   Paragraphs: {metrics.paragraph_count}\n"
            comparison += f"   Readability Score: {metrics.readability_score:.1f}/100\n"
            comparison += f"   Coherence Score: {metrics.coherence_score:.1f}/100\n"
            comparison += f"   Topic Relevance: {metrics.relevance_score:.1f}/100\n"
            comparison += f"   Generation Time: {metrics.generation_time:.2f}s\n"
        
        if len(valid_results) > 1:
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
        
        # Initial system check
        status = self.check_system_status()
        available_count = sum(status.values())
        
        if available_count == 0:
            print("\n❌ No LLMs are currently available!")
            print("Please ensure Ollama is running and models are installed.")
            return
        
        print(f"\n✅ {available_count}/{len(status)} LLMs are available")
        print(self.get_llm_information())
        print("\nCommands:")
        print("- 'generate <topic>' - Generate articles about a topic")
        print("- 'generate <topic> short/medium/long' - Specify article length")
        print("- 'info' - Show LLM information")
        print("- 'stats' - Show performance statistics")
        print("- 'status' - Check system status")
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
                elif user_input.lower() == 'status':
                    self.check_system_status()
                elif user_input.lower().startswith('generate '):
                    parts = user_input[9:].strip().split()
                    if not parts:
                        print("❌ Please provide a topic after 'generate'")
                        continue
                    
                    # Check if length is specified
                    length = "medium"
                    if len(parts) > 1 and parts[-1].lower() in ['short', 'medium', 'long']:
                        length = parts[-1].lower()
                        topic = ' '.join(parts[:-1])
                    else:
                        topic = ' '.join(parts)
                    
                    if topic:
                        results = self.generate_articles(topic, length)
                        
                        if not results:
                            continue
                        
                        # Show articles
                        print(f"\n📄 GENERATED ARTICLES (Topic: '{topic}', Length: {length}):")
                        print("="*80)
                        for name, (article, metrics) in results.items():
                            if metrics:
                                print(f"\n🤖 {name}:")
                                print("-" * 60)
                                # Show first 800 characters with proper truncation
                                display_text = article
                                if len(display_text) > 800:
                                    display_text = display_text[:800] + "\n\n[... article truncated for display ...]"
                                print(display_text)
                            else:
                                print(f"\n❌ {name}: {article}")
                        
                        # Show comparison
                        print(self.compare_performance(results))
                    else:
                        print("❌ Please provide a topic after 'generate'")
                else:
                    print("❌ Unknown command. Available commands:")
                    print("   generate <topic> [short/medium/long], info, stats, status, quit")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error in interactive session: {e}")
                print(f"❌ Unexpected error: {e}")
    
    def show_performance_stats(self):
        """Show accumulated performance statistics"""
        print("\n📊 ACCUMULATED PERFORMANCE STATISTICS")
        print("="*60)
        
        has_data = False
        for name, metrics_list in self.performance_data.items():
            if metrics_list:
                has_data = True
                avg_time = statistics.mean([m.generation_time for m in metrics_list])
                avg_words = statistics.mean([m.word_count for m in metrics_list])
                avg_readability = statistics.mean([m.readability_score for m in metrics_list])
                avg_coherence = statistics.mean([m.coherence_score for m in metrics_list])
                avg_relevance = statistics.mean([m.relevance_score for m in metrics_list])
                avg_quality = (avg_readability + avg_coherence + avg_relevance) / 3
                
                print(f"\n🤖 {name} (from {len(metrics_list)} generations):")
                print(f"   Average Generation Time: {avg_time:.2f}s")
                print(f"   Average Word Count: {avg_words:.0f}")
                print(f"   Average Readability: {avg_readability:.1f}/100")
                print(f"   Average Coherence: {avg_coherence:.1f}/100")
                print(f"   Average Relevance: {avg_relevance:.1f}/100")
                print(f"   Average Overall Quality: {avg_quality:.1f}/100")
        
        if not has_data:
            print("\n📝 No performance data available yet.")
            print("Generate some articles first to see statistics!")

def check_prerequisites():
    """Check if Ollama is installed and running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            required_models = ['llama2:7b', 'mistral:7b', 'codellama:7b']
            missing_models = [model for model in required_models if model not in model_names]
            
            print("✅ Ollama is running")
            print(f"📦 Available models: {', '.join(model_names) if model_names else 'None'}")
            
            if missing_models:
                print(f"⚠️  Missing models: {', '.join(missing_models)}")
                return False, missing_models
            else:
                print("✅ All required models are installed")
                return True, []
        else:
            print("❌ Ollama is not responding properly")
            return False, []
    except requests.exceptions.RequestException:
        print("❌ Ollama is not running or not accessible at localhost:11434")
        return False, []

def main():
    """Main function to run the chatbot"""
    print("🔧 Multi-LLM Article Generator Setup")
    print("="*50)
    
    # Check prerequisites
    is_ready, missing_models = check_prerequisites()
    
    if not is_ready:
        print("\n📋 Setup Instructions:")
        print("1. Install Ollama from: https://ollama.ai")
        print("2. Start Ollama service")
        print("3. Pull required models:")
        for model in ['llama2:7b', 'mistral:7b', 'codellama:7b']:
            print(f"   ollama pull {model}")
        print("\nThen run this script again.")
        return
    
    try:
        print("\n🚀 Starting chatbot...")
        chatbot = ArticleGeneratorChatbot()
        chatbot.run_interactive_session()
    except Exception as e:
        logger.error(f"Failed to start chatbot: {e}")
        print(f"❌ Failed to start chatbot: {e}")
        print("Please check the logs and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()
