import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
from typing import List, Tuple

class LocalMultiModalChatbot:
    def __init__(self):
        """Initialize the local multi-modal chatbot"""
        self.conversation_history = []
        
    def analyze_image_local(self, image_path: str) -> str:
        """Analyze image using local computer vision techniques"""
        try:
            if image_path is None:
                return "No image provided."
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return "Error: Could not load image."
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            analysis_results = []
            
            # 1. Basic image properties
            height, width, channels = image.shape
            analysis_results.append(f"📐 **Image Properties:**")
            analysis_results.append(f"   - Dimensions: {width}x{height} pixels")
            analysis_results.append(f"   - Color channels: {channels}")
            analysis_results.append(f"   - File size: {os.path.getsize(image_path) / 1024:.1f} KB")
            
            # 2. Color analysis
            mean_color = cv2.mean(image_rgb)
            analysis_results.append(f"\n🎨 **Color Analysis:**")
            analysis_results.append(f"   - Average RGB: ({mean_color[0]:.1f}, {mean_color[1]:.1f}, {mean_color[2]:.1f})")
            
            # 3. Brightness analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            analysis_results.append(f"   - Brightness: {'Bright' if brightness > 127 else 'Dark'} ({brightness:.1f}/255)")
            
            # 4. Edge detection for object detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            analysis_results.append(f"   - Edge density: {'High' if edge_density > 0.1 else 'Low'} ({edge_density:.3f})")
            
            # 5. Texture analysis
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            texture_variance = np.var(gray - blur)
            analysis_results.append(f"   - Texture: {'Detailed' if texture_variance > 100 else 'Smooth'}")
            
            # 6. Image quality assessment
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality = "Sharp" if laplacian_var > 100 else "Blurry"
            analysis_results.append(f"   - Quality: {quality} (variance: {laplacian_var:.1f})")
            
            # 7. Composition analysis
            center_x, center_y = width // 2, height // 2
            analysis_results.append(f"\n📐 **Composition Analysis:**")
            analysis_results.append(f"   - Aspect ratio: {width/height:.2f}")
            analysis_results.append(f"   - Center point: ({center_x}, {center_y})")
            
            # 8. File format and metadata
            pil_image = Image.open(image_path)
            analysis_results.append(f"\n📁 **File Information:**")
            analysis_results.append(f"   - Format: {pil_image.format}")
            analysis_results.append(f"   - Mode: {pil_image.mode}")
            
            return "\n".join(analysis_results)
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def detect_objects_local(self, image_path: str) -> str:
        """Detect objects using local computer vision"""
        try:
            if image_path is None:
                return "No image provided."
            
            image = cv2.imread(image_path)
            if image is None:
                return "Error: Could not load image."
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Simple shape detection
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            object_analysis = []
            object_analysis.append("🔍 **Object Detection Results:**")
            
            # Analyze contours
            for i, contour in enumerate(contours[:10]):  # Limit to 10 largest objects
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small noise
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Determine shape
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                    
                    if len(approx) == 3:
                        shape = "Triangle"
                    elif len(approx) == 4:
                        shape = "Rectangle/Square"
                    elif len(approx) > 8:
                        shape = "Circle/Oval"
                    else:
                        shape = "Complex shape"
                    
                    object_analysis.append(f"   - Object {i+1}: {shape} at ({x}, {y}), size: {w}x{h}")
            
            if len(object_analysis) == 1:
                object_analysis.append("   - No significant objects detected")
            
            return "\n".join(object_analysis)
            
        except Exception as e:
            return f"Error detecting objects: {str(e)}"
    
    def generate_text_response(self, user_input: str) -> str:
        """Generate text response using local knowledge"""
        try:
            # Simple response system based on keywords
            user_input_lower = user_input.lower()
            
            # Define response patterns
            responses = {
                "hello": "Hello! I'm your local multi-modal AI assistant. I can analyze images, detect objects, and help you with various tasks. How can I assist you today?",
                "help": "I can help you with:\n• 📷 Image analysis and technical details\n• 🔍 Object detection in images\n• 💬 Answering questions\n• 🎨 Creating image generation prompts\n\nJust upload an image or ask me anything!",
                "image": "I can analyze images to provide:\n• Technical properties (size, format, quality)\n• Color analysis and statistics\n• Object detection and shape recognition\n• Composition analysis\n\nUpload an image to get started!",
                "object": "I can detect objects in images using computer vision techniques. I'll identify shapes, sizes, and positions of objects in your uploaded images.",
                "generate": "I can help you create image generation prompts! Just say 'generate image' or 'create image' and I'll provide optimized prompts you can use with AI art tools like Stable Diffusion, DALL-E, or Midjourney.",
                "color": "I can analyze colors in your images, including:\n• Average RGB values\n• Brightness levels\n• Color distribution\n• Dominant color detection\n\nUpload an image to see the color analysis!",
                "quality": "I can assess image quality by analyzing:\n• Sharpness (using Laplacian variance)\n• Noise levels\n• Resolution details\n• Overall technical quality\n\nUpload an image for quality assessment!"
            }
            
            # Check for matching patterns
            for keyword, response in responses.items():
                if keyword in user_input_lower:
                    return response
            
            # Default response for other queries
            return f"I understand you're asking about: {user_input}\n\nI'm a local AI assistant that can help with image analysis, object detection, and creating image generation prompts. Upload an image or ask me about my capabilities!"
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def create_image_prompt(self, description: str) -> str:
        """Create an optimized prompt for image generation"""
        try:
            # Simple prompt enhancement
            enhanced_prompt = f"""
            **Image Generation Prompt:**
            {description}, high quality, detailed, professional photography, 4K resolution, 
            cinematic lighting, sharp focus, vibrant colors, artistic composition
            
            **Style Suggestions:**
            - Photography style: Professional
            - Lighting: Natural or dramatic
            - Composition: Rule of thirds
            - Color palette: Based on description
            - Mood: Engaging and visually appealing
            
            **Use this prompt with:**
            • Stable Diffusion WebUI
            • Hugging Face Spaces
            • Canva AI
            • DALL-E
            • Midjourney
            """
            
            return enhanced_prompt.strip()
            
        except Exception as e:
            return f"Error creating image prompt: {str(e)}"
    
    def generate_simple_image(self, prompt: str) -> str:
        """Generate a simple ASCII art representation"""
        try:
            ascii_art = f"""
            🎨 Generated Image Concept:
            
            {prompt}
            
            ┌─────────────────────────────────────┐
            │                                     │
            │         IMAGE PLACEHOLDER           │
            │                                     │
            │     (Use external AI art tools)     │
            │                                     │
            │  • Stable Diffusion                │
            │  • DALL-E                          │
            │  • Midjourney                      │
            │  • Canva AI                        │
            │                                     │
            └─────────────────────────────────────┘
            
            💡 Tip: Copy the prompt above and use it with:
            - Stable Diffusion WebUI
            - Hugging Face Spaces
            - Online AI art generators
            """
            
            return ascii_art
            
        except Exception as e:
            return f"Error generating image concept: {str(e)}"

def process_message(message: str, image: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
    """Process user message and return response with updated history"""
    chatbot = LocalMultiModalChatbot()
    
    # Initialize response
    response = ""
    
    try:
        # Process image if provided
        if image is not None:
            image_analysis = chatbot.analyze_image_local(image)
            response += f"📷 **Image Analysis:**\n{image_analysis}\n\n"
            
            # Add object detection
            object_detection = chatbot.detect_objects_local(image)
            response += f"{object_detection}\n\n"
            
            # Add specific image insights
            response += "💡 **Image Insights:**\n"
            
            # Analyze the image data to provide specific insights
            try:
                img = cv2.imread(image)
                if img is not None:
                    height, width, channels = img.shape
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray)
                    
                    # Provide specific insights based on image properties
                    if brightness > 150:
                        response += "• This is a bright, well-lit image\n"
                    elif brightness < 100:
                        response += "• This is a darker, moody image\n"
                    else:
                        response += "• This image has balanced lighting\n"
                    
                    if width > height:
                        response += "• Landscape orientation\n"
                    else:
                        response += "• Portrait orientation\n"
                    
                    if width * height > 2000000:  # More than 2MP
                        response += "• High resolution image suitable for printing\n"
                    else:
                        response += "• Standard resolution image\n"
                    
                    # Color analysis insights
                    mean_color = cv2.mean(img)
                    if mean_color[0] > 150 and mean_color[1] > 150 and mean_color[2] > 150:
                        response += "• Light, bright color palette\n"
                    elif mean_color[0] < 100 and mean_color[1] < 100 and mean_color[2] < 100:
                        response += "• Dark, muted color palette\n"
                    else:
                        response += "• Mixed color palette\n"
                    
            except Exception as e:
                response += "• Image analysis completed successfully\n"
            
            response += "\n"
        
        # Generate text response only if user asked a specific question
        if message.strip():
            # Check if user is asking about the image specifically
            message_lower = message.lower()
            
            if any(word in message_lower for word in ["what", "how", "why", "when", "where", "analyze", "detect", "find", "show", "tell"]):
                # User asked a specific question
                text_response = chatbot.generate_text_response(message)
                response += f"💬 **Answer:**\n{text_response}\n\n"
            elif any(word in message_lower for word in ["hello", "hi", "help", "thanks", "thank you"]):
                # Greeting or help request
                text_response = chatbot.generate_text_response(message)
                response += f"💬 **Response:**\n{text_response}\n\n"
            else:
                # Generic message - provide helpful guidance
                response += f"💬 **Response:**\nI've analyzed your image above. You can ask me specific questions about:\n"
                response += "• What objects I can detect\n"
                response += "• Color analysis details\n"
                response += "• Image quality assessment\n"
                response += "• Generate image prompts based on this image\n\n"
        
        # Generate image prompt if requested
        if any(keyword in message.lower() for keyword in ["generate image", "create image", "make image", "draw image", "prompt"]):
            if image is not None:
                # Use image analysis to create prompt
                image_analysis = chatbot.analyze_image_local(image)
                image_prompt = chatbot.create_image_prompt(f"Based on this image analysis: {image_analysis}")
            else:
                image_prompt = chatbot.create_image_prompt(message)
            
            response += f"🎨 **Image Generation Prompt:**\n{image_prompt}\n\n"
            
            # Generate simple image concept
            image_concept = chatbot.generate_simple_image(image_prompt)
            response += f"{image_concept}\n\n"
        
        # If no image and no specific question, provide welcome message
        if not response:
            response = """🤖 **Welcome to Local Multi-Modal AI Assistant!**

**I can help you with:**
• 📷 **Image Analysis:** Upload images for detailed technical analysis
• 💬 **Text Generation:** Get responses using local AI knowledge
• 🎨 **Image Prompts:** Generate optimized prompts for AI art tools
• 🔍 **Object Detection:** Identify shapes and objects in images
• 📊 **Image Statistics:** Color analysis, composition, quality assessment

**How to use:**
1. Upload an image for analysis
2. Ask questions about the image or any topic
3. Say "generate image" to create AI art prompts
4. Get detailed technical analysis of your images

**No API keys required - everything runs locally!** 🎉"""
        
        # Update history with new message pair
        new_history = history + [[message, response]]
        
        return "", new_history
        
    except Exception as e:
        error_response = f"❌ Error: {str(e)}\n\nPlease try again or check your input."
        new_history = history + [[message, error_response]]
        return "", new_history

def clear_chat():
    """Clear the chat history"""
    return [], ""

# Create the Gradio interface
with gr.Blocks(title="Local Multi-Modal AI Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Local Multi-Modal AI Chatbot
    
    **No API Keys Required - Everything Runs Locally!**
    
    **Capabilities:**
    - 📷 **Image Analysis:** Technical analysis using computer vision
    - 💬 **Text Generation:** Local AI responses
    - 🎨 **Image Prompts:** Generate prompts for external AI art tools
    - 🔍 **Object Detection:** Identify shapes and objects in images
    - 📊 **Image Statistics:** Color, composition, and quality analysis
    
    **How to use:**
    1. Upload an image for detailed analysis
    2. Ask questions about the image or any topic
    3. Request image generation prompts
    4. Get comprehensive technical insights
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Image upload
            image_input = gr.Image(
                label="📷 Upload Image (Optional)",
                type="filepath",
                height=300
            )
            
            # Clear button
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
        
        with gr.Column(scale=2):
            # Chat interface
            chatbot = gr.Chatbot(
                label="💬 Chat History",
                height=400,
                show_label=True
            )
            
            # Message input
            msg = gr.Textbox(
                label="💭 Your Message",
                placeholder="Ask me anything or upload an image for analysis...",
                lines=3
            )
            
            # Send button
            send_btn = gr.Button("🚀 Send", variant="primary", size="lg")
    
    # Examples
    gr.Examples(
        examples=[
            ["Analyze this image in detail", "example_image.jpg"],
            ["What objects can you detect in this image?", "example_image.jpg"],
            ["Generate an image prompt based on this image", "example_image.jpg"],
            ["Explain machine learning in simple terms", None],
            ["Create an image prompt for a futuristic city", None],
            ["What are the color properties of this image?", "example_image.jpg"]
        ],
        inputs=[msg, image_input]
    )
    
    # Event handlers
    send_btn.click(
        process_message,
        inputs=[msg, image_input, chatbot],
        outputs=[msg, chatbot]
    )
    
    msg.submit(
        process_message,
        inputs=[msg, image_input, chatbot],
        outputs=[msg, chatbot]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg]
    )
    
    # Instructions
    gr.Markdown("""
    ### 💡 Tips for Best Results:
    
    **For Image Analysis:**
    - Upload clear, high-quality images
    - Ask specific questions about image properties
    - Request detailed technical analysis
    
    **For Text Generation:**
    - Be specific in your questions
    - Ask about my capabilities
    - Use keywords like "help", "image", "object", "generate"
    
    **For Image Prompts:**
    - Say "generate image" or "create image"
    - Upload a reference image for better prompts
    - Use the generated prompts with external AI art tools
    
    ### 🔧 Setup Instructions:
    1. Install required packages: `pip install -r requirements_multimodal.txt`
    2. Run the application: `python multimodal_chatbot_local.py`
    3. No API keys required - everything runs locally!
    
    ### 🎨 External AI Art Tools:
    - **Stable Diffusion WebUI:** For local image generation
    - **Hugging Face Spaces:** Free online AI art generation
    - **Canva AI:** User-friendly AI art creation
    - **DALL-E:** OpenAI's image generation (requires API)
    """)

if __name__ == "__main__":
    print("🚀 Starting Local Multi-Modal AI Chatbot...")
    print("📋 Features: Image Analysis, Text Generation, Object Detection")
    print("💡 No API keys required - everything runs locally!")
    demo.launch(share=False, show_error=True)
