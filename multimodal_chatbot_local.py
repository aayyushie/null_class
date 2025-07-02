import os
import gradio as gr
import google.generativeai as genai
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# --- Load environment variables from .env if present ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set! Please set it in your environment or in a .env file.")

genai.configure(api_key=GOOGLE_API_KEY)
# Initialize Gemini multi-modal model
model = genai.GenerativeModel("gemini-pro-vision")

# --- Multi-modal chat function ---
def multimodal_chat(user_message, image=None, history=[]):
    """
    Handles both text and image input, integrates chat history, and uses Gemini for multi-modal reasoning.
    """
    parts = []
    if image is not None:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        parts.append(image)
    if user_message:
        parts.append(user_message)
    # Add last 3 turns of chat history for context
    for turn in history[-3:]:
        parts.append(f"User: {turn[0]}")
        parts.append(f"Assistant: {turn[1]}")
    try:
        response = model.generate_content(parts)
        answer = response.text
    except Exception as e:
        answer = f"Error: {str(e)}"
    history = history + [[user_message, answer]]
    return answer, history

# --- Image generation function ---
def generate_image(prompt):
    """
    Uses Gemini or PaLM for image generation if available. If not, returns a placeholder message.
    """
    # Gemini API does not currently support image generation (as of June 2024)
    # If you have access to a PaLM or Gemini image generation endpoint, add it here.
    return "Image generation is not supported by Gemini API yet. You can integrate Stable Diffusion, DALL-E, or future Gemini/PaLM endpoints here."

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("""
    # 🤖 Multi-Modal Chatbot (Gemini/PaLM)
    
    **Features:**
    - Accepts both text and image input
    - Uses Google Gemini API for multi-modal reasoning
    - (Pluggable) Image generation from text prompt (add your own API or model)
    - Maintains chat history for context-aware, seamless visual and textual conversation
    
    **Instructions:**
    1. Enter a message and/or upload an image.
    2. The chatbot will respond, integrating both modalities if provided.
    3. To generate an image from a prompt, use the image generation section (if supported).
    """)
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label="Chat History")
            user_msg = gr.Textbox(label="Your Message")
            image_input = gr.Image(type="pil", label="Upload Image (optional)")
            send_btn = gr.Button("Send")
            img_prompt = gr.Textbox(label="Image Generation Prompt")
            gen_img_btn = gr.Button("Generate Image")
        with gr.Column():
            gen_img_output = gr.Textbox(label="Generated Image (not supported by Gemini/PaLM as of now)")

    send_btn.click(
        multimodal_chat,
        inputs=[user_msg, image_input, chatbot],
        outputs=[user_msg, chatbot]
    )
    gen_img_btn.click(
        generate_image,
        inputs=img_prompt,
        outputs=gen_img_output
    )

demo.launch()
