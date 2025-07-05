# Gemini Multi-Modal Chatbot

This project is a multi-modal chatbot powered by Google Gemini API. It can understand both text and image inputs, and respond intelligently using state-of-the-art AI.

## Features
- Accepts both text and image input
- Uses Google Gemini API for multi-modal reasoning
- User-friendly Gradio web interface
- Maintains chat history for context-aware responses

## Setup Instructions

### 1. Clone the Repository
Clone or download this project to your local machine.

### 2. Install Requirements
Install the required Python packages:
```sh
pip install -r requirements_multimodal.txt
```

### 3. Get a Google Gemini API Key
- Go to [Google AI Studio](https://aistudio.google.com/) and sign in with your Google account.
- Create a project and generate an API key.
- **Do not share your API key with anyone.**

### 4. Set Your API Key Securely
You can set your API key in one of two ways:

#### Option A: Using a `.env` file (recommended for local development)
1. Create a file named `.env` in the project directory.
2. Add this line (replace with your actual key):
   ```
   GOOGLE_API_KEY=your-actual-api-key-here
   ```
3. The app will automatically load this key.

#### Option B: Using an Environment Variable
- **Windows:**
  ```sh
  set GOOGLE_API_KEY=your-actual-api-key-here
  ```
- **Mac/Linux:**
  ```sh
  export GOOGLE_API_KEY=your-actual-api-key-here
  ```

### 5. Run the App
```sh
python app.py
```
- Open the Gradio link in your browser to interact with the chatbot.

## Security Best Practices
- **Never share your API key** in code, screenshots, or public repositories.
- **Do not commit your `.env` file** to version control. Add `.env` to your `.gitignore`.
- If you accidentally share your API key, **revoke it immediately** in Google AI Studio and generate a new one.

## Notes
- Image generation is not currently supported by the Gemini API. You can integrate Stable Diffusion or DALL-E for this feature if needed.
- For any issues or questions, please refer to the official [Gemini API documentation](https://ai.google.dev/).

---

**Enjoy your secure, multi-modal AI chatbot!** 