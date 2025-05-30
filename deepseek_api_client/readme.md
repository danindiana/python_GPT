<div align="center">
  <img src="https://github.com/user-attachments/assets/8cd9c98d-4997-4148-8679-b1946e8274f4" alt="DeepSeek Chat Thumbnail" width="400"/>
</div>


This is a Python program that creates a simple GUI chat application for interacting with the DeepSeek AI API using the Flet framework. Here's a breakdown of what it does:

### Key Components:
1. **Imports**:
   - `flet` (as `ft`): For building the GUI
   - `requests`: For making HTTP requests to the DeepSeek API

   - `os` and `dotenv`: For loading the API key from environment variables

2. **Configuration**:
   - Loads environment variables from `.env` file (where API key should be stored)
   - Defines the DeepSeek API endpoint URL

3. **Main Function**:
   - Creates a Flet page titled "Deepseek Chat"
   - Sets up a chat interface with:
     - A scrollable chat history (`ListView`)
     - A text input field for new messages
     - A send button

4. **Functionality**:
   - When user sends a message:
     1. The message appears in the chat with "You:" prefix
     2. The message is sent to DeepSeek's API
     3. The AI's response (or error message) appears in the chat
   - Uses the `deepseek-chat` model for responses

5. **UI Layout**:
   - Vertical layout with:
     - Title at the top
     - Chat history in the middle (with border)
     - Input field and send button at the bottom

### How to Use:
1. You would need to:
   - Have a DeepSeek API key in a `.env` file as `DEEPSEEK_API_KEY`
   - Install the required packages (`flet`, `requests`, `python-dotenv`)
2. When run, it opens a window where you can chat with the AI

### Error Handling:
- Catches and displays any errors that occur during the API request
- Shows error messages in red text

This creates a basic but functional AI chat interface similar to many chatbot applications, specifically tailored for the DeepSeek API.

![image](https://github.com/user-attachments/assets/c7f731ed-f1d7-4467-9ca1-3c54879cd525)
