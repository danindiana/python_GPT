import flet as ft
import requests
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def main(page: ft.Page):
    page.title = "Deepseek Chat"
    chat = ft.ListView(expand=True)
    new_message = ft.TextField(hint_text="Type a message...", expand=True)
    
    def send_click(e):
        if not (msg := new_message.value.strip()): return
        
        chat.controls.append(ft.Text(f"You: {msg}"))
        new_message.value = ""
        page.update()
        
        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": msg}]
                }
            )
            response.raise_for_status()
            chat.controls.append(ft.Text(f"AI: {response.json()['choices'][0]['message']['content']}"))
        except Exception as e:
            chat.controls.append(ft.Text(f"Error: {str(e)}", color="red"))
        page.update()

    page.add(
        ft.Column([
            ft.Text("Deepseek Chat", size=20, weight="bold"),
            ft.Container(chat, border=ft.border.all(1), padding=10, expand=True),
            ft.Row([new_message, ft.ElevatedButton("Send", on_click=send_click)])
        ], expand=True)
    )

ft.app(main)
