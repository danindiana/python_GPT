import flet as ft
import requests
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def main(page: ft.Page):
    # Default theme
    page.theme_mode = ft.ThemeMode.LIGHT
    page.update()

    # Theme toggle logic
    def toggle_theme(e):
        page.theme_mode = (
            ft.ThemeMode.DARK 
            if page.theme_mode == ft.ThemeMode.LIGHT 
            else ft.ThemeMode.LIGHT
        )
        theme_icon_button.icon = (
            ft.Icons.WB_SUNNY if page.theme_mode == ft.ThemeMode.DARK else ft.Icons.NIGHTLIGHT_ROUND
        )
        page.update()

    # Theme toggle button with correct icon enum
    theme_icon_button = ft.IconButton(
        icon=ft.Icons.NIGHTLIGHT_ROUND,
        on_click=toggle_theme,
        tooltip="Toggle dark/light mode"
    )

    page.title = "Deepseek Chat"
    chat = ft.ListView(expand=True)
    new_message = ft.TextField(hint_text="Type a message...", expand=True)

    def send_click(e):
        if not (msg := new_message.value.strip()):
            return

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
            ai_response = response.json()['choices'][0]['message']['content']
            chat.controls.append(ft.Text(f"AI: {ai_response}"))
        except Exception as ex:
            chat.controls.append(ft.Text(f"Error: {str(ex)}", color="red"))
        page.update()

    # UI Layout
    page.add(
        ft.Column([
            ft.Row([
                ft.Text("Deepseek Chat", size=20, weight="bold"),
                theme_icon_button
            ], alignment="spaceBetween"),
            ft.Container(
                content=chat,
                border=ft.border.all(1),
                padding=10,
                expand=True
            ),
            ft.Row([
                new_message,
                ft.ElevatedButton("Send", on_click=send_click)
            ])
        ], expand=True)
    )

# Run the app
ft.app(target=main)
