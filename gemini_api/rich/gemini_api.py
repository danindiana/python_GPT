import google.generativeai as genai
import os
import sys
# --- Add imports for rich ---
from rich.console import Console
from rich.markdown import Markdown
# --------------------------

# --- Create a Console object (used by rich for printing) ---
console = Console()
# ---------------------------------------------------------

# --- Script Start ---
console.print(f"[bold cyan]--- Gemini Interactive Prompt Script (Model Selection) ---[/bold cyan]")

# --- Configure the API key ---
console.print(f"[info]Retrieving GEMINI_API_KEY from environment...[/info]")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    console.print(f"[bold red]ERROR:[/bold red] The 'GEMINI_API_KEY' environment variable was not found.")
    sys.exit("[bold red]Script aborted: API key is missing.[/bold red]")
else:
    console.print(f"[info]INFO:[/info] API key retrieved successfully.") # No need to print length

try:
    console.print("[info]INFO:[/info] Configuring genai...")
    genai.configure(api_key=api_key)
    console.print("[info]INFO:[/info] genai configured.")

    # --- List and Filter Available Models ---
    console.print("\n[info]INFO:[/info] Fetching available models supporting 'generateContent'...")
    suitable_models_data = [] # Store dicts with id and display name
    try:
        with console.status("[bold green]Listing available models...[/bold green]", spinner="dots"):
            for m in genai.list_models():
                # Check if the model supports the 'generateContent' method
                if 'generateContent' in m.supported_generation_methods:
                    # Extract the usable ID (part after 'models/')
                    model_id = m.name.split('/')[-1]
                    suitable_models_data.append({
                        "id": model_id,
                        # Use display_name if available, otherwise fall back to id
                        "display_name": m.display_name if hasattr(m, 'display_name') else model_id
                    })
    except Exception as e:
        console.print(f"\n[bold red]ERROR:[/bold red] Failed to list models: {e}")
        sys.exit("Script aborted: Could not retrieve model list.")

    if not suitable_models_data:
        console.print("[bold red]ERROR:[/bold red] No models found that support 'generateContent'. Cannot proceed.")
        sys.exit("Script aborted: No suitable models available.")

    # --- Present Model Selection Menu ---
    console.print("\n[bold green]Available Models for Text Generation:[/bold green]")
    for i, model_info in enumerate(suitable_models_data):
        # Display number, friendly name, and technical ID
        console.print(f"  [cyan]{i + 1}[/cyan]: {model_info['display_name']} ([dim]ID: {model_info['id']}[/dim])")

    selected_model_name = None
    while selected_model_name is None:
        try:
            choice_str = console.input(f"\n[bold yellow]Please select a model number (1-{len(suitable_models_data)}):[/bold yellow] ")
            choice_index = int(choice_str) - 1 # Convert input to 0-based index
            if 0 <= choice_index < len(suitable_models_data):
                selected_model_name = suitable_models_data[choice_index]['id'] # Get the ID
            else:
                # Invalid number range
                console.print(f"[yellow]Invalid choice. Please enter a number between 1 and {len(suitable_models_data)}.[/yellow]")
        except ValueError:
            # Input wasn't a number
            console.print("[yellow]Invalid input. Please enter a number.[/yellow]")
        except (EOFError, KeyboardInterrupt):
             # Handle Ctrl+D or Ctrl+C during selection prompt
             console.print("\n[bold red]Selection cancelled by user.[/bold red]")
             sys.exit("Script aborted.")

    # --- Instantiate the SELECTED model ---
    console.print(f"\n[info]INFO:[/info] Using model '[bold]{selected_model_name}[/bold]'...")
    try:
        model = genai.GenerativeModel(selected_model_name)
        console.print("[info]INFO:[/info] Model selected and ready.")
    except Exception as e:
         console.print(f"\n[bold red]ERROR:[/bold red] Failed to initialize selected model '{selected_model_name}': {e}")
         sys.exit("Script aborted: Could not initialize model.")

    # --- Get Prompt from User Input ---
    console.print("\n[bold]Now, enter your text prompt.[/bold]")
    prompt = console.input(f"[bold yellow]Prompt for {selected_model_name}:[/bold yellow] ")

    if not prompt.strip():
        console.print("[yellow]WARNING:[/yellow] You entered an empty prompt. Sending it anyway...")

    # --- Send Prompt and Get Response ---
    console.print(f"\n[info]INFO:[/info] Sending your prompt to Gemini (model: [bold]{selected_model_name}[/bold])...")
    try:
        with console.status("[bold green]Waiting for Gemini response...[/bold green]", spinner="dots"):
             response = model.generate_content(prompt)
        console.print("[info]INFO:[/info] Response received.")
    except Exception as e:
        console.print(f"\n[bold red]ERROR:[/bold red] Failed during API call to 'generate_content': {e}")
        sys.exit("Script aborted: API call failed.")


    # --- Save the Raw Response to a File (Optional but recommended) ---
    output_filename = "gemini_output.md"
    try:
        console.print(f"[info]INFO:[/info] Saving raw response to '[cyan]{output_filename}[/cyan]'...")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(response.text)
        console.print(f"[info]INFO:[/info] Successfully saved raw response.")
    except IOError as e:
        console.print(f"[bold red]ERROR:[/bold red] Could not write to file '{output_filename}': {e}")

    # --- Print the Response using rich Markdown rendering ---
    console.print("\n[bold magenta]--- Gemini Response ---[/bold magenta]")
    markdown_output = Markdown(response.text)
    console.print(markdown_output)
    console.print("[bold magenta]---------------------[/bold magenta]")

except Exception as e:
    # Catch any other unexpected errors
    console.print(f"\n[bold red]!!! An unexpected script error occurred: {e}[/bold red]")

finally:
    # This will run even if errors occur (unless sys.exit was called)
    console.print("\n[bold cyan]--- Script Finished ---[/bold cyan]")
