from textual.app import App, ComposeResult
from textual.widgets import Input, Static, Button, Label
from textual.containers import Vertical
from textual import work
from haikufy.converter import HaikuConverter


class HaikufyApp(App):
    CSS = """
    Screen {
        align: center middle;
    }

    #main-container {
        width: 80;
        height: auto;
        border: solid $primary;
        padding: 1 2;
    }

    #input-box {
        margin-bottom: 1;
    }

    #haikufy-button {
        width: 20;
        margin-bottom: 1;
    }

    #loading {
        color: $warning;
        margin-bottom: 1;
    }

    #haiku-output {
        color: $success;
        text-align: center;
        margin-top: 1;
        padding: 1;
        border: solid $accent;
    }
    """

    def __init__(self):
        super().__init__()
        self.converter = HaikuConverter()

    def compose(self) -> ComposeResult:
        with Vertical(id="main-container"):
            yield Label("Enter your text to convert to a haiku:", id="title")
            yield Input(placeholder="Type your message here...", id="input-box")
            yield Button("Haikufy", id="haikufy-button", variant="primary")
            yield Label("", id="loading")
            yield Static("", id="haiku-output")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button click"""
        if event.button.id == "haikufy-button":
            input_widget = self.query_one("#input-box", Input)
            text = input_widget.value.strip()

            if text:
                # Clear previous output and show loading
                self.query_one("#haiku-output", Static).update("")
                self.query_one("#loading", Label).update("Generating haiku...")

                # Disable button during generation
                event.button.disabled = True

                # Start the async haiku generation
                self.generate_haiku_async(text)
            else:
                self.query_one("#loading", Label).update("Please enter some text first!")

    @work(exclusive=True, thread=True)
    def generate_haiku_async(self, text: str) -> None:
        """Generate haiku in a background worker"""
        try:
            # This runs in a thread to avoid blocking the UI
            haiku, syllable_counts, is_valid = self.converter.generate_haiku(text)

            # Format the haiku output
            output = f"{haiku}\n\n"
            
            # Update the UI with the result (using call_from_thread for thread safety)
            self.call_from_thread(self.update_result, output, None)

        except Exception as e:
            # Update the UI with error (using call_from_thread for thread safety)
            self.call_from_thread(self.update_result, None, str(e))

    def update_result(self, output: str | None, error: str | None) -> None:
        """Update UI with results (called from worker thread)"""
        self.query_one("#loading", Label).update("")

        if error:
            self.query_one("#haiku-output", Static).update(f"Error: {error}")
        elif output:
            self.query_one("#haiku-output", Static).update(output)

        # Re-enable the button
        self.query_one("#haikufy-button", Button).disabled = False


def main():
    app = HaikufyApp()
    app.run()


if __name__ == "__main__":
    main()
