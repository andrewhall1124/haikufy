from textual.app import App, ComposeResult
from textual.widgets import Input, Static, Button, Label
from textual.containers import Vertical, Horizontal
from textual import work
from haikufy.converter import HaikuConverter
import pyperclip


class HaikufyApp(App):
    CSS_PATH = "app.tcss"

    def __init__(self):
        super().__init__()
        self.converter = HaikuConverter()
        self.current_haiku = ""  # Store the current haiku for copying

    def compose(self) -> ComposeResult:
        with Vertical(id="main-container"):
            yield Label("Enter your text to convert to a haiku:", id="title")
            yield Input(placeholder="Type your message here...", id="input-box")
            yield Button("Haikufy", id="haikufy-button", variant="primary")
            yield Static("", id="haiku-output")
            with Horizontal(id="button-container"):
                yield Button(
                    "Copy Haiku", id="copy-button", variant="success", disabled=True
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button click"""
        if event.button.id == "haikufy-button":
            input_widget = self.query_one("#input-box", Input)
            text = input_widget.value.strip()

            if text:
                # Show loading in output container
                self.query_one("#haiku-output", Static).update("Generating haiku...")

                # Disable buttons during generation
                event.button.disabled = True
                self.query_one("#copy-button", Button).disabled = True

                # Start the async haiku generation
                self.generate_haiku_async(text)
            else:
                self.query_one("#haiku-output", Static).update(
                    "Please enter some text first!"
                )

        elif event.button.id == "copy-button":
            # Copy the current haiku to clipboard
            if self.current_haiku:
                try:
                    pyperclip.copy(self.current_haiku)
                    # Temporarily update button text to show feedback
                    event.button.label = "Copied!"
                    self.set_timer(
                        1.5,
                        lambda: setattr(
                            self.query_one("#copy-button", Button),
                            "label",
                            "Copy Haiku",
                        ),
                    )
                except Exception as e:
                    self.query_one("#haiku-output", Static).update(
                        f"Error copying to clipboard: {e}"
                    )

    @work(exclusive=True, thread=True)
    def generate_haiku_async(self, text: str) -> None:
        """Generate haiku in a background worker"""
        try:
            # This runs in a thread to avoid blocking the UI
            haiku, syllable_counts, is_valid = self.converter.generate_haiku(text)

            # Format the haiku output
            output = f"{haiku}"

            # Update the UI with the result (using call_from_thread for thread safety)
            self.call_from_thread(self.update_result, output, None)

        except Exception as e:
            # Update the UI with error (using call_from_thread for thread safety)
            self.call_from_thread(self.update_result, None, str(e))

    def update_result(self, output: str | None, error: str | None) -> None:
        """Update UI with results (called from worker thread)"""
        if error:
            self.query_one("#haiku-output", Static).update(f"Error: {error}")
            self.current_haiku = ""
        elif output:
            self.query_one("#haiku-output", Static).update(output)
            # Store the haiku text (strip the trailing newlines for cleaner copying)
            self.current_haiku = output.strip()
            # Enable the copy button since we now have a haiku
            self.query_one("#copy-button", Button).disabled = False

        # Re-enable the haikufy button
        self.query_one("#haikufy-button", Button).disabled = False


if __name__ == "__main__":
    app = HaikufyApp()
    app.run()
