from textual.app import App, ComposeResult
from textual.widgets import Input, Static


class HelloWorldApp(App):
    """A simple Textual app with hello world text and an input box."""

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Static("hello, world!")
        yield Input(placeholder="Type something here...")


def main():
    """Run the app."""
    app = HelloWorldApp()
    app.run()


if __name__ == "__main__":
    main()
