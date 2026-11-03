"""
Gradio application for Sidekick.

Run with:

    uv run app.py
"""

import html

import gradio as gr

import styles

from sidekick import Sidekick


# ============================================================
# Launch configuration
# ============================================================

LAUNCH_STYLE = {
    "theme": styles.THEME,
    "css": styles.CSS,
    "head": styles.JS,
}


# ============================================================
# Header
# ============================================================

HEADER = """
<div id="header">
    <div class="context-label">
        Your personal co-worker
    </div>

    <h1>Sidekick</h1>

    <div class="brand-bar"></div>
</div>
"""


# ============================================================
# Render TODO plan
# ============================================================

def render_todos(todos):
    """
    Convert Sidekick TODOs into HTML for the plan panel.
    """

    if not todos:

        items = (
            '<div class="placeholder">'
            'The Sidekick will write its plan here as it works'
            '</div>'
        )

    else:

        items = (
            "<ul>"
            +
            "".join(
                (
                    f'<li class="{todo["status"]}">'
                    f'<span class="mark"></span>'
                    f'{html.escape(todo["content"])}'
                    '</li>'
                )
                for todo in todos
            )
            +
            "</ul>"
        )

    return f"<h3>Plan</h3>{items}"


# ============================================================
# Setup Sidekick
# ============================================================

async def setup():

    sidekick = Sidekick()

    await sidekick.setup()

    # Enable Go button
    return (
        sidekick,
        gr.update(interactive=True),
    )


# ============================================================
# Process user message
# ============================================================

async def process_message(
    sidekick,
    message,
    success_criteria,
    history,
):

    # User clicked before MCP setup completed
    if sidekick is None:

        return (
            history,
            gr.update(visible=False),
            sidekick,
        )

    results = await sidekick.run_turn(
        message,
        success_criteria,
        history,
    )

    return (
        results,
        gr.update(
            visible=sidekick.paused
        ),
        sidekick,
    )


# ============================================================
# Approve HITL action
# ============================================================

async def approve(
    sidekick,
    history,
):

    results = await sidekick.resume(
        history
    )

    return (
        results,
        gr.update(
            visible=sidekick.paused
        ),
        sidekick,
    )


# ============================================================
# Watch TODO plan
# ============================================================

def watch_todos(sidekick):
    """
    Called by the timer.

    The timer is the plan panel's only writer,
    allowing live updates while the Sidekick works.
    """

    if sidekick:

        return render_todos(
            sidekick.todos
        )

    return render_todos([])


# ============================================================
# Reset Sidekick
# ============================================================

async def reset(sidekick):

    if sidekick:
        sidekick.cleanup()

    new_sidekick = Sidekick()

    await new_sidekick.setup()

    return (
        "",
        "",
        None,
        gr.update(visible=False),
        new_sidekick,
    )


# ============================================================
# Resource cleanup
# ============================================================

def free_resources(sidekick):

    if sidekick:
        sidekick.cleanup()


# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks(
    title="Sidekick"
) as ui:

    # Header
    gr.HTML(HEADER)

    # --------------------------------------------------------
    # Chat + Plan
    # --------------------------------------------------------

    sidekick = gr.State(
        delete_callback=free_resources
    )

    with gr.Row():

        chatbot = gr.Chatbot(
            label="Sidekick",
            height=320,
            scale=3,
            elem_id="chat",
        )

        with gr.Column(scale=1):

            todos_panel = gr.HTML(
                render_todos([]),
                elem_id="plan-panel",
            )

    # --------------------------------------------------------
    # User input
    # --------------------------------------------------------

    with gr.Group(
        elem_id="ask-panel"
    ):

        with gr.Row():

            message = gr.Textbox(
                show_label=False,
                placeholder=(
                    "Your request to the Sidekick"
                ),
            )

        with gr.Row():

            success_criteria = gr.Textbox(
                show_label=False,
                placeholder=(
                    "What are your success criteria?"
                ),
            )

    # --------------------------------------------------------
    # Buttons
    # --------------------------------------------------------

    with gr.Row():

        reset_button = gr.Button(
            "Reset",
            elem_id="reset-button",
        )

        approve_button = gr.Button(
            "Approve and continue",
            visible=False,
            elem_id="approve-button",
        )

        go_button = gr.Button(
            "Go!",
            elem_id="go-button",
            interactive=False,
        )

    # --------------------------------------------------------
    # Timer
    # --------------------------------------------------------

    timer = gr.Timer(1)

    # --------------------------------------------------------
    # Event handlers
    # --------------------------------------------------------

    # Initialize Sidekick when application loads
    ui.load(
        setup,
        [],
        [sidekick, go_button],
    )

    # Update TODO plan every second
    timer.tick(
        watch_todos,
        [sidekick],
        [todos_panel],
        show_progress="hidden",
    )

    # Press Enter in message box
    message.submit(
        process_message,
        [
            sidekick,
            message,
            success_criteria,
            chatbot,
        ],
        [
            chatbot,
            approve_button,
            sidekick,
        ],
    )

    # Press Enter in success criteria box
    success_criteria.submit(
        process_message,
        [
            sidekick,
            message,
            success_criteria,
            chatbot,
        ],
        [
            chatbot,
            approve_button,
            sidekick,
        ],
    )

    # Click Go!
    go_button.click(
        process_message,
        [
            sidekick,
            message,
            success_criteria,
            chatbot,
        ],
        [
            chatbot,
            approve_button,
            sidekick,
        ],
    )

    # Click Approve
    approve_button.click(
        approve,
        [
            sidekick,
            chatbot,
        ],
        [
            chatbot,
            approve_button,
            sidekick,
        ],
    )

    # Click Reset
    reset_button.click(
        reset,
        [sidekick],
        [
            message,
            success_criteria,
            chatbot,
            approve_button,
            sidekick,
        ],
    )


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":

    ui.launch(
        inbrowser=True,
        **LAUNCH_STYLE,
    )