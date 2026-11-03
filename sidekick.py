"""
The Sidekick: a create_agent worker wrapped in an evaluator loop.

The worker is a single create_agent. Around it we run our own loop that checks
the worker's answer against the user's success criteria and either accepts it,
asks for another attempt, or returns to the user with a question.

Middleware provides:
- Planning
- PII protection
- Model-call limits
- Human-in-the-loop approval
- Tool-error recovery
"""

import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    TodoListMiddleware,
)
from langchain_core.messages import ToolMessage

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from sidekick_tools import get_all_tools


# ============================================================
# Configuration
# ============================================================

load_dotenv(override=True)

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.join(HERE, "sandbox")

MAX_ATTEMPTS = 3


# ============================================================
# Evaluator output
# ============================================================

class EvaluatorOutput(BaseModel):
    """Structured output returned by the evaluator."""

    feedback: str = Field(
        description="Feedback on the assistant's response"
    )

    success_criteria_met: bool = Field(
        description="Whether the success criteria have been met"
    )

    user_input_needed: bool = Field(
        description=(
            "True if the assistant has a question, "
            "needs clarification, or is stuck and needs the user"
        )
    )


# ============================================================
# Worker system prompt
# ============================================================

WORKER_PROMPT = """
You are Sidekick, a capable personal assistant who completes tasks for the user.

You have:
- A real web browser
- A sandbox filesystem
- Web search
- Wikipedia
- The ability to send push notifications

When you use the browser, navigate to a page and read it with a snapshot
rather than clicking around unnecessarily.

Dismiss cookie banners and popups yourself by clicking in the browser.

If you reach something only a human can do, such as:
- Logging in
- CAPTCHA
- Two-factor authentication

use the request_human_help tool to tell the user exactly what to do
in their browser window.

Then carry on once they have done it.

For flight searches, use Google Flights in your browser.

Go straight to:

https://www.google.com/travel/flights?q=...

with a natural-language query such as:

"flights from New York to London leaving 14 July returning 21 July"

Keep working on the task until:
- The success criteria are met, or
- You genuinely need to ask the user a question.

If you have a question, ask it plainly.

When you are finished, give your final answer clearly,
saying what you did, what you produced, and what you found.
"""


# ============================================================
# Middleware
# ============================================================

class TolerateToolErrors(AgentMiddleware):
    """
    Convert tool failures into messages that the model can understand.

    Instead of crashing the entire run, the model receives the error
    and can try another approach.
    """

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)

        except Exception as error:
            return ToolMessage(
                content=(
                    f"That tool call failed: {error}. "
                    "Try another approach."
                ),
                tool_call_id=request.tool_call["id"],
            )


# ============================================================
# Sidekick
# ============================================================

class Sidekick:
    """
    Main Sidekick application.

    Responsibilities:
    - Start tools and MCP servers
    - Create the worker agent
    - Create the evaluator
    - Run tasks
    - Handle HITL interruptions
    - Resume paused tasks
    - Retry failed tasks
    """

    def __init__(self):
        self.sidekick_id = str(uuid.uuid4())

        # LangGraph checkpoint memory
        self.memory = InMemorySaver()

        # Runtime objects
        self.tools = None
        self.sessions = None
        self.worker = None
        self.evaluator = None

        # Current task state
        self.task = ""
        self.success_criteria = ""
        self.attempts = 0

        # HITL state
        self.paused = False
        self.pending_actions = 0

        # Todo plan
        self.todos = []

    # ========================================================
    # Setup
    # ========================================================

    async def setup(self):
        """Initialize tools, MCP servers, worker and evaluator."""

        # Create sandbox directory
        os.makedirs(SANDBOX, exist_ok=True)

        # Load all tools
        self.tools, self.sessions = await get_all_tools(SANDBOX)

        # ----------------------------------------------------
        # Worker agent
        # ----------------------------------------------------

        self.worker = create_agent(
            model="openai:gpt-5.4-mini",
            tools=self.tools,

            system_prompt=(
                f"{WORKER_PROMPT}\n\n"
                f"Today is {datetime.now():%A %d %B %Y}."
            ),

            middleware=[
                # Recover from tool errors
                TolerateToolErrors(),

                # Maintain a task plan
                TodoListMiddleware(),

                # Protect email information
                PIIMiddleware("email"),

                # Protect credit-card information
                PIIMiddleware(
                    "credit_card",
                    apply_to_tool_results=True,
                ),

                # Prevent unlimited model calls
                ModelCallLimitMiddleware(
                    run_limit=30
                ),

                # Require human approval for sensitive actions
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "send_push_notification": True,
                        "request_human_help": True,
                    }
                ),
            ],

            # Enable checkpointing
            checkpointer=self.memory,
        )

        # ----------------------------------------------------
        # Evaluator
        # ----------------------------------------------------

        self.evaluator = (
            ChatOpenAI(
                model="gpt-5.4-mini"
            )
            .with_structured_output(EvaluatorOutput)
        )

    # ========================================================
    # Evaluator
    # ========================================================

    async def evaluate(
        self,
        message: str,
        success_criteria: str,
        last_reply: str,
        tools_used: list[str],
    ) -> EvaluatorOutput:
        """
        Ask the evaluator whether the worker completed the task.
        """

        prompt = f"""
You decide whether an assistant has met the success criteria for a task.

The user's request was:

{message}

The success criteria are:

{success_criteria}

The tools the assistant called while working, in order:

{", ".join(tools_used) or "none"}

The assistant's most recent reply was:

{last_reply}

Decide whether the success criteria are met, using the tool calls
as evidence of what was actually done.

Also decide whether the assistant needs more input from the user,
either because it asked a question, needs clarification, or seems stuck.

Give brief, concrete feedback.
"""

        return await self.evaluator.ainvoke(prompt)

    # ========================================================
    # Run a task
    # ========================================================

    async def run_turn(
        self,
        message: str,
        success_criteria: str,
        history: list,
    ) -> list:
        """
        Start a new Sidekick task.
        """

        self.task = message

        self.success_criteria = (
            success_criteria
            or "The answer should be clear, correct and complete"
        )

        self.attempts = 0
        self.todos = []

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{message}\n\n"
                        f"The success criteria for this task are: "
                        f"{self.success_criteria}"
                    ),
                }
            ]
        }

        new_history = history + [
            {
                "role": "user",
                "content": message,
            }
        ]

        return await self._advance(
            payload,
            new_history,
        )

    # ========================================================
    # Resume after HITL approval
    # ========================================================

    async def resume(self, history: list) -> list:
        """
        Approve the actions that caused the worker to pause
        and continue the same execution.
        """

        payload = Command(
            resume={
                "decisions": [
                    {"type": "approve"}
                ] * self.pending_actions
            }
        )

        return await self._advance(
            payload,
            history,
        )

    # ========================================================
    # Advance worker execution
    # ========================================================

    async def _advance(
        self,
        payload,
        history: list,
    ) -> list:
        """
        Execute the worker and handle:

        - Streaming
        - TODO updates
        - HITL interruptions
        - Evaluation
        - Retry logic
        """

        config = {
            "configurable": {
                "thread_id": self.sidekick_id
            }
        }

        while True:

            result = None

            # ------------------------------------------------
            # Run worker
            # ------------------------------------------------

            async for result in self.worker.astream(
                payload,
                config=config,
                stream_mode="values",
            ):
                self.todos = result.get(
                    "todos",
                    self.todos,
                )

            # ------------------------------------------------
            # Check for HITL interruption
            # ------------------------------------------------

            if "__interrupt__" in result:

                actions = (
                    result["__interrupt__"][0]
                    .value["action_requests"]
                )

                self.paused = True
                self.pending_actions = len(actions)

                described = "\n".join(
                    action["description"]
                    for action in actions
                )

                return history + [
                    {
                        "role": "assistant",
                        "content": (
                            "Waiting for your approval:\n"
                            f"{described}"
                        ),
                    }
                ]

            # ------------------------------------------------
            # Worker completed
            # ------------------------------------------------

            self.paused = False

            reply = result["messages"][-1].content

            # Extract tool calls
            tools_used = [
                call["name"]
                for message in result["messages"]
                for call in (
                    getattr(
                        message,
                        "tool_calls",
                        None,
                    )
                    or []
                )
            ]

            self.attempts += 1

            # ------------------------------------------------
            # Evaluate worker result
            # ------------------------------------------------

            verdict = await self.evaluate(
                self.task,
                self.success_criteria,
                reply,
                tools_used,
            )

            # ------------------------------------------------
            # Stop conditions
            # ------------------------------------------------

            if (
                verdict.success_criteria_met
                or verdict.user_input_needed
                or self.attempts >= MAX_ATTEMPTS
            ):

                return history + [
                    {
                        "role": "assistant",
                        "content": reply,
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"Evaluator: {verdict.feedback}"
                        ),
                    },
                ]

            # ------------------------------------------------
            # Retry
            # ------------------------------------------------

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Your last response did not meet "
                            "the success criteria.\n\n"
                            f"Here is the feedback:\n"
                            f"{verdict.feedback}\n\n"
                            "Please keep working and address it."
                        ),
                    }
                ]
            }

    # ========================================================
    # Cleanup
    # ========================================================

    def cleanup(self):
        """Shut down MCP servers and close browser resources."""

        if self.sessions:
            self.sessions.stop()