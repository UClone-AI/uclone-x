"""Tool for managing the interactive execution plan of an agent session."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext


class PlanUpdateParams(BaseModel):
    """Parameters for updating the agent's execution plan."""

    action: Literal["create", "update"] = Field(
        description="Whether to create a new plan or update the existing one"
    )
    title: str | None = Field(default=None, description="Title of the plan (required for create)")
    steps: list[dict[str, Any]] | None = Field(
        default=None,
        description="List of steps. For 'create', each step should have 'description'. "
        "For 'update', include 'index' and optionally 'completed' or 'verification'.",
    )
    status: Literal["proposed", "in_progress", "completed", "rejected"] | None = Field(
        default=None, description="Status of the plan (for update)"
    )

    from pydantic import model_validator

    @model_validator(mode="after")
    def validate_create_requires_title(self) -> PlanUpdateParams:
        if self.action == "create" and not self.title:
            raise ValueError("title is required when action is 'create'")
        if self.action == "create" and not self.steps:
            raise ValueError("steps are required when action is 'create'")
        return self


class PlanUpdateTool(BaseTool[PlanUpdateParams]):
    """Tool that allows the agent to create or update an execution plan."""

    name = "update_plan"
    writes_files: ClassVar[bool] = False  # writes no file on the host (#1167)
    description = "Manage the execution plan for the session. Use this tool to create a new plan or mark steps as completed."
    params_type = PlanUpdateParams

    def run(self, params: PlanUpdateParams, context: ToolContext) -> dict[str, Any]:
        """Return the validated intent to be processed by the agent."""
        return params.model_dump()
