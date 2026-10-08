"""The `browser` tool: a clone uses a real browser, U0's own Chrome (design `browser-agent.md`).

The tool is thin. It validates the call and hands it to the process's `BrowserService`,
which holds the link to Chrome and the tabs of each (conversation, clone). Step 1 reads
pages; step 2 acts on them (`click`, `type`, …). Step 5 adds `look`, a picture of the
page for a model that reads images, and `click` at a point of that picture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from uclone_x.browser.extension import default_browser_link
from uclone_x.browser.service import BrowserService, TabKey
from uclone_x.core.provenance import Provenance
from uclone_x.errors import PathTraversalError, PlainRefusalError
from uclone_x.llm.models import CANNOT_SEE_PICTURES
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext, ToolResult

#: What `look` says to a model that does not read images, beside the page's elements (G8).
NO_IMAGES_NOTE = f"{CANNOT_SEE_PICTURES}, so look returned the page's elements as text instead."

#: What `read` returns at most when the call names no `max_length`.
READ_MAX_LENGTH: Final = 20000

#: Documentation returned by `help` action describing all available actions and parameters.
HELP_TEXT: Final = (
    "browser actions:\n"
    "• open(url): load a web address and see page controls with ref IDs (e.g. e12).\n"
    "• snapshot: see current page controls again with ref IDs.\n"
    "• read(max_length=20000): page text as Markdown.\n"
    "• find(query): locate controls matching text.\n"
    "• look: picture of the page for vision models, or snapshot if model cannot see.\n"
    "• click(ref, double=False, right=False, hover=False) or click(x, y, double=False, right=False, hover=False): click control or coordinate.\n"
    "• type(ref, text, submit=False, append=False): type text into an input field.\n"
    "• select(ref, options): choose dropdown options.\n"
    "• check(ref, on=True): check or uncheck a checkbox.\n"
    "• press(key): press a key or chord (e.g. Enter, Escape, Tab, Meta+A).\n"
    "• scroll(direction='down', amount=1.0, ref=None): scroll page or element.\n"
    "• upload(ref, paths): choose workspace files in file picker.\n"
    "• wait(text=None, ref=None, ms=None): wait for text, ref, or milliseconds (max 10000).\n"
    "• back / forward / reload: tab history navigation.\n"
    "• tab(index=None, close=False): list tabs, switch to tab index, or close tab.\n"
    "• ask_user(text=None): ask user to take over tab for sign-in or CAPTCHA; text says why.\n"
    "Password and one-time-code fields are refused for safety: ask the user to take over."
)

_default_service: BrowserService | None = None


def default_browser_service() -> BrowserService:
    """The Core's one browser service. Chrome starts on its first call, not here."""
    global _default_service
    if _default_service is None:
        _default_service = BrowserService(default_browser_link)
    return _default_service


Action = Literal[
    "help",
    "open",
    "snapshot",
    "read",
    "find",
    "look",
    "click",
    "type",
    "select",
    "check",
    "press",
    "scroll",
    "upload",
    "wait",
    "back",
    "forward",
    "reload",
    "tab",
    "ask_user",
]
_NEEDS_REF = frozenset({"select", "check", "upload"})


def _unadvertised_default(schema: dict[str, Any]) -> None:
    """Leave a flag's `false` default out of the advertised schema (#2164).

    An absent flag is off; saying so six times cost about 27 tokens of every request, on
    a schema held to an 8K-window budget. The default still applies when the call omits it.
    """
    schema.pop("default", None)


def _off() -> Any:
    """A flag that is off unless the call turns it on, its default not advertised."""
    return Field(default=False, json_schema_extra=_unadvertised_default)


class BrowserParams(BaseModel):
    """Parameters for one browser call."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: Action
    url: str | None = None
    query: str | None = None
    # No advertised default: `help` states it, and the schema is held to the 8K budget (#2159).
    max_length: int | None = None
    ref: str | None = None
    x: int | None = Field(default=None, ge=0)
    y: int | None = Field(default=None, ge=0)
    text: str | None = None
    submit: bool = _off()
    append: bool = _off()
    double: bool = _off()
    right: bool = _off()
    hover: bool = _off()
    options: list[str] | None = None
    on: bool = True
    key: str | None = None
    direction: Literal["up", "down", "left", "right"] = "down"
    amount: float = 1.0
    paths: list[str] | None = None
    ms: int | None = Field(default=None, ge=1, le=10000)
    index: int | None = Field(default=None, ge=1)
    close: bool = _off()

    @model_validator(mode="after")
    def _arguments_for_action(self) -> BrowserParams:
        if self.action == "open" and not self.url:
            raise ValueError("open needs a url")
        if self.action == "find" and not (self.query or "").strip():
            raise ValueError("find needs a query")
        if self.action in _NEEDS_REF and not (self.ref or "").strip():
            raise ValueError(f"{self.action} needs a ref")
        if (
            self.action == "click"
            and bool((self.ref or "").strip())
            and (self.x is not None or self.y is not None)
        ):
            raise ValueError("click takes either ref or x, y, not both")
        if (
            self.action == "click"
            and not (self.ref or "").strip()
            and (self.x is None or self.y is None)
        ):
            raise ValueError("click needs a ref, or x and y, a point on the last look's picture")
        if self.action == "type" and self.text is None:
            raise ValueError("type needs text")
        if self.action == "select" and not self.options:
            raise ValueError("select needs options")
        if self.action == "press" and not (self.key or "").strip():
            raise ValueError("press needs a key")
        if self.action == "upload" and not self.paths:
            raise ValueError("upload needs paths")
        if (
            self.action == "wait"
            and sum(x is not None for x in (self.text, self.ref, self.ms)) != 1
        ):
            raise ValueError("wait needs one of text, ref or ms")
        return self


class BrowserTool(BaseTool[BrowserParams]):
    """Open, read and act on pages in a real Chrome window the person can watch."""

    name: str = "browser"
    # Pages are not workspace files: downloads land in the browser's own folder (§3.10).
    writes_files: ClassVar[bool] = False
    returns_images: ClassVar[bool] = True  # look (#2107)
    description: str = "Real Chrome browser. Action 'help'."

    def __init__(
        self,
        service: BrowserService | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self._service = service

    async def run(self, params: BrowserParams, context: ToolContext) -> dict[str, Any] | ToolResult:
        """Hand the call to the browser service for this conversation's tab.

        The call is announced to the dock's Browser tab while it runs (design §3.4), and the
        element it touched leads the result, so the conversation's step line can name it
        ("Scout clicked **Search**", §3.3) from the recorded call.
        """
        if params.action == "help":
            return {"help": HELP_TEXT}
        service = self._service or default_browser_service()
        key: TabKey = (context.room_id or context.session_id, context.agent_id)
        if params.action != "ask_user":
            await service.wait_for_control(key)
        async with service.acting(key, params.action) as step:
            result = await self._dispatch(service, key, params, context)
        element = step.get("element")
        if isinstance(element, str) and element and isinstance(result, dict):
            return {"element": element, **result}
        return result

    async def _dispatch(
        self,
        service: BrowserService,
        key: TabKey,
        params: BrowserParams,
        context: ToolContext,
    ) -> dict[str, Any] | ToolResult:
        ref = params.ref or ""
        if params.action == "click" and ref and (params.x is not None or params.y is not None):
            raise PlainRefusalError(
                "Provide either ref or x, y for click, not both.",
                reason_code="ambiguous_click",
            )
        match params.action:
            case "help":
                raise AssertionError("help action is handled in run() without reaching service")
            case "open":
                return await service.open(key, params.url or "")
            case "snapshot":
                return await service.snapshot(key)
            case "read":
                limit = READ_MAX_LENGTH if params.max_length is None else params.max_length
                return await service.read(key, limit)
            case "find":
                return await service.find(key, params.query or "")
            case "look":
                return await self._look(service, key, context)
            case "click" if not ref and params.x is not None and params.y is not None:
                return await service.click_at(
                    key,
                    params.x,
                    params.y,
                    double=params.double,  # coordinate click double
                    right=params.right,
                    hover=params.hover,
                )
            case "click":
                return await service.click(
                    key, ref, double=params.double, right=params.right, hover=params.hover
                )
            case "type":
                return await service.type_text(
                    key, params.ref, params.text or "", submit=params.submit, append=params.append
                )
            case "select":
                return await service.select(key, ref, params.options or [])
            case "check":
                return await service.check(key, ref, on=params.on)
            case "press":
                return await service.press(key, params.key or "")
            case "scroll":
                return await service.scroll(key, params.ref, params.direction, params.amount)
            case "upload":
                files = [self._workspace_file(p, context) for p in params.paths or []]
                return await service.upload(key, ref, files)
            case "wait":
                return await service.wait(key, text=params.text, ref=params.ref, ms=params.ms)
            case "back":
                return await service.back(key)
            case "forward":
                return await service.forward(key)
            case "reload":
                return await service.reload(key)
            case "tab":
                return await service.tab(key, index=params.index, close=params.close)
            case "ask_user":
                return await service.ask_user(key, message=params.text)

    async def _look(
        self, service: BrowserService, key: TabKey, context: ToolContext
    ) -> dict[str, Any] | ToolResult:
        """A picture for a model that reads images; the elements, and why, for one that does not.

        The picture travels beside the text (`ToolResult.images`), never inside it, so the
        event log and the stream carry no image bytes (#2107).
        """
        if not context.accepts_images:
            return {**await service.snapshot(key), "note": NO_IMAGES_NOTE}
        where, image = await service.look(key)
        return ToolResult(
            success=True,
            output={
                **where,
                "picture": f"Attached: {image.width} by {image.height} pixels. To press "
                "something only the picture shows, use click with x and y in its pixels.",
            },
            images=(image,),
            isolation_level=context.isolation.level,
            provenance=Provenance.primary(provider="local.builtin", model=self.name),
        )

    def _workspace_file(self, path: str, context: ToolContext) -> Path:
        """A file the clone may read, for the page's file picker."""
        if context.workspace_root is None:
            raise PlainRefusalError(
                "This conversation has no workspace, so there are no files to upload.",
                reason_code="no_workspace",
            )
        try:
            resolved, _ = self.resolve_read_path(path, context)
        except PathTraversalError as exc:
            raise PlainRefusalError(
                f"{path} is outside the files this clone can read.", reason_code="path_outside"
            ) from exc
        if not resolved.is_file():
            raise PlainRefusalError(
                f"There is no file {path} in the workspace.", reason_code="no_such_file"
            )
        return resolved
