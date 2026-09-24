"""Lightweight async HTTP client for the ComfyUI REST API.

Connects to ComfyUI via HTTP endpoints (/system_stats, /object_info, /prompt,
/history, /view) with structured error handling, node_errors unpacking, and
cancellation/timeout awareness.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Self, cast

import httpx

from uclone_x.errors import ComfyUIError, ComfyUIExecutionError

DEFAULT_COMFYUI_BASE_URL: str = "http://127.0.0.1:8188"
#: Overrides the checkpoint asked of the daemon — for the agent tool and the engine alike.
COMFY_CHECKPOINT_ENV: str = "UCX_COMFYUI_CHECKPOINT"
#: What the ComfyUI daemon is asked to load. It names a file in *that* daemon's model
#: directory, not a path in this filesystem.
#:
#: It is declared here rather than in `image.py` because both callers must read one value and
#: only this module sits below both: `image.py` already imports `build_txt2img_workflow` from
#: `comfy_image_tool`, so importing the constant back the other way would be a cycle (#1223).
COMFY_DEFAULT_CHECKPOINT: str = "anillustrious_v4.safetensors"


def default_comfy_checkpoint() -> str:
    """The checkpoint to ask the daemon for when the caller named none.

    Read from the environment **per call**, never captured at import. `ComfyUIImageEngine`
    reads it when it is constructed, and a pydantic field default or a function parameter
    default would instead freeze whatever `UCX_COMFYUI_CHECKPOINT` held when this module was
    first imported — so a process that sets the variable after import, and every test that
    sets it with `monkeypatch.setenv`, would get the wrong answer from one path and the right
    one from the other. That divergence is the defect this function exists to close (#1223),
    so its callers must call it rather than evaluate it into a default.
    """
    return os.getenv(COMFY_CHECKPOINT_ENV) or COMFY_DEFAULT_CHECKPOINT


class ComfyTimeoutError(ComfyUIError, TimeoutError):
    """Raised when a ComfyUI prompt polling operation exceeds timeout."""


def format_node_errors(payload: dict[str, Any]) -> str:
    """Unpack ComfyUI error and node_errors payloads into a human-readable string."""
    parts: list[str] = []

    err = payload.get("error")
    if isinstance(err, dict):
        err_dict = cast(dict[str, Any], err)
        msg_val = err_dict.get("message")
        if msg_val is not None:
            parts.append(str(msg_val))
        details_val = err_dict.get("details")
        if details_val is not None:
            parts.append(str(details_val))
    elif isinstance(err, str) and err:
        parts.append(err)

    node_errors = payload.get("node_errors")
    if isinstance(node_errors, dict):
        node_errors_dict = cast(dict[str, Any], node_errors)
        for node_id, node_err in node_errors_dict.items():
            if isinstance(node_err, dict):
                node_err_dict = cast(dict[str, Any], node_err)
                class_type = str(node_err_dict.get("class_type", "Unknown"))
                errors_val = node_err_dict.get("errors", [])
                if isinstance(errors_val, list):
                    for item in cast(list[object], errors_val):
                        if isinstance(item, dict):
                            item_dict = cast(dict[str, Any], item)
                            item_msg = str(item_dict.get("message", ""))
                            item_det = str(item_dict.get("details", ""))
                            combined = (
                                f"{item_msg}: {item_det}".strip(": ") if item_det else item_msg
                            )
                            parts.append(f"Node {node_id} ({class_type}): {combined}")
                        else:
                            parts.append(f"Node {node_id} ({class_type}): {item}")
            else:
                parts.append(f"Node {node_id}: {node_err}")

    return " | ".join(parts) if parts else "ComfyUI reported an error with no details"


def format_exec_error(status: dict[str, Any]) -> str:
    """Extract execution errors from a ComfyUI history entry's status blob."""
    parts: list[str] = []
    messages = status.get("messages", [])
    if isinstance(messages, list):
        for msg in cast(list[object], messages):
            if not isinstance(msg, (list, tuple)):
                continue
            msg_seq = cast(list[object] | tuple[object, ...], msg)
            if len(msg_seq) < 2:
                continue
            kind, payload = msg_seq[0], msg_seq[1]
            if kind == "execution_error" and isinstance(payload, dict):
                payload_dict = cast(dict[str, Any], payload)
                node_id_raw = payload_dict.get("node_id", "?")
                node_id = str(node_id_raw)
                node_type_raw = payload_dict.get("node_type", "")
                node_type = str(node_type_raw)
                node_label = f"Node {node_id} ({node_type})" if node_type else f"Node {node_id}"
                exc_type = str(payload_dict.get("exception_type", "")).strip()
                exc_msg = str(payload_dict.get("exception_message", "")).strip()
                details = f"{exc_type}: {exc_msg}".strip(": ")
                parts.append(f"{node_label}: {details}" if details else node_label)

    return " | ".join(parts) if parts else "ComfyUI execution failed with an unspecified error"


class ComfyClient:
    """Lightweight async HTTP client for interacting with a ComfyUI server."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        raw_url = (
            base_url
            if base_url is not None
            else os.environ.get("COMFYUI_BASE_URL", DEFAULT_COMFYUI_BASE_URL)
        )
        self.base_url: str = raw_url.rstrip("/")
        self.timeout: float = timeout
        self.client_id: str = str(uuid.uuid4())
        self._client: httpx.AsyncClient | None = client
        self._owns_client: bool = client is None

    def _url(self, path: str) -> str:
        clean_path = path if path.startswith("/") else f"/{path}"
        return f"{self.base_url}{clean_path}"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client session if owned by this instance."""
        if self._owns_client and self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.aclose()

    async def alive(self, timeout: float = 3.0) -> bool:
        """Check if the ComfyUI server is reachable and responsive."""
        try:
            client = self._get_client()
            resp = await client.get(self._url("/system_stats"), timeout=timeout)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def system_stats(self) -> dict[str, Any]:
        """Fetch system statistics including hardware devices and VRAM usage."""
        client = self._get_client()
        try:
            resp = await client.get(self._url("/system_stats"))
            resp.raise_for_status()
            data: object = resp.json()
            if isinstance(data, dict):
                return cast(dict[str, Any], data)
            raise ComfyUIError(f"Unexpected non-dict system_stats response: {type(data)}")
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Failed to fetch ComfyUI system stats: {exc}") from exc

    async def object_info(self) -> dict[str, Any]:
        """Fetch available node definitions and schema metadata."""
        client = self._get_client()
        try:
            resp = await client.get(self._url("/object_info"))
            resp.raise_for_status()
            data: object = resp.json()
            if isinstance(data, dict):
                return cast(dict[str, Any], data)
            raise ComfyUIError(f"Unexpected non-dict object_info response: {type(data)}")
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Failed to fetch ComfyUI object info: {exc}") from exc

    async def missing_nodes(self, required: list[str]) -> list[str]:
        """Check for the presence of required nodes and return missing node names.

        Raises:
            ComfyUIError: If the server is unreachable or fails to return object info.
        """
        info = await self.object_info()
        available = set(info.keys())
        return [node for node in required if node not in available]

    async def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """Submit a prompt workflow JSON to ComfyUI POST /prompt and return prompt_id.

        Raises ComfyUIError with unpacked node_errors if validation fails.
        """
        client = self._get_client()
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        try:
            resp = await client.post(self._url("/prompt"), json=payload)
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Failed to connect to ComfyUI at {self.base_url}: {exc}") from exc

        if resp.status_code != 200:
            detail: str = ""
            node_errors_data: dict[str, object] | None = None
            try:
                body_json: object = resp.json()
                if isinstance(body_json, dict):
                    str_dict = cast(dict[str, Any], body_json)
                    detail = format_node_errors(str_dict)
                    raw_errors = str_dict.get("node_errors")
                    if isinstance(raw_errors, dict):
                        node_errors_data = cast(dict[str, object], raw_errors)
            except Exception:
                detail = resp.text[:1000]

            error_msg = f"ComfyUI rejected prompt (HTTP {resp.status_code}): {detail}"
            raise ComfyUIError(
                error_msg,
                status_code=resp.status_code,
                node_errors=node_errors_data,
            )

        try:
            res_obj: object = resp.json()
        except Exception as exc:
            raise ComfyUIError(f"Failed to parse ComfyUI /prompt response JSON: {exc}") from exc

        if not isinstance(res_obj, dict):
            raise ComfyUIError(f"ComfyUI returned response without prompt_id: {res_obj}")

        res = cast(dict[str, Any], res_obj)
        if "prompt_id" not in res:
            raise ComfyUIError(f"ComfyUI returned response without prompt_id: {res}")

        return str(res["prompt_id"])

    async def get_history(self, prompt_id: str) -> dict[str, Any]:
        """Fetch execution history for a given prompt_id from GET /history/{prompt_id}."""
        client = self._get_client()
        try:
            resp = await client.get(self._url(f"/history/{prompt_id}"))
            resp.raise_for_status()
            try:
                data: object = resp.json()
            except Exception as exc:
                raise ComfyUIError(
                    f"Failed to parse ComfyUI /history response JSON: {exc}"
                ) from exc
            if not isinstance(data, dict):
                raise ComfyUIError(f"Unexpected non-dict /history response: {type(data)}")
            data_dict = cast(dict[str, Any], data)
            if prompt_id not in data_dict or data_dict[prompt_id] is None:
                return {}
            entry = data_dict[prompt_id]
            if not isinstance(entry, dict):
                raise ComfyUIError(
                    f"Unexpected non-dict history entry for {prompt_id}: {type(entry)}"
                )
            return cast(dict[str, Any], entry)
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Failed to get history for prompt {prompt_id}: {exc}") from exc

    @staticmethod
    def _extract_image_filenames(outputs: dict[str, Any]) -> list[str]:
        """Extract output image filenames from history outputs."""
        filenames: list[str] = []
        for node_output in outputs.values():
            if isinstance(node_output, dict):
                node_dict = cast(dict[str, Any], node_output)
                raw_imgs = node_dict.get("images")
                if isinstance(raw_imgs, list):
                    for img in cast(list[object], raw_imgs):
                        if isinstance(img, dict):
                            img_dict = cast(dict[str, Any], img)
                            fn_val = img_dict.get("filename")
                            if fn_val is not None:
                                fn = str(fn_val)
                                sub_val = img_dict.get("subfolder", "")
                                sub = str(sub_val).strip() if sub_val is not None else ""
                                filenames.append(f"{sub}/{fn}" if sub else fn)
        return filenames

    async def wait_for_output(
        self,
        prompt_id: str,
        timeout_seconds: float = 60.0,
        poll_interval: float = 0.5,
    ) -> list[str]:
        """Poll ComfyUI history until completion, returning generated image filenames."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds

        while loop.time() < deadline:
            history_entry = await self.get_history(prompt_id)
            if history_entry:
                raw_status = history_entry.get("status")
                if isinstance(raw_status, dict):
                    status_dict = cast(dict[str, Any], raw_status)
                    status_str = status_dict.get("status_str")
                    if status_str == "error":
                        raise ComfyUIExecutionError(format_exec_error(status_dict))

                    raw_outputs = history_entry.get("outputs")
                    outputs: dict[str, Any] = (
                        cast(dict[str, Any], raw_outputs) if isinstance(raw_outputs, dict) else {}
                    )
                    images = self._extract_image_filenames(outputs)
                    if images:
                        return images

                    if status_dict.get("completed") is True or status_str == "success":
                        raise ComfyUIExecutionError(
                            f"ComfyUI job {prompt_id} completed but produced no image outputs; "
                            f"the workflow has no SaveImage-style terminal node"
                        )

            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            sleep_time = min(poll_interval, remaining)
            await asyncio.sleep(sleep_time)

        raise ComfyTimeoutError(
            f"ComfyUI prompt {prompt_id} timed out after {timeout_seconds:.1f}s"
        )

    async def download_image(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> bytes:
        """Download raw image bytes from ComfyUI GET /view."""
        client = self._get_client()
        actual_subfolder = subfolder
        actual_filename = filename
        if not actual_subfolder and "/" in actual_filename:
            parts = actual_filename.rsplit("/", 1)
            actual_subfolder = parts[0]
            actual_filename = parts[1]

        params = {
            "filename": actual_filename,
            "subfolder": actual_subfolder,
            "type": folder_type,
        }
        try:
            resp = await client.get(self._url("/view"), params=params)
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Failed to download image '{filename}': {exc}") from exc
