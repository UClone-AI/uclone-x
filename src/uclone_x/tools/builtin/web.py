"""SSRF-guarded web fetch and pluggable web search tools for UClone-X runtime."""

from __future__ import annotations

import html
import ipaddress
import re
import socket
import urllib.parse
from html.parser import HTMLParser
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import SandboxViolationError
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext

# ======================================================================================
# SSRF Protection Helpers
# ======================================================================================

_EXTRA_BLOCKED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("255.255.255.255/32"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("2001:db8::/32"),
    ipaddress.ip_network("ff00::/8"),
]

_FORBIDDEN_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "0.0.0.0",
        "127.0.0.1",
        "::1",
        "[::1]",
        "169.254.169.254",
        "metadata.google.internal",
        "instance-data",
    }
)


def is_private_ip(ip: str | ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Determine if an IP address belongs to private, loopback, link-local, or reserved ranges.

    Args:
        ip: String or ipaddress object to evaluate.

    Returns:
        True if the IP is private/restricted, False if it is publicly routable.
    """
    if isinstance(ip, str):
        cleaned = ip.strip().strip("[]")
        try:
            ip_obj = ipaddress.ip_address(cleaned)
        except ValueError:
            return True
    else:
        ip_obj = ip

    if (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_reserved
        or ip_obj.is_multicast
        or ip_obj.is_unspecified
    ):
        return True

    for net in _EXTRA_BLOCKED_NETWORKS:
        if ip_obj.version == net.version and ip_obj in net:  # type: ignore[operator]
            return True

    return False


def validate_url_ssrf(url: str) -> None:
    """Validate a URL against SSRF vulnerabilities (forbidden protocols, private subnets, localhost).

    Args:
        url: The candidate URL string to validate.

    Raises:
        SandboxViolationError: If URL protocol is not http/https, hostname is missing,
            or hostname resolves to a private/loopback/restricted IP address.
    """
    if not url or not url.strip():
        raise SandboxViolationError("URL must be a non-empty string")

    parsed = urllib.parse.urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise SandboxViolationError(
            f"SSRF violation: Protocol '{scheme}' is forbidden. Only 'http' and 'https' are allowed."
        )

    hostname = parsed.hostname
    if not hostname:
        raise SandboxViolationError(f"SSRF violation: URL '{url}' does not have a valid hostname.")

    hostname_clean = hostname.lower().strip("[]")

    if hostname_clean in _FORBIDDEN_HOSTNAMES:
        raise SandboxViolationError(
            f"SSRF violation: Destination host '{hostname}' is blocked (local/metadata hostname)."
        )

    if (
        hostname_clean.endswith(".localhost")
        or hostname_clean.endswith(".local")
        or hostname_clean.endswith(".internal")
        or hostname_clean.endswith(".localdomain")
    ):
        raise SandboxViolationError(
            f"SSRF violation: Local/internal domain '{hostname}' is blocked."
        )

    try:
        ip_obj = ipaddress.ip_address(hostname_clean)
        if is_private_ip(ip_obj):
            raise SandboxViolationError(
                f"SSRF violation: Target IP '{hostname_clean}' is within a private or restricted subnet."
            )
        return
    except ValueError:
        pass

    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise SandboxViolationError(f"DNS resolution failed for hostname '{hostname}': {e}") from e

    if not addr_info:
        raise SandboxViolationError(f"No IP address records found for hostname '{hostname}'")

    for info in addr_info:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            if is_private_ip(ip_obj):
                raise SandboxViolationError(
                    f"SSRF violation: Host '{hostname}' resolved to private/restricted IP '{ip_str}'."
                )
        except ValueError as e:
            raise SandboxViolationError(
                f"SSRF violation: Host '{hostname}' resolved to unparseable IP '{ip_str}'."
            ) from e


# ======================================================================================
# HTML to Markdown Conversion
# ======================================================================================


class _HTMLToMarkdownParser(HTMLParser):
    """Converts HTML documents to clean markdown text while stripping non-content tags."""

    DISCARD_TAGS = frozenset(
        {
            "script",
            "style",
            "noscript",
            "svg",
            "canvas",
            "iframe",
            "nav",
            "footer",
            "header",
            "aside",
            "template",
            "head",
            "object",
            "embed",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.output: list[str] = []
        self._discard_depth = 0
        self._in_pre = False
        self._link_stack: list[dict[str, Any]] = []
        self._list_stack: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict: dict[str, str] = {k.lower(): (v or "") for k, v in attrs}
        tag_lower = tag.lower()

        if tag_lower in self.DISCARD_TAGS:
            self._discard_depth += 1
            return

        if self._discard_depth > 0:
            return

        if tag_lower in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag_lower[1])
            self.output.append(f"\n\n{'#' * level} ")
        elif tag_lower == "p":
            self.output.append("\n\n")
        elif tag_lower == "br":
            self.output.append("\n")
        elif tag_lower == "hr":
            self.output.append("\n\n---\n\n")
        elif tag_lower == "blockquote":
            self.output.append("\n\n> ")
        elif tag_lower in ("strong", "b"):
            self.output.append("**")
        elif tag_lower in ("em", "i"):
            self.output.append("*")
        elif tag_lower == "code":
            if not self._in_pre:
                self.output.append("`")
        elif tag_lower == "pre":
            self._in_pre = True
            self.output.append("\n\n```\n")
        elif tag_lower == "a":
            href = attr_dict.get("href", "").strip()
            self._link_stack.append({"href": href, "start_pos": len(self.output)})
        elif tag_lower == "ul":
            self._list_stack.append({"type": "ul", "index": 0})
        elif tag_lower == "ol":
            self._list_stack.append({"type": "ol", "index": 0})
        elif tag_lower == "li":
            indent = "  " * max(0, len(self._list_stack) - 1)
            if self._list_stack:
                current_list = self._list_stack[-1]
                if current_list["type"] == "ol":
                    current_list["index"] += 1
                    self.output.append(f"\n{indent}{current_list['index']}. ")
                else:
                    self.output.append(f"\n{indent}* ")
            else:
                self.output.append("\n* ")
        elif tag_lower == "img":
            alt = attr_dict.get("alt", "").strip()
            src = attr_dict.get("src", "").strip()
            if src:
                self.output.append(f"![{alt}]({src})")
        elif tag_lower == "tr":
            self.output.append("\n")
        elif tag_lower in ("td", "th"):
            self.output.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()

        if tag_lower in self.DISCARD_TAGS:
            if self._discard_depth > 0:
                self._discard_depth -= 1
            return

        if self._discard_depth > 0:
            return

        if tag_lower in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.output.append("\n\n")
        elif tag_lower == "p":
            self.output.append("\n\n")
        elif tag_lower == "blockquote":
            self.output.append("\n\n")
        elif tag_lower in ("strong", "b"):
            self.output.append("**")
        elif tag_lower in ("em", "i"):
            self.output.append("*")
        elif tag_lower == "code":
            if not self._in_pre:
                self.output.append("`")
        elif tag_lower == "pre":
            self._in_pre = False
            self.output.append("\n```\n\n")
        elif tag_lower == "a":
            if self._link_stack:
                link_info = self._link_stack.pop()
                href = link_info["href"]
                start_pos = link_info["start_pos"]
                anchor_text = "".join(self.output[start_pos:]).strip()
                del self.output[start_pos:]
                if href and href not in ("#", "javascript:void(0)", "javascript:;"):
                    if anchor_text:
                        self.output.append(f"[{anchor_text}]({href})")
                    else:
                        self.output.append(f"[{href}]({href})")
                else:
                    if anchor_text:
                        self.output.append(anchor_text)
        elif tag_lower in ("ul", "ol"):
            if self._list_stack:
                self._list_stack.pop()
            self.output.append("\n")

    def handle_data(self, data: str) -> None:
        if self._discard_depth > 0:
            return
        self.output.append(data)

    def get_markdown(self) -> str:
        raw = "".join(self.output)
        lines = [line.rstrip() for line in raw.split("\n")]
        cleaned = "\n".join(lines)
        collapsed = re.sub(r"\n{3,}", "\n\n", cleaned)
        return html.unescape(collapsed.strip())


def html_to_markdown(html_content: str) -> str:
    """Convert HTML string to clean markdown text.

    Args:
        html_content: Raw HTML input string.

    Returns:
        Structured markdown representation with non-content tags stripped.
    """
    if not html_content.strip():
        return ""
    parser = _HTMLToMarkdownParser()
    parser.feed(html_content)
    return parser.get_markdown()


# ======================================================================================
# 1. WebFetchTool
# ======================================================================================


class WebFetchParams(BaseModel):
    """Parameters for fetching and converting web content."""

    model_config = ConfigDict(extra="forbid", strict=True)

    url: str = Field(
        description="The HTTP or HTTPS URL to fetch.",
    )
    max_length: int = Field(
        default=30000,
        ge=1,
        le=500000,
        description="Maximum characters to return in markdown output (default: 30000).",
    )
    timeout_seconds: float = Field(
        default=15.0,
        ge=0.1,
        le=120.0,
        description="Request timeout in seconds (default: 15.0).",
    )


class WebFetchTool(BaseTool[WebFetchParams]):
    """Fetches web page content via HTTP/HTTPS with strict SSRF protection and markdown conversion."""

    name: str = "web_fetch"
    writes_files: ClassVar[bool] = False  # writes no file on the host (#1167)
    description: str = (
        "Fetch content from an HTTP/HTTPS URL with strict SSRF private subnet protection "
        "and clean HTML-to-markdown conversion."
    )

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self.transport = transport

    async def run(self, params: WebFetchParams, context: ToolContext) -> dict[str, Any]:
        """Fetch URL content, validate SSRF containment, and convert HTML to markdown."""
        validate_url_ssrf(params.url)

        async def _on_redirect(response: httpx.Response) -> None:
            if response.is_redirect:
                location = response.headers.get("Location")
                if location:
                    target_url = urllib.parse.urljoin(str(response.url), location)
                    validate_url_ssrf(target_url)

        headers = {
            "User-Agent": "UClone-X/0.1.0 (WebFetchTool; SSRF-Guarded; +https://github.com/UClone-AI/uclone-x)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
        }

        async with httpx.AsyncClient(
            transport=self.transport,
            timeout=httpx.Timeout(params.timeout_seconds),
            follow_redirects=True,
            event_hooks={"response": [_on_redirect]},
        ) as client:
            response = await client.get(params.url, headers=headers)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        raw_text = response.text

        if "html" in content_type.lower() or "<html" in raw_text[:500].lower():
            processed_text = html_to_markdown(raw_text)
        else:
            processed_text = raw_text

        truncated = False
        if len(processed_text) > params.max_length:
            processed_text = processed_text[: params.max_length]
            truncated = True

        return {
            "url": str(response.url),
            "status_code": response.status_code,
            "content": processed_text,
            "content_type": content_type,
            "truncated": truncated,
            "length": len(processed_text),
        }


# ======================================================================================
# 2. WebSearchTool & Search Providers
# ======================================================================================


@runtime_checkable
class SearchProviderProtocol(Protocol):
    """Protocol for pluggable web search backend providers."""

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """Execute a web search and return structured results.

        Each result dictionary must contain:
        - 'title': Title of the web page or result
        - 'url': Destination web URL
        - 'snippet': Descriptive excerpt or text snippet
        """
        ...


def _clean_ddg_url(raw_url: str) -> str:
    """Extract destination URL from DuckDuckGo redirect wrapper."""
    if not raw_url:
        return ""
    if "uddg=" in raw_url:
        parsed = urllib.parse.urlsplit(raw_url)
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            return urllib.parse.unquote(qs["uddg"][0])
    if raw_url.startswith("//"):
        return "https:" + raw_url
    return raw_url


class _DuckDuckGoHTMLParser(HTMLParser):
    """HTML parser to extract structured search results from DuckDuckGo HTML / Lite responses."""

    def __init__(self, max_results: int = 5) -> None:
        super().__init__(convert_charrefs=True)
        self.max_results = max_results
        self.results: list[dict[str, str]] = []

        self._in_title = False
        self._snippet_depth = 0
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []
        self._current_url = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = {k.lower(): (v or "") for k, v in attrs}
        tag_lower = tag.lower()
        css_class = attr_dict.get("class", "").lower()

        # Check for title/link
        if tag_lower == "a" and (
            "result__a" in css_class or "result-link" in css_class or "result__url" in css_class
        ):
            if self._current_title and self._current_url:
                self._commit_result()
            self._in_title = True
            self._current_title = []
            href = attr_dict.get("href", "")
            self._current_url = _clean_ddg_url(href)
        elif (
            "result__snippet" in css_class
            or "result-snippet" in css_class
            or "result__snippet" in attr_dict.get("id", "").lower()
            or (tag_lower in ("td", "div", "span", "p", "a") and "snippet" in css_class)
        ):
            self._snippet_depth += 1
        elif self._snippet_depth > 0:
            # Nested tag inside snippet (e.g. <b>, <span>, etc.)
            self._snippet_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower == "a" and self._in_title:
            self._in_title = False
        elif self._snippet_depth > 0:
            self._snippet_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current_title.append(data)
        elif self._snippet_depth > 0:
            self._current_snippet.append(data)

    def _commit_result(self) -> None:
        if len(self.results) >= self.max_results:
            return
        title = "".join(self._current_title).strip()
        snippet = " ".join("".join(self._current_snippet).split()).strip()
        url = self._current_url.strip()
        if title and url:
            self.results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )
        self._current_title = []
        self._current_snippet = []
        self._current_url = ""

    def close(self) -> None:
        super().close()
        if self._current_title and self._current_url:
            self._commit_result()


class DuckDuckGoSearchProvider:
    """Default zero-key search backend leveraging DuckDuckGo HTML and Instant Answer search."""

    def __init__(
        self,
        timeout_seconds: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """Search DuckDuckGo and parse returned HTML / JSON results."""
        clean_query = query.strip()
        if not clean_query:
            return []

        search_url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        async with httpx.AsyncClient(
            transport=self.transport,
            timeout=httpx.Timeout(self.timeout_seconds),
            follow_redirects=True,
        ) as client:
            # 1. Attempt POST to html.duckduckgo.com/html/
            try:
                response = await client.post(
                    search_url,
                    data={"q": clean_query, "b": ""},
                    headers=headers,
                )
                if response.status_code == 200:
                    results = self._parse_html(response.text, max_results)
                    if results:
                        return results
            except Exception:
                pass

            # 2. Attempt GET to html.duckduckgo.com/html/
            try:
                response = await client.get(
                    search_url,
                    params={"q": clean_query},
                    headers=headers,
                )
                if response.status_code == 200:
                    results = self._parse_html(response.text, max_results)
                    if results:
                        return results
            except Exception:
                pass

            # 3. Attempt POST to lite.duckduckgo.com/lite/
            try:
                lite_url = "https://lite.duckduckgo.com/lite/"
                response = await client.post(
                    lite_url,
                    data={"q": clean_query},
                    headers=headers,
                )
                if response.status_code == 200:
                    results = self._parse_html(response.text, max_results)
                    if results:
                        return results
            except Exception:
                pass

            # 4. Fallback to DuckDuckGo Instant Answer JSON API
            try:
                ia_url = "https://api.duckduckgo.com/"
                response = await client.get(
                    ia_url,
                    params={"q": clean_query, "format": "json", "no_html": "1"},
                    headers=headers,
                )
                if response.status_code == 200:
                    results = self._parse_instant_answer_json(response.json(), max_results)
                    if results:
                        return results
            except Exception:
                pass

        return []

    def _parse_html(self, html_text: str, max_results: int) -> list[dict[str, str]]:
        """Parse DuckDuckGo HTML results page."""
        parser = _DuckDuckGoHTMLParser(max_results=max_results)
        parser.feed(html_text)
        parser.close()
        return parser.results

    def _parse_instant_answer_json(
        self, data: dict[str, Any], max_results: int
    ) -> list[dict[str, str]]:
        """Extract results from DuckDuckGo Instant Answer JSON payload."""
        results: list[dict[str, str]] = []
        abstract = str(data.get("AbstractText", "") or "")
        abstract_url = str(data.get("AbstractURL", "") or "")
        heading = str(data.get("Heading", "") or "")
        if abstract and abstract_url:
            results.append(
                {
                    "title": heading or abstract[:60],
                    "url": abstract_url,
                    "snippet": abstract,
                }
            )

        related = data.get("RelatedTopics", [])
        if isinstance(related, list):
            for item in cast(list[Any], related):
                if len(results) >= max_results:
                    break
                if isinstance(item, dict):
                    item_dict = cast(dict[str, Any], item)
                    if "Text" in item_dict and "FirstURL" in item_dict:
                        text = str(item_dict.get("Text", ""))
                        url = str(item_dict.get("FirstURL", ""))
                        title = text.split(" - ")[0] if " - " in text else text[:60]
                        results.append(
                            {
                                "title": title,
                                "url": url,
                                "snippet": text,
                            }
                        )

        return results[:max_results]


class WebSearchParams(BaseModel):
    """Parameters for executing a web search."""

    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(
        description=(
            "Search query string. For hardware or technical inquiries, prioritize technical "
            "specifications, standards, and pinouts rather than raw commercial shopping listings."
        ),
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=25,
        description="Maximum number of search results to return (default: 5).",
    )


class WebSearchTool(BaseTool[WebSearchParams]):
    """Performs web search using a pluggable search provider (defaults to DuckDuckGo)."""

    name: str = "web_search"
    writes_files: ClassVar[bool] = False  # writes no file on the host (#1167)
    description: str = (
        "Search the web for queries and return structured results containing title, URL, and snippet. "
        "Prioritizes technical specifications, standards, and pinouts over commercial shopping listings."
    )

    def __init__(
        self,
        provider: SearchProviderProtocol | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self.provider: SearchProviderProtocol = (
            provider if provider is not None else DuckDuckGoSearchProvider()
        )

    async def run(self, params: WebSearchParams, context: ToolContext) -> list[dict[str, str]]:
        """Execute web search query via configured provider strategy."""
        return await self.provider.search(query=params.query, max_results=params.max_results)


__all__ = [
    "DuckDuckGoSearchProvider",
    "SearchProviderProtocol",
    "WebFetchParams",
    "WebFetchTool",
    "WebSearchParams",
    "WebSearchTool",
    "html_to_markdown",
    "is_private_ip",
    "validate_url_ssrf",
]
