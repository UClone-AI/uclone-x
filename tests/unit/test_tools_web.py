"""Comprehensive unit tests for SSRF-guarded WebFetchTool and WebSearchTool suite."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest

from uclone_x.errors import SandboxViolationError
from uclone_x.sandbox.models import WorkspaceIsolation
from uclone_x.tools import (
    DuckDuckGoSearchProvider,
    SearchProviderProtocol,
    ToolContext,
    WebFetchParams,
    WebFetchTool,
    WebSearchParams,
    WebSearchTool,
    create_default_registry,
    html_to_markdown,
    is_private_ip,
    validate_url_ssrf,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Provide an isolated workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def tool_context(workspace: Path) -> ToolContext:
    """Provide a standard ToolContext."""
    return ToolContext(
        agent_id="test_agent",
        session_id="test_session",
        workspace_root=workspace,
        isolation=WorkspaceIsolation(),
    )


# ======================================================================================
# 1. SSRF Guard Tests (is_private_ip & validate_url_ssrf)
# ======================================================================================


@pytest.mark.parametrize(
    ("ip_str", "expected_private"),
    [
        # Loopback IPv4
        ("127.0.0.1", True),
        ("127.1.2.3", True),
        # Private Class A/B/C
        ("10.0.0.1", True),
        ("10.254.0.1", True),
        ("172.16.0.1", True),
        ("172.31.255.255", True),
        ("192.168.1.1", True),
        ("192.168.0.254", True),
        # Link-local / Cloud Metadata
        ("169.254.169.254", True),
        ("169.254.1.1", True),
        # Zero / Reserved / CGNAT
        ("0.0.0.0", True),
        ("100.64.0.1", True),
        ("192.0.2.1", True),
        ("198.51.100.1", True),
        ("203.0.113.1", True),
        ("224.0.0.1", True),
        ("240.0.0.1", True),
        ("255.255.255.255", True),
        # IPv6 Loopback / Private / Link-local
        ("::1", True),
        ("::", True),
        ("fc00::1", True),
        ("fd00::1", True),
        ("fe80::1", True),
        ("ff02::1", True),
        ("2001:db8::1", True),
        # Public IPv4 & IPv6
        ("8.8.8.8", False),
        ("1.1.1.1", False),
        ("93.184.216.34", False),
        ("2606:4700:4700::1111", False),
        ("2001:4860:4860::8888", False),
    ],
)
def test_is_private_ip(ip_str: str, expected_private: bool) -> None:
    """Verify is_private_ip accurately identifies private and public IP ranges."""
    assert is_private_ip(ip_str) is expected_private


@pytest.mark.parametrize(
    "invalid_url",
    [
        # Forbidden schemes
        "file:///etc/passwd",
        "file:///C:/Windows/system32/cmd.exe",
        "ftp://ftp.example.com/file.txt",
        "gopher://gopher.floodgap.com",
        "ws://echo.websocket.org",
        "javascript:alert(1)",
        "data:text/html,<b>evil</b>",
        # Localhost / Loopback / Cloud metadata
        "http://127.0.0.1:8080/api",
        "http://127.0.0.1",
        "http://localhost:3000",
        "http://localhost/admin",
        "http://foo.localhost/test",
        "http://service.local/status",
        "http://db.internal/config",
        "http://10.0.0.1/secrets",
        "http://192.168.1.1/admin",
        "http://172.16.0.5/internal",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]:8080",
        "http://[fc00::1]/status",
        "http://0.0.0.0:8000",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://instance-data/latest/meta-data/",
    ],
)
def test_validate_url_ssrf_rejects_unsafe_targets(invalid_url: str) -> None:
    """Verify validate_url_ssrf raises SandboxViolationError for forbidden protocols and private hosts."""
    with pytest.raises(SandboxViolationError):
        validate_url_ssrf(invalid_url)


def test_validate_url_ssrf_empty_or_malformed() -> None:
    """Verify validate_url_ssrf rejects empty strings and URLs without hostname."""
    with pytest.raises(SandboxViolationError, match="URL must be a non-empty string"):
        validate_url_ssrf("")

    with pytest.raises(SandboxViolationError, match="does not have a valid hostname"):
        validate_url_ssrf("http://")


def test_validate_url_ssrf_dns_resolution_private_rejection() -> None:
    """Verify DNS resolution resolving to a private IP is blocked."""
    mock_addr_info = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 80)),
    ]
    with patch("socket.getaddrinfo", return_value=mock_addr_info):
        with pytest.raises(SandboxViolationError, match="resolved to private/restricted IP"):
            validate_url_ssrf("http://private-internal-company-domain.com/data")


def test_validate_url_ssrf_dns_resolution_failure() -> None:
    """Verify DNS resolution failures raise SandboxViolationError."""
    with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name or service not known")):
        with pytest.raises(SandboxViolationError, match="DNS resolution failed"):
            validate_url_ssrf("http://nonexistent-domain-xyz-12345.com")


def test_validate_url_ssrf_dns_resolution_no_records() -> None:
    """Verify empty DNS resolution records raise SandboxViolationError."""
    with patch("socket.getaddrinfo", return_value=[]):
        with pytest.raises(SandboxViolationError, match="No IP address records found"):
            validate_url_ssrf("http://empty-records-domain.com")


def test_validate_url_ssrf_allows_public_targets() -> None:
    """Verify validate_url_ssrf allows legitimate public HTTP and HTTPS endpoints."""
    mock_addr_info = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
    ]
    with patch("socket.getaddrinfo", return_value=mock_addr_info):
        # Should not raise
        validate_url_ssrf("https://example.com/test")
        validate_url_ssrf("http://example.org/page?query=1")


# ======================================================================================
# 2. HTML to Markdown Converter Tests
# ======================================================================================


def test_html_to_markdown_empty_or_blank() -> None:
    """Verify conversion of empty or whitespace strings."""
    assert html_to_markdown("") == ""
    assert html_to_markdown("   \n\t  ") == ""


def test_html_to_markdown_basic_formatting() -> None:
    """Verify headings, paragraphs, bold, italic, and line breaks conversion."""
    html_input = """
    <html>
    <body>
        <h1>Main Title</h1>
        <h2>Subtitle</h2>
        <p>This is a paragraph with <strong>bold</strong> and <em>italic</em> text.</p>
        <p>Another paragraph with<br>a line break.</p>
        <hr>
        <blockquote>Quoted wisdom here</blockquote>
    </body>
    </html>
    """
    md = html_to_markdown(html_input)
    assert "# Main Title" in md
    assert "## Subtitle" in md
    assert "**bold**" in md
    assert "*italic*" in md
    assert "Another paragraph with\na line break." in md
    assert "---" in md
    assert "> Quoted wisdom here" in md


def test_html_to_markdown_strips_scripts_styles_and_nav_footer_header() -> None:
    """Verify non-content tags (script, style, noscript, nav, header, footer, aside, svg) are stripped."""
    html_input = """
    <html>
    <head>
        <title>Document</title>
        <style>body { background: #fff; }</style>
        <script>console.log("secret tracker");</script>
    </head>
    <body>
        <header>
            <nav><a href="/menu">Menu Item 1</a></nav>
        </header>
        <main>
            <h1>Article Title</h1>
            <p>Important article text.</p>
        </main>
        <aside>Related ads and sidebar</aside>
        <footer>
            <p>Copyright 2026</p>
        </footer>
        <svg><circle cx="50" cy="50" r="40" /></svg>
        <noscript>Enable JS</noscript>
    </body>
    </html>
    """
    md = html_to_markdown(html_input)
    assert "secret tracker" not in md
    assert "background: #fff" not in md
    assert "Menu Item 1" not in md
    assert "Related ads" not in md
    assert "Copyright 2026" not in md
    assert "Enable JS" not in md
    assert "# Article Title" in md
    assert "Important article text." in md


def test_html_to_markdown_links_images_lists_code() -> None:
    """Verify links, images, lists (ul/ol), and code blocks."""
    html_input = """
    <div>
        <p>Check <a href="https://example.com/docs">the documentation</a>.</p>
        <img src="https://example.com/logo.png" alt="Company Logo">
        <ul>
            <li>Item Alpha</li>
            <li>Item Beta</li>
        </ul>
        <ol>
            <li>Step One</li>
            <li>Step Two</li>
        </ol>
        <p>Inline <code>variable_name</code> snippet.</p>
        <pre><code>def greet():\n    return "hello"</code></pre>
    </div>
    """
    md = html_to_markdown(html_input)
    assert "[the documentation](https://example.com/docs)" in md
    assert "![Company Logo](https://example.com/logo.png)" in md
    assert "* Item Alpha" in md
    assert "* Item Beta" in md
    assert "1. Step One" in md
    assert "2. Step Two" in md
    assert "`variable_name`" in md
    assert '```\ndef greet():\n    return "hello"\n```' in md


def test_html_to_markdown_entity_decoding() -> None:
    """Verify HTML entities are properly unescaped."""
    html_input = "<p>&copy; 2026 UClone-X &amp; Co. &lt;special&gt; &quot;quotes&quot; &#39;apostrophe&#39;</p>"
    md = html_to_markdown(html_input)
    assert "© 2026 UClone-X & Co. <special> \"quotes\" 'apostrophe'" in md


# ======================================================================================
# 3. WebFetchTool Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_web_fetch_tool_success_html(tool_context: ToolContext) -> None:
    """Verify WebFetchTool fetches public HTML content and converts to markdown."""
    tool = WebFetchTool()
    sample_html = """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <h1>UClone-X Framework</h1>
        <p>Next-gen agent runtime.</p>
        <a href="https://example.com/more">Learn more</a>
    </body>
    </html>
    """

    mock_response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        text=sample_html,
        request=httpx.Request("GET", "https://example.com/page"),
    )

    with patch(
        "socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))],
    ):
        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            result = await tool.execute(
                params={"url": "https://example.com/page"},
                context=tool_context,
            )

    assert result.success is True
    assert result.output is not None
    output = cast(dict[str, Any], result.output)
    assert output["status_code"] == 200
    assert output["url"] == "https://example.com/page"
    assert "# UClone-X Framework" in str(output["content"])
    assert "Next-gen agent runtime." in str(output["content"])
    assert "[Learn more](https://example.com/more)" in str(output["content"])
    assert output["truncated"] is False


@pytest.mark.asyncio
async def test_web_fetch_tool_plain_text_and_json(tool_context: ToolContext) -> None:
    """Verify WebFetchTool returns raw text for non-HTML content types."""
    tool = WebFetchTool()
    raw_json = '{"status": "ok", "count": 42}'

    mock_response = httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        text=raw_json,
        request=httpx.Request("GET", "https://api.example.com/status"),
    )

    with patch(
        "socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))],
    ):
        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            result = await tool.execute(
                params={"url": "https://api.example.com/status"},
                context=tool_context,
            )

    assert result.success is True
    output = cast(dict[str, Any], result.output)
    assert output["content"] == raw_json


@pytest.mark.asyncio
async def test_web_fetch_tool_max_length_truncation(tool_context: ToolContext) -> None:
    """Verify content exceeding max_length is truncated."""
    tool = WebFetchTool()
    long_html = "<html><body>" + "<p>A" * 500 + "</p></body></html>"

    mock_response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/html"},
        text=long_html,
        request=httpx.Request("GET", "https://example.com/long"),
    )

    with patch(
        "socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))],
    ):
        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            result = await tool.execute(
                params={"url": "https://example.com/long", "max_length": 50},
                context=tool_context,
            )

    assert result.success is True
    output = cast(dict[str, Any], result.output)
    assert output["truncated"] is True
    assert len(str(output["content"])) == 50


@pytest.mark.asyncio
async def test_web_fetch_tool_ssrf_rejection_in_execute(tool_context: ToolContext) -> None:
    """Verify SSRF attempts fail fast and return structured failure ToolResult."""
    tool = WebFetchTool()
    result = await tool.execute(
        params={"url": "http://127.0.0.1:8080/admin"},
        context=tool_context,
    )
    assert result.success is False
    assert "SSRF violation" in (result.error or "")

    file_result = await tool.execute(
        params={"url": "file:///etc/passwd"},
        context=tool_context,
    )
    assert file_result.success is False
    assert "SSRF violation" in (file_result.error or "")


@pytest.mark.asyncio
async def test_web_fetch_tool_http_error_statuses(tool_context: ToolContext) -> None:
    """Verify HTTP 404, 500, and timeout errors are cleanly caught and returned as errors."""
    tool = WebFetchTool()

    # 404 Not Found
    req = httpx.Request("GET", "https://example.com/missing")
    resp_404 = httpx.Response(status_code=404, request=req)
    with patch(
        "socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))],
    ):
        with patch.object(httpx.AsyncClient, "get", return_value=resp_404):
            res_404 = await tool.execute(
                params={"url": "https://example.com/missing"},
                context=tool_context,
            )
    assert res_404.success is False
    assert "404" in (res_404.error or "")

    # 500 Internal Server Error
    resp_500 = httpx.Response(status_code=500, request=req)
    with patch(
        "socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))],
    ):
        with patch.object(httpx.AsyncClient, "get", return_value=resp_500):
            res_500 = await tool.execute(
                params={"url": "https://example.com/error"},
                context=tool_context,
            )
    assert res_500.success is False
    assert "500" in (res_500.error or "")

    # Timeout Exception
    with patch(
        "socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))],
    ):
        with patch.object(
            httpx.AsyncClient, "get", side_effect=httpx.TimeoutException("Connection timed out")
        ):
            res_timeout = await tool.execute(
                params={"url": "https://example.com/slow"},
                context=tool_context,
            )
    assert res_timeout.success is False
    assert "Timeout" in (res_timeout.error or "")


@pytest.mark.asyncio
async def test_web_fetch_tool_redirect_ssrf_guard(tool_context: ToolContext) -> None:
    """Verify redirects to private IP addresses trigger SSRF violation."""

    def redirect_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=302,
            headers={"Location": "http://127.0.0.1:9000/internal"},
            request=request,
        )

    transport = httpx.MockTransport(redirect_handler)
    tool = WebFetchTool(transport=transport)

    with patch(
        "socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))],
    ):
        res = await tool.execute(
            params={"url": "https://example.com/redirect"},
            context=tool_context,
        )

    assert res.success is False
    assert "SSRF violation" in (res.error or "")


def test_web_fetch_tool_schema_and_parameters() -> None:
    """Verify WebFetchTool parameter model validation."""
    tool = WebFetchTool()
    schema = tool.parameters_schema
    assert schema["type"] == "object"
    assert "url" in schema["properties"]
    assert "max_length" in schema["properties"]
    assert "timeout_seconds" in schema["properties"]

    valid = WebFetchParams.model_validate({"url": "https://example.com"})
    assert valid.url == "https://example.com"
    assert valid.max_length == 30000
    assert valid.timeout_seconds == 15.0


# ======================================================================================
# 4. WebSearchTool & Search Provider Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_duckduckgo_provider_html_parsing() -> None:
    """Verify DuckDuckGoSearchProvider parses DuckDuckGo HTML results page."""
    provider = DuckDuckGoSearchProvider()

    sample_ddg_html = """
    <div class="result results_links results_links_deep web-result ">
      <div class="links_main links_deep result__body">
        <h2 class="result__title">
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpython.org&rut=1">Python Programming Language</a>
        </h2>
        <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpython.org&rut=1">Official home of Python programming language.</a>
      </div>
    </div>
    <div class="result results_links results_links_deep web-result ">
      <div class="links_main links_deep result__body">
        <h2 class="result__title">
          <a class="result__a" href="https://docs.python.org">Python Documentation</a>
        </h2>
        <a class="result__snippet" href="https://docs.python.org">Comprehensive Python standard library reference.</a>
      </div>
    </div>
    """

    mock_resp = httpx.Response(
        status_code=200,
        text=sample_ddg_html,
        request=httpx.Request("POST", "https://html.duckduckgo.com/html/"),
    )

    with patch.object(httpx.AsyncClient, "post", return_value=mock_resp):
        results = await provider.search("python programming", max_results=5)

    assert len(results) == 2
    assert results[0]["title"] == "Python Programming Language"
    assert results[0]["url"] == "https://python.org"
    assert results[0]["snippet"] == "Official home of Python programming language."

    assert results[1]["title"] == "Python Documentation"
    assert results[1]["url"] == "https://docs.python.org"
    assert results[1]["snippet"] == "Comprehensive Python standard library reference."


@pytest.mark.asyncio
async def test_duckduckgo_provider_lite_and_nested_snippet_parsing() -> None:
    """Verify DuckDuckGoSearchProvider parses Lite markup with nested tags and td.result-snippet."""
    provider = DuckDuckGoSearchProvider()

    sample_lite_html = """
    <table>
      <tr>
        <td valign="top">1.&nbsp;</td>
        <td>
          <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fnews.example.com%2Fai&rut=1" class="result-link">
            <b>AI</b> Breakthrough in 2026
          </a>
        </td>
      </tr>
      <tr>
        <td>&nbsp;</td>
        <td class="result-snippet">
          Researchers announce a <b>major</b> milestone in <span>autonomous agents</span>. Full report here.
        </td>
      </tr>
      <tr>
        <td valign="top">2.&nbsp;</td>
        <td>
          <a rel="nofollow" href="https://example.org/robotics" class="result-link">
            Robotics Update
          </a>
        </td>
      </tr>
      <tr>
        <td>&nbsp;</td>
        <td class="result-snippet">
          New humanoid models demonstrate real-time reasoning capabilities.
        </td>
      </tr>
    </table>
    """

    mock_resp = httpx.Response(
        status_code=200,
        text=sample_lite_html,
        request=httpx.Request("POST", "https://lite.duckduckgo.com/lite/"),
    )

    # Simulate html.duckduckgo.com failing (403), falling back to lite.duckduckgo.com
    mock_fail = httpx.Response(
        status_code=403,
        request=httpx.Request("POST", "https://html.duckduckgo.com/html/"),
    )

    with patch.object(httpx.AsyncClient, "post", side_effect=[mock_fail, mock_resp]):
        with patch.object(httpx.AsyncClient, "get", return_value=mock_fail):
            results = await provider.search("ai news", max_results=5)

    assert len(results) == 2
    assert results[0]["title"] == "AI Breakthrough in 2026"
    assert results[0]["url"] == "https://news.example.com/ai"
    assert (
        results[0]["snippet"]
        == "Researchers announce a major milestone in autonomous agents. Full report here."
    )

    assert results[1]["title"] == "Robotics Update"
    assert results[1]["url"] == "https://example.org/robotics"
    assert (
        results[1]["snippet"] == "New humanoid models demonstrate real-time reasoning capabilities."
    )


@pytest.mark.asyncio
async def test_duckduckgo_provider_instant_answer_fallback() -> None:
    """Verify DuckDuckGoSearchProvider falls back to Instant Answer JSON when HTML is unavailable."""
    provider = DuckDuckGoSearchProvider()

    ia_json = {
        "Heading": "Python (programming language)",
        "AbstractText": "Python is a high-level, general-purpose programming language.",
        "AbstractURL": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "RelatedTopics": [
            {
                "FirstURL": "https://duckduckgo.com/c/Python_software",
                "Text": "Python software - Software written in Python.",
            },
            {
                "FirstURL": "https://duckduckgo.com/c/Python_libraries",
                "Text": "Python libraries - Reusable packages.",
            },
        ],
    }

    mock_html_fail = httpx.Response(
        status_code=403, request=httpx.Request("POST", "https://html.duckduckgo.com/html/")
    )
    mock_get_fail = httpx.Response(
        status_code=403, request=httpx.Request("GET", "https://html.duckduckgo.com/html/")
    )
    mock_ia_resp = httpx.Response(
        status_code=200,
        json=ia_json,
        request=httpx.Request("GET", "https://api.duckduckgo.com/"),
    )

    with patch.object(httpx.AsyncClient, "post", return_value=mock_html_fail):
        with patch.object(httpx.AsyncClient, "get", side_effect=[mock_get_fail, mock_ia_resp]):
            results = await provider.search("python", max_results=2)

    assert len(results) == 2
    assert results[0]["title"] == "Python (programming language)"
    assert results[0]["url"] == "https://en.wikipedia.org/wiki/Python_(programming_language)"
    assert results[1]["title"] == "Python software"


@pytest.mark.asyncio
async def test_duckduckgo_provider_empty_query() -> None:
    """Verify empty query returns empty results immediately without making requests."""
    provider = DuckDuckGoSearchProvider()
    results = await provider.search("   ")
    assert results == []


@pytest.mark.asyncio
async def test_web_search_tool_execution_with_custom_provider(tool_context: ToolContext) -> None:
    """Verify WebSearchTool executes search via pluggable provider strategy."""

    class MockCustomProvider:
        async def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
            return [
                {
                    "title": f"Result for {query}",
                    "url": f"https://example.com/search?q={query}",
                    "snippet": f"Top match snippet for {query}",
                }
            ][:max_results]

    custom_provider = MockCustomProvider()
    tool = WebSearchTool(provider=custom_provider)
    assert isinstance(tool.provider, SearchProviderProtocol)

    result = await tool.execute(
        params={"query": "quantum computing", "max_results": 3},
        context=tool_context,
    )

    assert result.success is True
    assert isinstance(result.output, list)
    output = cast(list[dict[str, Any]], result.output)
    assert len(output) == 1
    assert output[0]["title"] == "Result for quantum computing"
    assert output[0]["url"] == "https://example.com/search?q=quantum computing"


@pytest.mark.asyncio
async def test_web_search_tool_default_provider_execution(tool_context: ToolContext) -> None:
    """Verify default WebSearchTool uses DuckDuckGo provider."""
    tool = WebSearchTool()
    assert isinstance(tool.provider, DuckDuckGoSearchProvider)

    mock_results = [
        {"title": "Test Title", "url": "https://example.com", "snippet": "Test Snippet"}
    ]

    with patch.object(tool.provider, "search", return_value=mock_results):
        result = await tool.execute(
            params={"query": "uclone framework"},
            context=tool_context,
        )

    assert result.success is True
    assert result.output == mock_results


def test_web_search_parameters_schema() -> None:
    """Verify WebSearchTool parameters schema validation."""
    tool = WebSearchTool()
    schema = tool.parameters_schema
    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert "max_results" in schema["properties"]

    valid = WebSearchParams.model_validate({"query": "ai agents", "max_results": 10})
    assert valid.query == "ai agents"
    assert valid.max_results == 10


# ======================================================================================
# 5. Tool Registry Integration Tests
# ======================================================================================


def test_default_registry_includes_web_tools() -> None:
    """Verify create_default_registry() and ToolRegistry.with_builtins() include web tools."""
    registry = create_default_registry()
    tools = registry.list_tools()
    tool_names = {t.name for t in tools}

    assert "web_fetch" in tool_names
    assert "web_search" in tool_names
    assert "file_read" in tool_names
    assert "file_write" in tool_names
    assert "file_edit" in tool_names
    assert "file_search" in tool_names
    assert "directory_list" in tool_names
    assert "bash_run" in tool_names
    assert "run_command" in tool_names
    assert "delegate_subagent" in tool_names
    assert "generate_image" in tool_names
    assert "tool_result_read" in tool_names
    assert len(tools) == 14

    web_fetch = registry.get("web_fetch")
    assert isinstance(web_fetch, WebFetchTool)

    web_search = registry.get("web_search")
    assert isinstance(web_search, WebSearchTool)
