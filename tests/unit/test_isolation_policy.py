"""Tests for the isolation model: the decided default, and what it deliberately cannot say.

The discriminated union exists so that a control which cannot be enforced at a level is
not a field of that level. `docs/security-threat-model.md` D1: "a security control that
is accepted and silently ignored is precisely what P6 forbids."
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from uclone_x.errors import SandboxViolationError
from uclone_x.sandbox import (
    ContainerIsolation,
    ExecutionRequest,
    IsolationLevel,
    NoIsolation,
    WasmIsolation,
    WorkspaceIsolation,
    WorkspaceSandboxRunner,
    effective_isolation_level,
    is_secret_env_name,
)
from uclone_x.sandbox.models import SECRET_ENV_EXACT_NAMES, SECRET_NAME_TAILS


def test_execution_defaults_to_workspace_isolation() -> None:
    """P3 as amended for issue -001: the default is `workspace`, decided by the project owner."""
    request = ExecutionRequest(
        command="python", cwd=Path("/srv/work"), workspace_root=Path("/srv/work")
    )

    assert request.isolation.level is IsolationLevel.WORKSPACE
    assert isinstance(request.isolation, WorkspaceIsolation)


def test_no_isolation_is_reachable_only_by_stating_it() -> None:
    """`none` "must never be reached by defaulting" — so it must be written out."""
    request = ExecutionRequest(
        command="python",
        cwd=Path("/srv/work"),
        workspace_root=Path("/srv/work"),
        isolation=NoIsolation(),
    )

    assert request.isolation.level is IsolationLevel.NONE


def test_unenforceable_controls_are_not_expressible() -> None:
    """The point of the union: no `allow_network` where egress cannot be enforced."""
    with pytest.raises(ValidationError):
        WorkspaceIsolation.model_validate({"allow_network": False})

    with pytest.raises(ValidationError):
        NoIsolation.model_validate({"memory_limit_mb": 512})

    with pytest.raises(ValidationError):
        WorkspaceIsolation.model_validate({"cpu_shares": 1.0})


def test_enforceable_controls_deny_egress_by_default() -> None:
    """Threat model D1 control 2: egress deny-by-default wherever it can be enforced."""
    container = ContainerIsolation(image="python:3.11")
    wasm = WasmIsolation()

    assert container.allow_network is False
    assert container.egress_allowlist == ()
    assert wasm.allow_network is False


def test_environment_is_constructed_not_inherited() -> None:
    """Threat model D1 control 1, which closes T3 at every level including `none`."""
    request = ExecutionRequest(
        command="python", cwd=Path("/srv/work"), workspace_root=Path("/srv/work")
    )

    assert request.env == {}
    assert request.env_allowlist == ()


def test_credential_shaped_names_are_recognised() -> None:
    assert is_secret_env_name("ANTHROPIC_API_KEY")
    assert is_secret_env_name("gh_token")
    assert is_secret_env_name("AWS_SECRET_ACCESS_KEY")
    assert not is_secret_env_name("PATH")
    assert not is_secret_env_name("HOME")


# The names below are the point of these two tests. Every pre-existing `env_allowlist=`
# in this suite used five benign placeholders (`PATH`, `HOME`, `HOST_SAFE_VAR`,
# `SAFE_VAR`, `NON_EXISTENT_VAR`), none of them anywhere near a matching boundary, so a
# green suite said nothing at all about what this predicate does. These name the real
# ones on both sides.

BARE_CREDENTIAL_NAMES: tuple[str, ...] = (
    # Every one of these returned False while the patterns were all `*_API_KEY`-shaped:
    # `fnmatch`'s `*` cannot match a zero-length segment before the literal `_`.
    "API_KEY",
    "TOKEN",
    "PASSWORD",
    "PRIVATE_KEY",
    "PASSWD",
    "SECRET",
    "CREDENTIALS",
    "SESSION_COOKIE",
    "KUBECONFIG",
    "NETRC",
    "DATABASE_URL",
    "REDIS_URL",
    "MONGODB_URI",
    "DOCKER_AUTH_CONFIG",
    # Not secrets themselves; each gets the child to one it could not otherwise reach.
    # These five replaced the AWS_*/GH_*/GITHUB_* namespace globs (#95).
    "SSH_AUTH_SOCK",
    "AWS_ACCESS_KEY_ID",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_CONFIG_FILE",
    "GH_CONFIG_DIR",
    # Container credential providers (ECS task role / EKS Pod Identity). Unlike the file
    # variables these have no default fallback: nothing resolves them without the var.
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
    # Header and dotted span-attribute spellings of the same credential.
    "x-api-key",
    "llm.api_key",
)

SUFFIXED_CREDENTIAL_NAMES: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "LANGFUSE_SECRET_KEY",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "DB_PASSWORD",
    "APP_CREDENTIALS",
    "API_SECRET",
    "SSH_PRIVATE_KEY",
    "BASIC_AUTH",
    "TEST_REDIS_URL",
    "CACHE_REDIS_URL",
    "STAGING_MONGODB_URI",
    "PROD_REDIS_URL",
    "PROD_DATABASE_URL",
)

BENIGN_NAMES: tuple[str, ...] = (
    # PROVENANCE IS PART OF THIS FIXTURE, PER NAME.
    #
    # An unsourceable name in the *credential* fixture is nearly harmless: over-refusing
    # a variable no tool sets costs nothing. An unsourceable name here is different — it
    # manufactures evidence. A green run reads as "the predicate does not over-match real
    # tools" when all it showed is that it does not over-match names the author invented.
    # Unsourced names previously removed in review (such as SECRETS_DIR and AWS_EXTERNAL_ID)
    # appeared in only three places — this fixture, the models.py docstring, and
    # sandbox-execution-architecture.md — with the test citing the docstring and the
    # docstring citing the test.
    #
    # Every name below cites where it comes from. If you add one and cannot, do not.
    #
    # --- git, via `git help environment`; a Git-native framework shells out to these ---
    "GIT_AUTHOR_NAME",
    "GIT_AUTHOR_EMAIL",
    "GIT_AUTHOR_DATE",
    "GIT_COMMITTER_NAME",
    # --- cargo, via the Cargo book "Environment variables Cargo sets for crates" ---
    "CARGO_PKG_AUTHORS",
    # --- npm, via `npm help scripts` package.json-to-env expansion ---
    "npm_package_author_name",
    "npm_package_author_email",
    # --- boundary specimens (contain credential substring 'TOKEN' but are benign configuration) ---
    # MAX_TOKENS: src/uclone_x/llm/connectors/gemini.py
    "MAX_TOKENS",
    # TOKEN_BUDGET: docs/dynamic-persona-interface.md, a P4 concept of this framework
    "TOKEN_BUDGET",
    # .env.example, all four. LANGFUSE_PUBLIC_KEY is the sharp one: it is publishable by
    # design and sits beside LANGFUSE_SECRET_KEY, which *_SECRET_KEY correctly refuses.
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_HOST",
    "OLLAMA_FAST_BASE_URL",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    # --- HuggingFace tokenizers, via its own docs; the `token` substring trap ---
    "TOKENIZERS_PARALLELISM",
    # --- pass(1), the standard unix password store; excluded by the artifact count:
    # GPG-encrypted blobs plus a keyring located elsewhere is two artifacts, not one ---
    "PASSWORD_STORE_DIR",
    # --- POSIX / IEEE Std 1003.1 environment variables ---
    "PATH",
    "HOME",
    "LANG",
    "PWD",
    "SHELL",
    "TERM",
    # --- python, via the Python docs ---
    "PYTHONPATH",
    "VIRTUAL_ENV",
    # --- GitHub Actions default environment, via the Actions docs. Refused at
    # construction by the GITHUB_* glob until #95; a runner sets ~30 of these ---
    "GITHUB_REPOSITORY",
    "GITHUB_REPOSITORY_OWNER",
    "GITHUB_REF",
    "GITHUB_HEAD_REF",
    "GITHUB_SHA",
    "GITHUB_ACTOR",
    "GITHUB_WORKSPACE",
    "GITHUB_RUN_ID",
    "GITHUB_EVENT_NAME",
    "GITHUB_API_URL",
    # GITHUB_EVENT_PATH: ends in a pointer suffix and is benign — the counterexample
    # that stops _POINTER_SUFFIXES being promoted into a classifier.
    "GITHUB_EVENT_PATH",
    # --- AWS SDK/CLI configuration, via the AWS CLI environment-variable docs ---
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_DEFAULT_OUTPUT",
    "AWS_ENDPOINT_URL",
    "AWS_RETRY_MODE",
    "AWS_MAX_ATTEMPTS",
    "AWS_PAGER",
    "AWS_EC2_METADATA_DISABLED",
    # Considered and excluded (#95): they select or name which credential resolves, they
    # do not locate one, and confer no capability a child that can read the default
    # credentials file does not already have (#109).
    "AWS_PROFILE",
    "AWS_ROLE_ARN",
    # --- gh CLI, via `gh help environment` ---
    "GH_HOST",
    "GH_REPO",
    "GH_EDITOR",
    "GH_PAGER",
    "GH_BROWSER",
    "GH_PROMPT_DISABLED",
)


# Names the removed AWS_*/GH_*/GITHUB_* globs used to catch, classified. This corpus is
# the whole basis of the guarantee below, so it is written out rather than sampled: the
# first version of this test asserted "no credential was released" over a hand-picked
# handful, which passed while three container-credential names were in fact released.
# An example list cannot express a universal property — it can only be made wide enough
# to be useful, and honest about being a list.
NAMESPACE_CREDENTIALS_COVERED_BY_TAIL: tuple[str, ...] = (
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    "GH_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "GITHUB_TOKEN",
    "GITHUB_ENTERPRISE_TOKEN",
)

NAMESPACE_CREDENTIALS_NEEDING_EXACT_ENTRY: tuple[str, ...] = (
    "AWS_ACCESS_KEY_ID",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_CONFIG_FILE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
    "GH_CONFIG_DIR",
)

# Pointer-shaped suffixes: a name ending in one of these does not *contain* a credential,
# it says where to find one. Every miss in this area has had this shape — the two AWS
# file variables, GH_CONFIG_DIR, and the three AWS_CONTAINER_* names all end here, and
# `AWS_CONTAINER_AUTHORIZATION_TOKEN` is caught by `*_TOKEN` while its `_FILE` sibling
# was not. No tail rule can cover them, because the tail describes the pointer.
_POINTER_SUFFIXES: tuple[str, ...] = ("_FILE", "_URI", "_DIR", "_SOCK", "_PATH")


def test_namespace_removal_releases_no_credential() -> None:
    """#95 dropped three namespace globs; no credential in the corpus was released.

    Guarantees exactly what it enumerates, and no more. The globs were a blanket over
    three vendor namespaces; ten exact names are not, so a credential in those
    namespaces is now refused only because a specific rule reaches it.
    """
    for name in NAMESPACE_CREDENTIALS_COVERED_BY_TAIL:
        assert is_secret_env_name(name) is True, name
    for name in NAMESPACE_CREDENTIALS_NEEDING_EXACT_ENTRY:
        assert is_secret_env_name(name) is True, name

    # No namespace glob survives: a non-credential sibling in each namespace is free.
    for name in ("AWS_REGION", "GH_HOST", "GITHUB_REPOSITORY"):
        assert is_secret_env_name(name) is False, name


def test_pointer_shaped_vendor_credentials_are_exact_listed() -> None:
    """The structural check the example list could not make.

    A pointer-shaped credential name cannot be reached by a tail rule, so it must appear
    in `SECRET_ENV_EXACT_NAMES` verbatim. Asserting membership in that tuple — rather
    than just that the predicate happens to return True — is what makes this fail loudly
    if someone "fixes" a future miss by widening a tail into a substring, which is the
    regression ruled out twice on #87.

    **What this guarantees, and what it does not.** It is structural about *mechanism*
    and merely enumerative about *membership*, so it cannot discover a name nobody
    listed. Mutation-checked in review, both halves: drop
    `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` from the exact list while widening the
    patterns with a `*CREDENTIALS*` substring glob and the predicate still returns True
    — `test_namespace_removal_releases_no_credential` passes, blind to it, and this test
    fails, which is the mechanism regression it exists to catch. But add a real
    pointer-shaped credential that no corpus names — `AWS_CONTAINER_CREDENTIALS_TOKEN_PATH`,
    `AWS_IDENTITY_PROVIDER_CONFIG_FILE`, `GH_CREDENTIAL_STORE_DIR` — and every test here
    stays green while the name goes free.

    `_POINTER_SUFFIXES` is therefore not a classifier and must never be promoted into
    one: it is safe only because it is applied to a domain already filtered to known
    credentials. Standalone it is wrong in both directions — it misses pointers that do
    not carry the suffix (`GNUPGHOME`, `DOCKER_CONFIG`, `GIT_CONFIG_GLOBAL`) and matches
    benign names that do (`GITHUB_EVENT_PATH`). That a substring or suffix predicate
    cannot express "credential-shaped" is the standing lesson of #87, #95 and #96; each
    fix here buys one spelling, and the sequence ends with a named set tested against a
    fixture of real variable names, not with a cleverer pattern.
    """
    for name in NAMESPACE_CREDENTIALS_NEEDING_EXACT_ENTRY:
        if name.endswith(_POINTER_SUFFIXES):
            assert name in SECRET_ENV_EXACT_NAMES, (
                f"{name} is pointer-shaped and must be listed exactly, not matched by a tail"
            )

    # The asymmetry that produced the AWS_CONTAINER_* miss: a bare credential name is
    # caught by a tail, and its pointer sibling needs its own entry. Assert the pair.
    assert is_secret_env_name("AWS_CONTAINER_AUTHORIZATION_TOKEN") is True
    assert "AWS_CONTAINER_AUTHORIZATION_TOKEN" not in SECRET_ENV_EXACT_NAMES
    assert "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE" in SECRET_ENV_EXACT_NAMES


GLUED_FORM_CREDENTIALS: tuple[str, ...] = (
    # No `_` precedes the credential word, so no tail reaches them (#96). Sourced:
    "PGPASSWORD",  # libpq, via the PostgreSQL docs; the commonest way a PG password
    # reaches a subprocess
    "MYSQL_PWD",  # mysql client, via the MySQL docs
)

CREDENTIAL_LOCATORS: tuple[str, ...] = (
    # Locators and code-execution vectors (#96 artifact count):
    "DOCKER_CONFIG",  # docker CLI: config.json holds a plaintext base64 `auths` block
    "GNUPGHOME",  # gnupg: the private keyring itself
    "GPG_AGENT_INFO",  # gpg-agent socket (historical; inert/removed in GnuPG 2.1+)
    "SSH_ASKPASS",  # an executable ssh/git will RUN to obtain a passphrase (code-execution vector)
)

GIT_CONFIG_FAMILY: tuple[str, ...] = (
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_KEY_0",
    "GIT_CONFIG_VALUE_0",
    "GIT_CONFIG_KEY_17",
    "GIT_CONFIG_VALUE_17",
    "GIT_CONFIG_KEY_999",
)


@pytest.mark.parametrize("name", GLUED_FORM_CREDENTIALS + CREDENTIAL_LOCATORS)
def test_glued_forms_and_locators_are_refused(name: str) -> None:
    """#96: the bug shape of #91, one spelling over.

    A glued name has no `_` before the credential word, so neither the bare entry nor the
    `*_` glob reaches it, and widening a tail to a substring is not available — the
    `PASSWORD` substring would refuse `PASSWORD_STORE_DIR` and the `AUTH` one would
    refuse `GIT_AUTHOR_NAME`. Named entries are the only mechanism left.
    """
    assert is_secret_env_name(name) is True

    with pytest.raises(ValidationError, match="Credential-shaped environment variable"):
        ExecutionRequest(
            command="python",
            cwd=Path("/srv/work"),
            workspace_root=Path("/srv/work"),
            env_allowlist=(name,),
        )


@pytest.mark.parametrize("name", GIT_CONFIG_FAMILY)
def test_git_config_family_is_refused_at_any_arity(name: str) -> None:
    """`GIT_CONFIG_*` reaches arbitrary code execution, verified rather than reasoned.

    A `credential.helper` beginning `!` runs through the shell. `GIT_CONFIG_GLOBAL` and
    `GIT_CONFIG_SYSTEM` redirect the config stack and execute it; `GIT_CONFIG_COUNT` with
    `GIT_CONFIG_KEY_<n>`/`GIT_CONFIG_VALUE_<n>` executes it with **no file at all**.

    The arity is unbounded, which is the third failure class beside pointers and glued
    forms: no finite named set expresses it, so `SECRET_ENV_FAMILY_PATTERNS` carries the
    two index globs. `_KEY_999` is here to assert the family is a family.
    """
    assert is_secret_env_name(name) is True


def test_which_list_criterion_holds_for_every_tail() -> None:
    """The mechanical rule that decides the class, asserted against the tails themselves.

    A tail is legitimate only when the credential word is the whole trailing segment.
    Checked structurally: no tail may contain `_` unless every one of its own segments is
    part of the credential word — which is what makes `APIKEY` and `KUBECONFIG` correct
    tails while `PGPASSWORD` is not one, and stops the next builder reaching for a
    substring when a glued form appears.
    """
    for tail in SECRET_NAME_TAILS:
        assert tail == tail.upper(), tail
        assert not tail.startswith("*"), f"{tail} is a glob, not a tail"
        assert not tail.endswith("_"), tail
        # A tail matches as a whole trailing segment, so a glued name that merely ends
        # with the tail's letters must NOT be caught by it.
        assert is_secret_env_name(f"PG{tail}") is (f"PG{tail}" in SECRET_ENV_EXACT_NAMES), (
            f"PG{tail} is matched by a tail — {tail} is behaving as a substring"
        )


def test_role_assumption_parameters_are_not_credentials() -> None:
    """Considered and excluded, recorded so the exclusions are not mistaken for misses.

    `AWS_PROFILE` and `AWS_ROLE_ARN` both select or name a credential the child must
    already be able to obtain. None of them locates one, and
    none confers a capability the child lacks — see #109, where a child with only `HOME`
    allowlisted read every profile in the default credentials file.
    """
    for name in ("AWS_PROFILE", "AWS_ROLE_ARN"):
        assert is_secret_env_name(name) is False, name


@pytest.mark.parametrize("name", BARE_CREDENTIAL_NAMES + SUFFIXED_CREDENTIAL_NAMES)
def test_credential_names_are_refused_at_construction(name: str) -> None:
    """A credential-shaped `env_allowlist` entry raises, bare form included."""
    assert is_secret_env_name(name) is True

    with pytest.raises(ValidationError, match="Credential-shaped environment variable"):
        ExecutionRequest(
            command="python",
            cwd=Path("/srv/work"),
            workspace_root=Path("/srv/work"),
            env_allowlist=(name,),
        )


@pytest.mark.parametrize("name", BENIGN_NAMES)
def test_benign_names_are_not_refused(name: str) -> None:
    """The predicate matches whole trailing segments, never substrings.

    Over-matching in a trace costs a `[REDACTED]`; over-matching here is a hard
    construction refusal that breaks the tool outright, so this direction is the one
    that has to be pinned by name.
    """
    assert is_secret_env_name(name) is False

    request = ExecutionRequest(
        command="git",
        cwd=Path("/srv/work"),
        workspace_root=Path("/srv/work"),
        env_allowlist=(name,),
    )
    assert request.env_allowlist == (name,)


@pytest.mark.parametrize(
    "name",
    (
        "REDIS_URL",
        "MONGODB_URI",
        "DATABASE_URL",
        "TEST_REDIS_URL",
        "CACHE_REDIS_URL",
        "STAGING_MONGODB_URI",
        "PROD_REDIS_URL",
        "PROD_DATABASE_URL",
    ),
)
def test_prefixed_database_and_cache_urls_are_refused(name: str) -> None:
    """Tails like DATABASE_URL, REDIS_URL, MONGODB_URI catch bare and prefixed forms."""
    assert is_secret_env_name(name) is True
    with pytest.raises(ValidationError, match="Credential-shaped environment variable"):
        ExecutionRequest(
            command="python",
            cwd=Path("/srv/work"),
            workspace_root=Path("/srv/work"),
            env_allowlist=(name,),
        )


@pytest.mark.parametrize(
    "name",
    (
        "AWS_ENDPOINT_URL",
        "GITHUB_API_URL",
        "GH_HOST",
        "LANGFUSE_HOST",
        "OLLAMA_FAST_BASE_URL",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ),
)
def test_benign_urls_and_endpoints_construct_successfully(name: str) -> None:
    """Benign endpoint/service URLs are not caught by URL/URI tail patterns."""
    assert is_secret_env_name(name) is False
    request = ExecutionRequest(
        command="python",
        cwd=Path("/srv/work"),
        workspace_root=Path("/srv/work"),
        env_allowlist=(name,),
    )
    assert request.env_allowlist == (name,)


def test_a_request_can_strengthen_isolation_but_never_weaken_it() -> None:
    """P3 as amended: a requesting artifact never raises its own ceiling.

    Issue #30: Requesting an isolation level with no available backend runner fails
    explicitly with SandboxViolationError (P6).
    """
    floor = IsolationLevel.WORKSPACE

    # A synthesized skill asking for host execution gets the floor instead.
    assert effective_isolation_level(IsolationLevel.NONE, floor) is IsolationLevel.WORKSPACE

    # A floor of `none` constrains nothing.
    assert (
        effective_isolation_level(IsolationLevel.NONE, IsolationLevel.NONE) is IsolationLevel.NONE
    )

    # Issue #30: Requesting CONTAINER or WASM without available runner backends fails explicitly
    with pytest.raises(SandboxViolationError, match="has no available runner backend"):
        effective_isolation_level(IsolationLevel.CONTAINER, floor)

    with pytest.raises(SandboxViolationError, match="has no available runner backend"):
        effective_isolation_level(IsolationLevel.WASM, floor)

    # When backends exist in the environment, asking for more is honoured
    all_levels = frozenset(
        {
            IsolationLevel.NONE,
            IsolationLevel.WORKSPACE,
            IsolationLevel.CONTAINER,
            IsolationLevel.WASM,
        }
    )
    assert (
        effective_isolation_level(IsolationLevel.CONTAINER, floor, available_levels=all_levels)
        is IsolationLevel.CONTAINER
    )
    assert (
        effective_isolation_level(IsolationLevel.WASM, floor, available_levels=all_levels)
        is IsolationLevel.WASM
    )


def test_isolation_policies_are_frozen() -> None:
    policy = WorkspaceIsolation()

    with pytest.raises(ValidationError):
        policy.write_paths = ()  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_home_allowlist_governs_variables_not_filesystem_reachability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Threat model D1 / #109: env allowlist constrains inherited variables, not filesystem reachability.

    Allowlisting `HOME` permits child processes to read disk-based credentials
    (~/.aws/credentials, ~/.config/gh/hosts.yml) at workspace isolation level,
    proving that environment allowlists do not create a filesystem boundary.
    """
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()

    aws_dir = fake_home / ".aws"
    aws_dir.mkdir()
    aws_creds = aws_dir / "credentials"
    aws_creds.write_text(
        "[default]\naws_access_key_id = DUMMY_KEY\naws_secret_access_key = DUMMY_SECRET\n"
    )

    gh_dir = fake_home / ".config" / "gh"
    gh_dir.mkdir(parents=True)
    gh_hosts = gh_dir / "hosts.yml"
    gh_hosts.write_text("github.com:\n    oauth_token: gho_dummy_token_12345\n    user: octocat\n")

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("HOME", str(fake_home))

    runner = WorkspaceSandboxRunner()
    code = (
        "import os, pathlib\n"
        "home = pathlib.Path(os.environ['HOME'])\n"
        "aws = (home / '.aws' / 'credentials').read_text()\n"
        "gh = (home / '.config' / 'gh' / 'hosts.yml').read_text()\n"
        "print('AWS_CREDS:' + aws.strip())\n"
        "print('GH_HOSTS:' + gh.strip())\n"
    )
    request = ExecutionRequest(
        command=sys.executable,
        args=("-c", code),
        cwd=workspace,
        workspace_root=workspace,
        env_allowlist=("HOME",),
    )

    result = await runner.execute(request)
    assert result.exit_code == 0
    assert "DUMMY_SECRET" in result.stdout
    assert "gho_dummy_token_12345" in result.stdout

    # Env allowlist allows HOME but refuses AWS_SECRET_ACCESS_KEY
    assert is_secret_env_name("HOME") is False
    assert is_secret_env_name("AWS_SECRET_ACCESS_KEY") is True
