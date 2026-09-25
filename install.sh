#!/usr/bin/env bash
# ==============================================================================
# UClone-X — beginner installer: the one install a first-time user runs.
#
#   curl -fsSL https://raw.githubusercontent.com/UClone-AI/uclone-x/main/install.sh | bash
#
# Installs the seed — `[cli,http]`, ~31 MB — which is the whole of what the
# dashboard needs. Then *offers* the image engine, which is two orders of
# magnitude larger. The seed install is the part that must not fail; the image
# install is allowed to fail, and when it does this script says so and leaves a
# working dashboard behind, because the dashboard is where the user asks the
# agent to finish the job.
#
#   ./install.sh                 seed, then ask about the image engine and the models
#   ./install.sh --with-image    seed, then install the image engine, no question
#   ./install.sh --no-image      seed only
#   ./install.sh --with-models   also fetch the local LLM and image model, no question
#   ./install.sh --no-models     leave the models to the dashboard
#   ./install.sh --dry-run       resolve everything, install nothing
#   ./install.sh --venv PATH     target a venv other than ./.venv
#   ./install.sh --yes           allow fetching uv, Python, the image engine and the models without asking
#   ./install.sh --no-start      do not offer to start the dashboard at the end
#
# It needs nothing but macOS or Linux and curl. When no Python 3.11+ is present
# it fetches a private Python 3.12 (~71 MB) through uv, installing uv first
# (~39 MB, ~/.local/bin) when it is missing, after asking once. Run from a checkout it installs that checkout into ./.venv; run on its
# own it installs the published package into ~/.uclone-x/venv and links `ucx` into
# ~/.local/bin. The questions are asked on the terminal even when the script is
# piped from curl, and the last one offers to start the dashboard.
#
# The models are the last step and the largest: Ollama and a local LLM, and the
# SDXL checkpoint the image engine loads. Software without weights cannot answer
# or draw, so `ucx install` fetches both — offered here, and offered again in the
# dashboard for anyone who says no now.
# ==============================================================================
set -u

SCRIPT_PATH="${BASH_SOURCE[0]:-}"
REPO_ROOT=""
if [ -n "$SCRIPT_PATH" ] && [ -f "$SCRIPT_PATH" ]; then
    REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
fi
# A checkout installs itself; anything else (the script saved on its own, or
# piped from curl) installs the published package.
if [ -n "$REPO_ROOT" ] && grep -q '^name = "uclone-x"' "$REPO_ROOT/pyproject.toml" 2>/dev/null; then
    SOURCE="$REPO_ROOT"
    VENV_DIR="$REPO_ROOT/.venv"
else
    SOURCE="uclone-x"
    VENV_DIR="$HOME/.uclone-x/venv"
fi
IMAGE_MODE="ask"
MODELS_MODE="ask"
DRY_RUN="no"
ASSUME_YES="no"
START_MODE="ask"
UV_PYTHON="3.12"
UV_INSTALLER_URL="https://astral.sh/uv/install.sh"
# Where macOS keeps the Command Line Tools shims. Overridable only for the tests.
CLT_SHIM_DIR="${UCX_INSTALL_CLT_SHIM_DIR:-/usr/bin}"
# The terminal the questions are asked on. Under `curl ... | bash` stdin is the
# script itself: a question read from it would swallow the script's next line, and
# `[ -t 0 ]` is false, so every question used to be skipped on the documented path.
# The terminal is still there as /dev/tty. Overridable only for the tests.
TTY="${UCX_INSTALL_TTY:-/dev/tty}"

# The seed: the smallest set of extras that still serves the dashboard.
SEED_EXTRAS="cli,http"
# The image engine. `media` carries pillow, diffusers, torch and transformers -- the
# in-process engine runs on any platform, on MPS where the host offers it (#1095).
# Measured 2026-09-17 on CPython 3.13.14: core+image = 1097 MB, 66 distributions,
# so the engine itself is ~1065 MB on top of the 31 MB seed. That measurement predates
# #1095 swapping mflux/mlx for diffusers/torch/transformers and has not been re-run; the
# order of magnitude holds -- torch 542 MB, transformers 53 MB and pillow 14 MB alone are
# 609 MB in this project's own venv -- but treat the exact figure as the earlier extra's.
IMAGE_EXTRAS="media"
MIN_MINOR=11

say()  { printf '%s\n' "$*"; }
step() { printf '\n\033[1;36m%s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m✔\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m✖\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m✖ %s\033[0m\n' "$*" >&2; exit 1; }

# usage: ask "<question>" <y|n default>. Sets ANSWER to y or n. Returns 1, asking
# nothing, when there is no terminal to ask on (CI, cron, a detached session).
ask() {
    { : < "$TTY"; } 2>/dev/null || return 1
    printf '  %s ' "$1"
    answer=""
    read -r answer < "$TTY" || true
    case "$answer" in
        [yY]*) ANSWER="y" ;;
        [nN]*) ANSWER="n" ;;
        *)     ANSWER="$2" ;;
    esac
    return 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --with-image)  IMAGE_MODE="yes"; shift ;;
        --no-image)    IMAGE_MODE="no"; shift ;;
        --with-models) MODELS_MODE="yes"; shift ;;
        --no-models)   MODELS_MODE="no"; shift ;;
        --dry-run)    DRY_RUN="yes"; shift ;;
        -y|--yes)     ASSUME_YES="yes"; shift ;;
        --no-start)   START_MODE="no"; shift ;;
        --venv)       [ $# -ge 2 ] || die "--venv needs a path"; VENV_DIR="$2"; shift 2 ;;
        -h|--help)    sed -n '2,34p' "${SCRIPT_PATH:-/dev/null}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)            die "unknown option: $1 (try --help)" ;;
    esac
done
# The link in ~/.local/bin points at this path, and a relative one would dangle.
case "$VENV_DIR" in /*) ;; *) VENV_DIR="$(pwd)/$VENV_DIR" ;; esac

# ------------------------------------------------------------------------------
# Step 0a — the interpreter. A normal Mac has Python 3.9 at best, and without the
# Xcode Command Line Tools its /usr/bin/python3 is a shim that opens an install
# dialog. Neither can run this project, so when no 3.11+ is on PATH the installer
# gets one through uv, which downloads a standalone CPython (~71 MB) into its own
# cache. It needs no admin rights, no Homebrew and no compiler.
# ------------------------------------------------------------------------------
step "Looking for a Python interpreter (need >= 3.$MIN_MINOR)"

# Without the Command Line Tools, running /usr/bin/python3 pops a GUI dialog
# instead of printing a version. Do not run it.
SKIP_DIR=""
if [ "$(uname -s 2>/dev/null)" = "Darwin" ] && ! xcode-select -p >/dev/null 2>&1; then
    SKIP_DIR="$CLT_SHIM_DIR"
fi

PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
    bin="$(command -v "$candidate" 2>/dev/null)" || continue
    [ -n "$SKIP_DIR" ] && [ "$(dirname "$bin")" = "$SKIP_DIR" ] && continue
    ver="$("$bin" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" || continue
    major="${ver%%.*}"; minor="${ver##*.}"
    if [ "$major" = "3" ] && [ "$minor" -ge "$MIN_MINOR" ] 2>/dev/null; then
        PYTHON="$bin"
        ok "Python $ver at $bin"
        break
    fi
done

find_uv() {
    for c in "$(command -v uv 2>/dev/null || true)" \
             ${UV_INSTALL_DIR:+"$UV_INSTALL_DIR/uv" "$UV_INSTALL_DIR/bin/uv"} \
             ${XDG_BIN_HOME:+"$XDG_BIN_HOME/uv"} \
             "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        [ -n "$c" ] && [ -x "$c" ] && { printf '%s\n' "$c"; return 0; }
    done
    return 1
}
UV="$(find_uv || true)"
# Whether the dashboard will find uv later: it looks on PATH only.
UV_ON_PATH="no"
command -v uv >/dev/null 2>&1 && UV_ON_PATH="yes"

# An environment from an earlier run already carries its own Python, so a re-run
# downloads nothing and needs no consent.
if [ -z "$PYTHON" ] && [ -x "$VENV_DIR/bin/python" ]; then
    ok "No Python 3.$MIN_MINOR+ on PATH; reusing the one in $VENV_DIR."
elif [ -z "$PYTHON" ]; then
    say "  No Python 3.$MIN_MINOR or newer on this machine."
    say "  This installer can fetch a private Python $UV_PYTHON (~71 MB) through uv. It goes in"
    say "  uv's own directory, needs no admin password and does not replace the system one."
    if [ -z "$UV" ]; then
        say "  uv is not installed either. It is a single program (~39 MB) that goes in"
        say "  ~/.local/bin and changes nothing else. Source: $UV_INSTALLER_URL"
    fi
    if [ "$DRY_RUN" = "yes" ]; then
        [ -z "$UV" ] && ok "Dry run: would install uv, then Python $UV_PYTHON. Stopping here."
        [ -n "$UV" ] && ok "Dry run: would fetch Python $UV_PYTHON with $UV. Stopping here."
        exit 0
    fi
    # The answer to this question is kept apart from ASSUME_YES, which every later step
    # reads as "the user passed --yes". Folding a typed Y back into ASSUME_YES would make
    # consenting to a 71 MB interpreter consent to the 1.0 GB engine and the ~12 GB of
    # weights too -- questions this run is about to ask properly, one at a time.
    FETCH_PYTHON="$ASSUME_YES"
    if [ "$FETCH_PYTHON" != "yes" ]; then
        question="Fetch Python $UV_PYTHON now? [Y/n]"
        [ -z "$UV" ] && question="Install uv and Python $UV_PYTHON now? [Y/n]"
        if ask "$question" y; then
            [ "$ANSWER" = "y" ] && FETCH_PYTHON="yes"
        else
            say "  Not a terminal, so not asking. Re-run with --yes to allow it."
        fi
    fi
    if [ "$FETCH_PYTHON" != "yes" ]; then
        warn "Nothing was installed."
        say "  Either re-run with --yes, or install Python 3.12 from"
        say "  https://www.python.org/downloads/ and run this again."
        exit 1
    fi
    if [ -z "$UV" ]; then
        step "Installing uv"
        # pipefail: without it a failed download is reported as sh's exit status.
        ( set -o pipefail
          curl -LsSf "$UV_INSTALLER_URL" | UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh ) \
            || die "The uv installer failed; the lines above are the reason."
        UV="$(find_uv || true)"
        [ -n "$UV" ] || die "The uv installer finished but this script cannot find uv (looked on PATH and in ~/.local/bin)."
    fi
fi
[ -n "$UV" ] && ok "uv at $UV"
[ -z "$UV" ] && say "  uv not found — falling back to venv + pip (slower, works the same)."

# ------------------------------------------------------------------------------
# Step 0b — the environment.
# ------------------------------------------------------------------------------
step "Preparing the environment at $VENV_DIR"

VENV_PY="$VENV_DIR/bin/python"
if [ -x "$VENV_PY" ]; then
    ok "Reusing the existing environment."
else
    # The environment is created even under --dry-run. A dry run that skipped it
    # would have to resolve against *some* interpreter, and the only one left is
    # the user's system Python — which is both the wrong answer and, on a
    # Homebrew or uv-managed interpreter, an externally-managed one that refuses.
    # Creating an empty venv costs milliseconds and makes the resolution real.
    mkdir -p "$(dirname "$VENV_DIR")" || die "cannot create $(dirname "$VENV_DIR")"
    if [ -n "$UV" ]; then
        venv_args=""
        # The dashboard installs later through uv on PATH, else through the venv's
        # pip. A uv venv has no pip, so when uv is off PATH give it one.
        [ "$UV_ON_PATH" = "no" ] && venv_args="--seed"
        if [ -n "$PYTHON" ]; then
            "$UV" venv $venv_args --python "$PYTHON" "$VENV_DIR" || die "uv venv failed at $VENV_DIR"
        else
            # Only uv's own Pythons: searching PATH would run /usr/bin/python3, which
            # without the Command Line Tools opens an install dialog.
            UV_PYTHON_PREFERENCE=only-managed "$UV" venv $venv_args --python "$UV_PYTHON" "$VENV_DIR" \
                || die "uv venv failed at $VENV_DIR"
        fi
    else
        "$PYTHON" -m venv "$VENV_DIR" || die "python -m venv failed at $VENV_DIR"
    fi
    ok "Created."
fi

# One installer for the rest of the script. A uv-created venv has no pip in it,
# so `$VENV_PY -m pip` is not a fallback that can be assumed to exist (#971);
# it is tried only after it has been shown to be importable.
install_into_venv() {
    # usage: install_into_venv <description> <target-spec>
    desc="$1"; spec="$2"
    dry=""
    [ "$DRY_RUN" = "yes" ] && dry="--dry-run"

    if [ -n "$UV" ]; then
        "$UV" pip install $dry --python "$VENV_PY" "$spec"
        return $?
    fi
    if "$VENV_PY" -m pip --version >/dev/null 2>&1; then
        if [ "$DRY_RUN" = "yes" ]; then
            "$VENV_PY" -m pip install --dry-run "$spec"
        else
            "$VENV_PY" -m pip install "$spec"
        fi
        return $?
    fi
    warn "Cannot install $desc: this environment has neither uv nor pip."
    say  "  Install uv (https://docs.astral.sh/uv/) or seed pip with:"
    say  "    $VENV_PY -m ensurepip --upgrade"
    return 127
}

# ------------------------------------------------------------------------------
# Step 0c — the seed. Measured 2026-09-17 into a fresh uv venv on CPython 3.13.14:
# 31 MB over an empty venv, 26 distributions. No compiler, no Node, no daemon.
# (An earlier measurement of the same extras reported 24 MB / 30 dists; that
# baseline was a pip-seeded `python -m venv`, which starts 13 MB heavier and
# already carries pip and setuptools. Same seed, different zero point.)
# ------------------------------------------------------------------------------
step "Installing the core (~31 MB)"

if ! install_into_venv "the core" "$SOURCE[$SEED_EXTRAS]"; then
    die "The core install failed. Nothing else was attempted; the lines above are the reason."
fi
if [ "$DRY_RUN" = "yes" ]; then
    ok "Core resolves (dry run — nothing was written)."
    CORE_OK="would install"
else
    ok "Core installed."
    CORE_OK="installed"
fi

# ------------------------------------------------------------------------------
# Step 0d — the image engine. Offered, metered, and allowed to fail.
# ------------------------------------------------------------------------------
IMAGE_OK="skipped"
# The published package declares the `media` extra (read from 0.2.1's METADATA), so the
# one-line install offers the engine exactly as a checkout does. uv still installs an extra a
# package does not declare as nothing while exiting 0, which is why the install below is
# judged by importing the engine's packages, never by the installer's exit status.
if [ "$IMAGE_MODE" = "ask" ]; then
    step "Image generation is optional and large"
    say "  The image engine adds about 1.0 GB of packages. It downloads no weights:"
    say "  it loads a single-file SDXL checkpoint you supply, at UCX_IMAGE_CHECKPOINT or"
    say "  in ~/ai_models/checkpoints/, so without one it installs but cannot generate."
    say "  Run \`ucx media status\` afterwards to see what is still missing."
    say "  The dashboard works without any of this."
    # --yes is read here exactly as step 0e reads it below, so the flag means one thing.
    # The documented one-liner -- `curl ... | bash -s -- --yes` -- has no terminal, so
    # deciding this on the terminal alone gave the person who consented to ~12 GB of weights
    # no engine and no checkpoint, under a summary that read `models installed`.
    if [ "$ASSUME_YES" = "yes" ]; then
        IMAGE_MODE="yes"
    elif ask "Install it now? [y/N]" n; then
        if [ "$ANSWER" = "y" ]; then IMAGE_MODE="yes"; else IMAGE_MODE="no"; fi
    else
        say "  Not a terminal, so not asking. Re-run with --with-image to include it."
        IMAGE_MODE="no"
    fi
fi

if [ "$IMAGE_MODE" = "yes" ]; then
    step "Installing the image engine (~1.0 GB)"
    if install_into_venv "the image engine" "$SOURCE[$IMAGE_EXTRAS]"; then
        if [ "$DRY_RUN" = "yes" ]; then
            IMAGE_OK="would install"
            ok "Image engine resolves (dry run — nothing was written)."
        elif "$VENV_PY" -c "import PIL, diffusers, torch, transformers" >/dev/null 2>&1; then
            IMAGE_OK="installed"
            ok "Image engine installed."
        else
            IMAGE_OK="failed"
            warn "The image engine did not install completely, although the installer reported success."
            say  "  The core is fine and the dashboard will start. Run \`ucx media status\` to see what is missing."
        fi
    else
        IMAGE_OK="failed"
        warn "The image engine did not install. The core is fine and the dashboard will start."
        say  "  Ask the agent in the dashboard to finish this — it can read the error above."
    fi
fi

# ------------------------------------------------------------------------------
# Step 0e — the weights. Everything above installs software that cannot answer or
# draw until a model is on disk, and until now this script left that entirely to
# the dashboard: a beginner who ran every documented step still had no LLM and no
# checkpoint. `ucx install` is the non-interactive twin of what `ucx start` does,
# so the sizes are stated once here and the download is one consented step.
#
# It is the last step on purpose. It is the largest by an order of magnitude and
# the likeliest to be interrupted, and everything before it already leaves a
# working dashboard behind.
# ------------------------------------------------------------------------------
UCX="$VENV_DIR/bin/ucx"
MODELS_OK="skipped"
# The published 0.1.2 has no `ucx install`; offering it there would fetch nothing
# and report success, which is the failure mode this script keeps guarding against.
if [ "$MODELS_MODE" != "no" ] && [ "$DRY_RUN" = "no" ] \
   && { [ ! -x "$UCX" ] || ! "$UCX" install --help >/dev/null 2>&1; }; then
    say ""
    say "  This build has no \`ucx install\`; the dashboard asks for the models instead."
    MODELS_MODE="no"
    MODELS_OK="not in this build"
fi
if [ "$MODELS_MODE" = "ask" ]; then
    step "The models are the big download"
    say "  A local LLM through Ollama (~1.4 GB for qwen3:1.7b, ~5.2 GB for qwen3:8b —"
    say "  which one depends on this machine's memory), and, when the image engine is"
    say "  installed, the SDXL base checkpoint (~6.9 GB)."
    say "  Without them the dashboard opens but nothing can answer or draw."
    say "  You can skip this and let the dashboard ask you later instead."
    if [ "$ASSUME_YES" = "yes" ]; then
        MODELS_MODE="yes"
    elif ask "Download them now? [y/N]" n; then
        if [ "$ANSWER" = "y" ]; then MODELS_MODE="yes"; else MODELS_MODE="no"; fi
    else
        say "  Not a terminal, so not asking. Re-run with --with-models to include them."
        MODELS_MODE="no"
    fi
fi

if [ "$MODELS_MODE" = "yes" ]; then
    if [ "$DRY_RUN" = "yes" ]; then
        step "The models"
        say "  Dry run: would run \`ucx install --yes\` to fetch the LLM and the checkpoint."
        MODELS_OK="would install"
    else
        step "Fetching the models"
        # No image engine means no use for a 6.9 GB checkpoint: the file would sit on
        # disk with nothing able to load it.
        MODEL_ARGS="--yes"
        [ "$IMAGE_OK" = "installed" ] || MODEL_ARGS="--yes --no-image"
        if "$UCX" install $MODEL_ARGS; then
            MODELS_OK="installed"
            ok "Models ready."
        else
            MODELS_OK="incomplete"
            warn "Not every model arrived. The dashboard still starts, and asks again there."
        fi
    fi
fi

# ------------------------------------------------------------------------------
# Memory. Disk is rarely the limit; RAM decides which local models are usable.
# Measured 2026-09-19 on an M-series Mac: the dashboard alone is ~80 MB,
# qwen3:1.7b ~1.7 GB, qwen3:8b ~5.5 GB, SDXL at 512px ~8.3 GB and at 768px
# ~13.2 GB of GPU memory. macOS and a browser want ~4 GB of their own.
# ------------------------------------------------------------------------------
ram_gb() {
    case "$(uname -s 2>/dev/null)" in
        # sysctl is in /usr/sbin, which a stripped-down PATH may not carry.
        Darwin) b="$( (command -v sysctl >/dev/null 2>&1 && sysctl -n hw.memsize) || /usr/sbin/sysctl -n hw.memsize 2>/dev/null)" \
                    && [ -n "$b" ] && echo $(( b / 1073741824 )) ;;
        Linux)  k="$(sed -n 's/^MemTotal: *\([0-9]*\) kB/\1/p' /proc/meminfo 2>/dev/null)" && [ -n "$k" ] && echo $(( (k + 524288) / 1048576 )) ;;
    esac
}
RAM="$(ram_gb || true)"

# ------------------------------------------------------------------------------
# Done.
# ------------------------------------------------------------------------------
START="start"
# The published 0.1.2 predates `ucx start`; its dashboard command is `ucx ui`.
if [ "$DRY_RUN" = "no" ] && [ -x "$UCX" ] && ! "$UCX" start --help >/dev/null 2>&1; then
    START="ui"
fi

# ------------------------------------------------------------------------------
# The command. A venv buried in ~/.uclone-x is not somewhere a beginner will type,
# so the published install links `ucx` into ~/.local/bin -- the directory uv and
# pipx use for the same purpose. A checkout keeps its own ./ucx and gets no link.
# An existing ucx there that is not this install's (`uv tool install`, say) is
# left alone: replacing someone's working command is not this script's call.
# ------------------------------------------------------------------------------
LINK_DIR="$HOME/.local/bin"
LINK="$LINK_DIR/ucx"
SHOW_UCX="$UCX"
PATH_HINT="no"
if [ "$SOURCE" != "$REPO_ROOT" ] && [ "$DRY_RUN" = "no" ] && [ -x "$UCX" ]; then
    if [ -e "$LINK" ] && ! { [ -L "$LINK" ] && [ "$(readlink "$LINK")" = "$UCX" ]; }; then
        warn "$LINK already exists and is not this install's; left it alone."
    elif mkdir -p "$LINK_DIR" && ln -sf "$UCX" "$LINK"; then
        case ":$PATH:" in
            *":$LINK_DIR:"*) SHOW_UCX="ucx" ;;
            *)               SHOW_UCX="$LINK"; PATH_HINT="yes" ;;
        esac
    else
        warn "Could not link ucx into $LINK_DIR; use the full path below."
    fi
fi

# ------------------------------------------------------------------------------
# Done.
# ------------------------------------------------------------------------------
step "Done"
say "  core          $CORE_OK"
say "  image engine  $IMAGE_OK"
say "  models        $MODELS_OK"
if [ -n "$RAM" ]; then
    say "  memory        $RAM GB"
    if [ "$RAM" -lt 8 ]; then
        say "                use a cloud model (an API key); local models will not fit well"
    elif [ "$RAM" -lt 16 ]; then
        say "                local model: qwen3:1.7b, or a cloud model; image generation will not fit"
    elif [ "$RAM" -lt 24 ]; then
        say "                local model: qwen3:8b; images at 512px"
    elif [ "$RAM" -lt 32 ]; then
        say "                local model: qwen3:8b; images at the default 768px"
    else
        say "                local model: qwen3:8b and 768px images, both loaded at once"
    fi
fi
say ""
say "Start it with:"
say "    $SHOW_UCX $START"
if [ "$PATH_HINT" = "yes" ]; then
    say ""
    say "To type just \`ucx\`, put $LINK_DIR on your PATH: add this line to"
    say "~/.zshrc (or ~/.bashrc), then open a new terminal:"
    say "    export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
say ""
say "That opens the dashboard at http://127.0.0.1:5180. Everything still missing —"
say "a model, the image engine, credentials — is something you ask for there."
# Only when that command exists: on the published build it does not, and pointing a
# beginner at a command that answers "No such command" is worse than saying nothing.
case "$MODELS_OK" in
    skipped|incomplete)
        say ""
        say "Or fetch the models from here instead:"
        say "    $SHOW_UCX install"
        ;;
esac

# Last, because it does not return: the dashboard runs in this terminal until
# Ctrl-C. Only on a terminal -- a scripted run (CI, --yes under a pipe) that started
# a server would never finish -- and `ucx start` gets the terminal as its stdin, so
# its own questions are asked there rather than read from curl's pipe.
if [ "$DRY_RUN" = "no" ] && [ "$START_MODE" != "no" ] && [ -x "$UCX" ]; then
    say ""
    if ask "Start UClone-X now? [Y/n]" y && [ "$ANSWER" = "y" ]; then
        exec "$UCX" "$START" < "$TTY"
    fi
fi
