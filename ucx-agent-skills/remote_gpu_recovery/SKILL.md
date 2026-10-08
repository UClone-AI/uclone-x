---
name: remote_gpu_recovery
description: Diagnostic and safe recovery playbooks for remote GPU, Ollama, and ComfyUI services.
version: 0.1.0
origin: human
status: active
content_sha256: 0eeb339253abc7e765e6c8c5245c0625d4d7963e6dcc4cb100080f500fd93ce6
tags:
  - ops
  - gpu
  - comfyui
  - ollama
  - recovery
---

# Remote GPU Worker Recovery Playbooks

This skill provides deterministic, fail-safe procedures to diagnose and recover from remote GPU worker errors (for example a workstation running Ollama and ComfyUI that you reach over SSH).

In the commands below, `gpu-host` stands for the SSH host alias of that machine, and ComfyUI is assumed to be installed in `~/ComfyUI` with its own `venv`. Substitute your own alias and paths.

## ⚠️ Mandatory Execution Rules (Strict Safety for 8B Models)

1. **Virtual Environment Isolation**:
   - **NEVER** run bare `pip install <pkg>` or bare `python3 main.py`.
   - **ALWAYS** use the explicit ComfyUI virtual environment binaries on the remote host:
     - Python: `~/ComfyUI/venv/bin/python3`
     - Pip: `~/ComfyUI/venv/bin/pip`

2. **Daemon Decoupling**:
   - When launching background daemons via SSH, **ALWAYS** redirect `stdin`, `stdout`, and `stderr`:
     ```bash
     nohup ~/ComfyUI/venv/bin/python3 ~/ComfyUI/main.py --listen 127.0.0.1 --port 8188 </dev/null >~/ComfyUI/comfy.log 2>&1 &
     ```
   - Failing to redirect file descriptors will cause SSH sessions to hang indefinitely.

3. **No Destructive Commands**:
   - NEVER execute recursive deletion, administrative superuser escalations, system power actions, or wildcard process kills.

---

## Playbook 1: Ollama Model Not Found (HTTP 404)

### Symptom & Signature
```
uclone_x.errors.LLMProviderError: Ollama streaming returned status 404: {"error":"model '<MODEL_NAME>' not found"}
```

### Root Cause
The active session or clone requested `<MODEL_NAME>`, but the remote Ollama daemon does not have this model pulled in its local storage.

### Recovery Procedure
1. **List installed models on remote host**:
   ```bash
   ssh gpu-host "ollama list"
   ```
2. **Determine the recovery path**:
   - **Path A (Download requested model)**: If the user requires the specific model:
     ```bash
     ssh gpu-host "ollama pull <MODEL_NAME>"
     ```
   - **Path B (Switch to available model)**: If fast response is needed, switch the session model to an already installed model (e.g. `qwen3:8b` or `deepseek-r1:14b`).
3. **Verify resolution**:
   Confirm model presence in `ollama list` and resume turn execution.

---

## Playbook 2: ComfyUI Missing Python Module (`ModuleNotFoundError`)

### Symptom & Signature
```
ModuleNotFoundError: No module named '<MODULE_NAME>'
```
(Found in `~/ComfyUI/comfy.log` or during worker launch).

### Root Cause
A custom node or pipeline requires a third-party Python package that has not been installed into the ComfyUI virtual environment.

### Recovery Procedure
1. **Inspect exact failure line in log**:
   ```bash
   ssh gpu-host "tail -n 25 ~/ComfyUI/comfy.log"
   ```
2. **Install missing package into ComfyUI venv**:
   ```bash
   ssh gpu-host "~/ComfyUI/venv/bin/pip install <MODULE_NAME>"
   ```
3. **Relaunch ComfyUI daemon**:
   ```bash
   ssh gpu-host "nohup ~/ComfyUI/venv/bin/python3 ~/ComfyUI/main.py --listen 127.0.0.1 --port 8188 </dev/null >~/ComfyUI/comfy.log 2>&1 &"
   ```
4. **Verify port 8188 listening**:
   ```bash
   ssh gpu-host "python3 -c 'import socket; s=socket.socket(); print(\"listening:\", s.connect_ex((\"127.0.0.1\", 8188)) == 0); s.close()'"
   ```

---

## Playbook 3: ComfyUI Port 8188 Inactive / Connection Refused

### Symptom & Signature
- Remote GPU status indicates port 8188 is not listening.
- UI displays "Connection refused" when sending image generation tasks.

### Root Cause
ComfyUI daemon crashed or was never launched on the remote machine.

### Recovery Procedure
1. **Check for active ComfyUI processes**:
   ```bash
   ssh gpu-host "ps aux | grep -i '[m]ain.py.*8188'"
   ```
2. **Inspect crash reason**:
   ```bash
   ssh gpu-host "tail -n 30 ~/ComfyUI/comfy.log"
   ```
3. **Clean restart**:
   ```bash
   ssh gpu-host "nohup ~/ComfyUI/venv/bin/python3 ~/ComfyUI/main.py --listen 127.0.0.1 --port 8188 </dev/null >~/ComfyUI/comfy.log 2>&1 &"
   ```
4. **Wait and verify**:
   Wait 3 seconds and verify TCP socket connection on 8188.
