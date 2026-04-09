import base64
import os
import subprocess

import pyperclip


def is_wsl() -> bool:
    return "WSL_DISTRO_NAME" in os.environ


def copy_to_clipboard(text: str) -> None:
    text = str(text)
    if not is_wsl():
        pyperclip.copy(text)
        return

    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    script = (
        "$text = [Text.Encoding]::UTF8.GetString("
        f"[Convert]::FromBase64String('{encoded}'));"
        "Set-Clipboard -Value $text"
    )
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
    )


def paste_from_clipboard() -> str:
    if not is_wsl():
        return pyperclip.paste()

    script = (
        "$text = Get-Clipboard -Raw -ErrorAction SilentlyContinue;"
        "if ([string]::IsNullOrEmpty($text)) { return };"
        "[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($text))"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        check=False,
        capture_output=True,
    )
    encoded = completed.stdout.decode("ascii", errors="ignore").strip()
    return base64.b64decode(encoded).decode("utf-8") if encoded else ""
