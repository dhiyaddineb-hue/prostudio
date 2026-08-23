"""Launch the ProStudio web UI."""

from __future__ import annotations

import os

import uvicorn

from youtube_auto_dub.ffmpeg_bin import ensure_ffmpeg_on_path


def main() -> None:
    ensure_ffmpeg_on_path()
    host = os.environ.get("PROSTUDIO_HOST", "0.0.0.0")
    port = int(os.environ.get("PROSTUDIO_PORT", "8080"))
    uvicorn.run(
        "web.app:app",
        host=host,
        port=port,
        reload=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
