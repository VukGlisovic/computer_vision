"""Web front end for combining AEB brackets.

Serves an HTML form that accepts a bracket of shots, runs the combine CLI against them with
the chosen settings, and returns the image the CLI writes.
"""

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

import aeb

APP_DIR = Path(__file__).parent

app = FastAPI(title="HDR Image Creation")
logger = logging.getLogger("uvicorn.error")
templates = Jinja2Templates(directory=str(APP_DIR))
_last_result: dict[str, bytes | str] = {}
_input_workdir = tempfile.TemporaryDirectory()
_input_dir = Path(_input_workdir.name)


def _uploaded_names() -> list[str]:
    return sorted(p.name for p in _input_dir.iterdir() if p.is_file())


def _render_page(
    request: Request,
    mode: str = "fusion",
    calibration: str = "debevec",
    tonemap: str = "reinhard",
    gamma: float = 2.2,
    align: bool = True,
    max_side: str = "",
    message: str = "",
    error: str = "",
    has_image: bool = False,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html.j2",
        context={
            "modes": ("fusion", "hdr"),
            "calibrators": tuple(aeb.CALIBRATORS),
            "tonemappers": tuple(aeb.TONEMAPPERS),
            "accept": ",".join(sorted(aeb.IMAGE_SUFFIXES)),
            "mode": mode,
            "calibration": calibration,
            "tonemap": tonemap,
            "gamma": gamma,
            "align": align,
            "max_side": max_side,
            "message": message,
            "error": error,
            "has_image": has_image,
            "file_name": str(_last_result.get("file_name", "combined.jpg")),
            "uploaded_files": _uploaded_names(),
        },
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return _render_page(request, has_image=bool(_last_result))


@app.post("/combine", response_class=HTMLResponse)
def combine(
    request: Request,
    uploads: list[UploadFile] = File(default=[]),
    mode: str = Form("fusion"),
    calibration: str = Form("debevec"),
    tonemap: str = Form("reinhard"),
    gamma: float = Form(2.2),
    align: bool = Form(False),
    max_side: str = Form(""),
) -> HTMLResponse:
    limit = max_side.strip()
    if limit and not limit.isdigit():
        return _render_page(
            request, mode, calibration, tonemap, gamma, align, max_side=max_side,
            error="Max side length has to be a positive whole number of pixels.",
        )

    new_uploads = [u for u in uploads if u.filename]
    if new_uploads:
        for existing in _input_dir.iterdir():
            existing.unlink()
        for upload in new_uploads:
            destination = _input_dir / Path(upload.filename).name
            with destination.open("wb") as target:
                shutil.copyfileobj(upload.file, target)

    with tempfile.TemporaryDirectory() as workdir:
        output_dir = Path(workdir) / "output"
        output_dir.mkdir()

        command = [
            sys.executable, "-m", "aeb",
            "-i", str(_input_dir),
            "-o", str(output_dir / "combined.jpg"),
            "-m", mode,
        ]
        if mode == "hdr":
            command += ["-c", calibration, "-t", tonemap, "-g", str(gamma)]
        if not align:
            command.append("--no-align")
        if limit:
            command += ["--max-side", limit]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            error_text = (
                result.stderr.strip()
                or result.stdout.strip()
                or f"aeb exited with code {result.returncode}."
            )
            logger.error(
                "aeb failed (exit %d): %s",
                result.returncode,
                error_text,
            )
            return _render_page(
                request, mode, calibration, tonemap, gamma, align, max_side=max_side, error=error_text
            )

        written = list(output_dir.iterdir())
        if written:
            _last_result["image_bytes"] = written[0].read_bytes()
            _last_result["file_name"] = written[0].name

        return _render_page(
            request,
            mode, calibration, tonemap, gamma, align, max_side,
            message=result.stdout.strip(),
            has_image=bool(_last_result),
        )


@app.get("/result")
def result_image() -> Response:
    file_name = str(_last_result.get("file_name", "combined.jpg"))
    return Response(
        content=bytes(_last_result.get("image_bytes", b"")),
        media_type="image/jpeg",
        headers={"Content-Disposition": f'inline; filename="{file_name}"'},
    )
