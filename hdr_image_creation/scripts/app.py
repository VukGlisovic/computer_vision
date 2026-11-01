"""Web front end for combine-aeb-images.py.

Serves an HTML form that accepts a bracket of shots, runs the combine script against them with
the chosen settings, and returns the image the script writes.
"""

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

SCRIPT_DIR = Path(__file__).parent
SCRIPT_PATH = SCRIPT_DIR / "combine-aeb-images.py"

# combine-aeb-images.py is loaded as a module, rather than imported by name, because its
# hyphenated filename is not a valid module name. This keeps the choices offered below, and the
# file extensions accepted by the uploader, drawn from the script that actually runs them instead
# of a second list that could drift from it. The module's __main__ guard keeps this from parsing
# command-line arguments.
_spec = importlib.util.spec_from_file_location("combine_aeb_images", SCRIPT_PATH)
combine_aeb_images = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(combine_aeb_images)

app = FastAPI(title="HDR Image Creation")
templates = Jinja2Templates(directory=str(SCRIPT_DIR))
_last_result: dict[str, bytes | str] = {}


def _render_page(
    request: Request,
    mode: str = "fusion",
    calibration: str = "debevec",
    tonemap: str = "reinhard",
    gamma: float = 2.2,
    align: bool = True,
    message: str = "",
    error: str = "",
    has_image: bool = False,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "modes": ("fusion", "hdr"),
            "calibrators": tuple(combine_aeb_images.CALIBRATORS),
            "tonemappers": tuple(combine_aeb_images.TONEMAPPERS),
            "accept": ",".join(sorted(combine_aeb_images.IMAGE_SUFFIXES)),
            "mode": mode,
            "calibration": calibration,
            "tonemap": tonemap,
            "gamma": gamma,
            "align": align,
            "message": message,
            "error": error,
            "has_image": has_image,
            "file_name": str(_last_result.get("file_name", "combined.jpg")),
        },
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return _render_page(request, has_image=bool(_last_result))


@app.post("/combine", response_class=HTMLResponse)
def combine(
    request: Request,
    uploads: list[UploadFile] = File(...),
    mode: str = Form("fusion"),
    calibration: str = Form("debevec"),
    tonemap: str = Form("reinhard"),
    gamma: float = Form(2.2),
    align: bool = Form(False),
) -> HTMLResponse:
    with tempfile.TemporaryDirectory() as workdir:
        input_dir = Path(workdir) / "input"
        output_dir = Path(workdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        for upload in uploads:
            if upload.filename:
                destination = input_dir / Path(upload.filename).name
                with destination.open("wb") as target:
                    shutil.copyfileobj(upload.file, target)

        command = [
            sys.executable, str(SCRIPT_PATH),
            "-i", str(input_dir),
            "-o", str(output_dir / "combined.jpg"),
            "-m", mode,
        ]
        if mode == "hdr":
            command += ["-c", calibration, "-t", tonemap, "-g", str(gamma)]
        if not align:
            command.append("--no-align")

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            error_text = result.stderr.strip() or "combine-aeb-images.py failed with no error output."
            return _render_page(request, mode, calibration, tonemap, gamma, align, error=error_text)

        written = list(output_dir.iterdir())
        if written:
            _last_result["image_bytes"] = written[0].read_bytes()
            _last_result["file_name"] = written[0].name

        return _render_page(
            request,
            mode, calibration, tonemap, gamma, align,
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
