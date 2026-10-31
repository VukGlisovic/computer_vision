"""Web front end for combine-aeb-images.py.

Serves an HTML form that accepts a bracket of shots, runs the combine script against them with
the chosen settings, and returns the image the script writes.
"""

import html
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, Response

SCRIPT_PATH = Path(__file__).parent / "combine-aeb-images.py"

# combine-aeb-images.py is loaded as a module, rather than imported by name, because its
# hyphenated filename is not a valid module name. This keeps the choices offered below, and the
# file extensions accepted by the uploader, drawn from the script that actually runs them instead
# of a second list that could drift from it. The module's __main__ guard keeps this from parsing
# command-line arguments.
_spec = importlib.util.spec_from_file_location("combine_aeb_images", SCRIPT_PATH)
combine_aeb_images = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(combine_aeb_images)

app = FastAPI(title="HDR Image Creation")
_last_result: dict[str, bytes | str] = {}


def _options(values: tuple[str, ...], selected: str) -> str:
    return "".join(
        f'<option value="{v}"{" selected" if v == selected else ""}>{v}</option>'
        for v in values
    )


def _render_page(
    mode: str = "fusion",
    calibration: str = "debevec",
    tonemap: str = "reinhard",
    gamma: float = 2.2,
    align: bool = True,
    message: str = "",
    error: str = "",
    has_image: bool = False,
) -> str:
    accept = ",".join(sorted(combine_aeb_images.IMAGE_SUFFIXES))
    hdr_display = "block" if mode == "hdr" else "none"
    status_html = ""
    if error:
        status_html = f'<p class="error">{html.escape(error)}</p>'
    elif message:
        status_html = f'<p class="success">{html.escape(message)}</p>'

    result_html = ""
    if has_image:
        file_name = html.escape(str(_last_result.get("file_name", "combined.jpg")))
        result_html = (
            f'<div class="result">'
            f'<a class="button" href="/result" download="{file_name}">Download {file_name}</a>'
            f'<img src="/result" alt="Combined image">'
            f'</div>'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HDR Image Creation</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 540px; margin: 1.5rem auto; padding: 0 1rem; }}
  label {{ display: block; margin-top: 1rem; font-weight: 600; }}
  select, input[type="number"], input[type="file"], button, .button {{
    width: 100%; box-sizing: border-box; margin-top: 0.35rem; padding: 0.65rem; font-size: 1rem;
  }}
  .checkbox {{ display: flex; align-items: center; gap: 0.5rem; margin-top: 1rem; font-weight: 600; }}
  button, .button {{
    display: inline-block; text-align: center; text-decoration: none;
    margin-top: 1.25rem; background: #1f6feb; color: #fff; border: 0; border-radius: 6px; cursor: pointer;
  }}
  .error {{ background: #ffebe9; color: #a40e26; padding: 0.75rem; border-radius: 6px; }}
  .success {{ background: #dafbe1; color: #1a7f37; padding: 0.75rem; border-radius: 6px; }}
  .result img {{ width: 100%; height: auto; margin-top: 1rem; border-radius: 6px; }}
</style>
</head>
<body>
<h1>HDR Image Creation</h1>
{status_html}
<form method="post" action="/combine" enctype="multipart/form-data">
  <label>Mode
    <select name="mode" onchange="document.getElementById('hdr-fields').style.display = this.value === 'hdr' ? 'block' : 'none'">
      {_options(("fusion", "hdr"), mode)}
    </select>
  </label>
  <label>Bracket shots
    <input type="file" name="uploads" accept="{accept}" multiple required>
  </label>
  <div id="hdr-fields" style="display: {hdr_display}">
    <label>Calibration
      <select name="calibration">{_options(tuple(combine_aeb_images.CALIBRATORS), calibration)}</select>
    </label>
    <label>Tonemap
      <select name="tonemap">{_options(tuple(combine_aeb_images.TONEMAPPERS), tonemap)}</select>
    </label>
    <label>Gamma
      <input type="number" name="gamma" value="{gamma}" min="0.1" step="0.1">
    </label>
  </div>
  <label class="checkbox">
    <input type="checkbox" name="align" value="true"{" checked" if align else ""}> Align shots
  </label>
  <button type="submit">Combine</button>
</form>
{result_html}
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _render_page(has_image=bool(_last_result))


@app.post("/combine", response_class=HTMLResponse)
def combine(
    uploads: list[UploadFile] = File(...),
    mode: str = Form("fusion"),
    calibration: str = Form("debevec"),
    tonemap: str = Form("reinhard"),
    gamma: float = Form(2.2),
    align: bool = Form(False),
) -> str:
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
            return _render_page(mode, calibration, tonemap, gamma, align, error=error_text)

        written = list(output_dir.iterdir())
        if written:
            _last_result["image_bytes"] = written[0].read_bytes()
            _last_result["file_name"] = written[0].name

        return _render_page(
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
