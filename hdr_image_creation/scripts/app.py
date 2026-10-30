"""Streamlit front end for combine-aeb-images.py.

Collects a bracket of shots from the browser, runs the combine script against them with the
chosen settings, and shows the image the script writes.
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

SCRIPT_PATH = Path(__file__).parent / "combine-aeb-images.py"

# combine-aeb-images.py is loaded as a module, rather than imported by name, because its
# hyphenated filename is not a valid module name. This keeps the choices offered below, and the
# file extensions accepted by the uploader, drawn from the script that actually runs them instead
# of a second list that could drift from it. The module's __main__ guard keeps this from parsing
# command-line arguments.
_spec = importlib.util.spec_from_file_location("combine_aeb_images", SCRIPT_PATH)
combine_aeb_images = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(combine_aeb_images)

st.set_page_config(page_title="HDR Image Creation")
st.title("HDR Image Creation")

# Mode sits outside the form so changing it redraws the form before submit. Fusion never reads
# calibration, tonemap, or gamma, so those controls are only offered for hdr.
mode = st.selectbox("Mode", ("fusion", "hdr"))

with st.form("bracket_form"):
    uploads = st.file_uploader(
        "Bracket shots",
        type=sorted(suffix.lstrip(".") for suffix in combine_aeb_images.IMAGE_SUFFIXES),
        accept_multiple_files=True,
    )
    if mode == "hdr":
        calibration = st.selectbox("Calibration", tuple(combine_aeb_images.CALIBRATORS))
        tonemap = st.selectbox("Tonemap", tuple(combine_aeb_images.TONEMAPPERS))
        gamma = st.number_input("Gamma", value=2.2, min_value=0.1)
    align = st.checkbox("Align shots", value=True)
    submitted = st.form_submit_button("Combine")

if submitted:
    with tempfile.TemporaryDirectory() as workdir:
        input_dir = Path(workdir) / "input"
        output_dir = Path(workdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        for upload in uploads or []:
            (input_dir / upload.name).write_bytes(upload.getvalue())

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
            st.session_state["combine_result"] = None
            st.error(result.stderr.strip() or "combine-aeb-images.py failed with no error output.")
        else:
            written = list(output_dir.iterdir())
            st.session_state["combine_result"] = {
                "image_bytes": written[0].read_bytes(),
                "file_name": written[0].name,
                "log": result.stdout.strip(),
            } if written else None

combine_result = st.session_state.get("combine_result")
if combine_result:
    st.success(combine_result["log"])
    st.image(combine_result["image_bytes"])
    st.download_button(
        "Download combined image",
        data=combine_result["image_bytes"],
        file_name=combine_result["file_name"],
    )
