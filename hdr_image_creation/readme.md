# HDR Image Creation

Sometimes you want to create amazing pictures, but unfortunately your camera simply can't handle the dynamic range in the frame you want to capture. In other words, it can't capture both the dark (undersatured) and the light (oversaturated) areas. In order to capture the perfect photo that captures both the dark and light areas nonetheless, a solution has been designed called (auto) expose bracketing (AEB). This methodology takes multiple photos of the same scene, but with different exposure times. This way, short exposure times can capture light areas and long exposure times can capture dark areas. After having collected the photos at different exposure times, you need to combine them into a single image. That's what this project is about: high dynamic range (HDR) image creation.


## Image capture background information
When taking pictures with a camera, what happens is the following:
1. photodiodes count photons hitting the diodes.
2. a camera response function (CRF) transforms the counted photons for each wavelength (RGB) non-linearly into an image.
