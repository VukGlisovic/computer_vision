# HDR Image Creation

Sometimes you want to create amazing pictures, but unfortunately your camera simply can't handle the dynamic range in the frame you want to capture. A camera records one exposure at a time and that exposure cannot capture both the shadows and the highlights of a high-contrast scene. The shadows fall into underexposure, the highlights into overexposure. Exposure bracketing, including its automatic form (auto exposure bracketing, AEB), takes several shots of the same scene at different exposure times. A short exposure keeps the highlights while a long exposure keeps the shadows. This project combines those shots into one high dynamic range image.


## Background info

### How is a picture captured?

A camera's sensor consists of photodiodes. These photodiodes collect charge in proportion to the light that reaches them during the exposure. It converts light into electric charge, pools that charge into tiny electrical reservoirs, and measures the accumulated voltage.
Each photodiode represents one pixel. When people talk about a bigger sensor, they do not mean more photodiodes, but the photodiodes themselves are bigger. Bigger photodiodes can capture more light before it needs to clip the value (before it reaches overexposure). Additionally, in low light areas it can still capture more photons compared to smaller photodiodes allowing it to exceed the random electrical interference (read noise) more easily.
Dynamic range is essentially the ratio between the maximum amount of light a pixel can hold before clipping and the minimum noise floor.

So comparing for example our mobile phones that have e.g. 48-megapixel cameras and a proper Canon camera with also 48-megapixels, we see that the 48 million photodiodes in the phone are spread over a much smaller area than in the Canon camera. In other words, the sensor is smaller. Because they have much less space, the photodiodes need to be made smaller to fit in the phone camera resulting in a smaller dynamic range.

### How are the captured photons converted to an image?

The photodiodes convert the electric charge they captured into 14-bit (2^14=16,384 steps) through Analog-to-Digital Converter (ADC). It's possible to save these raw 14-bit values, but it would yield very big images even in the 60MB+.
Therefore, cameras often encode the light captured into for example a JPEG file. The camera applies a non-linear camera response function (CRF) that converts the 14-bit information stream into an 8-bit information stream.

### How does the camera response function (CRF) work?

A sensor captures light **linearly**. However, humans perceive light **logarithmically**. We notice the difference between a dark shadow and a slightly brighter shadow much more easily than we notice the difference between two extremely bright highlights. The CRF basically translates the linearly captured photons into a so to say logarithmic scale for humans to observe.

Professional photgraphers usually capture their images in RAW format nonetheless. The camera in that case does not apply its CRF and thus does not throw away information (keeping the 14-bit). This allows them to do professional image editing in e.g. Lightroom or Photoshop.


## Combining images for an HDR image

So in exposure bracketing, we capture multiple images of the same scene with different exposure times. There's many different ways of combining the different images:

1. Fusion blends the well-exposed regions of the shots into a displayable image from the pixel values alone. 
2. HDR mode merges the shots into a radiance map and tonemaps that map for display. 
    a. On encoded shots (e.g. JPEG) it first recovers the CRF from the exposure times in the metadata. 
    b. On RAW images it can skip estimating the CRF (as it was never applied) and directly create the radiance map.


## Running code for creating an HDR image

The task runs from `scripts/`, so paths are relative to that directory. The written file keeps the given path and adds the settings to its name.

```bash
# Fusion, the default. Writes combined_fusion.jpg.
pixi run combine-aeb-images -i burst_dir/ -o combined.jpg

# Radiance map, then tonemap. Writes combined_hdr_debevec_reinhard.jpg.
pixi run combine-aeb-images -i under.jpg normal.jpg over.jpg -m hdr -o combined.jpg
```

Further settings are in the script's help.
