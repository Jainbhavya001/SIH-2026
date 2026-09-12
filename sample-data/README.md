# Sample data

Drop demo document images here for testing the pipeline locally (kept out
of git — real or synthetic ID documents shouldn't be committed to a repo).

A quick way to generate a synthetic passport-like test image with a valid,
checksummed MRZ (useful for demoing Module 1/2 without a real document):

```python
from PIL import Image, ImageDraw, ImageFont

img = Image.new("RGB", (900, 600), color=(245, 245, 240))
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("cour.ttf", 28)  # any monospace font

draw.text((30, 30), "REPUBLIC OF UTOPIA - PASSPORT", fill=(0, 0, 0), font=font)
draw.text((30, 80), "Name: ANNA MARIA ERIKSSON", fill=(0, 0, 0), font=font)

# The canonical ICAO 9303 worked example — every check digit validates.
line1 = "P<UTOERIKSSON<<ANNA<MARIA".ljust(44, "<")
line2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
draw.text((30, 480), line1, fill=(0, 0, 0), font=font)
draw.text((30, 520), line2, fill=(0, 0, 0), font=font)

img.save("sample-data/synthetic_passport.png")
```

Run this through `POST /api/documents/scan` with `document_type=passport`
(with Tesseract installed) to see all four modules light up on a document
that is internally consistent — a good baseline before testing tampered
inputs.
