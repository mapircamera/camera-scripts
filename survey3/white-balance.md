---
description: Correct the RAW image color balance of the Survey3 RGB images
---

# White Balance

White balancing is typically only done on the **RGB** filter models.

When a RAW file is converted to TIFF it will be overly green due to the image
sensor's Bayer filter's RGGB pattern. To correct the colors and make it less
green you need to apply a white balance to the images.

You can apply white balance corrections using a parameter during the conversion
of RAW+JPG to TIFF using the "Convert RAW+JPG to TIFF" script.

{% content-ref url="convert-raw-jpg-to-tiff.md" %}
[convert-raw-jpg-to-tiff.md](convert-raw-jpg-to-tiff.md)
{% endcontent-ref %}
