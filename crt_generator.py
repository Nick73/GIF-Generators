# crt_standby_generator.py
# Arch-friendly: uses system packages (python-pillow, python-numpy, python-imageio)
# Generates animated CRT-style “Connecting…” standby in:
#   - 1200x480 (Twitch banner)
#   - 1920x1080 (overlay)
# Cursor REMOVED. Background has CRT-style green glow.

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import numpy as np
import imageio
import random, os

TEXT_LINES = ["Connecting…", "Broadcast will start momentarily"]

# Arch font path first (ttf-dejavu)
FONT_PATHS = [
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",      # Arch
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Debian/Ubuntu
    "/Library/Fonts/Menlo.ttf",                     # macOS
    "C:/Windows/Fonts/consola.ttf",                 # Windows (Consolas)
]

# Colors
GREEN = (0, 255, 120)
BG = (0, 0, 0)

# Sizes
BANNER_SIZE = (1200, 480)       # Twitch banner
OVERLAY_SIZE = (1920, 1080)     # Stream overlay

# Animation
FRAMES = 48                     # 4s loop @ 12 FPS
FPS = 12
DURATION_MS = int(1000 / FPS)

# FX knobs
SCANLINE_OPACITY    = 0.22
BLOOM_RADIUS        = 2.5
BLOOM_GAIN          = 0.75
VIGNETTE_STRENGTH   = 0.35
NOISE_STRENGTH      = 0.06
TEAR_PROB           = 0.5
TEAR_BANDS          = (1, 3)
TEAR_SHIFT_PX       = 16
CHROMA_GLITCH_FRAMES= {0, 1}    # “kick” at loop start
CHROMA_SHIFT_PX     = 1
CURVATURE_K         = 0.08      # set 0.0 to disable
FLICKER_MIN, FLICKER_MAX = 0.95, 1.05

# CRT Background glow
CRT_BG_STRENGTH     = 0.18
CRT_BG_TINT         = (0, 140, 40) 

def load_font(px_size: int):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, px_size)
        except Exception:
            pass
    return ImageFont.load_default()

def center_text_block(draw, lines, font, canvas_w, canvas_h, line_spacing):
    bboxes = [draw.textbbox((0,0), line, font=font) for line in lines]
    heights = [b[3]-b[1] for b in bboxes]
    widths  = [b[2]-b[0] for b in bboxes]
    total_h = sum(heights) + line_spacing*(len(lines)-1)
    y = (canvas_h - total_h)//2
    x_list = [(canvas_w - w)//2 for w in widths]
    return x_list, y, widths, heights

def apply_scanlines(img, opacity=0.2):
    w, h = img.size
    arr = np.array(img).astype(np.float32) / 255.0
    mask = np.ones((h, 1), dtype=np.float32)
    mask[1::2, 0] = 1.0 - opacity
    arr *= mask[:, None, :]
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr*255).astype(np.uint8))

def add_bloom_simple(base, radius=2.0, gain=0.75):
    glow = base.filter(ImageFilter.GaussianBlur(radius))
    return Image.blend(base, glow, gain)

def vignette(img, strength=0.3):
    w, h = img.size
    y, x = np.ogrid[:h, :w]
    cx, cy = w/2, h/2
    r = np.sqrt((x-cx)**2 + (y-cy)**2)
    r /= r.max()
    mask = (1 - strength * (r**1.5))
    arr = np.array(img).astype(np.float32)/255.0
    arr *= mask[..., None]
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr*255).astype(np.uint8))

def add_noise(img, strength=0.05):
    arr = np.array(img).astype(np.float32)/255.0
    noise = np.random.uniform(-strength, strength, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr*255).astype(np.uint8))

def barrel_distort(img, k=0.08):
    if k <= 0:
        return img
    w, h = img.size
    cx, cy = (w-1)/2, (h-1)/2
    src = np.array(img)
    out = np.zeros_like(src)
    ys, xs = np.indices((h, w))
    x = (xs - cx) / cx
    y = (ys - cy) / cy
    r2 = x*x + y*y
    factor = 1 + k * r2
    xd = x / factor
    yd = y / factor
    xs_src = (xd * cx + cx).astype(np.float32)
    ys_src = (yd * cy + cy).astype(np.float32)
    xs_nn = np.clip(np.rint(xs_src), 0, w-1).astype(np.int32)
    ys_nn = np.clip(np.rint(ys_src), 0, h-1).astype(np.int32)
    out[:] = src[ys_nn, xs_nn]
    return Image.fromarray(out)

def chroma_shift(img, shift_px=1):
    if shift_px <= 0:
        return img
    r, g, b = img.split()
    r = ImageChops.offset(r, -shift_px, 0)
    b = ImageChops.offset(b, shift_px, 0)
    return Image.merge("RGB", (r, g, b))

def tearing(img, max_shift=12, bands=(1,3)):
    if max_shift <= 0:
        return img
    w, h = img.size
    base = img.copy()
    for _ in range(random.randint(bands[0], bands[1])):
        th = random.randint(4, max(6, int(h*0.06)))
        ty = random.randint(0, h-th)
        band = base.crop((0, ty, w, ty+th))
        shift = random.randint(-max_shift, max_shift)
        band = ImageChops.offset(band, shift, 0)
        base.paste(band, (0, ty))
    return base

def flicker(img, low=0.95, high=1.05):
    factor = random.uniform(low, high)
    arr = np.array(img).astype(np.float32)/255.0
    arr = np.clip(arr * factor, 0, 1)
    return Image.fromarray((arr*255).astype(np.uint8))

def crt_background(size, tint=(0,180,60), strength=0.35):
    """Return a black base with a soft green radial glow, to mimic warmed phosphor."""
    w, h = size
    base = Image.new("RGB", (w, h), BG)
    # radial gradient mask
    y, x = np.ogrid[:h, :w]
    cx, cy = w/2, h/2
    r = np.sqrt((x-cx)**2 + (y-cy)**2)
    r /= r.max()
    mask = (1 - (r**1.8)) * strength  # center bright, edge dark
    # apply tint
    arr = np.zeros((h, w, 3), dtype=np.float32)
    arr[..., 0] = (tint[0]/255.0) * mask
    arr[..., 1] = (tint[1]/255.0) * mask
    arr[..., 2] = (tint[2]/255.0) * mask
    arr = np.clip(arr, 0, 1)
    glow = Image.fromarray((arr*255).astype(np.uint8))
    return ImageChops.add(base, glow)

def render_sequence(out_path, size, font_px):
    w, h = size
    font = load_font(font_px)
    frames = []

    for i in range(FRAMES):
        # Start with CRT background
        frame = crt_background(size, tint=CRT_BG_TINT, strength=CRT_BG_STRENGTH)

        # Draw centered two-line text (NO cursor)
        draw = ImageDraw.Draw(frame)
        lines = [TEXT_LINES[0], TEXT_LINES[1]]
        line_spacing = int(font_px * 0.45)
        x_list, y, widths, heights = center_text_block(draw, lines, font, w, h, line_spacing)

        draw.text((x_list[0], y), lines[0], font=font, fill=GREEN)
        y += heights[0] + line_spacing
        draw.text((x_list[1], y), lines[1], font=font, fill=GREEN)

        # Bloom
        frame = Image.blend(frame, frame.filter(ImageFilter.GaussianBlur(BLOOM_RADIUS)), BLOOM_GAIN)

        # Glitch kick at loop start
        if i in CHROMA_GLITCH_FRAMES:
            frame = chroma_shift(frame, CHROMA_SHIFT_PX)
            frame = tearing(frame, max_shift=TEAR_SHIFT_PX, bands=(2,4))
        else:
            if random.random() < TEAR_PROB:
                frame = tearing(frame, max_shift=TEAR_SHIFT_PX//2, bands=TEAR_BANDS)

        # Scanlines, vignette, noise, flicker, curvature
        frame = apply_scanlines(frame, opacity=SCANLINE_OPACITY)
        frame = vignette(frame, strength=VIGNETTE_STRENGTH)
        frame = add_noise(frame, strength=NOISE_STRENGTH)
        frame = flicker(frame, FLICKER_MIN, FLICKER_MAX)
        frame = barrel_distort(frame, CURVATURE_K)

        frames.append(frame)

    imageio.mimsave(out_path, [np.array(f) for f in frames], duration=DURATION_MS/1000.0, loop=0)

def main():
    os.makedirs(".", exist_ok=True)
    render_sequence("connecting_banner_1200x480.gif", BANNER_SIZE, font_px=52)
    render_sequence("connecting_overlay_1920x1080.gif", OVERLAY_SIZE, font_px=84)
    print("Done:")
    print(" - connecting_banner_1200x480.gif")
    print(" - connecting_overlay_1920x1080.gif")

if __name__ == "__main__":
    main()
