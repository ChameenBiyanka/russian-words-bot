import os, io, sys, json, time, random
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types

# ================== SETTINGS ==================
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-3.5-flash-lite")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "")
POLLINATIONS_KEY = os.getenv("POLLINATIONS_API_KEY", "")
POSTED_FILE = "posted.json"
FB_API = "https://graph.facebook.com/v26.0"
SIZE = 1080

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

CATEGORIES = [
    "fruits", "vegetables", "farm animals", "wild animals", "pets", "birds",
    "insects", "colors in everyday objects", "family and people", "body parts",
    "clothes", "shoes and accessories", "household items", "kitchen items",
    "furniture", "vehicles and transport", "buildings and places", "food and drinks",
    "weather and nature", "jobs and tools", "school and office items",
    "sports equipment", "musical instruments", "flowers and trees",
    "electronics", "toys and games", "sea creatures", "city objects",
]

GEMINI_KEY = os.environ["GEMINI_API_KEY"]
FB_PAGE_ID = os.environ["FB_PAGE_ID"]
FB_PAGE_TOKEN = os.environ["FB_PAGE_TOKEN"]

client = genai.Client(api_key=GEMINI_KEY)


# ================== HELPERS ==================
def load_posted():
    if os.path.exists(POSTED_FILE):
        with open(POSTED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_posted(posted):
    with open(POSTED_FILE, "w", encoding="utf-8") as f:
        json.dump(posted[-3000:], f, ensure_ascii=False)


def ask_gemini(prompt, retries=3):
    for i in range(retries):
        try:
            res = client.models.generate_content(
                model=TEXT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            return json.loads(res.text)
        except Exception as e:
            print(f"Gemini text attempt {i+1} failed: {e}")
            time.sleep(10 * (i + 1))
    return None


def get_word(posted):
    recent = ", ".join(posted[-150:]) or "none"
    for attempt in range(4):
        category = random.choice(CATEGORIES)
        prompt = f"""You are building a Russian-English picture vocabulary for beginners.
Pick ONE common, easy-to-photograph noun from this category: {category}.
Do NOT use any of these already-posted words: {recent}
Return ONLY JSON with these keys:
- "english": the English word, singular, lowercase (example: "apple")
- "russian": the correct Russian word, lowercase, no stress marks
- "transliteration": simple English-friendly pronunciation of the Russian word (example: "YAH-bluh-kuh")
- "example_ru": a short, simple Russian sentence using the word
- "example_en": the English translation of that sentence
- "emoji": one emoji that matches the word
- "image_prompt": English prompt for a bright, high-quality photo showing several of this item (example: "a pile of fresh red apples"), clean simple background, subject centered in the upper two thirds of the frame with empty space at the bottom, no text, no letters, no logos
"""
        data = ask_gemini(prompt)
        if data and data.get("english") and data.get("russian"):
            if data["english"].strip().lower() not in posted:
                return data
        print("Word retry", attempt + 1)
    return None


def make_image(image_prompt):
    # 1) Gemini image model (only if IMAGE_MODEL is set)
    if IMAGE_MODEL:
        try:
            res = client.models.generate_content(
                model=IMAGE_MODEL,
                contents=image_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"]
                ),
            )
            for part in res.candidates[0].content.parts:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    return part.inline_data.data
        except Exception as e:
            print("Gemini image failed:", e)

    # 2) Pollinations (free AI image service)
    for attempt in range(3):
        try:
            seed = random.randint(1, 1000000)
            url = (
                "https://image.pollinations.ai/prompt/" + quote(image_prompt)
                + f"?width={SIZE}&height={SIZE}&nologo=true&seed={seed}"
            )
            headers = {}
            if POLLINATIONS_KEY:
                headers["Authorization"] = f"Bearer {POLLINATIONS_KEY}"
            r = requests.get(url, headers=headers, timeout=120)
            ctype = r.headers.get("content-type", "")
            if r.ok and ctype.startswith("image") and len(r.content) > 5000:
                return r.content
            print("Pollinations problem, status:", r.status_code)
        except Exception as e:
            print(f"Pollinations attempt {attempt+1} failed:", e)
        time.sleep(10)
    return None


def get_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default(size=size)


def fit_font(draw, text, path, max_width, start, min_size=36):
    size = start
    while size > min_size:
        font = get_font(path, size)
        if draw.textlength(text, font=font) <= max_width:
            return font, size
        size -= 4
    return get_font(path, min_size), min_size


def build_card(base_bytes, russian, english, translit):
    img = None
    if base_bytes:
        try:
            img = Image.open(io.BytesIO(base_bytes)).convert("RGB")
            w, h = img.size
            s = min(w, h)
            left, top = (w - s) // 2, (h - s) // 2
            img = img.crop((left, top, left + s, top + s)).resize((SIZE, SIZE))
        except Exception as e:
            print("Image open failed:", e)
            img = None
    if img is None:
        img = Image.new("RGB", (SIZE, SIZE), (24, 40, 72))

    img = img.convert("RGBA")
    band_top = int(SIZE * 0.66)
    overlay = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(
        [0, band_top, SIZE, SIZE], fill=(0, 0, 0, 175)
    )
    img = Image.alpha_composite(img, overlay)

    d = ImageDraw.Draw(img)
    max_w = SIZE - 100
    ru = russian[:1].upper() + russian[1:]
    en = english[:1].upper() + english[1:]
    tr = f"[{translit}]" if translit else ""

    items = []
    f, sz = fit_font(d, ru, FONT_BOLD, max_w, 130)
    items.append((ru, f, sz, (255, 255, 255)))
    f, sz = fit_font(d, en, FONT_BOLD, max_w, 84)
    items.append((en, f, sz, (255, 213, 79)))
    if tr:
        f, sz = fit_font(d, tr, FONT_REG, max_w, 50)
        items.append((tr, f, sz, (210, 210, 210)))

    total = sum(int(sz * 1.25) for _, _, sz, _ in items)
    y = band_top + max(10, (SIZE - band_top - total) // 2)
    for text, font, sz, color in items:
        w = d.textlength(text, font=font)
        d.text(((SIZE - w) / 2, y), text, font=font, fill=color)
        y += int(sz * 1.25)

    out = io.BytesIO()
    img.convert("RGB").save(out, format="JPEG", quality=92)
    return out.getvalue()


# ================== MAIN ==================
def main():
    posted = load_posted()

    word = get_word(posted)
    if not word:
        print("Word generation failed. Exit.")
        sys.exit(1)

    english = word["english"].strip()
    russian = word["russian"].strip()
    translit = word.get("transliteration", "").strip()
    print("Word:", english, "=", russian)

    base = make_image(word.get("image_prompt", f"photo of {english}"))
    if base is None:
        print("AI image failed, using plain card.")

    card = build_card(base, russian, english, translit)

    emoji = word.get("emoji", "")
    caption = (
        f"{emoji} {english.capitalize()} = {russian}\n"
        f"Pronunciation: {translit}\n\n"
        f"Example:\n{word.get('example_ru', '')}\n{word.get('example_en', '')}\n\n"
        f"A new Russian word every 2 hours. Follow to learn Russian daily!\n"
        f"(Image: AI-generated)\n\n"
        f"#LearnRussian #RussianWords #Russian #LanguageLearning #Vocabulary"
    )

    r = requests.post(
        f"{FB_API}/{FB_PAGE_ID}/photos",
        data={"caption": caption, "access_token": FB_PAGE_TOKEN},
        files={"source": ("word.jpg", card, "image/jpeg")},
        timeout=120,
    )
    print("Facebook:", r.status_code, r.text)

    if r.ok:
        posted.append(english.lower())
        save_posted(posted)
        print("Posted successfully.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
