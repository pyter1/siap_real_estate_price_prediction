from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import csv
import re
import time
import os

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL = (
    "https://www.nekretnine.rs/stambeni-objekti/stanovi/"
    "izdavanje-prodaja/prodaja/grad/beograd/lista/po-stranici/20/"
)

OUTPUT_FILE = "nekretnine_dataset.csv"
DELAY = 1  # seconds

START_INDEX = 6908  # RESUME FROM ADVERT 559 (0-based)

# -----------------------------
# HELPERS
# -----------------------------
def parse_price(price_text: str):
    """
    Extract numeric EUR price as float.
    Returns None if price is missing.
    """
    if not price_text:
        return None

    first_line = price_text.splitlines()[0]
    digits = re.findall(r"\d+", first_line)

    if not digits:
        return None

    return float("".join(digits))


# -----------------------------
# DRIVER SETUP (FAST)
# -----------------------------
chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--log-level=3")

prefs = {
    "profile.managed_default_content_settings.images": 2,
    "profile.managed_default_content_settings.stylesheets": 2,
    "profile.managed_default_content_settings.fonts": 2,
}
chrome_options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 6)

# -----------------------------
# STEP 1: COLLECT ALL ADVERT LINKS
# -----------------------------
all_links = []
page = 1

while True:
    if page == 1:
        url = BASE_URL
    else:
        url = BASE_URL + f"stranica/{page}/"

    print(f"\nLoading listing page {page}")
    driver.get(url)

    try:
        wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.advert-list")
            )
        )
    except:
        print("No advert list found — stopping pagination.")
        break

    links = [
        a.get_attribute("href")
        for a in driver.find_elements(
            By.CSS_SELECTOR,
            "div.advert-list div.row.offer h2.offer-title a"
        )
    ]

    links = [l for l in links if l]

    if not links:
        print("No adverts on this page — finished.")
        break

    print(f"Found {len(links)} adverts")
    all_links.extend(links)

    page += 1

# Remove duplicates
all_links = list(dict.fromkeys(all_links))
print(f"\nTOTAL adverts collected: {len(all_links)}")

# -----------------------------
# STEP 2: OPEN CSV IN APPEND MODE
# -----------------------------
file_exists = os.path.exists(OUTPUT_FILE)

csv_file = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

# Write header ONLY if file is new
if not file_exists:
    csv_writer.writerow([
        "Price_EUR",
        "Square_footage",
        "Number_of_rooms",
        "State",
        "Lift",
        "Heating",
        "Optical_internet",
        "Parking",
        "Floor",
        "Street"
    ])

# -----------------------------
# STEP 3: SCRAPE DETAIL PAGES (RESUME)
# -----------------------------
for i, link in enumerate(all_links[START_INDEX:], start=START_INDEX + 1):
    print(f"\n[{i}/{len(all_links)}] Opening advert")

    try:
        driver.get(link)

        # Wait once for details
        detalji_section = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "section#detalji")
            )
        )

        # PRICE
        raw_price_text = driver.find_element(
            By.CSS_SELECTOR, "h4.stickyBox__price"
        ).text.strip()
        price = parse_price(raw_price_text)

        # LOCATION / STREET
        location_text = driver.find_element(
            By.CSS_SELECTOR, "h3.stickyBox__Location"
        ).text.strip()

        street = (
            location_text.split(",")[-1].strip()
            if "," in location_text
            else location_text
        )

        # Defaults
        square_footage = None
        num_rooms = None
        state = None
        floor = "0"
        lift = 0
        heating = None
        optical_internet = 0
        parking = 0

        for div in detalji_section.find_elements(
            By.CSS_SELECTOR, "div.property__amenities"
        ):
            header = div.find_element(By.TAG_NAME, "h3").text.strip()

            if header == "Podaci o nekretnini":
                for li in div.find_elements(By.TAG_NAME, "li"):
                    text = li.text.strip()
                    if text.startswith("Kvadratura:"):
                        square_footage = li.find_element(
                            By.TAG_NAME, "strong"
                        ).text.strip()
                    elif text.startswith("Ukupan broj soba:"):
                        num_rooms = li.find_element(
                            By.TAG_NAME, "strong"
                        ).text.strip()
                    elif text.startswith("Stanje nekretnine:"):
                        state = li.find_element(
                            By.TAG_NAME, "strong"
                        ).text.strip()
                    elif text.startswith("Spratnost:"):
                        floor = li.find_element(
                            By.TAG_NAME, "strong"
                        ).text.strip()

            elif header == "Dodatna opremljenost":
                text = div.text
                lift = 1 if "Lift" in text else 0
                parking = 1 if (
                    "Garažno mesto" in text
                    or "Spoljno parking mesto" in text
                ) else 0

            elif header == "Tehnička opremljenost":
                optical_internet = 1 if "Optička mreža" in div.text else 0

            elif header == "Ostalo":
                for li in div.find_elements(By.TAG_NAME, "li"):
                    if "Grejanje:" in li.text:
                        heating = li.text.split("Grejanje:")[-1].strip()

        csv_writer.writerow([
            price,
            square_footage,
            num_rooms,
            state,
            lift,
            heating,
            optical_internet,
            parking,
            floor,
            street
        ])

        print(
            f"Saved: price={price}, m2={square_footage}, "
            f"rooms={num_rooms}, floor={floor}, street={street}"
        )

    except Exception as e:
        print(f"Skipping advert {i}: {e}")
        continue

# -----------------------------
# CLEANUP
# -----------------------------
csv_file.close()
driver.quit()

print("\nSCRAPING FINISHED SUCCESSFULLY")
