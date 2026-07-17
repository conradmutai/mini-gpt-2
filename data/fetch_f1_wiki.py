import wikipediaapi
import zipfile
import os
import re


def clean_filename(name):
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_') + ".txt"


DRIVERS = [
    "Lando Norris", "Oscar Piastri",            # McLaren
    "Charles Leclerc", "Lewis Hamilton",        # Ferrari
    "Max Verstappen", "Isack Hadjar",           # Red Bull
    "George Russell", "Kimi Antonelli",         # Mercedes
    "Fernando Alonso", "Lance Stroll",          # Aston Martin
    "Pierre Gasly", "Franco Colapinto",         # Alpine
    "Esteban Ocon", "Oliver Bearman",           # Haas
    "Liam Lawson", "Arvid Lindblad",            # Racing Bulls
    "Alexander Albon", "Carlos Sainz Jr.",      # Williams
    "Nico Hülkenberg", "Gabriel Bortoleto",     # Sauber/Audi
    "Sergio Pérez", "Valtteri Bottas",          # Cadillac
]

OUTPUT_DIR = "f1_wiki_pages"
ZIP_NAME = "f1_2026_drivers_wikipedia.zip"

wiki = wikipediaapi.Wikipedia(
    user_agent="mini-gpt2-data-collector/1.0 (personal ML project)",
    language="en"
)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fetched = []
    missing = []

    for driver in DRIVERS:
        page = wiki.page(driver)

        if page.exists():
            filepath = os.path.join(OUTPUT_DIR, clean_filename(driver))
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(page.text)
            fetched.append(driver)
            print(f"SAVED: {driver} Wikipedia Page, {len(page.text)} Characters")
        else:
            missing.append(driver)
            print(f"FAILED TO FIND {driver}!")

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(OUTPUT_DIR):
            zf.write(os.path.join(OUTPUT_DIR, fname), fname)

    total_chars = sum(len(open(os.path.join(OUTPUT_DIR, f), encoding="utf-8").read())
                      for f in os.listdir(OUTPUT_DIR))

    print(f"\nDone. {len(fetched)}/{len(DRIVERS)} pages fetched.")
    print(f"Total corpus size: {total_chars:,} characters")
    print(f"Zip created: {ZIP_NAME}")
    if missing:
        print(f"Missing (fix manually): {missing}")


if __name__ == "__main__":
    main()