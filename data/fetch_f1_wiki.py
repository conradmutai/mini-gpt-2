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

CIRCUITS = [
    "Albert Park Circuit",
    "Shanghai International Circuit",
    "Suzuka International Racing Course",
    "Bahrain International Circuit",
    "Jeddah Corniche Circuit",
    "Miami International Autodrome",
    "Circuit de Barcelona-Catalunya",
    "Circuit Gilles Villeneuve",
    "Circuit de Monaco",
    "Red Bull Ring",
    "Silverstone Circuit",
    "Circuit de Spa-Francorchamps",
    "Hungaroring",
    "Circuit Zandvoort",
    "Monza Circuit",
    "Baku City Circuit",
    "Marina Bay Street Circuit",
    "Circuit of the Americas",
    "Autódromo Hermanos Rodríguez",
    "Autódromo José Carlos Pace",
    "Las Vegas Strip Circuit",
    "Lusail International Circuit",
    "Yas Marina Circuit",
    "Madring",
]

SEASONS = [f"{year} Formula One World Championship" for year in range(2010, 2026)]

F2_DRIVERS = [
    "Sebastián Montoya", "Mari Boya",
    "Martinius Stenshorne", "Alexander Dunne",
    "Kush Maini", "Tasanapol Inthraphuvasak",
    "Cian Shields", "Nicolás Varrone",
    "Rafael Villagómez", "Laurens van Hoepen",
    "John Bennett", "Rafael Câmara",
    "Joshua Dürksen", "Nikola Tsolov",
    "Dino Beganovic", "Noel León",
    "Gabriele Minì",
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

    # for driver in DRIVERS:
    #     page = wiki.page(driver)
    #
    #     if page.exists():
    #         filepath = os.path.join(OUTPUT_DIR, clean_filename(driver))
    #         with open(filepath, "w", encoding="utf-8") as f:
    #             f.write(page.text)
    #         fetched.append(driver)
    #         print(f"SAVED: {driver} Wikipedia Page, {len(page.text)} Characters")
    #     else:
    #         missing.append(driver)
    #         print(f"FAILED TO FIND {driver}!")

    for f2_driver in F2_DRIVERS:
        page = wiki.page(f2_driver)

        if page.exists():
            filepath = os.path.join(OUTPUT_DIR, clean_filename(f2_driver))
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(page.text)
            fetched.append(f2_driver)
            print(f"SAVED: {f2_driver} Wikipedia Page, {len(page.text)} Characters")
        else:
            missing.append(f2_driver)
            print(f"FAILED TO FIND {f2_driver}!")

    for circuit in CIRCUITS:
        page = wiki.page(circuit)

        if page.exists():
            filepath = os.path.join(OUTPUT_DIR, clean_filename(circuit))
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(page.text)
            fetched.append(circuit)
            print(f"SAVED: {circuit} Wikipedia Page, {len(page.text)} Characters")
        else:
            missing.append(circuit)
            print(f"FAILED TO FIND {circuit}!")

    for season in SEASONS:
        page = wiki.page(season)

        if page.exists():
            filepath = os.path.join(OUTPUT_DIR, clean_filename(season))
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(page.text)
            fetched.append(season)
            print(f"SAVED: {season} Wikipedia Page, {len(page.text)} Characters")
        else:
            missing.append(season)
            print(f"FAILED TO season {season}!")

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(OUTPUT_DIR):
            zf.write(os.path.join(OUTPUT_DIR, fname), fname)

    total_chars = sum(len(open(os.path.join(OUTPUT_DIR, f), encoding="utf-8").read())
                      for f in os.listdir(OUTPUT_DIR))

    print(f"\nDone. {len(fetched)}/{len(DRIVERS) + len(F2_DRIVERS) + len(SEASONS) + len(CIRCUITS)} pages fetched.")
    print(f"Total corpus size: {total_chars:,} characters")
    print(f"Zip created: {ZIP_NAME}")
    if missing:
        print(f"Missing (fix manually): {missing}")


if __name__ == "__main__":
    main()