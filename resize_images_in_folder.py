"""
Autor: Matěj Klimeš
Vytvořeno v rámci bakalářské práce na Fsv ČVUT v Praze (2024)
Veškeré informace o práci jsou dostupné z odkazu:https://github.com/klimesm/BP

Pozn. Tento skript slouží k převzorkování stažených ortofot na velikost 5000x4000 pixelů
"""

import os
import cv2

def resize_images_in_folder(input_folder, output_folder):
    try:
        # kontrola existence vstupní a výstupní složky
        if not os.path.exists(input_folder):
            print(f"Složka {input_folder} neexistuje.")
            return
        if not os.path.exists(output_folder):
            print(f"Složka {output_folder} neexistuje. Vytvářím novou složku.")
            os.makedirs(output_folder)

        # Načtení seznamu souborů vstupní složky
        files = os.listdir(input_folder)

        # cyklus pres vsechny soubory ve slozce
        for file in files:
            # jen obrazova data
            if file.endswith((".jpg", ".jpeg", ".png", ".tiff")):
                # Načtení ortofota
                img_path = os.path.join(input_folder, file)
                img = cv2.imread(img_path)

                # Zmenšení ortofota, nejblizsí soused
                resized_img = cv2.resize(img, dsize=(5000, 4000), interpolation=cv2.INTER_NEAREST)

                # Uložení zmenšeného ortofota
                output_path = os.path.join(output_folder, f"resized_{os.path.splitext(file)[0]}.png")
                cv2.imwrite(output_path, resized_img)

                print(f"Uložen zmenšený obrázek: {output_path}")

    except Exception as e:
        print(f"Chyba: {e}")


input_folder = r"C:\skola\BP\data\vybrane_souc"
output_folder = r"C:\skola\BP\data\vybrane_souc_prevzorkovane"

resize_images_in_folder(input_folder, output_folder)


