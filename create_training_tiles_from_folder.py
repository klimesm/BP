"""
Autor: Matěj Klimeš
Vytvořeno v rámci bakalářské práce na Fsv ČVUT v Praze (2024)
Veškeré informace o práci jsou dostupné z odkazu:https://github.com/klimesm/BP

Pozn. Tento skript vytvoří ze složky obsahující ortofota trénovací a validační dlaždice, které uloží do vybraných složek
"""

import cv2
import numpy as np
import os

def create_training_tiles(ima):
    # rozměry snímku
    height, width = ima.shape[:2]

    # počet řádků a sloupců dlaždic
    num_rows = height // 512
    num_cols = width // 512

    # pole pro dlaždice
    tiles = []

    # snímek na dlaždice
    for i in range(num_rows):
        for j in range(num_cols):
            tile = ima[i * 512:(i + 1) * 512, j * 512:(j + 1) * 512]
            tiles.append(tile)

    return tiles


input_folder = r"C:\skola\BP\data\vstupni_data"
output_folder = r"C:\skola\BP\data\trenovaci_dlazdice"
validation_folder = r"C:\skola\BP\data\validacni_dlazdice"

# Získání seznamu všech souborů ve vstupní složce
image_files = os.listdir(input_folder)

# Iterace přes všechny soubory vstupní složky
for filename in image_files:
    # Načtení obrázku
    ima = cv2.imread(os.path.join(input_folder, filename))
    # menim nazev
    filename = os.path.splitext(filename)[0].split('.')[-1]
    # Rozdělení obrázku na dlaždice
    tiles = create_training_tiles(ima)

    # Iterace přes všechny dlaždice
    for i, tile in enumerate(tiles):
        if i < 55:
            # Uložení dlaždic z každého obrázku jako validační do jiné složky
            if (i == 2+i//9*9) or (i == 6+i//9*9):
                cv2.imwrite(os.path.join(validation_folder, f"{filename}_{i+1}.png"), tile)
            # Uložení dlaždice do výstupní složky s novým názvem
            else:
                cv2.imwrite(os.path.join(output_folder, f"{filename}_{i+1}.png"), tile)
        elif i > 54:
            if i == 59:
                cv2.imwrite(os.path.join(validation_folder, f"{filename}_{i+1}.png"), tile)
            else:
                cv2.imwrite(os.path.join(output_folder, f"{filename}_{i+1}.png"), tile)
