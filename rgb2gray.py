"""
Autor: Matěj Klimeš
Vytvořeno v rámci bakalářské práce na Fsv ČVUT v Praze (2024)
Veškeré informace o práci jsou dostupné z odkazu:https://github.com/klimesm/BP

Pozn. Tento skript slouží k převedení barevných testovacích dlaždic do stupňů šedi
"""

import os
import cv2

# vstupní složka
input_folder = r"C:\skola\BP\data\testovaci_dlazdice_vyber"

# výstupní složka
output_folder = r"C:\skola\BP\data\testovaci_dlazdice_vyber_gray"

# Vytvoření výstupní složky, pokud ještě neexistuje
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Procházení všech souborů ve vstupní složce
for filename in os.listdir(input_folder):
    # nacitam jen soubory s příponou jpg nebo png
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Načtení ortofota
        image_rgb = cv2.imread(os.path.join(input_folder, filename))

        # Převod z RGB na stupně šedi
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        # Uložení cernobileho ortofota
        output_path = os.path.join(output_folder, filename.split('.')[0] + '_gray.png')
        cv2.imwrite(output_path, image_gray)

        print(f"Ortofoto '{filename}' bylo úspěšně převedeno na stupně šedi a uloženo jako '{output_path}'.")
