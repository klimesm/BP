"""
Autor: Matěj Klimeš
Vytvořeno v rámci bakalářské práce na Fsv ČVUT v Praze (2024)
Veškeré informace o práci jsou dostupné z odkazu:https://github.com/klimesm/BP
Zvolený postup byl vybrán na základě článku: https://www.mdpi.com/1862998
Některé části skriptu a předtrénovaný model byly převzaty od autorů článku
Původní skripty jsou dostupné z odkazu: https://github.com/3DOM-FBK/Hyper_U_Net

Pozn. Tento skript je určen k obarvení ortofota pomocí předtrénované sítě Hyper-U-Net
      K použití je nutné mít ve stejné složce i soubor obsahující neuronovou síť (Hyper_U_NET_p.h5)
"""

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model


def LAB22RGB(L, a, b):
    # prevede barvy z sLABu do RGB
    a11 = 0.299
    a12 = 0.587
    a13 = 0.114
    a21 = (0.15 / 0.234)
    a22 = (-0.234 / 0.234)
    a23 = (0.084 / 0.234)
    a31 = (0.287 / 0.785)
    a32 = (0.498 / 0.785)
    a33 = (-0.785 / 0.785)

    aa = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    C0 = np.zeros((L.shape[0], 3))
    C0[:, 0] = L[:, 0]
    C0[:, 1] = a[:, 0]
    C0[:, 2] = b[:, 0]
    C = np.transpose(C0)

    X = np.linalg.inv(aa).dot(C)
    X1D = np.reshape(X, (X.shape[0] * X.shape[1], 1))
    p0 = np.where(X1D < 0)
    X1D[p0[0]] = 0
    p1 = np.where(X1D > 1)
    X1D[p1[0]] = 1
    Xr = np.reshape(X1D, (X.shape[0], X.shape[1]))

    Rr = Xr[0][:]
    Gr = Xr[1][:]
    Br = Xr[2][:]

    R = np.uint(np.round(Rr * 255))
    G = np.uint(np.round(Gr * 255))
    B = np.uint(np.round(Br * 255))
    return R, G, B

def create_tiles(ima):
    # Vytvori dlazdice z vetsiho ortofota
    # rozměry snímku
    height, width = ima.shape[:2]

    # počet řádků a sloupců dlaždic
    if height % 512 !=0:
        num_rows = height // 512 +1
    else:
        num_rows = height // 512
    if width % 512 !=0:
        num_cols = width // 512 + 1
    else:
        num_cols = width // 512

    # pole pro dlaždice
    tiles = []

    # snímek na dlaždice
    for i in range(num_rows):
        for j in range(num_cols):
            tile = ima[i*512:(i+1)*512, j*512:(j+1)*512]
            if tile.shape[0] < 512 or tile.shape[1] < 512:
                # když nevychází dlaždice na 512 x 512, doplní se nulami na tyto rozměry
                padded_tile = np.zeros((512, 512, 3), dtype=np.uint8)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile
            tiles.append(tile)

    return tiles

def color_tile(ima, model):
    # obarvi dlazdici
    sz0 = ima.shape[0]
    sz1 = ima.shape[1]
    bands = 2

    L0 = ima[:, :, 0]
    L = L0 / 255

    ima_gray = np.reshape(L, (1, sz0, sz1, 1))
    predicted = model.predict(ima_gray, verbose=0)

    predicted = np.reshape(predicted, (sz0 * sz1, bands))
    predicted = np.reshape(predicted, (sz0 * sz1, bands))
    a0 = predicted[:, 0:1]
    b0 = predicted[:, 1:2]
    Lr = np.reshape(L, (sz0 * sz1, 1))

    Rr, Gr, Br = LAB22RGB(Lr, a0, b0)
    Rr = np.reshape(Rr, (sz0, sz1))
    Gr = np.reshape(Gr, (sz0, sz1))
    Br = np.reshape(Br, (sz0, sz1))
    predicted255 = np.uint8(np.zeros((sz0, sz1, 3)))
    predicted255[:, :, 0] = Rr
    predicted255[:, :, 1] = Gr
    predicted255[:, :, 2] = Br
    return predicted255

def reconstruct_image(tiles, original_shape):
    # z dlazdic zpetne slozi puvodni rozmer

    # Získat rozměry ortofota
    height, width = original_shape[:2]
    # Vypočítat počet řádků a sloupců dlaždic

    if height % 512 !=0:
        num_rows = height // 512 +1
    else:
        num_rows = height // 512
    if width % 512 !=0:
        num_cols = width // 512 + 1
    else:
        num_cols = width // 512
    # Rozmer spojenych dlazdic
    shape = (num_rows*512, num_cols*512,3)
    # Vytvořit prázdnou matici pro rekonstruované ortofoto
    reconstructed_image = np.zeros(shape, dtype=np.uint8)

    # Spojit dlaždice
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            reconstructed_image[i*512:(i+1)*512, j*512:(j+1)*512] = tiles[idx]
            idx += 1
    # Oříznout ortofoto na rozměry původního
    cropped_image = reconstructed_image[:height, :width]
    return cropped_image


class obarveni_ortofota:
    def __init__(self, root):
        self.root = root
        self.root.title("Obarvení ortofota")

        self.original_image_path = None
        self.modified_image_path = None
        self.colored_image = None
        # Vytvoření rámců
        self.frame1 = tk.Frame(root)
        self.frame2 = tk.Frame(root)
        self.frame3 = tk.Frame(root)
        self.frame4 = tk.Frame(root)
        # Zobrazení rámce 1
        self.create_frame_1()
    def destroy_frames(self):
        if self.frame1.winfo_exists():
            self.frame1.destroy()
        if self.frame2.winfo_exists():
            self.frame2.destroy()
        if self.frame3.winfo_exists():
            self.frame3.destroy()
        if self.frame4.winfo_exists():
            self.frame4.destroy()
    def create_frame_1(self):
        self.destroy_frames()
        root.minsize(600, 400)  # minimální velikost okna
        self.frame1 = tk.Frame(root)
        self.frame1.pack(padx=10, pady=10)
        self.frame1.config(width=600, height=300)
        # Rámec 1 - vytvoření tlačítek
        self.button_1o = tk.Button(self.frame1, text="Obarvení ortofota", command=self.create_frame_2)
        self.info_label = tk.Label(self.frame1, text = "Toto GUI slouží k ovládání programu pro obarvení černobílého ortofota pomocí sítě Hyper-U-Net.",anchor="w",justify="left")
        self.info_label2 = tk.Label(self.frame1, text="Poznámky k použití: \n \nObarvení ortofot o větších rozměrech může na PC s nižším výpočetním výkonem trvat i několik minut. \nBěhem procesu barvení nezavírejte aplikaci.\nPrincip funkčnosti je popsán v bakalářské práci dostupné na GitHubu (https://github.com/klimesm/BP).\nPřípadné dotazy směřujte na autora: Matěj Klimeš (klimesmatej01@gmail.com).",anchor="w",justify="left")
        self.exit_button = tk.Button(self.frame1, text="Ukončit program", command=self.root.quit)
        self.info_label.pack(side=tk.TOP, padx=10, pady=10)
        self.button_1o.pack(side=tk.TOP, padx=10, pady=10)
        self.exit_button.pack(side=tk.TOP, padx=10, pady=10)
        self.info_label2.pack(side=tk.BOTTOM, padx=10, pady=50)
    def create_frame_2(self):
        self.destroy_frames()
        root.minsize(1150, 650)  # minimální velikost
        self.frame2 = tk.Frame(root)
        self.frame2.pack(padx=10, pady=10)
        # Rámec 2 - vytvoření tlačítek
        self.select_button = tk.Button(self.frame2, text="Načíst ortofoto", command=self.select_image)
        self.edit_button = tk.Button(self.frame2, text="Obarvit ortofoto", state=tk.DISABLED, command=self.edit_image)
        self.export_button = tk.Button(self.frame2, text="Exportovat obarvené ortofoto", state=tk.DISABLED, command=self.export_image)
        self.back_button = tk.Button(self.frame2, text="Zpět na hlavní okno", command=self.create_frame_1)
        self.exit_button = tk.Button(self.frame2, text="Ukončit program", command=self.root.quit)
        # Rámec 2 - rozložení prvků
        self.select_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.edit_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.export_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.back_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.exit_button.pack(side=tk.LEFT, padx=10, pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("*.png;*.jpg;*.jpeg; *.tiff;*.bmp;*.gif", "*.png;*.jpg;*.jpeg;*.tiff;*.bmp;*.gif")])

        if file_path:
            self.original_image_path = file_path
            # Zobrazení původního ortofota
            self.frame3.destroy()
            self.frame4.destroy()
            self.frame3 = tk.Frame(root)
            self.frame4 = tk.Frame(root)
            self.frame3.pack(padx=5, pady=5)
            self.frame4.pack(padx=5, pady=5)
            image = Image.open(file_path)
            image.thumbnail((512, 512))
            self.photo = ImageTk.PhotoImage(image)
            self.original_label = tk.Label(self.frame3, text="Původní ortofoto")
            self.original_image_label = tk.Label(self.frame4,image=self.photo)
            self.original_image_label.pack(side=tk.LEFT, padx=10)
            self.original_label.pack(side=tk.LEFT, padx=512/2-20)

            # Aktivovat tlačítko pro úpravu
            self.edit_button.config(state=tk.NORMAL)
            # Deaktivovat tlačítko pro export
            self.export_button.config(state=tk.DISABLED)
    def edit_image(self):
        if self.original_image_path:
            cwd = os.getcwd()
            # Načtení předtrénovaného modelu
            model1 = load_model(cwd + '/Hyper_U_NET_p.h5')
            # Načíst původní ortofoto
            ima = cv2.imread(self.original_image_path)
            # Vytvořit dlaždice
            tiles = create_tiles(ima)
            colored_tiles = []

            for i in range(len(tiles)):
                # Obarvení dlaždic
                colored_tile = color_tile(tiles[i],model1)
                colored_tiles.append(colored_tile)
            # rekonstrukce ortofota
            self.colored_image = reconstruct_image(colored_tiles,ima.shape)
            # Převod BGR na RGB
            modified_image = cv2.cvtColor(self.colored_image, cv2.COLOR_BGR2RGB)
            modified_image = Image.fromarray(modified_image)
            modified_image.thumbnail((512, 512))
            self.preview2 = ImageTk.PhotoImage(modified_image)
            # Zobrazení obarveného ortofota
            self.modified_label = tk.Label(self.frame3, text="Obarvené ortofoto")
            self.modified_image_label = tk.Label(self.frame4,image=self.preview2)
            self.modified_label.pack(side=tk.LEFT, padx=512/2-20)
            self.modified_image_label.pack(side=tk.LEFT, padx=10)

            # Aktivovat tlačítko pro export
            self.export_button.config(state=tk.NORMAL)
            # Deaktivovat tlačítko pro obarvení
            self.edit_button.config(state=tk.DISABLED)

    def export_image(self):
        original_image_name = self.original_image_path.split('/')[-1].split('.')[0]  # Získání názvu bez přípony
        export_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                   filetypes=[("Soubory PNG", "*.png"), ("Všechny soubory", "*.*")],
                                                   initialfile=f"{original_image_name}_rgb.png")
        if export_path:
            cv2.imwrite(export_path, self.colored_image)


if __name__ == "__main__":
    root = tk.Tk()
    app = obarveni_ortofota(root)
    root.mainloop()