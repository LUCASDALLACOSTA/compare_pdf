import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
import cv2
from shapely.geometry import Polygon, Point
import shapely.affinity
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import fitz  # PyMuPDF

class PDFVectorComparator:
    def __init__(self, root):
        self.root = root
        self.pdf1_path = None
        self.pdf2_path = None
        self.total_pages = 0
        self.progress_var = tk.DoubleVar(value=0)

        # AutoAligner pour alignement
        self.aligner = AutoAligner()

        # Configuration de l'interface
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Comparateur de PDF")
        self.root.geometry("500x400")

        # Label principal
        main_label = ttk.Label(self.root, text="Outil de comparaison de PDF", font=("Arial", 14, "bold"))
        main_label.pack(pady=10)

        # Cadre pour les sélections de fichiers
        file_frame = ttk.LabelFrame(self.root, text="Sélection des fichiers PDF", padding=10)
        file_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(file_frame, text="Choisir PDF 1", command=self.select_pdf1).grid(row=0, column=0, padx=5, pady=5)
        self.pdf1_label = ttk.Label(file_frame, text="Aucun fichier sélectionné", anchor='w')
        self.pdf1_label.grid(row=0, column=1, sticky='ew', padx=5)

        ttk.Button(file_frame, text="Choisir PDF 2", command=self.select_pdf2).grid(row=1, column=0, padx=5, pady=5)
        self.pdf2_label = ttk.Label(file_frame, text="Aucun fichier sélectionné", anchor='w')
        self.pdf2_label.grid(row=1, column=1, sticky='ew', padx=5)

        # Cadre pour les options d'alignement
        alignment_frame = ttk.LabelFrame(self.root, text="Options d'alignement", padding=10)
        alignment_frame.pack(fill='x', padx=10, pady=5)

        self.auto_align_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(alignment_frame, text="Auto-alignement", variable=self.auto_align_var).pack(side=tk.LEFT, padx=5)

        # Bouton pour réinitialiser l'alignement
        ttk.Button(alignment_frame, text="Réinitialiser l'alignement", command=self.reset_alignment).pack(side=tk.RIGHT, padx=5)

        # Bouton pour lancer la comparaison
        ttk.Button(self.root, text="Lancer la comparaison", command=self.compare_pdfs).pack(pady=10)

        # Barre de progression
        progress_frame = ttk.Frame(self.root)
        progress_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(progress_frame, text="Progression :").pack(side=tk.LEFT, padx=5)
        ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100).pack(side=tk.LEFT, fill='x', expand=True, padx=5)

        # Pied de page
        footer = ttk.Label(self.root, text="Développé pour la comparaison vectorielle", font=("Arial", 10),
                           anchor="center")
        footer.pack(side="bottom", pady=5)

    def select_pdf1(self):
        """Permet de sélectionner le premier fichier PDF."""
        file_path = filedialog.askopenfilename(filetypes=[("Fichiers PDF", "*.pdf")])
        if file_path:
            self.pdf1_path = file_path
            self.pdf1_label.config(text=file_path.split("/")[-1])

    def select_pdf2(self):
        """Permet de sélectionner le second fichier PDF."""
        file_path = filedialog.askopenfilename(filetypes=[("Fichiers PDF", "*.pdf")])
        if file_path:
            self.pdf2_path = file_path
            self.pdf2_label.config(text=file_path.split("/")[-1])

    def compare_pdfs(self):
        """Compare les PDF en mettant en évidence les pixels différents."""
        if not self.pdf1_path or not self.pdf2_path:
            messagebox.showerror("Erreur", "Veuillez sélectionner deux fichiers PDF.")
            return

        try:
            # Charger les documents pour vérifier le nombre de pages
            doc1 = fitz.open(self.pdf1_path)
            doc2 = fitz.open(self.pdf2_path)

            # Vérifier le nombre réel de pages minimum entre les deux documents
            self.total_pages = min(len(doc1), len(doc2))
            if self.total_pages == 0:
                raise ValueError("Un ou plusieurs documents sont vides.")

            for page_num in range(self.total_pages):
                self.progress_var.set((page_num + 1) / self.total_pages * 100)
                self.root.update_idletasks()

                # Comparer les pixels pour la page actuelle
                self.highlight_differences(page_num)

            messagebox.showinfo("Succès", "Comparaison terminée. Résultats sauvegardés.")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la comparaison : {str(e)}")

    def reset_alignment(self):
        """Réinitialise l'alignement à l'identité."""
        self.aligner.transformation_matrix = np.eye(3)
        self.progress_var.set(0)
        messagebox.showinfo("Info", "Alignement réinitialisé.")

    def highlight_differences(self, page_num):
        """Crée une image PNG mettant en évidence les pixels différents entre deux PDF, avec alignement automatique."""
        try:
            # Charger les deux PDF
            doc1 = fitz.open(self.pdf1_path)
            doc2 = fitz.open(self.pdf2_path)

            # Vérifier si la page existe dans les deux documents
            if page_num >= len(doc1) or page_num >= len(doc2):
                print(f"La page {page_num + 1} n'existe pas dans l'un des documents.")
                return

            # Récupérer les images des pages
            page1 = doc1[page_num]
            page2 = doc2[page_num]

            pix1 = page1.get_pixmap()
            pix2 = page2.get_pixmap()

            # Convertir les images en tableaux NumPy
            img1 = np.frombuffer(pix1.samples, dtype=np.uint8).reshape(pix1.height, pix1.width, pix1.n)
            img2 = np.frombuffer(pix2.samples, dtype=np.uint8).reshape(pix2.height, pix2.width, pix2.n)

            # Vérifier si les images ont un canal alpha et le retirer
            if img1.shape[2] == 4:
                img1 = img1[:, :, :3]
            if img2.shape[2] == 4:
                img2 = img2[:, :, :3]

            # Convertir les images en niveaux de gris pour l'alignement
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Détecter et calculer les points-clés et descripteurs
            orb = cv2.ORB_create()
            keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
            keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

            # Correspondance des descripteurs avec BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Extraire les points correspondants
            points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Calculer la transformation homographique
            matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

            if mask.sum() < len(mask) * 0.5:  # Si moins de 50 % des points sont valides
                print("Alignement insuffisant pour la page", page_num + 1)
                return

            # Appliquer la transformation pour aligner img2 sur img1
            height, width = img1.shape[:2]
            aligned_img2 = cv2.warpPerspective(img2, matrix, (width, height))

            # Calculer la différence pixel par pixel
            diff = cv2.absdiff(img1, aligned_img2)

            # Tolérance fixe
            tolerance = 100  # Ajustez cette valeur pour la sensibilité
            diff_mask = np.any(diff > tolerance, axis=2)

            # Mettre en évidence les pixels différents (ici en rouge)
            img_highlighted = np.copy(img1)
            img_highlighted[diff_mask] = [0, 0, 255]  # Pixels différents en rouge

            # Sauvegarde de l'image
            save_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"page_{page_num + 1}_differences.png")
            cv2.imwrite(save_path, img_highlighted)
            print(f"Image sauvegardée : {save_path}")

        except Exception as e:
            print(f"Erreur lors de la génération de l'image : {e}")


class AutoAligner:
    def __init__(self):
        self.transformation_matrix = np.eye(3)  # Matrice d'identité par défaut

    # Méthodes d'alignement (simplifiées pour cette interface)
    def compute_alignment(self, elements1, elements2):
        """Calcule une transformation d'alignement (simulation simplifiée)."""
        self.transformation_matrix = np.eye(3)  # Exemple de transformation
        return self.transformation_matrix


if __name__ == "__main__":
    root = tk.Tk()
    app = PDFVectorComparator(root)
    root.mainloop()