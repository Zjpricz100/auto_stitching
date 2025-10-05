# Homography Correspondence Tool
#
# This script provides a graphical user interface (GUI) to select corresponding
# points between two images. These points are essential for computing a
# homography matrix, which can be used to warp one image onto the other.

import tkinter as tk
from tkinter import filedialog, messagebox
# --- FIX: Import ImageOps to handle EXIF orientation ---
from PIL import Image, ImageTk, ImageOps
import numpy as np

class HomographyTool:
    """
    A GUI tool for selecting corresponding points in two images to compute homography.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Homography Point Selector")

        # --- State Variables ---
        self.left_image_path = None
        self.right_image_path = None
        self.left_photo_image = None
        self.right_photo_image = None
        self.left_img_display = None
        self.right_img_display = None
        self.left_original_image = None
        self.right_original_image = None

        self.points1 = []
        self.points2 = []

        self.temp_point = None
        self.is_left_turn = True

        # --- GUI Layout ---
        self.main_frame = tk.Frame(root, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.top_frame = tk.Frame(self.main_frame)
        self.top_frame.pack(fill=tk.X, pady=(0, 10))

        self.instruction_label = tk.Label(self.top_frame, text="Load two images to begin.", font=("Helvetica", 12))
        self.instruction_label.pack(side=tk.LEFT, expand=True)
        
        self.point_counter_label = tk.Label(self.top_frame, text="Pairs: 0", font=("Helvetica", 12))
        self.point_counter_label.pack(side=tk.RIGHT)

        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(1, weight=1)
        self.canvas_frame.rowconfigure(0, weight=1)

        self.canvas_left = tk.Canvas(self.canvas_frame, bg="gray90", highlightthickness=1, highlightbackground="gray70")
        self.canvas_left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.canvas_left.bind("<Button-1>", self.on_canvas_click)

        self.canvas_right = tk.Canvas(self.canvas_frame, bg="gray90", highlightthickness=1, highlightbackground="gray70")
        self.canvas_right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.canvas_right.bind("<Button-1>", self.on_canvas_click)
        
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.btn_load_left = tk.Button(self.button_frame, text="Load Left Image", command=lambda: self.load_image('left'))
        self.btn_load_left.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))
        
        self.btn_load_right = tk.Button(self.button_frame, text="Load Right Image", command=lambda: self.load_image('right'))
        self.btn_load_right.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))

        self.btn_reset = tk.Button(self.button_frame, text="Reset Points", command=self.reset_points, state=tk.DISABLED)
        self.btn_reset.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))

        self.btn_save = tk.Button(self.button_frame, text="Save Points", command=self.save_points, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        self.root.geometry("1000x600")
        self.root.bind('<Configure>', self.on_resize)

    def load_image(self, side):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )
        if not path:
            return

        if side == 'left':
            self.left_image_path = path
            self.canvas_left.delete("all")
        else:
            self.right_image_path = path
            self.canvas_right.delete("all")

        self.reset_points()
        self.display_image(side)

        if self.left_image_path and self.right_image_path:
            self.update_instructions()
            self.btn_reset.config(state=tk.NORMAL)

    def display_image(self, side):
        canvas = self.canvas_left if side == 'left' else self.canvas_right
        path = self.left_image_path if side == 'left' else self.right_image_path

        if not path:
            return

        original_img = Image.open(path)
        
        # --- FIX: Read and apply EXIF orientation tag from the image file ---
        original_img = ImageOps.exif_transpose(original_img)
        
        img_w, img_h = original_img.size
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            self.root.after(50, lambda: self.display_image(side))
            return

        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized_img = original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo_image = ImageTk.PhotoImage(resized_img)

        canvas.delete("all")
        img_display = canvas.create_image(canvas_w / 2, canvas_h / 2, image=photo_image, anchor=tk.CENTER)

        if side == 'left':
            self.left_photo_image = photo_image
            self.left_img_display = img_display
            self.left_original_image = original_img
        else:
            self.right_photo_image = photo_image
            self.right_img_display = img_display
            self.right_original_image = original_img

    def on_canvas_click(self, event):
        if not (self.left_image_path and self.right_image_path):
            return

        canvas = event.widget
        side = 'left' if canvas == self.canvas_left else 'right'

        if (self.is_left_turn and side == 'left') or (not self.is_left_turn and side == 'right'):
            original_coords = self.canvas_to_image_coords(event.x, event.y, side)
            if original_coords is None: return

            if self.is_left_turn:
                self.temp_point = original_coords
                self.draw_point(event.x, event.y, side, len(self.points1))
                self.is_left_turn = False
            else:
                self.points1.append(self.temp_point)
                self.points2.append(original_coords)
                self.draw_point(event.x, event.y, side, len(self.points2) - 1)
                self.is_left_turn = True
                self.temp_point = None

            self.update_instructions()
            self.update_point_counter()

    def canvas_to_image_coords(self, canvas_x, canvas_y, side):
        canvas = self.canvas_left if side == 'left' else self.canvas_right
        img_display_id = self.left_img_display if side == 'left' else self.right_img_display
        original_img = self.left_original_image if side == 'left' else self.right_original_image

        if not all([img_display_id, original_img]):
            return None

        try:
            bbox = canvas.bbox(img_display_id)
            if bbox is None: return None
            x1, y1, x2, y2 = bbox
        except Exception:
            return None
        
        displayed_w = x2 - x1
        displayed_h = y2 - y1
        original_w = original_img.width
        original_h = original_img.height

        if not (x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2) or displayed_w == 0 or displayed_h == 0:
            return None
            
        image_x = (canvas_x - x1) * (original_w / displayed_w)
        image_y = (canvas_y - y1) * (original_h / displayed_h)

        return (image_x, image_y)

    def draw_point(self, x, y, side, index):
        canvas = self.canvas_left if side == 'left' else self.canvas_right
        color = "cyan"
        radius = 5
        
        canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            fill=color, outline="black", width=1, tags=f"point_{index}"
        )
        canvas.create_text(
            x, y, text=str(index + 1), fill="black",
            font=("Helvetica", 10, "bold"), tags=f"point_{index}"
        )

    def on_resize(self, event):
        if self.left_image_path:
            self.display_image('left')
        if self.right_image_path:
            self.display_image('right')
        self.root.after(50, self.redraw_all_points)

    def redraw_all_points(self):
        all_points = list(self.points1)
        if self.temp_point and not self.is_left_turn:
            all_points.append(self.temp_point)
        
        for i, point in enumerate(all_points):
            coords = self.image_to_canvas_coords(point[0], point[1], 'left')
            if coords:
                self.draw_point(coords[0], coords[1], 'left', i)

        for i, point in enumerate(self.points2):
            coords = self.image_to_canvas_coords(point[0], point[1], 'right')
            if coords:
                self.draw_point(coords[0], coords[1], 'right', i)

    def image_to_canvas_coords(self, image_x, image_y, side):
        canvas = self.canvas_left if side == 'left' else self.canvas_right
        img_display_id = self.left_img_display if side == 'left' else self.right_img_display
        original_img = self.left_original_image if side == 'left' else self.right_original_image

        if not all([img_display_id, original_img]):
            return None
        
        try:
            bbox = canvas.bbox(img_display_id)
            if bbox is None: return None
            x1, y1, x2, y2 = bbox
        except Exception:
            return None
        
        displayed_w = x2 - x1
        displayed_h = y2 - y1
        original_w = original_img.width
        original_h = original_img.height

        if original_w == 0 or original_h == 0:
            return None
        
        canvas_x = (image_x * (displayed_w / original_w)) + x1
        canvas_y = (image_y * (displayed_h / original_h)) + y1
        
        return (canvas_x, canvas_y)

    def reset_points(self):
        self.points1.clear()
        self.points2.clear()
        self.temp_point = None
        self.is_left_turn = True
        
        self.canvas_left.delete("all")
        self.canvas_right.delete("all")
        
        if self.left_image_path:
            self.display_image('left')
        if self.right_image_path:
            self.display_image('right')

        self.update_instructions()
        self.update_point_counter()
        self.btn_save.config(state=tk.DISABLED)

    def save_points(self):
        if len(self.points1) < 4:
            messagebox.showwarning("Not Enough Points", "You need to select at least 4 point pairs to compute a homography.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy Array File", "*.npz")],
            title="Save Points"
        )
        if not filepath:
            return

        np.savez(filepath, points1=np.array(self.points1), points2=np.array(self.points2))
        messagebox.showinfo("Success", f"Points saved successfully to {filepath}")

    def update_instructions(self):
        if not (self.left_image_path and self.right_image_path):
            self.instruction_label.config(text="Load two images to begin.")
            return
        
        if self.is_left_turn:
            turn = "LEFT"
            num = len(self.points1) + 1
        else:
            turn = "RIGHT"
            num = len(self.points2) + 1
            
        self.instruction_label.config(text=f"Click point #{num} on the {turn} image.")

    def update_point_counter(self):
        num_pairs = len(self.points2)
        self.point_counter_label.config(text=f"Pairs: {num_pairs}")

        if num_pairs >= 4:
            self.btn_save.config(state=tk.NORMAL)
        else:
            self.btn_save.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = HomographyTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()