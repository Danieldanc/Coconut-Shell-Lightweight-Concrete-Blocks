# ============================================================================
#  SVR (RBF) - Compressive Strength Prediction GUI
#  Coconut Shell Concrete (Alccofine blended)  |  54 specimens embedded
#
#  HOW TO RUN (Visual Studio Code):
#    1) Install the two required libraries once, in the VS Code terminal:
#           pip install numpy scikit-learn
#       (tkinter ships with standard Python on Windows/macOS)
#    2) Open this file in VS Code and press  Run  (the play button)  or  F5.
#    3) The window opens, the SVR model trains automatically on the embedded
#       data, and you can enter mix proportions to predict strength.
# ============================================================================

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ----------------------------------------------------------------------------
#  EMBEDDED DATASET  (W/C ratio, Cement, Alccofine, Coarse agg., Coconut shell, fc)
#  Units: kg/m3 (except W/C ratio = dimensionless, fc = MPa = N/mm2)
# ----------------------------------------------------------------------------
DATA = [
    (0.520, 376.2, 3.8, 1100.0, 0.0, 7.94),
    (0.530, 376.2, 3.8, 1045.0, 55.0, 8.09),
    (0.530, 376.2, 3.8, 990.0, 110.0, 7.76),
    (0.540, 376.2, 3.8, 935.0, 165.0, 7.16),
    (0.540, 376.2, 3.8, 880.0, 220.0, 6.67),
    (0.550, 376.2, 3.8, 825.0, 275.0, 6.01),
    (0.550, 376.2, 3.8, 770.0, 330.0, 5.48),
    (0.510, 372.4, 7.6, 1100.0, 0.0, 8.28),
    (0.520, 372.4, 7.6, 1045.0, 55.0, 8.43),
    (0.520, 372.4, 7.6, 990.0, 110.0, 8.09),
    (0.530, 372.4, 7.6, 935.0, 165.0, 7.46),
    (0.530, 372.4, 7.6, 880.0, 220.0, 6.95),
    (0.540, 372.4, 7.6, 825.0, 275.0, 6.27),
    (0.540, 372.4, 7.6, 770.0, 330.0, 5.71),
    (0.500, 368.6, 11.4, 1100.0, 0.0, 8.62),
    (0.510, 368.6, 11.4, 1045.0, 55.0, 8.78),
    (0.510, 368.6, 11.4, 990.0, 110.0, 8.42),
    (0.520, 368.6, 11.4, 935.0, 165.0, 7.77),
    (0.520, 368.6, 11.4, 880.0, 220.0, 7.24),
    (0.530, 368.6, 11.4, 825.0, 275.0, 6.53),
    (0.530, 368.6, 11.4, 770.0, 330.0, 5.95),
    (0.490, 364.8, 15.2, 1100.0, 0.0, 8.98),
    (0.490, 364.8, 15.2, 1045.0, 55.0, 9.36),
    (0.500, 364.8, 15.2, 990.0, 110.0, 8.77),
    (0.510, 364.8, 15.2, 935.0, 165.0, 8.09),
    (0.510, 364.8, 15.2, 880.0, 220.0, 7.54),
    (0.520, 364.8, 15.2, 825.0, 275.0, 6.80),
    (0.520, 364.8, 15.2, 770.0, 330.0, 6.20),
    (0.480, 361.0, 19.0, 1100.0, 0.0, 9.34),
    (0.490, 361.0, 19.0, 1045.0, 55.0, 9.40),
    (0.490, 361.0, 19.0, 990.0, 110.0, 9.13),
    (0.500, 361.0, 19.0, 935.0, 165.0, 8.42),
    (0.500, 361.0, 19.0, 880.0, 220.0, 7.85),
    (0.510, 361.0, 19.0, 825.0, 275.0, 7.07),
    (0.510, 361.0, 19.0, 770.0, 330.0, 6.45),
    (0.600, 376.2, 3.8, 1100.0, 0.0, 6.58),
    (0.580, 376.2, 3.8, 1100.0, 0.0, 6.90),
    (0.600, 372.4, 7.6, 1100.0, 0.0, 6.70),
    (0.580, 368.6, 11.4, 1100.0, 0.0, 7.14),
    (0.570, 364.8, 15.2, 1100.0, 0.0, 7.44),
    (0.600, 376.2, 3.8, 990.0, 110.0, 6.58),
    (0.600, 372.4, 7.6, 990.0, 110.0, 6.70),
    (0.580, 368.6, 11.4, 990.0, 110.0, 7.15),
    (0.550, 364.8, 15.2, 990.0, 110.0, 7.80),
    (0.520, 361.0, 19.0, 990.0, 110.0, 8.51),
    (0.600, 376.2, 3.8, 880.0, 220.0, 5.80),
    (0.600, 372.4, 7.6, 880.0, 220.0, 5.90),
    (0.600, 368.6, 11.4, 880.0, 220.0, 6.00),
    (0.580, 364.8, 15.2, 880.0, 220.0, 6.40),
    (0.550, 361.0, 19.0, 880.0, 220.0, 6.98),
    (0.600, 376.2, 3.8, 770.0, 330.0, 4.88),
    (0.600, 372.4, 7.6, 770.0, 330.0, 4.96),
    (0.600, 368.6, 11.4, 770.0, 330.0, 5.05),
    (0.580, 364.8, 15.2, 770.0, 330.0, 5.38),
]

FEATURES = [
    ("X01", "W/C ratio",        "(-)",       0.52),
    ("X02", "Cement",           "(kg/m\u00b3)", 376.2),
    ("X03", "Alccofine",        "(kg/m\u00b3)", 3.8),
    ("X04", "Coarse aggregate", "(kg/m\u00b3)", 1100.0),
    ("X05", "Coconut shell",    "(kg/m\u00b3)", 0.0),
]

# Tuned SVR (RBF) hyper-parameters (from the study's hyper-parameter search)
SVR_PARAMS = dict(kernel="rbf", C=2582.485084490615,
                  gamma=0.03470490935437504, epsilon=0.007644457600399739)

# ----------------------------------------------------------------------------
#  Colour palette (Modern Professional)
# ----------------------------------------------------------------------------
NAVY   = "#0f2847"
WHITE  = "#ffffff"
LIGHT_BG = "#f8f9fb"
PRIMARY = "#2563eb"
ACCENT = "#dc2626"
SUCCESS = "#16a34a"
TEXT_DARK = "#1f2937"
TEXT_LIGHT = "#6b7280"
BORDER = "#e5e7eb"


def train_model():
    """Train SVR on the embedded data and return (deploy_model, metrics)."""
    arr = np.array(DATA, dtype=float)
    X, y = arr[:, :5], arr[:, 5]

    # 80 / 20 split (seed 42) -> reproduces the reported metrics
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=42)
    split_model = Pipeline([("sc", StandardScaler()), ("m", SVR(**SVR_PARAMS))])
    split_model.fit(Xtr, ytr)

    def trio(a, b):
        return (r2_score(a, b),
                float(np.sqrt(mean_squared_error(a, b))),
                mean_absolute_error(a, b))

    tr = trio(ytr, split_model.predict(Xtr))
    te = trio(yte, split_model.predict(Xte))
    metrics = {"train_r2": tr[0], "train_rmse": tr[1], "train_mae": tr[2],
               "test_r2": te[0], "test_rmse": te[1], "test_mae": te[2]}

    # Final deployment model: refit on ALL 54 specimens for best predictions
    deploy = Pipeline([("sc", StandardScaler()), ("m", SVR(**SVR_PARAMS))])
    deploy.fit(X, y)
    return deploy, metrics


# Function removed - strength class no longer displayed


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Compressive Strength Predictor")
        self.configure(bg=LIGHT_BG)
        self.resizable(False, False)
        self.geometry("600x650")

        self.model, self.metrics = train_model()
        self.entries = {}

        self._build_header()
        self._build_inputs()
        self._build_output()
        self._build_buttons()
        self.load_example()
        self.calculate()

    # ---------------- header ----------------
    def _build_header(self):
        head = tk.Frame(self, bg=NAVY, height=100)
        head.pack(fill="x", padx=0, pady=0)
        head.pack_propagate(False)
        
        tk.Label(head, text="Compressive Strength Predictor", bg=NAVY, fg=WHITE,
                 font=("Segoe UI", 26, "bold")).pack(pady=(20, 8))
        tk.Label(head, text="AI-Powered Material Analysis", bg=NAVY, fg="#93c5fd",
                 font=("Segoe UI", 11)).pack(pady=(0, 16))

    # ---------------- inputs ----------------
    def _build_inputs(self):
        box = tk.Frame(self, bg=LIGHT_BG)
        box.pack(fill="both", expand=True, padx=20, pady=(16, 12))
        
        title = tk.Label(box, text="Mix Design Parameters", bg=LIGHT_BG, fg=TEXT_DARK,
                        font=("Segoe UI", 12, "bold"))
        title.pack(anchor="w", pady=(0, 12))
        
        for i, (tag, name, unit, default) in enumerate(FEATURES):
            # Create a frame for each parameter
            row = tk.Frame(box, bg=WHITE, relief="solid", bd=1, highlightthickness=0)
            row.pack(fill="x", pady=6)
            
            # Label with parameter name and unit
            label_text = f"{name} {unit}"
            tk.Label(row, text=label_text, bg=WHITE, fg=TEXT_DARK, width=28,
                     anchor="w", font=("Segoe UI", 10), padx=12, pady=8).pack(side="left")
            
            # Entry field with modern styling
            e = tk.Entry(row, width=12, justify="center", font=("Segoe UI", 11),
                         relief="flat", bd=0, bg="#f3f4f6", fg=TEXT_DARK, insertbackground=PRIMARY)
            e.pack(side="right", padx=12, pady=8)
            self.entries[name] = e

    # ---------------- output ----------------
    def _build_output(self):
        box = tk.Frame(self, bg=PRIMARY, relief="solid", bd=0, highlightthickness=0)
        box.pack(fill="x", padx=20, pady=(0, 16))
        
        inner = tk.Frame(box, bg=PRIMARY, padx=20, pady=20)
        inner.pack(fill="both", expand=True)
        
        tk.Label(inner, text="Predicted Compressive Strength", bg=PRIMARY, fg="#e0e7ff",
                 font=("Segoe UI", 11)).pack(anchor="w")
        
        self.pred_var = tk.StringVar(value="--")
        tk.Label(inner, textvariable=self.pred_var, bg=PRIMARY, fg=WHITE,
                 font=("Segoe UI", 48, "bold")).pack(anchor="w", pady=(6, 0))
        
        tk.Label(inner, text="MPa", bg=PRIMARY, fg="#e0e7ff",
                 font=("Segoe UI", 13)).pack(anchor="w")

    # ---------------- buttons ----------------
    def _build_buttons(self):
        bar = tk.Frame(self, bg=LIGHT_BG)
        bar.pack(fill="x", padx=20, pady=(0, 20))
        
        tk.Button(bar, text="Calculate", bg=PRIMARY, fg=WHITE,
                  font=("Segoe UI", 11, "bold"), bd=0, padx=16, pady=10,
                  activebackground="#1d4ed8", relief="flat", cursor="hand2",
                  command=self.calculate).pack(side="left", expand=True, fill="x", padx=4)
        tk.Button(bar, text="Reset", bg=ACCENT, fg=WHITE,
                  font=("Segoe UI", 11, "bold"), bd=0, padx=16, pady=10,
                  activebackground="#b91c1c", relief="flat", cursor="hand2",
                  command=self.reset).pack(side="left", expand=True, fill="x", padx=4)
        tk.Button(bar, text="Example", bg=SUCCESS, fg=WHITE,
                  font=("Segoe UI", 11, "bold"), bd=0, padx=16, pady=10,
                  activebackground="#15803d", relief="flat", cursor="hand2",
                  command=self.load_example).pack(side="left", expand=True, fill="x", padx=4)

    # ---------------- actions ----------------
    def calculate(self):
        try:
            x = [float(self.entries[name].get()) for (_, name, _, _) in FEATURES]
        except ValueError:
            messagebox.showerror("Invalid Input",
                                 "Please enter a valid number in every field.")
            return
        fc = float(self.model.predict(np.array([x]))[0])
        self.pred_var.set(f"{fc:.2f}")

    def reset(self):
        for (_, name, _, _) in FEATURES:
            self.entries[name].delete(0, "end")
        self.pred_var.set("--")

    def load_example(self):
        for (_, name, _, default) in FEATURES:
            self.entries[name].delete(0, "end")
            self.entries[name].insert(0, str(default))
        self.calculate()


if __name__ == "__main__":
    App().mainloop()
