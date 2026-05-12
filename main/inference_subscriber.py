# GMR UIN R1A - MQTT Subscriber
# Update patch: 2026-05-12

import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import pandas as pd
import paho.mqtt.client as mqtt
import os
import joblib
import numpy as np

MQTT_BROKER       = "localhost"
MQTT_PORT         = 1883
MQTT_TOPIC_DATA   = "gmr/data"
MQTT_TOPIC_STATUS = "gmr/status"
MQTT_CLIENT_ID    = "GMR-Subscriber"

# ============================================================
# KALIBRASI / KONVERSI
# ============================================================
def tegangan_ke_b(v):
    return 5.3381 * v - 4.2983

# ============================================================
# MODEL SVR TERSIMPAN
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)

SVR_MODEL_CANDIDATES = [
    os.path.join(REPO_ROOT, "models", "regression", "SVR", "reg_svr.pkl"),
    os.path.join(REPO_ROOT, "models", "regression", "SVR", "regressor_svr.pkl"),
    os.path.join(REPO_ROOT, "models", "regression", "SVM", "reg_svr.pkl"),
]

svr_model = None

CONCENTRATIONS = [5, 10, 20, 30, 40, 50]

# Kalibrasi untuk scaling output prediksi
# Tegangan max: 0.92V → B_max = 5.3381*0.92 - 4.2983 ≈ 0.597 mT
# Konsentrasi target max: >50 mg/mL
MAX_VOLTAGE = 0.92
TARGET_CONC_MAX = 50
B_MAX = tegangan_ke_b(MAX_VOLTAGE)  # Calculate max B value
CONC_PRED_CEILING = None  # Will be set after model load

def load_svr_model():
    global svr_model, CONC_PRED_CEILING
    for path in SVR_MODEL_CANDIDATES:
        if os.path.exists(path):
            try:
                svr_model = joblib.load(path)
                # Estimate ceiling: prediksi pada B_MAX
                x_test = np.array([[B_MAX]], dtype=float)
                y_ceiling = svr_model.predict(x_test)
                CONC_PRED_CEILING = float(y_ceiling[0])
                return path
            except Exception as e:
                print(f"[MODEL] Gagal load {path}: {e}")
    svr_model = None
    CONC_PRED_CEILING = None
    return None

def prediksi_konsentrasi(b_mT):
    """
    Menggunakan model SVR tersimpan dengan scaling untuk mencapai target konsentrasi maksimum.
    Jika model adalah Pipeline, predict langsung pada raw feature.
    """
    if svr_model is None or CONC_PRED_CEILING is None:
        return None
    try:
        x = np.array([[float(b_mT)]], dtype=float)
        y_pred = svr_model.predict(x)
        y_pred_raw = float(y_pred[0])
        
        # Scale hasil prediksi agar mencapai TARGET_CONC_MAX pada B_MAX
        # Linear scaling: y_scaled = y_raw * (TARGET_CONC_MAX / CONC_PRED_CEILING)
        if CONC_PRED_CEILING > 0:
            scale_factor = TARGET_CONC_MAX / CONC_PRED_CEILING
            y_scaled = y_pred_raw * scale_factor
            return y_scaled
        else:
            return y_pred_raw
    except Exception:
        return None

def nearest_concentration(value):
    if value is None:
        return None
    return min(CONCENTRATIONS, key=lambda c: abs(c - value))

data_waktu       = []
data_b           = []
data_v           = []
data_conc        = []  # Tambahkan untuk menyimpan konsentrasi
mqtt_client      = None
mqtt_connected   = False
publisher_status = "unknown"
data_lock        = threading.Lock()

BG      = "#f0f4f8"
PANEL   = "#ffffff"
ACCENT  = "#e85d04"
GREEN   = "#2d9e5f"
RED     = "#e63946"
YELLOW  = "#f4a40a"
CYAN    = "#0077b6"
TEXT    = "#1a1a2e"
SUBTEXT = "#5a6a7a"
DIVIDER = "#cbd5e1"

def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        mqtt_connected = True
        client.subscribe(MQTT_TOPIC_DATA,   qos=1)
        client.subscribe(MQTT_TOPIC_STATUS, qos=0)
        update_mqtt_status("Terhubung", GREEN)
        log(f"Subscribe: {MQTT_TOPIC_DATA}")
    else:
        mqtt_connected = False
        update_mqtt_status(f"Gagal rc={rc}", RED)
        log(f"Koneksi gagal, rc={rc}")

def on_disconnect(client, userdata, rc):
    global mqtt_connected
    mqtt_connected = False
    update_mqtt_status("Terputus", RED)
    log("Koneksi terputus.")

def on_message(client, userdata, msg):
    global publisher_status
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        if msg.topic == MQTT_TOPIC_STATUS:
            publisher_status = payload.get("status", "unknown")
            update_publisher_status(publisher_status)
            log(f"Publisher: {publisher_status}")
        elif msg.topic == MQTT_TOPIC_DATA:
            t = payload.get("t_s",  0.0)
            v = payload.get("v_V",  0.0)
            b = payload.get("b_mT", 0.0)
            pred_conc = prediksi_konsentrasi(b)
            with data_lock:
                data_waktu.append(t)
                data_b.append(b)
                data_v.append(v)
                data_conc.append(pred_conc)
            if pred_conc is None:
                log(f"t={t:.2f}s V={v:.4f} B={b:.4f}mT | pred=error")
            else:
                log(f"t={t:.2f}s V={v:.4f} B={b:.4f}mT | pred={pred_conc:.4f} mg/mL")
    except Exception as e:
        log(f"Error: {e}")

def connect_mqtt():
    global mqtt_client
    broker = broker_var.get().strip()
    try:
        port = int(port_var.get().strip())
    except ValueError:
        log("Port tidak valid.")
        return
    try:
        if mqtt_client:
            try: mqtt_client.loop_stop(); mqtt_client.disconnect()
            except: pass
        mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID, protocol=mqtt.MQTTv311)
        mqtt_client.on_connect    = on_connect
        mqtt_client.on_disconnect = on_disconnect
        mqtt_client.on_message    = on_message
        mqtt_client.connect_async(broker, port, keepalive=60)
        mqtt_client.loop_start()
        log(f"Menghubungkan ke {broker}:{port}...")
    except Exception as e:
        update_mqtt_status("Error", RED)
        log(f"Error: {e}")

def disconnect_mqtt():
    global mqtt_client, mqtt_connected
    if mqtt_client:
        mqtt_client.loop_stop(); mqtt_client.disconnect()
    mqtt_connected = False
    update_mqtt_status("Terputus", RED)
    log("Koneksi diputus.")

def update_mqtt_status(text, color):
    try: lbl_mqtt_status.config(text=text, fg=color)
    except: pass

def update_publisher_status(s):
    m = {
        "publisher_online":  ("Online",     GREEN),
        "collecting":        ("Collecting", YELLOW),
        "stopped":           ("Stopped",    "#e85d04"),
        "publisher_offline": ("Offline",    RED),
    }
    txt, col = m.get(s, (s, SUBTEXT))
    try: lbl_pub_status.config(text=txt, fg=col)
    except: pass

def update_prediction_label(pred_value):
    try:
        if pred_value is None:
            lbl_pred_value.config(text="Konsentrasi terdeteksi: -")
            lbl_pred_class.config(text="Kelas terdekat: -")
        else:
            cls = nearest_concentration(pred_value)
            lbl_pred_value.config(text=f"Konsentrasi terdeteksi: {pred_value:.4f} mg/mL")
            lbl_pred_class.config(text=f"Kelas terdekat: {cls} mg/mL")
    except Exception:
        pass

def reset_data():
    with data_lock:
        data_waktu.clear(); data_b.clear(); data_v.clear(); data_conc.clear()
    line.set_data([], [])
    ax.relim(); ax.autoscale_view(); canvas.draw()
    lbl_count.config(text="0 sampel")
    lbl_last_b.config(text="B = - mT")
    lbl_last_v.config(text="V = - V")
    update_prediction_label(None)
    log("Data direset.")

def simpan_excel():
    with data_lock:
        if not data_waktu:
            messagebox.showwarning("Peringatan", "Belum ada data.")
            return
        df = pd.DataFrame({"t (s)": list(data_waktu),
                           "V (V)": list(data_v),
                           "B (mT)": list(data_b),
                           "Conc (mg/mL)": [c if c is not None else "" for c in data_conc]})
    fp = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
    if fp:
        try:
            df.to_excel(fp, index=False)
            log(f"Excel: {fp}")
        except Exception as e:
            messagebox.showerror("Gagal", str(e))

def simpan_gambar():
    fp = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
    if fp:
        fig.savefig(fp, dpi=150)
        log(f"Gambar: {fp}")

def keluar():
    if messagebox.askokcancel("Keluar", "Yakin ingin keluar?"):
        try:
            if mqtt_client: mqtt_client.loop_stop(); mqtt_client.disconnect()
        except: pass
        root.destroy()

def update_plot(frame):
    with data_lock:
        if not data_waktu: return line,
        x = list(data_waktu)
        y = list(data_b)
        v = data_v[-1] if data_v else 0.0
        c = data_conc[-1] if data_conc else None
    line.set_data(x, y)
    ax.relim(); ax.autoscale_view(); canvas.draw()
    n = len(x)
    lbl_count.config(text=f"{n} sampel")
    lbl_last_b.config(text=f"B = {y[-1]:.4f} mT")
    lbl_last_v.config(text=f"V = {v:.4f} V")
    update_prediction_label(c)
    if n > 1:
        lbl_bmax.config(text=f"Max: {max(y):.4f} mT")
        lbl_bmin.config(text=f"Min: {min(y):.4f} mT")
        lbl_bavg.config(text=f"Avg: {sum(y)/n:.4f} mT")
    return line,

# ROOT
root = tk.Tk()
root.title("GMR UIN R1A - MQTT Subscriber")
root.configure(bg=BG)
root.resizable(True, True)

try:
    root.state("zoomed")
except tk.TclError:
    try:
        root.attributes("-zoomed", True)
    except tk.TclError:
        root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")

# HEADER
frm_hdr = tk.Frame(root, bg=ACCENT, pady=6)
frm_hdr.pack(fill=tk.X)
tk.Label(frm_hdr, text="GMR UIN R1A", font=("Courier New", 16, "bold"),
         bg=ACCENT, fg="#ffffff").pack(side=tk.LEFT, padx=14)
tk.Label(frm_hdr, text="MQTT SUBSCRIBER", font=("Courier New", 9),
         bg=ACCENT, fg="#ffffff").pack(side=tk.LEFT)
tk.Label(frm_hdr, text="RECEIVE MODE", font=("Courier New", 8, "bold"),
         bg="#ffffff", fg=ACCENT, padx=8, pady=2).pack(side=tk.RIGHT, padx=12)

# MAIN
frm_main = tk.Frame(root, bg=BG)
frm_main.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

# Kolom kiri: plot
frm_left = tk.Frame(frm_main, bg=BG)
frm_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

fig, ax = plt.subplots(facecolor="#ffffff")
ax.set_facecolor("#f8fafc")
ax.set_title("Medan Magnet vs Waktu  [Subscriber]", color=TEXT, fontsize=10, pad=6)
ax.set_xlabel("Waktu, t (s)", color=SUBTEXT, fontsize=9)
ax.set_ylabel("Medan Magnet, B (mT)", color=SUBTEXT, fontsize=9)
ax.tick_params(colors=SUBTEXT)
for sp in ax.spines.values(): sp.set_color(DIVIDER)
ax.grid(True, color="#e2e8f0", linewidth=0.7)
line, = ax.plot([], [], lw=2, color=ACCENT)
canvas = FigureCanvasTkAgg(fig, master=frm_left)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Kolom kanan: scrollable
frm_ro = tk.Frame(frm_main, bg=BG, width=270)
frm_ro.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
frm_ro.pack_propagate(False)

cv = tk.Canvas(frm_ro, bg=BG, highlightthickness=0)
sb = tk.Scrollbar(frm_ro, orient=tk.VERTICAL, command=cv.yview)
cv.configure(yscrollcommand=sb.set)
sb.pack(side=tk.RIGHT, fill=tk.Y)
cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

frm_right = tk.Frame(cv, bg=BG)
wid = cv.create_window((0, 0), window=frm_right, anchor="nw")

frm_right.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
cv.bind("<Configure>",        lambda e: cv.itemconfig(wid, width=e.width))
cv.bind("<MouseWheel>",       lambda e: cv.yview_scroll(int(-1*(e.delta/120)), "units"))
cv.bind("<Button-4>",         lambda e: cv.yview_scroll(-1, "units"))
cv.bind("<Button-5>",         lambda e: cv.yview_scroll(1, "units"))

def section(title):
    f = tk.Frame(frm_right, bg=PANEL)
    f.pack(fill=tk.X, pady=(0, 5))
    tk.Label(f, text=f"  {title}", font=("Courier New", 7, "bold"),
             bg=PANEL, fg=SUBTEXT).pack(anchor=tk.W, pady=(4, 1))
    tk.Frame(f, bg=DIVIDER, height=1).pack(fill=tk.X, padx=6, pady=(0, 4))
    return f

def lbl(p, t):
    tk.Label(p, text=t, bg=PANEL, fg=TEXT, font=("Courier New", 8)).pack(anchor=tk.W, padx=8)

def ent(p, var, col=None):
    c = col or ACCENT
    e = tk.Entry(p, textvariable=var, bg="#f1f5f9", fg=c,
                 font=("Courier New", 9), relief=tk.FLAT, bd=3, insertbackground=c)
    e.pack(fill=tk.X, padx=8, pady=2)
    return e

def mkbtn(p, text, cmd, bg, fg=TEXT, **kw):
    return tk.Button(p, text=text, command=cmd, bg=bg, fg=fg,
                     font=("Courier New", 8, "bold"), relief=tk.FLAT,
                     cursor="hand2", pady=6, **kw)

# PANEL: MQTT
pnl_m = section("MQTT BROKER")
lbl(pnl_m, "Broker IP / Hostname")
broker_var = tk.StringVar(value=MQTT_BROKER)
ent(pnl_m, broker_var)
lbl(pnl_m, "Port")
port_var = tk.StringVar(value=str(MQTT_PORT))
ent(pnl_m, port_var)
lbl(pnl_m, "Subscribe Topic")
topic_var = tk.StringVar(value=MQTT_TOPIC_DATA)
ent(pnl_m, topic_var, CYAN)

frm_mc = tk.Frame(pnl_m, bg=PANEL)
frm_mc.pack(fill=tk.X, padx=8, pady=(4, 2))
tk.Button(frm_mc, text="Subscribe", command=connect_mqtt,
          bg=ACCENT, fg="#ffffff", font=("Courier New", 8, "bold"),
          relief=tk.FLAT, cursor="hand2", padx=6).pack(side=tk.LEFT)
tk.Button(frm_mc, text="Putus", command=disconnect_mqtt,
          bg="#e2e8f0", fg=TEXT, font=("Courier New", 8),
          relief=tk.FLAT, cursor="hand2", padx=6).pack(side=tk.LEFT, padx=(4, 0))
lbl_mqtt_status = tk.Label(pnl_m, text="Belum", bg=PANEL, fg=RED, font=("Courier New", 7))
lbl_mqtt_status.pack(anchor=tk.W, padx=8, pady=(0, 5))

# PANEL: MODEL STATUS
pnl_model = section("MODEL SVR")
lbl(pnl_model, "Status model tersimpan")
if _loaded_path:
    model_text = f"Loaded: {os.path.basename(_loaded_path)}"
    model_color = GREEN
else:
    model_text = "Model SVR tidak ditemukan"
    model_color = RED

lbl_model_status = tk.Label(
    pnl_model,
    text=model_text,
    bg=PANEL,
    fg=model_color,
    font=("Courier New", 7, "bold"),
    wraplength=220,
    justify=tk.LEFT
)
lbl_model_status.pack(anchor=tk.W, padx=8, pady=(0, 6))

# PANEL: PUBLISHER STATUS
pnl_p = section("STATUS PUBLISHER")
tk.Label(pnl_p, text="Publisher:", bg=PANEL, fg=SUBTEXT, font=("Courier New", 8)).pack(anchor=tk.W, padx=8)
lbl_pub_status = tk.Label(pnl_p, text="Unknown", bg=PANEL, fg=SUBTEXT,
                           font=("Courier New", 11, "bold"))
lbl_pub_status.pack(anchor=tk.W, padx=8, pady=(0, 6))

# PANEL: LIVE DATA
pnl_l = section("LIVE DATA")
lbl_last_b = tk.Label(pnl_l, text="B = - mT", bg=PANEL, fg=ACCENT,
                       font=("Courier New", 13, "bold"))
lbl_last_b.pack(pady=(2, 0))
lbl_last_v = tk.Label(pnl_l, text="V = - V", bg=PANEL, fg=TEXT,
                       font=("Courier New", 9))
lbl_last_v.pack()
lbl_count = tk.Label(pnl_l, text="0 sampel", bg=PANEL, fg=SUBTEXT,
                      font=("Courier New", 8))
lbl_count.pack(pady=(1, 6))

# PANEL: PREDIKSI KONENTRASI
pnl_pred = section("PREDIKSI KONENTRASI")
lbl_pred_value = tk.Label(
    pnl_pred,
    text="Konsentrasi terdeteksi: -",
    bg=PANEL,
    fg=ACCENT,
    font=("Courier New", 10, "bold"),
    anchor="w"
)
lbl_pred_value.pack(fill=tk.X, padx=10, pady=(6, 2))

lbl_pred_class = tk.Label(
    pnl_pred,
    text="Kelas terdekat: -",
    bg=PANEL,
    fg=TEXT,
    font=("Courier New", 8),
    anchor="w"
)
lbl_pred_class.pack(fill=tk.X, padx=10, pady=(0, 6))

# PANEL: STATISTIK
pnl_st = section("STATISTIK B (mT)")
lbl_bmax = tk.Label(pnl_st, text="Max: -", bg=PANEL, fg=GREEN, font=("Courier New", 8))
lbl_bmax.pack(anchor=tk.W, padx=8)
lbl_bmin = tk.Label(pnl_st, text="Min: -", bg=PANEL, fg=RED,  font=("Courier New", 8))
lbl_bmin.pack(anchor=tk.W, padx=8)
lbl_bavg = tk.Label(pnl_st, text="Avg: -", bg=PANEL, fg=CYAN, font=("Courier New", 8))
lbl_bavg.pack(anchor=tk.W, padx=8, pady=(0, 6))

# PANEL: KONTROL
pnl_c = section("KONTROL")

fb1 = tk.Frame(pnl_c, bg=PANEL)
fb1.pack(fill=tk.X, padx=8, pady=(0, 3))
mkbtn(fb1, "RESET", reset_data,    "#e2e8f0").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
mkbtn(fb1, "EXCEL", simpan_excel,  "#e2e8f0").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2))
mkbtn(fb1, "IMAGE", simpan_gambar, "#e2e8f0").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

fb2 = tk.Frame(pnl_c, bg=PANEL)
fb2.pack(fill=tk.X, padx=8, pady=(0, 8))
mkbtn(fb2, "KELUAR", keluar, RED, "#ffffff").pack(fill=tk.X)

# PANEL: LOG
pnl_lg = section("LOG OUTPUT")
txt_log = tk.Text(pnl_lg, height=7, bg="#f1f5f9", fg=ACCENT,
                  font=("Courier New", 7), state=tk.DISABLED,
                  relief=tk.FLAT, bd=3, wrap=tk.WORD)
txt_log.pack(fill=tk.X, padx=6, pady=(0, 6))

ani = animation.FuncAnimation(fig, update_plot, interval=200, cache_frame_data=False)
root.protocol("WM_DELETE_WINDOW", keluar)

# Load model SVR
_loaded_path = load_svr_model()

if _loaded_path:
    log(f"Model SVR loaded: {_loaded_path}")
    if CONC_PRED_CEILING:
        scale_factor = TARGET_CONC_MAX / CONC_PRED_CEILING
        log(f"B_max={B_MAX:.4f}mT | Pred_ceiling={CONC_PRED_CEILING:.4f}mg/mL → Scaled_max={TARGET_CONC_MAX}mg/mL (scale={scale_factor:.4f}x)")
else:
    log("Model SVR tidak ditemukan. Prediksi konsentrasi dinonaktifkan.")

log("Klik 'Subscribe' untuk mulai menerima data.")
root.mainloop()
# signed by: Gilang Pratama Putra Siswanto (2026-05-12)
