import tkinter as tk
from tkinter import messagebox
import threading
import time
from pynput import mouse

# Globale Variablen zur Aufnahme
recording = False
start_time = None
events = []
current_listener = None

# Callback-Funktionen für die Mausereignisse
def on_move(x, y):
    if recording:
        events.append({
            'time': time.time() - start_time,
            'type': 'move',
            'pos': (x, y)
        })

def on_click(x, y, button, pressed):
    if recording:
        events.append({
            'time': time.time() - start_time,
            'type': 'click',
            'pos': (x, y),
            'button': button,
            'pressed': pressed
        })

def on_scroll(x, y, dx, dy):
    if recording:
        events.append({
            'time': time.time() - start_time,
            'type': 'scroll',
            'pos': (x, y),
            'dx': dx,
            'dy': dy
        })

# Funktion zum Starten bzw. Stoppen der Aufnahme
def toggle_record():
    global recording, start_time, events, current_listener
    if not recording:
        # Aufnahme starten
        recording = True
        events = []  # Vorherige Events löschen
        start_time = time.time()
        current_listener = mouse.Listener(
            on_move=on_move,
            on_click=on_click,
            on_scroll=on_scroll
        )
        current_listener.start()
        record_button.config(text="Stop Recording")
    else:
        # Aufnahme beenden
        recording = False
        if current_listener is not None:
            current_listener.stop()
            current_listener = None
        record_button.config(text="Start Recording")
        messagebox.showinfo("Aufnahme beendet", f"{len(events)} Ereignisse aufgezeichnet.")

# Funktion zum Abspielen der aufgenommenen Events
def play_events():
    if not events:
        messagebox.showwarning("Keine Daten", "Es sind keine Mausaktionen aufgezeichnet.")
        return
    threading.Thread(target=playback_thread, daemon=True).start()

def playback_thread():
    mc = mouse.Controller()
    start_playback = time.time()
    for event in events:
        # Warten, bis der Zeitpunkt des Events erreicht ist
        time_to_wait = event['time'] - (time.time() - start_playback)
        if time_to_wait > 0:
            time.sleep(time_to_wait)
        if event['type'] == 'move':
            mc.position = event['pos']
        elif event['type'] == 'click':
            if event['pressed']:
                mc.press(event['button'])
            else:
                mc.release(event['button'])
        elif event['type'] == 'scroll':
            mc.scroll(event['dx'], event['dy'])

# GUI mit Tkinter
root = tk.Tk()
root.title("Maus Recorder & Player")

record_button = tk.Button(root, text="Start Recording", width=20, command=toggle_record)
record_button.pack(pady=10)

play_button = tk.Button(root, text="Play Recording", width=20, command=play_events)
play_button.pack(pady=10)

root.mainloop()
