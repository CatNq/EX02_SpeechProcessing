import tkinter as tk
from tkinter import messagebox

from predict import Predicter
import recorder



predicter = Predicter()




window = tk.Tk()
window.geometry("400x300")

frame0 = tk.Frame(master=window)
frame0.pack()

frame1 = tk.Frame(master=window)
frame1.pack()

frame2 = tk.Frame(master=window)
frame2.pack()

label = tk.Label(master=frame0, text="Speech recognition")
label.pack(padx=5, pady=10)

btn_record = tk.Button(master=frame1, width=13, height=2, text="Record", command=recorder.record)
btn_record.pack(side=tk.LEFT, padx=5, pady=5)

btn_playback = tk.Button(master=frame1, width=13, height=2, text="Playback", command=recorder.playback)
btn_playback.pack(side=tk.LEFT, padx=5, pady=5)

def predict():
    predicted = predicter.predict()
    messagebox.showinfo("Predicted result: ", predicted)


btn_predict = tk.Button(master=frame2, width=13, height=2, text="Predict", command=predict)
btn_predict.pack(side=tk.LEFT, padx=5, pady=5)


window.mainloop()