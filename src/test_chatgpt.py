import tkinter as tk

def on_button_press():
    if button["text"] == "Change color":
        button["text"] = "Original color"
        root.configure(bg="blue")
    else:
        button["text"] = "Change color"
        root.configure(bg="white")

root = tk.Tk()
root.title("GUI Example")
root.geometry("200x100")

button = tk.Button(root, text="Change color", command=on_button_press)
button.pack()

root.mainloop()