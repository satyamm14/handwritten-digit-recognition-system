import tkinter as tk
import torch
from torch import nn

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


model = NeuralNet()

state_dict = torch.load("./model/mnist_model.zip", weights_only=False)  
model.load_state_dict(state_dict)
model.eval()


class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition")

        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0, pady=10)

        self.prediction_label = tk.Label(root, text="Prediction: ?", font=("Helvetica", 16))
        self.prediction_label.grid(row=2, column=0, pady=10)

        self.pixel_size = 10 
        self.grid = [[0 for _ in range(28)] for _ in range(28)]

    def draw(self, event):
        x, y = event.x, event.y
        grid_x, grid_y = x // self.pixel_size, y // self.pixel_size

        if 0 <= grid_x < 28 and 0 <= grid_y < 28:
            self.grid[grid_y][grid_x] = 1
            self.canvas.create_rectangle(
                grid_x * self.pixel_size,
                grid_y * self.pixel_size,
                (grid_x + 1) * self.pixel_size,
                (grid_y + 1) * self.pixel_size,
                fill="black",
            )
            self.predict_digit()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.grid = [[0 for _ in range(28)] for _ in range(28)]
        self.prediction_label.config(text="Prediction: ?")

    def predict_digit(self):
        input_tensor = torch.tensor(self.grid, dtype=torch.float32).unsqueeze(0).view(-1, 28 * 28)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_digit = torch.argmax(output).item()

        self.prediction_label.config(text=f"Prediction: {predicted_digit}")



if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()

