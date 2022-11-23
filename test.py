import torch
import numpy as np

t1 = torch.tensor([[21,39],[31,30],[23,43],
                [11,46],[26,46],[31,25],[21,38],
                [22,39],[22,19],[18, 14]])


class Geeks:
  def __init__(self, name1 = "Arun", num2 = 46, name3 = "Rishab"):
    self.name1 = name1
    self.num2 = num2
    self.name3 = name3
   
GeeksforGeeks = Geeks()
print(GeeksforGeeks)
print(*vars(GeeksforGeeks))
print(t1)