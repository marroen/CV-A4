import matplotlib.pyplot as plt
import numpy as np

# Data from training logs
epochs = np.arange(1, 17)

# Total Loss
train_loss = [6.6098, 4.9670, 4.7692, 4.5208, 4.3661, 4.3011, 4.1433, 3.9941, 3.8839, 3.8231, 3.7071, 3.5816, 3.4946, 3.3541, 3.2459, 3.1490]
val_loss = [0.1299, 0.1262, 0.1380, 0.1176, 0.1258, 0.1263, 0.1125, 0.1225, 0.1103, 0.1036, 0.0893, 0.0968, 0.1078, 0.0921, 0.0970, 0.0900]

# Box Loss
train_box = [12.8268, 8.4741, 8.3773, 7.8901, 7.4728, 7.4320, 7.2988, 6.9274, 6.8524, 6.9558, 6.6385, 6.6155, 6.5471, 6.3673, 6.3156, 6.3025]
val_box = [0.2153, 0.1777, 0.2558, 0.2083, 0.1988, 0.2242, 0.1808, 0.2314, 0.1334, 0.2111, 0.1304, 0.1853, 0.2149, 0.1801, 0.1749, 0.0996]

# Confidence Loss
train_conf = [16.9629, 14.3247, 13.3130, 12.3894, 12.0255, 11.6313, 11.1454, 10.7092, 10.2580, 10.1177, 9.7703, 9.3277, 9.2112, 8.6751, 8.5677, 8.3151]
val_conf = [0.2808, 0.2533, 0.2720, 0.2421, 0.2719, 0.2309, 0.2259, 0.2072, 0.2130, 0.1880, 0.1592, 0.1562, 0.1754, 0.1787, 0.1790, 0.1350]

# Class Loss
train_class = [13.3315, 9.5108, 9.2458, 8.9449, 8.6731, 8.6396, 8.1500, 7.9409, 7.7500, 7.2987, 7.1368, 6.8362, 6.3739, 5.9462, 5.4118, 5.0031]
val_class = [0.1146, 0.1456, 0.1501, 0.1090, 0.1429, 0.1540, 0.1140, 0.1390, 0.1488, 0.0681, 0.0997, 0.0942, 0.1018, 0.0501, 0.0772, 0.1725]

# No-Object Loss
train_noobj = [9.7209, 7.4025, 7.1829, 6.9197, 6.7274, 6.6670, 6.5235, 6.3407, 6.1801, 6.1881, 6.0855, 5.8391, 5.8010, 5.8192, 5.6465, 5.5390]
val_noobj = [0.1689, 0.1805, 0.1503, 0.1459, 0.1410, 0.1488, 0.1542, 0.1574, 0.1664, 0.1545, 0.1465, 0.1449, 0.1546, 0.1439, 0.1506, 0.1331]

# Create plots
plt.figure(figsize=(15, 10))

# 1. Total Loss
plt.subplot(3, 2, 1)
plt.plot(epochs, train_loss, 'b-', label='Train')
plt.plot(epochs, val_loss, 'r-', label='Val')
plt.title('Total Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 2. Box Loss
plt.subplot(3, 2, 2)
plt.plot(epochs, train_box, 'b-', label='Train')
plt.plot(epochs, val_box, 'r-', label='Val')
plt.title('Box Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 3. Confidence Loss
plt.subplot(3, 2, 3)
plt.plot(epochs, train_conf, 'b-', label='Train')
plt.plot(epochs, val_conf, 'r-', label='Val')
plt.title('Confidence Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 4. Class Loss
plt.subplot(3, 2, 4)
plt.plot(epochs, train_class, 'b-', label='Train')
plt.plot(epochs, val_class, 'r-', label='Val')
plt.title('Class Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 5. No-Object Loss
plt.subplot(3, 2, 5)
plt.plot(epochs, train_noobj, 'b-', label='Train')
plt.plot(epochs, val_noobj, 'r-', label='Val')
plt.title('No-Object Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()