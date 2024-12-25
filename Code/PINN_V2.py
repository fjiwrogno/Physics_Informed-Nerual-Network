import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, bode, TransferFunction
from torch.autograd import grad
from scipy.signal import lsim, TransferFunction

# 定义传递函数
# Define the transfer function
num = [4]
den = [1, 1.2, 4]
system = TransferFunction(num, den)

# 生成训练数据
# Generate training data
def generate_sine_data(frequency, duration=10, fs=1000):
    t = np.linspace(0, duration, int(fs*duration))
    u = np.sin(2 * np.pi * frequency * t)
    tout, y, _ = lsim(system, U=u, T=t)
    return t, u, y

# 采样频率和持续时间
# Sampling frequency and duration
fs = 1000  # 1000 Hz
duration = 10  # 10 seconds

# 生成多个频率的正弦波数据
# Generate sine wave data for multiple frequencies
frequencies = np.linspace(0.1, 10, 20)  # 0.1 Hz to 10 Hz, 20 frequencies
t_train = []
u_train = []
y_train = []

for f in frequencies:
    t, u, y = generate_sine_data(f, duration, fs)
    t_train.append(t)
    u_train.append(u)
    y_train.append(y)

# 合并所有训练数据
# Concatenate all training data
t_train = np.concatenate(t_train)
u_train = np.concatenate(u_train)
y_train = np.concatenate(y_train)

# 转换为 PyTorch 张量
# Convert to PyTorch tensors
t_train_tensor = torch.tensor(t_train, dtype=torch.float32).reshape(-1, 1)
u_train_tensor = torch.tensor(u_train, dtype=torch.float32).reshape(-1, 1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

# 定义神经网络
# Define the neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
    
    def forward(self, t, u):
        inputs = torch.cat([t, u], dim=1)
        return self.net(inputs)

# 定义损失函数
# Define the loss function
def pinn_loss(model, t, u, y_true):
    # 预测 y
    # Predict y
    y_pred = model(t, u)
    
    # 计算一阶导数 y'
    # Compute the first derivative y'
    y_pred_grad = grad(y_pred, t, torch.ones_like(y_pred), retain_graph=True, create_graph=True)[0]
    
    # 计算二阶导数 y''
    # Compute the second derivative y''
    y_pred_grad2 = grad(y_pred_grad, t, torch.ones_like(y_pred_grad), retain_graph=True, create_graph=True)[0]
    
    # 物理约束 y'' + 1.2 y' + 4 y = 4 u
    # Physics constraint: y'' + 1.2 y' + 4 y = 4 u
    f = y_pred_grad2 + 1.2 * y_pred_grad + 4 * y_pred - 4 * u
    physics_loss = torch.mean(f**2)
    
    # 数据损失
    # Data loss
    data_loss = torch.mean((y_pred - y_true)**2)
    
    return data_loss + physics_loss

# 初始化模型和优化器
# Initialize the model and optimizer
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 转换训练数据到计算图
# Enable gradient computation for training data
t_train_tensor.requires_grad = True

# 训练参数
# Training parameters
epochs = 5000
print_interval = 500

# 训练循环
# Training loop
for epoch in range(epochs+1):
    optimizer.zero_grad()
    loss = pinn_loss(model, t_train_tensor, u_train_tensor, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % print_interval == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 生成 chirp 信号测试数据
# Generate chirp signal test data
def generate_chirp_data(f_start=0.1, f_end=20, duration=10, fs=1000):
    t = np.linspace(0, duration, int(fs*duration))
    u = chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')
    tout, y, _ = lsim(system, U=u, T=t)
    return t, u, y

t_test, u_test, y_test = generate_chirp_data()

# 转换为 PyTorch 张量
# Convert test data to PyTorch tensors
t_test_tensor = torch.tensor(t_test, dtype=torch.float32).reshape(-1, 1)
u_test_tensor = torch.tensor(u_test, dtype=torch.float32).reshape(-1, 1)

# 预测
# Prediction
model.eval()
with torch.no_grad():
    y_pred_tensor = model(t_test_tensor, u_test_tensor)
y_pred = y_pred_tensor.numpy().flatten()

# 计算真实系统的波特图
# Compute the Bode plot of the true system
w, mag_true, phase_true = bode(system)

# 计算模型预测系统的传递函数
# Compute the transfer function of the predicted system using FFT
# Since the model directly predicts y(t) from u(t), approximate the frequency response via FFT
Y_pred = y_pred
U_test = u_test

# 计算 FFT
# Compute FFT
fft_len = len(t_test)
freqs = np.fft.rfftfreq(fft_len, d=1/fs)
U_fft = np.fft.rfft(U_test)
Y_fft = np.fft.rfft(Y_pred)

# 避免除以零
# Avoid division by zero
U_fft = np.where(U_fft == 0, 1e-10, U_fft)

# 计算频率响应
# Compute frequency response
H_pred = Y_fft / U_fft
mag_pred = 20 * np.log10(np.abs(H_pred))
phase_pred = np.angle(H_pred, deg=True)

# 绘制波特图
# Plot the Bode plot
plt.figure(figsize=(12, 8))

# 幅频响应
# Magnitude response
plt.subplot(2, 1, 1)
plt.semilogx(w/(2*np.pi), mag_true, label='True')
plt.semilogx(freqs, mag_pred, '--', label='PINN Predicted')
plt.title('Bode Plot')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True, which='both', linestyle='--')

# 相频响应
# Phase response
plt.subplot(2, 1, 2)
plt.semilogx(w/(2*np.pi), phase_true, label='True')
plt.semilogx(freqs, phase_pred, '--', label='PINN Predicted')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')
plt.legend()
plt.grid(True, which='both', linestyle='--')

plt.tight_layout()
plt.show()
