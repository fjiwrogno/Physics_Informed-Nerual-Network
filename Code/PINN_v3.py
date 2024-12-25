import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, bode, TransferFunction, lsim
from torch.autograd import grad

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

# 生成多个正弦波测试数据
# Generate multiple sine wave test data
def generate_multiple_sine_tests(frequencies, duration=10, fs=1000):
    t = np.linspace(0, duration, int(fs*duration))
    y_total = np.zeros_like(t)
    y_pred_total = np.zeros_like(t)
    responses = {}
    
    for f in frequencies:
        u = np.sin(2 * np.pi * f * t)
        tout, y, _ = lsim(system, U=u, T=t)
        responses[f] = {'input': u, 'output': y}
    return t, responses

# 生成脉冲信号测试数据
# Generate impulse signal test data
def generate_impulse_data(duration=10, fs=1000):
    t = np.linspace(0, duration, int(fs*duration))
    u = np.zeros_like(t)
    u[0] = 1  # 脉冲信号在 t=0
    tout, y, _ = lsim(system, U=u, T=t)
    return t, u, y

# 生成测试数据
# Generate test data
# 定义测试频率
# Define test frequencies
test_frequencies = np.linspace(0.1, 20, 40)  # 0.1 Hz 到 20 Hz, 40 个频率

# 生成多个正弦波测试数据
# Generate multiple sine wave test data
t_sine_test, sine_responses = generate_multiple_sine_tests(test_frequencies, duration, fs)

# 生成脉冲信号测试数据
# Generate impulse signal test data
t_impulse_test, impulse_u, impulse_y = generate_impulse_data(duration, fs)

# 转换为 PyTorch 张量
# Convert test data to PyTorch tensors
# 正弦波测试数据
# Sine wave test data
sine_predictions = {}
for f in test_frequencies:
    u = sine_responses[f]['input']
    t = t_sine_test
    u_tensor = torch.tensor(u, dtype=torch.float32).reshape(-1, 1)
    t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        y_pred_tensor = model(t_tensor, u_tensor)
    y_pred = y_pred_tensor.numpy().flatten()
    sine_predictions[f] = y_pred

# 脉冲信号测试数据
# Impulse signal test data
u_impulse_tensor = torch.tensor(impulse_u, dtype=torch.float32).reshape(-1, 1)
t_impulse_tensor = torch.tensor(t_impulse_test, dtype=torch.float32).reshape(-1, 1)
with torch.no_grad():
    y_pred_impulse_tensor = model(t_impulse_tensor, u_impulse_tensor)
y_pred_impulse = y_pred_impulse_tensor.numpy().flatten()

# 计算真实系统的波特图
# Compute the Bode plot of the true system
w, mag_true, phase_true = bode(system)

# 计算模型预测系统的传递函数
# Compute the transfer function of the predicted system using sine wave tests and impulse
# 通过正弦波测试计算幅频和相频响应
# Compute magnitude and phase response from sine wave tests
mag_pred = []
phase_pred = []

for f in test_frequencies:
    y_pred = sine_predictions[f]
    u = sine_responses[f]['input']
    
    # 计算输入和输出的FFT
    U_fft = np.fft.rfft(u)
    Y_fft = np.fft.rfft(y_pred)
    
    # 计算频率响应
    # Avoid division by zero
    U_fft = np.where(U_fft == 0, 1e-10, U_fft)
    H_pred = Y_fft / U_fft
    
    # 计算幅度和相位
    mag = 20 * np.log10(np.abs(H_pred))
    phase = np.angle(H_pred, deg=True)
    
    # 取对应频率的幅度和相位
    idx = np.argmin(np.abs(np.fft.rfftfreq(len(u), d=1/fs) - f))
    mag_pred.append(mag[idx])
    phase_pred.append(phase[idx])

# 通过脉冲信号测试计算幅频和相频响应
# Compute frequency response from impulse signal test using FFT
# 计算FFT
fft_len = len(t_impulse_test)
freqs = np.fft.rfftfreq(fft_len, d=1/fs)
U_fft_impulse = np.fft.rfft(impulse_u)
Y_fft_impulse = np.fft.rfft(y_pred_impulse)

# 避免除以零
U_fft_impulse = np.where(U_fft_impulse == 0, 1e-10, U_fft_impulse)

# 计算频率响应
H_pred_impulse = Y_fft_impulse / U_fft_impulse
mag_impulse = 20 * np.log10(np.abs(H_pred_impulse))
phase_impulse = np.angle(H_pred_impulse, deg=True)

# 将脉冲信号的频率响应与正弦波测试的频率响应结合
# Combine frequency responses from impulse and sine wave tests
# 这里主要依赖于正弦波测试的频率响应，因为脉冲信号的FFT可能会包含噪声
# Here we mainly rely on sine wave test frequency response, as impulse signal's FFT might contain noise

# 绘制波特图
# Plot the Bode plot
plt.figure(figsize=(12, 8))

# 幅频响应
# Magnitude response
plt.subplot(2, 1, 1)
plt.semilogx(w/(2*np.pi), mag_true, label='True')
plt.semilogx(test_frequencies, mag_pred, '--', label='PINN Predicted (Sine Tests)')
plt.title('Bode Plot')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True, which='both', linestyle='--')

# 相频响应
# Phase response
plt.subplot(2, 1, 2)
plt.semilogx(w/(2*np.pi), phase_true, label='True')
plt.semilogx(test_frequencies, phase_pred, '--', label='PINN Predicted (Sine Tests)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')
plt.legend()
plt.grid(True, which='both', linestyle='--')

plt.tight_layout()
plt.show()

# 可选：绘制脉冲响应对比
# Optional: Plot impulse response comparison
plt.figure(figsize=(12, 6))
plt.plot(t_impulse_test, impulse_y, label='True Impulse Response')
plt.plot(t_impulse_test, y_pred_impulse, '--', label='PINN Predicted Impulse Response')
plt.title('Impulse Response Comparison')
plt.xlabel('Time (s)')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.show()
