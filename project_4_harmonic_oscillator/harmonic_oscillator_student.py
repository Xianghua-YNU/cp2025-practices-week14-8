#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def harmonic_oscillator_ode(state, t, omega=1.0):
    """简谐振子的一阶微分方程组。"""
    x, v = state
    return np.array([v, -omega**2 * x])

def anharmonic_oscillator_ode(state, t, omega=1.0):
    """非谐振子的一阶微分方程组。"""
    x, v = state
    return np.array([v, -omega**2 * x**3])

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """使用四阶龙格-库塔方法进行一步数值积分。"""
    k1 = ode_func(state, t, **kwargs)
    k2 = ode_func(state + 0.5*dt*k1, t + 0.5*dt, **kwargs)
    k3 = ode_func(state + 0.5*dt*k2, t + 0.5*dt, **kwargs)
    k4 = ode_func(state + dt*k3, t + dt, **kwargs)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """求解常微分方程组。"""
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    states = np.zeros((len(t), len(initial_state)))
    states[0] = initial_state
    
    for i in range(1, len(t)):
        states[i] = rk4_step(ode_func, states[i-1], t[i-1], dt, **kwargs)
    
    return t, states

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """Plot the time evolution of states."""
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], label='Position x(t)')
    plt.plot(t, states[:, 1], label='Velocity v(t)')
    plt.xlabel('Time t')
    plt.ylabel('State Variables')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """Plot the phase space trajectory."""
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('Position x')
    plt.ylabel('Velocity v')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def analyze_period(t: np.ndarray, states: np.ndarray) -> float:
    """分析振动周期。"""
    # 通过寻找位置的极大值点来估计周期
    x = states[:, 0]
    peaks = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(t[i])
    
    if len(peaks) < 2:
        return np.nan
    
    # 计算相邻峰值之间的时间差的平均值
    periods = np.diff(peaks)
    return np.mean(periods)

def main():
    # Set parameters
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    
    # Task 1 - Numerical solution of harmonic oscillator
    initial_state = np.array([1.0, 0.0])  # x(0)=1, v(0)=0
    t, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_time_evolution(t, states, 'Time Evolution of Harmonic Oscillator')
    period = analyze_period(t, states)
    print(f'Harmonic Oscillator Period: {period:.4f} (Theoretical: {2*np.pi/omega:.4f})')
    
    # Task 2 - Analysis of amplitude effect on period
    amplitudes = [0.5, 1.0, 2.0]
    periods = []
    for A in amplitudes:
        initial_state = np.array([A, 0.0])
        t, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        period = analyze_period(t, states)
        periods.append(period)
        print(f'Amplitude {A}: Period = {period:.4f}')
    
    # Task 3 - Numerical analysis of anharmonic oscillator
    for A in amplitudes:
        initial_state = np.array([A, 0.0])
        t, states = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        period = analyze_period(t, states)
        print(f'Anharmonic Oscillator - Amplitude {A}: Period = {period:.4f}')
        plot_time_evolution(t, states, f'Time Evolution of Anharmonic Oscillator (A={A})')
    
    # Task 4 - Phase space analysis
    initial_state = np.array([1.0, 0.0])
    t, states_harmonic = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_phase_space(states_harmonic, 'Phase Space Trajectory of Harmonic Oscillator')
    
    t, states_anharmonic = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_phase_space(states_anharmonic, 'Phase Space Trajectory of Anharmonic Oscillator')

if __name__ == "__main__":
    main()
    
    # TODO: 任务1 - 简谐振子的数值求解
    # 1. 设置初始条件 x(0)=1, v(0)=0
    # 2. 求解方程
    # 3. 绘制时间演化图
    
    # TODO: 任务2 - 振幅对周期的影响分析
    # 1. 使用不同的初始振幅
    # 2. 分析周期变化
    
    # TODO: 任务3 - 非谐振子的数值分析
    # 1. 求解非谐振子方程
    # 2. 分析不同振幅的影响
    
    # TODO: 任务4 - 相空间分析
    # 1. 绘制相空间轨迹
    # 2. 比较简谐和非谐振子
