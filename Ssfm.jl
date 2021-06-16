module Ssfm 
using FFTW
using Plots
using Printf
import Plotly
import ProgressMeter
export solve

struct Fiber
  T::Float64
  dt::Float64
  L::Float64
  dz::Float64
  alpha::Float64
  beta_2::Float64
  gamma::Float64
end

struct Pulse
  P0::Float64
  T0::Float64
  sech_flag::Float64
end

function solve(fiber::Fiber, pulse::Pulse)

  time_steps = Int(floor(fiber.T/fiber.dt))
  space_steps = Int(floor(fiber.L/fiber.dz))
  
  LD = pulse.T0^2 / abs(fiber.beta_2)
  LNL = 1 / (fiber.gamma*pulse.P0)
  
  disp_ratio = fiber.L / LD
  nonl_ratio = fiber.L / LNL
  print("Simulation diagnositics:\n")
  @printf("\tL/LD  = %6.4f\n", disp_ratio)
  @printf("\tL/LNL = %6.4f\n", nonl_ratio)
  @printf("\tN     = %6.4f \n", sqrt(LD / LNL))
  @printf("time steps : %6d \n", time_steps)
  @printf("space steps: %6d \n", space_steps)

  time = LinRange(-fiber.T/2, fiber.T/2, time_steps)
  space = LinRange(0, fiber.L, space_steps)
  
  w = 2*pi* LinRange(-1/(2*fiber.dt), 1/(2*fiber.dt), time_steps)
  w = fftshift(w)
  A = zeros(ComplexF32, time_steps, space_steps)
  A_spect = zeros(ComplexF32, time_steps, space_steps)
  
  if (pulse.sech_flag == 1)
    waveform = sqrt(pulse.P0) .* sech(time/pulse.T0)
  else
    waveform = sqrt(pulse.P0) .* exp.(-(time/pulse.T0) .^ 2 ./ 2)
  end
  
  A[:, 1] = waveform
  #fig = Plots.plot(time, abs.(A[:, 1]), show=true)
  A_spect[:, 1] = fft(A[:, 1])

  fwd_disp = exp.(fiber.dz * (im * fiber.beta_2/2 * w .^ 2 .- fiber.alpha/2))
  for n = 1:space_steps-1
    A_spect[:, n] = fft(A[:, n])
    A_spect[:, n+1] = A_spect[:, n] .* fwd_disp
    A[:, n+1] = ifft(A_spect[:, n+1])
    A[:, n+1] = A[:, n+1] .* exp.(im * fiber.dz * fiber.gamma .* (abs.(A[:, n+1])).^2)
  end
  A_spect[:, space_steps] = fft(A[:, space_steps])
  fig2 = Plots.heatmap(abs.(A), show=true)
end

function mem_estimate(fiber::Fiber)
  return 2* fiber.T/fiber.dt * fiber.L/fiber.dz * 128
end

function config()
  Plots.plotly()
  Plots.default(size=(1300, 800),
                guidefont=("times", 10), 
                legendfont=("times", 12),
                tickfont=("times", 10)
                )
end
config()

dispersive = Fiber(2e-9,
               5e-13,
               77500,
               100,
               0.0,
               -20e-27,
               0.00)
realistic = Fiber(2e-9,
                  5e-13,
                  77500,
                  100,
                  0.0002,
                  20e-27,
                  2.00)
pulse1 =  Pulse(0.001,
                10e-12,
                0)

configs = [(dispersive, pulse1),
           (realistic, pulse1)]
mem_limit = 15000000000 #byte

for (fiber, pulse) in configs
  estimate = mem_estimate(fiber)
  if estimate < mem_limit
    @printf("Estimated memory consumption: %4.1f MiB\n", estimate/(1024^2))
    @time solve(fiber, pulse)
  else
    @printf("Estimated memory consumption (%4.1f MiB) exceed maximum!\n", estimate/(1024^2))
  end
end

end
