module Ssfm 
using FFTW
using Plots
using Printf
import Plotly
using ProgressMeter
export solve, config

struct Fiber
  T::Float64
  dt::Float64
  L::Float64
  dz::Float64
  alpha::Float64 # dB/km
  beta_2::Float64 # ps^2/km
  gamma::Float64 # km^-1
end

struct Pulse
  P0::Float64
  T0::Float64
  sech_flag::Float64
end

function solve(fiber::Fiber, pulse::Pulse)
  collision = 0;
  alpha = fiber.alpha * 0.23025 / 1e3;
  time_steps = Int(floor(fiber.T/fiber.dt))
  space_steps = Int(floor(fiber.L/fiber.dz))
  
  LD = pulse.T0^2 / abs(fiber.beta_2)
  LNL = 1 / (fiber.gamma*pulse.P0)
  
  disp_ratio = fiber.L / LD
  nonl_ratio = fiber.L / LNL
  print("Simulation diagnositics:\n")
  @printf("\tT/T0  = %6.4f  window form factor\n", fiber.T/pulse.T0)
  @printf("\tL/LD  = %6.4f\n", disp_ratio)
  @printf("\tL/LNL = %6.4f\n", nonl_ratio)
  @printf("\tN     = %6.4f \n", sqrt(LD / LNL))
  @printf("time steps : %6d \n", time_steps)
  @printf("space steps: %6d \n", space_steps)

  time = LinRange(-fiber.T/2, fiber.T/2, time_steps)
  space = LinRange(0, fiber.L, space_steps)
  
  w = 2*pi* LinRange(-1/(2*fiber.dt), 1/(2*fiber.dt), time_steps)
  w = fftshift(w)
  A = zeros(ComplexF64, time_steps, space_steps)
  A_spect = zeros(ComplexF64, time_steps, space_steps)
  
  if (pulse.sech_flag == 1)
    waveform = sqrt(pulse.P0) .* 2 ./(exp.(time/pulse.T0).+exp.(-time/pulse.T0))
    if collision == 1
      waveform = sqrt(pulse.P0) .* ( 2 ./ (exp.((time.-10*pulse.T0)./pulse.T0).+exp.(-(time.-10*pulse.T0)./pulse.T0)) + 2 ./ (exp.((time.+10*pulse.T0)./pulse.T0).+exp.(-(time.+10*pulse.T0)./pulse.T0)))
    end
  else
    waveform = sqrt(pulse.P0) .* exp.(-(time/pulse.T0) .^ 2 ./ 2)
  end
  
  A[:, 1] = waveform
#  fig = Plots.plot(time, abs.(A[:, 1]), show=true)
  A_spect[:, 1] = fft(A[:, 1])

  fwd_disp = exp.(fiber.dz * (im * fiber.beta_2/2 * w .^ 2 .- alpha/2))
  @showprogress "Propagating the field... " for n = 1:space_steps-1
    A_spect[:, n] = fft(A[:, n])
    A_spect[:, n+1] = A_spect[:, n] .* fwd_disp
    A[:, n+1] = ifft(A_spect[:, n+1])
    A[:, n+1] = A[:, n+1] .* exp.(im * fiber.dz * fiber.gamma .* (abs.(A[:, n+1])).^2)
  end
  A_spect[:, space_steps] = fft(A[:, space_steps])

  tmargin_l = Int(floor(time_steps/2 - 10 * pulse.T0/fiber.dt))
  tmargin_r = Int(ceil(time_steps/2 + 10 * pulse.T0/fiber.dt))
  t_points = 10000
  z_points = 1000
  t_skip = Int(ceil(20 * pulse.T0/fiber.dt /t_points))
  z_skip = Int(ceil(space_steps/z_points))
  fig2 = Plots.heatmap(abs.(A[tmargin_l:t_skip:tmargin_r, 1:z_skip:space_steps]), show=true)
  # fig3 = Plots.surface(abs.(A[tmargin_l:t_skip:tmargin_r, 1:z_skip:space_steps]), show=true)
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


## Fibers
attenuative = Fiber(20e-9,
               1e-12,
               50000,
               100,
               0.2,
               0e-27,
               0.00e-3)
dispersive = Fiber(2e-9,
               1e-13,
               77500,
               100,
               0.0,
               20e-27,
               0.00e-3)
nonlinear = Fiber(20e-9,
               1e-12,
               27000,
               100,
               0.0,
               0e-27,
               3.00e-3)
solitonic = Fiber(20e-9,
               1e-12,
               27000,
               100,
               0.0,
               -4e-27,
               3.00e-3)
smf_28 =Fiber(10e-9,
              1e-13,
              55500,
              100,
              0.18,
              -21.58e-27,
              0.78e-3)
hnlf = Fiber(1e-9,
            1e-13,
            5300,
            1,
            0.81,
            -7.09e-27,
            10.68e-3)
## Pulses      
pulse1 =  Pulse(0.001,
                20e-12,
                0)
pulse2 =  Pulse(0.1,
                20e-12,
                0)
pulse3 =  Pulse(0.46,
                7e-12,
                1)
pulse4 =  Pulse(0.152,
                2.09e-12,
                1)  
high_power =  Pulse(0.152*9,
                2.09e-12,
                1)         
soliton = Pulse(3.4,
                20e-12,
                1)       
## Configuration
configs = [(hnlf, pulse4)      # first order soliton (attenuation limited)
           (hnlf, high_power)] # third order soliton (attenuation limited)


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