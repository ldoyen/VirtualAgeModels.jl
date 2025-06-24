using VirtualAgeModels
using DataFrames

m = @vam(Temps & Type ~ (ARA1(.5) | Weibull(0.01,2.5)))
df = simulate(m, @stop(size < 5),system=4)
m.data
data!(m,df)
m.data