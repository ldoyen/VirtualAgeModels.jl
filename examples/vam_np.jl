using VirtualAgeModels
using DataFrames
using Plots
using LaTeXStrings

df = DataFrame(System =[1,1,1,1,1,1,2,2,2,2], Time=[3.3,4.0,4.9,5.1,6.4,6.7,2.4,4.5,5.4,6.7], Type=[-1,-1,-1,-1,-1,0,-1,-1,-1,0])
m = @vam System & Time & Type ~ (ARAInf(0.4) | Weibull(0.001,2.5)) 
θ = [0.3,1.4,0.6]
mNP = np(m, 1.0, df)
size(mNP.model.data[1])[1]


VirtualAgeModels.NP_Compute!(mNP,6.,gradient=true)
mNP.Bound_atRiIn[mNP.SortPerm_atRiIn]


tmax = 5.8
N_V = CountingProcessInVA(mNP,[0.4],tmax)
Y_V = AtRiskInVA(mNP,[0.4], tmax)
data!(mNP.model, 1)
infos1 = VirtualAgeModels.virtual_age_infos(mNP.model, mNP.model.time[1], mNP.model.time[end], type=:v)
infos1
data!(mNP.model, 2)
infos2 = VirtualAgeModels.virtual_age_infos(mNP.model, mNP.model.time[1], mNP.model.time[end], type=:v)
infos2
maxy = 3.7
p1 = plot(infos1.x,infos1.y, color=:blue, legend=nothing, ylims=(0, maxy))
#epsilon = 0.05 #to distinguish virtual age of new system on plot
#infos2.y[1] = infos2.y[1] .+ epsilon
tick = (Y_V[!,3], repeat([""],length(Y_V[!,3])))
p1 = plot!(infos2.x, infos2.y, yticks = tick, xticks=[0], color=:red, legend=nothing, xlabel = "Calendar time", ylabel = "Virtual Age")
p1 = plot!([6], seriestype = :vline, color="black")
p1 = scatter!(maximum.(infos1.x[1:(end-1)]), maximum.(infos1.y[1:(end-1)]), legend=nothing, markershape = :x, markercolor= "blue")
p1 = scatter!(maximum.(infos2.x[1:(end-1)]), maximum.(infos2.y[1:(end-1)]), legend=nothing, markershape = :x, markercolor= "red")
x = zeros(length(N_V[!,2]) + 1)
y = zeros(length(N_V[!,2]) + 1)
x[2:end] = N_V[!,1]
y[2:end] = N_V[!,2]
p2 = plot(x/2, y, xticks = 1:maximum(x/2), yticks = tick, ylims=(0, maxy), linetype=:steppre, legend = nothing, color="green", xlabel = "Counting process", ylabel = "Virtual Age")
x2 = zeros(length(Y_V[!,2]) + 2)
y2 = zeros(length(Y_V[!,2]) + 2)
x2[2:(end-1)] = Y_V[!,1]
y2[2:(end-1)] = Y_V[!,2]
y2[end] = Y_V[end,3]
p3 = plot(x2/2, y2, xticks = 1:maximum(x2/2), yticks = tick, ylims=(0, maxy), linetype=:steppre, legend = nothing, color = "green", xlabel = "At-risk", ylabel = "Virtual Age")
p= plot(p3, p2, p1, layout=(1,3))
savefig(p, "../plot2.png")

mNP.nb_CM
mNP.length_atRiIn
mNP.lengthSortPerm_atRiIn
mNP.lengthSortPerm_CMatRiIn
mNP.Bound_atRiIn
mNP.dBound_atRiIn
mNP.BoundTypes_atRiIn
mNP.SortPerm_atRiIn
mNP.SortPermCM_atRiIn
mNP.BoundSlope_atRiIn
mNP.dBoundSlope_atRiIn

ρ = 0.
V = zeros(14)
V[2] = df[1,2]
#syst1
i = 1
V[ 2*i + 1] = (1-ρ) * V[2*i]
i += 1
V[ 2*i] = V[2*i-1] + df[i,2] - df[i-1,2]
V[ 2*i + 1] = (1-ρ) * V[2*i]
i += 1
V[ 2*i] = V[2*i-1] + df[i,2] - df[i-1,2]
V[ 2*i + 1] = (1-ρ) * V[2*i]
i += 1
V[ 2*i] = V[2*i-1] + df[i,2] - df[i-1,2]
#syst2
#V[2*i+1] =0
i += 1
V[ 2*i] = df[i,2]
V[ 2*i + 1] = (1-ρ) * V[2*i]
i += 1
V[ 2*i] = V[2*i-1] + df[i,2] - df[i-1,2]
V[ 2*i + 1] = (1-ρ) * V[2*i]
i += 1
V[ 2*i] = V[2*i-1] + df[i,2] - df[i-1,2]
V
mNP.Bound_atRiIn
V - mNP.Bound_atRiIn

SmoothHazardRateInVA([1.5; 2.5; 3.1], mNP)
SmoothCumHazardRateInVA([1.5; 2.5; 3.1; 3.36; 4; 10], mNP)


m = @vam system & time & type ~ (ARAInf(0.3) | Weibull(0.001,3.0))
θ0 = params(m)
df = simulate(m, 5,system=10)
mNP = np(m, 0.6, df)
ρ = 0.01:0.01:1
f(x) = ContrastNP(mNP,[x])
c = f.(ρ)
plot(ρ,c)

ml = mle(m, [1, 2.5, 0.5], df)
est_θ = params(ml)
x = 0:0.1:30
l(x) = θ0[1] * x^(θ0[2])
est_l(x) = est_θ[1] * x^(est_θ[2])
NPest_l(x) = SmoothCumHazardRateInVA(x, mNP, [est_θ[3]]) #using MLE estimation of ARA part model
plot(x, [l.(x) est_l.(x) NPest_l(convert(Vector{Float64}, x))], title = "Cummulative Hazard rate in VA time scale", label = ["True one" "MLE plug-in estimation" "NP estimation using MLE of ρ"])

ρ = 0.01:0.0001:1
v = 10.1
ϵ = 0.0000001
H = ones(length(ρ))
Y = ones(length(ρ))
dH = ones(length(ρ))
approxdH = ones(length(ρ))
for i in 1:length(ρ)
    H[i] = SmoothCumHazardRateInVA([v] , mNP, [ρ[i]])[1]
    tsY = AtRiskInVA(mNP)
    Y[i] = tsY[sum(tsY[!,2] .< v) , 1]
    dH[i] = dSmoothCumHazardRateInVA([v] , mNP, [ρ[i]])[1]
    approxdH[i] = (SmoothCumHazardRateInVA([v], mNP, [ρ[i] + ϵ]) .- SmoothCumHazardRateInVA([v], mNP, [ρ[i]]))[1] / ϵ
end
plot(ρ, H)
plot(ρ, Y)
plot(ρ, [dH approxdH], label=["computed dH" "approximated dH"])
plot!(ylims = (0, 8))

ρ = 0.1:0.05:1 #0.01:0.01:1
ϵ = 0.0000001
v = 0.:0.05:17.
H = ones(length(v), length(ρ))
dH = ones(length(v), length(ρ))
approxdH = ones(length(v), length(ρ))
for i in 1:length(ρ)
    H[:,i] = SmoothCumHazardRateInVA(convert(Vector{Float64},v) , mNP, [ρ[i]])
    dH[:,i] = dSmoothCumHazardRateInVA(convert(Vector{Float64},v) , mNP, [ρ[i]])
    approxdH[:,i] = (SmoothCumHazardRateInVA(convert(Vector{Float64},v), mNP, [ρ[i] + ϵ]) .- SmoothCumHazardRateInVA(convert(Vector{Float64},v), mNP, [ρ[i]])) ./ ϵ
end
dH .- approxdH
maximum(abs.(dH .-approxdH))
surface(ρ,v,H,xlabel="rho",ylabel="v",zlabel="H")
surface(ρ,v,dH,xlabel="rho",ylabel="v",zlabel="d_{rho}H")

