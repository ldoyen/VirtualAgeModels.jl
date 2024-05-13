using VirtualAgeModels
using DataFrames
using RCall

m = @vam system & time & type ~ (ARAInf(0.3) | Weibull(0.001,3.0))
df_j = simulate(m, 50,system=40)

ml = mle(m, [1, 2.5, 0.5], df)
res_j = params(ml)

@rput df_j

R"""
	require(VAM)
    s <- sim.vam(system & time & type ~ (ARAInf(0.3) | Weibull(0.001,3)))
    df_r<- simulate(s, 50, nb.system=40)
    m <- mle.vam(system & time & type ~ (ARAInf(0.5) | Weibull(1,2.5)), data = df_j)
    res_r <- coef(m, c(1, 2.5, 0.5)
    resC_r <- contrast(m, c(TRUE,TRUE,TRUE))
	"""
@rget res_r
@rget resC_r