using VirtualAgeModels
using DataFrames
using RCall

m = @vam system & time & type ~ (ARAInf(0.3) | Weibull(0.001,3.0))
df_j = simulate(m, 5,system=3)

#df_j = DataFrame(system=[1,1,1,1,2,2,2,3],time=[3.36,4.04,4.97,5.16,2.34,3.46,5.02,4],type=[-1,-1,-1,0,-1,-1,-1,0])


ml = mle(m, [1, 0.8, 0.5],df_j)
data(ml)

res_j = params(ml)

@rput df_j

R"""
	require(VAM)
    s <- sim.vam(system & time & type ~ (ARAInf(0.3) | Weibull(0.001,3)))
    df_r<- simulate(s, 50, nb.system=40)
    m <- mle.vam(system & time & type ~ (ARAInf(0.5) | Weibull(1,2.5)), data = df_j)
    res_r <- coef(m, c(1, 2.5, 0.5))
    resC_r <- contrast(m, res_r, TRUE,TRUE,TRUE)
    resL_r <- logLik(m, res_r, TRUE,TRUE,TRUE)
	"""
@rget res_r
@rget resC_r
@rget resL_r

data(ml.model)
contrast(ml, res_r, profile=true)
gradient(ml, res_r, profile=true)
hessian(ml, res_r, profile=true)
contrast(ml, res_r, profile=false)
gradient(ml, res_r, profile=false)
hessian(ml, res_r, profile=false)