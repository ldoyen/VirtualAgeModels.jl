mutable struct NP
    model::Model
    data::Union{DataFrame,Vector{DataFrame}}#The complet data set which can differ from the one used in model because only event befor t_max are considered for computuation

    #smoothing kernel:
    k_bandwidth::Float64
    k_d::UnivariateDistribution

    t_max_data::Float64 #maximum calendar time:
    t_max::Float64 #maximum considered calendar time :
     #all maintenances taking place after that times are note considered

    nb_CM::Int64#total number of CM
    length_atRiIn::Int64#length of the following vectors
    lengthSortPerm_atRiIn::Int64
    lengthSortPerm_CMatRiIn::Int64

    #in virtual age time scale
    Bound_atRiIn::Vector{Float64}#bounds of at risks intervals
    BoundTypes_atRiIn::Vector{Int64}#types of bounds of at risks intervals (left: 1 and right: -1 if CM and 0 otherwise) 
    SortPerm_atRiIn::Vector{Int64}#permutation vector to sort Bound_atRiIn values
    Nb_atRiIn::Vector{Int64}#number of intervals that are at risk at corresponding Bound_atRiIn times
    dBound_atRiIn::Matrix{Float64}#derivatives (vs maintenance parameters) of bounds of at risks intervals
    d2Bound_atRiIn::Matrix{Float64}#second order derivatives (vs maintenance parameters) of bounds of at risks intervals
    BoundSlope_atRiIn::Vector{Float64}#successive slopes of the virtual age, CAREFUL ! there are half many slope as there are bounds
    dBoundSlope_atRiIn::Matrix{Float64}#derivatives (vs maintenance parameters) of successive slopes of the virtual age 
    d2BoundSlope_atRiIn::Matrix{Float64}#second order derivatives (vs maintenance parameters) of successive slopes of the virtual age 
    SortPermCM_atRiIn::Vector{Int64}#permutation vector to sort Bound_atRiIn values corresponding only to CM actions
    SortPermCM_Y_atRiIn::Vector{Int64}#permutation vector given the At risk value corresponding to the previous bounds


    #Does the previous different quantities in virtual age time scale have been computes
    is_V_computed::Bool
    is_dV_computed::Bool
    is_d2V_computed::Bool
    #and with which parameters values of the maintenance parameters
    θ_maint::Vector{Float64}

    NP() = new()
end

#For now, only Gaussian kernel are considered
kernel_dist(::Type{Normal},bandwidth::Float64)::UnivariateDistribution = Normal(0.0,bandwidth)

function init!(np::NP)
    np.t_max_data = 0
    np.length_atRiIn = 0
    np.nb_CM = 0
    for i in 1:(np.model.nb_system)
        dataSysti = data(np.model,i)
        d = size(dataSysti)[1]
        np.length_atRiIn += 2 * d #each maintenance time corresponds to a right and left VA except the last one for which there is only a left VA
        np.t_max_data = max(np.t_max_data , maximum(dataSysti[!,1]))
        np.nb_CM += sum(dataSysti[!,2] .== (-1))
    end
    np.lengthSortPerm_CMatRiIn = np.nb_CM
    np.lengthSortPerm_atRiIn = np.length_atRiIn
    np.Bound_atRiIn = zeros(np.length_atRiIn) #first bound is 0
    np.BoundTypes_atRiIn = ones(Int64, np.length_atRiIn) #first bound 0 is a left bound
    np.BoundSlope_atRiIn = zeros(convert(Int64, np.length_atRiIn / 2)) #first slope is 1
    np.SortPerm_atRiIn = ones(Int64, np.length_atRiIn)
    np.Nb_atRiIn = zeros(Int64, np.length_atRiIn)
    np.dBound_atRiIn = zeros(np.length_atRiIn , np.model.nb_params_maintenance)# derivatives of first bound 0 are null
    np.d2Bound_atRiIn = zeros(np.length_atRiIn , ind_nb(np.model.nb_params_maintenance))
    np.dBoundSlope_atRiIn = zeros(convert(Int64, np.length_atRiIn / 2) , np.model.nb_params_maintenance)# derivatives of first slope (1) are null
    np.d2BoundSlope_atRiIn = zeros(convert(Int64, np.length_atRiIn / 2) , ind_nb(np.model.nb_params_maintenance))
    np.SortPermCM_atRiIn = zeros(Int64, np.nb_CM)
    np.SortPermCM_Y_atRiIn = zeros(Int64, np.nb_CM)
    np.θ_maint = params(np.model)[(np.model.nb_params_family + 1) : (np.model.nb_params_family + np.model.nb_params_maintenance)]
    np.is_V_computed = false
    np.is_dV_computed = false
    np.is_d2V_computed = false
end

#Only Normal kernel for now
function NP(model::Model , bandwidth::Float64 , data::DataFrame=DataFrame(); kernel=Normal)::NP
    np = NP()
    np.k_bandwidth = bandwidth
    np.k_d = kernel_dist(kernel,bandwidth)
    np.model = model
    init!(np.model)
    ## only if not already set!
    ###like MLE but why ??
    np.data = data
    if data==DataFrame()
        np.data = np.model.data 
        #data!(np.model, data, DataFrame())#covariates not considered in this case for now
    end
    init!(np)
    return np
end

function np(model::Model , bandwidth::Float64 , data::DataFrame=DataFrame(); kernel=Normal)::NP
    np = NP(model, bandwidth, data, kernel=kernel)
    return np
end

function NP_Compute!(np::NP , t_max::Float64=np.t_max_data ; gradient::Bool=false, hessian::Bool=false )
    np.is_V_computed=true
    np.t_max=t_max
    i = 1   
    for k in 1:np.model.nb_system
        data!(np.model, k)
        n = sum(np.model.time .<= np.t_max)
        change_lasttime = false
        if n<length(np.model.time)#adding end of observation calendar time in data
            change_lasttime = true
            n = n+1
            last_time = np.model.time[n]
            last_type = np.model.type[n]
            np.model.time[n] = np.t_max
            np.model.type[n] = 0#like censoring
        end

        #init
        for mm in np.model.models
            init!(mm)
        end
        np.model.Vright = 0
        np.model.Vleft = 0
        np.model.A = 1
        np.model.k = 1
        np.model.id_mod = 0 #id of current model
        np.model.id_params = 1
        init!(np.model.comp, deriv=gradient|hessian)
        for type in np.model.type
            if type < 0 
                np.model.comp.S0 += 1
            end
        end
        if gradient
            fill!(np.model.dVright, 0.0)
            fill!(np.model.dA, 0.0)
        end
        if hessian
            fill!(np.model.d2Vright, 0.0)
            fill!(np.model.d2A, 0.0)
        end

        while np.model.k < n
            update_Vleft!(np.model, gradient=gradient, hessian=hessian)
            np.Bound_atRiIn[i] = np.model.Vright
            np.BoundTypes_atRiIn[i] = 1
            np.Bound_atRiIn[i+1] = np.model.Vleft
            np.BoundTypes_atRiIn[i+1] = (np.model.type[np.model.k + 1] < 0 ? -1 : 0)
            np.BoundSlope_atRiIn[convert(Int64, (i + 1) / 2) ] = np.model.A
            if gradient
                np.is_dV_computed=true
                np.dBound_atRiIn[i,:] .= np.model.dVright
                np.dBound_atRiIn[i+1,:] .= np.model.dVleft
                np.dBoundSlope_atRiIn[convert(Int64, (i + 1) / 2),:] .= np.model.dA
            end
            if hessian
                np.is_d2V_computed=true
                np.d2Bound_atRiIn[i,:] .= np.model.d2Vright
                np.d2Bound_atRiIn[i+1,:] .= np.model.d2Vleft
                np.d2BoundSlope_atRiIn[convert(Int64, (i + 1) / 2),:] .= np.model.d2A
            end
            i += 2
            #// previous model for the next step
            type = np.model.type[np.model.k + 1]
            if type < 0 
                type = 0
            end
            #//model.indMode = (type < 0 ? 0 : type);
            update_maintenance!(np.model, type, gradient=gradient, hessian=hessian)
        end

        if change_lasttime #put back last data value if changed
            np.model.time[n] = last_time
            np.model.type[n] = last_type
        end
    end
    np.lengthSortPerm_atRiIn = i-1
    np.SortPerm_atRiIn = sortperm(np.Bound_atRiIn[1:np.lengthSortPerm_atRiIn])
    j = 1
    k = 1
    indSortPerm_ok = ones(Int64,np.lengthSortPerm_atRiIn)
    ind_last = 1
    np.Nb_atRiIn[np.SortPerm_atRiIn[1]] = 1
    np.SortPermCM_atRiIn = ones(Float64,np.nb_CM)
    np.SortPermCM_Y_atRiIn = ones(Float64,np.nb_CM)
    for i in 2:np.lengthSortPerm_atRiIn
        if np.Bound_atRiIn[np.SortPerm_atRiIn[ind_last]] == np.Bound_atRiIn[np.SortPerm_atRiIn[i]]
            #same bound as previous, do not duplicate
            if np.BoundTypes_atRiIn[np.SortPerm_atRiIn[i]] <= 0
                #end bound of at risk interval: Vleft
                np.Nb_atRiIn[np.SortPerm_atRiIn[ind_last]] -= 1
                if np.BoundTypes_atRiIn[np.SortPerm_atRiIn[i]] < 0
                    #CM: virtual age event time for the counting process
                    np.SortPermCM_Y_atRiIn[j] = np.SortPerm_atRiIn[ind_last-1]
                    np.SortPermCM_atRiIn[j] = np.SortPerm_atRiIn[ind_last]
                    j += 1
                end
            else
                #begining bound of at risk interval: Vright
                np.Nb_atRiIn[np.SortPerm_atRiIn[ind_last]] += 1
            end
        else
            if np.BoundTypes_atRiIn[np.SortPerm_atRiIn[i]] <= 0
                #end bound of at risk interval: Vleft
                np.Nb_atRiIn[np.SortPerm_atRiIn[i]] = np.Nb_atRiIn[np.SortPerm_atRiIn[ind_last]] - 1
                if np.BoundTypes_atRiIn[np.SortPerm_atRiIn[i]] < 0
                    #CM: virtual age event time for the counting process
                    np.SortPermCM_Y_atRiIn[j] = np.SortPerm_atRiIn[ind_last]
                    np.SortPermCM_atRiIn[j] = np.SortPerm_atRiIn[i]
                    j += 1
                end
            else
                #begining bound of at risk interval: Vright
                np.Nb_atRiIn[np.SortPerm_atRiIn[i]] = np.Nb_atRiIn[np.SortPerm_atRiIn[ind_last]] + 1
            end
            ind_last = i
            k += 1
            indSortPerm_ok[k] = i
        end
    end
    np.SortPerm_atRiIn = np.SortPerm_atRiIn[indSortPerm_ok[1:k]]
    np.SortPermCM_atRiIn = np.SortPermCM_atRiIn[1:(j-1)]
    np.SortPermCM_Y_atRiIn = np.SortPermCM_Y_atRiIn[1:(j-1)]
    np.lengthSortPerm_atRiIn = k
    np.lengthSortPerm_CMatRiIn = j-1

end

function Need_to_Compute_V!(np::NP, θ_maint::Vector{Float64}=np.θ_maint, t_max::Float64=np.t_max_data ; gradient::Bool=false, hessian::Bool=false )
    if  ((gradient && !np.is_dV_computed) ||
            (hessian && !np.is_d2V_computed) ||
            (!np.is_V_computed) || 
            (np.θ_maint != θ_maint) || 
            (t_max != np.t_max)
            )
        
        #Values of the derivatives of the virtual age are no more valid if the parameter value or t_max has changed
        np.is_dV_computed = (np.is_dV_computed) && !(((np.θ_maint != θ_maint) || (t_max != np.t_max)))
        np.is_d2V_computed = (np.is_d2V_computed) && !(((np.θ_maint != θ_maint) || (t_max != np.t_max))) 
        
        np.θ_maint = θ_maint
        θ = params(np.model)
        θ[(np.model.nb_params_family + 1) : (np.model.nb_params_family + np.model.nb_params_maintenance)] = np.θ_maint
        params!(np.model , θ)
        NP_Compute!(np , t_max, gradient=gradient, hessian=hessian)
    end
end

#####
##TODO:
##All the following functions does not take into count the slope of the virtual age
#####

function CountingProcessInVA(np::NP, θ_maint::Vector{Float64}=np.θ_maint, t_max::Float64=np.t_max_data)
    Need_to_Compute_V!(np, θ_maint, t_max) 
    return DataFrame(N = 1:np.lengthSortPerm_CMatRiIn, V = np.Bound_atRiIn[np.SortPermCM_atRiIn[1:np.lengthSortPerm_CMatRiIn]])
end

function AtRiskInVA(np::NP, θ_maint::Vector{Float64}=np.θ_maint, t_max::Float64=np.t_max_data)
    Need_to_Compute_V!(np, θ_maint, t_max) 
    return DataFrame(Y = np.Nb_atRiIn[np.SortPerm_atRiIn[1:(np.lengthSortPerm_atRiIn-1)]], 
                     V_inf = np.Bound_atRiIn[np.SortPerm_atRiIn[1:(np.lengthSortPerm_atRiIn-1)]],
                     V_sup = np.Bound_atRiIn[np.SortPerm_atRiIn[2:(np.lengthSortPerm_atRiIn)]])
end

function SmoothHazardRateInVA(x::Vector{Float64}, np::NP, θ_maint::Vector{Float64}=np.θ_maint, t_max::Float64=np.t_max_data)
    Need_to_Compute_V!(np, θ_maint, t_max) 
    nx = length(x)
    res = zeros(nx)
    for j in 1:nx
        for i in 1:np.lengthSortPerm_CMatRiIn
            res[j] += (pdf(np.k_d, x[j] - np.Bound_atRiIn[np.SortPermCM_atRiIn[i]])) / (np.Nb_atRiIn[np.SortPermCM_Y_atRiIn[i]])
        end
    end
    return res
end

function SmoothCumHazardRateInVA(x::Vector{Float64}, np::NP, θ_maint::Vector{Float64}=np.θ_maint, t_max::Float64=np.t_max_data)
    Need_to_Compute_V!(np, θ_maint, t_max) 
    nx = length(x)
    res = zeros(nx)
    for j in 1:nx
        for i in 1:np.lengthSortPerm_CMatRiIn
            res[j] += (cdf(np.k_d, x[j] - np.Bound_atRiIn[np.SortPermCM_atRiIn[i]])) / (np.Nb_atRiIn[np.SortPermCM_Y_atRiIn[i]])
        end
    end
    return res
end

function dSmoothCumHazardRateInVA(x::Vector{Float64}, np::NP, θ_maint::Vector{Float64}=np.θ_maint, t_max::Float64=np.t_max_data)
    Need_to_Compute_V!(np, θ_maint, t_max, gradient=true) 
    nx = length(x)
    res = zeros(nx,np.model.nb_params_maintenance)
    for j in 1:nx
        for i in 1:np.lengthSortPerm_CMatRiIn
            res[j,:] .-= np.dBound_atRiIn[np.SortPermCM_atRiIn[i],] .* ((pdf(np.k_d, x[j] - np.Bound_atRiIn[np.SortPermCM_atRiIn[i]])) / (np.Nb_atRiIn[np.SortPermCM_Y_atRiIn[i]]))
        end
    end
    return res
end

function ContrastNP(np::NP, θ_maint::Vector{Float64}=np.θ_maint, t_max::Float64=np.t_max_data)
    Need_to_Compute_V!(np, θ_maint, t_max) 
    res = 0.0
    for i in 1:np.lengthSortPerm_CMatRiIn
        npint = 0.0
        x = np.Bound_atRiIn[np.SortPermCM_atRiIn[i]]
        for j in 1:np.lengthSortPerm_CMatRiIn
            npint += (pdf(np.k_d, x - np.Bound_atRiIn[np.SortPermCM_atRiIn[j]])) / (np.Nb_atRiIn[np.SortPermCM_Y_atRiIn[j]])
        end
        res += log(npint)
    end
    return res
end