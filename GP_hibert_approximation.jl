using AbstractGPs
using KernelFunctions

function get_indices(ml...)
    sm = 1:ml[end]
    for m in ml[end-1:-1:1]
        sm = [repeat(1:m, inner=size(sm, 1)) repeat(sm, m)]
    end
    sm
end

function get_sqrt_lambda(indices, L)
    abs.(π * indices / 2 ./ reshape(L, 1, :))
end

function get_phi(x, sqrt_lam, L)
    x = reshape(x, 1, :)
    L = reshape(L, 1, :)
    return prod(sin.(sqrt_lam .* (x + L)) ./ sqrt.(L), dims=2)[:,1]
end

function get_S(alpha, rho, sqrt_lambda)
    D = length(rho)
    cons = alpha^2 * sqrt(2 * π)^D * prod(rho)
    return cons * exp.(sqrt_lambda.^2 * rho.^2 / (-2))
end

function get_se_hibert_transform(alpha, rho, m_list, L_list)
    indices = get_indices(m_list...)
    sqrt_lambda = get_sqrt_lambda(indices, L_list)
    S = get_S(alpha, rho, sqrt_lambda)
    sqrt_S = sqrt.(S)
    abs_alpha = abs(alpha)
    
    return function(x)
        get_phi(x, sqrt_lambda, L_list) .* sqrt_S * abs_alpha
    end
end

