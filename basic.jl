### A Pluto.jl notebook ###
# v0.11.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 9945486e-e2d4-11ea-1821-595d5640259f
begin
	using AbstractGPs
	using KernelFunctions
	using Plots
end

# ╔═╡ f2ed8e58-e2d4-11ea-19b1-a13e091d9d58
function get_indices(ml...)
    sm = 1:ml[end]
    for m in ml[end-1:-1:1]
        sm = [repeat(1:m, inner=size(sm, 1)) repeat(sm, m)]
    end
    sm
end

# ╔═╡ 0df8323c-e2d5-11ea-1a27-b142cde516fe
function get_sqrt_lambda(indices, L)
    abs.(π * indices / 2 ./ reshape(L, 1, :))
end

# ╔═╡ 181a8a30-e2d5-11ea-1b4d-1334e2098827
function get_phi(x, sqrt_lam, L)
    x = reshape(x, 1, :)
    L = reshape(L, 1, :)
    return prod(sin.(sqrt_lam .* (x + L)) ./ sqrt.(L), dims=2)[:,1]
end

# ╔═╡ 19ff6406-e2d5-11ea-1b85-113aa473a6fd
function get_S(alpha, rho, sqrt_lambda)
    D = length(rho)
    cons = alpha^2 * sqrt(2 * π)^D * prod(rho)
    return cons * exp.(sqrt_lambda.^2 * rho.^2 / (-2))
end

# ╔═╡ 1ff71ac0-e2d5-11ea-306e-0d71815fb0b7
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

# ╔═╡ 43060f30-e2d5-11ea-205d-29a382e73740
se_hibert_transform = get_se_hibert_transform(1., [1., 2., 3.], [2, 3, 4], [1.5, 1.5, 1.5])

# ╔═╡ 483c7200-e2d5-11ea-3bf7-f7660f4f8f43
se_hibert_transform([1.,2.,3])' * se_hibert_transform([2.,3.,4.])

# ╔═╡ 4c6e9c06-e2d5-11ea-3ea5-019d341523cf
se_ha_kernel = TransformedKernel(LinearKernel(), FunctionTransform(se_hibert_transform)) 
# Yeah, kernel of Hibert approximation is just a basic LinearKernel with special transformation.
# Though it's not efficient to explicitly calculate the Covariance matrix, sampling is the true use case.

# ╔═╡ 53c8ec34-e2d5-11ea-0464-5bdc8e7c78c3
se_ha_kernel([1.,2.,3], [2.,3.,4.])

# ╔═╡ 5df6f66a-e2d5-11ea-1b39-691263db15bb
# This function is borrowed from https://github.com/gabriuma/basis_functions_approach_to_GP/blob/master/uni_dimensional/BF_1D_notebook.Rmd
F = function(x)
    1/10*(x/10+1.7)^3 - 2/5*(x/10+1.7)^2 + 1/5*sin((x/10+1.7)^3)
end

# ╔═╡ 98775bfe-e2d5-11ea-3f6c-bde78e9ced36
x = range(-5, 5, step=0.1)

# ╔═╡ ada5139a-e2d5-11ea-0730-c9da4b32e83a
N = length(x)

# ╔═╡ b4750400-e2d5-11ea-2e2e-8540cf2c6084
y = F.(x) + randn(N) * 0.08

# ╔═╡ b91c682c-e2d5-11ea-157e-a7daef19f016
begin
	plot(x, F.(x), color=:black, legend=false)
	scatter!(x, y, markersize=2, color=:black)
end

# ╔═╡ f6a450d8-e2d5-11ea-39d0-bf840ddcec52
alpha = 2.

# ╔═╡ fd513a4a-e2d5-11ea-2efa-893014ecd0bb
rho = [1.]

# ╔═╡ d433c22c-e2d5-11ea-3fba-e77eb854a96b
M = [40]

# ╔═╡ cfde845a-e2d5-11ea-00cf-a99f700df72e
L = [2.5 * maximum(x)] # we treat 1d as special case of nD cases so [x] is required

# ╔═╡ 00184714-e2d6-11ea-10e9-bb1ce8081a86
se_hibert_transform_ap = get_se_hibert_transform(alpha, rho, M, L)

# ╔═╡ 5a6af630-e2d6-11ea-1997-610f4712ee57
se_ha_kernel_ap = TransformedKernel(LinearKernel(), FunctionTransform(se_hibert_transform_ap))

# ╔═╡ d91a9ff6-e2d6-11ea-0887-874277fce3da
f = GP(se_ha_kernel_ap);

# ╔═╡ e1bc2c62-e2d6-11ea-00df-959849ab67da
fx = f(map(x->[x], x), 0.08);

# ╔═╡ e798424c-e2d6-11ea-39d7-450c82235f41
plot(x, [rand(fx) for i in 1:5], legend=false)

# ╔═╡ f7af3de6-e2d6-11ea-3b78-21fa0219d6f8
fxy = posterior(fx, y);

# ╔═╡ 4538740a-e2d7-11ea-345c-81c8dbd7816b
mdl = marginals(fxy(map(x->[x], x)))

# ╔═╡ 4d78f99e-e2d7-11ea-2733-e10b44ddfb80
mu_l = [d.μ for d in mdl]

# ╔═╡ 71d88e76-e2d7-11ea-00fe-0bf6fb9cc8cd
sigma_l = [d.σ for d in mdl]

# ╔═╡ 77e26468-e2d7-11ea-12fc-8f14dc773419
begin
	plot(x, F.(x), color=:black, legend=false)
	scatter!(x, y, markersize=2, color=:black)
	plot!(x, mu_l, ribbon=sigma_l, fillalpha=0.5)
end

# ╔═╡ 586c318a-e2d8-11ea-0498-e780b7cb9fdb
f_exact = GP(SEKernel())

# ╔═╡ ad786be2-e2d8-11ea-2751-9d5640c68731
f_exact_x = f_exact(x, 0.08);

# ╔═╡ b192f2e4-e2d8-11ea-132c-35a6666e93ad
f_exact_x_y = posterior(f_exact_x, y);

# ╔═╡ b67c6cb8-e2d8-11ea-2524-dfb36f823166
mdl_exact = marginals(f_exact_x_y(x))

# ╔═╡ 04e457ce-e2db-11ea-27c0-4706dd5d94ad
mu_l_exact = [d.μ for d in mdl_exact]

# ╔═╡ c2b192d8-e2d8-11ea-19a1-b94e1d7366d9
sigma_l_exact = [d.σ for d in mdl_exact]

# ╔═╡ c88f0a5a-e2d8-11ea-2907-4d11cd07f26e
begin
	plot(x, F.(x), color=:black, legend=false)
	scatter!(x, y, markersize=2, color=:black)
	plot!(x, mu_l_exact, ribbon=sigma_l_exact)
end

# ╔═╡ d62c3532-e2d8-11ea-1880-ed8cd96d2d21
begin
	scatter(x, y, markersize=2, color=:black, label="")
	
	plot!(x, mu_l, color=:blue, label="BF model, M=$(M[1]) L=$(L[1])")
	plot!(x, mu_l + sigma_l, color=:blue, linestyle=:dash, label="")
	plot!(x, mu_l - sigma_l, color=:blue, linestyle=:dash, label="")
	
	plot!(x, mu_l_exact, color=:red, label="GP model")
	plot!(x, mu_l_exact + sigma_l_exact, color=:red, linestyle=:dash, label="")
	plot!(x, mu_l_exact - sigma_l_exact, color=:red, linestyle=:dash, label="")
end

# ╔═╡ 9d288402-e2da-11ea-1024-d769ec84030d
function get_mu_sigma_l(alpha, rho1, M1, L1)
	rho = [rho1]
	M = [M1]
	L = [L1]
	
	se_hibert_transform_ap = get_se_hibert_transform(alpha, rho, M, L)
	se_ha_kernel_ap = TransformedKernel(LinearKernel(), FunctionTransform(se_hibert_transform_ap))
	f = GP(se_ha_kernel_ap);
	fx = f(map(x->[x], x), 0.08);
	fxy = posterior(fx, y);
	mdl = marginals(fxy(map(x->[x], x)))
	mu_l = [d.μ for d in mdl]
	sigma_l = [d.σ for d in mdl]
	return mu_l, sigma_l
end

# ╔═╡ d66597fc-e2dd-11ea-212f-d118aa9cb8fb
@bind alpha_bind html"alpha <input type=range min=0.5 max=3 step=0.1 value=2>"

# ╔═╡ caab2b40-e2db-11ea-3824-d9021a7b7e7c
@bind rho1 html"rho <input type=range step=0.01 min=0.5 max=2 value=1>"

# ╔═╡ 2fb4f5b2-e2db-11ea-0a6b-ed160dae834a
@bind M1 html"M <input type=range min=1 max=50 value=5>"

# ╔═╡ fb380ed6-e2db-11ea-1531-797d85d73807
@bind L1 html"L <input type=range min=1 max=20 step=0.01 value=12.5>"

# ╔═╡ 1744e8ae-e2dc-11ea-150d-5b7a4bd6256c
"alpha=$(alpha_bind) rho=$(rho1) M=$(M1) L=$(L1)"

# ╔═╡ c148c2b8-e2dd-11ea-04eb-2585a820508d
begin
	mu_l_bind, sigma_l_bind = get_mu_sigma_l(alpha_bind, rho1, M1, L1)
	
	scatter(x, y, markersize=2, color=:black, label="")
	
	plot!(x, mu_l_bind, color=:blue, label="BF model, M=$(M1) L=$(L1)")
	plot!(x, mu_l_bind + sigma_l_bind, color=:blue, linestyle=:dash, label="")
	plot!(x, mu_l_bind - sigma_l_bind, color=:blue, linestyle=:dash, label="")
	
	plot!(x, mu_l_exact, color=:red, label="GP model")
	plot!(x, mu_l_exact + sigma_l_exact, color=:red, linestyle=:dash, label="")
	plot!(x, mu_l_exact - sigma_l_exact, color=:red, linestyle=:dash, label="")
end

# ╔═╡ Cell order:
# ╠═9945486e-e2d4-11ea-1821-595d5640259f
# ╠═f2ed8e58-e2d4-11ea-19b1-a13e091d9d58
# ╠═0df8323c-e2d5-11ea-1a27-b142cde516fe
# ╠═181a8a30-e2d5-11ea-1b4d-1334e2098827
# ╠═19ff6406-e2d5-11ea-1b85-113aa473a6fd
# ╠═1ff71ac0-e2d5-11ea-306e-0d71815fb0b7
# ╠═43060f30-e2d5-11ea-205d-29a382e73740
# ╠═483c7200-e2d5-11ea-3bf7-f7660f4f8f43
# ╠═4c6e9c06-e2d5-11ea-3ea5-019d341523cf
# ╠═53c8ec34-e2d5-11ea-0464-5bdc8e7c78c3
# ╠═5df6f66a-e2d5-11ea-1b39-691263db15bb
# ╠═98775bfe-e2d5-11ea-3f6c-bde78e9ced36
# ╠═ada5139a-e2d5-11ea-0730-c9da4b32e83a
# ╠═b4750400-e2d5-11ea-2e2e-8540cf2c6084
# ╠═b91c682c-e2d5-11ea-157e-a7daef19f016
# ╠═f6a450d8-e2d5-11ea-39d0-bf840ddcec52
# ╠═fd513a4a-e2d5-11ea-2efa-893014ecd0bb
# ╠═d433c22c-e2d5-11ea-3fba-e77eb854a96b
# ╠═cfde845a-e2d5-11ea-00cf-a99f700df72e
# ╠═00184714-e2d6-11ea-10e9-bb1ce8081a86
# ╠═5a6af630-e2d6-11ea-1997-610f4712ee57
# ╠═d91a9ff6-e2d6-11ea-0887-874277fce3da
# ╠═e1bc2c62-e2d6-11ea-00df-959849ab67da
# ╠═e798424c-e2d6-11ea-39d7-450c82235f41
# ╠═f7af3de6-e2d6-11ea-3b78-21fa0219d6f8
# ╠═4538740a-e2d7-11ea-345c-81c8dbd7816b
# ╠═4d78f99e-e2d7-11ea-2733-e10b44ddfb80
# ╠═71d88e76-e2d7-11ea-00fe-0bf6fb9cc8cd
# ╠═77e26468-e2d7-11ea-12fc-8f14dc773419
# ╠═586c318a-e2d8-11ea-0498-e780b7cb9fdb
# ╠═ad786be2-e2d8-11ea-2751-9d5640c68731
# ╠═b192f2e4-e2d8-11ea-132c-35a6666e93ad
# ╠═b67c6cb8-e2d8-11ea-2524-dfb36f823166
# ╠═04e457ce-e2db-11ea-27c0-4706dd5d94ad
# ╠═c2b192d8-e2d8-11ea-19a1-b94e1d7366d9
# ╠═c88f0a5a-e2d8-11ea-2907-4d11cd07f26e
# ╠═d62c3532-e2d8-11ea-1880-ed8cd96d2d21
# ╠═9d288402-e2da-11ea-1024-d769ec84030d
# ╟─d66597fc-e2dd-11ea-212f-d118aa9cb8fb
# ╟─caab2b40-e2db-11ea-3824-d9021a7b7e7c
# ╟─2fb4f5b2-e2db-11ea-0a6b-ed160dae834a
# ╟─fb380ed6-e2db-11ea-1531-797d85d73807
# ╟─1744e8ae-e2dc-11ea-150d-5b7a4bd6256c
# ╠═c148c2b8-e2dd-11ea-04eb-2585a820508d
