### A Pluto.jl notebook ###
# v0.11.7

using Markdown
using InteractiveUtils

# ╔═╡ 5e43be36-e2e0-11ea-11fd-7fb62aceb719
begin
	using Plots
	using AbstractGPs
	using KernelFunctions
end

# ╔═╡ 7fdb7d68-e2e0-11ea-352b-bdc15cd99085
include("GP_hibert_approximation.jl");

# ╔═╡ 88d0f150-e2e0-11ea-0309-cf59fab3e8f8
dist(x1,y1,x2,y2) = sqrt((x1-x2)^2 + (y1-y2)^2)

# ╔═╡ a253137c-e2e0-11ea-3a13-0b11565639b6
function recon_path(r, rad, rad_rot, speed, length)
    angle_1 = (rad - 90) / 360 * 2π
    angle_2 = angle_1 + rad_rot / 360 * 2π
    
    angle_1 = -angle_1
    angle_2 = -angle_2
    
    p1_x = r * cos(angle_1)
    p1_y = r * sin(angle_1)
    p2_x = r * cos(angle_2)
    p2_y = r * sin(angle_2)
    
    d1 = dist(p1_x, p1_y, 0, 0)
    d2 = dist(p2_x, p2_y, 0, 0)
    d12 = dist(p1_x, p1_y, p2_x, p2_y)
    d = d1 + d2 + d12
    d1_p = d1 / d
    d2_p = d2 / d
    d12_p = d12 / d
    
    total_t = d / speed
    
    path = Array{Float64}(undef, 3, length)
    for n in 1:length
        p = (n-1) / (length-1)
        if p < d1_p
            pp = p / d1_p
            path[1, n] = pp * p1_x
            path[2, n] = pp * p1_y
        elseif p < (d1_p + d12_p)
            pp = (p - d1_p) / d12_p
            path[1, n] = pp * p2_x + (1 - pp) * p1_x
            path[2, n] = pp * p2_y + (1 - pp) * p1_y
        else
            pp = (p - d1_p - d12_p) / d2_p
            path[1, n] = (1-pp) * p2_x
            path[2, n] = (1-pp) * p2_y
        end
        path[3, n] = p * total_t
    end
    
    return path
end

# ╔═╡ a5a3633a-e2e0-11ea-1850-75ee59eae07c
rad_list = [90, 105, 120, 135, 150, 165]

# ╔═╡ bafbf72e-e2e0-11ea-3e02-4718466e8676
r = 10

# ╔═╡ bd5c4b4a-e2e0-11ea-25d8-ff5e6911d78b
l = 50

# ╔═╡ bf2cfb4a-e2e0-11ea-30c5-65914da5c11b
rad_rot = 7.5

# ╔═╡ c14fce02-e2e0-11ea-3f4f-ddcb5ed08931
path_list = [recon_path(r, rad, rad_rot, r, l) for rad in rad_list]

# ╔═╡ d89e23d8-e2e0-11ea-060a-3ba4775a9b33
let
	f = plot()
	for path in path_list
		plot!(f, path[1,:], path[2,:], legend=false)
		scatter!(f, path[1,:], path[2,:], legend=false)
	end
	f
end

# ╔═╡ 8446f636-e2e3-11ea-0116-5bf9d2b8aafc
x = hcat(path_list...)

# ╔═╡ b4c10ef0-e2e3-11ea-0acc-b161afdb6d24
begin
	x_dummy = Matrix{Float64}(undef, 3, 100*100)
	
	xr = range(0, 10, length=100)
	yr = range(-10, 0, length=100)

	for (i,x) in enumerate(xr)
		for (j,y) in enumerate(yr)
			idx = (i-1) * 100 + j
			x_dummy[1, idx] = x
			x_dummy[2, idx] = y
			x_dummy[3, idx] = 2.1308062584602863
		end
	end
end

# ╔═╡ 868a3ece-e2e5-11ea-1a79-47051621284f
y = zeros(size(x, 2)) .- 1

# ╔═╡ ff873ea6-e2e2-11ea-18ea-89a2edfb966d
alpha = 0.5

# ╔═╡ 0b788e4a-e2e3-11ea-22d6-1daceea6affc
rho = [4., 4.5, 1.5]

# ╔═╡ eea5d3ba-e2e0-11ea-1a0f-0f4f95f05949
M = [5,5,5]

# ╔═╡ f4bfcc7a-e2e2-11ea-3f67-ad05b1246138
L = [15., 15., 15.]

# ╔═╡ 1583d138-e2e3-11ea-2af7-fbcbbf4acfa3
se_hibert_transform_ap = get_se_hibert_transform(alpha, rho, M, L)

# ╔═╡ 24bb3a74-e2e3-11ea-3b43-11958b2bbd62
se_ha_kernel_ap = TransformedKernel(LinearKernel(), FunctionTransform(se_hibert_transform_ap))

# ╔═╡ 342c9124-e2e3-11ea-031a-03568a2dc52f
f = GP(se_ha_kernel_ap)

# ╔═╡ 62a702e4-e2e5-11ea-04b0-cf0be72d36e5
fx = f(KernelFunctions.vec_of_vecs(x), 0.02);

# ╔═╡ 7999f602-e2e5-11ea-32f6-eb4ac0f89cd3
fxy = posterior(fx, y);

# ╔═╡ 94112792-e2e5-11ea-3492-93cc790990c9
fxy_dummy = fxy(KernelFunctions.vec_of_vecs(x_dummy), 0.02);

# ╔═╡ 6382eaae-e2e6-11ea-3d26-b70686d3cc54
mdl = marginals(fxy_dummy)

# ╔═╡ 6cff6618-e2e6-11ea-200a-2f41054f777e
mu_l = [md.μ for md in mdl]

# ╔═╡ 775f0b2c-e2e6-11ea-2ae6-671da8573a08
sigma_l = [md.σ for md in mdl]

# ╔═╡ 089579ec-e2e6-11ea-298a-d11c0af90a4b
contour(xr, yr, reshape(mu_l, 100, 100))

# ╔═╡ Cell order:
# ╠═5e43be36-e2e0-11ea-11fd-7fb62aceb719
# ╠═7fdb7d68-e2e0-11ea-352b-bdc15cd99085
# ╠═88d0f150-e2e0-11ea-0309-cf59fab3e8f8
# ╟─a253137c-e2e0-11ea-3a13-0b11565639b6
# ╠═a5a3633a-e2e0-11ea-1850-75ee59eae07c
# ╠═bafbf72e-e2e0-11ea-3e02-4718466e8676
# ╠═bd5c4b4a-e2e0-11ea-25d8-ff5e6911d78b
# ╠═bf2cfb4a-e2e0-11ea-30c5-65914da5c11b
# ╠═c14fce02-e2e0-11ea-3f4f-ddcb5ed08931
# ╠═d89e23d8-e2e0-11ea-060a-3ba4775a9b33
# ╠═8446f636-e2e3-11ea-0116-5bf9d2b8aafc
# ╠═b4c10ef0-e2e3-11ea-0acc-b161afdb6d24
# ╠═868a3ece-e2e5-11ea-1a79-47051621284f
# ╠═ff873ea6-e2e2-11ea-18ea-89a2edfb966d
# ╠═0b788e4a-e2e3-11ea-22d6-1daceea6affc
# ╠═eea5d3ba-e2e0-11ea-1a0f-0f4f95f05949
# ╠═f4bfcc7a-e2e2-11ea-3f67-ad05b1246138
# ╠═1583d138-e2e3-11ea-2af7-fbcbbf4acfa3
# ╠═24bb3a74-e2e3-11ea-3b43-11958b2bbd62
# ╠═342c9124-e2e3-11ea-031a-03568a2dc52f
# ╠═62a702e4-e2e5-11ea-04b0-cf0be72d36e5
# ╠═7999f602-e2e5-11ea-32f6-eb4ac0f89cd3
# ╠═94112792-e2e5-11ea-3492-93cc790990c9
# ╠═6382eaae-e2e6-11ea-3d26-b70686d3cc54
# ╠═6cff6618-e2e6-11ea-200a-2f41054f777e
# ╠═775f0b2c-e2e6-11ea-2ae6-671da8573a08
# ╠═089579ec-e2e6-11ea-298a-d11c0af90a4b
