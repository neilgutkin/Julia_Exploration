using CUDA
using BenchmarkTools

function axpby_kernel(N, a, b, x, y, z)
    # Collect thread/block indexing
    t = threadIdx().x
    bb = blockIdx().x
    B = blockDim().x
    n = t + (bb-1)*B # 1-based indexing!

    # Perform z=aX+bY
    if(n <= N)
        @inbounds z[n] = a*x[n] + b*y[n]
    end

    return nothing
end

function main()
    # Check invocation
    if (length(ARGS) != 1)
        println("Usage: ./axpby N")
    end

    # Initialize axpby variables
    N = parse(Int32, ARGS[1])
    a, b  = rand(Float64), rand(Float64)
    x, y = rand(Float64, N), rand(Float64, N)
    # Device arrays
    c_x, c_y = CuArray(x), CuArray(y)
    c_z = similar(c_x)

    # Compute ax+by using the kernel
    n_threads = 256
    n_blocks = cld(N, n_threads)
    @cuda threads=n_threads blocks=n_blocks axpby_kernel(N, a, b, c_x, c_y, c_z)
    # Collect result
    z_ker = CuArray(c_z)

    # Compute ax+by using vector ops
    z_vec = CUDA.@sync a.*c_x .+ b.*c_y

    # Get error between kernel-computed and vector-computed z
    z_diff = CUDA.@sync z_vec .- z_ker
    diff = CUDA.@sync sum(z_diff)
    printstyled("Error between kernel and vector functions: $(diff)\n"; color=:red)

    # Benchmark the vector and kernel implementations
    printstyled("Kernel execution:"; color=:green)
    @btime CUDA.@sync @cuda threads=$n_threads blocks=$n_blocks axpby_kernel($N, $a, $b, $c_x, $c_y, $c_z)
    printstyled("Vector execution:"; color=:green)
    @btime CUDA.@sync $a.*$c_x .+ $b.*$c_y
    
    return 0
end

main()