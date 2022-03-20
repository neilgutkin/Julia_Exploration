using CUDA
using BenchmarkTools
using Printf
using Cthulhu

const matrix_width = ARGS[2]

function hostPLU(N::Int32, M::Int32, A::Array{Float64}, LU::Array{Float64}, P::Array{Int32}, tol::Float64)
    
    @inbounds for n in 1:N

        z::Int32 = (n-1)*M*M # Starting index -1

        for j in 1:M
            # Initialize pivots
            P[(n-1)*M+j] = j
            for i in 1:M 
                # Initialize LU
                LU[z+j+M*(i-1)] = A[z+j+M*(i-1)]
            end
        end

        for i in 1:M 
            # Find index of maximum element in column
            maxLU::Float64 = 0
            absLUji::Float64 = 0
            imax = i
            
            for j in i:M
                absLUji = abs(LU[z+j+(i-1)*M])

                if absLUji > maxLU
                    maxLU = absLUji
                    imax = j
                end
            end

            if imax != i
                # Pivoting P
                k::Int32 = P[(n-1)*M+i]
                P[(n-1)*M+i] = P[(n-1)*M+imax]
                P[(n-1)*M+imax] = k

                # Pivoting rows of A
                for j in 1:M
                    tmp::Float64 = LU[z+i+M*(j-1)]
                    LU[z+i+M*(j-1)] = LU[z+imax+M*(j-1)]
                    LU[z+imax+M*(j-1)] = tmp
                end
            end
         
            # Perform LU decomp on row-permuted matrix
            # Loop over rows starting from diagonal
            for j in (i+1):M
            # Check for breakdown
                if (abs(LU[z+i+M*(i-1)]) < tol)
                    @printf("hostLU: matrix diagonal %g indicates LU breakdown, exiting\n",
                            LU[z+i+M*(i-1)])
                end

                LU[z+j+M*(i-1)] /=  LU[z+i+M*(i-1)]

                for t in (i+1):M 
                    LU[z+j+M*(t-1)] -= LU[z+j+M*(i-1)] * LU[z+i+M*(t-1)]
                end
            end
        end
    end

    return nothing
end

function devicePLU(N, M, A, LU, P, tol)
    # Indexing
    n = blockIdx().x
    j = threadIdx().x
    z = (n-1)*M*M

    # Initialize current matrix in shared memory
    s_LUn = @cuStaticSharedMem(Float64, (matrix_width, matrix_width))
    for i in 1:M 
        s_LUn[j,i] = A[z+j+M*(i-1)]
    end

    # Initialize variables for pivot reduction
    P[(n-1)*M+j] = j
    pVal::Float64 = 0
    pInd::Int32 = 0
    topVal::Float64 = 0
    topInd::Int32 = 0
    colSize::Int32 = 0
    alive::Int32 = 0
    imax::Int32 = 0
    tmpInd::Int32 = 0
    tmpVal::Float64 = 0

    # Initialize mask and warp size for shfl operations
    mask::UInt32 = 0xffffffff
    warp::Int32 = 32

    # Loop over columns
    @inbounds for i in 1:M
        # Initialize pivot reduction values and indices
        # Store visually lower values first for reduction functionality
        pInd = M-j+1
        if (j > M-i+1)
            pVal = 0
        else 
            pVal = s_LUn[pInd,i]
        end

        sync_threads()

        # Determine column size and round up to next power of 2
        colSize = 1
        while (colSize < (M-i+1))
            colSize *= 2
        end
        alive = div(colSize, 2)

        # Find pivot row
        while (alive >= 1)
            # Shuffle down values and associated indices
            topVal = shfl_down_sync(mask, pVal, alive, warp)
            topInd = shfl_down_sync(mask, pInd, alive, warp)

            # Only bottom half of threads do comparison
            if (j <= alive && j+alive <= M)
                # Compare and reset as necessary
                if (abs(topVal) - abs(pVal) > 0)
                    pVal = topVal
                    pInd = topInd
                end
            end

            alive = div(alive, 2)
        end

        # Thread 0 shares the result
        imax = shfl_sync(mask, pInd, 1, warp)

        # Swap entries according to reduction results
        if (imax != i)
            if (j == 1)
                tmpInd = P[(n-1)*M+i]
                P[(n-1)*M+i] = P[(n-1)*M+imax]
                P[(n-1)*M+imax] = tmpInd
            end

            tmpVal = s_LUn[i,j]
            s_LUn[i,j] = s_LUn[imax,j]
            s_LUn[imax,j] = tmpVal
        end

        sync_threads()

        # Perform LU decomp on row-permuted matrix
        if (j >= i+1)
            s_LUn[j,i] /= s_LUn[i,i]

            for k in (i+1):M 
                s_LUn[j,k] -= s_LUn[j,i] * s_LUn[i,k]
            end
        end
    end

    # Write results
    @inbounds for i in 1:M 
        LU[z+j+M*(i-1)] = s_LUn[j,i]
    end

    return nothing
end

function main()
    # Check usage
    if length(ARGS) != 2
        println("usage: ./cudaLU totalMatrixEntries matrixWidth")
        exit(1)
    end

    # Parse args and initialize scalars
    Ntotal::Int32 = ARGS[1] 
    M::Int32 = ARGS[2]
    if (M > 32)
        println("Matrix size cannot exceed 32")
    end
    N = cld(Ntotal, M)
    tol = 1e-14

    # Initialize host arrays
    h_A = rand(Float64, N*M*M)
    h_LU = zeros(Float64, N*M*M)
    h_P = zeros(Int32, N*M)

    # Initialize device arrays
    c_A = CuArray(h_A)
    c_LU = CuArray(h_LU)
    c_P = CuArray(h_P)

    # Perform pivoted LU on the host
    hostPLU(N, M, h_A, h_LU, h_P, tol)

    # Perform pivoted LU on the device
    CUDA.@sync @cuda threads=M blocks=N devicePLU(N, M, c_A, c_LU, c_P, tol)

    # Check for differences between host & device implementations
    h_gpuLU = Array(c_LU)
    diff = sum(broadcast(abs, h_LU .- h_gpuLU))
    printstyled("Error between host and device functions: "; color=:red)
    print("$(diff)\n\n")

    # Benchmark host and device implementations
    printstyled("Timing host code:\n"; color=:red)
    @btime hostPLU($N, $M, $h_A, $h_LU, $h_P, $tol) samples=1
    printstyled("Timing device code:\n"; color=:red)
    @btime CUDA.@sync @cuda threads=$M blocks=$N devicePLU($N, $M, $c_A, $c_LU, $c_P, $tol)

    return nothing
end

main()