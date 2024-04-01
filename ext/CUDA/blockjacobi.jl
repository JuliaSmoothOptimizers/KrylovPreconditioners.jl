KP.BlockJacobiPreconditioner(J::CUSPARSE.CuSparseMatrixCSR; options...) = BlockJacobiPreconditioner(SparseMatrixCSC(J); options...)

function KP.create_blocklist(cublocks::CuArray, npart)
    blocklist = Array{CuMatrix{Float64}}(undef, npart)
    for b in 1:npart
        blocklist[b] = CuMatrix{Float64}(undef, size(cublocks)...)
    end
    return blocklist
end

function _update_gpu(p, j_rowptr, j_colval, j_nzval, device::CUDABackend)
    nblocks = p.nblocks
    blocksize = p.blocksize
    fillblock_gpu_kernel! = KP._fillblock_gpu!(device)
    # Fill Block Jacobi" begin
    fillblock_gpu_kernel!(
        p.cublocks, size(p.id,1),
        p.cupartitions, p.cumap,
        j_rowptr, j_colval, j_nzval,
        p.cupart, p.culpartitions, p.id,
        ndrange=(nblocks, blocksize),
    )
    KA.synchronize(device)
    # Invert blocks begin
    for b in 1:nblocks
        p.blocklist[b] .= p.cublocks[:,:,b]
    end
    CUDA.@sync pivot, info = CUBLAS.getrf_batched!(p.blocklist, true)
    CUDA.@sync pivot, info, p.blocklist = CUBLAS.getri_batched(p.blocklist, pivot)
    for b in 1:nblocks
        p.cublocks[:,:,b] .= p.blocklist[b]
    end
    return
end

"""
    function update!(J::CuSparseMatrixCSR, p)

Update the preconditioner `p` from the sparse Jacobian `J` in CSR format for CUDA

1) The dense blocks `cuJs` are filled from the sparse Jacobian `J`
2) To a batch inversion of the dense blocks using CUBLAS
3) Extract the preconditioner matrix `p.P` from the dense blocks `cuJs`

"""
function KP.update!(p::BlockJacobiPreconditioner, J::CUSPARSE.CuSparseMatrixCSR)
    _update_gpu(p, J.rowPtr, J.colVal, J.nzVal, p.device)
end
