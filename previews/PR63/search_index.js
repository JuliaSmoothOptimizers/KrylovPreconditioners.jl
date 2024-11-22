var documenterSearchIndex = {"docs":
[{"location":"triangular_operators/#triangular_operators","page":"Triangular operators","title":"Triangular operators","text":"","category":"section"},{"location":"triangular_operators/","page":"Triangular operators","title":"Triangular operators","text":"TriangularOperator\nupdate!(::AbstractTriangularOperator, ::Any)","category":"page"},{"location":"triangular_operators/#KrylovPreconditioners.TriangularOperator","page":"Triangular operators","title":"KrylovPreconditioners.TriangularOperator","text":"TriangularOperator(A, uplo::Char, diag::Char; nrhs::Int=1, transa::Char='N')\n\nCreate a triangular operator for efficient solution of sparse triangular systems on GPU architectures.  Supports sparse matrices stored on NVIDIA, AMD, and Intel GPUs.\n\nInput arguments\n\nA: A sparse matrix on the GPU representing the triangular system to be solved;\nuplo: Specifies whether the triangular matrix A is upper triangular ('U') or lower triangular ('L');\ndiag: Indicates whether the diagonal is unit ('U') or non-unit ('N');\nnrhs: Specifies the number of columns for the right-hand side(s). Defaults to 1, corresponding to solving triangular systems with a single vector as the right-hand side;\ntransa: Determines how the matrix A is applied during the triangle solves; 'N' for no transposition, 'T' for transpose, and 'C' for conjugate transpose.\n\nOutput argument\n\nop: An instance of AbstractTriangularOperator representing the triangular operator for the specified sparse matrix and parameters.\n\n\n\n\n\n","category":"function"},{"location":"triangular_operators/#KrylovPreconditioners.update!-Tuple{AbstractTriangularOperator, Any}","page":"Triangular operators","title":"KrylovPreconditioners.update!","text":"update!(op::AbstractTriangularOperator, A)\n\nUpdate the sparse matrix A associated with the given AbstractTriangularOperator without the need to reallocate buffers  or repeat the structural analysis phase for detecting parallelism for sparse triangular solves. A and the operator op must have the same sparsity pattern, enabling efficient reuse of existing resources.\n\nInput arguments\n\nop: The triangular operator to update;\nA: The new sparse matrix to associate with the operator.\n\n\n\n\n\n","category":"method"},{"location":"triangular_operators/#Nvidia-GPUs","page":"Triangular operators","title":"Nvidia GPUs","text":"","category":"section"},{"location":"triangular_operators/","page":"Triangular operators","title":"Triangular operators","text":"Sparse matrices have a specific storage on Nvidia GPUs (CuSparseMatrixCSC, CuSparseMatrixCSR or CuSparseMatrixCOO):","category":"page"},{"location":"triangular_operators/","page":"Triangular operators","title":"Triangular operators","text":"using CUDA, CUDA.CUSPARSE\nusing SparseArrays\nusing KrylovPreconditioners\n\nif CUDA.functional()\n  # CPU Arrays\n  A_cpu = sprand(100, 100, 0.3)\n\n  # GPU Arrays\n  A_csc_gpu = CuSparseMatrixCSC(A_cpu)\n  A_csr_gpu = CuSparseMatrixCSR(A_cpu)\n  A_coo_gpu = CuSparseMatrixCOO(A_cpu)\n\n  # Triangular operators\n  op_csc = TriangularOperator(A_csc_gpu; uplo='L', diag='U', nrhs=1, transa='N')\n  op_csr = TriangularOperator(A_csr_gpu; uplo='U', diag='N', nrhs=1, transa='T')\n  op_coo = TriangularOperator(A_coo_gpu; uplo='L', diag='N', nrhs=5, transa='N')\nend","category":"page"},{"location":"triangular_operators/#AMD-GPUs","page":"Triangular operators","title":"AMD GPUs","text":"","category":"section"},{"location":"triangular_operators/","page":"Triangular operators","title":"Triangular operators","text":"Sparse matrices have a specific storage on AMD GPUs (ROCSparseMatrixCSC, ROCSparseMatrixCSR or ROCSparseMatrixCOO):","category":"page"},{"location":"triangular_operators/","page":"Triangular operators","title":"Triangular operators","text":"using AMDGPU, AMDGPU.rocSPARSE\nusing SparseArrays\nusing KrylovPreconditioners\n\nif AMDGPU.functional()\n  # CPU Arrays\n  A_cpu = sprand(200, 100, 0.3)\n\n  # GPU Arrays\n  A_csc_gpu = ROCSparseMatrixCSC(A_cpu)\n  A_csr_gpu = ROCSparseMatrixCSR(A_cpu)\n  A_coo_gpu = ROCSparseMatrixCOO(A_cpu)\n\n  # Triangular operators\n  op_csc = TriangularOperator(A_csc_gpu; uplo='L', diag='U', nrhs=1, transa='N')\n  op_csr = TriangularOperator(A_csr_gpu; uplo='L', diag='U', nrhs=1, transa='T')\n  op_coo = TriangularOperator(A_coo_gpu; uplo='L', diag='U', nrhs=5, transa='N')\nend","category":"page"},{"location":"triangular_operators/#Intel-GPUs","page":"Triangular operators","title":"Intel GPUs","text":"","category":"section"},{"location":"triangular_operators/","page":"Triangular operators","title":"Triangular operators","text":"Sparse matrices have a specific storage on Intel GPUs (oneSparseMatrixCSR):","category":"page"},{"location":"triangular_operators/","page":"Triangular operators","title":"Triangular operators","text":"using oneAPI, oneAPI.oneMKL\nusing SparseArrays\nusing KrylovPreconditioners\n\nif oneAPI.functional()\n  # CPU Arrays\n  A_cpu = sprand(T, 20, 10, 0.3)\n\n  # GPU Arrays\n  A_csr_gpu = oneSparseMatrixCSR(A_cpu)\n\n  # Triangular operator\n  op_csr = TriangularOperator(A_csr_gpu; uplo='L', diag='U', nrhs=1, transa='N')\nend","category":"page"},{"location":"krylov_operators/#krylov_operators","page":"Krylov operators","title":"Krylov operators","text":"","category":"section"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"KrylovOperator\nupdate!(::AbstractKrylovOperator, ::Any)","category":"page"},{"location":"krylov_operators/#KrylovPreconditioners.KrylovOperator","page":"Krylov operators","title":"KrylovPreconditioners.KrylovOperator","text":"KrylovOperator(A; nrhs::Int=1, transa::Char='N')\n\nCreate a Krylov operator to accelerate sparse matrix-vector or matrix-matrix products on GPU architectures. The operator is compatible with sparse matrices stored on NVIDIA, AMD, and Intel GPUs.\n\nInput arguments\n\nA: The sparse matrix on the GPU that serves as the operator for matrix-vector or matrix-matrix products;\nnrhs: Specifies the number of columns for the right-hand sides. Defaults to 1 for standard matrix-vector products;\ntransa: Determines how the matrix A is applied during the products; 'N' for no transposition, 'T' for transpose, and 'C' for conjugate transpose.\n\nOutput argument\n\nop: An instance of AbstractKrylovOperator representing the Krylov operator for the specified sparse matrix and parameters.\n\n\n\n\n\n","category":"function"},{"location":"krylov_operators/#KrylovPreconditioners.update!-Tuple{AbstractKrylovOperator, Any}","page":"Krylov operators","title":"KrylovPreconditioners.update!","text":"update!(op::AbstractKrylovOperator, A)\n\nUpdate the sparse matrix A associated with the given AbstractKrylovOperator without the need to reallocate buffers  or repeat the structural analysis phase for detecting parallelism for sparse matrix-vector or matrix-matrix products. A and the operator op must have the same sparsity pattern, enabling efficient reuse of existing resources.\n\nInput arguments\n\nop: The Krylov operator to update;\nA: The new sparse matrix to associate with the operator.\n\n\n\n\n\n","category":"method"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"\n## Nvidia GPUs\n\nSparse matrices have a specific storage on Nvidia GPUs (`CuSparseMatrixCSC`, `CuSparseMatrixCSR` or `CuSparseMatrixCOO`):\n","category":"page"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"julia using CUDA, CUDA.CUSPARSE using SparseArrays using KrylovPreconditioners","category":"page"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"if CUDA.functional()","category":"page"},{"location":"krylov_operators/#CPU-Arrays","page":"Krylov operators","title":"CPU Arrays","text":"","category":"section"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"A_cpu = sprand(200, 100, 0.3)","category":"page"},{"location":"krylov_operators/#GPU-Arrays","page":"Krylov operators","title":"GPU Arrays","text":"","category":"section"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"Acscgpu = CuSparseMatrixCSC(Acpu)   Acsrgpu = CuSparseMatrixCSR(Acpu)   Acoogpu = CuSparseMatrixCOO(A_cpu)","category":"page"},{"location":"krylov_operators/#Krylov-operators","page":"Krylov operators","title":"Krylov operators","text":"","category":"section"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"opcsc = KrylovOperator(Acscgpu; nrhs=1, transa='N')   opcsr = KrylovOperator(Acsrgpu; nrhs=1, transa='T')   opcoo = KrylovOperator(Acoo_gpu; nrhs=5, transa='N') end","category":"page"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"\n## AMD GPUs\n\nSparse matrices have a specific storage on AMD GPUs (`ROCSparseMatrixCSC`, `ROCSparseMatrixCSR` or `ROCSparseMatrixCOO`):\n","category":"page"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"julia using AMDGPU, AMDGPU.rocSPARSE using SparseArrays using KrylovPreconditioners","category":"page"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"if AMDGPU.functional()","category":"page"},{"location":"krylov_operators/#CPU-Arrays-2","page":"Krylov operators","title":"CPU Arrays","text":"","category":"section"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"A_cpu = sprand(200, 100, 0.3)","category":"page"},{"location":"krylov_operators/#GPU-Arrays-2","page":"Krylov operators","title":"GPU Arrays","text":"","category":"section"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"Acscgpu = ROCSparseMatrixCSC(Acpu)   Acsrgpu = ROCSparseMatrixCSR(Acpu)   Acoogpu = ROCSparseMatrixCOO(A_cpu)","category":"page"},{"location":"krylov_operators/#Krylov-operators-2","page":"Krylov operators","title":"Krylov operators","text":"","category":"section"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"opcsc = KrylovOperator(Acscgpu; nrhs=1, transa='N')   opcsr = KrylovOperator(Acsrgpu; nrhs=1, transa='T')   opcoo = KrylovOperator(Acoo_gpu; nrhs=5, transa='N') end","category":"page"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"\n## Intel GPUs\n\nSparse matrices have a specific storage on Intel GPUs (`oneSparseMatrixCSR`):\n","category":"page"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"julia using oneAPI, oneAPI.oneMKL using SparseArrays using KrylovPreconditioners","category":"page"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"if oneAPI.functional()","category":"page"},{"location":"krylov_operators/#CPU-Arrays-3","page":"Krylov operators","title":"CPU Arrays","text":"","category":"section"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"A_cpu = sprand(Float32, 20, 10, 0.3)","category":"page"},{"location":"krylov_operators/#GPU-Arrays-3","page":"Krylov operators","title":"GPU Arrays","text":"","category":"section"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"Acsrgpu = oneSparseMatrixCSR(A_cpu)","category":"page"},{"location":"krylov_operators/#Krylov-operator","page":"Krylov operators","title":"Krylov operator","text":"","category":"section"},{"location":"krylov_operators/","page":"Krylov operators","title":"Krylov operators","text":"opcsr = KrylovOperator(Acsr_gpu; nrhs=1, transa='N') end ```","category":"page"},{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"KrylovPreconditioners.update!(::BlockJacobiPreconditioner, ::SparseMatrixCSC)\nKrylovPreconditioners.BlockJacobiPreconditioner\nKrylovPreconditioners.backward_substitution!\nKrylovPreconditioners.forward_substitution!","category":"page"},{"location":"reference/#KrylovPreconditioners.update!-Tuple{BlockJacobiPreconditioner, SparseMatrixCSC}","page":"Reference","title":"KrylovPreconditioners.update!","text":"function update!(p, J::SparseMatrixCSC)\n\nUpdate the preconditioner p from the sparse Jacobian J in CSC format for the CPU\n\nNote that this implements the same algorithm as for the GPU and becomes very slow on CPU with growing number of blocks.\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.BlockJacobiPreconditioner","page":"Reference","title":"KrylovPreconditioners.BlockJacobiPreconditioner","text":"BlockJacobiPreconditioner\n\nOverlapping-Schwarz preconditioner.\n\nAttributes\n\nnblocks::Int64: Number of partitions or blocks.\nblocksize::Int64: Size of each block.\npartitions::Vector{Vector{Int64}}:npart` partitions stored as lists\ncupartitions: partitions transfered to the GPU\nlpartitions::Vector{Int64}`: Length of each partitions.\nculpartitions::Vector{Int64}`: Length of each partitions, on the GPU.\nblocks: Dense blocks of the block-Jacobi\ncublocks: Js transfered to the GPU\nmap: The partitions as a mapping to construct views\ncumap: cumap transferred to the GPU`\npart: Partitioning as output by Metis\ncupart: part transferred to the GPU\n\n\n\n\n\n","category":"type"},{"location":"reference/#KrylovPreconditioners.backward_substitution!","page":"Reference","title":"KrylovPreconditioners.backward_substitution!","text":"Applies in-place backward substitution with the U factor of F, under the assumptions:\n\nU is stored transposed / row-wise\nU has no lower-triangular elements stored\nU has (nonzero) diagonal elements stored.\n\n\n\n\n\n","category":"function"},{"location":"reference/#KrylovPreconditioners.forward_substitution!","page":"Reference","title":"KrylovPreconditioners.forward_substitution!","text":"Applies in-place forward substitution with the L factor of F, under the assumptions:\n\nL is stored column-wise (unlike U)\nL has no upper triangular elements\nL has no diagonal elements\n\n\n\n\n\n","category":"function"},{"location":"#Home","page":"Home","title":"KrylovPreconditioners.jl documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package provides a collection of preconditioners.","category":"page"},{"location":"#How-to-Cite","page":"Home","title":"How to Cite","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you use KrylovPreconditioners.jl in your work, please cite using the format given in CITATION.cff.","category":"page"},{"location":"#How-to-Install","page":"Home","title":"How to Install","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"KrylovPreconditioners.jl can be installed and tested through the Julia package manager:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> ]\npkg> add KrylovPreconditioners\npkg> test KrylovPreconditioners","category":"page"},{"location":"#Bug-reports-and-discussions","page":"Home","title":"Bug reports and discussions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you think you found a bug, feel free to open an issue. Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.","category":"page"},{"location":"","page":"Home","title":"Home","text":"If you want to ask a question not suited for a bug report, feel free to start a discussion here. This forum is for general discussion about this repository and the JuliaSmoothOptimizers organization, so questions about any of our packages are welcome.","category":"page"}]
}
