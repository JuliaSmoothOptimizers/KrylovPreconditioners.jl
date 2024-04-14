var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [KrylovPreconditioners]\nOrder   = [:function, :type]","category":"page"},{"location":"reference/#KrylovPreconditioners._fillblock_gpu!-Tuple{Any}","page":"Reference","title":"KrylovPreconditioners._fillblock_gpu!","text":"_fillblock_gpu\n\nFill the dense blocks of the preconditioner from the sparse CSR matrix arrays\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.build_adjmatrix-Tuple{Any}","page":"Reference","title":"KrylovPreconditioners.build_adjmatrix","text":"build_adjmatrix\n\nBuild the adjacency matrix of a matrix A corresponding to the undirected graph\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.overlap-Tuple{Any, Any}","page":"Reference","title":"KrylovPreconditioners.overlap","text":"overlap(Graph, subset, level)\n\nGiven subset embedded within Graph, compute subset2 such that subset2 contains subset and all of its adjacent vertices.\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.update!-Tuple{BlockJacobiPreconditioner, SparseArrays.SparseMatrixCSC}","page":"Reference","title":"KrylovPreconditioners.update!","text":"function update!(p, J::SparseMatrixCSC)\n\nUpdate the preconditioner p from the sparse Jacobian J in CSC format for the CPU\n\nNote that this implements the same algorithm as for the GPU and becomes very slow on CPU with growing number of blocks.\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.BlockJacobiPreconditioner","page":"Reference","title":"KrylovPreconditioners.BlockJacobiPreconditioner","text":"BlockJacobiPreconditioner\n\nOverlapping-Schwarz preconditioner.\n\nAttributes\n\nnblocks::Int64: Number of partitions or blocks.\nblocksize::Int64: Size of each block.\npartitions::Vector{Vector{Int64}}:npart` partitions stored as lists\ncupartitions: partitions transfered to the GPU\nlpartitions::Vector{Int64}`: Length of each partitions.\nculpartitions::Vector{Int64}`: Length of each partitions, on the GPU.\nblocks: Dense blocks of the block-Jacobi\ncublocks: Js transfered to the GPU\nmap: The partitions as a mapping to construct views\ncumap: cumap transferred to the GPU`\npart: Partitioning as output by Metis\ncupart: part transferred to the GPU\n\n\n\n\n\n","category":"type"},{"location":"#Home","page":"Home","title":"KrylovPreconditioners.jl documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package provides a collection of preconditioners.","category":"page"},{"location":"#How-to-Cite","page":"Home","title":"How to Cite","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you use KrylovPreconditioners.jl in your work, please cite using the format given in CITATION.cff.","category":"page"},{"location":"#How-to-Install","page":"Home","title":"How to Install","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"KrylovPreconditioners.jl can be installed and tested through the Julia package manager:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> ]\npkg> add KrylovPreconditioners\npkg> test KrylovPreconditioners","category":"page"},{"location":"#Bug-reports-and-discussions","page":"Home","title":"Bug reports and discussions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you think you found a bug, feel free to open an issue. Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.","category":"page"},{"location":"","page":"Home","title":"Home","text":"If you want to ask a question not suited for a bug report, feel free to start a discussion here. This forum is for general discussion about this repository and the JuliaSmoothOptimizers organization, so questions about any of our packages are welcome.","category":"page"}]
}
