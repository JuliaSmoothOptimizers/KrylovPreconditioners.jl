var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [KrylovPreconditioners]\nOrder   = [:function, :type]","category":"page"},{"location":"reference/#Base.empty!-Tuple{KrylovPreconditioners.InsertableSparseVector}","page":"Reference","title":"Base.empty!","text":"Empties the InsterableSparseVector in O(1) operations.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.empty!-Tuple{KrylovPreconditioners.SortedSet}","page":"Reference","title":"Base.empty!","text":"Make the head pointer do a self-loop.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.empty!-Tuple{KrylovPreconditioners.SparseVectorAccumulator}","page":"Reference","title":"Base.empty!","text":"Empty the SparseVectorAccumulator in O(1) operations.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.push!-Tuple{KrylovPreconditioners.LinkedLists, Integer, Integer}","page":"Reference","title":"Base.push!","text":"For the L-factor: insert in row head column value For the U-factor: insert in column head row value\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.push!-Tuple{KrylovPreconditioners.SortedSet, Int64, Int64}","page":"Reference","title":"Base.push!","text":"Insert index after a known value after\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners._fillblock_gpu!-Tuple{Any}","page":"Reference","title":"KrylovPreconditioners._fillblock_gpu!","text":"_fillblock_gpu\n\nFill the dense blocks of the preconditioner from the sparse CSR matrix arrays\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.add!-Tuple{KrylovPreconditioners.InsertableSparseVector, Any, Integer, Integer}","page":"Reference","title":"KrylovPreconditioners.add!","text":"Sets v[idx] += a when idx is occupied, or sets v[idx] = a. Complexity is O(nnz). The prev_idx can be used to start the linear search at prev_idx, useful when multiple already sorted values are added.\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.add!-Tuple{KrylovPreconditioners.InsertableSparseVector, Any, Integer}","page":"Reference","title":"KrylovPreconditioners.add!","text":"Add without providing a previous index.\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.add!-Tuple{KrylovPreconditioners.SparseVectorAccumulator, Any, Any}","page":"Reference","title":"KrylovPreconditioners.add!","text":"Sets v[idx] += a when idx is occupied, or sets v[idx] = a. Complexity is O(1).\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.append_col!","page":"Reference","title":"KrylovPreconditioners.append_col!","text":"Basically A[:, j] = scale * drop(y), where drop removes values less than drop. Note: sorts the nzind's of y,  so that the column can be appended to a SparseMatrixCSC.\n\nResets the SparseVectorAccumulator.\n\nNote: does not update A.colptr for columns > j + 1, as that is done during the steps.\n\n\n\n\n\n","category":"function"},{"location":"reference/#KrylovPreconditioners.append_col!-Union{Tuple{Tv}, Tuple{SparseArrays.SparseMatrixCSC{Tv}, KrylovPreconditioners.InsertableSparseVector{Tv}, Int64, Tv}, Tuple{SparseArrays.SparseMatrixCSC{Tv}, KrylovPreconditioners.InsertableSparseVector{Tv}, Int64, Tv, Tv}} where Tv","page":"Reference","title":"KrylovPreconditioners.append_col!","text":"Basically A[:, j] = scale * drop(y), where drop removes values less than drop.\n\nResets the InsertableSparseVector.\n\nNote: does not update A.colptr for columns > j + 1, as that is done during the steps.\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.backward_substitution!-Tuple{KrylovPreconditioners.ILUFactorization, AbstractVector}","page":"Reference","title":"KrylovPreconditioners.backward_substitution!","text":"Applies in-place backward substitution with the U factor of F, under the assumptions:\n\nU is stored transposed / row-wise\nU has no lower-triangular elements stored\nU has (nonzero) diagonal elements stored.\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.build_adjmatrix-Tuple{Any}","page":"Reference","title":"KrylovPreconditioners.build_adjmatrix","text":"build_adjmatrix\n\nBuild the adjacency matrix of a matrix A corresponding to the undirected graph\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.forward_substitution!-Tuple{KrylovPreconditioners.ILUFactorization, AbstractVector}","page":"Reference","title":"KrylovPreconditioners.forward_substitution!","text":"Applies in-place forward substitution with the L factor of F, under the assumptions:\n\nL is stored column-wise (unlike U)\nL has no upper triangular elements\nL has no diagonal elements\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.isoccupied-Tuple{KrylovPreconditioners.SparseVectorAccumulator, Integer}","page":"Reference","title":"KrylovPreconditioners.isoccupied","text":"Check whether idx is nonzero.\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.overlap-Tuple{Any, Any}","page":"Reference","title":"KrylovPreconditioners.overlap","text":"overlap(Graph, subset, level)\n\nGiven subset embedded within Graph, compute subset2 such that subset2 contains subset and all of its adjacent vertices.\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.update!-Tuple{BlockJacobiPreconditioner, SparseArrays.SparseMatrixCSC}","page":"Reference","title":"KrylovPreconditioners.update!","text":"function update!(p, J::SparseMatrixCSC)\n\nUpdate the preconditioner p from the sparse Jacobian J in CSC format for the CPU\n\nNote that this implements the same algorithm as for the GPU and becomes very slow on CPU with growing number of blocks.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearAlgebra.axpy!-Tuple{Any, SparseArrays.SparseMatrixCSC, Any, Any, KrylovPreconditioners.SparseVectorAccumulator}","page":"Reference","title":"LinearAlgebra.axpy!","text":"Add a part of a SparseMatrixCSC column to a SparseVectorAccumulator, starting at a given index until the end.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SparseArrays.nnz-Tuple{KrylovPreconditioners.ILUFactorization}","page":"Reference","title":"SparseArrays.nnz","text":"Returns the number of nonzeros of the L and U factor combined.\n\nExcludes the unit diagonal of the L factor, which is not stored.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.Vector-Tuple{KrylovPreconditioners.SortedSet}","page":"Reference","title":"Base.Vector","text":"For debugging and testing\n\n\n\n\n\n","category":"method"},{"location":"reference/#KrylovPreconditioners.BlockJacobiPreconditioner","page":"Reference","title":"KrylovPreconditioners.BlockJacobiPreconditioner","text":"BlockJacobiPreconditioner\n\nOverlapping-Schwarz preconditioner.\n\nAttributes\n\nnblocks::Int64: Number of partitions or blocks.\nblocksize::Int64: Size of each block.\npartitions::Vector{Vector{Int64}}:npart` partitions stored as lists\ncupartitions: partitions transfered to the GPU\nlpartitions::Vector{Int64}`: Length of each partitions.\nculpartitions::Vector{Int64}`: Length of each partitions, on the GPU.\nblocks: Dense blocks of the block-Jacobi\ncublocks: Js transfered to the GPU\nmap: The partitions as a mapping to construct views\ncumap: cumap transferred to the GPU`\npart: Partitioning as output by Metis\ncupart: part transferred to the GPU\n\n\n\n\n\n","category":"type"},{"location":"reference/#KrylovPreconditioners.InsertableSparseVector","page":"Reference","title":"KrylovPreconditioners.InsertableSparseVector","text":"InsertableSparseVector accumulates the sparse vector result from SpMV. Initialization requires O(N) work, therefore the data structure is reused. Insertion requires O(nnz) at worst, as insertion sort is used.\n\n\n\n\n\n","category":"type"},{"location":"reference/#KrylovPreconditioners.LinkedLists","page":"Reference","title":"KrylovPreconditioners.LinkedLists","text":"The factor L is stored column-wise, but we need all nonzeros in row row. We already keep track of  the first nonzero in each column (at most n indices). Take l = LinkedLists(n). Let l.head[row] be the column of some nonzero in row row. Then we can store the column of the next nonzero of row row in l.next[l.head[row]], etc. That \"spot\" is empty and there will never be a conflict because as long as we only store the first nonzero per column:  the column is then a unique identifier.\n\n\n\n\n\n","category":"type"},{"location":"reference/#KrylovPreconditioners.SortedSet","page":"Reference","title":"KrylovPreconditioners.SortedSet","text":"SortedSet keeps track of a sorted set of integers ≤ N using insertion sort with a linked list structure in a pre-allocated  vector. Requires O(N + 1) memory. Insertion goes via a linear scan in O(n) where n is the number of stored elements, but can be accelerated  by passing along a known value in the set (which is useful when pushing in an already sorted list). The insertion itself requires O(1) operations due to the linked list structure. Provides iterators:\n\nints = SortedSet(10)\npush!(ints, 5)\npush!(ints, 3)\nfor value in ints\n    println(value)\nend\n\n\n\n\n\n","category":"type"},{"location":"#Home","page":"Home","title":"KrylovPreconditioners.jl documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package provides a collection of preconditioners.","category":"page"},{"location":"#How-to-Cite","page":"Home","title":"How to Cite","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you use KrylovPreconditioners.jl in your work, please cite using the format given in CITATION.cff.","category":"page"},{"location":"#How-to-Install","page":"Home","title":"How to Install","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"KrylovPreconditioners.jl can be installed and tested through the Julia package manager:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> ]\npkg> add KrylovPreconditioners\npkg> test KrylovPreconditioners","category":"page"},{"location":"#Bug-reports-and-discussions","page":"Home","title":"Bug reports and discussions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you think you found a bug, feel free to open an issue. Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.","category":"page"},{"location":"","page":"Home","title":"Home","text":"If you want to ask a question not suited for a bug report, feel free to start a discussion here. This forum is for general discussion about this repository and the JuliaSmoothOptimizers organization, so questions about any of our packages are welcome.","category":"page"}]
}