// code from: https://iree.dev/community/blog/2024-01-29-iree-mlir-linalg-tutorial/
// The 1D identity map, used below.
#map_1d_identity = affine_map<(m) -> (m)>

// Define a function @foo taking two tensor arguments `%lhs` and `%rhs` and returning a tensor.
func.func @foo(
      %lhs : tensor<10xf32>,
      %rhs : tensor<10xf32>
    ) -> tensor<10xf32> {
  // A constant used below.
  %c0f32 = arith.constant 0.0 : f32
  // Create a result "init value". Think of it as an abstract "allocation",
  // creating a tensor but not giving its elements any particular value. It would be
  // undefined behavior to read any element from this tensor.
  %result_empty =  tensor.empty() : tensor<10xf32>

  // Perform the computation. The following is all a single linalg.generic op.

  %result = linalg.generic {
    // This {...} section is the "attributes" - some compile-time settings for this op.
    indexing_maps=[
      // Indexing maps for the parameters listed in `ins(...)`
      #map_1d_identity,
      #map_1d_identity,
      // Indexing maps for the parameters listed in `outs(...)`
      #map_1d_identity
    ],
    // There is one tensor dimension, and it's a parallel-iteration dimension,
    // meaning that it occurs also as a result tensor dimension. The alternative
    // would be "reduction", for dimensions that do not occur in the result tensor.
    iterator_types=["parallel"]
  } // End of the attributes for this linalg.generic. Next come the parameters:
    // `ins` is where we pass regular input-parameters
    ins(%lhs, %rhs : tensor<10xf32>, tensor<10xf32>)
    // `outs` is where we pass the "outputs", but that term has a subtle meaning
    // in linalg. Here we are passing a tensor.empty, meaning just a placeholder
    // for the output with no preexisting element values. In other examples with
    // an accumulator, this is where the accumulator would be passed.
    outs(%result_empty : tensor<10xf32>)
    // End of parameters. The next {...} part is the "code block".
  {
    // bb0 is a code block taking one scalar from each input tensor as argument, and
    // computing and "yielding" (ie returning) the corresponding output tensor element.
    ^bb0(%lhs_entry : f32, %rhs_entry : f32, %unused_result_entry : f32):
      %add = arith.addf %lhs_entry, %rhs_entry : f32
      linalg.yield %add : f32
  } // End of the basic block. Finally, we describe the return type.
  -> tensor<10xf32>

  // End of the linalg.generic op.

  // Return the function's return value.
  return %result : tensor<10xf32>
}