import onnx

batch_size = 1

def change_input_dim(model):

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim 
    inputs = model.graph.input
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_value = batch_size


def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)

apply(change_input_dim, "flowers-152-opset7.onnx", "flowers-152-opset7-b{}.onnx".format(batch_size))
