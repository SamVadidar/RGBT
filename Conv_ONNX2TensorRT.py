from onnx import ModelProto
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

def build_engine(onnx_path, shape):

    """
    This is the function to create the TensorRT engine
    Args:
        onnx_path : Path to onnx_file. 
        shape : Shape of the input of the ONNX file. 
    """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.max_workspace_size = (256 << 20)
        # with open(onnx_path, 'rb') as model:
        #     parser.parse(model.read())
        success = parser.parse_from_file(onnx_path)
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))

        if not success:
            print('FAILED TO OPEN THE ONNX FILE')
            # pass # Error handling code here

        network.get_input(0).shape = shape[0] # RGB input
        network.get_input(1).shape = shape[1] # IR input
        # engine = builder.build_engine(network, config)
        engine = builder.build_serialized_network(network, config)
        return engine

def save_engine(engine, file_name):
#    buf = engine.serialize()
   with open(file_name, 'wb') as f:
    #    f.write(buf)
       f.write(engine)

def load_engine(trt_runtime, plan_path):
   with open(plan_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine


if __name__ == "__main__":

    engine_name = "./RGBT_new_4Channel_2.plan"
    onnx_path = "./runs/train/exp_RGBT640_500_HACBC_CS2/weights/best_val_loss_Ver2.onnx"
    batch_size = 1

    model = ModelProto()
    with open(onnx_path, "rb") as f:
        model.ParseFromString(f.read())

    d0_rgb = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value # rgb input channel
    d0_ir = model.graph.input[1].type.tensor_type.shape.dim[1].dim_value # ir input channel
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    shape_rgb = [batch_size , d0_rgb, d1 ,d2]
    shape_ir = [batch_size , d0_ir, d1 ,d2]

    engine = build_engine(onnx_path, shape = [shape_rgb, shape_ir])
    save_engine(engine, engine_name)