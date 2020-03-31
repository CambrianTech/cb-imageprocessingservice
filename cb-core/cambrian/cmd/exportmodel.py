import os
import tensorflow as tf
import argparse
from time import time

def freeze_saved_model(input_path, output_path, output_names):
    from tensorflow.python.tools import freeze_graph
    from tensorflow.python.saved_model import tag_constants

    output_node_names = output_names
    input_binary = False
    input_saver_def_path = False
    restore_op_name = None
    filename_tensor_name = None
    clear_devices = False
    input_meta_graph = False
    checkpoint_path = None
    input_graph_filename = None
    saved_model_tags = tag_constants.SERVING

    freeze_graph.freeze_graph(input_graph_filename, input_saver_def_path,
                                input_binary, checkpoint_path, output_node_names,
                                restore_op_name, filename_tensor_name,
                                output_path, clear_devices, "", "", "",
                                input_meta_graph, input_path,
                                saved_model_tags)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to the saved model")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Output file name")
    parser.add_argument("-f", "--format", type=str, default="coreml", help="Output format (coreml|tflite)")
    parser.add_argument("-on", "--output_names", type=str, default="smooth/conv2d_36/BiasAdd", help="Comma-seperated output tensor names. Required for coreml format. Use cb-savedmodelinfo command to find out name.")
    args = parser.parse_args()

    print(args)

    assert args.format == "coreml" or args.format == "tflite"
    
    if not os.path.exists(args.input_path):
        raise ValueError("Error: input_path", args.input_path, "not found")

    print("Converting saved model at", args.input_path, "to", args.format)

    if args.format == "coreml":
        import tfcoreml

        output_names = args.output_names

        frozen_path = "frozen_%d.pb" % int(time())
        print("Creating frozen graph")
        freeze_saved_model(args.input_path, frozen_path, output_names)

        out_path = args.output_path
        if not out_path.endswith(".mlmodel"):
            out_path += ".mlmodel"

        tfcoreml.convert(frozen_path, out_path, [name + ":0" for name in output_names.split(",")])

        print("Removing frozen graph file")
        os.remove(frozen_path)
    elif args.format == "tflite":
        from tensorflow.contrib import lite
        
        print("Creating tflite converter")
        converter = lite.TFLiteConverter.from_saved_model(args.input_path)

        print("Converting")
        tflite_model = converter.convert()

        out_path = args.output_path
        if not out_path.endswith(".tflite"):
            out_path += ".tflite"

        with open(out_path, "wb") as out_file:
            out_file.write(tflite_model)
    else:
        raise ValueError("Invalid output format %s"  % args.format)
    
    print("Done.")

if __name__ == "__main__":
    main()
