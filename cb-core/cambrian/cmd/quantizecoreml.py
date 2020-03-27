from coremltools.models.utils import save_spec
from coremltools.models.neural_network.quantization_utils import quantize_weights
from coremltools.models import MLModel
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="Path to the coreml model")
    parser.add_argument("output_path", type=str, help="Path to save the quantized coreml model at")
    args = parser.parse_args()

    model = MLModel(args.input_path)
    quantized = quantize_weights(model, nbits=8)
    save_spec(quantized, args.output_path)