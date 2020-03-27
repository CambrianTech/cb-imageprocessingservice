# Cambrian core
## Usage
- Install with `pip install .` or `pip install -e .` for symlinking so when making changes to this repository it will automatically take over the changes without having to install again.
- `import cambrian`

## Modules
- cambrian.diagnostics
- cambrian.geometry
- cambrian.image_processing
- cambrian.transformations
- cambrian.utils
- cambrian.nn

## Command line commands (use --help to get more info)
- cb-splitdataset (Split a dataset into train and test set)
- cb-savedmodelinfo (Display input and output tensors for a TensorFlow SavedModel)
- cb-exportmodel (Export a TensorFlow SavedModel to tflite or coreml)