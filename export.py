import torch
import torch.onnx
import tensorflow as tf
from unet_model import UNet  # Make sure this import works with your file structure

def export_to_onnx(model, input_shape, output_path):
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(model, dummy_input, output_path, verbose=True, input_names=['input'], output_names=['output'])
    print(f"Model exported to ONNX format: {output_path}")

def export_to_tflite(onnx_path, output_path):
    import onnx
    from onnx_tf.backend import prepare

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(output_path)
    
    converter = tf.lite.TFLiteConverter.from_saved_model(output_path)
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model exported to TFLite format: {output_path}")

def export_to_h5(model, output_path):
    dummy_input = torch.randn(1, 3, 256, 256)
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, 'temp_model.pt')
    
    import torchvision
    model = torchvision.models.resnet18()  # This is a placeholder, replace with your actual model
    torch.save(model.state_dict(), output_path)
    print(f"Model exported to H5 format: {output_path}")

if __name__ == "__main__":
    # Load your trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=3).to(device)
    checkpoint = torch.load('model_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Export to ONNX
    export_to_onnx(model, (1, 3, 256, 256), 'unet_model.onnx')

    # Export to TFLite
    export_to_tflite('unet_model.onnx', 'unet_model.tflite')

    # Export to H5
    export_to_h5(model, 'unet_model.h5')