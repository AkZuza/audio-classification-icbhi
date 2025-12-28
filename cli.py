"""Command-line interface for audio classification inference."""

import argparse
import torch
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm

from src.models.cnn import LightweightCNN
from src.models. resnet import CompactResNet
from src.data.preprocessing import AudioPreprocessor
from src.utils.config import get_device


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config")

    if config is None:
        raise ValueError("Checkpoint does not contain configuration.  Please provide config file.")

    # Create model
    if config["model"]["architecture"] == "cnn":
        model = LightweightCNN(
            num_classes=config["model"]["num_classes"], dropout=config["model"]["dropout"]
        )
    else: 
        model = CompactResNet(
            num_classes=config["model"]["num_classes"],
            pretrained=False,
            dropout=config["model"]["dropout"],
        )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


def classify_audio(model, preprocessor, audio_path, device, class_names):
    """Classify a single audio file."""
    # Preprocess audio
    mel_spec = preprocessor.preprocess(audio_path)
    mel_spec = mel_spec.unsqueeze(0).to(device)  # Add batch dimension

    # Inference
    with torch.no_grad():
        output = model(mel_spec)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    result = {
        "audio_path": str(audio_path),
        "predicted_class": class_names[predicted_class],
        "confidence":  confidence,
        "probabilities": {
            class_names[i]: probabilities[0, i].item() for i in range(len(class_names))
        },
    }

    return result


def classify_command(args):
    """Handle single file classification."""
    # Get device
    device = get_device(args.device == "cuda")

    # Load model
    print(f"Loading model from {args. model}...")
    model, config = load_model(args.model, device)

    # Create preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=config["data"]["sample_rate"],
        n_mels=config["data"]["n_mels"],
        n_fft=config["data"]["n_fft"],
        hop_length=config["data"]["hop_length"],
        duration=config["data"]["duration"],
        augment=False,
    )

    class_names = config["classes"]

    # Classify
    print(f"\nClassifying {args.audio}...")
    result = classify_audio(model, preprocessor, args. audio, device, class_names)

    # Print results
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"Audio: {result['audio_path']}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:. 4f}")
    print("\nProbabilities:")
    for class_name, prob in result["probabilities"].items():
        print(f"  {class_name}: {prob:.4f}")
    print("=" * 60)


def classify_batch_command(args):
    """Handle batch classification."""
    # Get device
    device = get_device(args.device == "cuda")

    # Load model
    print(f"Loading model from {args. model}...")
    model, config = load_model(args.model, device)

    # Create preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=config["data"]["sample_rate"],
        n_mels=config["data"]["n_mels"],
        n_fft=config["data"]["n_fft"],
        hop_length=config["data"]["hop_length"],
        duration=config["data"]["duration"],
        augment=False,
    )

    class_names = config["classes"]

    # Get audio files
    input_dir = Path(args.input_dir)
    audio_files = list(input_dir.glob("*.wav"))

    if not audio_files:
        print(f"No . wav files found in {input_dir}")
        return

    print(f"\nFound {len(audio_files)} audio files")

    # Classify all files
    results = []
    for audio_path in tqdm(audio_files, desc="Classifying"):
        try:
            result = classify_audio(model, preprocessor, audio_path, device, class_names)
            results.append(result)
        except Exception as e: 
            print(f"Error processing {audio_path}: {e}")

    # Save results
    output_path = args.output
    if output_path. endswith(".json"):
        with open(output_path, "w") as f:
            json. dump(results, f, indent=2)
    else:  # CSV
        df_data = []
        for result in results: 
            row = {
                "audio_path": result["audio_path"],
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
            }
            # Add probabilities
            row.update(result["probabilities"])
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)

    print(f"\n✓ Results saved to {output_path}")
    print(f"✓ Processed {len(results)}/{len(audio_files)} files successfully")


def info_command(args):
    """Display model information."""
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location="cpu")
    config = checkpoint. get("config")

    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    print(f"Checkpoint: {args.model}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation Loss: {checkpoint.get('val_loss', 'unknown')}")

    if config: 
        print(f"\nModel Architecture: {config['model']['architecture']}")
        print(f"Number of Classes: {config['model']['num_classes']}")
        print(f"Classes: {', '.join(config['classes'])}")
        print(f"\nAudio Configuration:")
        print(f"  Sample Rate: {config['data']['sample_rate']} Hz")
        print(f"  Mel Bins: {config['data']['n_mels']}")
        print(f"  Duration: {config['data']['duration']} seconds")
    else:
        print("\nNo configuration found in checkpoint")

    print("=" * 60)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Audio Classification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Classify single file
    classify_parser = subparsers. add_parser("classify", help="Classify a single audio file")
    classify_parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    classify_parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    classify_parser. add_argument(
        "--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Device to use"
    )

    # Classify batch
    batch_parser = subparsers. add_parser("classify-batch", help="Classify multiple audio files")
    batch_parser.add_argument(
        "--input-dir", type=str, required=True, help="Directory containing audio files"
    )
    batch_parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    batch_parser.add_argument(
        "--output", type=str, default="results.csv", help="Output file (CSV or JSON)"
    )
    batch_parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Device to use"
    )

    # Model info
    info_parser = subparsers.add_parser("info", help="Display model information")
    info_parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")

    args = parser.parse_args()

    if args.command == "classify": 
        classify_command(args)
    elif args.command == "classify-batch":
        classify_batch_command(args)
    elif args.command == "info": 
        info_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
