import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import random
import torch

from dataset import prepare_data
from models import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "mnist", "fmnist"],
        help="须与训练时所用数据集一致",
    )
    parser.add_argument(
        "--model-path",
        default="weights/adaptive/cifar10_dirichlet.pth",
        help="相对项目根的权重路径，须与训练保存路径一致",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_dataset, _, _ = prepare_data(dataset=args.dataset)

    model = create_model(args.dataset).to(device)
    ckpt = _PROJECT_ROOT / args.model_path
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.eval()

    num_samples = 6
    indices = random.sample(range(len(test_dataset)), num_samples)

    plt.figure(figsize=(12, 8))
    plt.suptitle("Federated Learning Model Inference Demo", fontsize=16)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label = test_dataset[idx]
            input_tensor = image.unsqueeze(0).to(device)
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

            plt.subplot(2, 3, i + 1)
            img_display = image.cpu().numpy()

            if img_display.shape[0] == 1:
                plt.imshow(img_display[0], cmap="gray", vmin=0, vmax=1)
            else:
                img_display = image.permute(1, 2, 0).numpy()
                img_display = (img_display * 0.2) + 0.5
                plt.imshow(img_display.clip(0, 1))

            color = "green" if predicted.item() == label else "red"
            plt.title(f"True: {label} | Pred: {predicted.item()}", color=color)
            plt.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("demo_inference.png", dpi=300)
    plt.show()
    print("Success: Demo plot generated! You can use 'demo_inference.png' in your PPT.")


if __name__ == "__main__":
    main()
