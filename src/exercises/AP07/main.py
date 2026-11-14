import itertools
from typing import Any
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

# Gera todas as combinações de parâmetros como uma lista de dicionários.
def grid_search(params: dict[str, Any]) -> list[dict[str, Any]]:
    keys = params.keys()
    return [
        dict(zip(keys, values))
        for values in itertools.product(*params.values())
    ]

def plot_results(img1: np.ndarray, img2: np.ndarray, mean_grid: list[dict[str, Any]], gaussian_grid: list[dict[str, Any]],
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # --- 1ª Linha: Exibição das imagens ---
    axs[0, 0].imshow(img1, cmap="gray", vmin=0, vmax=255)
    axs[0, 0].set_title("Imagem 01 (img01.jpg)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(img2, cmap="gray", vmin=0, vmax=255)
    axs[0, 1].set_title("Imagem 02 (img02_m_30.0_dp_40.0.jpeg)")
    axs[0, 1].axis("off")

    # --- 2ª Linha, Coluna 1: Gráfico dos parâmetros para Filtro Média
    mean_experiments = range(len(mean_grid))
    mean_radii = [item["radius"] for item in mean_grid]

    axs[1, 0].plot(mean_experiments, mean_radii, marker="o", color="tab:blue", label="Radius")
    axs[1, 0].set_title("Filtro Média - Grid Search")
    axs[1, 0].set_xlabel("Experimento")
    axs[1, 0].set_ylabel("Valor")
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)

    # --- 2ª Linha, Coluna 2: Gráfico dos parâmetros para Filtro Gaussiano
    gaussian_experiments = range(len(gaussian_grid))
    gaussian_radii = [item["radius"] for item in gaussian_grid]
    gaussian_sigmas = [item["sigma"] for item in gaussian_grid]
    axs[1, 1].plot(gaussian_experiments, gaussian_radii, color="tab:green", label="Radius")
    axs[1, 1].plot(gaussian_experiments, gaussian_sigmas, color="tab:orange", label="Sigma")
    axs[1, 1].set_title("Filtro Gaussiano - Grid Search")
    axs[1, 1].set_xlabel("Experimento")
    axs[1, 1].set_ylabel("Valor")
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

def main() -> None:
    # Carrega as imagens locais
    img01 = cv2.imread(os.path.join(str(Path(__file__).parent).replace("src", "images"), "img01.jpeg"), cv2.IMREAD_GRAYSCALE)
    img02 = cv2.imread(os.path.join(str(Path(__file__).parent).replace("src", "output"), "img02_m_30.0_dp_40.0.jpeg"), cv2.IMREAD_GRAYSCALE)

    # Define os parâmetros e executa o Grid Search para o Filtro Média
    mean_params = {
        "radius": range(1, 6)
    }
    mean_grid = grid_search(mean_params)
    for i, experiment in enumerate(mean_grid):
        # Chama função para executar o Filtro Média com os parâmetros do experimento
        print(f"Média [{i}]:", experiment)

    # Define os parâmetros e executa o Grid Search para o Filtro Gaussiano
    gaussian_params = {
        "radius": range(1, 6),
        "sigma": [round(i * 0.1, 1) for i in range(1, 81)],
    }
    gaussian_grid = grid_search(gaussian_params)
    for i, experiment in enumerate(gaussian_grid):
        # Chama função para executar o Filtro Gaussiano com os parâmetros do experimento
        print(f"Gaussiano [{i}]:", experiment)

    # Plota as imagens e os gráficos dos experimentos
    plot_results(img01, img02, mean_grid, gaussian_grid)

if __name__ == "__main__":
    main()