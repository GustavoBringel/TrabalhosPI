import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# =========================================================
# CONFIGURAÇÃO: APENAS MUDAR ESTE NOME DE ARQUIVO
# =========================================================
NOME_IMAGEM_ENTRADA = "image4.png"
# =========================================================

# -------------------------
# Interpolação - Vizinho Mais Próximo
# -------------------------
def nearest_neighbor_resize(img, new_width, new_height):
    height, width = img.shape[:2]
    if new_width <= 0 or new_height <= 0:
        return np.zeros_like(img, dtype=img.dtype)
    resized = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)
    x_ratio = width / new_width
    y_ratio = height / new_height
    for y in range(new_height):
        for x in range(new_width):
            orig_x = int(round(x * x_ratio))
            orig_y = int(round(y * y_ratio))
            orig_x = min(orig_x, width - 1)
            orig_y = min(orig_y, height - 1)
            resized[y, x] = img[orig_y, orig_x]
    return resized

# -------------------------
# Interpolação - Bilinear
# -------------------------
def bilinear_resize(img, new_width, new_height):
    height, width = img.shape[:2]
    if new_width <= 0 or new_height <= 0:
        return np.zeros_like(img, dtype=img.dtype)
    resized = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)
    x_ratio = (width - 1) / (new_width - 1) if new_width > 1 else 0
    y_ratio = (height - 1) / (new_height - 1) if new_height > 1 else 0
    for y in range(new_height):
        for x in range(new_width):
            x_orig_float = x * x_ratio
            y_orig_float = y * y_ratio
            x_l = int(np.floor(x_orig_float))
            y_l = int(np.floor(y_orig_float))
            x_h = min(x_l + 1, width - 1)
            y_h = min(y_l + 1, height - 1)
            x_weight = x_orig_float - x_l
            y_weight = y_orig_float - y_l
            a = img[y_l, x_l]
            b = img[y_l, x_h]
            c = img[y_h, x_l]
            d = img[y_h, x_h]
            pixel = (a * (1 - x_weight) * (1 - y_weight) +
                     b * x_weight * (1 - y_weight) +
                     c * (1 - x_weight) * y_weight +
                     d * x_weight * y_weight)
            resized[y, x] = pixel.astype(img.dtype)
    return resized

# -------------------------
# Função para carregar ou gerar a imagem
# -------------------------
def carregar_ou_gerar_imagem(nome_arquivo):
    img = cv2.imread(nome_arquivo)
    if img is None:
        print(f"Arquivo '{nome_arquivo}' não encontrado. Gerando uma imagem de teste...")
        largura_amostra, altura_amostra = 200, 200
        img_teste = np.zeros((altura_amostra, largura_amostra, 3), dtype=np.uint8)
        
        # Desenha um gradiente diagonal
        for y in range(altura_amostra):
            for x in range(largura_amostra):
                img_teste[y, x, 0] = int((x + y) / (largura_amostra + altura_amostra) * 255)
                img_teste[y, x, 1] = int(x / largura_amostra * 255)
                img_teste[y, x, 2] = int(y / altura_amostra * 255)

        # Adiciona formas para testar bordas nítidas
        cv2.line(img_teste, (10, 10), (190, 190), (255, 255, 255), 2)
        cv2.circle(img_teste, (100, 100), 40, (255, 0, 255), 3)
        cv2.rectangle(img_teste, (20, 160), (60, 180), (0, 255, 255), -1)

        cv2.imwrite(nome_arquivo, img_teste)
        img = cv2.imread(nome_arquivo)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------------------------
# Testando com a imagem especificada
# -------------------------
if __name__ == "__main__":
    img = carregar_ou_gerar_imagem(NOME_IMAGEM_ENTRADA)

    if img is None:
        exit()

    original_height, original_width = img.shape[:2]

    # Define os tamanhos para a redução de amostragem
    tamanhos_amostragem = [original_width // 2, original_width // 4, original_width // 8, original_width // 16, original_width // 32]
    
    # Listas para armazenar os resultados e tempos de processamento
    resultados_nn = []
    tempos_nn = []
    resultados_bl = []
    tempos_bl = []
    
    for tamanho in tamanhos_amostragem:
        # Processamento com Vizinho Mais Próximo
        start_time_nn = time.time()
        img_reduzida_nn = nearest_neighbor_resize(img, tamanho, tamanho)
        img_ampliada_nn = nearest_neighbor_resize(img_reduzida_nn, original_width, original_height)
        end_time_nn = time.time()
        resultados_nn.append(img_ampliada_nn)
        tempos_nn.append(end_time_nn - start_time_nn)
        
        # Processamento com Bilinear
        start_time_bl = time.time()
        img_reduzida_bl = bilinear_resize(img, tamanho, tamanho)
        img_ampliada_bl = bilinear_resize(img_reduzida_bl, original_width, original_height)
        end_time_bl = time.time()
        resultados_bl.append(img_ampliada_bl)
        tempos_bl.append(end_time_bl - start_time_bl)
        
    # Exibe os resultados em uma grade
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('Comparação de Interpolação', fontsize=18)
    
    # Plota a imagem original na primeira linha (ocupa 5 posições)
    for i in range(5):
        if i == 2:  # Plota a imagem original no meio da primeira linha
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Original ({original_width}x{original_height})", fontsize=14)
        axes[0, i].axis('off')

    # Plota os resultados de Vizinho Mais Próximo na segunda linha
    for i in range(5):
        axes[1, i].imshow(resultados_nn[i])
        axes[1, i].set_title(f"Vizinho\n{tamanhos_amostragem[i]}px\nTempo: {tempos_nn[i]:.4f}s", fontsize=10)
        axes[1, i].axis('off')

    # Plota os resultados de Bilinear na terceira linha
    for i in range(5):
        axes[2, i].imshow(resultados_bl[i])
        axes[2, i].set_title(f"Bilinear\n{tamanhos_amostragem[i]}px\nTempo: {tempos_bl[i]:.4f}s", fontsize=10)
        axes[2, i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()