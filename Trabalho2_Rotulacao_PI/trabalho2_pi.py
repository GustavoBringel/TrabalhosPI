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
# Implementação do Algoritmo de Rotulação
# -------------------------
def rotular_componentes_conectados(imagem_binaria, conectividade=8):
    """
    Identifica e rotula componentes conectados em uma imagem binária.
    
    Args:
        imagem_binaria (np.ndarray): Uma imagem binária (0s e 255s).
        conectividade (int): Tipo de conectividade a ser usado (4 ou 8).

    Returns:
        tuple: (num_labels, labels, stats, centroids)
    """
    # Garante que a imagem é do tipo correto para a função do OpenCV
    if imagem_binaria.dtype != np.uint8:
        imagem_binaria = imagem_binaria.astype(np.uint8)

    # Aplica o algoritmo otimizado de rotulação do OpenCV
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        imagem_binaria, connectivity=conectividade, ltype=cv2.CV_32S
    )

    return num_labels, labels, stats, centroids

# -------------------------
# Função para carregar ou gerar a imagem e binarizá-la
# -------------------------
def carregar_ou_gerar_imagem_binaria(nome_arquivo):
    img_colorida = cv2.imread(nome_arquivo)
    
    if img_colorida is None:
        print(f"Arquivo '{nome_arquivo}' não encontrado. Gerando uma imagem de teste...")
        largura, altura = 400, 300
        img_teste = np.zeros((altura, largura), dtype=np.uint8)
        
        # Adiciona formas para criar componentes conectados
        cv2.circle(img_teste, (100, 100), 50, 255, -1)
        cv2.rectangle(img_teste, (250, 50), (350, 150), 255, -1)
        cv2.line(img_teste, (50, 200), (150, 250), 255, 5)
        
        # Formas próximas para testar a conectividade
        cv2.circle(img_teste, (200, 200), 20, 255, -1)
        cv2.circle(img_teste, (230, 200), 20, 255, -1)

        cv2.imwrite(nome_arquivo, img_teste)
        img_binaria = img_teste
    else:
        # Converte a imagem para escala de cinza e binariza
        img_gray = cv2.cvtColor(img_colorida, cv2.COLOR_BGR2GRAY)
        _, img_binaria = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        
    return img_binaria

# -------------------------
# Execução principal do script
# -------------------------
if __name__ == "__main__":
    # Carrega ou gera a imagem binária
    img_binaria = carregar_ou_gerar_imagem_binaria(NOME_IMAGEM_ENTRADA)

    if img_binaria is None:
        print("Erro ao carregar ou gerar a imagem.")
        exit()

    # Aplica a rotulação de componentes conectados e mede o tempo
    start_time = time.time()
    num_labels, labels, stats, centroids = rotular_componentes_conectados(img_binaria)
    end_time = time.time()
    tempo_processamento = end_time - start_time

    print(f"Tempo de processamento: {tempo_processamento:.4f}s")
    print(f"Número total de rótulos (incluindo o background): {num_labels}")
    print(f"Número de objetos/componentes conectados: {num_labels - 1}")

    # Cria uma imagem colorida para visualização dos rótulos
    output_image = np.zeros((img_binaria.shape[0], img_binaria.shape[1], 3), dtype=np.uint8)

    # Percorre cada rótulo, exceto o background (rótulo 0)
    for i in range(1, num_labels):
        cor = np.random.randint(0, 256, size=3, dtype=np.uint8)
        output_image[labels == i] = cor

    # Exibe as imagens lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Imagem Original Binarizada
    axes[0].imshow(img_binaria, cmap='gray')
    axes[0].set_title('Imagem Binária Original')
    axes[0].axis('off')
    
    # Imagem com Componentes Rotulados
    axes[1].imshow(output_image)
    axes[1].set_title(f'Componentes Rotulados\nTempo: {tempo_processamento:.4f}s')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()