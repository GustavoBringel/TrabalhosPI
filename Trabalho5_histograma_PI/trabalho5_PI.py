import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# =========================================================
# CONFIGURAÇÃO: NOME DO ARQUIVO DE IMAGEM
# =========================================================
NOME_IMAGEM_ENTRADA = "image1.png"
# =========================================================

# -------------------------
# Algoritmo de Equalização de Histograma (completo)
# -------------------------
def equalizar_histograma_completo(imagem_gray):
    """
    Aplica o algoritmo completo de equalização de histograma em uma imagem
    em escala de cinza, sem usar funções prontas de bibliotecas.
    """
    # Passo 1: Calcular o histograma da imagem
    hist = np.zeros(256, dtype=int)
    altura, largura = imagem_gray.shape
    for y in range(altura):
        for x in range(largura):
            pixel_value = imagem_gray[y, x]
            hist[pixel_value] += 1
    
    # Passo 2: Calcular a Função de Distribuição Acumulada (CDF)
    cdf = np.zeros(256, dtype=float)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
        
    # Passo 3: Normalizar a CDF
    # O valor máximo da CDF é o número total de pixels na imagem.
    # Normalizamos para o intervalo [0, 255].
    total_pixels = altura * largura
    cdf_normalizada = np.round((cdf / total_pixels) * 255)
    
    # Passo 4: Mapear os pixels da imagem original para os novos valores
    imagem_equalizada = np.zeros_like(imagem_gray, dtype=np.uint8)
    for y in range(altura):
        for x in range(largura):
            pixel_value = imagem_gray[y, x]
            imagem_equalizada[y, x] = cdf_normalizada[pixel_value]
            
    return imagem_equalizada

# -------------------------
# Função para carregar ou gerar a imagem
# -------------------------
def carregar_ou_gerar_imagem(nome_arquivo):
    img = cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Arquivo '{nome_arquivo}' não encontrado. Gerando uma imagem de teste...")
        img_teste = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img_teste, (0, 0), (256, 256), 50, -1)
        cv2.rectangle(img_teste, (50, 50), (200, 200), 120, -1)
        cv2.circle(img_teste, (128, 128), 50, 180, -1)
        
        cv2.imwrite(nome_arquivo, img_teste)
        img = cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)
    
    return img

# -------------------------
# Execução Principal
# -------------------------
if __name__ == "__main__":
    
    img_gray = carregar_ou_gerar_imagem(NOME_IMAGEM_ENTRADA)
    if img_gray is None:
        exit()
    
    # Aplica a equalização do histograma e mede o tempo
    start_time = time.time()
    img_equalizada = equalizar_histograma_completo(img_gray)
    end_time = time.time()
    
    tempo_processamento = end_time - start_time
    
    # Exibe as imagens original e equalizada
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'Equalização de Histograma\nTempo de Processamento: {tempo_processamento:.4f}s', fontsize=16)
    
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Original (Baixo Contraste)')
    axes[0].axis('off')
    
    axes[1].imshow(img_equalizada, cmap='gray')
    axes[1].set_title('Equalizada (Alto Contraste)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()