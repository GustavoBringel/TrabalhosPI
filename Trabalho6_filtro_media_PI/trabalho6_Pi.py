import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# =========================================================
# CONFIGURAÇÃO GERAL
# =========================================================
NOME_IMAGEM_ENTRADA = "image1.png"
TAMANHO_KERNEL = 9  # Tamanho da máscara do filtro (deve ser ímpar: 3, 5, 9, 15, 35)
# =========================================================

# -------------------------
# Implementação do Filtro da Média
# -------------------------
def filtro_da_media(imagem_gray, tamanho_kernel):
    """
    Aplica o filtro da média em uma imagem em escala de cinza.
    A função trata as bordas com padding de zeros.
    
    Args:
        imagem_gray (np.ndarray): A imagem em escala de cinza.
        tamanho_kernel (int): O tamanho da máscara do filtro.
        
    Returns:
        np.ndarray: A imagem suavizada.
    """
    if tamanho_kernel % 2 == 0:
        raise ValueError("O tamanho do kernel deve ser um número ímpar.")

    altura, largura = imagem_gray.shape
    borda = tamanho_kernel // 2
    
    # Padding: Preenchimento da imagem com zeros para tratar as bordas
    imagem_padded = np.zeros((altura + 2 * borda, largura + 2 * borda), dtype=np.float32)
    imagem_padded[borda:borda + altura, borda:borda + largura] = imagem_gray.astype(np.float32)
    
    imagem_suavizada = np.zeros_like(imagem_gray, dtype=np.float32)
    
    # Convolução
    for y in range(altura):
        for x in range(largura):
            roi = imagem_padded[y:y + tamanho_kernel, x:x + tamanho_kernel]
            media = np.sum(roi) / (tamanho_kernel * tamanho_kernel)
            imagem_suavizada[y, x] = media
            
    return imagem_suavizada.astype(np.uint8)

# -------------------------
# Função para carregar ou gerar a imagem de teste
# -------------------------
def carregar_ou_gerar_imagem(nome_arquivo):
    img = cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Arquivo '{nome_arquivo}' não encontrado. Gerando uma imagem de teste...")
        largura, altura = 500, 500
        img_teste = np.zeros((altura, largura), dtype=np.uint8)
        
        # Adiciona elementos de teste (quadrados, texto, ruído)
        quadrados = [3, 5, 9, 15, 25, 35]
        pos_x = 20
        for tamanho in quadrados:
            cv2.rectangle(img_teste, (pos_x, 20), (pos_x + tamanho, 20 + tamanho), 255, -1)
            pos_x += tamanho + 25

        ruido = np.random.normal(0, 20, (50, 120)).astype(np.uint8)
        img_teste[400:450, 50:170] = ruido

        cv2.putText(img_teste, "a", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 5)
        
        cv2.imwrite(nome_arquivo, img_teste)
        img = cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)
        
    return img

# -------------------------
# Execução Principal
# -------------------------
if __name__ == "__main__":
    
    img_gray = carregar_ou_gerar_imagem(NOME_IMAGEM_ENTRADA)
    if img_gray is None:
        print("Erro: Imagem não encontrada ou gerada.")
        exit()
    
    # Mede o tempo de processamento
    start_time = time.time()
    img_suavizada = filtro_da_media(img_gray, TAMANHO_KERNEL)
    end_time = time.time()
    tempo_processamento = end_time - start_time
    
    print(f"Tempo de processamento: {tempo_processamento:.4f} segundos")
    
    # Exibe as imagens original e suavizada com o tempo e o título
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    fig.suptitle(f'Filtro da Média (Kernel {TAMANHO_KERNEL}x{TAMANHO_KERNEL})\nTempo: {tempo_processamento:.4f}s', fontsize=16)
    
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')
    
    axes[1].imshow(img_suavizada, cmap='gray')
    axes[1].set_title('Imagem Suavizada')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()