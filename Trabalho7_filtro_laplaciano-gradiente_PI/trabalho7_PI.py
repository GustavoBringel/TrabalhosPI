import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# =========================================================
# CONFIGURAÇÃO GERAL
# =========================================================
NOME_IMAGEM_ENTRADA = "image1.png"
# Opção para lidar com valores negativos:
# 1 = Atribuir 0
# 2 = Reescalar para [0, 255]
OPCAO_VALORES_NEGATIVOS = 2 
# =========================================================

# -------------------------
# Filtros de Detecção de Borda
# -------------------------

def filtro_laplaciano(imagem_gray, mascara, opcao_negativos):
    """
    Aplica o Filtro Laplaciano para detecção de bordas.
    """
    altura, largura = imagem_gray.shape
    borda = mascara.shape[0] // 2
    
    imagem_padded = np.zeros((altura + 2 * borda, largura + 2 * borda), dtype=np.float32)
    imagem_padded[borda:borda + altura, borda:borda + largura] = imagem_gray.astype(np.float32)
    
    imagem_filtrada = np.zeros_like(imagem_gray, dtype=np.float32)
    
    for y in range(altura):
        for x in range(largura):
            roi = imagem_padded[y:y + mascara.shape[0], x:x + mascara.shape[1]]
            conv = np.sum(roi * mascara)
            imagem_filtrada[y, x] = conv
            
    # Tratamento de valores negativos
    if opcao_negativos == 1:
        imagem_filtrada[imagem_filtrada < 0] = 0
    elif opcao_negativos == 2:
        imagem_filtrada = cv2.normalize(imagem_filtrada, None, 0, 255, cv2.NORM_MINMAX)
        
    return imagem_filtrada.astype(np.uint8)


def filtro_sobel(imagem_gray, opcao_negativos):
    """
    Aplica o Filtro Gradiente (Sobel) para detecção de bordas.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    altura, largura = imagem_gray.shape
    borda = 1 # Para kernel 3x3

    imagem_padded = np.zeros((altura + 2 * borda, largura + 2 * borda), dtype=np.float32)
    imagem_padded[borda:borda + altura, borda:borda + largura] = imagem_gray.astype(np.float32)
    
    gradiente_x = np.zeros_like(imagem_gray, dtype=np.float32)
    gradiente_y = np.zeros_like(imagem_gray, dtype=np.float32)

    for y in range(altura):
        for x in range(largura):
            roi = imagem_padded[y:y + 3, x:x + 3]
            gradiente_x[y, x] = np.sum(roi * sobel_x)
            gradiente_y[y, x] = np.sum(roi * sobel_y)

    # Magnitude do gradiente: G = sqrt(Gx^2 + Gy^2)
    gradiente_magnitude = np.sqrt(np.square(gradiente_x) + np.square(gradiente_y))

    # Tratamento de valores negativos (não aplicável à magnitude, mas incluso por consistência)
    if opcao_negativos == 1:
        gradiente_magnitude[gradiente_magnitude < 0] = 0
    elif opcao_negativos == 2:
        gradiente_magnitude = cv2.normalize(gradiente_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    return gradiente_magnitude.astype(np.uint8)

# -------------------------
# Funções Auxiliares
# -------------------------
def carregar_ou_gerar_imagem(nome_arquivo):
    img = cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Arquivo '{nome_arquivo}' não encontrado. Gerando uma imagem de teste...")
        img_teste = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(img_teste, (200, 200), 100, 255, -1)
        cv2.rectangle(img_teste, (50, 50), (150, 150), 100, -1)
        cv2.putText(img_teste, 'Teste', (250, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
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

    # Máscaras Laplacianas propostas
    mascaras_laplacianas = {
        'Máscara 1': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
        'Máscara 2': np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32),
        'Máscara 3': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
        'Máscara 4': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),
    }

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    axes = axes.flatten()

    # Imagem Original
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')
    
    # Filtro de Sobel (Gradiente)
    start_time_sobel = time.time()
    img_sobel = filtro_sobel(img_gray, OPCAO_VALORES_NEGATIVOS)
    end_time_sobel = time.time()
    axes[1].imshow(img_sobel, cmap='gray')
    axes[1].set_title(f'Filtro de Sobel\nTempo: {end_time_sobel - start_time_sobel:.4f}s')
    axes[1].axis('off')
    
    # Filtros Laplacianos
    i = 2
    for nome, mascara in mascaras_laplacianas.items():
        start_time_lap = time.time()
        img_laplaciano = filtro_laplaciano(img_gray, mascara, OPCAO_VALORES_NEGATIVOS)
        end_time_lap = time.time()
        
        axes[i].imshow(img_laplaciano, cmap='gray')
        axes[i].set_title(f'{nome}\nTempo: {end_time_lap - start_time_lap:.4f}s')
        axes[i].axis('off')
        i += 1
    
    plt.tight_layout()
    plt.show()