import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURAÇÃO: Nomes de arquivos de imagem
# =========================================================
NOME_IMAGEM_1 = "image1.png"
NOME_IMAGEM_2 = "image4.png"
# =========================================================

# -------------------------
# Operações Aritméticas
# -------------------------
def adicao_imagens(img1, img2):
    """Soma os valores de pixel de duas imagens, com saturação."""
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.add(img1, img2_resized)

def subtracao_imagens(img1, img2):
    """Subtrai os valores de pixel, com saturação em zero."""
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.subtract(img1, img2_resized)

def multiplicacao_imagens(img1, img2):
    """Multiplica os valores de pixel de duas imagens, normalizando o resultado."""
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    multiplicacao = cv2.multiply(img1.astype(np.float32), img2_resized.astype(np.float32))
    multiplicacao = np.clip(multiplicacao, 0, 255).astype(np.uint8)
    return multiplicacao

def divisao_imagens(img1, img2):
    """Divide os valores de pixel da primeira imagem pela segunda."""
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img2_resized[img2_resized == 0] = 1
    divisao = cv2.divide(img1, img2_resized, scale=255)
    return divisao

# -------------------------
# Operações Geométricas
# -------------------------
def rotacao_imagem(img, angulo, escala=1.0):
    """Rotaciona a imagem por um determinado ângulo."""
    (h, w) = img.shape[:2]
    centro = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centro, angulo, escala)
    return cv2.warpAffine(img, M, (w, h))

def translacao_imagem(img, dx, dy):
    """Translada (move) a imagem por (dx, dy) pixels."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def espelhamento_horizontal(img):
    """Espelha a imagem no eixo vertical (reflexão horizontal)."""
    return cv2.flip(img, 1)

def espelhamento_vertical(img):
    """Espelha a imagem no eixo horizontal (reflexão vertical)."""
    return cv2.flip(img, 0)

# -------------------------
# Função para carregar ou gerar as imagens de teste
# -------------------------
def carregar_ou_gerar_imagens(nome1, nome2):
    img1 = cv2.imread(nome1)
    img2 = cv2.imread(nome2)
    
    if img1 is None or img2 is None:
        print(f"Arquivos '{nome1}' e/ou '{nome2}' não encontrados. Gerando imagens de teste...")
        
        # Gerar imagem de teste 1 (círculo)
        img_teste1 = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img_teste1, (100, 100), 50, (255, 0, 0), -1)
        cv2.putText(img_teste1, 'Img 1', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(nome1, img_teste1)
        
        # Gerar imagem de teste 2 (quadrado)
        img_teste2 = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img_teste2, (75, 75), (150, 150), (0, 255, 0), -1)
        cv2.putText(img_teste2, 'Img 2', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(nome2, img_teste2)
        
        img1 = img_teste1
        img2 = img_teste2
        
    return img1, img2

if __name__ == "__main__":
    img_orig1, img_orig2 = carregar_ou_gerar_imagens(NOME_IMAGEM_1, NOME_IMAGEM_2)

    img1_rgb = cv2.cvtColor(img_orig1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img_orig2, cv2.COLOR_BGR2RGB)

    # Aplica as operações
    img_adicao = adicao_imagens(img_orig1, img_orig2)
    img_subtracao = subtracao_imagens(img_orig1, img_orig2)
    img_multiplicacao = multiplicacao_imagens(img_orig1, img_orig2)
    img_divisao = divisao_imagens(img_orig1, img_orig2)
    
    img_rotacao = rotacao_imagem(img_orig1, 45)
    img_translacao = translacao_imagem(img_orig1, 50, 30)
    img_esp_horiz = espelhamento_horizontal(img_orig1)
    img_esp_vert = espelhamento_vertical(img_orig1)

    # Exibe os resultados
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Operações Aritméticas e Geométricas em Imagens', fontsize=18)

    # Linha 1: Imagens Originais
    axes[0, 0].imshow(img1_rgb)
    axes[0, 0].set_title('Original (Imagem 1)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_rgb)
    axes[0, 1].set_title('Original (Imagem 2)')
    axes[0, 1].axis('off')

    # Linha 1: Operações Aritméticas
    axes[0, 2].imshow(cv2.cvtColor(img_adicao, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Adição')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(cv2.cvtColor(img_subtracao, cv2.COLOR_BGR2RGB))
    axes[0, 3].set_title('Subtração')
    axes[0, 3].axis('off')

    # Linha 2: Operações Aritméticas
    axes[1, 0].imshow(cv2.cvtColor(img_multiplicacao, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Multiplicação')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(img_divisao, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Divisão')
    axes[1, 1].axis('off')
    
    # Linha 2: Operações Geométricas
    axes[1, 2].imshow(cv2.cvtColor(img_rotacao, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Rotação (45°)')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(cv2.cvtColor(img_translacao, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title('Translação (+50, +30)')
    axes[1, 3].axis('off')
    
    # Linha 3: Operações Geométricas
    axes[2, 0].imshow(cv2.cvtColor(img_esp_horiz, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title('Espelhamento Horizontal')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(cv2.cvtColor(img_esp_vert, cv2.COLOR_BGR2RGB))
    axes[2, 1].set_title('Espelhamento Vertical')
    axes[2, 1].axis('off')
    
    # Remove os subplots vazios
    for i in range(2, 4):
        axes[2, i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()