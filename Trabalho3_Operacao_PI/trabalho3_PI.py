import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# =========================================================
# CONFIGURAÇÃO: NOME DO ARQUIVO DE IMAGEM
# =========================================================
NOME_IMAGEM_ENTRADA = "image4.png"
# =========================================================

# -------------------------
# Funções de Operações Aritméticas e Geométricas
# -------------------------

# Adição (com uma constante)
def adicao_constante(img, valor):
    """Adiciona um valor constante aos pixels da imagem."""
    return cv2.add(img, valor)

# Subtração (com uma constante)
def subtracao_constante(img, valor):
    """Subtrai um valor constante dos pixels da imagem."""
    return cv2.subtract(img, valor)

# Multiplicação (com uma constante)
def multiplicacao_constante(img, valor):
    """Multiplica os pixels da imagem por um valor constante, com saturação."""
    return cv2.multiply(img, valor)

# Divisão (com uma constante)
def divisao_constante(img, valor):
    """Divide os pixels da imagem por um valor constante."""
    return cv2.divide(img, valor)

# Rotação
def rotacao_imagem(img, angulo, escala=1.0):
    """Rotaciona a imagem por um determinado ângulo."""
    (h, w) = img.shape[:2]
    centro = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centro, angulo, escala)
    return cv2.warpAffine(img, M, (w, h))

# Translação
def translacao_imagem(img, dx, dy):
    """Translada (move) a imagem por (dx, dy) pixels."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# Espelhamento Horizontal
def espelhamento_horizontal(img):
    """Espelha a imagem no eixo vertical (reflexão horizontal)."""
    return cv2.flip(img, 1)

# Espelhamento Vertical
def espelhamento_vertical(img):
    """Espelha a imagem no eixo horizontal (reflexão vertical)."""
    return cv2.flip(img, 0)

# -------------------------
# Função para carregar ou gerar a imagem de teste
# -------------------------
def carregar_ou_gerar_imagem(nome_arquivo):
    img = cv2.imread(nome_arquivo)
    
    if img is None:
        print(f"Arquivo '{nome_arquivo}' não encontrado. Gerando uma imagem de teste...")
        img_teste = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img_teste, (100, 100), 50, (255, 0, 0), -1)
        cv2.putText(img_teste, 'Teste', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(nome_arquivo, img_teste)
        img = img_teste
    
    return img

# -------------------------
# Execução principal do script
# -------------------------
if __name__ == "__main__":
    img_original = carregar_ou_gerar_imagem(NOME_IMAGEM_ENTRADA)
    if img_original is None:
        exit()

    # Conversão para o formato RGB para exibição no matplotlib
    img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

    # Dicionário para armazenar os resultados e tempos
    resultados = {}

    # --- Medição de Tempo e Execução das Operações ---

    # Adição
    start_time = time.time()
    img_adicao = adicao_constante(img_original, 50)
    end_time = time.time()
    resultados['Adição (+50)'] = (img_adicao, end_time - start_time)

    # Subtração
    start_time = time.time()
    img_subtracao = subtracao_constante(img_original, 50)
    end_time = time.time()
    resultados['Subtração (-50)'] = (img_subtracao, end_time - start_time)
    
    # Multiplicação
    start_time = time.time()
    img_multiplicacao = multiplicacao_constante(img_original, 1.5)
    end_time = time.time()
    resultados['Multiplicação (*1.5)'] = (img_multiplicacao, end_time - start_time)

    # Divisão
    start_time = time.time()
    img_divisao = divisao_constante(img_original, 2)
    end_time = time.time()
    resultados['Divisão (/2)'] = (img_divisao, end_time - start_time)

    # Rotação
    start_time = time.time()
    img_rotacao = rotacao_imagem(img_original, 45)
    end_time = time.time()
    resultados['Rotação (45°)'] = (img_rotacao, end_time - start_time)

    # Translação
    start_time = time.time()
    img_translacao = translacao_imagem(img_original, -50, -30)
    end_time = time.time()
    resultados['Translação (+50, +30)'] = (img_translacao, end_time - start_time)

    # Espelhamento Horizontal
    start_time = time.time()
    img_esp_horiz = espelhamento_horizontal(img_original)
    end_time = time.time()
    resultados['Espelhamento Horizontal'] = (img_esp_horiz, end_time - start_time)

    # Espelhamento Vertical
    start_time = time.time()
    img_esp_vert = espelhamento_vertical(img_original)
    end_time = time.time()
    resultados['Espelhamento Vertical'] = (img_esp_vert, end_time - start_time)

    # --- Exibição dos Resultados ---
    num_operacoes = len(resultados)
    num_cols = 3
    num_rows = (num_operacoes + 1 + num_cols - 1) // num_cols # +1 para a imagem original
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    # Plota a imagem original
    axes[0].imshow(img_original_rgb)
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')

    # Plota os resultados das operações
    for i, (titulo, (img_result, tempo)) in enumerate(resultados.items()):
        ax = axes[i + 1]
        ax.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{titulo}\nTempo: {tempo:.4f}s')
        ax.axis('off')

    # Oculta eixos não utilizados
    for i in range(num_operacoes + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()