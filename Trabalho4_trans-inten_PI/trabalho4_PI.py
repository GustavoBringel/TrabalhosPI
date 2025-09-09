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
# Funções de Transformação de Intensidade
# -------------------------
def transformacao_negativa(img):
    """Calcula o negativo da imagem."""
    # Transforma pixels P em P_neg = 255 - P
    return 255 - img

def transformacao_logaritmica(img, c=1.0):
    """
    Aplica a transformação logarítmica para expandir valores escuros.
    f(P) = c * log(1 + P)
    """
    img_float = img.astype(np.float32) + 1
    log_transformed = c * np.log(img_float)
    # Normaliza e converte de volta para o tipo original
    normalized = (log_transformed / np.max(log_transformed)) * 255
    return normalized.astype(np.uint8)

def transformacao_potencia(img, gamma=1.5):
    """
    Aplica a transformação de potência (correção gama).
    f(P) = P^gamma
    """
    # Normaliza os valores de pixel para o intervalo [0, 1]
    normalized_img = img.astype(np.float32) / 255.0
    # Aplica a transformação de potência
    gamma_corrected = np.power(normalized_img, gamma)
    # Re-escala para o intervalo [0, 255]
    return (gamma_corrected * 255).astype(np.uint8)

# -------------------------
# Função para carregar ou gerar a imagem
# -------------------------
def carregar_ou_gerar_imagem(nome_arquivo):
    img = cv2.imread(nome_arquivo)
    if img is None:
        print(f"Arquivo '{nome_arquivo}' não encontrado. Gerando uma imagem de teste...")
        largura, altura = 256, 256
        img_teste = np.zeros((altura, largura, 3), dtype=np.uint8)
        
        # Desenha um gradiente de cinza para testar as transformações
        for y in range(altura):
            img_teste[y, :, 0] = y
            img_teste[y, :, 1] = y
            img_teste[y, :, 2] = y
        
        cv2.imwrite(nome_arquivo, img_teste)
        img = cv2.imread(nome_arquivo)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------------------------
# Execução principal do script
# -------------------------
if __name__ == "__main__":
    img = carregar_ou_gerar_imagem(NOME_IMAGEM_ENTRADA)
    if img is None:
        exit()
    
    # Dicionário para armazenar resultados e tempos
    resultados = {}

    # --- Medição de Tempo e Execução das Transformações ---
    
    # Transformação Negativa
    start_time_neg = time.time()
    img_neg = transformacao_negativa(img)
    end_time_neg = time.time()
    resultados['Negativa'] = (img_neg, end_time_neg - start_time_neg)
    
    # Transformação Logarítmica
    start_time_log = time.time()
    img_log = transformacao_logaritmica(img)
    end_time_log = time.time()
    resultados['Logarítmica'] = (img_log, end_time_log - start_time_log)
    
    # Transformação de Potência
    start_time_pot = time.time()
    img_pot = transformacao_potencia(img, gamma=1.5)
    end_time_pot = time.time()
    resultados['Potência (γ=1.5)'] = (img_pot, end_time_pot - start_time_pot)
    
    # --- Exibição dos Resultados ---
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes = axes.flatten()
    
    # Plota a imagem original
    axes[0].imshow(img)
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')
    
    # Plota os resultados das transformações
    axes[1].imshow(resultados['Negativa'][0])
    axes[1].set_title(f"Negativa\nTempo: {resultados['Negativa'][1]:.4f}s")
    axes[1].axis('off')

    axes[2].imshow(resultados['Logarítmica'][0])
    axes[2].set_title(f"Logarítmica\nTempo: {resultados['Logarítmica'][1]:.4f}s")
    axes[2].axis('off')

    axes[3].imshow(resultados['Potência (γ=1.5)'][0])
    axes[3].set_title(f"Potência (γ=1.5)\nTempo: {resultados['Potência (γ=1.5)'][1]:.4f}s")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()