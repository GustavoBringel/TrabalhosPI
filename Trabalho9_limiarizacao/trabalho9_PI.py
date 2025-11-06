import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# =========================================================
# CONFIGURAÇÃO GERAL
# =========================================================
NOME_IMAGEM_ENTRADA = "imagem_limiarizacao.png"
# =========================================================

# --- Funções Auxiliares de Carregamento e Geração ---

def gerar_imagem_com_vale(nome_arquivo):
    """Gera uma imagem de teste com um histograma bimodal (dois picos/um vale)."""
    print("Gerando imagem de teste bimodal para o Método do Vale...")
    img_teste = np.zeros((300, 300), dtype=np.uint8)
    
    # Fundo: Nível de cinza baixo (Pico 1)
    img_teste.fill(60) 
    
    # Objeto: Nível de cinza alto (Pico 2)
    cv2.circle(img_teste, (150, 150), 100, 180, -1)
    
    # Adiciona um pequeno ruído gaussiano para simular um histograma real
    ruido = np.random.normal(0, 5, (300, 300)).astype(np.uint8)
    img_teste = cv2.add(img_teste, ruido)
    
    cv2.imwrite(nome_arquivo, img_teste)
    return img_teste

def carregar_ou_gerar_imagem(nome_arquivo):
    """Carrega a imagem se existir; caso contrário, gera e salva uma nova."""
    if os.path.exists(nome_arquivo):
        print(f"Carregando imagem existente: {nome_arquivo}")
        img = cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    
    return gerar_imagem_com_vale(nome_arquivo)

# ----------------------------------------------------
# ALGORITMO COMPLETO: Método do Vale para Limiarização
# ----------------------------------------------------

def _calcular_histograma(imagem_gray):
    """Calcula o histograma manualmente."""
    hist = np.zeros(256, dtype=int)
    for pixel in imagem_gray.flatten():
        hist[pixel] += 1
    return hist

def encontrar_limiar_metodo_vale(hist):
    """
    Encontra o limiar pelo método do vale.
    O algoritmo busca o ponto mais baixo (o vale) entre os dois picos mais altos.
    """
    
    # Suaviza o histograma para facilitar a identificação de picos e vales
    # Uma suavização simples por média móvel é suficiente
    hist_suave = np.convolve(hist, np.ones(5)/5, mode='same')
    
    # 1. Encontrar os dois picos mais altos
    # Simplesmente encontra o primeiro e o segundo maior valor no histograma suavizado
    picos_candidatos = []
    # Usamos uma técnica simples: procurar pontos altos
    for i in range(1, 255):
        if hist_suave[i] > hist_suave[i-1] and hist_suave[i] > hist_suave[i+1] and hist_suave[i] > 100:
            picos_candidatos.append((hist_suave[i], i))
    
    if len(picos_candidatos) < 2:
        print("Aviso: Menos de dois picos detectados. Retornando 127 como padrão.")
        return 127
    
    # Seleciona os dois picos mais proeminentes (não necessariamente os maiores absolutos)
    # Aqui, simplificamos selecionando o primeiro e o último pico significativo
    pico1_idx = picos_candidatos[0][1]
    pico2_idx = picos_candidatos[-1][1]

    # Garante que pico1_idx < pico2_idx
    if pico1_idx > pico2_idx:
        pico1_idx, pico2_idx = pico2_idx, pico1_idx
        
    # 2. Encontrar o vale (mínimo) entre os dois picos
    vale_min_val = np.inf
    limiar_vale = -1
    
    # Procura o valor mínimo (o vale) apenas no intervalo entre os dois picos
    for i in range(pico1_idx, pico2_idx):
        if hist_suave[i] < vale_min_val:
            vale_min_val = hist_suave[i]
            limiar_vale = i
            
    return limiar_vale

def aplicar_limiarizacao_manual(imagem_gray, limiar):
    """
    Pixels > Limiar = 255 (Objeto)
    Pixels <= Limiar = 0 (Fundo)
    """
    imagem_segmentada = np.zeros_like(imagem_gray, dtype=np.uint8)
    
    for y in range(imagem_gray.shape[0]):
        for x in range(imagem_gray.shape[1]):
            if imagem_gray[y, x] > limiar:
                imagem_segmentada[y, x] = 255
            else:
                imagem_segmentada[y, x] = 0
                
    return imagem_segmentada

# ----------------------------------------------------
# Execução Principal
# ----------------------------------------------------

if __name__ == "__main__":
    img_gray = carregar_ou_gerar_imagem(NOME_IMAGEM_ENTRADA)
    if img_gray is None:
        exit()
    
    # 1. Cálculo do Histograma
    hist = _calcular_histograma(img_gray)
    
    # 2. Encontrar o Limiar (Método do Vale) e medir o tempo
    start_time_limiar = time.time()
    limiar_encontrado = encontrar_limiar_metodo_vale(hist)
    tempo_limiar = time.time() - start_time_limiar
    
    # 3. Aplicar a Limiarização e medir o tempo
    start_time_segmentacao = time.time()
    img_segmentada = aplicar_limiarizacao_manual(img_gray, limiar_encontrado)
    tempo_segmentacao = time.time() - start_time_segmentacao
    
    
    # --- SALVAMENTO E EXIBIÇÃO DE RESULTADOS ---
    cv2.imwrite("resultado_segmentacao_vale.png", img_segmentada)
    print(f"\nLimiar Encontrado (Método do Vale): {limiar_encontrado}")
    print(f"Tempo para encontrar o Limiar: {tempo_limiar:.4f}s")
    print(f"Tempo para aplicar a Segmentação: {tempo_segmentacao:.4f}s")
    print("Imagem segmentada salva como 'resultado_segmentacao_vale.png'")
    
    # Plotagem
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Imagem Original
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # 2. Histograma com o Limiar
    axes[1].plot(hist, color='black')
    axes[1].axvline(x=limiar_encontrado, color='red', linestyle='--', label=f'Limiar = {limiar_encontrado}')
    axes[1].set_title('Histograma e Limiar do Vale')
    axes[1].set_xlabel('Nível de Cinza')
    axes[1].set_ylabel('Frequência de Pixels')
    axes[1].legend()
    
    # 3. Imagem Segmentada
    axes[2].imshow(img_segmentada, cmap='gray')
    axes[2].set_title(f'Segmentação Binária (Limiar: {limiar_encontrado})\nTempo Total: {(tempo_limiar + tempo_segmentacao):.4f}s')
    axes[2].axis('off')

    plt.suptitle('Implementação Completa: Segmentação por Limiarização (Método do Vale)', fontsize=16)
    plt.tight_layout()
    plt.show()