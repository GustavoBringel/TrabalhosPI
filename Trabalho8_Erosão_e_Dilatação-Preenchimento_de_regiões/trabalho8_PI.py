import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os # Para verificar a existência do arquivo

# =========================================================
# CONFIGURAÇÃO GERAL
# =========================================================
NOME_IMAGEM_BINARIA = "img2.png"
TAMANHO_KERNEL = 5
TAMANHO_IMG = 300
# =========================================================

# --- Funções Auxiliares de Carregamento e Geração ---

def gerar_imagem_binaria_aleatoria(nome_arquivo):
    """Gera uma imagem de teste binária com formas aleatórias e ruído."""
    
    print("Gerando imagem de teste binária aleatória...")
    # Usa o tempo como semente para garantir aleatoriedade
    np.random.seed(int(time.time() * 1000) % 10000) 
    
    img_teste = np.zeros((TAMANHO_IMG, TAMANHO_IMG), dtype=np.uint8)
    
    # 1. Desenha formas aleatórias (Objetos Principais)
    num_formas = np.random.randint(3, 7)
    for _ in range(num_formas):
        tipo_forma = np.random.choice(['circle', 'rectangle'])
        
        # Gera posições e tamanhos aleatórios
        x = np.random.randint(50, TAMANHO_IMG - 50)
        y = np.random.randint(50, TAMANHO_IMG - 50)
        
        if tipo_forma == 'circle':
            raio = np.random.randint(10, 40)
            cv2.circle(img_teste, (x, y), raio, 255, -1)
        else: # rectangle
            w = np.random.randint(20, 80)
            h = np.random.randint(20, 80)
            cv2.rectangle(img_teste, (x - w//2, y - h//2), (x + w//2, y + h//2), 255, -1)

    # 2. Cria um buraco aleatório para o teste de Preenchimento
    buraco_x = np.random.randint(100, 200)
    buraco_y = np.random.randint(100, 200)
    raio_buraco = np.random.randint(15, 25)
    cv2.circle(img_teste, (buraco_x, buraco_y), raio_buraco, 0, -1)
    
    # 3. Adiciona Ruído (Salt and Pepper)
    ruido_prob = 0.05 
    for i in range(TAMANHO_IMG):
        for j in range(TAMANHO_IMG):
            r = np.random.random()
            if r < ruido_prob / 2:
                img_teste[i, j] = 0
            elif r < ruido_prob:
                img_teste[i, j] = 255
    
    # Binariza a imagem e salva
    _, img_binaria = cv2.threshold(img_teste, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(nome_arquivo, img_binaria)
    
    return img_binaria

def carregar_ou_gerar_imagem_binaria(nome_arquivo):
    """Carrega a imagem se existir; caso contrário, gera e salva uma nova."""
    if os.path.exists(nome_arquivo):
        print(f"Carregando imagem existente: {nome_arquivo}")
        img = cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Garante que a imagem carregada seja binária
            _, img_binaria = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            return img_binaria
    
    # Se o carregamento falhar ou o arquivo não existir, gera uma nova
    return gerar_imagem_binaria_aleatoria(nome_arquivo)


# ----------------------------------------------------
# ALGORITMO COMPLETO: Opção 1 - Erosão e Dilatação (IMPLEMENTAÇÃO MANUAL)
# ----------------------------------------------------

def _aplicar_operacao_morfologica(img_binaria, kernel_size, operacao):
    """Implementação manual da Erosão (min) ou Dilatação (max)."""
    altura, largura = img_binaria.shape
    borda = kernel_size // 2
    
    imagem_padded = np.pad(img_binaria, borda, mode='constant', constant_values=0)
    imagem_result = np.zeros_like(img_binaria, dtype=np.uint8)
    
    for y in range(altura):
        for x in range(largura):
            roi = imagem_padded[y:y + kernel_size, x:x + kernel_size]
            
            if operacao == 'Erosao':
                valor_saida = np.min(roi) 
            elif operacao == 'Dilatacao':
                valor_saida = np.max(roi)
            
            imagem_result[y, x] = valor_saida
            
    return imagem_result

def aplicar_erosao_completo(img_binaria, kernel_size):
    return _aplicar_operacao_morfologica(img_binaria, kernel_size, 'Erosao')

def aplicar_dilatacao_completo(img_binaria, kernel_size):
    return _aplicar_operacao_morfologica(img_binaria, kernel_size, 'Dilatacao')


# ----------------------------------------------------
# ALGORITMO COMPLETO: Opção 4 - Preenchimento de Regiões (Hole Filling)
# ----------------------------------------------------

def preenchimento_de_regioes_completo(img_binaria):
    """
    Implementa o Preenchimento de Regiões (Hole Filling)
    usando a inversão e o preenchimento por inundação (Flood Fill).
    """
    altura, largura = img_binaria.shape
    
    # 1. Imagem invertida (buracos internos = 255)
    img_inv = cv2.bitwise_not(img_binaria)
    
    # 2. Máscara para o Flood Fill (H+2, W+2)
    mask = np.zeros((altura + 2, largura + 2), np.uint8)
    
    # 3. Aplica o Flood Fill partindo da borda (0, 0) na imagem invertida
    cv2.floodFill(img_inv, mask, (0, 0), 0)
    
    # 4. Inverte o resultado do Flood Fill.
    img_buracos = cv2.bitwise_not(img_inv)
    
    # 5. Combina (OR) a imagem original com os buracos preenchidos.
    img_preenchida = cv2.bitwise_or(img_binaria, img_buracos)
    
    return img_preenchida

# ----------------------------------------------------
# Execução Principal
# ----------------------------------------------------

if __name__ == "__main__":
    img_original = carregar_ou_gerar_imagem_binaria(NOME_IMAGEM_BINARIA)
    
    resultados = {}
    
    # --- 1. Erosão ---
    start_time = time.time()
    img_erosao = aplicar_erosao_completo(img_original, TAMANHO_KERNEL)
    tempo_erosao = time.time() - start_time
    resultados[f'1. Erosão (T: {tempo_erosao:.4f}s)'] = img_erosao
    
    # --- 1. Dilatação ---
    start_time = time.time()
    img_dilatacao = aplicar_dilatacao_completo(img_original, TAMANHO_KERNEL)
    tempo_dilatacao = time.time() - start_time
    resultados[f'1. Dilatação (T: {tempo_dilatacao:.4f}s)'] = img_dilatacao
    
    # --- 4. Preenchimento de Regiões ---
    start_time = time.time()
    img_preenchida = preenchimento_de_regioes_completo(img_original)
    tempo_preenchimento = time.time() - start_time
    resultados[f'4. Preenchimento (T: {tempo_preenchimento:.4f}s)'] = img_preenchida
    
    # --- SALVAMENTO DAS NOVAS IMAGENS ---
    cv2.imwrite("resultado_erosao.png", img_erosao)
    cv2.imwrite("resultado_dilatacao.png", img_dilatacao)
    cv2.imwrite("resultado_preenchimento.png", img_preenchida)
    print("\nNovas imagens salvas:")
    print(" - resultado_erosao.png")
    print(" - resultado_dilatacao.png")
    print(" - resultado_preenchimento.png")
    
    # --- Plotagem de Resultados ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes = axes.flatten()
    
    axes[0].imshow(img_original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for i, (titulo, img_result) in enumerate(resultados.items()):
        ax = axes[i + 1]
        ax.imshow(img_result, cmap='gray')
        ax.set_title(titulo)
        ax.axis('off')

    plt.suptitle('Implementação Completa: Erosão, Dilatação e Preenchimento de Regiões', fontsize=16)
    plt.tight_layout()
    plt.show()