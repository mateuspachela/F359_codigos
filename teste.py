import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft

def selecionar_ponto(event, x, y, flags, param):
    global ponto_selecionado, ponto_x, ponto_y, escala_definida, escala_ponto1, escala_ponto2
    if event == cv2.EVENT_LBUTTONDOWN:
        if not escala_definida:
            if escala_ponto1 is None:
                escala_ponto1 = (x, y)
                print(f"Ponto 1 da escala selecionado: {escala_ponto1}")
            elif escala_ponto2 is None:
                escala_ponto2 = (x, y)
                escala_definida = True
                print(f"Ponto 2 da escala selecionado: {escala_ponto2}")
        elif escala_definida and not ponto_selecionado:
            ponto_selecionado = True
            ponto_x, ponto_y = x, y
            print(f"Ponto de rastreamento selecionado: ({ponto_x}, {ponto_y})")

# Variáveis globais para rastrear o ponto
ponto_selecionado = False
escala_definida = False
escala_ponto1 = None
escala_ponto2 = None
ponto_x, ponto_y = -1, -1

# Carregar o vídeo
video_path = r"C:\Users\giova\Downloads\material_unicamp\f359\primeiro_4mm.mp4"
cap = cv2.VideoCapture(video_path)

# Verificar se o vídeo foi carregado corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Obtendo o tamanho e FPS do vídeo
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Resolução do vídeo: {video_width}x{video_height} - FPS: {fps}")

# Criando uma janela que mantém a proporção original
cv2.namedWindow("Video", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Video", selecionar_ponto)

frame_index = 0
paused = True

# Exibir o primeiro frame
ret, frame = cap.read()
while True:
    if paused:
        # Exibir o quadro com o tamanho correto e sem zoom
        display_frame = frame.copy()
        if escala_ponto1:
            cv2.circle(display_frame, escala_ponto1, 5, (0, 255, 0), -1)
        if escala_ponto2:
            cv2.circle(display_frame, escala_ponto2, 5, (0, 255, 0), -1)
        if ponto_selecionado:
            cv2.circle(display_frame, (ponto_x, ponto_y), 5, (0, 0, 255), -1)
        cv2.imshow("Video", display_frame)

        # Controles do usuário
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Sair
            break
        elif key == ord('d'):  # Avançar um frame
            frame_index += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
        elif key == ord('a'):  # Voltar um frame
            frame_index = max(0, frame_index - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
        elif key == 13 and ponto_selecionado and escala_definida:  # Iniciar rastreamento (Enter)
            print("Ponto selecionado no quadro:", frame_index)
            break

# Verificar se a escala foi definida corretamente
if not escala_definida or not ponto_selecionado:
    print("Erro: A escala e o ponto de rastreamento devem ser definidos.")
    exit()

# Calculando a escala (distância em cm por pixel)
escala_pixels = np.sqrt((escala_ponto2[0] - escala_ponto1[0]) ** 2 +
                        (escala_ponto2[1] - escala_ponto1[1]) ** 2)
escala_cm = 10  # Ajuste para o valor da régua (por exemplo, 10 cm)
escala = escala_cm / escala_pixels
print(f"Escala: {escala:.4f} cm/pixel")

# Inicializando o rastreador CSRT (usando legacy)
tracker = cv2.legacy.TrackerCSRT_create()
trajetoria_y = []
tempos = []

# Iniciar o rastreamento do ponto selecionado
caixa_inicial = (ponto_x - 10, ponto_y - 10, 20, 20)
tracker.init(frame, caixa_inicial)
print("Rastreador inicializado no ponto:", caixa_inicial)

# Rastreamento automático do ponto selecionado
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Atualizando o rastreador
    success, bbox = tracker.update(frame)
    if success:
        _, y, _, h = [int(v) for v in bbox]
        centro_y = y + h // 2

        # Convertendo para centímetros e armazenando
        trajetoria_y.append((video_height - centro_y) * escala)
        tempos.append(len(trajetoria_y) / fps)

        # Desenhando o ponto rastreado
        cv2.circle(frame, (int(ponto_x), int(centro_y)), 5, (0, 0, 255), -1)
        cv2.putText(frame, "Rastreando", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Exibindo o tempo atual do vídeo
        tempo_atual = tempos[-1] if len(tempos) > 0 else 0
        cv2.putText(frame, f"Tempo: {tempo_atual:.2f} s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    # Exibir o quadro com o ponto rastreado
    cv2.imshow("Video", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Aplicando a Transformada de Fourier na Amplitude
trajetoria_y_centralizada = np.array(trajetoria_y) - np.mean(trajetoria_y)
fft_result = fft(trajetoria_y_centralizada)
fft_frequencies = fftfreq(len(trajetoria_y_centralizada), 1 / fps)

# Identificando a frequência dominante
positive_freqs = fft_frequencies[fft_frequencies > 1]
positive_magnitude = np.abs(fft_result[fft_frequencies > 1])

dominant_freq = positive_freqs[np.argmax(positive_magnitude)]
print(f"Frequência Dominante: {dominant_freq:.4f} Hz")

# Calculando a frequência teórica
m = 145.2 / 1000  # Convertendo g para kg
k = 13.175        # Constante elástica (N/m)
f_teorica = (1 / (2 * np.pi)) * np.sqrt(k / m)
print(f"Frequência Teórica (Modelo): {f_teorica:.4f} Hz")

# Calculando o Modelo Teórico
A0 = max(trajetoria_y)  # Amplitude inicial (valor máximo da amplitude experimental)
m0 = 145.2             # Massa inicial em kg (145,2 g)
c = 15.83           # Taxa de perda de massa (medido teoricamente pelo diametro)
b = 0.277  * 1000            # Coeficiente de amortecimento (medido teoricamente pelo diametro)

# Calculando o parâmetro beta
beta = (b / (2 * c)) + (1 / 4)
A_teorico = A0 * (1 - (c * np.array(tempos)) / m0) ** beta
A_teorico = np.where(A_teorico < 0, 0, A_teorico)  # Garantindo que a amplitude não seja negativa

# Plotando o Gráfico da Amplitude com o Modelo Teórico
plt.figure(figsize=(12, 6))
plt.plot(tempos, trajetoria_y, label="Amplitude Real (Experimental)", color='b')
#plt.plot(tempos, A_teorico, label="Modelo Teórico", linestyle='--', color='r')
#plt.title("Comparação entre Amplitude Real e Modelo Teórico")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude Y (cm)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# PLOT DAS TRANSFORMADAS DE FOURIER

# Parâmetros
window_duration = 5  # Duração de cada janela em segundos
samples_per_window = int(window_duration * fps)  # Número de amostras por janela

# Preparando as janelas
total_samples = len(trajetoria_y)
num_windows = total_samples // samples_per_window

for i in range(num_windows):
    # Selecionar os dados da janela
    start = i * samples_per_window
    end = start + samples_per_window
    window_y = np.array(trajetoria_y[start:end])
    window_t = np.array(tempos[start:end])

    # Centralizar a janela
    window_y_centralizado = window_y - np.mean(window_y)

    # FFT
    fft_result = fft(window_y_centralizado)
    fft_freqs = fftfreq(len(window_y_centralizado), 1 / fps)
    
    # Filtrar apenas frequências positivas
    mask = fft_freqs > 0
    fft_freqs = fft_freqs[mask]
    fft_magnitude = np.abs(fft_result[mask])

    # Salvar o gráfico da FFT
    plt.figure(figsize=(10, 5))
    plt.plot(fft_freqs, fft_magnitude, color='purple')
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Magnitude")
    plt.title(f"FFT da Janela {i+1} ({window_t[0]:.2f}s a {window_t[-1]:.2f}s)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'fft_janela_{i+1}.pdf')
    plt.close()

print(f"Análise por janelas finalizada: {num_windows} gráficos de FFT salvos em PDF.")