import cv2

cap = cv2.VideoCapture("q1B.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    formas = detectar_formas(frame)
    
    maior_massa = detectar_maior_massa(formas)
    if maior_massa:
        x, y, w, h, _, _ = maior_massa
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if detectar_colisao(formas):
        cv2.putText(frame, "COLISÃO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def detectar_formas(frame):
    """Detecta formas geométricas por cor e retorna as informações."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    limites = {
        "azul": ((100, 150, 50), (140, 255, 255)),
        "vermelho": ((0, 120, 70), (10, 255, 255)),
    }
    
    formas_detectadas = []
    for cor, (lower, upper) in limites.items():
        mascara = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contornos:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                formas_detectadas.append((x, y, w, h, cor, cv2.contourArea(cnt)))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                
    return formas_detectadas

def detectar_colisao(formas):
    """Verifica se há colisão entre formas."""
    for i in range(len(formas)):
        for j in range(i + 1, len(formas)):
            x1, y1, w1, h1, _, _ = formas[i]
            x2, y2, w2, h2, _, _ = formas[j]
            
            if (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2):
                return True
    return False

def detectar_maior_massa(formas):
    """Forma de maior massa."""
    if not formas:
        return None
    return max(formas, key=lambda f: f[-1])


    # Exibe resultado
    cv2.imshow("Feed", frame)

    # Wait for key 'ESC' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# That's how you exit
cap.release()
cv2.destroyAllWindows()