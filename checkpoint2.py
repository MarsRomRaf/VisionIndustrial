import numpy as np
import cv2

# Mantener el background frame para quitarlo posteriormente
background = None
# Guarda los datos de la mano para que todos sus detalles estén en un solo lugar.
hand = None
# Variables para contar cuántos fotogramas han pasado y para establecer el tamaño de la ventana.
frames_elapsed = 0
FRAME_HEIGHT = 300
FRAME_WIDTH = 400
# Prueba a editarlas si tu programa tiene problemas para reconocer tu tono de piel.
CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18

class HandData:
    top = (0, 0)
    bottom = (0, 0)
    left = (0, 0)
    right = (0, 0)
    centerX = 0
    prevCenterX = 0
    isInFrame = False
    isWaving = False
    fingers = None

    def __init__(self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX
        self.prevCenterX = 0
        self.isInFrame = False
        self.isWaving = False

    def update(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
    
    def check_for_waving(self, centerX):
        self.prevCenterX = self.centerX
        self.centerX = centerX
        if abs(self.centerX - self.prevCenterX > 3):
            self.isWaving = True
        else:
            self.isWaving = False

#Función para buscar las extremidades de la mano y colócalas en el objeto global mano
def get_hand_data(thresholded_image, segmented_image):
    global hand
    # Encierra el área alrededor de las extremidades en un "convex hull" para conectar todos los afloramientos.
    convexHull = cv2.convexHull(segmented_image)
    # Encuentra las extremidades del "convex hull" y guárdalas como puntos.
    top = tuple(convexHull[convexHull[:, :, 1].argmin()][0])
    bottom = tuple(convexHull[convexHull[:, :, 1].argmax()][0])
    left = tuple(convexHull[convexHull[:, :, 0].argmin()][0])
    right = tuple(convexHull[convexHull[:, :, 0].argmax()][0])
    # Obtiene el centro de la palma, así podremos comprobar si ondea y encontrar los dedos.
    centerX = int((left[0] + right[0]) / 2)
    # Ponemos toda la información en un objeto para extraerla facilmente
    if hand == None:
        hand = HandData(top, bottom, left, right, centerX)
    else:
        hand.update(top, bottom, left, right)
    # Sólo comprueba la ondulación cada 6 fotogramas.
    if frames_elapsed % 6 == 0:
        hand.check_for_waving(centerX)

# Función para escribir en el frame
def write_on_image(frame, frames_elapsed, hand):
    # Determina el texto a mostrar según el estado de la calibración y de la mano
    text = "Buscando..."
    if frames_elapsed < CALIBRATION_TIME:
        text = "Calibrando..."
    elif hand is None or hand.isInFrame is False:
        text = "Mano no detectada"
    else:
        if hand.isWaving:
            text = "Moviendo"
        elif hand.fingers == 0:
            text = "Paro"
        elif hand.fingers == 1:
            text = "Encendido"
    
    # Definir la posición del texto en el centro inferior de toda la imagen
    text_position = (frame.shape[1] // 2 - 50, frame.shape[0] - 10)  # Centrado y 10 píxeles desde el borde inferior

    # Escribir el texto en la posición calculada
    cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

def get_region(frame):
    #Separa la región de interés del resto del fotograma
    region=frame[region_top:region_bottom, region_left:region_right]
    #Transforma a escala de grises para que podamos detectar los bordes más fácilmente. 
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    #Utiliza un desenfoque gaussiano para evitar que el ruido del fotograma se etiquete como borde. 
    region = cv2.GaussianBlur(region,(5,5),0)

    return region

def get_average(region):
    #Tenemos que utilizar la palabra clave global porque queremos editar la variable global. 
    global background
    #Si aún no hemos capturado el fondo, haz que la región actual sea el fondo. 
    if background is None:
        background = region.copy().astype("float")
        return
    #De lo contrario, añade este fotograma capturado a la media de los fondos
    cv2.accumulateWeighted(region,background,BG_WEIGHT)

#Aquí utilizamos la diferenciación para separar el fondo del objeto de interés. 
def segment(region):
    global hand
    #Encuentra la diferencia absoluta entre el fondo y el fotograma actual. 
    diff = cv2.absdiff(background.astype(np.uint8),region)

    #Umbral de esa región con un estricto 0 o 1, por lo que sólo el primer plano se mantiene.
    thresholded_region = cv2.threshold(diff, OBJ_THRESHOLD,255, cv2.THRESH_BINARY)[1]

    # Obtener los contornos de la región, que devolverá un contorno de la mano.
    (contours, _) = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Si no conseguimos nada, no hay mano
    if len(contours) == 0:
        if hand is not None:
            hand.isInFrame = False
        return
    # En caso contrario, devuelve una tupla de la mano rellena (thresholded_region), junto con el contorno (segmented_region).
    else:
        if hand is not None:
            hand.isInFrame = True
        segmented_region = max(contours, key = cv2.contourArea)
        return (thresholded_region, segmented_region)

# Nuestra región de interés será la parte superior derecha del frame
region_top = 0              
region_bottom = int(2 * FRAME_HEIGHT / 3)     
region_left = int(FRAME_WIDTH / 2)             
region_right = FRAME_WIDTH       
frames_elapsed = 0

capture = cv2.VideoCapture(0)

while True:
    # Guarda el fotograma de la captura de vídeo y redimensiona al tamaño de la ventana.
    ret, frame = capture.read()
    frame=cv2.resize(frame,(FRAME_WIDTH,FRAME_HEIGHT))
    frame = cv2.flip(frame, 1)

    region = get_region(frame)
   
    if frames_elapsed < CALIBRATION_TIME:
        get_average(region)
    else:
        region_2 = segment(region)
        if region_2 is not None:
                (thresholded_region, segmented_region) = region_2
                cv2.drawContours(frame, [segmented_region], -1, (0, 255, 0))
                cv2.imshow("Segmented Region", thresholded_region)

                get_hand_data(thresholded_region, segmented_region)

    # Llama a la función para escribir en la imagen
    write_on_image(frame, frames_elapsed, hand)
    
    # Resalta la región de interés con un recuadro
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255, 255, 255), 2)
    
    # Muestra el fotograma capturado anteriormente.
    cv2.imshow("Camera Input", frame)
    frames_elapsed += 1
    
    # Comprueba si el usuario desea tomar una foto de la región de interés
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Recorta la región de interés
        roi = frame[region_top:region_bottom, region_left:region_right]
        print(f"Foto tomada")
    
    # Comprueba si el usuario desea salir.
    if key == ord('x'):
        break

# Cuando salimos del bucle, también detenemos la captura.
capture.release()
cv2.destroyAllWindows()