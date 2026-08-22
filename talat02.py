import customtkinter as ctk
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import serial
import threading

# =====================
# CONFIGURACIÓN GENERAL
# =====================
ctk.set_appearance_mode("dark")

app = ctk.CTk()
app.title("T'ALAT")
app.geometry("1000x700")

rosa = "#FF3ECF"
morado = "#8A4DFF"
azul = "#4DA6FF"

# =====================
# MEDIAPIPE
# =====================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
estado_actual = 'REPOSO'
print("Cámara abierta:", cap.isOpened())

# Inicialización segura de Arduino
try:
    arduino = serial.Serial("COM6", 9600, timeout=1)
except Exception as e:
    print(f"No se pudo abrir el puerto COM6: {e}")
    arduino = None


# =====================
# TRANSMISIÓN SEGUNDO PLANO
# =====================
def enviar_arduino(mensaje):
    def hilo_envio():
        if arduino and arduino.is_open:
            try:
                arduino.write(mensaje)
                arduino.flush()
            except Exception as e:
                print("Error de comunicación serial:", e)

    threading.Thread(target=hilo_envio, daemon=True).start()


# =====================
# CAMBIAR PANTALLAS
# =====================
def mostrar(frame):
    for f in (inicio_frame, expresion_frame):
        f.pack_forget()
    frame.pack(fill="both", expand=True)


# =====================
# PANTALLA INICIO
# =====================
inicio_frame = ctk.CTkFrame(app)

titulo = ctk.CTkLabel(inicio_frame, text="T'ALAT 🎹", font=("Arial", 42, "bold"), text_color=rosa)
titulo.pack(pady=40)

subtitulo = ctk.CTkLabel(inicio_frame, text="Música a través de expresiones faciales", font=("Arial", 20))
subtitulo.pack(pady=10)

btn_piano = ctk.CTkButton(inicio_frame, text="🎵 Expresiones", fg_color=azul, command=lambda: mostrar(expresion_frame))
btn_piano.pack(pady=15)

btn_salir = ctk.CTkButton(inicio_frame, text="❌ Salir", fg_color="red", command=app.destroy)
btn_salir.pack(pady=15)

# =====================
# MODO EXPRESIÓN
# =====================
expresion_frame = ctk.CTkFrame(app)

titulo_expresion = ctk.CTkLabel(expresion_frame, text="Modo Piano", font=("Arial", 32, "bold"), text_color=azul)
titulo_expresion.pack(pady=20)

estado_label = ctk.CTkLabel(expresion_frame, text="Estado: Reposo 😐", font=("Arial", 24))
estado_label.pack(pady=10)

acorde_label = ctk.CTkLabel(expresion_frame, text="🎹 ---", font=("Arial", 40, "bold"), text_color=rosa)
acorde_label.pack(pady=10)

camara_frame = ctk.CTkFrame(expresion_frame, width=640, height=480)
camara_frame.pack_propagate(False)
camara_frame.pack(pady=20)

camara_label = ctk.CTkLabel(camara_frame, text="")
camara_label.pack(fill="both", expand=True)


# =====================
# DETECCIÓN FACIAL CON PUNTOS GUÍA AVANZADOS
# =====================
def actualizar_camara():
    global estado_actual
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            h, w, _ = frame.shape

            # Mapeado de puntos principales para los cálculos
            boca_izq = face.landmark[61]
            boca_der = face.landmark[291]
            boca_sup = face.landmark[13]
            boca_inf = face.landmark[14]

            ojo_izq_sup = face.landmark[159]
            ojo_izq_inf = face.landmark[145]
            ojo_der_sup = face.landmark[386]
            ojo_der_inf = face.landmark[374]

            ceja_izq = face.landmark[52]
            ceja_izq_inicio = face.landmark[55]
            ceja_der_inicio = face.landmark[285]

            # Cálculos geométricos del rostro
            ancho_boca = abs(int(boca_der.x * w) - int(boca_izq.x * w))
            alto_boca = abs(int(boca_inf.y * h) - int(boca_sup.y * h))
            apertura_ojo_izq = abs(int(ojo_izq_inf.y * h) - int(ojo_izq_sup.y * h))
            apertura_ojo_der = abs(int(ojo_der_inf.y * h) - int(ojo_der_sup.y * h))
            altura_ceja = abs(int(ojo_izq_sup.y * h) - int(ceja_izq.y * h))
            cercania_cejas = abs(int(ceja_der_inicio.x * w) - int(ceja_izq_inicio.x * w))

            # --------------------------------------------------
            # DIBUJADO DE LISTAS DE PUNTOS GUÍA (CIBER-MÁSCARA)
            # --------------------------------------------------
            puntos_ceja_izq = [70, 63, 105, 66, 107]
            puntos_ceja_der = [336, 296, 334, 293, 300]
            puntos_ojo_izq  = [33, 160, 158, 133, 153, 144]
            puntos_ojo_der  = [263, 387, 385, 362, 380, 373]
            puntos_boca     = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 312, 13, 82, 81, 42]
            puntos_silueta  = [10,  67, 103, 54, 21, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 251, 284, 332, 297, 338]

            # Dibujar Cejas (Color Azul)
            for idx in puntos_ceja_izq + puntos_ceja_der:
                pt = face.landmark[idx]
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (255, 166, 77), -1)

            # Dibujar Ojos (Color Rosa)
            for idx in puntos_ojo_izq + puntos_ojo_der:
                pt = face.landmark[idx]
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (207, 62, 255), -1)

            # Dibujar Contorno de la Boca (Color Morado)
            for idx in puntos_boca:
                pt = face.landmark[idx]
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (255, 77, 138), -1)

            # Dibujar Puntos de la Silueta/Mentón (Color Blanco traslúcido)
            for idx in puntos_silueta:
                pt = face.landmark[idx]
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (240, 240, 240), -1)

            # Destacar los puntos de control clave con un tamaño mayor
            cv2.circle(frame, (int(boca_izq.x * w), int(boca_izq.y * h)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(boca_der.x * w), int(boca_der.y * h)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(ceja_izq_inicio.x * w), int(ceja_izq_inicio.y * h)), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(ceja_der_inicio.x * w), int(ceja_der_inicio.y * h)), 5, (0, 0, 255), -1)

            # --------------------------------------------------
            # LÓGICA DE DECISIÓN EN CASCADA (MODO NOTAS)
            # --------------------------------------------------
            nuevo_estado = "REPOSO"
            texto_estado = "Estado: Reposo 😐"
            texto_piano = "🎹 ---"
            caracter_serial = "0"

            if apertura_ojo_izq < 4.5 and apertura_ojo_der < 4.5:
                nuevo_estado = "ABURRIMIENTO"
                texto_estado = "Estado: Aburrimiento 😑"
                texto_piano = "🎹 DO (Aburrimiento)"
                caracter_serial = "C"
            elif alto_boca > 35 and altura_ceja > 40:
                nuevo_estado = "SORPRESA"
                texto_estado = "Estado: Sorpresa 😮"
                texto_piano = "🎹 RE (Sorpresa)"
                caracter_serial = "D"
            elif cercania_cejas < 18:
                nuevo_estado = "IRA"
                texto_estado = "Estado: Ira 😠"
                texto_piano = "🎹 MI (Ira)"
                caracter_serial = "E"
            elif boca_izq.y > boca_inf.y and boca_der.y > boca_inf.y:
                nuevo_estado = "TRISTEZA"
                texto_estado = "Estado: Tristeza 🙁"
                texto_piano = "🎹 FA (Tristeza)"
                caracter_serial = "F"
            elif ancho_boca > 90:
                nuevo_estado = "ALEGRIA"
                texto_estado = "Estado: Alegría 😄"
                texto_piano = "🎹 SOL (Alegría)"
                caracter_serial = "G"

            # Enviar el carácter de la nota si cambia el estado
            if nuevo_estado != estado_actual:
                enviar_arduino(f"{caracter_serial}\n".encode())
                estado_actual = nuevo_estado

            estado_label.configure(text=texto_estado)
            acorde_label.configure(text=texto_piano)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(img)

        if expresion_frame.winfo_ismapped():
            camara_label.configure(image=imgtk)
            camara_label.image = imgtk

    app.after(20, actualizar_camara)


# =====================
# CERRAR PROGRAMA
# =====================
def cerrar():
    cap.release()
    face_mesh.close()
    if arduino and arduino.is_open:
        arduino.close()
    app.destroy


app.protocol("WM_DELETE_WINDOW", cerrar)

# =====================
# INICIAR APP
# =====================
mostrar(inicio_frame)
actualizar_camara()
app.mainloop()
