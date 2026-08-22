import customtkinter as ctk
from PIL import Image
import cv2
import mediapipe as mp
import threading
from PIL import ImageTk
import json
import os
from math import dist
from datetime import datetime

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


app = ctk.CTk()

app.title("T'ALAT")

app.geometry("1280x720")

app.configure(
    fg_color="#000000"
)


# ==========================================
# COLORES T'ALAT
# ==========================================

NEGRO = "#000000"
BLANCO = "#FFFFFF"
AZUL = "#3B9DFF"
MORADO = "#9B4DFF"
GRIS = "#BFBFBF"

# ==========================================
# MEDIAPIPE
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

camara = None
camara_activa = False
frame_camara = None

# ==========================================
# MINI BASE DE DATOS T'ALAT
# ==========================================

ARCHIVO_USUARIOS = "usuarios.json"

usuario_actual = None
hora_inicio_sesion = None
estadisticas_sesion = {
    "expresiones": 0,
    "notas": 0
}
ultima_expresion_registrada = "REPOSO"
registros_expresiones_sesion = []
contador_frames = 0

def cargar_usuarios():
    if not os.path.exists(ARCHIVO_USUARIOS):
        return {}

    try:
        with open(ARCHIVO_USUARIOS, "r", encoding="utf-8") as archivo:
            datos = json.load(archivo)
            return datos if isinstance(datos, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}

def guardar_usuarios(datos):
    try:
        with open(ARCHIVO_USUARIOS, "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo, ensure_ascii=False, indent=4)
    except OSError as e:
        print("No se pudieron guardar los usuarios:", e)

usuarios_db = cargar_usuarios()

for _nombre, _datos in usuarios_db.items():
    _datos.setdefault("sesiones", 0)
    _datos.setdefault("tiempo_total", 0)
    _datos.setdefault("expresiones", 0)
    _datos.setdefault("notas", 0)
    _datos.setdefault("canciones_completadas", 0)
    _datos.setdefault("ultima_sesion", "")
    _datos.setdefault("comentarios", "")
    _datos.setdefault("historial_sesiones", [])

def crear_datos_usuario():
    return {
        "sesiones": 0,
        "tiempo_total": 0,
        "expresiones": 0,
        "notas": 0,
        "canciones_completadas": 0,
        "ultima_sesion": "",
        "comentarios": "",
        "historial_sesiones": []
    }



# ==========================================
# DETECTOR DE GESTOS T'ALAT
# ==========================================

class DetectorGestos:
    """
    Detector geométrico robusto para T'ALAT.
    La calibración solo define el rostro neutro de la sesión actual.
    No aprende ni guarda una expresión personal.
    """

    def __init__(self):
        self.emocion = "REPOSO"
        self.emoji = "😐"
        self.nota = "--"
        self.color = BLANCO
        self.mensaje = "Haz una expresión"

        self.calibrado = False
        self.calibrando = False
        self.muestras = []
        self.total_muestras_calibracion = 45
        self.base = None

        self.candidato = "REPOSO"
        self.contador_candidato = 0
        self.ultima_estable = "REPOSO"
        self.ultima_medicion = None

    def iniciar_calibracion(self):
        self.calibrado = False
        self.calibrando = True
        self.muestras = []
        self.base = None
        self.ultima_medicion = None
        self.candidato = "REPOSO"
        self.contador_candidato = 0
        self.ultima_estable = "REPOSO"

        self.emocion = "CALIBRANDO"
        self.emoji = "😐"
        self.nota = "--"
        self.color = AZUL
        self.mensaje = "Mantén tu rostro relajado"

    @staticmethod
    def _p(lm, i):
        return (lm[i].x, lm[i].y)

    @staticmethod
    def _dist(a, b):
        return dist(a, b)

    def _caracteristicas(self, rostro):
        lm = rostro.landmark

        ancho_rostro = self._dist(self._p(lm, 33), self._p(lm, 263))
        alto_rostro = self._dist(self._p(lm, 10), self._p(lm, 152))

        if ancho_rostro < 1e-6 or alto_rostro < 1e-6:
            return None

        # Boca
        boca_izq = self._p(lm, 61)
        boca_der = self._p(lm, 291)
        boca_sup = self._p(lm, 13)
        boca_inf = self._p(lm, 14)
        comisura_izq = self._p(lm, 78)
        comisura_der = self._p(lm, 308)
        labio_sup_centro = self._p(lm, 0)
        labio_inf_centro = self._p(lm, 17)
        labio_sup_izq = self._p(lm, 78)
        labio_sup_der = self._p(lm, 308)
        labio_inf_izq = self._p(lm, 95)
        labio_inf_der = self._p(lm, 324)

        ancho_boca = self._dist(boca_izq, boca_der) / ancho_rostro
        apertura_boca = self._dist(boca_sup, boca_inf) / alto_rostro

        centro_comisuras_y = (comisura_izq[1] + comisura_der[1]) / 2
        centro_boca_y = (boca_sup[1] + boca_inf[1]) / 2
        curva_boca = (centro_comisuras_y - centro_boca_y) / alto_rostro

        inclinacion_boca = (
            comisura_der[1] - comisura_izq[1]
        ) / ancho_rostro

        distancia_nariz_boca = self._dist(
            self._p(lm, 1), self._p(lm, 13)
        ) / alto_rostro

        distancia_boca_menton = self._dist(
            self._p(lm, 14), self._p(lm, 152)
        ) / alto_rostro

        altura_labios = self._dist(
            labio_sup_centro,
            labio_inf_centro
        ) / alto_rostro

        ancho_labios_sup = self._dist(
            labio_sup_izq,
            labio_sup_der
        ) / ancho_rostro

        ancho_labios_inf = self._dist(
            labio_inf_izq,
            labio_inf_der
        ) / ancho_rostro

        ancho_labios_promedio = (
            ancho_labios_sup + ancho_labios_inf
        ) / 2

        # Ojos
        ojo_izq_sup = self._p(lm, 159)
        ojo_izq_inf = self._p(lm, 145)
        ojo_der_sup = self._p(lm, 386)
        ojo_der_inf = self._p(lm, 374)

        apertura_ojo_izq = self._dist(
            ojo_izq_sup, ojo_izq_inf
        ) / alto_rostro

        apertura_ojo_der = self._dist(
            ojo_der_sup, ojo_der_inf
        ) / alto_rostro

        apertura_ojos = (apertura_ojo_izq + apertura_ojo_der) / 2

        diferencia_ojos = abs(
            apertura_ojo_izq - apertura_ojo_der
        )

        # Cejas
        altura_ceja_izq = self._dist(
            self._p(lm, 159), self._p(lm, 52)
        ) / alto_rostro

        altura_ceja_der = self._dist(
            self._p(lm, 386), self._p(lm, 285)
        ) / alto_rostro

        altura_ceja = (altura_ceja_izq + altura_ceja_der) / 2

        cercania_cejas = self._dist(
            self._p(lm, 55), self._p(lm, 285)
        ) / ancho_rostro

        # Contorno
        ancho_medio_cara = self._dist(
            self._p(lm, 234), self._p(lm, 454)
        ) / ancho_rostro

        return {
            "ancho_boca": ancho_boca,
            "apertura_boca": apertura_boca,
            "curva_boca": curva_boca,
            "inclinacion_boca": inclinacion_boca,
            "distancia_nariz_boca": distancia_nariz_boca,
            "distancia_boca_menton": distancia_boca_menton,
            "apertura_ojo_izq": apertura_ojo_izq,
            "apertura_ojo_der": apertura_ojo_der,
            "apertura_ojos": apertura_ojos,
            "diferencia_ojos": diferencia_ojos,
            "altura_ceja_izq": altura_ceja_izq,
            "altura_ceja_der": altura_ceja_der,
            "altura_ceja": altura_ceja,
            "cercania_cejas": cercania_cejas,
            "ancho_medio_cara": ancho_medio_cara,
            "altura_labios": altura_labios,
            "ancho_labios_sup": ancho_labios_sup,
            "ancho_labios_inf": ancho_labios_inf,
            "ancho_labios_promedio": ancho_labios_promedio
        }

    def agregar_muestra(self, rostro):
        caracteristicas = self._caracteristicas(rostro)
        if caracteristicas is None:
            return False

        self.muestras.append(caracteristicas)

        if len(self.muestras) >= self.total_muestras_calibracion:
            self._finalizar_calibracion()

        return True

    def _finalizar_calibracion(self):
        claves = self.muestras[0].keys()

        self.base = {
            clave: sum(m[clave] for m in self.muestras) / len(self.muestras)
            for clave in claves
        }

        self.calibrado = True
        self.calibrando = False

        self.emocion = "REPOSO"
        self.emoji = "😐"
        self.nota = "--"
        self.color = BLANCO
        self.mensaje = "Calibración completada"

    def _delta(self, f, clave):
        return f[clave] - self.base[clave]

    def _clasificar(self, f):
        # Diferencias respecto al rostro neutro de esta sesión.
        da_boca = self._delta(f, "ancho_boca")
        dap_boca = self._delta(f, "apertura_boca")
        dcurva = self._delta(f, "curva_boca")
        dincl = self._delta(f, "inclinacion_boca")
        dnose = self._delta(f, "distancia_nariz_boca")
        dmenton = self._delta(f, "distancia_boca_menton")

        d_altura_labios = self._delta(f, "altura_labios")
        d_ancho_labios = self._delta(f, "ancho_labios_promedio")
        d_ojos = self._delta(f, "apertura_ojos")
        d_cejas = self._delta(f, "altura_ceja")

        dojos = self._delta(f, "apertura_ojos")
        dojo_i = self._delta(f, "apertura_ojo_izq")
        dojo_d = self._delta(f, "apertura_ojo_der")

        dceja = self._delta(f, "altura_ceja")
        dceja_i = self._delta(f, "altura_ceja_izq")
        dceja_d = self._delta(f, "altura_ceja_der")
        dcejas = self._delta(f, "cercania_cejas")

        scores = {
            "ALEGRÍA": 0.0,
            "SORPRESA": 0.0,
            "TRISTEZA": 0.0,
            "IRA": 0.0,
            "ABURRIMIENTO": 0.0
        }

        # ALEGRÍA
        if da_boca > 0.018:
            scores["ALEGRÍA"] += 1.2
        if dcurva < -0.004:
            scores["ALEGRÍA"] += 1.4
        if dap_boca > 0.003:
            scores["ALEGRÍA"] += 0.7
        if dnose < -0.002:
            scores["ALEGRÍA"] += 0.4
        if dojos > 0.002:
            scores["ALEGRÍA"] += 0.3

        # SORPRESA
        if d_altura_labios > LIMITE:
            scores["SORPRESA"] += 1.5
        if d_ancho_labios > LIMITE:
            scores["SORPRESA"] += 1.0
        if d_ojo_izq > LIMITE:
            scores["SORPRESA"] += 0.8
        if d_ojo_der > LIMITE:
            scores["SORPRESA"] += 0.8
        if d_ceja_izq > LIMITE:
            scores["SORPRESA"] += 0.7
        if d_ceja_der > LIMITE:
            scores["SORPRESA"] += 0.7

        # TRISTEZA
        if dcurva > 0.006:
            scores["TRISTEZA"] += 1.3
        if da_boca < 0.008:
            scores["TRISTEZA"] += 0.7
        if dojos < -0.004:
            scores["TRISTEZA"] += 0.7
        if dmenton < 0.004:
            scores["TRISTEZA"] += 0.4
        if dincl > 0.004:
            scores["TRISTEZA"] += 0.6

        # IRA
        if dcejas < -0.012:
            scores["IRA"] += 1.4
        if dceja < -0.004:
            scores["IRA"] += 1.1
        if dcurva > 0.003:
            scores["IRA"] += 0.5
        if dap_boca > 0.004:
            scores["IRA"] += 0.4
        if dceja_i < -0.006 and dceja_d < -0.006:
            scores["IRA"] += 0.6

        # ABURRIMIENTO
        if dojos < -0.012:
            scores["ABURRIMIENTO"] += 1.5
        if dojo_i < -0.008 and dojo_d < -0.008:
            scores["ABURRIMIENTO"] += 0.8
        if abs(da_boca) < 0.015:
            scores["ABURRIMIENTO"] += 0.4
        if abs(dap_boca) < 0.010:
            scores["ABURRIMIENTO"] += 0.4

        emocion, puntuacion = max(scores.items(), key=lambda item: item[1])

        if puntuacion < 2.0:
            return "REPOSO"

        ordenadas = sorted(scores.values(), reverse=True)
        if len(ordenadas) > 1 and ordenadas[0] - ordenadas[1] < 0.25:
            return "REPOSO"

        return emocion

    def _estabilizar(self, emocion):
        if emocion == self.candidato:
            self.contador_candidato += 1
        else:
            self.candidato = emocion
            self.contador_candidato = 1

        minimo_frames = 3 if emocion == "REPOSO" else 5

        if self.contador_candidato >= minimo_frames:
            self.ultima_estable = emocion

        return self.ultima_estable

    def detectar(self, rostro, ancho, alto):
        if self.calibrando:
            self.agregar_muestra(rostro)
            return

        if not self.calibrado:
            return

        f = self._caracteristicas(rostro)
        if f is None:
            return

        self.ultima_medicion = f

        emocion = self._estabilizar(self._clasificar(f))

        datos = {
            "REPOSO": ("😐", "--", BLANCO, "Haz una expresión"),
            "ABURRIMIENTO": ("😑", "DO", GRIS, "Mantén la expresión"),
            "SORPRESA": ("😮", "RE", AZUL, "¡Muy bien!"),
            "IRA": ("😠", "MI", "#FF4444", "Excelente"),
            "TRISTEZA": ("🙁", "FA", "#3B6BFF", "Muy bien"),
            "ALEGRÍA": ("😄", "SOL", "#FFD93D", "¡Excelente!")
        }

        self.emocion = emocion
        self.emoji, self.nota, self.color, self.mensaje = datos[emocion]


detector = DetectorGestos()

# ==========================================
# SISTEMA DE CÁMARA T'ALAT
# ==========================================

def iniciar_camara():
    global camara
    global camara_activa
    global hora_inicio_sesion
    global estadisticas_sesion
    global ultima_expresion_registrada
    global registros_expresiones_sesion
    global contador_frames

    if camara_activa:
        return

    camara = cv2.VideoCapture(0)

    # Resolución suficiente para detectar rostro y más ligera para la Raspberry.
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camara.set(cv2.CAP_PROP_FPS, 30)

    if not camara.isOpened():
        print("No se pudo abrir la cámara.")
        camara = None
        return

    camara_activa = True

    hora_inicio_sesion = datetime.now()

    estadisticas_sesion = {
        "expresiones": 0,
        "notas": 0
    }

    ultima_expresion_registrada = "REPOSO"
    registros_expresiones_sesion = []
    contador_frames = 0

    # La calibración solo sirve para ESTA sesión.
    # No se guarda en usuarios.json.
    detector.iniciar_calibracion()

    if "instruccion_label" in globals():
        instruccion_label.configure(
            text="INSTRUCCIÓN: Mantén tu rostro relajado mientras T'ALAT calibra tu rostro."
        )

    if "mensaje_label" in globals():
        mensaje_label.configure(
            text="Mantén tu rostro relajado..."
        )

    actualizar_camara()


def guardar_sesion():
    global hora_inicio_sesion
    global estadisticas_sesion
    global usuario_actual
    global registros_expresiones_sesion

    if usuario_actual is None or hora_inicio_sesion is None:
        return

    ahora = datetime.now()
    duracion = max(
        0,
        int((ahora - hora_inicio_sesion).total_seconds())
    )

    usuario = usuarios_db.setdefault(
        usuario_actual,
        crear_datos_usuario()
    )

    # Historial de esta sesión.
    sesion = {
        "fecha": ahora.strftime("%Y-%m-%d %H:%M"),
        "duracion_segundos": duracion,
        "expresiones": list(registros_expresiones_sesion)
    }

    usuario["sesiones"] += 1
    usuario["tiempo_total"] += duracion
    usuario["expresiones"] += estadisticas_sesion["expresiones"]
    usuario["notas"] += estadisticas_sesion["notas"]
    usuario["ultima_sesion"] = ahora.strftime("%Y-%m-%d %H:%M")

    usuario.setdefault("historial_sesiones", [])
    usuario["historial_sesiones"].append(sesion)

    # No guardamos la calibración.
    # Solo guardamos las mediciones realizadas durante la sesión.

    if "caja_comentarios" in globals():
        comentario = caja_comentarios.get("1.0", "end").strip()
        if comentario:
            usuario["comentarios"] = comentario

    guardar_usuarios(usuarios_db)

    if "actualizar_estadisticas_perfil" in globals():
        actualizar_estadisticas_perfil()

    if "actualizar_comentarios_perfil" in globals():
        actualizar_comentarios_perfil()

    hora_inicio_sesion = None


def detener_camara():
    global camara
    global camara_activa

    if not camara_activa and camara is None:
        return

    guardar_sesion()

    camara_activa = False

    if camara is not None:
        camara.release()
        camara = None

    if "camara_label" in globals():
        camara_label.configure(
            image=None,
            text="📷 Cámara T'ALAT"
        )
        camara_label.image = None


def actualizar_camara():
    global camara
    global camara_activa
    global estadisticas_sesion
    global ultima_expresion_registrada
    global registros_expresiones_sesion
    global contador_frames

    if not camara_activa:
        return

    if camara is None:
        return

    ret, frame = camara.read()

    if not ret or frame is None:
        app.after(40, actualizar_camara)
        return

    frame = cv2.flip(frame, 1)

    # Reducimos el tamaño que entra a MediaPipe para acelerar el procesamiento.
    frame_procesamiento = cv2.resize(frame, (640, 480))

    rgb = cv2.cvtColor(
        frame_procesamiento,
        cv2.COLOR_BGR2RGB
    )

    contador_frames += 1

    # Procesamos la mayoría de los frames para mantener sensibilidad sin saturar la CPU.
    procesar_rostro = (contador_frames % 4 != 0)

    if procesar_rostro:
        resultados = face_mesh.process(rgb)

        if resultados.multi_face_landmarks:

            rostro = resultados.multi_face_landmarks[0]

            alto, ancho, _ = frame_procesamiento.shape

            detector.detectar(
                rostro,
                ancho,
                alto
            )

            if detector.calibrando:

                progreso = len(detector.muestras)

                emoji_label.configure(text="😐")

                estado_expresion.configure(
                    text="CALIBRANDO",
                    text_color=AZUL
                )

                estado_nota.configure(
                    text=f"📐 {progreso}/{detector.total_muestras_calibracion}",
                    text_color=AZUL
                )

                mensaje_label.configure(
                    text="Mantén tu rostro relajado",
                    text_color=AZUL
                )

                instruccion_label.configure(
                    text="INSTRUCCIÓN: No sonrías ni hagas gestos. Mira al frente y mantén el rostro relajado."
                )

            elif detector.calibrado:

                emoji_label.configure(text=detector.emoji)

                estado_expresion.configure(
                    text=detector.emocion,
                    text_color=detector.color
                )

                estado_nota.configure(
                    text=f"🎹 {detector.nota}",
                    text_color=detector.color
                )

                mensaje_label.configure(
                    text=detector.mensaje,
                    text_color=detector.color
                )

                if detector.emocion == "REPOSO":
                    instruccion_label.configure(
                        text="INSTRUCCIÓN: Haz una expresión clara y mantenla unos instantes."
                    )
                else:
                    instruccion_label.configure(
                        text=f"INSTRUCCIÓN: Expresión detectada: {detector.emocion}. Manténla unos instantes."
                    )

                info_sesion.configure(
                    border_width=4,
                    border_color=detector.color
                )

                # Registramos SOLO el momento en que una expresión cambia.
                # La calibración NO entra en este historial.
                if (
                    detector.ultima_estable != "REPOSO"
                    and detector.ultima_estable != ultima_expresion_registrada
                    and detector.ultima_medicion is not None
                ):
                    medicion = detector.ultima_medicion
                    apertura = medicion["apertura_boca"]

                    anterior = None

                    # Buscamos la medición anterior de ESTA MISMA expresión.
                    for registro in reversed(registros_expresiones_sesion):
                        if registro["emocion"] == detector.ultima_estable:
                            anterior = registro
                            break

                    if anterior is None:
                        comparacion = "Primera medición de esta expresión"
                    else:
                        diferencia = apertura - anterior["apertura_boca"]
                        if diferencia > 0.002:
                            comparacion = f"Se abrió más que la pasada (+{diferencia:.4f})"
                        elif diferencia < -0.002:
                            comparacion = f"Se abrió menos que la pasada ({diferencia:.4f})"
                        else:
                            comparacion = "Apertura muy similar a la pasada"

                    registros_expresiones_sesion.append({
                        "emocion": detector.ultima_estable,
                        "emoji": detector.emoji,
                        "apertura_boca": round(medicion["apertura_boca"], 5),
                        "ancho_boca": round(medicion["ancho_boca"], 5),
                        "apertura_ojos": round(medicion["apertura_ojos"], 5),
                        "altura_ceja": round(medicion["altura_ceja"], 5),
                        "cercania_cejas": round(medicion["cercania_cejas"], 5),
                        "curva_boca": round(medicion["curva_boca"], 5),
                        "comparacion": comparacion,
                        "momento": datetime.now().strftime("%H:%M:%S")
                    })

                    estadisticas_sesion["expresiones"] += 1

                    if detector.nota != "--":
                        estadisticas_sesion["notas"] += 1

                    ultima_expresion_registrada = detector.ultima_estable

                elif detector.ultima_estable == "REPOSO":
                    ultima_expresion_registrada = "REPOSO"

    # Mostrar la imagen reducida; 640x480 es mucho más ligero que 1000x750.
    imagen = Image.fromarray(rgb)
    imagen = imagen.resize((640, 360))

    foto = ImageTk.PhotoImage(imagen)

    camara_label.configure(
        image=foto,
        text=""
    )

    camara_label.image = foto

    app.after(
        33,
        actualizar_camara
    )

# ==========================================
# CAMBIO DE PANTALLAS
# ==========================================

def mostrar_bienvenido():
    inicio_frame.pack_forget()

    if not bienvenido_frame.winfo_ismapped():
        bienvenido_frame.pack(fill="both", expand=True)

def mostrar_inicio():
    bienvenido_frame.pack_forget()

    if not inicio_frame.winfo_ismapped():
        inicio_frame.pack(fill="both", expand=True)

def mostrar_perfil():
    bienvenido_frame.pack_forget()
    inicio_frame.pack_forget()
    perfil_frame.pack(fill="both", expand=True)

def iniciar_sesion():
    perfil_frame.pack_forget()
    sesion_frame.pack(
        fill="both",
        expand=True
    )

    iniciar_camara()

def regresar_perfil():

    detener_camara()

    sesion_frame.pack_forget()
    perfil_frame.pack(
        fill="both",
        expand=True
    )


def terminar_sesion():
    if not camara_activa and hora_inicio_sesion is None:
        regresar_perfil()
        return

    detener_camara()
    sesion_frame.pack_forget()
    perfil_frame.pack(
        fill="both",
        expand=True
    )

    actualizar_estadisticas_perfil()
    actualizar_comentarios_perfil()

# ==========================================
# CONTENEDOR PRINCIPAL
# ==========================================

bienvenido_frame = ctk.CTkFrame(
    app,
    fg_color=NEGRO
)

bienvenido_frame.pack(
    fill="both",
    expand=True
)

# ==========================================
# CABECERA
# ==========================================

header = ctk.CTkFrame(
    bienvenido_frame,
    fg_color=NEGRO,
    height=60
)

header.pack(
    fill="x",
    padx=40,
    pady=20
)


titulo_header = ctk.CTkLabel(
    header,
    text="T'ALAT",
    font=("Montserrat",35,"bold"),
    text_color=BLANCO
)

titulo_header.pack(
    side="left"
)



# ==========================================
# ZONA PRINCIPAL
# ==========================================

content = ctk.CTkFrame(
    bienvenido_frame,
    fg_color=NEGRO
)

content.pack(
    fill="both",
    expand=True,
    padx=60,
    pady=20
)



# ==========================================
# ZONA IZQUIERDA (LOGO)
# ==========================================

left_frame = ctk.CTkFrame(
    content,
    fg_color=NEGRO
)

left_frame.pack(

    side="left",
    expand=True
)



try:

    logo_img = ctk.CTkImage(
        light_image=Image.open("IMG_3131.jpeg"),
        dark_image=Image.open("IMG_3131.jpeg"),
        size=(500,430)
    )


    logo_label = ctk.CTkLabel(
        left_frame,
        image=logo_img,
        text=""
    )

    logo_label.pack(
        pady=20
    )


except Exception as e:

    print(
        "Error cargando logo:",
        e
    )


    logo_label = ctk.CTkLabel(
        left_frame,
        text="T'ALAT",
        font=("Montserrat",80,"bold"),
        text_color=BLANCO
    )

    logo_label.pack()



slogan = ctk.CTkLabel(
    left_frame,
    text="La música al alcance de todos",
    font=("Montserrat",22),
    text_color=GRIS
)

slogan.pack()

instruccion_bienvenida = ctk.CTkLabel(
    left_frame,
    text="Primero crea o selecciona un usuario.\nDespués inicia una sesión y mantén tu rostro relajado durante la calibración.",
    font=("Montserrat",16),
    text_color=GRIS,
    justify="center"
)

instruccion_bienvenida.pack(
    pady=(15,0)
)

# ==========================================
# ZONA DERECHA (BOTONES)
# ==========================================


right_frame = ctk.CTkFrame(
    content,
    fg_color=NEGRO
)


right_frame.pack(
    side="right",
    expand=True
)



bienvenida = ctk.CTkLabel(
    right_frame,
    text="Bienvenido",
    font=("Montserrat",45,"bold"),
    text_color=BLANCO
)


bienvenida.pack(
    pady=(20,10)
)


# ==========================================
# BOTONES
# ==========================================

btn_reporte = ctk.CTkButton(
    right_frame,
    text="🎹 INICIO",
    width=420,
    height=75,
    corner_radius=35,
    fg_color=AZUL,
    hover_color=MORADO,
    text_color=BLANCO,
    font=("Montserrat",24,"bold"),
    command=mostrar_inicio
)


btn_reporte.pack(
    pady=15
)



btn_config = ctk.CTkButton(
    right_frame,
    text="⚙ idioma",
    width=420,
    height=75,
    corner_radius=35,
    fg_color="#111111",
    border_width=2,
    border_color=MORADO,
    hover_color="#222222",
    text_color=BLANCO,
    font=("Montserrat",24,"bold")
)


btn_config.pack(
    pady=15
)



# ==========================================
# PIE
# ==========================================


footer = ctk.CTkLabel(
    bienvenido_frame,
    text="Sistema Musical T'ALAT",
    font=("Montserrat",14),
    text_color="#555555"
)


footer.pack(
    side="bottom",
    pady=10
)

# ==========================================
# PANTALLA DE INICIO (USUARIOS)
# ==========================================

inicio_frame = ctk.CTkFrame(
    app,
    fg_color=NEGRO
)

inicio_frame.pack_forget()

# ==========================================
# CABECERA
# ==========================================

header_inicio = ctk.CTkFrame(
    inicio_frame,
    fg_color=NEGRO,
    height=70
)

header_inicio.pack(
    fill="x",
    padx=30,
    pady=20
)

# Botón regresar
btn_regresar = ctk.CTkButton(
    header_inicio,
    text="← Bienvenido",
    width=140,
    height=40,
    fg_color="#222222",
    hover_color="#333333",
    command=mostrar_bienvenido
)

btn_regresar.pack(side="left")

# Título
titulo_inicio = ctk.CTkLabel(
    header_inicio,
    text="Usuarios",
    font=("Montserrat", 34, "bold"),
    text_color=BLANCO
)

titulo_inicio.pack(side="left", padx=30)

def agregar_usuario():

    ventana = ctk.CTkInputDialog(
        text="Ingresa el nombre del usuario:",
        title="Nuevo usuario"
    )

    nombre = ventana.get_input()

    if nombre:
        nombre = nombre.strip()

        if not nombre:
            return

        if nombre not in usuarios_db:
            usuarios_db[nombre] = crear_datos_usuario()
            guardar_usuarios(usuarios_db)
            mensaje_vacio.pack_forget()
            crear_tarjeta_usuario(nombre)
        else:
            mensaje_vacio.pack_forget()
            print("El usuario ya existe.")

# Botón agregar usuario
btn_agregar = ctk.CTkButton(
    header_inicio,
    text="➕ Agregar usuario",
    width=180,
    height=45,
    fg_color=AZUL,
    hover_color=MORADO,
    command=agregar_usuario
)

btn_agregar.pack(side="right")

# ==========================================
# CONTENEDOR DE USUARIOS
# ==========================================

usuarios_container = ctk.CTkScrollableFrame(
    inicio_frame,
    fg_color=NEGRO
)

usuarios_container.pack(
    fill="both",
    expand=True,
    padx=40,
    pady=(0, 30)
)

# ==========================================
# TARJETA DE USUARIO
# ==========================================

def crear_tarjeta_usuario(nombre):


    tarjeta = ctk.CTkFrame(
        usuarios_container,
        fg_color="#1A1A1A",
        corner_radius=20,
        height=90
    )

    tarjeta.pack(
        fill="x",
        pady=10
    )

    tarjeta.pack_propagate(False)

    nombre_label = ctk.CTkLabel(
        tarjeta,
        text=f"👤 {nombre}",
        font=("Montserrat", 24, "bold"),
        text_color=BLANCO
    )

    nombre_label.pack(
        side="left",
        padx=25
    )

    btn_perfil = ctk.CTkButton(
        tarjeta,
        text="📊 Ver perfil",
        width=180,
        height=45,
        fg_color=AZUL,
        hover_color=MORADO,
        command=lambda: abrir_perfil(nombre)
    )

    btn_perfil.pack(
        side="right",
        padx=20
    )


def cargar_tarjetas_existentes():
    if usuarios_db:
        mensaje_vacio.pack_forget()
        for nombre in usuarios_db:
            crear_tarjeta_usuario(nombre)


mensaje_vacio = ctk.CTkLabel(
    usuarios_container,
    text="Aún no hay usuarios registrados.\n\nPresiona 'Agregar usuario' para comenzar.",
    font=("Montserrat", 22),
    text_color=GRIS,
    justify="center"
)

mensaje_vacio.pack(expand=True, pady=150)

cargar_tarjetas_existentes()


# ==========================================
# PANTALLA SESIÓN T'ALAT
# ==========================================

sesion_frame = ctk.CTkFrame(
    app,
    fg_color=NEGRO
)


# ==========================================
# CABECERA SESIÓN
# ==========================================

header_sesion = ctk.CTkFrame(
    sesion_frame,
    fg_color=NEGRO,
    height=70
)

header_sesion.pack(
    fill="x",
    padx=30,
    pady=20
)


btn_regresar_sesion = ctk.CTkButton(
    header_sesion,
    text="⏹ Terminar sesión",
    width=190,
    height=45,
    fg_color="#B22222",
    hover_color="#8B0000",
    command=terminar_sesion)

btn_regresar_sesion.pack(
    side="left"
)


titulo_sesion = ctk.CTkLabel(
    header_sesion,
    text="🎵 Sesión T'ALAT",
    font=("Montserrat",34,"bold"),
    text_color=BLANCO
)

titulo_sesion.pack(
    side="left",
    padx=30
)



# ==========================================
# CONTENIDO PRINCIPAL
# ==========================================

zona_sesion = ctk.CTkFrame(
    sesion_frame,
    fg_color=NEGRO
)

zona_sesion.pack(
    fill="both",
    expand=True,
    padx=40,
    pady=10
)



# ==========================================
# ESPACIO DE CÁMARA
# ==========================================

camara_label = ctk.CTkLabel(
    zona_sesion,
    text="📷 Cámara T'ALAT",
    width=640,
    height=360,
    fg_color="#111111",
    text_color=GRIS,
    font=("Montserrat",25)
)

camara_label.pack(
    side="left",
    padx=20
)

# ==========================================
# INFORMACIÓN DE SESIÓN
# ==========================================

info_sesion = ctk.CTkFrame(
    zona_sesion,
    fg_color="#1A1A1A",
    corner_radius=20,
    width=460,
    height=400,

)

info_sesion.pack(
    side="right",
    fill="both",
    padx=20
)

info_sesion.pack_propagate(False)

# ------------------------
# PICTOGRAMA
# ------------------------

emoji_label = ctk.CTkLabel(
    info_sesion,
    text="😐",
    font=("Segoe UI Emoji", 250)
)

emoji_label.pack(
    pady=(20,5)
)

# ------------------------
# INSTRUCCIÓN
# ------------------------

instruccion_label = ctk.CTkLabel(
    info_sesion,
    text="INSTRUCCIÓN: Primero calibramos tu rostro. Después registraremos cuánto se mueve tu boca en cada expresión para compararlo con tus propias mediciones anteriores.",
    font=("Montserrat",16,"bold"),
    wraplength=390,
    justify="center",
    text_color=AZUL
)

instruccion_label.pack(
    pady=(5,10),
    padx=15
)

# ------------------------
# EMOCIÓN
# ------------------------

estado_expresion = ctk.CTkLabel(
    info_sesion,
    text="REPOSO",
    font=("Montserrat",30,"bold"),
    text_color=BLANCO
)

estado_expresion.pack()

# ------------------------
# NOTA
# ------------------------

estado_nota = ctk.CTkLabel(
    info_sesion,
    text="🎹 --",
    font=("Montserrat",26,"bold"),
    text_color=AZUL
)

estado_nota.pack(
    pady=20
)

# ------------------------
# MENSAJE
# ------------------------

mensaje_label = ctk.CTkLabel(
    info_sesion,
    text="Haz una expresión",
    font=("Montserrat",22),
    wraplength=260,
    justify="center",
    text_color=GRIS
)

mensaje_label.pack(
    pady=20
)

# ------------------------
# MODO
# ------------------------

modo_actual = ctk.CTkLabel(
    info_sesion,
    text="🎹 Piano Libre",
    font=("Montserrat",20),
    text_color=GRIS
)

modo_actual.pack(
    side="bottom",
    pady=25
)

# ==========================================
# SELECCIÓN DE MODOS
# ==========================================

botones_modo = ctk.CTkFrame(
    sesion_frame,
    fg_color=NEGRO
)

botones_modo.pack(
    pady=20
)


btn_piano = ctk.CTkButton(
    botones_modo,
    text="🎹 Piano Libre",
    width=250,
    height=55,
    fg_color=AZUL,
    hover_color=MORADO,
    font=("Montserrat",18,"bold")
)

btn_piano.grid(
    row=0,
    column=0,
    padx=20
)



btn_cancion = ctk.CTkButton(
    botones_modo,
    text="🎼 Canción Guiada",
    width=250,
    height=55,
    fg_color="#222222",
    hover_color=MORADO,
    font=("Montserrat",18,"bold")
)

btn_cancion.grid(
    row=0,
    column=1,
    padx=20
)

# ==========================================
# PANTALLA PERFIL
# ==========================================
perfil_frame = ctk.CTkFrame(app, fg_color=NEGRO)

# Cabecera (NO se mueve)
header_perfil = ctk.CTkFrame(
    perfil_frame,
    fg_color=NEGRO,
    height=70
)

header_perfil.pack(fill="x")

# Contenido con scroll
contenido_scroll = ctk.CTkScrollableFrame(
    perfil_frame,
    fg_color=NEGRO
)

contenido_scroll.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=10
)


# ==========================================
# USUARIO ACTUAL
# ==========================================

usuario_actual = None

# ==========================================
# ABRIR PERFIL DEL USUARIO
# ==========================================

def actualizar_estadisticas_perfil():

    if usuario_actual is None:
        return

    datos = usuarios_db.get(usuario_actual)

    if datos is None:
        return

    minutos = datos.get("tiempo_total", 0) // 60

    tiempo_total_label.configure(
        text=f"Tiempo total: {minutos} min"
    )

    sesiones_label.configure(
        text=f"Sesiones realizadas: {datos.get('sesiones', 0)}"
    )

    expresiones_label.configure(
        text=f"Expresiones registradas: {datos.get('expresiones', 0)}"
    )

    notas_label.configure(
        text=f"Notas detectadas: {datos.get('notas', 0)}"
    )

    ultima = datos.get("ultima_sesion", "")

    ultima_label.configure(
        text=(
            f"Última sesión: {ultima}"
            if ultima
            else "Última sesión: Sin sesiones"
        )
    )


def mostrar_detalle_estadisticas():
    if usuario_actual is None:
        return

    datos = usuarios_db.get(usuario_actual, {})
    historial = datos.get("historial_sesiones", [])

    ventana = ctk.CTkToplevel(app)
    ventana.title(f"Estadísticas detalladas - {usuario_actual}")
    ventana.geometry("900x650")
    ventana.configure(fg_color=NEGRO)
    ventana.grab_set()

    titulo = ctk.CTkLabel(
        ventana,
        text=f"📈 Mediciones detalladas de {usuario_actual}",
        font=("Montserrat",28,"bold"),
        text_color=BLANCO
    )
    titulo.pack(pady=(20,10))

    explicacion = ctk.CTkLabel(
        ventana,
        text="Las mediciones se guardan por sesión y se comparan con la medición anterior de la misma expresión.",
        font=("Montserrat",15),
        text_color=GRIS,
        wraplength=820,
        justify="center"
    )
    explicacion.pack(pady=(0,15), padx=20)

    scroll = ctk.CTkScrollableFrame(
        ventana,
        fg_color=NEGRO
    )
    scroll.pack(fill="both", expand=True, padx=25, pady=(0,20))

    if not historial:
        ctk.CTkLabel(
            scroll,
            text="Aún no hay mediciones detalladas registradas.",
            font=("Montserrat",20),
            text_color=GRIS
        ).pack(pady=100)
    else:
        for numero, sesion in enumerate(reversed(historial), start=1):
            bloque = ctk.CTkFrame(
                scroll,
                fg_color="#1A1A1A",
                corner_radius=18
            )
            bloque.pack(fill="x", pady=10, padx=5)

            ctk.CTkLabel(
                bloque,
                text=f"Sesión {len(historial) - numero + 1}  •  {sesion.get('fecha', 'Sin fecha')}  •  {sesion.get('duracion_segundos', 0)//60} min",
                font=("Montserrat",20,"bold"),
                text_color=BLANCO
            ).pack(anchor="w", padx=20, pady=(15,8))

            expresiones = sesion.get("expresiones", [])

            if not expresiones:
                ctk.CTkLabel(
                    bloque,
                    text="No hubo expresiones registradas en esta sesión.",
                    font=("Montserrat",15),
                    text_color=GRIS
                ).pack(anchor="w", padx=20, pady=(0,15))
                continue

            # Agrupar visualmente por expresión, pero conservar el orden temporal.
            por_expresion = {}
            for registro in expresiones:
                por_expresion.setdefault(registro.get("emocion", "DESCONOCIDA"), []).append(registro)

            for emocion, registros in por_expresion.items():
                sub = ctk.CTkFrame(
                    bloque,
                    fg_color="#111111",
                    corner_radius=12
                )
                sub.pack(fill="x", padx=20, pady=7)

                ctk.CTkLabel(
                    sub,
                    text=f"{registros[0].get('emoji', '') if registros[0].get('emoji') else ''} {emocion}",
                    font=("Montserrat",18,"bold"),
                    text_color=AZUL
                ).pack(anchor="w", padx=15, pady=(10,5))

                for i, registro in enumerate(registros, start=1):
                    apertura = registro.get("apertura_boca", 0)
                    ancho = registro.get("ancho_boca", 0)
                    comparacion = registro.get("comparacion", "")
                    momento = registro.get("momento", "")

                    ojos = registro.get("apertura_ojos", 0)
                    cejas = registro.get("altura_ceja", 0)
                    cercania = registro.get("cercania_cejas", 0)
                    curva = registro.get("curva_boca", 0)

                    ctk.CTkLabel(
                        sub,
                        text=(
                            f"{i}. {momento}\n"
                            f"   Boca: apertura {apertura:.5f} | ancho {ancho:.5f}\n"
                            f"   Ojos: apertura {ojos:.5f}\n"
                            f"   Cejas: altura {cejas:.5f} | separación {cercania:.5f}\n"
                            f"   Curvatura boca: {curva:.5f}\n"
                            f"   {comparacion}"
                        ),
                        font=("Montserrat",15),
                        text_color=BLANCO,
                        justify="left"
                    ).pack(anchor="w", padx=15, pady=6)

                ctk.CTkFrame(sub, height=5, fg_color="#111111").pack()

    ctk.CTkButton(
        ventana,
        text="Cerrar",
        width=160,
        height=42,
        fg_color=AZUL,
        hover_color=MORADO,
        command=ventana.destroy
    ).pack(pady=(0,20))


def actualizar_comentarios_perfil():
    if usuario_actual is None:
        return

    datos = usuarios_db.get(usuario_actual, {})
    comentario = datos.get("comentarios", "")

    caja_comentarios.delete("1.0", "end")
    if comentario:
        caja_comentarios.insert("1.0", comentario)


def abrir_perfil(nombre):

    global usuario_actual

    usuario_actual = nombre

    inicio_frame.pack_forget()

    perfil_frame.pack(
        fill="both",
        expand=True
    )

    titulo_perfil.configure(
        text=f"👤 {usuario_actual}"
    )

    actualizar_estadisticas_perfil()
    actualizar_comentarios_perfil()

# ==========================================
# CABECERA PERFIL
# ==========================================

header_perfil = ctk.CTkFrame(
    perfil_frame,
    fg_color=NEGRO,
    height=70
)

header_perfil.pack(
    fill="x",
    padx=30,
    pady=20
)

btn_regresar_perfil = ctk.CTkButton(
    header_perfil,
    text="← Usuarios",
    width=140,
    command=lambda: (
        perfil_frame.pack_forget(),
        inicio_frame.pack(fill="both", expand=True)
    )
)

btn_regresar_perfil.pack(
    side="left"
)

titulo_perfil = ctk.CTkLabel(
    header_perfil,
    text="",
    font=("Montserrat",34,"bold"),
    text_color=BLANCO
)

titulo_perfil.pack(
    side="left",
    padx=25
)

contenido_perfil = ctk.CTkLabel(
    contenido_scroll,
    text="Aquí aparecerán las estadísticas del usuario.",
    font=("Montserrat",24),
    text_color=GRIS
)

contenido_perfil.pack(
    pady=120
)

estadisticas = ctk.CTkFrame(
    contenido_scroll,
    fg_color="#1A1A1A",
    corner_radius=20
)

estadisticas.pack(
    fill="x",
    padx=40,
    pady=20
)

titulo_estadisticas = ctk.CTkLabel(
    estadisticas,
    text="📊 Estadísticas",
    font=("Montserrat",28,"bold"),
    text_color=BLANCO
)

titulo_estadisticas.pack(
    pady=(20,10)
)

grafica = ctk.CTkLabel(
    estadisticas,
    text="(Aquí aparecerá la gráfica de progreso)",
    font=("Montserrat",18),
    text_color=GRIS
)

grafica.pack(
    pady=(20,10)
)

btn_detalle_estadisticas = ctk.CTkButton(
    estadisticas,
    text="🔍 Ver mediciones detalladas",
    width=300,
    height=48,
    fg_color=AZUL,
    hover_color=MORADO,
    font=("Montserrat",16,"bold"),
    command=mostrar_detalle_estadisticas
)

btn_detalle_estadisticas.pack(
    pady=(0,20)
)

# Todo el bloque de estadísticas también responde al clic sobre el botón.

info = ctk.CTkFrame(
    contenido_scroll,
    fg_color="#1A1A1A",
    corner_radius=20
)

info.pack(
    fill="x",
    padx=40,
    pady=15
)

tiempo_total_label = ctk.CTkLabel(
    info,
    text="Tiempo total: 0 min",
    font=("Montserrat",20)
)

tiempo_total_label.pack(
    anchor="w",
    padx=20,
    pady=8
)

sesiones_label = ctk.CTkLabel(
    info,
    text="Sesiones realizadas: 0",
    font=("Montserrat",20)
)

sesiones_label.pack(
    anchor="w",
    padx=20,
    pady=8
)

ultima_label = ctk.CTkLabel(
    info,
    text="Última sesión: Sin sesiones",
    font=("Montserrat",20)
)

ultima_label.pack(
    anchor="w",
    padx=20,
    pady=8
)

expresiones_label = ctk.CTkLabel(
    info,
    text="Expresiones registradas: 0",
    font=("Montserrat",20)
)

expresiones_label.pack(
    anchor="w",
    padx=20,
    pady=8
)

notas_label = ctk.CTkLabel(
    info,
    text="Notas detectadas: 0",
    font=("Montserrat",20)
)

notas_label.pack(
    anchor="w",
    padx=20,
    pady=8
)

comentarios = ctk.CTkFrame(
    contenido_scroll,
    fg_color="#1A1A1A",
    corner_radius=20
)

comentarios.pack(
    fill="x",
    padx=40,
    pady=15
)

ctk.CTkLabel(
    comentarios,
    text="📝 Comentarios",
    font=("Montserrat",28,"bold")
).pack(pady=(15,10))

caja_comentarios = ctk.CTkTextbox(
    comentarios,
    height=140
)

caja_comentarios.pack(
    fill="x",
    padx=20,
    pady=(0,20)
)

botones = ctk.CTkFrame(
    contenido_scroll,
    fg_color=NEGRO
)

botones.pack(
    pady=20
)

ctk.CTkButton(
    botones,
    text="▶ Iniciar sesión",
    width=220,
    fg_color=AZUL,
    hover_color=MORADO,
    command=iniciar_sesion
).grid(row=0, column=0, padx=10)

ctk.CTkButton(
    botones,
    text="✏ Editar usuario",
    width=220
).grid(row=0, column=1, padx=10)

ctk.CTkButton(
    botones,
    text="🗑 Eliminar usuario",
    width=220,
    fg_color="#B22222",
    hover_color="#8B0000"
).grid(row=0, column=2, padx=10)


# ==========================================
# EJECUTAR
# ==========================================

app.mainloop()