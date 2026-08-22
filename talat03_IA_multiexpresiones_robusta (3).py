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

app.title("TALAT")

app.geometry("1280x720")

app.configure(
    fg_color="#000000"
)


# ==========================================
# COLORES TALAT
# ==========================================

NEGRO = "#000000"
BLANCO = "#FFFFFF"
AZUL = "#3B9DFF"
MORADO = "#9B4DFF"
GRIS = "#BFBFBF"
MORADO2 = "#5D2AA6"
ROJO = "#FF0000"
AMARILLO = "#FFD63D"
VERDE = "#4CF405"
CELESTE = "#05CEFA"

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
# MINI BASE DE DATOS TALAT
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
    # Datos personales: los usuarios creados antes de esta versión
    # simplemente quedan vacíos hasta que se editen.
    _datos.setdefault("edad", "")
    _datos.setdefault("motivo", "")
    # Instrumento del Modo Libre: los gestos que grabó esta persona.
    _datos.setdefault("instrumento_libre", [])

def crear_datos_usuario():
    return {
        "sesiones": 0,
        "tiempo_total": 0,
        "expresiones": 0,
        "notas": 0,
        "canciones_completadas": 0,
        "ultima_sesion": "",
        "comentarios": "",
        "historial_sesiones": [],
        "edad": "",
        "motivo": "",
        "instrumento_libre": []
    }



# ==========================================
# DETECTOR DE GESTOS TALAT
# ==========================================

class DetectorGestos:
    """
    Detector geométrico robusto para TALAT.
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
        if da_boca > 0.010 and dcurva < -0.002:
            scores["ALEGRÍA"] += 0.8

        # SORPRESA
        if d_altura_labios > 0.028:
            scores["SORPRESA"] += 0.5
        if d_ancho_labios > 0.020:
            scores["SORPRESA"] += 0.3
        if dap_boca > 0.012:
            scores["SORPRESA"] += 1.0
        if dojo_i > 0.004:
            scores["SORPRESA"] += 1.0
        if dojo_d > 0.004:
            scores["SORPRESA"] += 1.0
        if dceja_i > 0.005:
            scores["SORPRESA"] += 0.9
        if dceja_d > 0.005:
            scores["SORPRESA"] += 0.9
        if dcurva > 0.004:
            scores["SORPRESA"] -= 1.5
        if dojos < 0.002 and dceja < 0.002:
            scores["SORPRESA"] -= 1.2

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
        if (
            d_altura_labios > 0.012
            and dap_boca < 0.010
            and dcurva > -0.002
            and da_boca < 0.012
        ):
            scores["TRISTEZA"] += 1.2
        if d_altura_labios > 0.012 and dcurva > 0.003:
            scores["TRISTEZA"] += 0.8
        if dcurva < -0.004:
            scores["TRISTEZA"] -= 1.5
        if da_boca > 0.018:
            scores["TRISTEZA"] -= 1.0

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
            "ABURRIMIENTO": ("😑", "DO", MORADO2, "Mantén la expresión"),
            "SORPRESA": ("😮", "RE", VERDE, "¡Muy bien!"),
            "IRA": ("😠", "MI", ROJO, "Excelente"),
            "TRISTEZA": ("🙁", "FA", CELESTE, "Muy bien"),
            "ALEGRÍA": ("😄", "SOL", AMARILLO, "¡Excelente!")
        }

        self.emocion = emocion
        self.emoji, self.nota, self.color, self.mensaje = datos[emocion]


detector = DetectorGestos()


# ==========================================
# MOTOR DE NOTAS Y ACORDES (SONIDO)
# ==========================================
#
# Aquí se produce el sonido. Un acorde son varias notas sonando a la vez.
# Cuando el proyecto pase a la Raspberry Pi con los relevadores SSR,
# solo hay que rellenar enviar_a_hardware(): recibe la lista de notas
# del acorde y activa un relevador por cada una.

FRECUENCIAS_NOTAS = {
    "DO": 261.63,
    "RE": 293.66,
    "MI": 329.63,
    "FA": 349.23,
    "SOL": 392.00,
    "LA": 440.00,
    "SI": 493.88,
    # Segunda octava, necesaria para completar los acordes.
    "DO5": 523.25,
    "RE5": 587.33,
    "MI5": 659.25,
    "FA5": 698.46,
    "SOL5": 783.99
}

# Cuando conectes el piano físico, asigna aquí tu objeto PianoSSR.
piano_hardware = None


class MotorDeNotas:
    """
    Reproduce notas sueltas y acordes.

    Busca un motor de sonido disponible en este orden:
      1. pygame   (Windows, Linux y Raspberry Pi)
      2. winsound (solo Windows, viene con Python)
      3. silencio (la app sigue funcionando, solo sin audio)
    """

    def __init__(self):
        self.backend = "silencio"
        self.cache = {}
        self._preparar()

    def _preparar(self):
        try:
            import pygame
            pygame.mixer.pre_init(22050, -16, 1, 512)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(8)
            self.pygame = pygame
            self.backend = "pygame"
            return
        except Exception:
            pass

        try:
            import winsound
            self.winsound = winsound
            self.backend = "winsound"
            return
        except Exception:
            pass

        print("TALAT: sin motor de sonido. Instala pygame para escuchar los acordes.")

    def disponible(self):
        return self.backend != "silencio"

    def _crear_sonido(self, nota, duracion=0.85):
        """Genera una onda parecida a un piano, sin librerías extra."""
        import array
        import math

        frecuencia = FRECUENCIAS_NOTAS.get(nota)

        if frecuencia is None:
            return None

        muestreo = 22050
        total = int(muestreo * duracion)
        datos = array.array("h")

        for i in range(total):
            t = i / muestreo
            avance = i / total

            # Ataque rápido y caída lenta: evita el "clic" y suena más a piano.
            if avance < 0.02:
                envolvente = avance / 0.02
            else:
                envolvente = (1.0 - avance) ** 1.6

            onda = (
                math.sin(2 * math.pi * frecuencia * t) * 0.60
                + math.sin(2 * math.pi * frecuencia * 2 * t) * 0.22
                + math.sin(2 * math.pi * frecuencia * 3 * t) * 0.08
            )

            datos.append(int(max(-1.0, min(1.0, onda * envolvente)) * 16000))

        return self.pygame.mixer.Sound(buffer=datos.tobytes())

    def _reproducir_notas(self, notas):
        try:
            if self.backend == "pygame":
                for nota in notas:
                    if nota not in self.cache:
                        self.cache[nota] = self._crear_sonido(nota)

                    sonido = self.cache[nota]

                    if sonido is not None:
                        sonido.play()

            elif self.backend == "winsound":
                # winsound no puede sonar dos notas a la vez.
                # Las tocamos muy rápido, como un arpegio.
                for nota in notas:
                    frecuencia = FRECUENCIAS_NOTAS.get(nota)
                    if frecuencia:
                        self.winsound.Beep(int(frecuencia), 120)

        except Exception as e:
            print("No se pudo reproducir el sonido:", e)

    def enviar_a_hardware(self, notas):
        """
        Punto de conexión con los relevadores SSR del piano.

        Recibe la lista de notas del acorde. Cuando tengas listo el
        módulo HARWARE, basta con:  piano_hardware = PianoSSR()
        """
        if piano_hardware is None:
            return

        try:
            for nota in notas:
                piano_hardware.tocar(nota)
        except Exception as e:
            print("No se pudo activar el relevador:", e)

    def tocar(self, nota):
        """Una sola nota (lo que usa el Modo Terapia)."""
        if nota in ("--", None):
            return

        self.tocar_notas([nota])

    def tocar_acorde(self, clave_acorde):
        """Un acorde completo, por su clave: DO, FA, SOL7..."""
        acorde = ACORDES.get(clave_acorde)

        if acorde is None:
            return

        self.tocar_notas(acorde["notas"])

    def tocar_notas(self, notas):
        """Suena una o varias notas a la vez, sin congelar la interfaz."""
        threading.Thread(
            target=self._reproducir_notas,
            args=(list(notas),),
            daemon=True
        ).start()

        self.enviar_a_hardware(notas)


motor_notas = MotorDeNotas()


# ==========================================
# ACORDES Y GESTOS
# ==========================================
#
# Cada gesto produce un ACORDE completo, no una nota suelta.
# Así se tocan canciones de verdad con muy pocos gestos.

ACORDES = {
    "DO": {
        "nombre": "DO mayor",
        "grado": "I",
        "notas": ["DO", "MI", "SOL"],
        "color": "#FFD93D"
    },
    "FA": {
        "nombre": "FA mayor",
        "grado": "IV",
        "notas": ["FA", "LA", "DO5"],
        "color": "#FF4444"
    },
    "SOL7": {
        "nombre": "SOL séptima",
        "grado": "V7",
        "notas": ["SOL", "SI", "RE5", "FA5"],
        "color": AZUL
    }
}

# Qué cara hay que poner para cada acorde.
# Elegimos las tres expresiones MÁS distintas entre sí, para que el
# detector no las confunda. TRISTEZA y ABURRIMIENTO quedan libres
# para cuando agreguemos acordes menores (LAm, REm).
GESTO_POR_ACORDE = {
    "DO": "ALEGRÍA",
    "SOL7": "SORPRESA",
    "FA": "IRA"
}

ACORDE_POR_GESTO = {
    gesto: acorde for acorde, gesto in GESTO_POR_ACORDE.items()
}

GUIA_DE_GESTOS = {
    "ABURRIMIENTO": {
        "emoji": "😑",
        "titulo": "Cara de aburrimiento",
        "como": "Entrecierra los ojos y relaja la cara, como si tuvieras sueño."
    },
    "SORPRESA": {
        "emoji": "😮",
        "titulo": "Cara de sorpresa",
        "como": "Abre la boca y sube las cejas, con los ojos bien abiertos."
    },
    "IRA": {
        "emoji": "😠",
        "titulo": "Cara de enojo",
        "como": "Frunce el ceño juntando las cejas hacia abajo."
    },
    "TRISTEZA": {
        "emoji": "🙁",
        "titulo": "Cara de tristeza",
        "como": "Baja las esquinas de la boca y saca un poco el labio de abajo."
    },
    "ALEGRÍA": {
        "emoji": "😄",
        "titulo": "Cara de alegría",
        "como": "Sonríe estirando la boca y subiendo las esquinas."
    }
}


# ==========================================
# CANCIONES GUIADAS#
# ==========================================

CANCIONES = {
    "cucaracha": {
        "titulo": "La cucaracha",
        "subtitulo": "2 gestos · canción fácil",
        "dificultad": "Fácil",
        "bloques": [
            {
                "letra": (
                    "La cucaracha, la cucaracha\n"
                    "ya no puede caminar"
                ),
                "gesto": "ALEGRÍA",
                "acordes": [
                    "DO",
                    "DO",
                    "SOL7",
                    "DO"
                ]
            },
            {
                "letra": (
                    "Porque no tiene, porque le falta\n"
                    "las dos patitas de atrás"
                ),
                "gesto": "SORPRESA",
                "acordes": [
                    "SOL7",
                    "SOL7",
                    "SOL7",
                    "DO"
                ]
            }
        ]
    },

    "estrellita_corta": {
        "titulo": "Estrellita, ¿dónde estás?",
        "subtitulo": "2 gestos · versión corta",
        "dificultad": "Media",
        "bloques": [
            {
                "letra": (
                    "Estrellita, ¿dónde estás?\n"
                    "Me pregunto quién serás"
                ),
                "gesto": "ALEGRÍA",
                "acordes": [
                    "DO",
                    "DO",
                    "FA",
                    "DO",
                    "FA",
                    "DO",
                    "SOL7",
                    "DO"
                ]
            },
            {
                "letra": (
                    "En el cielo o en el mar\n"
                    "un diamante de verdad"
                ),
                "gesto": "SORPRESA",
                "acordes": [
                    "DO",
                    "FA",
                    "DO",
                    "SOL7",
                    "DO",
                    "FA",
                    "DO",
                    "SOL7"
                ]
            }
        ]
    },

    "estrellita_completa": {
        "titulo": "Estrellita, ¿dónde estás?",
        "subtitulo": "3 gestos · canción completa",
        "dificultad": "Larga",
        "bloques": [
            {
                "letra": (
                    "Estrellita, ¿dónde estás?\n"
                    "Me pregunto quién serás"
                ),
                "gesto": "ALEGRÍA",
                "acordes": [
                    "DO",
                    "DO",
                    "FA",
                    "DO",
                    "FA",
                    "DO",
                    "SOL7",
                    "DO"
                ]
            },
            {
                "letra": (
                    "En el cielo o en el mar\n"
                    "un diamante de verdad"
                ),
                "gesto": "SORPRESA",
                "acordes": [
                    "DO",
                    "FA",
                    "DO",
                    "SOL7",
                    "DO",
                    "FA",
                    "DO",
                    "SOL7"
                ]
            },
            {
                "letra": (
                    "Estrellita, ¿dónde estás?\n"
                    "Me pregunto quién serás"
                ),
                "gesto": "ALEGRÍA",
                "acordes": [
                    "DO",
                    "DO",
                    "FA",
                    "DO",
                    "FA",
                    "DO",
                    "SOL7",
                    "DO"
                ]
            }
        ]
    }
}


def gestos_usados(cancion):
    return [
        bloque["gesto"]
        for bloque in cancion["bloques"]
    ]

class CancionGuiada:
    """
    Canción Guiada:

    UN GESTO = UN BLOQUE MUSICAL COMPLETO.

    La persona hace una expresión una sola vez.
    TALAT reproduce automáticamente todos los acordes
    del bloque y la persona puede relajar la cara mientras suena.
    """

    def __init__(self, motor):
        self.motor = motor
        self.clave = None
        self.cancion = None
        self.bloques = []
        self.indice = 0

        self.estado = "detenida"
        # detenida / tocando / pausada / reproduciendo / terminada

        self.esperando_reposo = False
        self.gesto_detectado = "REPOSO"

        self.aciertos = 0
        self.ya_registrada = None

    # --------------------------------------
    # CONSULTAS
    # --------------------------------------

    def cargada(self):
        return self.cancion is not None

    def total_bloques(self):
        return len(self.bloques)

    def bloque_actual(self):
        if not self.bloques or self.indice >= len(self.bloques):
            return None

        return self.bloques[self.indice]

    def gesto_actual(self):
        bloque = self.bloque_actual()

        if bloque is None:
            return None

        return bloque.get("gesto")

    def letra_actual(self):
        bloque = self.bloque_actual()

        if bloque is None:
            return ""

        return bloque.get("letra", "")

    def acordes_actuales(self):
        bloque = self.bloque_actual()

        if bloque is None:
            return []

        return bloque.get("acordes", [])

    def progreso(self):
        if not self.bloques:
            return 0.0

        return self.indice / len(self.bloques)

    # --------------------------------------
    # CARGAR
    # --------------------------------------

    def cargar(self, clave):
        cancion = CANCIONES.get(clave)

        if cancion is None:
            return False

        self.clave = clave
        self.cancion = cancion
        self.bloques = list(cancion.get("bloques", []))

        self.indice = 0
        self.aciertos = 0

        self.estado = "detenida"
        self.esperando_reposo = False

        self.gesto_detectado = "REPOSO"
        self.ya_registrada = None

        return True

    # --------------------------------------
    # CONTROL
    # --------------------------------------

    def iniciar(self):
        if not self.cargada():
            return

        if self.estado == "terminada":
            self.reiniciar()

        self.estado = "tocando"

        # Obliga a comenzar desde rostro relajado.
        self.esperando_reposo = True

    def pausar(self):
        if self.estado == "tocando":
            self.estado = "pausada"

        elif self.estado == "pausada":
            self.estado = "tocando"

            # Debe volver a reposo antes del siguiente gesto.
            self.esperando_reposo = True

    def reiniciar(self):
        self.indice = 0
        self.aciertos = 0

        self.estado = "detenida"
        self.esperando_reposo = False

        self.gesto_detectado = "REPOSO"
        self.ya_registrada = None

    def salir(self):
        self.reiniciar()

    # --------------------------------------
    # PROCESAMIENTO DEL GESTO
    # --------------------------------------

    def procesar(self, gesto):
        """
        Un solo gesto correcto inicia un bloque completo.
        """

        self.gesto_detectado = gesto

        if self.estado != "tocando":
            return False

        # Mientras un bloque está sonando no aceptamos
        # otro gesto.
        if self.estado == "reproduciendo":
            return False

        # Primero necesitamos volver a reposo.
        if gesto == "REPOSO":
            self.esperando_reposo = False
            return False

        if self.esperando_reposo:
            return False

        objetivo = self.gesto_actual()

        if objetivo is None:
            return False

        if gesto != objetivo:
            return False

        return self._acertar_bloque()

    # --------------------------------------
    # EJECUTAR BLOQUE
    # --------------------------------------

    def _acertar_bloque(self):
        bloque = self.bloque_actual()

        if bloque is None:
            return False

        acordes = bloque.get("acordes", [])

        if not acordes:
            return False

        self.estado = "reproduciendo"
        self.aciertos += 1

        # Guardamos el bloque actual antes de avanzar.
        indice_bloque = self.indice

        # Reproduce el bloque completo.
        threading.Thread(
            target=self._reproducir_bloque,
            args=(acordes, indice_bloque),
            daemon=True
        ).start()

        return True

    def _reproducir_bloque(self, acordes, indice_bloque):
        try:
            for acorde in acordes:

                if self.estado == "pausada":
                    return

                self.motor.tocar_acorde(acorde)

                # Tiempo entre acordes para que se escuche
                # como una pequeña progresión musical.
                import time
                time.sleep(0.75)

            # Solo la interfaz principal debe modificar
            # el estado de la canción.
            app.after(
                100,
                lambda i=indice_bloque: self._terminar_bloque(i)
            )

        except Exception as e:
            print("Error reproduciendo bloque:", e)

            app.after(
                100,
                lambda i=indice_bloque: self._terminar_bloque(i)
            )

    def _terminar_bloque(self, indice_bloque):
        # Evita avanzar accidentalmente si ya cambió la canción.
        if indice_bloque != self.indice:
            return

        self.indice += 1

        if self.indice >= len(self.bloques):
            self.estado = "terminada"
            self.esperando_reposo = False

        else:
            self.estado = "tocando"

            # Para activar el siguiente bloque:
            # primero debe relajar la cara.
            self.esperando_reposo = True

        actualizar_panel_cancion()

    # --------------------------------------
    # ESTADO DEL BLOQUE
    # --------------------------------------

    def bloque_reproduciendose(self):
        return self.estado == "reproduciendo"

# ==========================================
# MODO LIBRE
# ==========================================
#
# La persona graba SU propio gesto para cada sonido.
# No hay gestos predefinidos: puede usar la boca, los ojos, las cejas
# o cualquier combinación que le resulte cómoda.
#
# Dos decisiones importantes de precisión:
#
#  1. Distancia NORMALIZADA. Las mediciones no son comparables entre sí:
#     la boca se mueve 0.060 y una ceja apenas 0.013. Sumando diferencias
#     crudas, la boca aplasta a todo lo demás y los gestos de ojos y cejas
#     nunca se reconocen. Dividimos cada medición entre su recorrido
#     típico para que todas pesen igual.
#
#  2. PROMEDIO de varias lecturas. El temblor de los puntos de MediaPipe
#     es del mismo tamaño que el movimiento de una ceja. Promediando las
#     últimas lecturas ese temblor se reduce a la tercera parte.
#
#  Midiendo las dos juntas: el acierto sube de 79% a 99%, y los disparos
#  con la cara quieta bajan a cero.

CLAVES_MODO_LIBRE = [
    "apertura_boca",
    "ancho_boca",
    "curva_boca",
    "apertura_ojos",
    "altura_ceja",
    "cercania_cejas"
]

# Cuánto se mueve cada medición en un gesto completo. Es el divisor
# que pone a todas las mediciones en la misma escala.
RECORRIDO_MEDICION = {
    "apertura_boca": 0.060,
    "ancho_boca": 0.050,
    "curva_boca": 0.012,
    "apertura_ojos": 0.018,
    "altura_ceja": 0.013,
    "cercania_cejas": 0.025
}

# Cómo se llama cada movimiento en palabras, según hacia dónde va.
NOMBRES_MOVIMIENTO = {
    "apertura_boca": ("boca abierta", "boca cerrada"),
    "ancho_boca": ("boca estirada", "labios juntos"),
    "curva_boca": ("comisuras abajo", "sonrisa"),
    "apertura_ojos": ("ojos muy abiertos", "ojos cerrados"),
    "altura_ceja": ("cejas arriba", "cejas abajo"),
    "cercania_cejas": ("cejas separadas", "ceño fruncido")
}

# Lecturas que se promedian antes de decidir qué gesto es.
LECTURAS_SUAVIZADO = 8

# Un gesto por debajo de esto no se distingue del temblor de la cámara.
# Un movimiento completo vale 1.00 en su propio eje; el ruido, ya
# promediado sobre las 20 lecturas de la grabación, vale unos 0.21.
MOVIMIENTO_MINIMO_LIBRE = 0.60

# Qué tan distintos deben ser dos gestos para poder usarse en sonidos
# diferentes. Por debajo de 0.80 la confusión al tocar sube muchísimo.
DIFERENCIA_MINIMA_GESTOS = 0.80

# El gesto debe ganarle claramente al rostro relajado para que suene.
MARGEN_REPOSO_LIBRE = 0.70

TOTAL_MUESTRAS_MOVIMIENTO_LIBRE = 20
SEGUNDOS_CUENTA_REGRESIVA = 3

# Sonidos que se pueden asignar: notas sueltas y acordes.
SONIDOS_LIBRES = {
    "DO": {"etiqueta": "🎵 DO", "notas": ["DO"]},
    "RE": {"etiqueta": "🎵 RE", "notas": ["RE"]},
    "MI": {"etiqueta": "🎵 MI", "notas": ["MI"]},
    "FA": {"etiqueta": "🎵 FA", "notas": ["FA"]},
    "SOL": {"etiqueta": "🎵 SOL", "notas": ["SOL"]},
    "LA": {"etiqueta": "🎵 LA", "notas": ["LA"]},
    "SI": {"etiqueta": "🎵 SI", "notas": ["SI"]},
    "DO5": {"etiqueta": "🎵 DO alto", "notas": ["DO5"]},
    "RE5": {"etiqueta": "🎵 RE alto", "notas": ["RE5"]},
    "MI5": {"etiqueta": "🎵 MI alto", "notas": ["MI5"]},
    "AC_DO": {"etiqueta": "🎹 Acorde DO", "notas": ["DO", "MI", "SOL"]},
    "AC_FA": {"etiqueta": "🎹 Acorde FA", "notas": ["FA", "LA", "DO5"]},
    "AC_SOL7": {"etiqueta": "🎹 Acorde SOL7", "notas": ["SOL", "SI", "RE5", "FA5"]}
}

ETIQUETA_A_SONIDO = {d["etiqueta"]: c for c, d in SONIDOS_LIBRES.items()}
SONIDO_A_ETIQUETA = {c: d["etiqueta"] for c, d in SONIDOS_LIBRES.items()}


def diferencias_contra_reposo(medicion):
    """Cuánto se movió cada parte del rostro respecto al rostro relajado."""
    if detector.base is None:
        return None

    return {
        clave: medicion.get(clave, 0.0) - detector.base.get(clave, 0.0)
        for clave in CLAVES_MODO_LIBRE
    }


def tamano_movimiento(diferencias):
    """Qué tan marcado es el gesto, ya normalizado."""
    return sum(
        abs(diferencias[clave]) / RECORRIDO_MEDICION[clave]
        for clave in CLAVES_MODO_LIBRE
    )


def distancia_normalizada(diferencias_a, diferencias_b):
    """Distancia entre dos gestos con todas las mediciones pesando igual."""
    return sum(
        abs(diferencias_a[clave] - diferencias_b[clave]) / RECORRIDO_MEDICION[clave]
        for clave in CLAVES_MODO_LIBRE
    )


def perfil_movimiento(diferencias):
    """La DIRECCIÓN del gesto, sin su intensidad."""
    normalizadas = {
        clave: diferencias[clave] / RECORRIDO_MEDICION[clave]
        for clave in CLAVES_MODO_LIBRE
    }

    total = sum(abs(v) for v in normalizadas.values())

    if total < 1e-9:
        return None

    return {clave: v / total for clave, v in normalizadas.items()}


def diferencia_entre_gestos(perfil_a, perfil_b):
    """0 = el mismo gesto (aunque uno sea más fuerte). 2 = opuestos."""
    return sum(
        abs(perfil_a[clave] - perfil_b[clave])
        for clave in CLAVES_MODO_LIBRE
    )


def describir_gesto(diferencias):
    """
    Pone en palabras el gesto grabado, para que la persona pueda
    reconocer cuál es cuál al editarlos.
    """
    normalizadas = {
        clave: diferencias[clave] / RECORRIDO_MEDICION[clave]
        for clave in CLAVES_MODO_LIBRE
    }

    ordenadas = sorted(
        normalizadas.items(),
        key=lambda par: abs(par[1]),
        reverse=True
    )

    partes = []

    for clave, valor in ordenadas[:2]:
        if abs(valor) < 0.25:
            continue

        arriba, abajo = NOMBRES_MOVIMIENTO[clave]
        partes.append(arriba if valor > 0 else abajo)

    if not partes:
        return "gesto suave"

    return " + ".join(partes)


class InstrumentoLibre:
    """
    El instrumento que arma la persona: una lista de sonidos,
    cada uno con el gesto que ella misma grabó.
    """

    def __init__(self):
        self.ranuras = []          # [{"sonido": "DO", "gesto": {...}, "texto": "..."}]
        self.historial = []        # últimas lecturas, para promediar
        self.ultimo_sonando = None

    # ---------- edición ----------

    def agregar(self, sonido):
        self.ranuras.append({"sonido": sonido, "gesto": None, "texto": ""})

    def quitar(self, indice):
        if 0 <= indice < len(self.ranuras):
            self.ranuras.pop(indice)
            self.ultimo_sonando = None

    def cambiar_sonido(self, indice, sonido):
        if 0 <= indice < len(self.ranuras):
            self.ranuras[indice]["sonido"] = sonido

    def borrar_gesto(self, indice):
        if 0 <= indice < len(self.ranuras):
            self.ranuras[indice]["gesto"] = None
            self.ranuras[indice]["texto"] = ""
            self.ultimo_sonando = None

    def guardar_gesto(self, indice, diferencias):
        self.ranuras[indice]["gesto"] = dict(diferencias)
        self.ranuras[indice]["texto"] = describir_gesto(diferencias)
        self.ultimo_sonando = None

    def revisar_gesto(self, diferencias, indice):
        """
        ¿Se puede usar este gesto? Devuelve (True, "") o (False, motivo).
        """
        if tamano_movimiento(diferencias) < MOVIMIENTO_MINIMO_LIBRE:
            return False, "Ese gesto es muy suave. Haz un movimiento más marcado."

        perfil = perfil_movimiento(diferencias)

        if perfil is None:
            return False, "No se detectó ningún movimiento."

        for i, ranura in enumerate(self.ranuras):
            if i == indice or ranura["gesto"] is None:
                continue

            perfil_guardado = perfil_movimiento(ranura["gesto"])

            if perfil_guardado is None:
                continue

            if diferencia_entre_gestos(perfil, perfil_guardado) < DIFERENCIA_MINIMA_GESTOS:
                etiqueta = SONIDO_A_ETIQUETA.get(ranura["sonido"], ranura["sonido"])
                return False, f"Ese gesto ya lo usa {etiqueta}. Haz uno diferente."

        return True, ""

    # ---------- tocar ----------

    def reconocer(self, medicion):
        """
        Devuelve el índice de la ranura que corresponde al gesto actual,
        o None si la persona está en reposo.
        """
        diferencias = diferencias_contra_reposo(medicion)

        if diferencias is None:
            return None

        # Promediamos las últimas lecturas para quitarle el temblor.
        self.historial.append(diferencias)

        if len(self.historial) > LECTURAS_SUAVIZADO:
            self.historial.pop(0)

        if len(self.historial) < LECTURAS_SUAVIZADO:
            return None

        suave = {
            clave: sum(h[clave] for h in self.historial) / len(self.historial)
            for clave in CLAVES_MODO_LIBRE
        }

        reposo = {clave: 0.0 for clave in CLAVES_MODO_LIBRE}
        distancia_reposo = distancia_normalizada(suave, reposo)

        mejor = None
        menor = float("inf")

        for i, ranura in enumerate(self.ranuras):
            if ranura["gesto"] is None:
                continue

            distancia = distancia_normalizada(suave, ranura["gesto"])

            if distancia < menor:
                menor = distancia
                mejor = i

        if mejor is None:
            return None

        # El rostro relajado compite como una opción más: si el gesto no
        # le gana claramente, preferimos el silencio.
        if menor > distancia_reposo * MARGEN_REPOSO_LIBRE:
            return None

        return mejor

    def limpiar_historial(self):
        self.historial = []
        self.ultimo_sonando = None

    # ---------- guardar y cargar ----------

    def exportar(self):
        return [
            {
                "sonido": r["sonido"],
                "gesto": r["gesto"],
                "texto": r["texto"]
            }
            for r in self.ranuras
        ]

    def importar(self, datos):
        self.ranuras = []

        for r in datos or []:
            sonido = r.get("sonido")

            if sonido not in SONIDOS_LIBRES:
                continue

            gesto = r.get("gesto")

            if isinstance(gesto, dict):
                gesto = {
                    clave: float(gesto.get(clave, 0.0))
                    for clave in CLAVES_MODO_LIBRE
                }
            else:
                gesto = None

            self.ranuras.append({
                "sonido": sonido,
                "gesto": gesto,
                "texto": r.get("texto", "")
            })

        self.limpiar_historial()


instrumento_libre = InstrumentoLibre()

# Estado de la grabación de un gesto.
grabando_indice = None
grabando_muestras = []
cuenta_regresiva_libre = 0



cancion_guiada = CancionGuiada(motor_notas)

# Modo de la sesión: "terapia" o "cancion".
modo_sesion = "terapia"


# ==========================================
# SISTEMA DE CÁMARA TALAT
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
            text="INSTRUCCIÓN: Mantén tu rostro relajado mientras TALAT calibra tu rostro."
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
        "total_expresiones": estadisticas_sesion["expresiones"],
        "total_notas": estadisticas_sesion["notas"],
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
            text="📷 Cámara TALAT"
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
    # 640x480 conserva la proporción de la cámara. Deformarla a un
    # cuadrado altera las mediciones que mezclan alto con ancho.
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
                    instruccion_label.configure(text="")

                info_sesion.configure(
                    border_width=10,
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

                # ----- MODO LIBRE -----
                if modo_sesion == "libre":
                    if detector.ultima_medicion is not None:
                        procesar_modo_libre(detector.ultima_medicion)

                # ----- MODO CANCIÓN GUIADA -----
                # El modo libre de arriba queda intacto: esto solo se activa
                # cuando el usuario eligió "Canción Guiada".
                if modo_sesion == "cancion":
                    avanzo = cancion_guiada.procesar(detector.ultima_estable)

                    if "actualizar_panel_cancion" in globals():
                        if avanzo:
                            actualizar_panel_cancion()
                        else:
                            refrescar_deteccion_cancion()

    # Mostrar la imagen reducida; 640x480 es mucho más ligero que 1000x750.
    imagen = Image.fromarray(rgb)
    imagen = imagen.resize((1000, 750))

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

    # Toda sesión empieza en Modo Terapia.
    if "activar_modo_terapia" in globals():
        activar_modo_terapia()

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
    text="TALAT",
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
        text="TALAT",
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
    text="Sistema Musical TALAT",
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

def dialogo_datos_usuario(titulo, nombre="", edad="", motivo=""):
    """
    Formulario de nombre, edad y motivo de uso.

    Se usa igual para crear un usuario nuevo y para editar uno existente.
    Devuelve un diccionario con los datos, o None si se cancela.
    """
    resultado = {}

    ventana = ctk.CTkToplevel(app)
    ventana.title(titulo)
    ventana.geometry("560x560")
    ventana.configure(fg_color=NEGRO)
    ventana.resizable(False, False)

    # transient + grab_set: la ventana queda encima y bloquea el resto.
    ventana.transient(app)
    ventana.grab_set()

    ctk.CTkLabel(
        ventana,
        text=titulo,
        font=("Montserrat", 26, "bold"),
        text_color=BLANCO
    ).pack(pady=(25, 20))

    # ---- Nombre ----
    ctk.CTkLabel(
        ventana,
        text="Nombre",
        font=("Montserrat", 16, "bold"),
        text_color=BLANCO,
        anchor="w"
    ).pack(fill="x", padx=40)

    entrada_nombre = ctk.CTkEntry(
        ventana,
        height=42,
        font=("Montserrat", 16),
        placeholder_text="Nombre de la persona"
    )

    entrada_nombre.pack(fill="x", padx=40, pady=(4, 14))
    entrada_nombre.insert(0, nombre)

    # ---- Edad ----
    ctk.CTkLabel(
        ventana,
        text="Edad",
        font=("Montserrat", 16, "bold"),
        text_color=BLANCO,
        anchor="w"
    ).pack(fill="x", padx=40)

    entrada_edad = ctk.CTkEntry(
        ventana,
        height=42,
        font=("Montserrat", 16),
        placeholder_text="Años cumplidos"
    )

    entrada_edad.pack(fill="x", padx=40, pady=(4, 14))
    entrada_edad.insert(0, str(edad))

    # ---- Motivo ----
    ctk.CTkLabel(
        ventana,
        text="¿Por qué usa TALAT?",
        font=("Montserrat", 16, "bold"),
        text_color=BLANCO,
        anchor="w"
    ).pack(fill="x", padx=40)

    ctk.CTkLabel(
        ventana,
        text="Escríbelo con tus palabras. Este texto aparecerá en su perfil.",
        font=("Montserrat", 13),
        text_color=GRIS,
        anchor="w",
        wraplength=470,
        justify="left"
    ).pack(fill="x", padx=40)

    caja_motivo = ctk.CTkTextbox(
        ventana,
        height=110,
        font=("Montserrat", 15)
    )

    caja_motivo.pack(fill="x", padx=40, pady=(6, 6))
    caja_motivo.insert("1.0", motivo)

    aviso = ctk.CTkLabel(
        ventana,
        text="",
        font=("Montserrat", 14, "bold"),
        text_color="#FF9F43"
    )

    aviso.pack(pady=(0, 4))

    def confirmar():
        nombre_nuevo = entrada_nombre.get().strip()
        edad_nueva = entrada_edad.get().strip()
        motivo_nuevo = caja_motivo.get("1.0", "end").strip()

        if not nombre_nuevo:
            aviso.configure(text="El nombre no puede quedar vacío.")
            return

        # La edad puede quedar vacía, pero si se escribe debe ser un número real.
        if edad_nueva:
            if not edad_nueva.isdigit():
                aviso.configure(text="La edad debe ser un número.")
                return

            if not (1 <= int(edad_nueva) <= 120):
                aviso.configure(text="La edad debe estar entre 1 y 120.")
                return

        resultado["nombre"] = nombre_nuevo
        resultado["edad"] = edad_nueva
        resultado["motivo"] = motivo_nuevo

        ventana.destroy()

    botones_dialogo = ctk.CTkFrame(ventana, fg_color="transparent")
    botones_dialogo.pack(pady=(4, 20))

    ctk.CTkButton(
        botones_dialogo,
        text="Guardar",
        width=180,
        height=44,
        fg_color=AZUL,
        hover_color=MORADO,
        font=("Montserrat", 16, "bold"),
        command=confirmar
    ).grid(row=0, column=0, padx=10)

    ctk.CTkButton(
        botones_dialogo,
        text="Cancelar",
        width=180,
        height=44,
        fg_color="#3A3A3A",
        hover_color="#B22222",
        font=("Montserrat", 16, "bold"),
        command=ventana.destroy
    ).grid(row=0, column=1, padx=10)

    entrada_nombre.focus()

    # Detiene la ejecución aquí hasta que se cierre la ventana.
    ventana.wait_window()

    return resultado if resultado else None


def agregar_usuario():

    datos = dialogo_datos_usuario("Nuevo usuario")

    if datos is None:
        return

    nombre = datos["nombre"]

    if nombre in usuarios_db:
        print("El usuario ya existe.")
        return

    usuarios_db[nombre] = crear_datos_usuario()
    usuarios_db[nombre]["edad"] = datos["edad"]
    usuarios_db[nombre]["motivo"] = datos["motivo"]

    guardar_usuarios(usuarios_db)

    mensaje_vacio.pack_forget()
    crear_tarjeta_usuario(nombre)

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

def resumen_usuario(datos):
    """Línea corta con la edad y el motivo, para la tarjeta de la lista."""
    edad = str(datos.get("edad", "")).strip()
    motivo = str(datos.get("motivo", "")).strip()

    partes = []

    if edad:
        partes.append(f"{edad} años")

    if motivo:
        # En la tarjeta solo cabe una línea; el texto completo va en el perfil.
        corto = motivo.replace("\n", " ")

        if len(corto) > 60:
            corto = corto[:57] + "..."

        partes.append(corto)

    if not partes:
        return "Sin datos personales · edítalos desde su perfil"

    return "  ·  ".join(partes)


def crear_tarjeta_usuario(nombre):


    tarjeta = ctk.CTkFrame(
        usuarios_container,
        fg_color="#1A1A1A",
        corner_radius=20,
        height=110
    )

    tarjeta.pack(
        fill="x",
        pady=10
    )

    tarjeta.pack_propagate(False)

    datos = usuarios_db.get(nombre, {})

    # Bloque de texto a la izquierda: nombre arriba, datos personales abajo.
    columna = ctk.CTkFrame(tarjeta, fg_color="transparent")

    columna.pack(side="left", padx=25, fill="both", expand=True)

    nombre_label = ctk.CTkLabel(
        columna,
        text=f"👤 {nombre}",
        font=("Montserrat", 24, "bold"),
        text_color=BLANCO,
        anchor="w"
    )

    nombre_label.pack(anchor="w", pady=(14, 0))

    ctk.CTkLabel(
        columna,
        text=resumen_usuario(datos),
        font=("Montserrat", 14),
        text_color=GRIS,
        anchor="w",
        justify="left"
    ).pack(anchor="w")

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


def recargar_tarjetas_usuarios():
    """Vuelve a dibujar la lista completa (tras editar o renombrar)."""
    for hijo in usuarios_container.winfo_children():
        if hijo is not mensaje_vacio:
            hijo.destroy()

    if usuarios_db:
        mensaje_vacio.pack_forget()
        for nombre in usuarios_db:
            crear_tarjeta_usuario(nombre)
    else:
        mensaje_vacio.pack(expand=True, pady=150)


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
# PANTALLA SESIÓN TALAT
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
    text="🎵 Sesión TALAT",
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
    text="📷 Cámara TALAT",
    width=600,
    height=450,
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
    font=("Segoe UI Emoji", 300)
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
    text="🧠 Modo Terapia",
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


btn_terapia = ctk.CTkButton(
    botones_modo,
    text="🧠 Modo Terapia",
    width=250,
    height=55,
    fg_color=AZUL,
    hover_color=MORADO,
    font=("Montserrat",18,"bold")
)

btn_terapia.grid(
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


btn_libre = ctk.CTkButton(
    botones_modo,
    text="🎛 Modo Libre",
    width=250,
    height=55,
    fg_color="#222222",
    hover_color=MORADO,
    font=("Montserrat",18,"bold")
)

btn_libre.grid(
    row=0,
    column=2,
    padx=20
)


# ==========================================
# PANEL DE CANCIÓN GUIADA
# ==========================================
#
# El panel tiene DOS vistas que se alternan:
#   1. vista_menu   -> elegir canción
#   2. vista_tocar  -> tocarla con los gestos
#
# Ocupa el mismo lugar que el panel de información del modo libre.

panel_cancion = ctk.CTkFrame(
    zona_sesion,
    fg_color="#1A1A1A",
    corner_radius=20,
    width=470
)

panel_cancion.pack_propagate(False)


# ------------------------------------------
# VISTA 1: MENÚ DE CANCIONES
# ------------------------------------------

vista_menu = ctk.CTkFrame(
    panel_cancion,
    fg_color="transparent"
)

ctk.CTkLabel(
    vista_menu,
    text="🎼 ELIGE UNA CANCIÓN",
    font=("Montserrat", 22, "bold"),
    text_color=BLANCO
).pack(pady=(18, 2))

ctk.CTkLabel(
    vista_menu,
    text="Cada gesto de tu cara toca un acorde completo.",
    font=("Montserrat", 14),
    text_color=GRIS,
    wraplength=410
).pack(pady=(0, 12))

lista_canciones = ctk.CTkScrollableFrame(
    vista_menu,
    fg_color="transparent"
)

lista_canciones.pack(fill="both", expand=True, padx=12)


# ------------------------------------------
# VISTA 2: PANTALLA DE TOCAR
# ------------------------------------------

vista_tocar = ctk.CTkFrame(
    panel_cancion,
    fg_color="transparent"
)

barra_superior = ctk.CTkFrame(
    vista_tocar,
    fg_color="transparent"
)

barra_superior.pack(fill="x", padx=15, pady=(12, 0))

cancion_titulo = ctk.CTkLabel(
    barra_superior,
    text="",
    font=("Montserrat", 18, "bold"),
    text_color=BLANCO,
    wraplength=300,
    justify="left"
)

cancion_titulo.pack(side="left")

# Letra de la frase y recuadros de acordes.

cancion_letra = ctk.CTkLabel(
    vista_tocar,
    text="",
    font=("Montserrat", 14),
    text_color=GRIS,
    wraplength=430
)

cancion_letra.pack(pady=(6, 6))

fila_acordes = ctk.CTkFrame(
    vista_tocar,
    fg_color="transparent"
)

fila_acordes.pack(pady=(0, 10))

# Guardamos los recuadros para repintarlos sin recrearlos.
chips_cancion = []
frase_dibujada = -1

# Acorde actual y gesto que hay que hacer.

zona_gesto = ctk.CTkFrame(
    vista_tocar,
    fg_color="#111111",
    corner_radius=14
)

zona_gesto.pack(fill="x", padx=18, pady=(0, 8))

cancion_acorde_actual = ctk.CTkLabel(
    zona_gesto,
    text="ACORDE: --",
    font=("Montserrat", 21, "bold"),
    text_color=AZUL
)

cancion_acorde_actual.pack(pady=(10, 0))

cancion_notas_acorde = ctk.CTkLabel(
    zona_gesto,
    text="",
    font=("Montserrat", 13),
    text_color=GRIS
)

cancion_notas_acorde.pack()

cancion_emoji = ctk.CTkLabel(
    zona_gesto,
    text="🎵",
    font=("Segoe UI Emoji", 80)
)

cancion_emoji.pack()

cancion_gesto_titulo = ctk.CTkLabel(
    zona_gesto,
    text="",
    font=("Montserrat", 16, "bold"),
    text_color=BLANCO
)

cancion_gesto_titulo.pack()

cancion_gesto_como = ctk.CTkLabel(
    zona_gesto,
    text="",
    font=("Montserrat", 13),
    text_color=GRIS,
    wraplength=390,
    justify="center"
)

cancion_gesto_como.pack(pady=(2, 6))

cancion_feedback = ctk.CTkLabel(
    zona_gesto,
    text="",
    font=("Montserrat", 14, "bold"),
    text_color=GRIS
)

cancion_feedback.pack(pady=(0, 10))

# Progreso.

cancion_progreso_texto = ctk.CTkLabel(
    vista_tocar,
    text="Progreso: ░░░░░░░░░░ 0%",
    font=("Consolas", 15, "bold"),
    text_color=GRIS
)

cancion_progreso_texto.pack(pady=(0, 4))

cancion_barra = ctk.CTkProgressBar(
    vista_tocar,
    height=14,
    corner_radius=7,
    progress_color=AZUL
)

cancion_barra.set(0)

cancion_barra.pack(fill="x", padx=25, pady=(0, 10))

# Botones de control.

controles_cancion = ctk.CTkFrame(
    vista_tocar,
    fg_color="transparent"
)

controles_cancion.pack(pady=(0, 12))


# ------------------------------------------
# FUNCIONES DEL PANEL
# ------------------------------------------

def texto_barra(proporcion, bloques=10):
    llenos = int(round(proporcion * bloques))
    return "█" * llenos + "░" * (bloques - llenos)

def construir_menu_canciones():
    """Crea una tarjeta por cada canción."""

    for hijo in lista_canciones.winfo_children():
        hijo.destroy()

    for clave, cancion in CANCIONES.items():

        tarjeta = ctk.CTkFrame(
            lista_canciones,
            fg_color="#111111",
            corner_radius=14
        )

        tarjeta.pack(fill="x", pady=6)

        ctk.CTkLabel(
            tarjeta,
            text=cancion["titulo"],
            font=("Montserrat", 17, "bold"),
            text_color=BLANCO,
            wraplength=360,
            justify="left"
        ).pack(
            anchor="w",
            padx=15,
            pady=(12, 0)
        )

        ctk.CTkLabel(
            tarjeta,
            text=cancion["subtitulo"],
            font=("Montserrat", 13),
            text_color=GRIS
        ).pack(
            anchor="w",
            padx=15
        )

        # Gestos que se necesitan.
        fila_gestos = ctk.CTkFrame(
            tarjeta,
            fg_color="transparent"
        )

        fila_gestos.pack(
            anchor="w",
            padx=15,
            pady=(6, 0)
        )

        for gesto in gestos_usados(cancion):

            guia = GUIA_DE_GESTOS.get(gesto, {})

            ctk.CTkLabel(
                fila_gestos,
                text=f"{guia.get('emoji', '')} {gesto}",
                font=("Montserrat", 13, "bold"),
                text_color=BLANCO,
                fg_color="#2B2B2B",
                corner_radius=8,
                width=120,
                height=32
            ).pack(
                side="left",
                padx=3
            )

        total_bloques = len(cancion["bloques"])

        ctk.CTkLabel(
            tarjeta,
            text=f"{total_bloques} bloques musicales · "
                 f"{total_bloques} gestos",
            font=("Montserrat", 12),
            text_color=GRIS
        ).pack(
            anchor="w",
            padx=15,
            pady=(6, 0)
        )

        ctk.CTkButton(
            tarjeta,
            text="▶ TOCAR ESTA CANCIÓN",
            height=38,
            fg_color=AZUL,
            hover_color=MORADO,
            font=("Montserrat", 14, "bold"),
            command=lambda c=clave: abrir_cancion(c)
        ).pack(
            fill="x",
            padx=15,
            pady=12
        )

def mostrar_menu_canciones():
    """Vuelve al menú de selección."""
    cancion_guiada.salir()

    vista_tocar.pack_forget()
    vista_menu.pack(fill="both", expand=True)


def abrir_cancion(clave):
    """Carga la canción elegida y muestra la pantalla de tocar."""
    global frase_dibujada

    if not cancion_guiada.cargar(clave):
        return

    frase_dibujada = -1

    vista_menu.pack_forget()
    vista_tocar.pack(fill="both", expand=True)

    dibujar_acordes_frase(forzar=True)
    actualizar_panel_cancion()


def refrescar_deteccion_cancion():
    """Solo actualiza el mensaje de lo que la cámara está viendo ahora."""
    if cancion_guiada.estado != "tocando":
        return

    gesto = cancion_guiada.gesto_detectado
    objetivo = cancion_guiada.gesto_actual()

    if gesto == "REPOSO":
        cancion_feedback.configure(
            text="Esperando tu gesto...",
            text_color=GRIS
        )
    elif cancion_guiada.esperando_reposo:
        cancion_feedback.configure(
            text="Relaja la cara para el siguiente acorde",
            text_color=AZUL
        )
    elif gesto != objetivo:
        acorde_equivocado = ACORDE_POR_GESTO.get(gesto)

        if acorde_equivocado:
            aviso = f"Eso es {acorde_equivocado}, no {cancion_guiada.acorde_actual()}"
        else:
            aviso = f"{gesto} no se usa en esta canción"

        cancion_feedback.configure(text=aviso, text_color="#FF9F43")

def dibujar_acordes_frase(forzar=False):
    """
    Muestra los acordes que forman el bloque actual.
    Todos pertenecen al mismo bloque musical.
    """

    if not cancion_guiada.cargada():
        return

    bloque = cancion_guiada.bloque_actual()

    if bloque is None:
        return

    for chip in chips_cancion:
        chip.destroy()

    chips_cancion.clear()

    for acorde in bloque["acordes"]:

        chip = ctk.CTkLabel(
            fila_acordes,
            text=acorde,
            width=70,
            height=40,
            corner_radius=10,
            fg_color="#2B2B2B",
            text_color=GRIS,
            font=("Montserrat", 14, "bold")
        )

        chip.pack(
            side="left",
            padx=3
        )

        chips_cancion.append(chip)

    # Mientras el bloque se reproduce,
    # todos sus acordes se muestran activos.
    if cancion_guiada.estado == "reproduciendo":

        for i, chip in enumerate(chips_cancion):

            acorde = bloque["acordes"][i]

            chip.configure(
                fg_color=ACORDES[acorde]["color"],
                text_color=NEGRO
            )

    elif cancion_guiada.estado in ("tocando", "pausada"):

        for chip in chips_cancion:
            chip.configure(
                fg_color=MORADO,
                text_color=BLANCO
            )

def actualizar_panel_cancion():
    """Actualiza la pantalla de Canción Guiada."""

    if not cancion_guiada.cargada():
        return

    cancion = cancion_guiada.cancion

    cancion_titulo.configure(
        text=cancion["titulo"]
    )

    total_bloques = cancion_guiada.total_bloques()

    if cancion_guiada.estado == "terminada":

        cancion_letra.configure(
            text="🎉 ¡Terminaste la canción!"
        )

        cancion_acorde_actual.configure(
            text="¡CANCIÓN COMPLETADA!",
            text_color="#FFD93D"
        )

        cancion_notas_acorde.configure(
            text=""
        )

        cancion_emoji.configure(
            text="🌟"
        )

        cancion_gesto_titulo.configure(
            text=f"Tocaste {total_bloques} bloques con {total_bloques} gestos",
            text_color=BLANCO
        )

        cancion_gesto_como.configure(
            text="Pulsa REINICIAR para tocarla otra vez."
        )

        cancion_feedback.configure(
            text=""
        )

        cancion_barra.set(1)
        cancion_progreso_texto.configure(
            text=f"Progreso: {texto_barra(1.0)} 100%"
        )

        cancion_barra.configure(
            progress_color="#FFD93D"
        )

        registrar_cancion_completada()

        return

    bloque = cancion_guiada.bloque_actual()

    if bloque is None:
        return

    numero = cancion_guiada.indice + 1

    cancion_letra.configure(
        text=(
            f"Bloque {numero} de {total_bloques}\n\n"
            f"{bloque['letra']}"
        )
    )

    dibujar_acordes_frase(forzar=True)

    proporcion = cancion_guiada.progreso()
    porcentaje = int(round(proporcion * 100))

    cancion_barra.set(proporcion)

    cancion_progreso_texto.configure(
        text=(
            f"Progreso: {texto_barra(proporcion)} "
            f"{porcentaje}%   "
            f"({numero}/{total_bloques})"
        )
    )

    gesto = cancion_guiada.gesto_actual()

    guia = GUIA_DE_GESTOS.get(
        gesto,
        {}
    )

    color_gesto = COLORES_POR_EMOCION.get(
        gesto,
        MORADO
    )

    cancion_emoji.configure(
        text=guia.get("emoji", "🎵")
    )

    if cancion_guiada.estado == "reproduciendo":

        cancion_acorde_actual.configure(
            text="🎵 TOCANDO EL BLOQUE...",
            text_color=AZUL
        )

        cancion_notas_acorde.configure(
            text="Puedes relajar tu rostro"
        )

        cancion_gesto_titulo.configure(
            text="Muy bien",
            text_color=AZUL
        )

        cancion_gesto_como.configure(
            text="El bloque está sonando. Descansa."
        )

        cancion_feedback.configure(
            text="🎶 Escucha y descansa...",
            text_color=AZUL
        )

    else:

        cancion_acorde_actual.configure(
            text=f"BLOQUE {numero}",
            text_color=color_gesto
        )

        cancion_notas_acorde.configure(
            text=" + ".join(bloque["acordes"])
        )

        cancion_gesto_titulo.configure(
            text=guia.get("titulo", ""),
            text_color=color_gesto
        )

        cancion_gesto_como.configure(
            text=guia.get("como", "")
        )

        if cancion_guiada.estado == "detenida":

            cancion_feedback.configure(
                text="Pulsa ▶ INICIAR",
                text_color=GRIS
            )

        elif cancion_guiada.estado == "pausada":

            cancion_feedback.configure(
                text="⏸ En pausa",
                text_color="#FF9F43"
            )

        elif cancion_guiada.esperando_reposo:

            cancion_feedback.configure(
                text="Relaja tu rostro para continuar",
                text_color=AZUL
            )

        else:

            cancion_feedback.configure(
                text="Haz el gesto indicado una vez",
                text_color=GRIS
            )

def registrar_cancion_completada():
    """Suma la canción al perfil del usuario, una sola vez."""
    if usuario_actual is None:
        return

    if cancion_guiada.ya_registrada == cancion_guiada.clave:
        return

    usuario = usuarios_db.setdefault(usuario_actual, crear_datos_usuario())
    usuario["canciones_completadas"] = usuario.get("canciones_completadas", 0) + 1

    guardar_usuarios(usuarios_db)

    cancion_guiada.ya_registrada = cancion_guiada.clave


def iniciar_cancion():
    cancion_guiada.iniciar()
    actualizar_panel_cancion()


def pausar_cancion():
    cancion_guiada.pausar()
    actualizar_panel_cancion()


def reiniciar_cancion():
    cancion_guiada.reiniciar()
    dibujar_acordes_frase(forzar=True)
    actualizar_panel_cancion()



# ==========================================
# PANEL DEL MODO LIBRE
# ==========================================

panel_libre = ctk.CTkFrame(
    zona_sesion,
    fg_color="#1A1A1A",
    corner_radius=20,
    width=470
)

panel_libre.pack_propagate(False)

ctk.CTkLabel(
    panel_libre,
    text="🎛 MODO LIBRE",
    font=("Montserrat", 22, "bold"),
    text_color=BLANCO
).pack(pady=(14, 0))

lista_libre = ctk.CTkScrollableFrame(
    panel_libre,
    fg_color="transparent"
)

lista_libre.pack(fill="both", expand=True, padx=10)

libre_feedback = ctk.CTkLabel(
    panel_libre,
    text="Agrega un sonido y graba su gesto.",
    font=("Montserrat", 14, "bold"),
    text_color=GRIS,
    wraplength=420
)

libre_feedback.pack(pady=(6, 4))

botones_libre = ctk.CTkFrame(panel_libre, fg_color="transparent")
botones_libre.pack(fill="x", padx=14, pady=(0, 12))


def dibujar_lista_libre():
    """Repinta la lista completa de sonidos y gestos."""
    for hijo in lista_libre.winfo_children():
        hijo.destroy()

    if not instrumento_libre.ranuras:
        ctk.CTkLabel(
            lista_libre,
            text="Todavía no hay sonidos.\nPulsa «Agregar sonido» para empezar.",
            font=("Montserrat", 14),
            text_color=GRIS,
            justify="center"
        ).pack(pady=40)
        return

    for indice, ranura in enumerate(instrumento_libre.ranuras):

        fila = ctk.CTkFrame(lista_libre, fg_color="#111111", corner_radius=12)
        fila.pack(fill="x", pady=4)

        arriba = ctk.CTkFrame(fila, fg_color="transparent")
        arriba.pack(fill="x", padx=10, pady=(8, 2))

        selector = ctk.CTkOptionMenu(
            arriba,
            values=list(ETIQUETA_A_SONIDO.keys()),
            font=("Montserrat", 12),
            fg_color="#2B2B2B",
            button_color=AZUL,
            button_hover_color=MORADO,
            dynamic_resizing=False,
            width=145,
            command=lambda etiqueta, i=indice: cambiar_sonido_libre(i, etiqueta)
        )

        selector.set(SONIDO_A_ETIQUETA.get(ranura["sonido"], "🎵 DO"))
        selector.pack(side="left")

        ctk.CTkButton(
            arriba,
            text="🎬 Grabar" if ranura["gesto"] is None else "🔄 Regrabar",
            width=95,
            height=50,
            fg_color=AZUL if ranura["gesto"] is None else "#2B2B2B",
            hover_color=MORADO,
            font=("Montserrat", 12, "bold"),
            command=lambda i=indice: iniciar_grabacion_gesto(i)
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            arriba,
            text="🗑",
            width=34,
            height=50,
            fg_color="#3A3A3A",
            hover_color="#B22222",
            font=("Montserrat", 12, "bold"),
            command=lambda i=indice: quitar_sonido_libre(i)
        ).pack(side="left", padx=2)

        if ranura["gesto"] is None:
            texto = "Sin gesto grabado"
            color = "#FF9F43"
        else:
            texto = f"Tu gesto:  {ranura['texto']}"
            color = AZUL

        ctk.CTkLabel(
            fila,
            text=texto,
            font=("Montserrat", 12),
            text_color=color,
            anchor="w"
        ).pack(anchor="w", padx=14, pady=(0, 8))


def cambiar_sonido_libre(indice, etiqueta):
    instrumento_libre.cambiar_sonido(indice, ETIQUETA_A_SONIDO.get(etiqueta, "DO"))
    guardar_instrumento_libre()


def agregar_sonido_libre():
    if len(instrumento_libre.ranuras) >= 8:
        libre_feedback.configure(
            text="Ocho sonidos es el máximo.",
            text_color="#FF9F43"
        )
        return

    # Proponemos el siguiente sonido que no esté usado.
    usados = {r["sonido"] for r in instrumento_libre.ranuras}

    siguiente = "DO"

    for clave in SONIDOS_LIBRES:
        if clave not in usados:
            siguiente = clave
            break

    instrumento_libre.agregar(siguiente)

    dibujar_lista_libre()
    guardar_instrumento_libre()


def quitar_sonido_libre(indice):
    instrumento_libre.quitar(indice)
    dibujar_lista_libre()
    guardar_instrumento_libre()



def borrar_gesto_libre(indice):
    instrumento_libre.borrar_gesto(indice)
    dibujar_lista_libre()
    guardar_instrumento_libre()


def guardar_instrumento_libre():
    """Guarda el instrumento en el perfil, para no regrabarlo cada sesión."""
    if usuario_actual is None:
        return

    usuario = usuarios_db.get(usuario_actual)

    if usuario is None:
        return

    usuario["instrumento_libre"] = instrumento_libre.exportar()

    guardar_usuarios(usuarios_db)


# ------------------------------------------
# GRABAR UN GESTO
# ------------------------------------------

def iniciar_grabacion_gesto(indice):
    """Cuenta regresiva y después captura el gesto."""
    global grabando_indice
    global grabando_muestras
    global cuenta_regresiva_libre

    if not detector.calibrado:
        libre_feedback.configure(
            text="Espera a que termine la calibración.",
            text_color="#FF9F43"
        )
        return

    grabando_indice = None
    grabando_muestras = []
    cuenta_regresiva_libre = SEGUNDOS_CUENTA_REGRESIVA

    contar_para_grabar(indice)


def contar_para_grabar(indice):
    """
    Cuenta 3, 2, 1 antes de capturar.

    Sin esta pausa se grababa el momento en que la persona todavía
    estaba soltando el mouse, y el gesto salía a medias.
    """
    global grabando_indice
    global cuenta_regresiva_libre

    if cuenta_regresiva_libre > 0:
        libre_feedback.configure(
            text=f"Prepara tu gesto...  {cuenta_regresiva_libre}",
            text_color=MORADO
        )

        cuenta_regresiva_libre -= 1

        app.after(1000, lambda: contar_para_grabar(indice))
        return

    grabando_indice = indice

    libre_feedback.configure(
        text="🎬 ¡Mantén el gesto!",
        text_color=AZUL
    )


def guardar_muestra_gesto(medicion):
    """Va juntando lecturas mientras dura la grabación."""
    global grabando_muestras

    diferencias = diferencias_contra_reposo(medicion)

    if diferencias is None:
        return

    grabando_muestras.append(diferencias)

    if len(grabando_muestras) >= TOTAL_MUESTRAS_MOVIMIENTO_LIBRE:
        terminar_grabacion_gesto()


def terminar_grabacion_gesto():
    """Promedia las lecturas, revisa que el gesto sirva y lo guarda."""
    global grabando_indice
    global grabando_muestras

    indice = grabando_indice
    muestras = grabando_muestras

    grabando_indice = None
    grabando_muestras = []

    if indice is None or not muestras:
        return

    promedio = {
        clave: sum(m[clave] for m in muestras) / len(muestras)
        for clave in CLAVES_MODO_LIBRE
    }

    sirve, motivo = instrumento_libre.revisar_gesto(promedio, indice)

    if not sirve:
        libre_feedback.configure(text=f"✖ {motivo}", text_color="#FF9F43")
        return

    instrumento_libre.guardar_gesto(indice, promedio)

    dibujar_lista_libre()
    guardar_instrumento_libre()

    libre_feedback.configure(
        text=f"✓ Guardado: {instrumento_libre.ranuras[indice]['texto']}",
        text_color=AZUL
    )


# ------------------------------------------
# TOCAR
# ------------------------------------------

def procesar_modo_libre(medicion):
    """Se llama en cada frame mientras el Modo Libre está activo."""
    if grabando_indice is not None:
        guardar_muestra_gesto(medicion)
        return

    indice = instrumento_libre.reconocer(medicion)

    # Volvió al reposo: se libera el sonido para poder repetirlo.
    if indice is None:
        if instrumento_libre.ultimo_sonando is not None:
            libre_feedback.configure(

            )

        instrumento_libre.ultimo_sonando = None
        return

    # Mientras se sostenga el mismo gesto, no se repite el sonido.
    if indice == instrumento_libre.ultimo_sonando:
        return

    instrumento_libre.ultimo_sonando = indice

    ranura = instrumento_libre.ranuras[indice]
    sonido = SONIDOS_LIBRES[ranura["sonido"]]

    motor_notas.tocar_notas(sonido["notas"])

    libre_feedback.configure(
        text=f"{sonido['etiqueta']}   ←   {ranura['texto']}",
        text_color=AZUL
    )

    if "estado_nota" in globals():
        estado_nota.configure(text=sonido["etiqueta"], text_color=AZUL)


ctk.CTkButton(
    botones_libre,
    text="➕ Agregar sonido",
    height=34,
    fg_color="#2B2B2B",
    hover_color=MORADO,
    font=("Montserrat", 13, "bold"),
    command=agregar_sonido_libre
).pack(side="left", expand=True, fill="x", padx=(0, 4))

dibujar_lista_libre()


# ------------------------------------------
# CAMBIO ENTRE MODOS
# ------------------------------------------

def activar_modo_terapia():
    global modo_sesion

    modo_sesion = "terapia"

    cancion_guiada.salir()

    panel_cancion.pack_forget()
    panel_libre.pack_forget()
    info_sesion.pack(side="right", fill="both", padx=20)

    btn_terapia.configure(fg_color=AZUL)
    btn_cancion.configure(fg_color="#222222")
    btn_libre.configure(fg_color="#222222")

    modo_actual.configure(text="🧠 Modo Terapia")

    instruccion_label.configure(
        text="INSTRUCCIÓN: Haz una expresión clara y mantenla unos instantes."
    )


def activar_modo_cancion():
    global modo_sesion

    modo_sesion = "cancion"

    info_sesion.pack_forget()
    panel_libre.pack_forget()
    panel_cancion.pack(side="right", fill="both", padx=20)

    btn_terapia.configure(fg_color="#222222")
    btn_cancion.configure(fg_color=AZUL)
    btn_libre.configure(fg_color="#222222")

    modo_actual.configure(text="🎼 Canción Guiada")

    # Siempre se entra por el menú de selección.
    mostrar_menu_canciones()


def activar_modo_libre():
    global modo_sesion
    global grabando_indice
    global grabando_muestras

    modo_sesion = "libre"

    grabando_indice = None
    grabando_muestras = []

    cancion_guiada.salir()
    instrumento_libre.limpiar_historial()

    info_sesion.pack_forget()
    panel_cancion.pack_forget()
    panel_libre.pack(side="right", fill="both", padx=20)

    btn_terapia.configure(fg_color="#222222")
    btn_cancion.configure(fg_color="#222222")
    btn_libre.configure(fg_color=AZUL)

    modo_actual.configure(text="🎛 Modo Libre")

    # Recuperamos el instrumento que esta persona ya había armado.
    if usuario_actual is not None:
        guardado = usuarios_db.get(usuario_actual, {}).get("instrumento_libre")

        if guardado:
            instrumento_libre.importar(guardado)

    dibujar_lista_libre()

    if not detector.calibrado:
        libre_feedback.configure(
            text="Espera a que termine la calibración.",
            text_color="#FF9F43"
        )


btn_terapia.configure(command=activar_modo_terapia)
btn_cancion.configure(command=activar_modo_cancion)
btn_libre.configure(command=activar_modo_libre)

# Botón para volver al menú desde la pantalla de tocar.
ctk.CTkButton(
    barra_superior,
    text="← Canciones",
    width=110,
    height=30,
    fg_color="#2B2B2B",
    hover_color=MORADO,
    font=("Montserrat", 12, "bold"),
    command=mostrar_menu_canciones
).pack(side="right")

ctk.CTkButton(
    controles_cancion,
    text="▶ INICIAR",
    width=125,
    height=40,
    fg_color=AZUL,
    hover_color=MORADO,
    font=("Montserrat", 15, "bold"),
    command=iniciar_cancion
).grid(row=0, column=0, padx=5)

ctk.CTkButton(
    controles_cancion,
    text="⏸ PAUSA",
    width=125,
    height=40,
    fg_color="#2B2B2B",
    hover_color=MORADO,
    font=("Montserrat", 15, "bold"),
    command=pausar_cancion
).grid(row=0, column=1, padx=5)

ctk.CTkButton(
    controles_cancion,
    text="↻ REINICIAR",
    width=125,
    height=40,
    fg_color="#2B2B2B",
    hover_color=MORADO,
    font=("Montserrat", 15, "bold"),
    command=reiniciar_cancion
).grid(row=0, column=2, padx=5)


construir_menu_canciones()


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

# ==========================================
# LECTURA HUMANA DE LAS MEDICIONES
# ==========================================
#
# El detector guarda números normalizados (0.03421, 0.58810...).
# Esos números son correctos, pero nadie los entiende a simple vista.
# Aquí los traducimos a palabras y a barras de colores.
#
#   "escala"  -> rango normal esperado, sirve para llenar la barra
#   "niveles" -> palabras de MENOR a MAYOR valor

ORDEN_MEDICIONES = [
    "apertura_boca",
    "ancho_boca",
    "curva_boca",
    "apertura_ojos",
    "altura_ceja",
    "cercania_cejas"
]

MEDICIONES_LEGIBLES = {
    "apertura_boca": {
        "titulo": "Boca abierta",
        "icono": "👄",
        "explicacion": "Qué tanto separaste los labios.",
        "escala": (0.000, 0.140),
        "niveles": ["Cerrada", "Apenas abierta", "Abierta", "Muy abierta"]
    },
    "ancho_boca": {
        "titulo": "Boca estirada",
        "icono": "😬",
        "explicacion": "Qué tanto estiraste la boca hacia los lados.",
        "escala": (0.450, 0.850),
        "niveles": ["Recogida", "Normal", "Estirada", "Muy estirada"]
    },
    "curva_boca": {
        "titulo": "Forma de la sonrisa",
        "icono": "🙂",
        "explicacion": "Hacia dónde apuntan las esquinas de la boca.",
        "escala": (-0.020, 0.020),
        "niveles": ["Muy sonriente", "Sonriente", "Recta", "Hacia abajo"]
    },
    "apertura_ojos": {
        "titulo": "Ojos abiertos",
        "icono": "👁",
        "explicacion": "Qué tanto abriste los ojos.",
        "escala": (0.010, 0.060),
        "niveles": ["Casi cerrados", "Entrecerrados", "Normales", "Muy abiertos"]
    },
    "altura_ceja": {
        "titulo": "Cejas levantadas",
        "icono": "🤨",
        "explicacion": "Qué tanto subiste las cejas.",
        "escala": (0.040, 0.120),
        "niveles": ["Muy bajas", "Bajas", "Normales", "Levantadas"]
    },
    "cercania_cejas": {
        "titulo": "Separación de cejas",
        "icono": "😠",
        "explicacion": "Qué tanto juntaste las cejas (el ceño).",
        "escala": (0.150, 0.450),
        "niveles": ["Muy juntas", "Juntas", "Normales", "Separadas"]
    }
}

NOTAS_POR_EMOCION = {
    "ABURRIMIENTO": "DO",
    "SORPRESA": "RE",
    "IRA": "MI",
    "TRISTEZA": "FA",
    "ALEGRÍA": "SOL"
}

COLORES_POR_EMOCION = {
    "ABURRIMIENTO": MORADO2,
    "SORPRESA": AZUL,
    "IRA": ROJO,
    "TRISTEZA": CELESTE,
    "ALEGRÍA": AMARILLO
}


def escalas_del_usuario(historial):
    """
    Ajusta las barras al rostro de cada persona.

    Una boca grande y una boca pequeña dan números distintos aunque
    hagan la misma expresión. Si el usuario ya tiene suficientes
    mediciones, usamos SU propio rango; si no, usamos el rango general.
    """
    valores = {clave: [] for clave in ORDEN_MEDICIONES}

    for sesion in historial:
        for registro in sesion.get("expresiones", []):
            for clave in ORDEN_MEDICIONES:
                if clave in registro:
                    valores[clave].append(registro[clave])

    escalas = {}

    for clave in ORDEN_MEDICIONES:
        lista = valores[clave]
        escalas[clave] = MEDICIONES_LEGIBLES[clave]["escala"]

        if len(lista) >= 5:
            minimo = min(lista)
            maximo = max(lista)
            if maximo - minimo > 1e-6:
                margen = (maximo - minimo) * 0.25
                escalas[clave] = (minimo - margen, maximo + margen)

    return escalas


def porcentaje_medicion(clave, valor, escalas=None):
    """Convierte el número crudo en un valor de 0.0 a 1.0 para la barra."""
    info = MEDICIONES_LEGIBLES.get(clave)

    if info is None:
        return 0.0

    if escalas and clave in escalas:
        minimo, maximo = escalas[clave]
    else:
        minimo, maximo = info["escala"]

    if maximo - minimo < 1e-9:
        return 0.0

    proporcion = (valor - minimo) / (maximo - minimo)

    return max(0.0, min(1.0, proporcion))


def etiqueta_medicion(clave, proporcion):
    """Devuelve la palabra que describe el nivel de la medición."""
    info = MEDICIONES_LEGIBLES.get(clave)

    if info is None:
        return ""

    niveles = info["niveles"]
    indice = int(proporcion * len(niveles))

    return niveles[min(indice, len(niveles) - 1)]


def color_medicion(proporcion):
    if proporcion >= 0.75:
        return MORADO
    if proporcion >= 0.35:
        return AZUL
    return "#4F6B85"


def comparar_humano(clave, actual, anterior, escalas=None):
    """Explica con palabras si la persona hizo más o menos que la vez pasada."""
    if anterior is None:
        return "Primera vez que se mide esta expresión.", GRIS

    info = MEDICIONES_LEGIBLES.get(clave)

    if info is None:
        return "", GRIS

    if escalas and clave in escalas:
        minimo, maximo = escalas[clave]
    else:
        minimo, maximo = info["escala"]

    rango = max(maximo - minimo, 1e-9)
    diferencia = (actual - anterior) / rango

    if abs(diferencia) < 0.05:
        return "Casi igual que la vez anterior.", GRIS

    palabra = "más" if diferencia > 0 else "menos"
    intensidad = "bastante" if abs(diferencia) > 0.20 else "un poco"
    color = AZUL if diferencia > 0 else MORADO

    return f"{intensidad.capitalize()} {palabra} que la vez anterior.", color


def barra_medicion(padre, clave, valor, escalas=None, mostrar_numero=False):
    """Dibuja una fila: icono + título + palabra + barra de progreso."""
    info = MEDICIONES_LEGIBLES.get(clave)

    if info is None:
        return

    proporcion = porcentaje_medicion(clave, valor, escalas)
    palabra = etiqueta_medicion(clave, proporcion)
    color = color_medicion(proporcion)

    fila = ctk.CTkFrame(padre, fg_color="transparent")
    fila.pack(fill="x", padx=15, pady=5)

    encabezado = ctk.CTkFrame(fila, fg_color="transparent")
    encabezado.pack(fill="x")

    ctk.CTkLabel(
        encabezado,
        text=f"{info['icono']}  {info['titulo']}",
        font=("Montserrat", 15),
        text_color=BLANCO,
        anchor="w"
    ).pack(side="left")

    texto_derecha = palabra

    if mostrar_numero:
        texto_derecha = f"{palabra}   ({valor:.5f})"

    ctk.CTkLabel(
        encabezado,
        text=texto_derecha,
        font=("Montserrat", 15, "bold"),
        text_color=color,
        anchor="e"
    ).pack(side="right")

    # Barra: un riel gris y un relleno de color encima.
    riel = ctk.CTkFrame(
        fila,
        fg_color="#2B2B2B",
        height=12,
        corner_radius=6
    )
    riel.pack(fill="x", pady=(4, 0))
    riel.pack_propagate(False)

    relleno = ctk.CTkFrame(
        riel,
        fg_color=color,
        corner_radius=6
    )
    relleno.place(
        relx=0,
        rely=0,
        relwidth=max(proporcion, 0.02),
        relheight=1
    )


# ==========================================
# GRÁFICA DE PROGRESO
# ==========================================

metrica_grafica = "Expresiones"


def datos_grafica(datos, metrica, maximo_sesiones=10):
    """Arma la lista de (etiqueta, valor) de las últimas sesiones."""
    historial = datos.get("historial_sesiones", [])
    puntos = []

    for sesion in historial:
        expresiones = sesion.get("expresiones", [])

        if metrica == "Expresiones":
            valor = sesion.get("total_expresiones", len(expresiones))
        else:
            valor = round(sesion.get("duracion_segundos", 0) / 60, 1)

        fecha = sesion.get("fecha", "")
        etiqueta = fecha[5:10] if len(fecha) >= 10 else "--"

        puntos.append((etiqueta, valor))

    return puntos[-maximo_sesiones:]


def dibujar_grafica_progreso(event=None):
    """Dibuja la gráfica de barras a mano sobre un canvas (ligero para la Raspberry)."""
    if "grafica_canvas" not in globals():
        return

    canvas = grafica_canvas
    canvas.delete("all")

    ancho = canvas.winfo_width()
    alto = canvas.winfo_height()

    if ancho < 60:
        ancho = 780
    if alto < 60:
        alto = 240

    if usuario_actual is None:
        return

    datos = usuarios_db.get(usuario_actual, {})
    puntos = datos_grafica(datos, metrica_grafica)

    if not puntos:
        canvas.create_text(
            ancho / 2,
            alto / 2,
            text="Aún no hay sesiones.\nInicia una sesión para ver tu progreso.",
            fill=GRIS,
            font=("Montserrat", 15),
            justify="center"
        )
        return

    margen_izq = 55
    margen_der = 20
    margen_sup = 25
    margen_inf = 45

    area_ancho = ancho - margen_izq - margen_der
    area_alto = alto - margen_sup - margen_inf

    if area_ancho <= 10 or area_alto <= 10:
        return

    valores = [valor for _, valor in puntos]
    maximo = max(valores)

    if maximo <= 0:
        maximo = 1

    # Líneas guía horizontales y sus números.
    for i in range(4):
        proporcion = i / 3
        y = margen_sup + area_alto - (area_alto * proporcion)

        canvas.create_line(
            margen_izq,
            y,
            margen_izq + area_ancho,
            y,
            fill="#2B2B2B"
        )

        canvas.create_text(
            margen_izq - 10,
            y,
            text=f"{maximo * proporcion:.0f}",
            fill=GRIS,
            font=("Montserrat", 10),
            anchor="e"
        )

    # Promedio, como referencia de progreso.
    promedio = sum(valores) / len(valores)
    y_promedio = margen_sup + area_alto - (promedio / maximo) * area_alto

    canvas.create_line(
        margen_izq,
        y_promedio,
        margen_izq + area_ancho,
        y_promedio,
        fill=MORADO,
        dash=(5, 4)
    )

    canvas.create_text(
        margen_izq + area_ancho,
        y_promedio - 9,
        text=f"promedio {promedio:.1f}",
        fill=MORADO,
        font=("Montserrat", 10),
        anchor="e"
    )

    # Barras.
    paso = area_ancho / len(puntos)
    ancho_barra = min(paso * 0.55, 55)

    for indice, (etiqueta, valor) in enumerate(puntos):
        centro = margen_izq + paso * (indice + 0.5)
        altura = (valor / maximo) * area_alto

        if altura < 3 and valor > 0:
            altura = 3

        x0 = centro - ancho_barra / 2
        x1 = centro + ancho_barra / 2
        y0 = margen_sup + area_alto - altura
        y1 = margen_sup + area_alto

        # La última sesión se resalta en morado.
        color = MORADO if indice == len(puntos) - 1 else AZUL

        canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            fill=color,
            outline=""
        )

        texto_valor = f"{valor:g}"

        canvas.create_text(
            centro,
            y0 - 10,
            text=texto_valor,
            fill=BLANCO,
            font=("Montserrat", 11, "bold")
        )

        canvas.create_text(
            centro,
            y1 + 15,
            text=etiqueta,
            fill=GRIS,
            font=("Montserrat", 10)
        )

    # Eje inferior.
    canvas.create_line(
        margen_izq,
        margen_sup + area_alto,
        margen_izq + area_ancho,
        margen_sup + area_alto,
        fill="#3A3A3A"
    )

    unidad = {
        "Expresiones": "expresiones registradas por sesión",
        "Minutos": "minutos de práctica por sesión"
    }.get(metrica_grafica, "")

    canvas.create_text(
        margen_izq,
        alto - 10,
        text=unidad,
        fill=GRIS,
        font=("Montserrat", 11),
        anchor="w"
    )


def cambiar_metrica_grafica(valor):
    global metrica_grafica
    metrica_grafica = valor
    dibujar_grafica_progreso()


def actualizar_estadisticas_perfil():

    if usuario_actual is None:
        return

    datos = usuarios_db.get(usuario_actual)

    if datos is None:
        return

    edad = str(datos.get("edad", "")).strip()

    edad_label.configure(
        text=f"Edad: {edad} años" if edad else "Edad: sin registrar"
    )

    motivo = str(datos.get("motivo", "")).strip()

    motivo_label.configure(
        text=motivo if motivo else "Sin registrar",
        text_color=BLANCO if motivo else GRIS
    )

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

    # La gráfica se vuelve a dibujar con los datos recién guardados.
    dibujar_grafica_progreso()


# Qué mirar en cada expresión. Mostrar las seis mediciones en todas
# las expresiones confundía: para la alegría lo que importa es la
# sonrisa, no la separación de las cejas.
LO_QUE_IMPORTA = {
    "ALEGRÍA": ["curva_boca", "ancho_boca"],
    "SORPRESA": ["apertura_boca", "altura_ceja"],
    "IRA": ["cercania_cejas", "altura_ceja"],
    "TRISTEZA": ["curva_boca", "apertura_boca"],
    "ABURRIMIENTO": ["apertura_ojos", "altura_ceja"]
}

# Cómo se lee cada medición dentro de cada expresión, en palabras que
# cualquiera entiende. El orden va de menos a más logrado.
LOGRO_POR_MEDICION = {
    "curva_boca": {
        "pregunta": "¿Levantaste las esquinas de la boca?",
        "invertida": True,
        "niveles": ["Casi nada", "Un poco", "Bastante", "Mucho"]
    },
    "ancho_boca": {
        "pregunta": "¿Estiraste la boca hacia los lados?",
        "invertida": False,
        "niveles": ["Casi nada", "Un poco", "Bastante", "Mucho"]
    },
    "apertura_boca": {
        "pregunta": "¿Abriste la boca?",
        "invertida": False,
        "niveles": ["Casi nada", "Un poco", "Bastante", "Mucho"]
    },
    "altura_ceja": {
        "pregunta": "¿Levantaste las cejas?",
        "invertida": False,
        "niveles": ["Casi nada", "Un poco", "Bastante", "Mucho"]
    },
    "cercania_cejas": {
        "pregunta": "¿Juntaste las cejas?",
        "invertida": True,
        "niveles": ["Casi nada", "Un poco", "Bastante", "Mucho"]
    },
    "apertura_ojos": {
        "pregunta": "¿Cerraste los ojos?",
        "invertida": True,
        "niveles": ["Casi nada", "Un poco", "Bastante", "Mucho"]
    }
}


def nivel_logro(clave, valor, escalas):
    """
    Convierte la medición en una barra de 0 a 1 y una palabra.

    'invertida' quiere decir que el número BAJA cuando la persona hace
    más el movimiento (cerrar los ojos, fruncir el ceño, sonreír).
    """
    info = LOGRO_POR_MEDICION.get(clave)

    if info is None:
        return 0.0, ""

    proporcion = porcentaje_medicion(clave, valor, escalas)

    if info["invertida"]:
        proporcion = 1.0 - proporcion

    niveles = info["niveles"]
    indice = min(int(proporcion * len(niveles)), len(niveles) - 1)

    return proporcion, niveles[indice]


def barra_logro(padre, clave, valor, escalas):
    """Una pregunta en lenguaje normal, la respuesta y una barra."""
    info = LOGRO_POR_MEDICION.get(clave)

    if info is None:
        return

    proporcion, palabra = nivel_logro(clave, valor, escalas)

    if proporcion >= 0.66:
        color = "#4CF405"
    elif proporcion >= 0.33:
        color = AZUL
    else:
        color = "#FF9F43"

    fila = ctk.CTkFrame(padre, fg_color="transparent")
    fila.pack(fill="x", padx=18, pady=5)

    encabezado = ctk.CTkFrame(fila, fg_color="transparent")
    encabezado.pack(fill="x")

    ctk.CTkLabel(
        encabezado,
        text=info["pregunta"],
        font=("Montserrat", 16),
        text_color=BLANCO,
        anchor="w"
    ).pack(side="left")

    ctk.CTkLabel(
        encabezado,
        text=palabra,
        font=("Montserrat", 16, "bold"),
        text_color=color,
        anchor="e"
    ).pack(side="right")

    riel = ctk.CTkFrame(fila, fg_color="#2B2B2B", height=14, corner_radius=7)
    riel.pack(fill="x", pady=(4, 0))
    riel.pack_propagate(False)

    relleno = ctk.CTkFrame(riel, fg_color=color, corner_radius=7)
    relleno.place(relx=0, rely=0, relwidth=max(proporcion, 0.03), relheight=1)


def mostrar_detalle_estadisticas():
    """
    Detalle de las sesiones, en lenguaje de todos los días.

    Antes había una "vista técnica" con números de cinco decimales.
    Eso servía para depurar el detector, no para una persona en terapia
    ni para su familia, así que se quitó por completo.
    """
    if usuario_actual is None:
        return

    datos = usuarios_db.get(usuario_actual, {})
    historial = datos.get("historial_sesiones", [])
    escalas = escalas_del_usuario(historial)

    ventana = ctk.CTkToplevel(app)
    ventana.title(f"Cómo te ha ido - {usuario_actual}")
    ventana.geometry("900x700")
    ventana.configure(fg_color=NEGRO)
    ventana.transient(app)
    ventana.grab_set()

    ctk.CTkLabel(
        ventana,
        text=f"Cómo le ha ido a {usuario_actual}",
        font=("Montserrat", 28, "bold"),
        text_color=BLANCO
    ).pack(pady=(20, 4))

    ctk.CTkLabel(
        ventana,
        text="Cada barra responde una pregunta sencilla sobre lo que hizo "
             "tu cara. Entre más llena y más verde, más marcado te salió "
             "el movimiento.",
        font=("Montserrat", 15),
        text_color=GRIS,
        wraplength=780,
        justify="center"
    ).pack(pady=(0, 12), padx=25)

    scroll = ctk.CTkScrollableFrame(ventana, fg_color=NEGRO)
    scroll.pack(fill="both", expand=True, padx=25, pady=(0, 12))

    if not historial:
        ctk.CTkLabel(
            scroll,
            text="Todavía no hay sesiones.\n\nInicia una sesión y haz "
                 "algunas expresiones frente a la cámara.",
            font=("Montserrat", 19),
            text_color=GRIS,
            justify="center"
        ).pack(pady=110)
    else:
        total_sesiones = len(historial)

        for numero, sesion in enumerate(reversed(historial), start=1):

            indice_real = total_sesiones - numero + 1

            bloque = ctk.CTkFrame(scroll, fg_color="#1A1A1A", corner_radius=18)
            bloque.pack(fill="x", pady=10, padx=5)

            expresiones = sesion.get("expresiones", [])

            minutos = sesion.get("duracion_segundos", 0) // 60
            segundos = sesion.get("duracion_segundos", 0) % 60

            if minutos > 0:
                duracion_texto = f"{minutos} min {segundos} s"
            else:
                duracion_texto = f"{segundos} s"

            ctk.CTkLabel(
                bloque,
                text=f"Sesión {indice_real}   ·   {sesion.get('fecha', 'Sin fecha')}",
                font=("Montserrat", 21, "bold"),
                text_color=BLANCO
            ).pack(anchor="w", padx=20, pady=(15, 2))

            if not expresiones:
                ctk.CTkLabel(
                    bloque,
                    text=f"Practicaste {duracion_texto}, pero la cámara no "
                         f"alcanzó a registrar ninguna expresión.\n"
                         f"Consejo: mantén cada gesto un par de segundos.",
                    font=("Montserrat", 16),
                    text_color=GRIS,
                    wraplength=760,
                    justify="left"
                ).pack(anchor="w", padx=20, pady=(0, 16))
                continue

            conteo = {}

            for registro in expresiones:
                nombre = registro.get("emocion", "DESCONOCIDA")
                conteo[nombre] = conteo.get(nombre, 0) + 1

            favorita = max(conteo.items(), key=lambda par: par[1])

            ctk.CTkLabel(
                bloque,
                text=f"Practicaste {duracion_texto} y lograste "
                     f"{len(expresiones)} expresiones de {len(conteo)} tipos. "
                     f"La que más te salió fue {favorita[0].lower()}, "
                     f"{favorita[1]} veces.",
                font=("Montserrat", 16),
                text_color=GRIS,
                wraplength=760,
                justify="left"
            ).pack(anchor="w", padx=20, pady=(0, 10))

            # Agrupamos por expresión, conservando el orden en que ocurrieron.
            por_expresion = {}

            for registro in expresiones:
                por_expresion.setdefault(
                    registro.get("emocion", "DESCONOCIDA"), []
                ).append(registro)

            for emocion, registros in por_expresion.items():

                tarjeta = ctk.CTkFrame(bloque, fg_color="#111111", corner_radius=14)
                tarjeta.pack(fill="x", padx=20, pady=8)

                cabecera = ctk.CTkFrame(tarjeta, fg_color="transparent")
                cabecera.pack(fill="x", padx=18, pady=(12, 6))

                veces = "vez" if len(registros) == 1 else "veces"

                ctk.CTkLabel(
                    cabecera,
                    text=f"{registros[0].get('emoji', '')}  {emocion.capitalize()}",
                    font=("Montserrat", 20, "bold"),
                    text_color=COLORES_POR_EMOCION.get(emocion, AZUL)
                ).pack(side="left")

                ctk.CTkLabel(
                    cabecera,
                    text=f"{len(registros)} {veces}",
                    font=("Montserrat", 16),
                    text_color=GRIS
                ).pack(side="right")

                ultimo = registros[-1]

                for clave in LO_QUE_IMPORTA.get(emocion, ["apertura_boca"]):
                    if clave in ultimo:
                        barra_logro(tarjeta, clave, ultimo[clave], escalas)

                # Comparación con la vez anterior, en una frase.
                if len(registros) > 1:
                    claves = LO_QUE_IMPORTA.get(emocion, ["apertura_boca"])
                    clave = claves[0]

                    if clave in ultimo and clave in registros[-2]:
                        anterior_prop, _ = nivel_logro(
                            clave, registros[-2][clave], escalas
                        )
                        ultima_prop, _ = nivel_logro(
                            clave, ultimo[clave], escalas
                        )

                        cambio = ultima_prop - anterior_prop

                        if abs(cambio) < 0.05:
                            frase = "Te salió igual que la vez anterior."
                            color = GRIS
                        elif cambio > 0:
                            frase = "Te salió mejor que la vez anterior."
                            color = "#4CF405"
                        else:
                            frase = "Esta vez te salió un poco menos marcado."
                            color = "#FF9F43"

                        ctk.CTkLabel(
                            tarjeta,
                            text=frase,
                            font=("Montserrat", 15),
                            text_color=color
                        ).pack(anchor="w", padx=18, pady=(6, 12))
                else:
                    ctk.CTkLabel(
                        tarjeta,
                        text="Primera vez que haces esta expresión.",
                        font=("Montserrat", 15),
                        text_color=GRIS
                    ).pack(anchor="w", padx=18, pady=(6, 12))

            ctk.CTkFrame(bloque, height=8, fg_color="#1A1A1A").pack()

    ctk.CTkButton(
        ventana,
        text="Cerrar",
        width=160,
        height=44,
        fg_color=AZUL,
        hover_color=MORADO,
        font=("Montserrat", 16, "bold"),
        command=ventana.destroy
    ).pack(pady=(0, 18))


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
    text="Resumen de tu práctica en TALAT",
    font=("Montserrat",22),
    text_color=GRIS
)

# Antes tenía pady=120 y empujaba la gráfica fuera de la pantalla.
contenido_perfil.pack(
    pady=(10, 5)
)

# ------------------------------------------
# FICHA DE DATOS PERSONALES
# ------------------------------------------

ficha_personal = ctk.CTkFrame(
    contenido_scroll,
    fg_color="#1A1A1A",
    corner_radius=20
)

ficha_personal.pack(
    fill="x",
    padx=40,
    pady=15
)

ctk.CTkLabel(
    ficha_personal,
    text="🪪 Datos de la persona",
    font=("Montserrat",24,"bold"),
    text_color=BLANCO
).pack(anchor="w", padx=20, pady=(15,10))

edad_label = ctk.CTkLabel(
    ficha_personal,
    text="Edad: sin registrar",
    font=("Montserrat",20),
    text_color=BLANCO,
    anchor="w"
)

edad_label.pack(anchor="w", padx=20, pady=(0,10))

ctk.CTkLabel(
    ficha_personal,
    text="¿Por qué usa TALAT?",
    font=("Montserrat",17,"bold"),
    text_color=GRIS,
    anchor="w"
).pack(anchor="w", padx=20)

motivo_label = ctk.CTkLabel(
    ficha_personal,
    text="Sin registrar",
    font=("Montserrat",18),
    text_color=BLANCO,
    anchor="w",
    justify="left",
    wraplength=850
)

motivo_label.pack(anchor="w", padx=20, pady=(2,18))


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

grafica_subtitulo = ctk.CTkLabel(
    estadisticas,
    text="Tu progreso en las últimas 10 sesiones",
    font=("Montserrat",16),
    text_color=GRIS
)

grafica_subtitulo.pack(
    pady=(0,10)
)

selector_metrica = ctk.CTkSegmentedButton(
    estadisticas,
    values=["Expresiones", "Minutos"],
    font=("Montserrat",14,"bold"),
    selected_color=AZUL,
    selected_hover_color=MORADO,
    unselected_color="#2B2B2B",
    command=cambiar_metrica_grafica
)

selector_metrica.set("Expresiones")

selector_metrica.pack(
    pady=(0,15)
)

grafica_canvas = ctk.CTkCanvas(
    estadisticas,
    height=240,
    bg="#1A1A1A",
    highlightthickness=0
)

grafica_canvas.pack(
    fill="x",
    padx=25,
    pady=(0,15)
)

# Al cambiar de tamaño la ventana, la gráfica se redibuja sola.
grafica_canvas.bind(
    "<Configure>",
    dibujar_grafica_progreso
)

btn_detalle_estadisticas = ctk.CTkButton(
    estadisticas,
    text="🔍 Ver cómo te ha ido",
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

def editar_usuario():
    """Edita nombre, edad y motivo del usuario abierto."""
    global usuario_actual

    if usuario_actual is None:
        return

    datos = usuarios_db.get(usuario_actual)

    if datos is None:
        return

    nuevos = dialogo_datos_usuario(
        "Editar usuario",
        nombre=usuario_actual,
        edad=datos.get("edad", ""),
        motivo=datos.get("motivo", "")
    )

    if nuevos is None:
        return

    nombre_nuevo = nuevos["nombre"]

    # Si cambió el nombre hay que mover la ficha completa a la nueva clave,
    # porque el nombre es el identificador del usuario en el archivo.
    if nombre_nuevo != usuario_actual:
        if nombre_nuevo in usuarios_db:
            print("Ya existe un usuario con ese nombre.")
            return

        usuarios_db[nombre_nuevo] = usuarios_db.pop(usuario_actual)
        usuario_actual = nombre_nuevo

    usuarios_db[usuario_actual]["edad"] = nuevos["edad"]
    usuarios_db[usuario_actual]["motivo"] = nuevos["motivo"]

    guardar_usuarios(usuarios_db)

    titulo_perfil.configure(text=f"👤 {usuario_actual}")

    actualizar_estadisticas_perfil()
    recargar_tarjetas_usuarios()


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
    width=220,
    command=editar_usuario
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