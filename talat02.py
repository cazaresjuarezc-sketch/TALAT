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
# MOTOR DE SONIDO
# ==========================================
#
# Aquí se produce el sonido de la computadora, el que se oye mientras
# el piano físico no está conectado. Las canciones se tocan nota por
# nota; el Modo Terapia puede además sonar tres notas a la vez.

# ------------------------------------------
# LAS NOTAS
# ------------------------------------------
#
# Doce notas: los doce semitonos, una por cada relevador del piano.
# La POSICIÓN de la nota en esta lista es el número de relevador que
# se le manda al Arduino, así que el orden de aquí y el cableado del
# piano tienen que coincidir.
#
# Las frecuencias no se escriben a mano: cada semitono es la anterior
# multiplicada por la raíz doceava de 2. Así ninguna queda desafinada
# por un error de tecleo.

NOTAS = [
    "DO", "DO#", "RE", "RE#", "MI", "FA",
    "FA#", "SOL", "SOL#", "LA", "LA#", "SI"
]

FRECUENCIA_DO = 261.63

FRECUENCIAS_NOTAS = {
    nota: FRECUENCIA_DO * (2 ** (posicion / 12))
    for posicion, nota in enumerate(NOTAS)
}

# ------------------------------------------
# QUÉ NOTAS TIENE EL PIANO FÍSICO
# ------------------------------------------
#
# La app maneja las doce notas, pero el piano tiene ocho relevadores.
# Esta tabla dice cuál relevador le toca a cada nota.
#
# Las ocho elegidas NO son al azar: son todas las que usan las dos
# canciones (DO, RE, MI, FA#, SOL, LA, SI) más el FA, que hace falta
# para los acordes de FA mayor y RE menor del Modo Terapia. Con ocho
# relevadores el piano toca el 100% de las canciones y de los acordes.
#
# El orden respeta el cableado que ya existía: DO, RE, MI, FA y SOL
# son los cinco relevadores originales.
#
# Las cuatro que faltan (DO#, RE#, SOL#, LA#) solo aparecen si alguien
# las elige en Modo Libre: esas suenan por la computadora y no mueven
# ningún relevador. No se rompe nada.
#
# Si cableas distinto, cambia SOLO los números de aquí; el número es
# la entrada del módulo (IN1 = 0, IN2 = 1...) y el pin lo define el
# sketch del Arduino.

RELEVADOR_POR_NOTA = {
    # Los cinco que ya tenías cableados, en el mismo orden.
    "DO": 0,
    "RE": 1,
    "MI": 2,
    "FA": 3,
    "SOL": 4,

    # Los tres que faltan para completar el repertorio.
    # Si todavía no los conectas, no pasa nada: el sketch revisa que
    # el relevador exista y esas notas suenan solo en la computadora.
    "FA#": 5,
    "LA": 6,
    "SI": 7
}

# Notas que sí mueven un relevador, en orden de relevador.
NOTAS_DEL_PIANO = sorted(
    RELEVADOR_POR_NOTA,
    key=lambda nota: RELEVADOR_POR_NOTA[nota]
)


class MotorDeNotas:
    """
    Reproduce una nota sola o varias a la vez.

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

        print("TALAT: sin motor de sonido. Instala pygame para escuchar las notas.")

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
        Manda las notas al piano físico.

        Van todas en un solo mensaje: si se enviaran una por una, las
        tres notas de un acorde llegarían separadas por milisegundos
        y se oiría como un arpegio rápido en vez de un acorde.
        """
        if piano_hardware is None or not piano_hardware.disponible():
            return

        try:
            piano_hardware.tocar(notas)
        except Exception as e:
            print("No se pudo activar el relevador:", e)

    def precargar(self, notas):
        """
        Genera de una vez todos los sonidos.

        Crear una nota cuesta ~19,000 cuentas en Python. Si eso pasa
        mientras la melodía suena, la primera nota de cada frase llega
        tarde y se oye un tirón.
        """
        if self.backend != "pygame":
            return

        for nota in notas:
            if nota not in self.cache:
                self.cache[nota] = self._crear_sonido(nota)

    def tocar(self, nota):
        """Una sola nota."""
        if nota in ("--", None):
            return

        self.tocar_notas([nota])

    def tocar_notas(self, notas):
        """Suena una o varias notas a la vez, sin congelar la interfaz."""
        threading.Thread(
            target=self._reproducir_notas,
            args=(list(notas),),
            daemon=True
        ).start()

        self.enviar_a_hardware(notas)


motor_notas = MotorDeNotas()

# En segundo plano, para que la ventana abra igual de rápido.
threading.Thread(
    target=lambda: motor_notas.precargar(list(FRECUENCIAS_NOTAS)),
    daemon=True
).start()


# ==========================================
# PIANO FÍSICO (ARDUINO UNO + RELEVADORES)
# ==========================================
#
# El Arduino recibe por el cable USB una línea de texto y cierra los
# relevadores que le pidan. El formato es a propósito lo más simple
# posible, para poder probarlo a mano desde el Monitor Serie:
#
#       T:0,4,7:600\n     -> cierra los relevadores 0, 4 y 7 por 600 ms
#       X\n               -> abre todos (paro de emergencia)
#
# Los números salen de RELEVADOR_POR_NOTA: DO = 0, RE = 1, MI = 2,
# FA = 3, SOL = 4, FA# = 5, LA = 6, SI = 7.
#
# Todo está envuelto en try: si no hay Arduino conectado, si falta la
# librería pyserial o si el cable se desconecta a media demostración,
# TALAT sigue funcionando con el sonido de la computadora. En un
# concurso eso es la diferencia entre una falla y un imprevisto.

PUERTO_ARDUINO = None      # None = buscarlo solo. O ponlo a mano: "COM3"

# Ponlo en True para ver en la terminal cada mensaje que sale hacia el
# Arduino y cada respuesta que llega. Es lo primero que hay que revisar
# cuando "no se está mandando la señal".
DEPURAR_PIANO = True
VELOCIDAD_ARDUINO = 115200
DURACION_TECLA_MS = 600

# Palabras que aparecen en el nombre del puerto cuando hay un Arduino.
# CH340 y CP210 son los chips USB de las placas compatibles.
PISTAS_ARDUINO = ("arduino", "ch340", "ch341", "cp210", "usb serial", "wch")


class PianoArduino:
    """Puente entre TALAT y los relevadores del piano."""

    def __init__(self):
        self.puerto = None
        self.conexion = None
        self.error = ""

    def disponible(self):
        return self.conexion is not None

    # ---------- conexión ----------

    def buscar_puerto(self):
        try:
            from serial.tools import list_ports
        except ImportError:
            self.error = "Falta pyserial (pip install pyserial)"
            return None

        candidatos = list(list_ports.comports())

        for puerto in candidatos:
            descripcion = f"{puerto.description} {puerto.manufacturer}".lower()

            if any(pista in descripcion for pista in PISTAS_ARDUINO):
                return puerto.device

        # Si solo hay un puerto, probablemente es ese.
        if len(candidatos) == 1:
            return candidatos[0].device

        if candidatos:
            nombres = ", ".join(p.device for p in candidatos)
            self.error = f"Ningún puerto parece Arduino ({nombres})"
        else:
            self.error = "No hay puertos: revisa el cable USB"

        return None

    def conectar(self):
        """
        Abre el puerto. Tarda unos segundos, así que se llama desde un
        hilo aparte para que la ventana de TALAT no se congele.
        """
        try:
            import serial
        except ImportError:
            self.error = "Falta pyserial (pip install pyserial)"
            return False

        puerto = PUERTO_ARDUINO or self.buscar_puerto()

        if puerto is None:
            return False

        try:
            self.conexion = serial.Serial(
                puerto,
                VELOCIDAD_ARDUINO,
                timeout=0.2,
                write_timeout=0.2
            )

            self.puerto = puerto

            # Al abrir el puerto el Arduino se reinicia solo.
            # Si le hablamos antes de tiempo, pierde el primer mensaje.
            import time
            time.sleep(2.0)

            # El sketch saluda con "TALAT listo". Si no saluda, el puerto
            # abrió pero del otro lado no está nuestro programa.
            saludo = self.conexion.read_all().decode(errors="ignore")

            if DEPURAR_PIANO:
                print(f"Arduino en {puerto} saludó: {saludo.strip()!r}")

            if "TALAT" not in saludo:
                self.error = "Responde, pero no trae el sketch de TALAT"
            else:
                self.error = ""

            return True

        except Exception as e:
            self.conexion = None
            self.error = str(e)
            return False

    def soltar_todo(self):
        """Abre todos los relevadores. Paro de emergencia."""
        if self.conexion is None:
            return

        try:
            self.conexion.write(b"X\n")
        except Exception:
            pass

    def cerrar(self):
        if self.conexion is None:
            return

        try:
            self.conexion.write(b"X\n")
            self.conexion.close()
        except Exception:
            pass

        self.conexion = None

    # ---------- tocar ----------

    def tocar(self, notas, duracion_ms=DURACION_TECLA_MS):
        """Cierra los relevadores de esas notas."""
        if self.conexion is None:
            return

        relevadores = [
            str(RELEVADOR_POR_NOTA[nota])
            for nota in notas
            if nota in RELEVADOR_POR_NOTA
        ]

        if not relevadores:
            return

        mensaje = f"T:{','.join(relevadores)}:{int(duracion_ms)}\n"

        try:
            self.conexion.write(mensaje.encode("ascii"))

            # flush obliga a que los bytes salgan YA por el cable.
            # Sin esto pueden quedarse esperando en el buffer del
            # sistema y la nota llega tarde o no llega.
            self.conexion.flush()

            if DEPURAR_PIANO:
                print("-> Arduino:", mensaje.strip())

            # Vaciamos lo que el Arduino contesta ("OK 3"). Si no se lee,
            # el buffer del puerto se llena y a los pocos minutos las
            # escrituras empiezan a tardar.
            if self.conexion.in_waiting:
                respuesta = self.conexion.read_all()

                if DEPURAR_PIANO:
                    print("<- Arduino:", respuesta.decode(errors="ignore").strip())

        except Exception as e:
            # El cable se soltó: seguimos con el sonido de la compu.
            print("Se perdió el Arduino:", e)
            self.conexion = None
            self.error = "Se desconectó"


    def probar(self):
        """
        Recorre las doce teclas, una por una.

        Sirve para revisar el cableado sin tener que hacer caras
        frente a la cámara: si el relevador 5 no suena, ese es el
        cable flojo.
        """
        if self.conexion is None:
            return

        import time

        for nota in NOTAS_DEL_PIANO:
            self.tocar([nota], 300)
            time.sleep(0.4)


piano_hardware = PianoArduino()


def conectar_piano():
    """Busca el Arduino en segundo plano y avisa en pantalla."""
    piano_hardware.conectar()

    def avisar():
        if "estado_piano" not in globals():
            return

        if piano_hardware.disponible():
            if piano_hardware.error:
                # Abrió el puerto pero el sketch no contestó bien.
                estado_piano.configure(
                    text=f"⚠ {piano_hardware.error}",
                    text_color="#FF9F43"
                )
            else:
                estado_piano.configure(
                    text=f"🎹 Piano conectado ({piano_hardware.puerto})",
                    text_color=VERDE
                )
        else:
            # Decir el motivo real: "falta pyserial", "puerto ocupado",
            # "no hay puertos". Un genérico "sin piano" no ayuda a nadie
            # a las once de la noche antes del concurso.
            motivo = piano_hardware.error or "sin piano"

            estado_piano.configure(
                text=f"🔌 {motivo}",
                text_color="#FF9F43"
            )

        if DEPURAR_PIANO:
            print("Piano:", piano_hardware.error or "conectado en " + str(piano_hardware.puerto))

    app.after(0, avisar)


threading.Thread(target=conectar_piano, daemon=True).start()


# ==========================================
# GESTOS Y SU GUÍA
# ==========================================
#
# Las cinco expresiones que reconoce TALAT, con la explicación que se
# le muestra a la persona. Cada canción elige cuáles usa.

# Un acorde por expresión, para el Modo Terapia.
#
# Son los cinco primeros grados de la escala de DO mayor (I, ii, iii,
# IV, V). Al pertenecer todos a la misma escala, suenen en el orden que
# suenen nunca chocan entre sí: la persona no puede "equivocarse" de
# armonía, que es justo lo que se busca en una sesión de terapia.
ACORDES_TERAPIA = {
    "ABURRIMIENTO": {
        "nombre": "DO mayor",
        "notas": ["DO", "MI", "SOL"]
    },
    "SORPRESA": {
        "nombre": "RE menor",
        "notas": ["RE", "FA", "LA"]
    },
    "IRA": {
        "nombre": "MI menor",
        "notas": ["MI", "SOL", "SI"]
    },
    "TRISTEZA": {
        "nombre": "FA mayor",
        "notas": ["FA", "LA", "DO"]
    },
    "ALEGRÍA": {
        "nombre": "SOL mayor",
        "notas": ["SOL", "SI", "RE"]
    }
}

# Nota suelta de cada expresión, para cuando el Modo Terapia
# está puesto en "Notas".
NOTAS_POR_EMOCION = {
    "ABURRIMIENTO": "DO",
    "SORPRESA": "RE",
    "IRA": "MI",
    "TRISTEZA": "FA",
    "ALEGRÍA": "SOL"
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
# CANCIONES GUIADAS
# ==========================================
#
# Cada canción se divide en FRASES. Una frase es un renglón de la
# partitura con sus notas en orden: una sola expresión de la cara
# dispara la frase entera.
#
# Para agregar otra canción basta con copiar el mismo formato:
# letra, gesto que la dispara y la lista de notas.
# Tiempo entre notas. Súbelo a 0.55 si la melodía va muy rápido
# para tu usuario, o bájalo a 0.35 si se oye arrastrada.
PULSO_MELODIA = 0.42

# Color de la etiqueta que aparece en el menú de canciones.
COLOR_DIFICULTAD = {
    "Fácil": VERDE,
    "Media": AMARILLO,
    "Larga": "#FF9F43"
}

CANCIONES = {
    "cucaracha": {
        "titulo": "La cucaracha",
        "subtitulo": "2 frases · 2 expresiones",
        "dificultad": "Fácil",
        "frases": [
            {
                "letra": (
                    "La cucaracha, la cucaracha\n"
                    "ya no puede caminar"
                ),
                "gesto": "ALEGRÍA",
                "notas": [
                    "RE", "RE", "RE", "SOL", "SI",
                    "RE", "RE", "RE", "SOL", "SI",
                    "SOL", "SOL", "FA#", "FA#", "MI", "MI", "RE"
                ]
            },
            {
                "letra": (
                    "Porque no tiene, porque le falta\n"
                    "las dos patitas de atrás"
                ),
                "gesto": "SORPRESA",
                "notas": [
                    "RE", "RE", "RE", "FA#", "LA",
                    "RE", "RE", "RE", "FA#", "LA",
                    "RE", "MI", "RE", "DO", "SI", "LA", "SOL"
                ]
            }
        ]
    },

    "estrellita": {
        "titulo": "Estrellita, ¿dónde estás?",
        "subtitulo": "3 frases · 3 expresiones",
        "dificultad": "Media",
        "frases": [
            {
                "letra": (
                    "Estrellita, ¿dónde estás?\n"
                    "Me pregunto quién serás"
                ),
                "gesto": "ALEGRÍA",
                "notas": [
                    "SOL", "SOL", "RE", "RE", "MI", "MI", "RE",
                    "DO", "DO", "SI", "SI", "LA", "LA", "SOL"
                ]
            },
            {
                "letra": (
                    "En el cielo o en el mar\n"
                    "un diamante de verdad"
                ),
                "gesto": "SORPRESA",
                "notas": [
                    "RE", "RE", "DO", "DO", "SI", "SI", "LA",
                    "RE", "RE", "DO", "DO", "SI", "SI", "LA"
                ]
            },
            {
                "letra": (
                    "Estrellita, ¿dónde estás?\n"
                    "Me pregunto quién serás"
                ),
                "gesto": "ALEGRÍA",
                "notas": [
                    "SOL", "SOL", "RE", "RE", "MI", "MI", "RE",
                    "DO", "DO", "SI", "SI", "LA", "LA", "SOL"
                ]
            }
        ]
    }
}


def gestos_usados(cancion):
    return [
        frase["gesto"]
        for frase in cancion["frases"]
    ]

class CancionGuiada:
    """
    Canción Guiada:

    UNA EXPRESIÓN = UNA FRASE COMPLETA DE LA MELODÍA.

    La persona hace la expresión una sola vez.
    TALAT toca las notas de esa frase, una tras otra, y la persona
    puede relajar la cara mientras suena.
    """

    def __init__(self, motor):
        self.motor = motor
        self.clave = None
        self.cancion = None
        self.frases = []
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

    def total_frases(self):
        return len(self.frases)

    def frase_actual(self):
        if not self.frases or self.indice >= len(self.frases):
            return None

        return self.frases[self.indice]

    def gesto_actual(self):
        frase = self.frase_actual()

        if frase is None:
            return None

        return frase.get("gesto")

    def letra_actual(self):
        frase = self.frase_actual()

        if frase is None:
            return ""

        return frase.get("letra", "")

    def notas_actuales(self):
        frase = self.frase_actual()

        if frase is None:
            return []

        return frase.get("notas", [])

    def progreso(self):
        if not self.frases:
            return 0.0

        return self.indice / len(self.frases)

    # --------------------------------------
    # CARGAR
    # --------------------------------------

    def cargar(self, clave):
        cancion = CANCIONES.get(clave)

        if cancion is None:
            return False

        self.clave = clave
        self.cancion = cancion
        self.frases = list(cancion.get("frases", []))

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
        Una sola expresión correcta dispara la frase completa.
        """

        self.gesto_detectado = gesto

        # Mientras la frase suena, el estado es "reproduciendo"
        # y aquí no se acepta ninguna expresión nueva.
        if self.estado != "tocando":
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

        return self._acertar_frase()

    # --------------------------------------
    # EJECUTAR BLOQUE
    # --------------------------------------

    def _acertar_frase(self):
        frase = self.frase_actual()

        if frase is None:
            return False

        notas = frase.get("notas", [])

        if not notas:
            return False

        self.estado = "reproduciendo"
        self.aciertos += 1

        # Guardamos en qué frase vamos antes de avanzar.
        indice_frase = self.indice

        # Reproduce la frase completa.
        threading.Thread(
            target=self._reproducir_frase,
            args=(notas, indice_frase),
            daemon=True
        ).start()

        return True

    def _reproducir_frase(self, notas, indice_frase):
        import time

        try:
            for posicion, nota in enumerate(notas):

                # Si se pausó o se salió de la canción, el hilo muere.
                # Antes solo revisaba "pausada" y seguía sonando aunque
                # el usuario ya se hubiera ido al menú.
                if self.estado != "reproduciendo":
                    return

                self.motor.tocar(nota)

                # La interfaz solo se puede tocar desde el hilo principal.
                app.after(
                    0,
                    lambda p=posicion, i=indice_frase: resaltar_nota(i, p)
                )

                time.sleep(PULSO_MELODIA)

            # Solo la interfaz principal debe modificar
            # el estado de la canción.
            app.after(
                100,
                lambda i=indice_frase: self._terminar_frase(i)
            )

        except Exception as e:
            print("Error reproduciendo la frase:", e)

            app.after(
                100,
                lambda i=indice_frase: self._terminar_frase(i)
            )

    def _terminar_frase(self, indice_frase):
        # Evita avanzar accidentalmente si ya cambió la canción.
        if indice_frase != self.indice:
            return

        self.indice += 1

        if self.indice >= len(self.frases):
            self.estado = "terminada"
            self.esperando_reposo = False

        else:
            self.estado = "tocando"

            # Para activar la siguiente frase:
            # primero debe relajar la cara.
            self.esperando_reposo = True

        actualizar_panel_cancion()

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

# Sonidos que la persona puede asignar a sus propios gestos:
# las doce notas, las mismas que tiene el piano.

SONIDOS_LIBRES = {
    nota: {"etiqueta": f"🎵 {nota}", "notas": [nota]}
    for nota in NOTAS
}

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



# Modo Terapia: "notas" (una nota por expresión) o "acordes".
sonido_terapia = "notas"


def sonar_expresion(emocion):
    """
    Suena la expresión que acaba de hacer la persona.

    Se llama UNA sola vez, en el momento en que la expresión se vuelve
    estable, no en cada frame: si no, la misma cara dispararía la nota
    treinta veces por segundo.
    """
    if sonido_terapia == "acordes":
        acorde = ACORDES_TERAPIA.get(emocion)

        if acorde is None:
            return ""

        motor_notas.tocar_notas(acorde["notas"])

        return acorde["nombre"]

    nota = NOTAS_POR_EMOCION.get(emocion)

    if nota is None:
        return ""

    motor_notas.tocar(nota)

    return nota


def cambiar_sonido_terapia(valor):
    global sonido_terapia

    sonido_terapia = "acordes" if valor.startswith("🎹") else "notas"


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

    # Abrir todos los relevadores: ninguna tecla se queda pisada
    # si la sesión termina justo mientras sonaba un acorde.
    if piano_hardware is not None and piano_hardware.disponible():
        piano_hardware.soltar_todo()

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

                    # En Modo Terapia la expresión suena aquí mismo.
                    # Los otros dos modos tienen su propio disparador.
                    if modo_sesion == "terapia":
                        sonando = sonar_expresion(detector.ultima_estable)

                        if sonando:
                            estado_nota.configure(
                                text=f"🎹 {sonando}",
                                text_color=detector.color
                            )

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
                    # Sin este try, cualquier error del panel corta el
                    # app.after de abajo y la cámara se queda congelada.
                    try:
                        avanzo = cancion_guiada.procesar(detector.ultima_estable)

                        if avanzo:
                            actualizar_panel_cancion()
                        else:
                            refrescar_deteccion_cancion()

                    except Exception as e:
                        print("Error en Canción Guiada:", e)

    # Reescalar cuesta CPU 30 veces por segundo: BILINEAR es la
    # interpolación más barata que se ve bien.
    imagen = Image.fromarray(rgb)
    imagen = imagen.resize((930, 650), Image.BILINEAR)

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

# Estado del piano físico. Se actualiza solo cuando termina la
# búsqueda del Arduino, que corre en segundo plano.
estado_piano = ctk.CTkLabel(
    header_sesion,
    text="🔌 Buscando el piano...",
    font=("Montserrat", 14, "bold"),
    text_color=GRIS,
    wraplength=260,
    justify="right"
)

estado_piano.pack(side="right", padx=10)


def probar_piano():
    """Recorre las doce teclas del piano físico, para revisar el cableado."""
    if not piano_hardware.disponible():
        estado_piano.configure(
            text="🔌 No hay piano conectado",
            text_color="#FF9F43"
        )
        return

    estado_piano.configure(
        text=f"🎹 Probando las {len(NOTAS_DEL_PIANO)} teclas...",
        text_color=AZUL
    )

    def terminar():
        estado_piano.configure(
            text=f"🎹 Piano conectado ({piano_hardware.puerto})",
            text_color=VERDE
        )

    def correr():
        piano_hardware.probar()
        app.after(0, terminar)

    threading.Thread(target=correr, daemon=True).start()


btn_probar_piano = ctk.CTkButton(
    header_sesion,
    text="🔧 Probar piano",
    width=140,
    height=38,
    fg_color="#2B2B2B",
    hover_color=MORADO,
    font=("Montserrat", 13, "bold"),
    command=probar_piano
)

btn_probar_piano.pack(side="right", padx=6)



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
    font=("Segoe UI Emoji", 190)
)

emoji_label.pack(
    pady=(10,0)
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
    pady=(10, 10)
)

# ------------------------
# MENSAJE
# ------------------------

mensaje_label = ctk.CTkLabel(
    info_sesion,
    text="Haz una expresión",
    font=("Montserrat",20),
    wraplength=380,
    justify="center",
    text_color=GRIS
)

mensaje_label.pack(
    pady=(0, 8)
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
# SONIDO DE LAS EXPRESIONES
# ==========================================
#
# Cuelga del panel de las expresiones, no de dentro: ese panel se
# quita y se vuelve a poner cada vez que se cambia de modo, y el
# selector desaparecía con él.
#
# Solo se muestra en Modo Terapia, que es el único donde aplica.
# Se empaqueta con before=botones_modo para que siempre vuelva a su
# lugar; sin eso, al reaparecer se iría hasta el final de la ventana.

barra_sonido = ctk.CTkFrame(
    sesion_frame,
    fg_color="#1A1A1A",
    corner_radius=16
)

etiqueta_sonido_terapia = ctk.CTkLabel(
    barra_sonido,
    text="SONIDO",
    font=("Montserrat", 12, "bold"),
    text_color=GRIS
)

etiqueta_sonido_terapia.pack(side="left", padx=(16, 10), pady=8)

selector_sonido_terapia = ctk.CTkSegmentedButton(
    barra_sonido,
    values=["🎵 Notas", "🎹 Acordes"],
    font=("Montserrat", 15, "bold"),
    height=34,
    selected_color=AZUL,
    selected_hover_color=MORADO,
    unselected_color="#2B2B2B",
    command=cambiar_sonido_terapia
)

selector_sonido_terapia.set("🎵 Notas")

selector_sonido_terapia.pack(side="left", padx=(0, 12), pady=8)


def mostrar_sonido_terapia(visible):
    """Aparece pegado al panel de las expresiones, solo en Terapia."""
    if visible:
        barra_sonido.pack(
            before=botones_modo,
            anchor="e",
            padx=(0, 60),
            pady=(4, 0)
        )
    else:
        barra_sonido.pack_forget()


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
    text="Una sola expresión toca una frase completa de la melodía.",
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

# Letra de la frase, cabecera y renglón de notas.

cancion_cabecera = ctk.CTkLabel(
    vista_tocar,
    text="FRASE 1 DE 1",
    font=("Montserrat", 20, "bold"),
    text_color=AZUL
)

cancion_cabecera.pack(pady=(8, 2))

cancion_barra = ctk.CTkProgressBar(
    vista_tocar,
    height=10,
    corner_radius=5,
    progress_color=AZUL
)

cancion_barra.set(0)

cancion_barra.pack(fill="x", padx=25, pady=(0, 8))

# La letra es lo que la persona canta o sigue con la vista:
# va en grande y en blanco, no como texto secundario.
cancion_letra = ctk.CTkLabel(
    vista_tocar,
    text="",
    font=("Montserrat", 16, "bold"),
    text_color=BLANCO,
    wraplength=420,
    justify="center"
)

cancion_letra.pack(pady=(0, 8))

# Renglón de notas. Alto fijo para tres filas: si creciera y encogiera
# con cada frase, todo el panel daría saltos al cambiar de frase.
fila_notas = ctk.CTkFrame(
    vista_tocar,
    fg_color="transparent",
    width=406,
    height=116
)

fila_notas.pack(pady=(0, 8))
fila_notas.pack_propagate(False)
fila_notas.grid_propagate(False)

# Guardamos los recuadros para repintarlos sin recrearlos.
chips_cancion = []
frase_dibujada = -1

# Qué expresión hay que hacer para disparar la frase.

zona_gesto = ctk.CTkFrame(
    vista_tocar,
    fg_color="#111111",
    corner_radius=14
)

zona_gesto.pack(fill="x", padx=18, pady=(0, 8))

# Emoji a la izquierda y textos a la derecha: en vertical no cabían
# el emoji grande, las tres filas de notas y los botones.
cancion_emoji = ctk.CTkLabel(
    zona_gesto,
    text="🎵",
    font=("Segoe UI Emoji", 52),
    width=80
)

cancion_emoji.grid(row=0, column=0, rowspan=2, padx=(14, 6), pady=10)

cancion_gesto_titulo = ctk.CTkLabel(
    zona_gesto,
    text="",
    font=("Montserrat", 17, "bold"),
    text_color=BLANCO,
    anchor="w",
    justify="left"
)

cancion_gesto_titulo.grid(row=0, column=1, sticky="sw", padx=(0, 14))

cancion_gesto_como = ctk.CTkLabel(
    zona_gesto,
    text="",
    font=("Montserrat", 13),
    text_color=GRIS,
    wraplength=290,
    anchor="w",
    justify="left"
)

cancion_gesto_como.grid(row=1, column=1, sticky="nw", padx=(0, 14))

zona_gesto.grid_columnconfigure(1, weight=1)

cancion_feedback = ctk.CTkLabel(
    zona_gesto,
    text="",
    font=("Montserrat", 15, "bold"),
    text_color=GRIS,
    wraplength=400
)

cancion_feedback.grid(row=2, column=0, columnspan=2, pady=(0, 10))

# Botones de control.

controles_cancion = ctk.CTkFrame(
    vista_tocar,
    fg_color="transparent"
)

controles_cancion.pack(pady=(0, 12))


# ------------------------------------------
# FUNCIONES DEL PANEL
# ------------------------------------------

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

        # Subtítulo y etiqueta de dificultad, en la misma línea.
        linea_subtitulo = ctk.CTkFrame(tarjeta, fg_color="transparent")
        linea_subtitulo.pack(anchor="w", fill="x", padx=15)

        ctk.CTkLabel(
            linea_subtitulo,
            text=cancion["subtitulo"],
            font=("Montserrat", 13),
            text_color=GRIS
        ).pack(side="left")

        dificultad = cancion.get("dificultad", "")

        if dificultad:
            ctk.CTkLabel(
                linea_subtitulo,
                text=f" {dificultad} ",
                font=("Montserrat", 12, "bold"),
                text_color=NEGRO,
                fg_color=COLOR_DIFICULTAD.get(dificultad, GRIS),
                corner_radius=8,
                height=22
            ).pack(side="right")

        # Expresiones que hacen falta, sin repetir.
        fila_gestos = ctk.CTkFrame(
            tarjeta,
            fg_color="transparent"
        )

        fila_gestos.pack(
            anchor="w",
            padx=15,
            pady=(6, 0)
        )

        for gesto in dict.fromkeys(gestos_usados(cancion)):

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

        total_frases = len(cancion["frases"])

        total_notas = sum(
            len(frase.get("notas", []))
            for frase in cancion["frases"]
        )

        ctk.CTkLabel(
            tarjeta,
            text=f"{total_frases} frases · {total_frases} expresiones · "
                 f"{total_notas} notas",
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

    cancion_barra.configure(progress_color=AZUL)

    dibujar_notas_frase(forzar=True)
    actualizar_panel_cancion()


def poner_feedback(texto, color):
    """No repinta si el mensaje es el mismo que ya está en pantalla."""
    if cancion_feedback.cget("text") == texto:
        return

    cancion_feedback.configure(text=texto, text_color=color)


def refrescar_deteccion_cancion():
    """Solo actualiza el mensaje de lo que la cámara está viendo ahora."""
    if cancion_guiada.estado != "tocando":
        return

    gesto = cancion_guiada.gesto_detectado
    objetivo = cancion_guiada.gesto_actual()

    if gesto == "REPOSO":
        poner_feedback("Esperando tu expresión...", GRIS)

    elif cancion_guiada.esperando_reposo:
        poner_feedback("Relaja la cara para la siguiente frase", AZUL)

    elif gesto != objetivo:
        guia_objetivo = GUIA_DE_GESTOS.get(objetivo, {})
        titulo = guia_objetivo.get("titulo", objetivo).lower()

        poner_feedback(f"Ahora toca: {titulo}", "#FF9F43")


def dibujar_notas_frase(forzar=False):
    """
    Dibuja los recuadros con las notas de la frase actual.

    Solo se rehacen cuando cambia la frase. Rehacerlos en cada
    actualización borraba los recuadros justo cuando la melodía
    empezaba a iluminarlos, y la primera nota nunca se veía.
    """
    global frase_dibujada

    if not cancion_guiada.cargada():
        return

    frase = cancion_guiada.frase_actual()

    if frase is None:
        return

    if not forzar and frase_dibujada == cancion_guiada.indice:
        return

    frase_dibujada = cancion_guiada.indice

    for chip in chips_cancion:
        chip.destroy()

    chips_cancion.clear()

    notas = frase.get("notas", [])

    activo = cancion_guiada.estado in ("tocando", "pausada", "reproduciendo")

    for posicion, nota in enumerate(notas):

        chip = ctk.CTkLabel(
            fila_notas,
            text=nota,
            width=52,
            height=30,
            corner_radius=10,
            fg_color=MORADO if activo else "#2B2B2B",
            text_color=BLANCO if activo else GRIS,
            font=("Montserrat", 13, "bold")
        )

        chip.grid(
            row=posicion // 7,
            column=posicion % 7,
            padx=3,
            pady=3
        )

        chips_cancion.append(chip)


def resaltar_nota(indice_frase, posicion):
    """
    Ilumina la nota que suena en este instante (efecto karaoke).

    Solo toca DOS recuadros: el que acaba de sonar y el que suena.
    Repintar los diecisiete en cada nota sí se notaría en la Raspberry.
    """
    if indice_frase != cancion_guiada.indice:
        return

    try:
        if 0 < posicion < len(chips_cancion):
            chips_cancion[posicion - 1].configure(
                fg_color="#3A2A5C",
                text_color=GRIS
            )

        if posicion < len(chips_cancion):
            chips_cancion[posicion].configure(
                fg_color=AMARILLO,
                text_color=NEGRO
            )

    except Exception:
        # Los recuadros se rehicieron mientras sonaba la frase.
        pass


def actualizar_panel_cancion():
    """Actualiza la pantalla de Canción Guiada."""

    if not cancion_guiada.cargada():
        return

    cancion = cancion_guiada.cancion

    cancion_titulo.configure(
        text=cancion["titulo"]
    )

    total_frases = cancion_guiada.total_frases()

    # ---------- canción terminada ----------

    if cancion_guiada.estado == "terminada":

        cancion_cabecera.configure(
            text="🎉 ¡CANCIÓN COMPLETADA!",
            text_color=AMARILLO
        )

        cancion_letra.configure(
            text=f"Tocaste la canción entera con {total_frases} expresiones."
        )

        cancion_emoji.configure(text="🌟")

        cancion_gesto_titulo.configure(
            text="Muy bien",
            text_color=AMARILLO
        )

        cancion_gesto_como.configure(
            text="Pulsa ↻ REINICIAR para tocarla otra vez, "
                 "o ← Canciones para elegir otra."
        )

        poner_feedback("", GRIS)

        cancion_barra.set(1)
        cancion_barra.configure(progress_color=AMARILLO)

        registrar_cancion_completada()

        return

    frase = cancion_guiada.frase_actual()

    if frase is None:
        return

    numero = cancion_guiada.indice + 1
    notas = cancion_guiada.notas_actuales()

    gesto = cancion_guiada.gesto_actual()
    guia = GUIA_DE_GESTOS.get(gesto, {})
    color_gesto = COLORES_POR_EMOCION.get(gesto, MORADO)

    cancion_letra.configure(text=cancion_guiada.letra_actual())

    dibujar_notas_frase()

    cancion_barra.set(cancion_guiada.progreso())
    cancion_barra.configure(progress_color=color_gesto)

    # ---------- la melodía está sonando ----------

    if cancion_guiada.estado == "reproduciendo":

        cancion_cabecera.configure(
            text=f"🎶 SONANDO LA FRASE {numero}",
            text_color=AZUL
        )

        cancion_emoji.configure(text="🎧")

        cancion_gesto_titulo.configure(
            text="Puedes relajar la cara",
            text_color=AZUL
        )

        cancion_gesto_como.configure(
            text="Sigue las notas amarillas mientras suena la melodía."
        )

        poner_feedback("Escucha y descansa...", AZUL)

        return

    # ---------- esperando la expresión ----------

    cancion_cabecera.configure(
        text=f"FRASE {numero} DE {total_frases}   ·   {len(notas)} notas",
        text_color=color_gesto
    )

    cancion_emoji.configure(text=guia.get("emoji", "🎵"))

    cancion_gesto_titulo.configure(
        text=guia.get("titulo", ""),
        text_color=color_gesto
    )

    cancion_gesto_como.configure(text=guia.get("como", ""))

    if cancion_guiada.estado == "detenida":
        poner_feedback("Pulsa ▶ INICIAR para empezar", GRIS)

    elif cancion_guiada.estado == "pausada":
        poner_feedback("⏸ En pausa", "#FF9F43")

    elif cancion_guiada.esperando_reposo:
        poner_feedback("Relaja la cara para la siguiente frase", AZUL)

    else:
        poner_feedback("Haz esta expresión una vez", GRIS)


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

    cancion_barra.configure(progress_color=AZUL)

    # forzar: los recuadros pasan de gris apagado a morado activo.
    dibujar_notas_frase(forzar=True)
    actualizar_panel_cancion()


def pausar_cancion():
    cancion_guiada.pausar()
    actualizar_panel_cancion()


def reiniciar_cancion():
    cancion_guiada.reiniciar()

    cancion_barra.configure(progress_color=AZUL)

    dibujar_notas_frase(forzar=True)
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

# Igual que el menú de canciones: una línea que explica de qué va.
ctk.CTkLabel(
    panel_libre,
    text="Tú eliges el sonido y tú grabas la cara que lo toca.",
    font=("Montserrat", 13),
    text_color=GRIS,
    wraplength=410
).pack(pady=(0, 2))

libre_contador = ctk.CTkLabel(
    panel_libre,
    text="0 de 8 sonidos",
    font=("Montserrat", 12, "bold"),
    text_color=MORADO
)

libre_contador.pack(pady=(0, 6))

lista_libre = ctk.CTkScrollableFrame(
    panel_libre,
    fg_color="transparent"
)

lista_libre.pack(fill="both", expand=True, padx=10)

# Barra de estado: antes el texto flotaba suelto sobre el fondo y no
# se distinguía de la lista. Con su propio recuadro se lee como un
# renglón fijo donde siempre aparece lo que está pasando.
barra_estado_libre = ctk.CTkFrame(
    panel_libre,
    fg_color="#111111",
    corner_radius=12,
    height=46
)

barra_estado_libre.pack(fill="x", padx=14, pady=(8, 6))
barra_estado_libre.pack_propagate(False)

libre_feedback = ctk.CTkLabel(
    barra_estado_libre,
    text="Agrega un sonido y graba su gesto.",
    font=("Montserrat", 14, "bold"),
    text_color=GRIS,
    wraplength=400
)

libre_feedback.pack(expand=True)

botones_libre = ctk.CTkFrame(panel_libre, fg_color="transparent")
botones_libre.pack(fill="x", padx=14, pady=(0, 12))

# Cada fila de la lista, para poder iluminar la que suena.
filas_libre = []


def poner_estado_libre(texto, color):
    """Escribe en la barra de estado sin repintar si no cambió."""
    if libre_feedback.cget("text") == texto:
        return

    libre_feedback.configure(text=texto, text_color=color)


def marcar_fila_libre(indice, modo="normal"):
    """
    Resalta una fila: 'sonando' mientras el gesto está activo,
    'grabando' mientras se captura, 'normal' el resto del tiempo.
    """
    colores = {
        "normal": ("#111111", 0, "#111111"),
        "sonando": ("#1B2E45", 2, AZUL),
        "grabando": ("#2A1B45", 2, MORADO)
    }

    fondo, grosor, borde = colores.get(modo, colores["normal"])

    try:
        for i, fila in enumerate(filas_libre):
            if i == indice:
                fila.configure(
                    fg_color=fondo,
                    border_width=grosor,
                    border_color=borde
                )
            else:
                fila.configure(fg_color="#111111", border_width=0)

    except Exception:
        # La lista se redibujó mientras tanto.
        pass


def dibujar_lista_libre():
    """Repinta la lista completa de sonidos y gestos."""
    for hijo in lista_libre.winfo_children():
        hijo.destroy()

    filas_libre.clear()

    total = len(instrumento_libre.ranuras)

    libre_contador.configure(text=f"{total} de 8 sonidos")

    if not instrumento_libre.ranuras:
        vacio = ctk.CTkFrame(lista_libre, fg_color="transparent")
        vacio.pack(expand=True, pady=50)

        ctk.CTkLabel(
            vacio,
            text="🎹",
            font=("Segoe UI Emoji", 46)
        ).pack()

        ctk.CTkLabel(
            vacio,
            text="Todavía no hay sonidos.\n"
                 "Pulsa «Agregar sonido» para empezar.",
            font=("Montserrat", 14),
            text_color=GRIS,
            justify="center"
        ).pack(pady=(4, 0))
        return

    for indice, ranura in enumerate(instrumento_libre.ranuras):

        fila = ctk.CTkFrame(
            lista_libre,
            fg_color="#111111",
            corner_radius=12
        )

        fila.pack(fill="x", pady=4)

        filas_libre.append(fila)

        arriba = ctk.CTkFrame(fila, fg_color="transparent")
        arriba.pack(fill="x", padx=10, pady=(9, 2))

        # Número de la ranura: ayuda a saber de cuál habla el mensaje
        # de error cuando dos gestos se parecen demasiado.
        ctk.CTkLabel(
            arriba,
            text=str(indice + 1),
            width=26,
            height=26,
            corner_radius=13,
            fg_color="#2B2B2B",
            text_color=GRIS,
            font=("Montserrat", 12, "bold")
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            arriba,
            text=SONIDO_A_ETIQUETA.get(ranura["sonido"], "🎵 DO"),
            width=150,
            height=34,
            corner_radius=8,
            fg_color="#2B2B2B",
            hover_color=MORADO,
            font=("Montserrat", 13, "bold"),
            command=lambda i=indice: abrir_teclado_sonidos(i)
        ).pack(side="left")

        # Todos los controles a 34 de alto: antes los botones medían 50
        # y el menú 28, así que la fila se veía escalonada.
        ctk.CTkButton(
            arriba,
            text="🗑",
            width=34,
            height=34,
            fg_color="#2B2B2B",
            hover_color="#B22222",
            font=("Montserrat", 14),
            command=lambda i=indice: quitar_sonido_libre(i)
        ).pack(side="right", padx=(6, 0))

        ctk.CTkButton(
            arriba,
            text="🎬 Grabar" if ranura["gesto"] is None else "🔄 Regrabar",
            width=104,
            height=34,
            fg_color=AZUL if ranura["gesto"] is None else "#2B2B2B",
            hover_color=MORADO,
            font=("Montserrat", 13, "bold"),
            command=lambda i=indice: iniciar_grabacion_gesto(i)
        ).pack(side="right")

        if ranura["gesto"] is None:
            texto = "●  Sin gesto grabado"
            color = "#FF9F43"
        else:
            texto = f"●  {ranura['texto']}"
            color = VERDE

        ctk.CTkLabel(
            fila,
            text=texto,
            font=("Montserrat", 12, "bold"),
            text_color=color,
            anchor="w"
        ).pack(anchor="w", padx=14, pady=(0, 9))


def abrir_teclado_sonidos(indice):
    """
    Teclado para elegir el sonido de una ranura.

    Antes era un menú desplegable. Un teclado se entiende sin leer nada
    y se atina con el mouse mucho más rápido.
    """
    if not (0 <= indice < len(instrumento_libre.ranuras)):
        return

    # ---- medidas del teclado ----

    ANCHO_BLANCA = 62
    ALTO_BLANCA = 200
    ANCHO_NEGRA = 40
    ALTO_NEGRA = 126

    blancas = ["DO", "RE", "MI", "FA", "SOL", "LA", "SI"]

    # Después de cuáles teclas blancas viene una negra.
    # MI y SI no tienen: por eso el piano se ve agrupado en 2 y 3.
    lleva_negra = [True, True, False, True, True, True, False]

    ancho_teclado = len(blancas) * ANCHO_BLANCA

    ventana = ctk.CTkToplevel(app)
    ventana.title("Elegir sonido")
    ventana.geometry(f"{ancho_teclado + 90}x{ALTO_BLANCA + 250}")
    ventana.configure(fg_color=NEGRO)
    ventana.resizable(False, False)
    ventana.transient(app)
    ventana.grab_set()

    ctk.CTkLabel(
        ventana,
        text=f"Sonido del gesto {indice + 1}",
        font=("Montserrat", 26, "bold"),
        text_color=BLANCO
    ).pack(pady=(20, 2))

    ctk.CTkLabel(
        ventana,
        text="Toca una tecla para escucharla y asignarla.",
        font=("Montserrat", 15),
        text_color=GRIS
    ).pack(pady=(0, 18))

    actual = instrumento_libre.ranuras[indice]["sonido"]

    def elegir(clave):
        instrumento_libre.cambiar_sonido(indice, clave)
        motor_notas.tocar_notas(SONIDOS_LIBRES[clave]["notas"])

        dibujar_lista_libre()
        guardar_instrumento_libre()

        ventana.destroy()

    # Las teclas se colocan con place(), y place() no le avisa al marco
    # cuánto espacio necesita. Sin estas dos medidas el marco se queda
    # con su tamaño por omisión y el teclado sale cortado a la mitad.
    teclado = ctk.CTkFrame(
        ventana,
        fg_color="transparent",
        width=ancho_teclado,
        height=ALTO_BLANCA
    )

    teclado.pack()
    teclado.pack_propagate(False)

    # ---- teclas blancas ----

    for posicion, nombre in enumerate(blancas):

        clave = nombre

        # Un punto en las teclas que sí mueven un relevador del piano.
        etiqueta = clave if clave in RELEVADOR_POR_NOTA else f"{clave}\n·"

        ctk.CTkButton(
            teclado,
            text=etiqueta,
            width=ANCHO_BLANCA - 3,
            height=ALTO_BLANCA,
            corner_radius=6,
            fg_color=AZUL if clave == actual else "#F2F2F2",
            hover_color=MORADO,
            text_color=BLANCO if clave == actual else "#333333",
            font=("Montserrat", 14, "bold"),
            anchor="s",
            command=lambda c=clave: elegir(c)
        ).place(x=posicion * ANCHO_BLANCA, y=0)

    # ---- teclas negras, encima y corridas a la derecha ----

    for posicion, nombre in enumerate(blancas):

        if not lleva_negra[posicion]:
            continue

        clave = f"{nombre}#"

        # Los sostenidos que no tienen relevador se ven apagados.
        tiene_piano = clave in RELEVADOR_POR_NOTA

        ctk.CTkButton(
            teclado,
            text=clave,
            width=ANCHO_NEGRA,
            height=ALTO_NEGRA,
            corner_radius=4,
            fg_color=MORADO if clave == actual else "#1A1A1A",
            hover_color=MORADO2,
            text_color=BLANCO if tiene_piano else "#6A6A6A",
            font=("Montserrat", 11, "bold"),
            anchor="s",
            command=lambda c=clave: elegir(c)
        ).place(
            x=posicion * ANCHO_BLANCA + ANCHO_BLANCA - ANCHO_NEGRA // 2,
            y=0
        )

    ctk.CTkLabel(
        ventana,
        text="Las notas en gris solo suenan en la computadora:\n"
             "el piano físico tiene ocho teclas.",
        font=("Montserrat", 12),
        text_color=GRIS
    ).pack(pady=(10, 0))

    ctk.CTkButton(
        ventana,
        text="Cancelar",
        width=170,
        height=42,
        fg_color="#3A3A3A",
        hover_color="#B22222",
        font=("Montserrat", 15, "bold"),
        command=ventana.destroy
    ).pack(pady=22)


def agregar_sonido_libre():
    if len(instrumento_libre.ranuras) >= 8:
        poner_estado_libre(
            "Ocho sonidos es el máximo.",
            "#FF9F43"
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
        poner_estado_libre(
            "Espera a que termine la calibración.",
            "#FF9F43"
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

    marcar_fila_libre(indice, "grabando")

    if cuenta_regresiva_libre > 0:
        poner_estado_libre(
            f"Prepara tu gesto...   {cuenta_regresiva_libre}",
            MORADO
        )

        cuenta_regresiva_libre -= 1

        app.after(1000, lambda: contar_para_grabar(indice))
        return

    grabando_indice = indice

    poner_estado_libre("🎬 ¡Mantén el gesto!", AZUL)


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
        marcar_fila_libre(None)
        poner_estado_libre(f"✖ {motivo}", "#FF9F43")
        return

    instrumento_libre.guardar_gesto(indice, promedio)

    dibujar_lista_libre()
    guardar_instrumento_libre()

    poner_estado_libre(
        f"✓ Guardado: {instrumento_libre.ranuras[indice]['texto']}",
        VERDE
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
            marcar_fila_libre(None)
            poner_estado_libre("Haz uno de tus gestos.", GRIS)

        instrumento_libre.ultimo_sonando = None
        return

    # Mientras se sostenga el mismo gesto, no se repite el sonido.
    if indice == instrumento_libre.ultimo_sonando:
        return

    instrumento_libre.ultimo_sonando = indice

    ranura = instrumento_libre.ranuras[indice]
    sonido = SONIDOS_LIBRES[ranura["sonido"]]

    motor_notas.tocar_notas(sonido["notas"])

    marcar_fila_libre(indice, "sonando")

    poner_estado_libre(
        f"{sonido['etiqueta']}   ←   {ranura['texto']}",
        AZUL
    )

    if "estado_nota" in globals():
        estado_nota.configure(text=sonido["etiqueta"], text_color=AZUL)


ctk.CTkButton(
    botones_libre,
    text="➕ Agregar sonido",
    height=42,
    corner_radius=12,
    fg_color=AZUL,
    hover_color=MORADO,
    font=("Montserrat", 15, "bold"),
    command=agregar_sonido_libre
).pack(fill="x")

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

    mostrar_sonido_terapia(True)

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

    mostrar_sonido_terapia(False)

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

    mostrar_sonido_terapia(False)

    modo_actual.configure(text="🎛 Modo Libre")

    # Recuperamos el instrumento que esta persona ya había armado.
    if usuario_actual is not None:
        guardado = usuarios_db.get(usuario_actual, {}).get("instrumento_libre")

        if guardado:
            instrumento_libre.importar(guardado)

    dibujar_lista_libre()

    if not detector.calibrado:
        poner_estado_libre(
            "Espera a que termine la calibración.",
            "#FF9F43"
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
    ventana.title(f"Cómo le ha ido - {usuario_actual}")
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