import customtkinter as ctk
from PIL import Image
import cv2
import mediapipe as mp
import threading
from PIL import ImageTk
import json
import os
import sqlite3
import csv
from tkinter import messagebox
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
NEGRO2 = "#1A1A1A"
BLANCO = "#FFFFFF"
AZUL = "#3B9DFF"
MORADO = "#9B4DFF"
GRIS = "#BFBFBF"
GRIS2 = "#2B2B2B"
GRIS3 = "#5A5A5A"
GRIS4 = "#333333"
MORADO2 = "#5D2AA6"
ROJO = "#FF0000"
AMARILLO = "#FFD63D"
VERDE = "#4CF405"
CELESTE = "#05CEFA"
NARANJA = "#FF9F43"
ROJOOS = "#B22222"

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
# BASE DE DATOS SQL TALAT
# ==========================================
# La app conserva su estructura interna de diccionarios para que toda la
# interfaz y la lógica existente sigan funcionando igual, pero la
# persistencia real queda en SQLite.
BASE_DATOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos")
os.makedirs(BASE_DATOS_DIR, exist_ok=True)
ARCHIVO_BD = os.path.join(BASE_DATOS_DIR, "talat.db")
ARCHIVO_USUARIOS = "usuarios.json"          # solo migración inicial
ARCHIVO_CSV_LEGADO = "graficaparaimprimir_talat.csv"  # solo migración/exportación

def conexion_bd():
    conn = sqlite3.connect(ARCHIVO_BD, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def inicializar_base_datos():
    with conexion_bd() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS usuarios (
                id_usuario INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL UNIQUE,
                edad TEXT DEFAULT '',
                motivo TEXT DEFAULT '',
                comentarios TEXT DEFAULT '',
                sesiones INTEGER DEFAULT 0,
                tiempo_total INTEGER DEFAULT 0,
                expresiones INTEGER DEFAULT 0,
                notas INTEGER DEFAULT 0,
                canciones_completadas INTEGER DEFAULT 0,
                ultima_sesion TEXT DEFAULT '',
                fecha_alta TEXT DEFAULT CURRENT_TIMESTAMP,
                activo INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS sesiones (
                id_sesion INTEGER PRIMARY KEY AUTOINCREMENT,
                id_usuario INTEGER NOT NULL,
                fecha TEXT NOT NULL,
                duracion_segundos INTEGER DEFAULT 0,
                total_expresiones INTEGER DEFAULT 0,
                total_notas INTEGER DEFAULT 0,
                FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS expresiones (
                id_expresion INTEGER PRIMARY KEY AUTOINCREMENT,
                id_sesion INTEGER NOT NULL,
                emocion TEXT NOT NULL,
                emoji TEXT DEFAULT '',
                apertura_boca REAL,
                ancho_boca REAL,
                apertura_ojos REAL,
                altura_ceja REAL,
                cercania_cejas REAL,
                curva_boca REAL,
                comparacion TEXT DEFAULT '',
                momento TEXT DEFAULT '',
                FOREIGN KEY (id_sesion) REFERENCES sesiones(id_sesion) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS instrumento_libre (
                id_instrumento INTEGER PRIMARY KEY AUTOINCREMENT,
                id_usuario INTEGER NOT NULL UNIQUE,
                datos_json TEXT NOT NULL DEFAULT '[]',
                fecha_actualizacion TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS canciones_completadas (
                id_cancion_completada INTEGER PRIMARY KEY AUTOINCREMENT,
                id_usuario INTEGER NOT NULL,
                clave_cancion TEXT NOT NULL,
                titulo TEXT DEFAULT '',
                aciertos INTEGER DEFAULT 0,
                fecha TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS eventos (
                id_evento INTEGER PRIMARY KEY AUTOINCREMENT,
                id_usuario INTEGER,
                fecha_hora TEXT NOT NULL,
                registro TEXT NOT NULL,
                descripcion TEXT DEFAULT '',
                duracion_min TEXT DEFAULT '',
                expresiones TEXT DEFAULT '',
                expresion TEXT DEFAULT '',
                intensidad TEXT DEFAULT '',
                comparacion TEXT DEFAULT '',
                observaciones TEXT DEFAULT '',
                FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario) ON DELETE SET NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sesiones_usuario ON sesiones(id_usuario);
            CREATE INDEX IF NOT EXISTS idx_expresiones_sesion ON expresiones(id_sesion);
            CREATE INDEX IF NOT EXISTS idx_eventos_usuario ON eventos(id_usuario);
            CREATE INDEX IF NOT EXISTS idx_eventos_fecha ON eventos(fecha_hora);
        """)

def migrar_json_y_csv_anteriores():
    """Migra una instalación anterior una sola vez, sin volver a escribir JSON."""
    try:
        with conexion_bd() as conn:
            if conn.execute("SELECT COUNT(*) FROM usuarios").fetchone()[0] == 0 and os.path.exists(ARCHIVO_USUARIOS):
                with open(ARCHIVO_USUARIOS, "r", encoding="utf-8") as archivo:
                    datos = json.load(archivo)
                if isinstance(datos, dict):
                    for nombre, usuario in datos.items():
                        if not isinstance(usuario, dict):
                            continue
                        conn.execute("""
                            INSERT OR IGNORE INTO usuarios
                            (nombre, edad, motivo, comentarios, sesiones, tiempo_total,
                             expresiones, notas, canciones_completadas, ultima_sesion)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (nombre, usuario.get("edad", ""), usuario.get("motivo", ""),
                              usuario.get("comentarios", ""), usuario.get("sesiones", 0),
                              usuario.get("tiempo_total", 0), usuario.get("expresiones", 0),
                              usuario.get("notas", 0), usuario.get("canciones_completadas", 0),
                              usuario.get("ultima_sesion", "")))
                        uid = conn.execute("SELECT id_usuario FROM usuarios WHERE nombre=?", (nombre,)).fetchone()[0]
                        for sesion in usuario.get("historial_sesiones", []) or []:
                            cur = conn.execute("""
                                INSERT INTO sesiones (id_usuario, fecha, duracion_segundos, total_expresiones, total_notas)
                                VALUES (?, ?, ?, ?, ?)
                            """, (uid, sesion.get("fecha", ""), sesion.get("duracion_segundos", 0),
                                  sesion.get("total_expresiones", 0), sesion.get("total_notas", 0)))
                            sid = cur.lastrowid
                            for exp in sesion.get("expresiones", []) or []:
                                conn.execute("""
                                    INSERT INTO expresiones
                                    (id_sesion, emocion, emoji, apertura_boca, ancho_boca, apertura_ojos,
                                     altura_ceja, cercania_cejas, curva_boca, comparacion, momento)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (sid, exp.get("emocion", ""), exp.get("emoji", ""), exp.get("apertura_boca"),
                                      exp.get("ancho_boca"), exp.get("apertura_ojos"), exp.get("altura_ceja"),
                                      exp.get("cercania_cejas"), exp.get("curva_boca"), exp.get("comparacion", ""),
                                      exp.get("momento", "")))
                        instrumento = usuario.get("instrumento_libre", []) or []
                        if instrumento:
                            conn.execute("""
                                INSERT OR REPLACE INTO instrumento_libre (id_usuario, datos_json, fecha_actualizacion)
                                VALUES (?, ?, CURRENT_TIMESTAMP)
                            """, (uid, json.dumps(instrumento, ensure_ascii=False)))
            if conn.execute("SELECT COUNT(*) FROM eventos").fetchone()[0] == 0 and os.path.exists(ARCHIVO_CSV_LEGADO):
                with open(ARCHIVO_CSV_LEGADO, "r", encoding="utf-8-sig", newline="") as archivo:
                    for fila in csv.DictReader(archivo):
                        nombre = (fila.get("Persona") or "-").strip()
                        encontrado = conn.execute("SELECT id_usuario FROM usuarios WHERE nombre=?", (nombre,)).fetchone() if nombre not in ("", "-") else None
                        uid = encontrado[0] if encontrado else None
                        try:
                            fh = datetime.strptime(f"{fila.get('Fecha','')} {fila.get('Hora','')}", "%d/%m/%Y %H:%M").strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            fh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        conn.execute("""
                            INSERT INTO eventos
                            (id_usuario, fecha_hora, registro, descripcion, duracion_min, expresiones,
                             expresion, intensidad, comparacion, observaciones)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (uid, fh, fila.get("Registro", ""), fila.get("Qué ocurrió", ""),
                              fila.get("Duración (min)", ""), fila.get("Expresiones logradas", ""),
                              fila.get("Expresión trabajada", ""), fila.get("Intensidad", ""),
                              fila.get("Comparación con la vez anterior", ""), fila.get("Observaciones de la terapeuta", "")))
    except Exception as e:
        print("No se pudo completar la migración a SQL:", e)

inicializar_base_datos()
migrar_json_y_csv_anteriores()

# ==========================================
# IDIOMA
# ==========================================
#
ARCHIVO_CONFIGURACION = "configuracion.json"

TRADUCCION_EN = {
    # --- bienvenida ---
    "Bienvenido": "Welcome",
    "🎹 INICIO": "🎹 START",
    "⚙ Idioma": "⚙ Language",
    "La música al alcance de todos": "Music within everyone's reach",
    "Sistema Musical TALAT": "TALAT Music System",

    # --- usuarios ---
    "Usuarios": "Users",
    "➕ Agregar usuario": "➕ Add user",
    "📊 Ver perfil": "📊 View profile",
    "← Bienvenido": "← Welcome",
    "📄 Gráfica para imprimir": "📄 Printable graph",
    "Nuevo usuario": "New user",
    "Editar usuario": "Edit user",
    "✏ Editar usuario": "✏ Edit user",
    "🗑 Eliminar usuario": "🗑 Delete user",
    "▶ Iniciar sesión": "▶ Start session",
    "Nombre": "Name",
    "Edad": "Age",
    "Nombre de la persona": "Person's name",
    "Años cumplidos": "Years old",
    "¿Por qué usa TALAT?": "Why do they use TALAT?",
    "Escríbelo con tus palabras. Este texto aparecerá en su perfil.":
        "Write it in your own words. It will appear in their profile.",
    "Guardar": "Save",
    "Cancelar": "Cancel",
    "Cerrar": "Close",
    "El nombre no puede quedar vacío.": "The name cannot be empty.",
    "La edad debe ser un número.": "Age must be a number.",
    "La edad debe estar entre 1 y 120.": "Age must be between 1 and 120.",
    "Aún no hay usuarios registrados.\n\nPresiona 'Agregar usuario' para comenzar.":
        "No users yet.\n\nPress 'Add user' to begin.",
    "Sin datos personales · edítalos desde su perfil":
        "No personal data · edit it from their profile",

    # --- sesión ---
    "🎵 Sesión TALAT": "🎵 TALAT Session",
    "⏹ Terminar sesión": "⏹ End session",
    "🔧 Probar piano": "🔧 Test piano",
    "📷 Cámara TALAT": "📷 TALAT Camera",
    "🔌 Buscando el piano...": "🔌 Looking for the piano...",
    "🔌 No hay piano conectado": "🔌 No piano connected",
    "🧠 Modo Terapia": "🧠 Therapy Mode",
    "🎼 Canción Guiada": "🎼 Guided Song",
    "🎛 Modo Libre": "🎛 Free Mode",
    "SONIDO": "SOUND",
    "🎵 Notas": "🎵 Notes",
    "🎹 Acordes": "🎹 Chords",
    "REPOSO": "AT REST",
    "CALIBRANDO": "CALIBRATING",
    "ALEGRÍA": "JOY",
    "SORPRESA": "SURPRISE",
    "TRISTEZA": "SADNESS",
    "IRA": "ANGER",
    "ABURRIMIENTO": "BOREDOM",
    "Haz una expresión": "Make an expression",
    "Mantén la expresión": "Hold the expression",
    "¡Muy bien!": "Very good!",
    "Excelente": "Excellent",
    "Muy bien": "Well done",
    "¡Excelente!": "Excellent!",
    "Mantén tu rostro relajado": "Keep your face relaxed",
    "Mantén tu rostro relajado...": "Keep your face relaxed...",
    "Calibración completada": "Calibration complete",
    "INSTRUCCIÓN: Haz una expresión clara y mantenla unos instantes.":
        "INSTRUCTION: Make a clear expression and hold it for a moment.",
    "INSTRUCCIÓN: No sonrías ni hagas gestos. Mira al frente y mantén el rostro relajado.":
        "INSTRUCTION: Don't smile or make faces. Look ahead and keep your face relaxed.",
    "INSTRUCCIÓN: Mantén tu rostro relajado mientras TALAT calibra tu rostro.":
        "INSTRUCTION: Keep your face relaxed while TALAT calibrates it.",

    # --- canción guiada ---
    "🎼 ELIGE UNA CANCIÓN": "🎼 CHOOSE A SONG",
    "Una sola expresión toca una frase completa de la melodía.":
        "A single expression plays a whole phrase of the melody.",
    "▶ TOCAR ESTA CANCIÓN": "▶ PLAY THIS SONG",
    "← Canciones": "← Songs",
    "▶ INICIAR": "▶ START",
    "⏸ PAUSA": "⏸ PAUSE",
    "↻ REINICIAR": "↻ RESTART",
    "Fácil": "Easy",
    "Media": "Medium",
    "Larga": "Long",
    "Esperando tu expresión...": "Waiting for your expression...",
    "Relaja la cara para la siguiente frase": "Relax your face for the next phrase",
    "Pulsa ▶ INICIAR para empezar": "Press ▶ START to begin",
    "⏸ En pausa": "⏸ Paused",
    "Haz esta expresión una vez": "Make this expression once",
    "Escucha y descansa...": "Listen and rest...",
    "Puedes relajar la cara": "You can relax your face",
    "Sigue las notas amarillas mientras suena la melodía.":
        "Follow the yellow notes while the melody plays.",
    "🎉 ¡CANCIÓN COMPLETADA!": "🎉 SONG COMPLETED!",
    "Cara de alegría": "Happy face",
    "Cara de sorpresa": "Surprised face",
    "Cara de enojo": "Angry face",
    "Cara de tristeza": "Sad face",
    "Cara de aburrimiento": "Bored face",
    "Sonríe estirando la boca y subiendo las esquinas.":
        "Smile, stretching your mouth and lifting the corners.",
    "Abre la boca y sube las cejas, con los ojos bien abiertos.":
        "Open your mouth and raise your eyebrows, eyes wide open.",
    "Frunce el ceño juntando las cejas hacia abajo.":
        "Frown, pulling your eyebrows down and together.",
    "Baja las esquinas de la boca y saca un poco el labio de abajo.":
        "Lower the corners of your mouth and push out your bottom lip.",
    "Entrecierra los ojos y relaja la cara, como si tuvieras sueño.":
        "Narrow your eyes and relax your face, as if you were sleepy.",
    "frases": "phrases",
    "segundos de música": "seconds of music",
    "2 frases · 2 expresiones": "2 phrases · 2 expressions",
    "3 frases · 3 expresiones": "3 phrases · 3 expressions",

    # --- modo libre ---
    "🎛 MODO LIBRE": "🎛 FREE MODE",
    "Tú eliges el sonido y tú grabas la cara que lo toca.":
        "You choose the sound and record the face that plays it.",
    "➕ Agregar sonido": "➕ Add sound",
    "Agrega un sonido y graba su gesto.": "Add a sound and record its gesture.",
    "Haz uno de tus gestos.": "Make one of your gestures.",
    "🎬 ¡Mantén el gesto!": "🎬 Hold the gesture!",
    "●  Sin gesto grabado": "●  No gesture recorded",
    "🎬 Grabar": "🎬 Record",
    "🔄 Regrabar": "🔄 Re-record",
    "Ocho sonidos es el máximo.": "Eight sounds is the maximum.",
    "Espera a que termine la calibración.": "Wait for calibration to finish.",
    "Todavía no hay sonidos.\nPulsa «Agregar sonido» para empezar.":
        "No sounds yet.\nPress «Add sound» to begin.",
    "Elegir sonido": "Choose sound",
    "Toca una tecla para escucharla y asignarla.":
        "Tap a key to hear it and assign it.",
    "Las notas en gris solo suenan en la computadora:\nel piano físico tiene ocho teclas.":
        "Grey notes only play on the computer:\nthe physical piano has eight keys.",
    "Ese gesto es muy suave. Haz un movimiento más marcado.":
        "That gesture is too soft. Make a stronger movement.",
    "No se detectó ningún movimiento.": "No movement was detected.",

    # --- perfil ---
    "← Usuarios": "← Users",
    "Resumen de tu práctica en TALAT": "Summary of your TALAT practice",
    "🪪 Datos de la persona": "🪪 Personal information",
    "Edad: sin registrar": "Age: not recorded",
    "Sin registrar": "Not recorded",
    "📊 Estadísticas": "📊 Statistics",
    "Tu progreso en las últimas 10 sesiones": "Your progress over the last 10 sessions",
    "Expresiones": "Expressions",
    "Minutos": "Minutes",
    "🔍 Ver cómo te ha ido": "🔍 See how it has gone",
    "📝 Comentarios": "📝 Notes",
    "Tiempo total: 0 min": "Total time: 0 min",
    "Sesiones realizadas: 0": "Sessions completed: 0",
    "Última sesión: Sin sesiones": "Last session: No sessions",
    "Expresiones registradas: 0": "Expressions recorded: 0",
    "Notas detectadas: 0": "Notes detected: 0",
    "Cada barra responde una pregunta sencilla sobre lo que hizo tu cara. Entre más llena y más verde, más marcado te salió el movimiento.":
        "Each bar answers a simple question about what your face did. The fuller and greener it is, the stronger the movement was.",
    "Tiempo total:": "Total time:",
    "Sesiones realizadas:": "Sessions completed:",
    "Expresiones registradas:": "Expressions recorded:",
    "Notas detectadas:": "Notes detected:",
    "Última sesión:": "Last session:",
    "Sin sesiones": "No sessions",

    # --- bitácora ---
    "📄  Gráfica para imprimir de TALAT": "📄  TALAT printable graph",
    "📊 Abrir en Excel": "📊 Open in Excel",
    "📊 Abrir el registro (CSV)": "📊 Open the log (CSV)",
    "📄 Gráfica para imprimir": "📄 Activity log",

    # --- hoja para imprimir ---
    "📄 Hoja para imprimir": "📄 Printable sheet",
    "Hoja para imprimir": "Printable sheet",
    "Primero abre el perfil de una persona.": "Open someone's profile first.",
    "Todavía no hay sesiones de esta persona.": "This person has no sessions yet.",
    "Cierra el archivo en Excel y vuelve a intentarlo.":
        "Close the file in Excel and try again.",
    "No se pudo generar el archivo": "The file could not be created",
    "El archivo se guardó aquí:": "The file was saved here:",
    "No se pudo abrir solo": "It could not be opened automatically",
    "Reporte generado": "Report generated",
    "Indicadores para la terapeuta": "Indicators for the therapist",
    "Progreso por sesión": "Progress by session",
    "Calidad de los movimientos": "Movement quality",
    "Promedio por sesión": "Average per session",
    "Duración promedio": "Average duration",
    "Movimiento más practicado": "Most practiced movement",
    "Cambio observado": "Observed change",
    "Sin cambio suficiente para comparar": "Not enough data to compare",
    "Primeras sesiones": "First sessions",
    "Sesiones recientes": "Recent sessions",
    "Reporte PDF de TALAT": "TALAT PDF Report",
    "📄 Generar reporte PDF": "📄 Generate PDF report",
    "Reporte de actividad TALAT": "TALAT Activity Report",
    "Resumen general": "General summary",
    "Historial de sesiones": "Session history",
    "Actividad por movimiento": "Activity by movement",
    "Observaciones de la terapeuta": "Therapist's notes",
    "Fecha": "Date",
    "Duración": "Duration",
    "Expresiones realizadas": "Expressions completed",
    "Sesión": "Session",
    "Veces realizada": "Times completed",
    "Intensidad": "Intensity",
    "Sin datos": "No data",
    "Reporte generado correctamente.": "Report generated successfully.",
    "Todavía no hay nada registrado.": "Nothing recorded yet.",
    "Todavía no hay nada registrado.\n\nInicia una sesión y vuelve a entrar aquí.":
        "Nothing recorded yet.\n\nStart a session and come back here.",
    "Abriendo...": "Opening...",

    # --- idioma ---
    "🌎  Idioma": "🌎  Language",
    "Se guarda y la próxima vez TALAT abre en ese idioma.":
        "It is saved and TALAT will open in that language next time.",
    "Pronto habrá más idiomas.": "More languages coming soon.",

    # --- textos dinámicos y ventanas secundarias ---
    "Todo queda guardado en:": "Everything is saved in:",
    "registros en total": "total records",
    "se muestran los 300 más recientes": "showing the 300 most recent",
    "Inicio de sesión": "Session started",
    "Fin de sesión": "Session ended",
    "Expresión lograda": "Expression achieved",
    "Actividad completada": "Activity completed",
    "Gesto personalizado": "Custom gesture",
    "Alta de la persona": "Person added",
    "Equipo": "Equipment",
    "Configuración": "Settings",
    "Marcada": "Strong",
    "Moderada": "Moderate",
    "Leve": "Mild",
    "Aún no hay sesiones.\nInicia una sesión para ver tu progreso.":
        "No sessions yet.\nStart a session to see your progress.",
    "Todavía no hay sesiones.\n\nInicia una sesión y haz alguna expresión para que aquí aparezcan tus mediciones.":
        "No sessions yet.\n\nStart a session and make an expression to see your measurements here.",
    "Primera vez que haces esta expresión.": "First time making this expression.",
    "Te salió igual que la vez anterior.": "It was the same as last time.",
    "Te salió mejor que la vez anterior.": "You did better than last time.",
    "Esta vez te salió un poco menos marcado.": "This time it was a little less pronounced.",
    "Puedes relajar la cara": "You can relax your face",
    "Pulsa ↻ REINICIAR para tocarla otra vez,\nla expresión volverá a aparecer.":
        "Press ↻ RESTART to play it again,\nthe expression will appear again.",
    "FRASE 1 DE 1": "PHRASE 1 OF 1",
    "notas": "notes",
    "frases": "phrases",
    "segundos de música": "seconds of music",
    "SONANDO LA FRASE": "PLAYING PHRASE",
    "Ahora toca:": "Now play:",
    "¿Seguro que quieres eliminar este usuario?": "Are you sure you want to delete this user?",
    "El usuario fue eliminado.": "The user was deleted.",
    "El usuario no pudo eliminarse.": "The user could not be deleted.",
    "No se encontró el usuario.": "The user was not found.",
    "0 de 8 sonidos": "0 of 8 sounds"

}

TRADUCCION_ES = {v: k for k, v in TRADUCCION_EN.items()}

TEXTOS = {
    "es": {"_nombre": "Español", "_bandera": "🇲🇽"},
    "en": {"_nombre": "English", "_bandera": "🇺🇸"}
}

IDIOMA = "es"


def cargar_configuracion():
    if not os.path.exists(ARCHIVO_CONFIGURACION):
        return {}

    try:
        with open(ARCHIVO_CONFIGURACION, "r", encoding="utf-8") as archivo:
            datos = json.load(archivo)
            return datos if isinstance(datos, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def guardar_configuracion(datos):
    try:
        with open(ARCHIVO_CONFIGURACION, "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo, ensure_ascii=False, indent=4)
    except OSError as e:
        print("No se pudo guardar la configuración:", e)


configuracion = cargar_configuracion()

if configuracion.get("idioma") in TEXTOS:
    IDIOMA = configuracion["idioma"]


def traducir(texto):
    """
    Pasa un texto al idioma actual.

    Si no está en la tabla lo deja igual: así los nombres propios
    (las canciones, las notas DO-RE-MI) nunca se tocan.
    """
    if not isinstance(texto, str) or not texto.strip():
        return texto

    limpio = texto.strip()

    if IDIOMA == "en":
        return TRADUCCION_EN.get(limpio, texto)

    return TRADUCCION_ES.get(limpio, texto)


def t(clave):
    """Se conserva por compatibilidad: traduce el texto que reciba."""
    return traducir(clave)


def traducir_ventana(ventana):
    """Traduce textos visibles, opciones y títulos de ventanas."""
    try:
        titulo = ventana.title()
        nuevo_titulo = traducir(titulo)
        if nuevo_titulo != titulo:
            ventana.title(nuevo_titulo)
    except Exception:
        pass

    try:
        texto = ventana.cget("text")
        nuevo = traducir(texto)
        if nuevo != texto:
            ventana.configure(text=nuevo)
    except Exception:
        pass

    try:
        valores = ventana.cget("values")
        if valores:
            actual = ventana.get()
            ventana.configure(values=[traducir(v) for v in valores])
            ventana.set(traducir(actual))
    except Exception:
        pass

    try:
        for hijo in ventana.winfo_children():
            traducir_ventana(hijo)
    except Exception:
        pass


# ==========================================
# GRAFICA PARA IMPRMIR
# ==========================================

ARCHIVO_GRAFICA_PARA_IMPRIMIR = "graficaparaimprimir_talat.csv"

COLUMNAS_GRAFICA_PARA_IMPRIMIR = [
    "Fecha", "Hora", "Persona", "Edad",
    "Registro", "Qué ocurrió",
    "Duración (min)", "Expresiones logradas",
    "Expresión trabajada", "Intensidad",
    "Comparación con la vez anterior",
    "Observaciones de la terapeuta"
]

def registrar_evento(registro, descripcion="", usuario=None,
                     duracion_min="", expresiones="",
                     expresion="", intensidad="", comparacion=""):
    """Guarda la bitácora en SQL. La app sigue usando la misma función."""
    try:
        usuario = usuario if usuario is not None else (usuario_actual or "-")
        with conexion_bd() as conn:
            fila = conn.execute("SELECT id_usuario FROM usuarios WHERE nombre=?", (usuario,)).fetchone() if usuario != "-" else None
            uid = fila[0] if fila else None
            conn.execute("""
                INSERT INTO eventos
                (id_usuario, fecha_hora, registro, descripcion, duracion_min, expresiones,
                 expresion, intensidad, comparacion, observaciones)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (uid, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), registro, descripcion,
                  duracion_min, expresiones, expresion, intensidad, comparacion, ""))
    except Exception as e:
        print("No se pudo guardar el evento SQL:", e)

def exportar_bitacora_csv():
    """Genera el CSV desde SQL únicamente cuando la interfaz lo necesita."""
    try:
        with conexion_bd() as conn:
            filas = conn.execute("""
                SELECT strftime('%d/%m/%Y', e.fecha_hora), strftime('%H:%M', e.fecha_hora),
                       COALESCE(u.nombre, '-'), COALESCE(u.edad, ''), e.registro, e.descripcion,
                       e.duracion_min, e.expresiones, e.expresion, e.intensidad,
                       e.comparacion, e.observaciones
                FROM eventos e LEFT JOIN usuarios u ON u.id_usuario=e.id_usuario
                ORDER BY e.id_evento ASC
            """).fetchall()
        with open(ARCHIVO_GRAFICA_PARA_IMPRIMIR, "w", newline="", encoding="utf-8-sig") as archivo:
            escritor=csv.writer(archivo)
            escritor.writerow(COLUMNAS_GRAFICA_PARA_IMPRIMIR)
            escritor.writerows(filas)
        return True
    except Exception as e:
        print("No se pudo generar la bitácora CSV:", e)
        return False
def intensidad_en_palabras(emocion, medicion):
    """
    Traduce la medición a Leve / Moderada / Marcada.

    Mira la medición que de verdad importa en esa expresión: en la
    alegría la sonrisa, en el enojo el ceño. Un número de cinco
    decimales no le dice nada a nadie; esta palabra sí.
    """
    try:
        clave = LO_QUE_IMPORTA.get(emocion, ["apertura_boca"])[0]

        if clave not in medicion:
            return ""

        proporcion = porcentaje_medicion(clave, medicion[clave])

        if LOGRO_POR_MEDICION.get(clave, {}).get("invertida"):
            proporcion = 1.0 - proporcion

        if proporcion < 0.34:
            return "Leve"

        if proporcion < 0.67:
            return "Moderada"

        return "Marcada"

    except Exception:
        return ""


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
    """Carga desde SQL y devuelve el mismo diccionario que esperaba la app."""
    datos={}
    try:
        with conexion_bd() as conn:
            usuarios=conn.execute("""
                SELECT id_usuario,nombre,edad,motivo,comentarios,sesiones,tiempo_total,
                       expresiones,notas,canciones_completadas,ultima_sesion
                FROM usuarios WHERE activo=1 ORDER BY nombre COLLATE NOCASE
            """).fetchall()
            for uid,nombre,edad,motivo,comentarios,sesiones,tiempo_total,expresiones,notas,canciones,ultima in usuarios:
                historial=[]
                for sid,fecha,duracion,total_exp,total_notas in conn.execute("""
                    SELECT id_sesion,fecha,duracion_segundos,total_expresiones,total_notas
                    FROM sesiones WHERE id_usuario=? ORDER BY id_sesion
                """,(uid,)).fetchall():
                    exps=[]
                    for e in conn.execute("""
                        SELECT emocion,emoji,apertura_boca,ancho_boca,apertura_ojos,altura_ceja,
                               cercania_cejas,curva_boca,comparacion,momento
                        FROM expresiones WHERE id_sesion=? ORDER BY id_expresion
                    """,(sid,)).fetchall():
                        exps.append({"emocion":e[0],"emoji":e[1],"apertura_boca":e[2],"ancho_boca":e[3],
                                     "apertura_ojos":e[4],"altura_ceja":e[5],"cercania_cejas":e[6],
                                     "curva_boca":e[7],"comparacion":e[8],"momento":e[9]})
                    historial.append({"fecha":fecha,"duracion_segundos":duracion or 0,
                                      "total_expresiones":total_exp or 0,"total_notas":total_notas or 0,
                                      "expresiones":exps})
                inst=conn.execute("SELECT datos_json FROM instrumento_libre WHERE id_usuario=?",(uid,)).fetchone()
                try: instrumento=json.loads(inst[0]) if inst else []
                except Exception: instrumento=[]
                datos[nombre]={"sesiones":sesiones or 0,"tiempo_total":tiempo_total or 0,
                    "expresiones":expresiones or 0,"notas":notas or 0,"canciones_completadas":canciones or 0,
                    "ultima_sesion":ultima or "","comentarios":comentarios or "","historial_sesiones":historial,
                    "edad":edad or "","motivo":motivo or "","instrumento_libre":instrumento}
    except Exception as e:
        print("No se pudieron cargar los usuarios desde SQL:",e)
    return datos

def guardar_usuarios(datos):
    """Persiste en SQL el mismo modelo de datos que usa actualmente la interfaz."""
    try:
        with conexion_bd() as conn:
            existentes={n:u for u,n in conn.execute("SELECT id_usuario,nombre FROM usuarios").fetchall()}
            for uid,nombre in list(existentes.items()):
                if nombre not in datos:
                    conn.execute("UPDATE usuarios SET activo=0 WHERE id_usuario=?",(uid,))
            for nombre,u in datos.items():
                uid=existentes.get(nombre)
                vals=(nombre,u.get("edad",""),u.get("motivo",""),u.get("comentarios",""),u.get("sesiones",0),
                      u.get("tiempo_total",0),u.get("expresiones",0),u.get("notas",0),u.get("canciones_completadas",0),u.get("ultima_sesion",""))
                if uid:
                    conn.execute("""UPDATE usuarios SET nombre=?,edad=?,motivo=?,comentarios=?,sesiones=?,
                                    tiempo_total=?,expresiones=?,notas=?,canciones_completadas=?,ultima_sesion=?,activo=1
                                    WHERE id_usuario=?""",vals+(uid,))
                else:
                    uid=conn.execute("""INSERT INTO usuarios
                        (nombre,edad,motivo,comentarios,sesiones,tiempo_total,expresiones,notas,canciones_completadas,ultima_sesion)
                        VALUES (?,?,?,?,?,?,?,?,?,?)""",vals).lastrowid
                conn.execute("DELETE FROM sesiones WHERE id_usuario=?",(uid,))
                for sesion in u.get("historial_sesiones",[]) or []:
                    sid=conn.execute("""INSERT INTO sesiones
                        (id_usuario,fecha,duracion_segundos,total_expresiones,total_notas) VALUES (?,?,?,?,?)""",
                        (uid,sesion.get("fecha",""),sesion.get("duracion_segundos",0),sesion.get("total_expresiones",0),sesion.get("total_notas",0))).lastrowid
                    for e in sesion.get("expresiones",[]) or []:
                        conn.execute("""INSERT INTO expresiones
                            (id_sesion,emocion,emoji,apertura_boca,ancho_boca,apertura_ojos,altura_ceja,cercania_cejas,curva_boca,comparacion,momento)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",(sid,e.get("emocion",""),e.get("emoji",""),e.get("apertura_boca"),e.get("ancho_boca"),
                            e.get("apertura_ojos"),e.get("altura_ceja"),e.get("cercania_cejas"),e.get("curva_boca"),e.get("comparacion",""),e.get("momento","")))
                conn.execute("DELETE FROM instrumento_libre WHERE id_usuario=?",(uid,))
                instrumento=u.get("instrumento_libre",[]) or []
                if instrumento:
                    conn.execute("INSERT INTO instrumento_libre (id_usuario,datos_json,fecha_actualizacion) VALUES (?,?,CURRENT_TIMESTAMP)",
                                 (uid,json.dumps(instrumento,ensure_ascii=False)))
    except Exception as e:
        print("No se pudieron guardar los usuarios en SQL:",e)

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
    _datos.setdefault("edad", "")
    _datos.setdefault("motivo", "")
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
# NOTAS
# ==========================================


NOTAS = [
    "DO", "DO#", "RE", "RE#", "MI", "FA",
    "FA#", "SOL", "SOL#", "LA", "LA#", "SI"
]

FRECUENCIA_DO = 261.63

FRECUENCIAS_NOTAS = {
    nota: FRECUENCIA_DO * (2 ** (posicion / 12))
    for posicion, nota in enumerate(NOTAS)
}


RELEVADOR_POR_NOTA = {
    "DO": 8,
    "RE": 7,
    "MI": 6,
    "FA": 5,
    "SOL": 4,
    "LA": 3,
    "SI": 2,
    "FA#": 1
}

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
                for nota in notas:
                    frecuencia = FRECUENCIAS_NOTAS.get(nota)
                    if frecuencia:
                        self.winsound.Beep(int(frecuencia), 120)

        except Exception as e:
            print("No se pudo reproducir el sonido:", e)

    def enviar_a_hardware(self, notas, duracion_ms=None):
        """
        Manda las notas al piano físico.

        Van todas en un solo mensaje: si se enviaran una por una, las
        tres notas de un acorde llegarían separadas por milisegundos
        y se oiría como un arpegio rápido en vez de un acorde.

        La duración viaja con ellas: así una corchea le pide al piano
        media tecla de tiempo y una blanca el doble.
        """
        if piano_hardware is None or not piano_hardware.disponible():
            return

        try:
            if duracion_ms:
                piano_hardware.tocar(notas, duracion_ms)
            else:
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

    def tocar(self, nota, duracion_ms=None):
        """Una sola nota."""
        if nota in ("--", None):
            return

        self.tocar_notas([nota], duracion_ms)

    def tocar_notas(self, notas, duracion_ms=None):
        """Suena una o varias notas a la vez, sin congelar la interfaz."""
        threading.Thread(
            target=self._reproducir_notas,
            args=(list(notas),),
            daemon=True
        ).start()

        self.enviar_a_hardware(notas, duracion_ms)


motor_notas = MotorDeNotas()

# En segundo plano, para que la ventana abra igual de rápido.
threading.Thread(
    target=lambda: motor_notas.precargar(list(FRECUENCIAS_NOTAS)),
    daemon=True
).start()


# ==========================================
# PIANO FÍSICO (ARDUINO UNO + RELEVADORES)
# ==========================================

# Todo está envuelto en try: si no hay Arduino conectado, si falta la

#       PUERTO_ARDUINO = "COM5"          en Windows
#       PUERTO_ARDUINO = "/dev/ttyUSB0"  en Linux / Raspberry

PUERTO_ARDUINO = None      # None = buscarlo solo

DEPURAR_PIANO = True
VELOCIDAD_ARDUINO = 115200
DURACION_TECLA_MS = 600

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

        if PUERTO_ARDUINO:
            return self._abrir(PUERTO_ARDUINO)

        preferido = self.buscar_puerto()

        if preferido and self._abrir(preferido):
            return True

        try:
            from serial.tools import list_ports
            candidatos = [p.device for p in list_ports.comports()]
        except Exception:
            candidatos = []

        for puerto in candidatos:
            if puerto == preferido:
                continue

            if DEPURAR_PIANO:
                print(f"Probando {puerto}...")

            if self._abrir(puerto):
                return True

        if not self.error:
            self.error = "No respondió ningún puerto"

        return False

    def _abrir(self, puerto):
        """Abre un puerto y revisa si del otro lado está nuestro sketch."""
        import serial
        import time

        try:
            conexion = serial.Serial(
                puerto,
                VELOCIDAD_ARDUINO,
                timeout=0.2,
                write_timeout=0.5
            )

        except Exception as e:
            if DEPURAR_PIANO:
                print(f"  {puerto}: {e}")

            self.error = str(e)
            return False

        # Al abrir el puerto el Arduino se reinicia solo.
        # Si le hablamos antes de tiempo, pierde el primer mensaje.
        time.sleep(2.0)

        saludo = conexion.read_all().decode(errors="ignore")

        if DEPURAR_PIANO:
            print(f"  {puerto} saludó: {saludo.strip()!r}")

        self.conexion = conexion
        self.puerto = puerto

        if "TALAT" not in saludo:
            self.error = "Conectado, pero no saludó (¿sketch viejo?)"
        else:
            self.error = ""

        return True

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

            self.conexion.flush()

            if DEPURAR_PIANO:
                print("-> Arduino:", mensaje.strip())

            if self.conexion.in_waiting:
                respuesta = self.conexion.read_all()

                if DEPURAR_PIANO:
                    print("<- Arduino:", respuesta.decode(errors="ignore").strip())

        except Exception as e:
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
                    text_color=NARANJA
                )
            else:
                estado_piano.configure(
                    text=f"🎹 Piano conectado ({piano_hardware.puerto})",
                    text_color=VERDE
                )
        else:

            motivo = piano_hardware.error or "sin piano"

            estado_piano.configure(
                text=f"🔌 {motivo}",
                text_color=NARANJA
            )

        registrar_evento(
            "Equipo",
            piano_hardware.error or f"Piano conectado en {piano_hardware.puerto}",
            "-"
        )

        if DEPURAR_PIANO:
            print("Piano:", piano_hardware.error or "conectado en " + str(piano_hardware.puerto))

    app.after(0, avisar)


threading.Thread(target=conectar_piano, daemon=True).start()


# ==========================================
# ACORDES Y NOTAS POR GESTOS
# ==========================================

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

# ------------------------------------------
# EL RITMO
# ------------------------------------------

ARTICULACION = 0.85

# Por si alguna canción no trae pulso propio.
PULSO_MELODIA = 0.45

# Color de la etiqueta que aparece en el menú de canciones.
COLOR_DIFICULTAD = {
    "Fácil": VERDE,
    "Media": AMARILLO,
    "Larga": NARANJA
}

CANCIONES = {
    "cucaracha": {
        "titulo": "La cucaracha",
        "subtitulo": "2 frases · 2 expresiones",
        "dificultad": "Fácil",
        "pulso": 0.42,
        "frases": [
            {
                "letra": (
                    "La cucaracha, la cucaracha\n"
                    "ya no puede caminar"
                ),
                "gesto": "ALEGRÍA",
                "notas": [
                    # "La cu-ca-" van rápidas y "ra-cha" se asienta.
                    ("RE", 0.5), ("RE", 0.5), ("RE", 0.5),
                    ("SOL", 1.0), ("SI", 1.0),

                    ("RE", 0.5), ("RE", 0.5), ("RE", 0.5),
                    ("SOL", 1.0), ("SI", 1.0),

                    # "ya no pue-de ca-mi-" corridas y "nar" larga.
                    ("SOL", 0.5), ("SOL", 0.5),
                    ("FA#", 0.5), ("FA#", 0.5),
                    ("MI", 0.5), ("MI", 0.5),
                    ("RE", 2.0)
                ]
            },
            {
                "letra": (
                    "Porque no tiene, porque le falta\n"
                    "las dos patitas de atrás"
                ),
                "gesto": "SORPRESA",
                "notas": [
                    ("RE", 0.5), ("RE", 0.5), ("RE", 0.5),
                    ("FA#", 1.0), ("LA", 1.0),

                    ("RE", 0.5), ("RE", 0.5), ("RE", 0.5),
                    ("FA#", 1.0), ("LA", 1.0),

                    ("RE", 0.5), ("MI", 0.5), ("RE", 0.5),
                    ("DO", 0.5), ("SI", 0.5), ("LA", 0.5),
                    ("SOL", 2.0)
                ]
            }
        ]
    },

    "estrellita": {
        "titulo": "Estrellita, ¿dónde estás?",
        "subtitulo": "3 frases · 3 expresiones",
        "dificultad": "Media",
        # Más lenta que La cucaracha: es una canción de cuna.
        "pulso": 0.55,
        "frases": [
            {
                "letra": (
                    "Estrellita, ¿dónde estás?\n"
                    "Me pregunto quién serás"
                ),
                "gesto": "ALEGRÍA",
                "notas": [
                    ("SOL", 1.0), ("SOL", 1.0),
                    ("RE", 1.0), ("RE", 1.0),
                    ("MI", 1.0), ("MI", 1.0),
                    ("RE", 2.0),

                    ("DO", 1.0), ("DO", 1.0),
                    ("SI", 1.0), ("SI", 1.0),
                    ("LA", 1.0), ("LA", 1.0),
                    ("SOL", 2.0)
                ]
            },
            {
                "letra": (
                    "En el cielo o en el mar\n"
                    "un diamante de verdad"
                ),
                "gesto": "SORPRESA",
                "notas": [
                    ("RE", 1.0), ("RE", 1.0),
                    ("DO", 1.0), ("DO", 1.0),
                    ("SI", 1.0), ("SI", 1.0),
                    ("LA", 2.0),

                    ("RE", 1.0), ("RE", 1.0),
                    ("DO", 1.0), ("DO", 1.0),
                    ("SI", 1.0), ("SI", 1.0),
                    ("LA", 2.0)
                ]
            },
            {
                "letra": (
                    "Estrellita, ¿dónde estás?\n"
                    "Me pregunto quién serás"
                ),
                "gesto": "ALEGRÍA",
                "notas": [
                    ("SOL", 1.0), ("SOL", 1.0),
                    ("RE", 1.0), ("RE", 1.0),
                    ("MI", 1.0), ("MI", 1.0),
                    ("RE", 2.0),

                    ("DO", 1.0), ("DO", 1.0),
                    ("SI", 1.0), ("SI", 1.0),
                    ("LA", 1.0), ("LA", 1.0),
                    ("SOL", 2.0)
                ]
            }
        ]
    }
}


def partes_de_nota(nota):
    """
    Devuelve (nombre, figura).

    Acepta las dos formas: ("RE", 0.5) con su figura, o "RE" a secas
    para no romper las canciones escritas antes del ritmo.
    """
    if isinstance(nota, (tuple, list)):
        return nota[0], float(nota[1])

    return nota, 1.0


def nombres_de_notas(notas):
    """Solo los nombres, que es lo que se ve en pantalla."""
    return [partes_de_nota(n)[0] for n in notas]


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
        """Los nombres de las notas, para dibujarlas en pantalla."""
        frase = self.frase_actual()

        if frase is None:
            return []

        return nombres_de_notas(frase.get("notas", []))

    def pulso(self):
        """Cuánto dura una negra en esta canción, en segundos."""
        if self.cancion is None:
            return PULSO_MELODIA

        return float(self.cancion.get("pulso", PULSO_MELODIA))

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

        # Reproduce la frase completa, con el pulso de esta canción.
        threading.Thread(
            target=self._reproducir_frase,
            args=(notas, indice_frase, self.pulso()),
            daemon=True
        ).start()

        return True

    def _reproducir_frase(self, notas, indice_frase, pulso):
        import time

        try:
            for posicion, nota in enumerate(notas):

                # Si se pausó o se salió de la canción, el hilo muere.
                # Antes solo revisaba "pausada" y seguía sonando aunque
                # el usuario ya se hubiera ido al menú.
                if self.estado != "reproduciendo":
                    return

                nombre, figura = partes_de_nota(nota)

                # Lo que dura la nota en el compás.
                largo = figura * pulso

                # Lo que se queda pisada la tecla: un poco menos, para
                # que se oiga el corte entre una nota y la siguiente.
                self.motor.tocar(
                    nombre,
                    duracion_ms=int(largo * ARTICULACION * 1000)
                )

                # La interfaz solo se puede tocar desde el hilo principal.
                app.after(
                    0,
                    lambda p=posicion, i=indice_frase: resaltar_nota(i, p)
                )

                time.sleep(largo)

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

LECTURAS_SUAVIZADO = 8

MOVIMIENTO_MINIMO_LIBRE = 0.60

DIFERENCIA_MINIMA_GESTOS = 0.80

MARGEN_REPOSO_LIBRE = 0.70

TOTAL_MUESTRAS_MOVIMIENTO_LIBRE = 20
SEGUNDOS_CUENTA_REGRESIVA = 3


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
        self.ranuras = []
        self.historial = []
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

    registrar_evento("Inicio de sesión", "Comienza la calibración del rostro")

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

    registrar_evento(
        "Fin de sesión",
        "Sesión de trabajo completa",
        duracion_min=str(round(duracion / 60, 1)),
        expresiones=str(estadisticas_sesion["expresiones"])
    )

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
                    text=traducir("CALIBRANDO"),
                    text_color=AZUL
                )

                estado_nota.configure(
                    text=f"📐 {progreso}/{detector.total_muestras_calibracion}",
                    text_color=AZUL
                )

                mensaje_label.configure(
                    text=traducir("Mantén tu rostro relajado"),
                    text_color=AZUL
                )

                instruccion_label.configure(
                    text=traducir("INSTRUCCIÓN: No sonrías ni hagas gestos. Mira al frente y mantén el rostro relajado.")
                )

            elif detector.calibrado:

                emoji_label.configure(text=detector.emoji)

                estado_expresion.configure(
                    text=traducir(detector.emocion),
                    text_color=detector.color
                )

                estado_nota.configure(
                    text=f"🎹 {detector.nota}",
                    text_color=detector.color
                )

                mensaje_label.configure(
                    text=traducir(detector.mensaje),
                    text_color=detector.color
                )

                if detector.emocion == "REPOSO":
                    instruccion_label.configure(
                        text=traducir("INSTRUCCIÓN: Haz una expresión clara y mantenla unos instantes.")
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

                    registrar_evento(
                        "Expresión lograda",
                        f"La persona sostuvo la expresión de "
                        f"{detector.ultima_estable.lower()}",
                        expresion=detector.ultima_estable.capitalize(),
                        intensidad=intensidad_en_palabras(
                            detector.ultima_estable, medicion
                        ),
                        comparacion=comparacion
                    )

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
    text=traducir("La música al alcance de todos"),
    font=("Montserrat",22),
    text_color=GRIS
)

slogan.pack()

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
    text=traducir("Bienvenido"),
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
    text=traducir("🎹 INICIO"),
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



def elegir_idioma():
    """
    Ventana para escoger el idioma.

    Hoy solo hay español. La ventana ya recorre TEXTOS, así que el día
    que se agregue otro idioma aparece aquí solo, sin tocar esta parte.
    """
    ventana = ctk.CTkToplevel(app)
    ventana.title(traducir("⚙ Idioma"))
    ventana.geometry("460x420")
    ventana.configure(fg_color=NEGRO)
    ventana.resizable(False, False)
    ventana.transient(app)
    ventana.grab_set()

    ctk.CTkLabel(
        ventana,
        text=traducir("🌎  Idioma"),
        font=("Montserrat", 28, "bold"),
        text_color=BLANCO
    ).pack(pady=(26, 4))

    ctk.CTkLabel(
        ventana,
        text=traducir("Se guarda y la próxima vez TALAT abre en ese idioma."),
        font=("Montserrat", 14),
        text_color=GRIS,
        wraplength=380
    ).pack(pady=(0, 18))

    def poner(clave):
        global IDIOMA

        IDIOMA = clave

        configuracion["idioma"] = clave
        guardar_configuracion(configuracion)

        registrar_evento(
            "Configuración",
            f"Idioma: {TEXTOS[clave].get('_nombre', clave)}",
            "-"
        )

        aplicar_idioma()
        ventana.destroy()

    for clave, textos in TEXTOS.items():

        actual = (clave == IDIOMA)

        ctk.CTkButton(
            ventana,
            text=f"{textos.get('_bandera', '')}  {textos.get('_nombre', clave)}"
                 + ("     ✓" if actual else ""),
            width=340,
            height=58,
            corner_radius=14,
            fg_color=AZUL if actual else GRIS2,
            hover_color=MORADO,
            font=("Montserrat", 18, "bold"),
            command=lambda c=clave: poner(c)
        ).pack(pady=6)

    ctk.CTkLabel(
        ventana,
        text=traducir("Pronto habrá más idiomas."),
        font=("Montserrat", 13),
        text_color=GRIS3
    ).pack(pady=(14, 0))

    ctk.CTkButton(
        ventana,
        text="Cerrar",
        width=160,
        height=42,
        fg_color=GRIS3,
        hover_color=ROJOOS,
        font=("Montserrat", 15, "bold"),
        command=ventana.destroy
    ).pack(pady=20)

    traducir_ventana(ventana)


btn_config = ctk.CTkButton(
    right_frame,
    text=traducir("⚙ Idioma"),
    width=420,
    height=75,
    corner_radius=35,
    fg_color=NEGRO,
    border_width=2,
    border_color=MORADO,
    hover_color=GRIS2,
    text_color=BLANCO,
    font=("Montserrat",24,"bold"),
    command=elegir_idioma
)


btn_config.pack(
    pady=15
)



# ==========================================
# PIE
# ==========================================


footer = ctk.CTkLabel(
    bienvenido_frame,
    text=traducir("Sistema Musical TALAT"),
    font=("Montserrat",14),
    text_color=GRIS3
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
    text=traducir("← Bienvenido"),
    width=140,
    height=40,
    fg_color=GRIS2,
    hover_color=GRIS4,
    command=mostrar_bienvenido
)

btn_regresar.pack(side="left")

# Título
titulo_inicio = ctk.CTkLabel(
    header_inicio,
    text=traducir("Usuarios"),
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
    ventana.title(traducir(titulo))
    ventana.geometry("560x560")
    ventana.configure(fg_color=NEGRO)
    ventana.resizable(False, False)

    # transient + grab_set: la ventana queda encima y bloquea el resto.
    ventana.transient(app)
    ventana.grab_set()

    ctk.CTkLabel(
        ventana,
        text=traducir(titulo),
        font=("Montserrat", 26, "bold"),
        text_color=BLANCO
    ).pack(pady=(25, 20))

    # ---- Nombre ----
    ctk.CTkLabel(
        ventana,
        text=traducir("Nombre"),
        font=("Montserrat", 16, "bold"),
        text_color=BLANCO,
        anchor="w"
    ).pack(fill="x", padx=40)

    entrada_nombre = ctk.CTkEntry(
        ventana,
        height=42,
        font=("Montserrat", 16),
        placeholder_text=traducir("Nombre de la persona")
    )

    entrada_nombre.pack(fill="x", padx=40, pady=(4, 14))
    entrada_nombre.insert(0, nombre)

    # ---- Edad ----
    ctk.CTkLabel(
        ventana,
        text=traducir("Edad"),
        font=("Montserrat", 16, "bold"),
        text_color=BLANCO,
        anchor="w"
    ).pack(fill="x", padx=40)

    entrada_edad = ctk.CTkEntry(
        ventana,
        height=42,
        font=("Montserrat", 16),
        placeholder_text=traducir("Años cumplidos")
    )

    entrada_edad.pack(fill="x", padx=40, pady=(4, 14))
    entrada_edad.insert(0, str(edad))

    # ---- Motivo ----
    ctk.CTkLabel(
        ventana,
        text=traducir("¿Por qué usa TALAT?"),
        font=("Montserrat", 16, "bold"),
        text_color=BLANCO,
        anchor="w"
    ).pack(fill="x", padx=40)

    ctk.CTkLabel(
        ventana,
        text=traducir("Escríbelo con tus palabras. Este texto aparecerá en su perfil."),
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
        text_color=NARANJA
    )

    aviso.pack(pady=(0, 4))

    def confirmar():
        nombre_nuevo = entrada_nombre.get().strip()
        edad_nueva = entrada_edad.get().strip()
        motivo_nuevo = caja_motivo.get("1.0", "end").strip()

        if not nombre_nuevo:
            aviso.configure(text=traducir("El nombre no puede quedar vacío."))
            return

        # La edad puede quedar vacía, pero si se escribe debe ser un número real.
        if edad_nueva:
            if not edad_nueva.isdigit():
                aviso.configure(text=traducir("La edad debe ser un número."))
                return

            if not (1 <= int(edad_nueva) <= 120):
                aviso.configure(text=traducir("La edad debe estar entre 1 y 120."))
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
        fg_color=GRIS3,
        hover_color=ROJOOS,
        font=("Montserrat", 16, "bold"),
        command=ventana.destroy
    ).grid(row=0, column=1, padx=10)

    entrada_nombre.focus()

    traducir_ventana(ventana)

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

    registrar_evento(
        "Alta de la persona",
        datos["motivo"] or "Sin motivo registrado",
        nombre
    )

    mensaje_vacio.pack_forget()
    crear_tarjeta_usuario(nombre)

def abrir_GRAFICA_PARA_IMPRIMIR_en_excel():
    """
    Abre el CSV con el programa que tenga la computadora
    (Excel en Windows, Numbers en Mac, LibreOffice en Linux).
    """
    exportar_bitacora_csv()
    ruta = os.path.abspath(ARCHIVO_GRAFICA_PARA_IMPRIMIR)

    if not os.path.exists(ruta):
        return False, "Todavía no hay nada registrado."

    try:
        import subprocess
        import sys

        if sys.platform.startswith("win"):
            os.startfile(ruta)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", ruta])
        else:
            subprocess.Popen(["xdg-open", ruta])

        return True, ""

    except Exception as e:
        return False, str(e)


def mostrar_graficaparaimprimir():
    """
    Muestra la bitácora dentro de TALAT.

    Sirve para enseñarla en el momento aunque la computadora no tenga
    Excel, y trae el botón para abrirla en Excel si sí lo tiene.
    """
    ventana = ctk.CTkToplevel(app)
    ventana.title(traducir("Gráfica para imprimir de TALAT"))
    ventana.geometry("1180x640")
    ventana.configure(fg_color=NEGRO)
    ventana.transient(app)
    ventana.grab_set()

    ctk.CTkLabel(
        ventana,
        text=traducir ("📄  Gráfica para imprimir de TALAT"),
        font=("Montserrat", 28, "bold"),
        text_color=BLANCO
    ).pack(pady=(20, 2))

    ruta = os.path.abspath(ARCHIVO_GRAFICA_PARA_IMPRIMIR)

    ctk.CTkLabel(
        ventana,
        text=f"{traducir('Todo queda guardado en:')}  {ruta}",
        font=("Montserrat", 12),
        text_color=GRIS,
        wraplength=900
    ).pack(pady=(0, 12))

    aviso = ctk.CTkLabel(
        ventana,
        text="",
        font=("Montserrat", 14, "bold"),
        text_color=NARANJA
    )

    aviso.pack()

    def en_excel():
        listo, motivo = abrir_GRAFICA_PARA_IMPRIMIR_en_excel()

        if listo:
            aviso.configure(text=traducir("Abriendo..."), text_color=VERDE)
        else:
            aviso.configure(text=traducir(motivo), text_color=NARANJA)

    ctk.CTkButton(
        ventana,
        text=traducir("📊 Abrir el registro (CSV)"),
        width=240,
        height=44,
        fg_color=VERDE,
        text_color=NEGRO,
        hover_color=MORADO,
        font=("Montserrat", 16, "bold"),
        command=en_excel
    ).pack(pady=(6, 14))

    tabla = ctk.CTkScrollableFrame(ventana, fg_color=NEGRO)
    tabla.pack(fill="both", expand=True, padx=20, pady=(0, 10))

    filas = []
    exportar_bitacora_csv()

    try:
        with open(ARCHIVO_GRAFICA_PARA_IMPRIMIR, "r", encoding="utf-8-sig") as archivo:
            filas = [f for f in csv.reader(archivo) if f]

    except FileNotFoundError:
        filas = []
    except Exception as e:
        print("No se pudo leer la Gráfica:", e)

    if len(filas) <= 1:
        ctk.CTkLabel(
            tabla,
            text="Todavía no hay nada registrado.\n\n"
                 "Inicia una sesión y vuelve a entrar aquí.",
            font=("Montserrat", 18),
            text_color=GRIS,
            justify="center"
        ).pack(pady=120)

    else:
        # Se muestran las columnas más útiles de un vistazo. El resto
        # (edad, duración, observaciones) están en el archivo de Excel.
        columnas_visibles = [0, 1, 2, 4, 5, 8, 9, 10]
        anchos = [90, 60, 130, 150, 240, 110, 100, 200]

        encabezado = filas[0]

        # Las más nuevas arriba, y como mucho 300 para que no se
        # tarde en abrir cuando el archivo ya tenga meses de uso.
        cuerpo = filas[1:][-300:][::-1]

        ctk.CTkLabel(
            ventana,
            text=f"{len(filas) - 1} {traducir('registros en total')}"
                 + (f" · {traducir('se muestran los 300 más recientes')}"
                    if len(cuerpo) >= 300 else ""),
            font=("Montserrat", 13),
            text_color=GRIS
        ).pack(before=tabla, pady=(0, 6))

        fila_titulos = ctk.CTkFrame(tabla, fg_color=GRIS2, corner_radius=8)
        fila_titulos.pack(fill="x", pady=(0, 4))

        for posicion, columna in enumerate(columnas_visibles):
            titulo = encabezado[columna] if columna < len(encabezado) else ""

            ctk.CTkLabel(
                fila_titulos,
                text=titulo.upper(),
                font=("Montserrat", 11, "bold"),
                text_color=BLANCO,
                width=anchos[posicion],
                anchor="w"
            ).pack(side="left", padx=6, pady=6)

        colores_evento = {
            "Inicio de sesión": AZUL,
            "Fin de sesión": AZUL,
            "Expresión lograda": BLANCO,
            "Actividad completada": AMARILLO,
            "Gesto personalizado": VERDE,
            "Alta de la persona": MORADO,
            "Equipo": GRIS,
            "Configuración": GRIS
        }

        for numero, fila in enumerate(cuerpo):

            marco = ctk.CTkFrame(
                tabla,
                fg_color=NEGRO2 if numero % 2 == 0 else NEGRO2,
                corner_radius=6
            )

            marco.pack(fill="x", pady=1)

            for posicion, columna in enumerate(columnas_visibles):
                texto = fila[columna] if columna < len(fila) else ""

                # La columna 4 es el tipo de registro: va en color.
                # La 9 es la intensidad: verde cuando fue marcada.
                if columna == 4:
                    color = colores_evento.get(texto, BLANCO)
                elif columna == 9:
                    color = {"Marcada": VERDE,
                             "Moderada": AZUL,
                             "Leve": NARANJA}.get(texto, GRIS)
                elif columna == 2:
                    color = BLANCO
                else:
                    color = GRIS

                ctk.CTkLabel(
                    marco,
                    text=texto,
                    font=("Montserrat", 11,
                          "bold" if columna in (4, 9) else "normal"),
                    text_color=color,
                    width=anchos[posicion],
                    anchor="w"
                ).pack(side="left", padx=6, pady=4)

    ctk.CTkButton(
        ventana,
        text="Cerrar",
        width=160,
        height=42,
        fg_color=AZUL,
        hover_color=MORADO,
        font=("Montserrat", 15, "bold"),
        command=ventana.destroy
    ).pack(pady=(0, 16))

    traducir_ventana(ventana)


btn_graficaparaimprimir = ctk.CTkButton(
    header_inicio,
    text=traducir("📄 Generar reporte PDF"),
    width=150,
    height=45,
    fg_color=GRIS2,
    hover_color=MORADO,
    command=mostrar_graficaparaimprimir
)

btn_graficaparaimprimir.pack(side="right", padx=(0, 10))


# Botón agregar usuario
btn_agregar = ctk.CTkButton(
    header_inicio,
    text=traducir("➕ Agregar usuario"),
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
        # "9 años" o "9 years", según el idioma que esté puesto.
        partes.append(
            f"{edad} years" if IDIOMA == "en" else f"{edad} años"
        )

    if motivo:
        # En la tarjeta solo cabe una línea; el texto completo va en el perfil.
        corto = motivo.replace("\n", " ")

        if len(corto) > 60:
            corto = corto[:57] + "..."

        partes.append(corto)

    if not partes:
        return traducir("Sin datos personales · edítalos desde su perfil")

    return "  ·  ".join(partes)


def crear_tarjeta_usuario(nombre):


    tarjeta = ctk.CTkFrame(
        usuarios_container,
        fg_color=NEGRO2,
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
        text=traducir("📊 Ver perfil"),
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
    text=traducir("⏹ Terminar sesión"),
    width=190,
    height=45,
    fg_color=ROJOOS,
    command=terminar_sesion)

btn_regresar_sesion.pack(
    side="left"
)


titulo_sesion = ctk.CTkLabel(
    header_sesion,
    text=traducir("🎵 Sesión TALAT"),
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
            text_color=NARANJA
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
    text=traducir("🔧 Probar piano"),
    width=140,
    height=38,
    fg_color=GRIS2,
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
    fg_color=NEGRO,
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
    fg_color=NEGRO2,
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
    text=traducir("🧠 Modo Terapia"),
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
    text=traducir("🎼 Canción Guiada"),
    width=250,
    height=55,
    fg_color=GRIS2,
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
    text=traducir("🎛 Modo Libre"),
    width=250,
    height=55,
    fg_color=GRIS2,
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
    fg_color=NEGRO2,
    corner_radius=16
)

etiqueta_sonido_terapia = ctk.CTkLabel(
    barra_sonido,
    text=traducir("SONIDO"),
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
    unselected_color=GRIS2,
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
    fg_color=NEGRO2,
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
    text=traducir("🎼 ELIGE UNA CANCIÓN"),
    font=("Montserrat", 22, "bold"),
    text_color=BLANCO
).pack(pady=(18, 2))

ctk.CTkLabel(
    vista_menu,
    text=traducir("Una sola expresión toca una frase completa de la melodía."),
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
    fg_color=NEGRO,
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
            fg_color=NEGRO,
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
            text=traducir(cancion["subtitulo"]),
            font=("Montserrat", 13),
            text_color=GRIS
        ).pack(side="left")

        dificultad = cancion.get("dificultad", "")

        if dificultad:
            ctk.CTkLabel(
                linea_subtitulo,
                text=f" {traducir(dificultad)} ",
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
                text=f"{guia.get('emoji', '')} {traducir(gesto)}",
                font=("Montserrat", 13, "bold"),
                text_color=BLANCO,
                fg_color=GRIS2,
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

        segundos = sum(
            partes_de_nota(nota)[1]
            for frase in cancion["frases"]
            for nota in frase.get("notas", [])
        ) * cancion.get("pulso", PULSO_MELODIA)

        ctk.CTkLabel(
            tarjeta,
            text=f"{total_frases} {traducir('frases')} · "
                 f"{total_notas} {traducir('notas')} · "
                 f"{int(segundos)} {traducir('segundos de música')}",
            font=("Montserrat", 12),
            text_color=GRIS
        ).pack(
            anchor="w",
            padx=15,
            pady=(6, 0)
        )

        ctk.CTkButton(
            tarjeta,
            text=traducir("▶ TOCAR ESTA CANCIÓN"),
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
    texto = traducir(texto)

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

        poner_feedback(f"Ahora toca: {titulo}", NARANJA)


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

    notas = nombres_de_notas(frase.get("notas", []))

    activo = cancion_guiada.estado in ("tocando", "pausada", "reproduciendo")

    for posicion, nota in enumerate(notas):

        chip = ctk.CTkLabel(
            fila_notas,
            text=nota,
            width=52,
            height=30,
            corner_radius=10,
            fg_color=MORADO if activo else GRIS2,
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

    if IDIOMA == "en":
        texto_cabecera = f"PHRASE {numero} OF {total_frases}   ·   {len(notas)} notes"
    else:
        texto_cabecera = f"FRASE {numero} DE {total_frases}   ·   {len(notas)} notas"

    cancion_cabecera.configure(
        text=texto_cabecera,
        text_color=color_gesto
    )

    cancion_emoji.configure(text=guia.get("emoji", "🎵"))
    cancion_gesto_titulo.configure(
        text=traducir(guia.get("titulo", "")),
        text_color=color_gesto
    )

    cancion_gesto_como.configure(text=traducir(guia.get("como", "")))

    if cancion_guiada.estado == "detenida":
        poner_feedback("Pulsa ▶ INICIAR para empezar", GRIS)

    elif cancion_guiada.estado == "pausada":
        poner_feedback("⏸ En pausa", NARANJA)

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

    registrar_evento(
        "Actividad completada",
        f"Canción: {cancion_guiada.cancion.get('titulo', cancion_guiada.clave)}",
        expresiones=str(cancion_guiada.aciertos)
    )

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
    fg_color=NEGRO2,
    corner_radius=20,
    width=470
)

panel_libre.pack_propagate(False)

ctk.CTkLabel(
    panel_libre,
    text=traducir("🎛 MODO LIBRE"),
    font=("Montserrat", 22, "bold"),
    text_color=BLANCO
).pack(pady=(14, 0))

# Igual que el menú de canciones: una línea que explica de qué va.
ctk.CTkLabel(
    panel_libre,
    text=traducir("Tú eliges el sonido y tú grabas la cara que lo toca."),
    font=("Montserrat", 13),
    text_color=GRIS,
    wraplength=410
).pack(pady=(0, 2))

libre_contador = ctk.CTkLabel(
    panel_libre,
    text=traducir("0 de 8 sonidos"),
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
    fg_color=NEGRO,
    corner_radius=12,
    height=46
)

barra_estado_libre.pack(fill="x", padx=14, pady=(8, 6))
barra_estado_libre.pack_propagate(False)

libre_feedback = ctk.CTkLabel(
    barra_estado_libre,
    text=traducir("Agrega un sonido y graba su gesto."),
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
    texto = traducir(texto)

    if libre_feedback.cget("text") == texto:
        return

    libre_feedback.configure(text=texto, text_color=color)


def marcar_fila_libre(indice, modo="normal"):
    """
    Resalta una fila: 'sonando' mientras el gesto está activo,
    'grabando' mientras se captura, 'normal' el resto del tiempo.
    """
    colores = {
        "normal": (NEGRO2, 0, NEGRO2),
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
                fila.configure(fg_color=NEGRO, border_width=0)

    except Exception:
        # La lista se redibujó mientras tanto.
        pass


def dibujar_lista_libre():
    """Repinta la lista completa de sonidos y gestos."""
    for hijo in lista_libre.winfo_children():
        hijo.destroy()

    filas_libre.clear()

    total = len(instrumento_libre.ranuras)

    libre_contador.configure(
        text=f"{total} de 8 sonidos" if IDIOMA == "es" else f"{total} of 8 sounds"
    )

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
            fg_color=NEGRO,
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
            fg_color=GRIS2,
            text_color=GRIS,
            font=("Montserrat", 12, "bold")
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            arriba,
            text=SONIDO_A_ETIQUETA.get(ranura["sonido"], "🎵 DO"),
            width=150,
            height=34,
            corner_radius=8,
            fg_color=GRIS2,
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
            fg_color=GRIS2,
            hover_color=ROJOOS,
            font=("Montserrat", 14),
            command=lambda i=indice: quitar_sonido_libre(i)
        ).pack(side="right", padx=(6, 0))

        ctk.CTkButton(
            arriba,
            text=traducir(
                "🎬 Grabar" if ranura["gesto"] is None else "🔄 Regrabar"
            ),
            width=104,
            height=34,
            fg_color=AZUL if ranura["gesto"] is None else GRIS2,
            hover_color=MORADO,
            font=("Montserrat", 13, "bold"),
            command=lambda i=indice: iniciar_grabacion_gesto(i)
        ).pack(side="right")

        if ranura["gesto"] is None:
            texto = "●  Sin gesto grabado"
            color = NARANJA
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
        text=traducir("Toca una tecla para escucharla y asignarla."),
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
            text_color=BLANCO if clave == actual else GRIS4,
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
            fg_color=MORADO if clave == actual else NEGRO2,
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
        fg_color=GRIS3,
        hover_color=ROJOOS,
        font=("Montserrat", 15, "bold"),
        command=ventana.destroy
    ).pack(pady=22)

    traducir_ventana(ventana)


def agregar_sonido_libre():
    if len(instrumento_libre.ranuras) >= 8:
        poner_estado_libre(
            "Ocho sonidos es el máximo.",
            NARANJA
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
            NARANJA
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
        poner_estado_libre(f"✖ {motivo}", NARANJA)
        return

    instrumento_libre.guardar_gesto(indice, promedio)

    dibujar_lista_libre()
    guardar_instrumento_libre()

    registrar_evento(
        "Gesto personalizado",
        f"Movimiento propio ({instrumento_libre.ranuras[indice]['texto']}) "
        f"asignado a la nota {instrumento_libre.ranuras[indice]['sonido']}"
    )

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
    btn_cancion.configure(fg_color=GRIS2)
    btn_libre.configure(fg_color=GRIS2)

    mostrar_sonido_terapia(True)

    modo_actual.configure(text=traducir("🧠 Modo Terapia"))

    instruccion_label.configure(
        text=traducir("INSTRUCCIÓN: Haz una expresión clara y mantenla unos instantes.")
    )


def activar_modo_cancion():
    global modo_sesion

    modo_sesion = "cancion"

    info_sesion.pack_forget()
    panel_libre.pack_forget()
    panel_cancion.pack(side="right", fill="both", padx=20)

    btn_terapia.configure(fg_color=GRIS2)
    btn_cancion.configure(fg_color=AZUL)
    btn_libre.configure(fg_color=GRIS2)

    mostrar_sonido_terapia(False)

    modo_actual.configure(text=traducir("🎼 Canción Guiada"))

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

    btn_terapia.configure(fg_color=GRIS2)
    btn_cancion.configure(fg_color=GRIS2)
    btn_libre.configure(fg_color=AZUL)

    mostrar_sonido_terapia(False)

    modo_actual.configure(text=traducir("🎛 Modo Libre"))

    # Recuperamos el instrumento que esta persona ya había armado.
    if usuario_actual is not None:
        guardado = usuarios_db.get(usuario_actual, {}).get("instrumento_libre")

        if guardado:
            instrumento_libre.importar(guardado)

    dibujar_lista_libre()

    if not detector.calibrado:
        poner_estado_libre(
            "Espera a que termine la calibración.",
            NARANJA
        )


def aplicar_idioma():
    """Aplica el idioma a widgets actuales y a los paneles dinámicos."""
    traducir_ventana(app)
    try: recargar_tarjetas_usuarios()
    except Exception: pass
    try: dibujar_lista_libre()
    except Exception: pass
    try: construir_menu_canciones()
    except Exception: pass
    try: actualizar_panel_cancion()
    except Exception: pass
    try:
        actualizar_estadisticas_perfil()
        actualizar_comentarios_perfil()
    except Exception: pass


btn_terapia.configure(command=activar_modo_terapia)
btn_cancion.configure(command=activar_modo_cancion)
btn_libre.configure(command=activar_modo_libre)

# Botón para volver al menú desde la pantalla de tocar.
ctk.CTkButton(
    barra_superior,
    text=traducir("← Canciones"),
    width=110,
    height=30,
    fg_color=GRIS2,
    hover_color=MORADO,
    font=("Montserrat", 12, "bold"),
    command=mostrar_menu_canciones
).pack(side="right")

ctk.CTkButton(
    controles_cancion,
    text=traducir("▶ INICIAR"),
    width=125,
    height=40,
    fg_color=AZUL,
    hover_color=MORADO,
    font=("Montserrat", 15, "bold"),
    command=iniciar_cancion
).grid(row=0, column=0, padx=5)

ctk.CTkButton(
    controles_cancion,
    text=traducir("⏸ PAUSA"),
    width=125,
    height=40,
    fg_color=GRIS2,
    hover_color=MORADO,
    font=("Montserrat", 15, "bold"),
    command=pausar_cancion
).grid(row=0, column=1, padx=5)

ctk.CTkButton(
    controles_cancion,
    text=traducir("↻ REINICIAR"),
    width=125,
    height=40,
    fg_color=GRIS2,
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
            fill=GRIS2
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
        fill=GRIS3
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

    # El selector puede venir traducido ("Expressions"). Guardamos
    # siempre el nombre en español, que es con el que compara la
    # gráfica más abajo.
    metrica_grafica = TRADUCCION_ES.get(valor.strip(), valor)

    dibujar_grafica_progreso()


def actualizar_estadisticas_perfil():

    if usuario_actual is None:
        return

    datos = usuarios_db.get(usuario_actual)

    if datos is None:
        return

    edad = str(datos.get("edad", "")).strip()

    if edad:
        edad_label.configure(
            text=f"Age: {edad} years" if IDIOMA == "en"
                 else f"Edad: {edad} años"
        )
    else:
        edad_label.configure(text=traducir("Edad: sin registrar"))

    motivo = str(datos.get("motivo", "")).strip()

    motivo_label.configure(
        text=motivo if motivo else "Sin registrar",
        text_color=BLANCO if motivo else GRIS
    )

    minutos = datos.get("tiempo_total", 0) // 60

    # Estas cinco líneas se arman con el número pegado al texto, así
    # que el barrido de traducción no las alcanza: hay que armarlas
    # ya traducidas. La etiqueta se traduce y el número se le pega.
    tiempo_total_label.configure(
        text=f"{traducir('Tiempo total:')} {minutos} min"
    )

    sesiones_label.configure(
        text=f"{traducir('Sesiones realizadas:')} {datos.get('sesiones', 0)}"
    )

    expresiones_label.configure(
        text=f"{traducir('Expresiones registradas:')} "
             f"{datos.get('expresiones', 0)}"
    )

    notas_label.configure(
        text=f"{traducir('Notas detectadas:')} {datos.get('notas', 0)}"
    )

    ultima = datos.get("ultima_sesion", "")

    ultima_label.configure(
        text=(
            f"{traducir('Última sesión:')} {ultima}"
            if ultima
            else f"{traducir('Última sesión:')} {traducir('Sin sesiones')}"
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


# ==========================================
# REPORTE PDF PARA LA TERAPEUTA
# ==========================================

# Este reporte NO crea otra base de datos.
# Lee directamente el historial que TALAT ya guarda en usuarios.json.

MOVIMIENTOS_REPORTE = {
    "ALEGRÍA": ("Sonrisa", "Smile"),
    "SORPRESA": ("Apertura de boca y elevación de cejas",
                 "Mouth opening and eyebrow elevation"),
    "IRA": ("Fruncimiento del ceño", "Frowning"),
    "TRISTEZA": ("Movimiento de las comisuras", "Mouth-corner movement"),
    "ABURRIMIENTO": ("Entrecierre de ojos", "Eye narrowing")
}


def _texto_reporte(espanol, ingles):
    return ingles if IDIOMA == "en" else espanol


def _nombre_movimiento_reporte(emocion):
    nombres = MOVIMIENTOS_REPORTE.get(emocion)
    return _texto_reporte(*nombres) if nombres else traducir(emocion).capitalize()


def _abrir_archivo_generado(ruta):
    try:
        import subprocess
        import sys

        if sys.platform.startswith("win"):
            os.startfile(ruta)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", ruta])
        else:
            subprocess.Popen(["xdg-open", ruta])

        return True, ""
    except Exception as e:
        return False, str(e)


def _sesiones_validas(datos):
    return [
        sesion for sesion in datos.get("historial_sesiones", [])
        if isinstance(sesion, dict)
    ]


def _expresiones_de_sesion(sesion):
    expresiones = sesion.get("expresiones", [])
    return expresiones if isinstance(expresiones, list) else []


def _duracion_minutos(sesion):
    try:
        return round(float(sesion.get("duracion_segundos", 0) or 0) / 60.0, 1)
    except (TypeError, ValueError):
        return 0.0


def _fecha_corta(sesion):
    fecha = str(sesion.get("fecha", "") or "")
    if len(fecha) >= 10:
        # Guardada como YYYY-MM-DD HH:MM.
        yyyy, mm, dd = fecha[:10].split("-")
        return f"{dd}/{mm}" if IDIOMA == "es" else f"{mm}/{dd}"
    return "—"


def _conteo_movimientos(datos):
    conteo = {emocion: 0 for emocion in MOVIMIENTOS_REPORTE}

    for sesion in _sesiones_validas(datos):
        for registro in _expresiones_de_sesion(sesion):
            if not isinstance(registro, dict):
                continue
            emocion = registro.get("emocion")
            if emocion in conteo:
                conteo[emocion] += 1

    return conteo


def _calidad_registro(registro, escalas):
    """
    Convierte las mediciones útiles de una expresión a 0-100.

    Usa las mismas reglas que la pantalla de estadísticas:
    LO_QUE_IMPORTA decide qué mediciones mirar y LOGRO_POR_MEDICION
    indica si una medición aumenta o disminuye cuando el gesto se marca.
    """
    emocion = registro.get("emocion")
    claves = LO_QUE_IMPORTA.get(emocion, [])
    valores = []

    for clave in claves:
        if clave not in registro:
            continue

        valor = porcentaje_medicion(clave, registro[clave], escalas)

        if LOGRO_POR_MEDICION.get(clave, {}).get("invertida"):
            valor = 1.0 - valor

        valores.append(max(0.0, min(1.0, valor)))

    if not valores:
        return None

    return sum(valores) / len(valores) * 100.0


def _calidad_por_movimiento(datos):
    historial = _sesiones_validas(datos)
    escalas = escalas_del_usuario(historial)
    acumulado = {emocion: [] for emocion in MOVIMIENTOS_REPORTE}

    for sesion in historial:
        for registro in _expresiones_de_sesion(sesion):
            if not isinstance(registro, dict):
                continue

            emocion = registro.get("emocion")
            if emocion not in acumulado:
                continue

            calidad = _calidad_registro(registro, escalas)
            if calidad is not None:
                acumulado[emocion].append(calidad)

    return {
        emocion: round(sum(valores) / len(valores), 1) if valores else None
        for emocion, valores in acumulado.items()
    }


def _calidad_bloque_sesiones(sesiones, escalas):
    valores = []

    for sesion in sesiones:
        for registro in _expresiones_de_sesion(sesion):
            if not isinstance(registro, dict):
                continue
            calidad = _calidad_registro(registro, escalas)
            if calidad is not None:
                valores.append(calidad)

    if not valores:
        return None

    return sum(valores) / len(valores)


def _indicadores_terapeuta(datos):
    sesiones = _sesiones_validas(datos)
    conteo = _conteo_movimientos(datos)

    total_expresiones = sum(
        int(s.get("total_expresiones", len(_expresiones_de_sesion(s))) or 0)
        for s in sesiones
    )
    total_minutos = sum(_duracion_minutos(s) for s in sesiones)

    promedio_exp = total_expresiones / len(sesiones) if sesiones else 0
    promedio_min = total_minutos / len(sesiones) if sesiones else 0

    practicados = [(e, n) for e, n in conteo.items() if n > 0]
    if practicados:
        emocion_favorita, cantidad_favorita = max(practicados, key=lambda x: x[1])
        movimiento_favorito = (
            f"{_nombre_movimiento_reporte(emocion_favorita)} ({cantidad_favorita})"
        )
    else:
        movimiento_favorito = "—"

    cambio = None
    if len(sesiones) >= 2:
        escalas = escalas_del_usuario(sesiones)
        cantidad_bloque = min(3, max(1, len(sesiones) // 2))
        primeras = sesiones[:cantidad_bloque]
        recientes = sesiones[-cantidad_bloque:]

        calidad_inicio = _calidad_bloque_sesiones(primeras, escalas)
        calidad_actual = _calidad_bloque_sesiones(recientes, escalas)

        if calidad_inicio is not None and calidad_actual is not None:
            cambio = round(calidad_actual - calidad_inicio, 1)

    return {
        "promedio_exp": round(promedio_exp, 1),
        "promedio_min": round(promedio_min, 1),
        "movimiento_favorito": movimiento_favorito,
        "cambio": cambio
    }


def _grafica_barras_sesiones(datos):
    """Gráfica: expresiones logradas en las últimas 10 sesiones."""
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.textlabels import Label
    from reportlab.lib import colors

    sesiones = _sesiones_validas(datos)[-10:]
    valores = [
        int(s.get("total_expresiones", len(_expresiones_de_sesion(s))) or 0)
        for s in sesiones
    ]
    etiquetas = [_fecha_corta(s) for s in sesiones]

    dibujo = Drawing(500, 210)

    if not sesiones:
        return dibujo

    grafica = VerticalBarChart()
    grafica.x = 42
    grafica.y = 35
    grafica.height = 135
    grafica.width = 425
    grafica.data = [valores]
    grafica.categoryAxis.categoryNames = etiquetas
    grafica.categoryAxis.labels.fontName = "Helvetica"
    grafica.categoryAxis.labels.fontSize = 8
    grafica.valueAxis.valueMin = 0
    grafica.valueAxis.valueMax = max(5, max(valores) * 1.2)
    grafica.valueAxis.valueStep = max(1, int(grafica.valueAxis.valueMax / 5))
    grafica.valueAxis.labels.fontName = "Helvetica"
    grafica.valueAxis.labels.fontSize = 8
    grafica.bars[0].fillColor = colors.HexColor("#3B9DFF")
    grafica.barWidth = 16
    dibujo.add(grafica)

    etiqueta = Label()
    etiqueta.setOrigin(250, 190)
    etiqueta.setText(
        _texto_reporte(
            "Expresiones logradas por sesión",
            "Expressions completed per session"
        )
    )
    etiqueta.fontName = "Helvetica-Bold"
    etiqueta.fontSize = 11
    etiqueta.textAnchor = "middle"
    dibujo.add(etiqueta)

    return dibujo


def _grafica_calidad_movimientos(datos):
    """
    Gráfica: qué tan marcado fue cada movimiento, de 0 a 100.

    No muestra números crudos de MediaPipe. Resume las mediciones
    faciales que ya usa TALAT para cada expresión.
    """
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import HorizontalBarChart
    from reportlab.graphics.charts.textlabels import Label
    from reportlab.lib import colors

    calidad = _calidad_por_movimiento(datos)

    emociones = [
        e for e in MOVIMIENTOS_REPORTE
        if calidad.get(e) is not None
    ]

    dibujo = Drawing(500, 225)

    if not emociones:
        return dibujo

    valores = [calidad[e] for e in emociones]
    nombres = [_nombre_movimiento_reporte(e) for e in emociones]

    grafica = HorizontalBarChart()
    grafica.x = 165
    grafica.y = 32
    grafica.height = 145
    grafica.width = 300
    grafica.data = [valores]
    grafica.categoryAxis.categoryNames = nombres
    grafica.categoryAxis.labels.fontName = "Helvetica"
    grafica.categoryAxis.labels.fontSize = 7.5
    grafica.valueAxis.valueMin = 0
    grafica.valueAxis.valueMax = 100
    grafica.valueAxis.valueStep = 20
    grafica.valueAxis.labels.fontName = "Helvetica"
    grafica.valueAxis.labels.fontSize = 8
    grafica.bars[0].fillColor = colors.HexColor("#9B4DFF")
    dibujo.add(grafica)

    etiqueta = Label()
    etiqueta.setOrigin(250, 205)
    etiqueta.setText(
        _texto_reporte(
            "Qué tan marcado fue cada movimiento (0–100)",
            "How pronounced each movement was (0–100)"
        )
    )
    etiqueta.fontName = "Helvetica-Bold"
    etiqueta.fontSize = 11
    etiqueta.textAnchor = "middle"
    dibujo.add(etiqueta)

    return dibujo


def construir_reporte_pdf(nombre):
    datos = usuarios_db.get(nombre)

    if datos is None:
        return False, traducir("No se encontró el usuario.")

    sesiones = _sesiones_validas(datos)
    if not sesiones:
        return False, traducir("Todavía no hay sesiones de esta persona.")

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            KeepTogether
        )
    except ImportError:
        return False, "Falta reportlab (pip install reportlab)"

    try:
        limpio = "".join(
            c for c in nombre if c.isalnum() or c in " -_"
        ).strip().replace(" ", "_")

        archivo = (
            f"reporte_talat_{limpio}_"
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}.pdf"
        )
        ruta = os.path.abspath(archivo)

        documento = SimpleDocTemplate(
            ruta,
            pagesize=letter,
            rightMargin=1.35 * cm,
            leftMargin=1.35 * cm,
            topMargin=1.25 * cm,
            bottomMargin=1.25 * cm
        )

        estilos = getSampleStyleSheet()

        estilo_titulo = ParagraphStyle(
            "TalatTitulo",
            parent=estilos["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=23,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#3B9DFF"),
            spaceAfter=5
        )

        estilo_subtitulo = ParagraphStyle(
            "TalatSubtitulo",
            parent=estilos["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#666666"),
            spaceAfter=10
        )

        estilo_seccion = ParagraphStyle(
            "TalatSeccion",
            parent=estilos["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=15,
            textColor=colors.HexColor("#5D2AA6"),
            spaceBefore=8,
            spaceAfter=5
        )

        estilo_normal = ParagraphStyle(
            "TalatNormal",
            parent=estilos["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13
        )

        elementos = []

        # ---------------- ENCABEZADO ----------------
        elementos.append(Paragraph(
            _texto_reporte(
                "REPORTE DE SEGUIMIENTO TALAT",
                "TALAT FOLLOW-UP REPORT"
            ),
            estilo_titulo
        ))

        edad = str(datos.get("edad", "")).strip()
        fecha_impresion = datetime.now().strftime(
            "%d/%m/%Y" if IDIOMA == "es" else "%m/%d/%Y"
        )

        linea = nombre
        if edad:
            linea += _texto_reporte(
                f" · {edad} años",
                f" · {edad} years old"
            )
        linea += _texto_reporte(
            f"<br/>Generado el {fecha_impresion}",
            f"<br/>Generated on {fecha_impresion}"
        )
        elementos.append(Paragraph(linea, estilo_subtitulo))

        motivo = str(datos.get("motivo", "")).strip()
        if motivo:
            elementos.append(Paragraph(
                _texto_reporte(
                    f"<b>Motivo registrado:</b> {motivo}",
                    f"<b>Recorded reason:</b> {motivo}"
                ),
                estilo_normal
            ))
            elementos.append(Spacer(1, 5))

        # ---------------- INDICADORES ----------------
        indicadores = _indicadores_terapeuta(datos)

        elementos.append(Paragraph(
            _texto_reporte(
                "INDICADORES DE ACTIVIDAD",
                "ACTIVITY INDICATORS"
            ),
            estilo_seccion
        ))

        cambio = indicadores["cambio"]
        if cambio is None:
            cambio_texto = _texto_reporte(
                "Aún no hay suficientes mediciones para comparar",
                "Not enough measurements to compare yet"
            )
        elif cambio > 2:
            cambio_texto = _texto_reporte(
                f"+{cambio:.1f} puntos en movimientos recientes",
                f"+{cambio:.1f} points in recent movements"
            )
        elif cambio < -2:
            cambio_texto = _texto_reporte(
                f"{cambio:.1f} puntos respecto al inicio",
                f"{cambio:.1f} points compared with the beginning"
            )
        else:
            cambio_texto = _texto_reporte(
                "Se mantiene similar al inicio",
                "Similar to the beginning"
            )

        indicadores_tabla = [
            [
                _texto_reporte("Sesiones", "Sessions"),
                _texto_reporte("Tiempo total", "Total time"),
                _texto_reporte("Promedio por sesión", "Average per session"),
                _texto_reporte("Duración promedio", "Average duration")
            ],
            [
                str(len(sesiones)),
                f"{sum(_duracion_minutos(s) for s in sesiones):.1f} min",
                f"{indicadores['promedio_exp']:.1f} "
                + _texto_reporte("expresiones", "expressions"),
                f"{indicadores['promedio_min']:.1f} min"
            ]
        ]

        tabla = Table(
            indicadores_tabla,
            colWidths=[4.1 * cm, 4.1 * cm, 4.1 * cm, 4.1 * cm]
        )
        tabla.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A1A1A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F4F4F4")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.3),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D0D0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elementos.append(tabla)
        elementos.append(Spacer(1, 6))

        detalles = Table([
            [
                Paragraph(
                    "<b>" + _texto_reporte(
                        "Movimiento más practicado:",
                        "Most practiced movement:"
                    ) + "</b><br/>" + indicadores["movimiento_favorito"],
                    estilo_normal
                ),
                Paragraph(
                    "<b>" + _texto_reporte(
                        "Cambio observado:",
                        "Observed change:"
                    ) + "</b><br/>" + cambio_texto,
                    estilo_normal
                )
            ]
        ], colWidths=[8.2 * cm, 8.2 * cm])
        detalles.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F7F7")),
            ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D0D0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D0D0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        elementos.append(detalles)

        # ---------------- GRÁFICA 1 ----------------
        elementos.append(Paragraph(
            _texto_reporte(
                "PROGRESO POR SESIÓN",
                "PROGRESS BY SESSION"
            ),
            estilo_seccion
        ))
        elementos.append(_grafica_barras_sesiones(datos))

        elementos.append(Paragraph(
            _texto_reporte(
                "La gráfica muestra cuántas expresiones fueron reconocidas en cada una "
                "de las últimas sesiones. Sirve para observar participación y cantidad "
                "de práctica, no para calificar a la persona.",
                "The chart shows how many expressions were recognized in each recent "
                "session. It reflects participation and amount of practice, not a grade."
            ),
            estilo_normal
        ))

        # ---------------- GRÁFICA 2 ----------------
        elementos.append(Paragraph(
            _texto_reporte(
                "CALIDAD DE LOS MOVIMIENTOS FACIALES",
                "FACIAL MOVEMENT QUALITY"
            ),
            estilo_seccion
        ))
        elementos.append(_grafica_calidad_movimientos(datos))

        elementos.append(Paragraph(
            _texto_reporte(
                "El valor 0–100 resume qué tan marcado fue cada movimiento a partir "
                "de las mediciones que TALAT ya utiliza. No son porcentajes médicos "
                "ni un diagnóstico.",
                "The 0–100 value summarizes how pronounced each movement was using "
                "the measurements TALAT already analyzes. These are not medical "
                "percentages or a diagnosis."
            ),
            estilo_normal
        ))

        # ---------------- ÚLTIMAS SESIONES ----------------
        elementos.append(Paragraph(
            _texto_reporte(
                "ÚLTIMAS SESIONES",
                "RECENT SESSIONS"
            ),
            estilo_seccion
        ))

        historial_tabla = [[
            _texto_reporte("Fecha", "Date"),
            _texto_reporte("Duración", "Duration"),
            _texto_reporte("Expresiones", "Expressions"),
            _texto_reporte("Notas", "Notes")
        ]]

        for sesion in sesiones[-8:]:
            historial_tabla.append([
                _fecha_corta(sesion),
                f"{_duracion_minutos(sesion):.1f} min",
                str(sesion.get(
                    "total_expresiones",
                    len(_expresiones_de_sesion(sesion))
                )),
                str(sesion.get("total_notas", 0))
            ])

        historial = Table(
            historial_tabla,
            colWidths=[4.1 * cm, 4.1 * cm, 4.1 * cm, 4.1 * cm],
            repeatRows=1
        )
        historial.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3B9DFF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D0D0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F4F4F4")]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elementos.append(historial)

        # ---------------- NOTAS ----------------
        elementos.append(Paragraph(
            _texto_reporte(
                "OBSERVACIONES DE LA TERAPEUTA",
                "THERAPIST'S NOTES"
            ),
            estilo_seccion
        ))

        notas = Table(
            [[""] for _ in range(5)],
            colWidths=[16.4 * cm],
            rowHeights=[0.62 * cm] * 5
        )
        notas.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFBFBF"))
        ]))
        elementos.append(notas)

        def pie(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(colors.HexColor("#777777"))
            canvas.drawCentredString(
                letter[0] / 2,
                0.65 * cm,
                _texto_reporte(
                    "TALAT · Reporte de seguimiento de actividad",
                    "TALAT · Activity follow-up report"
                )
            )
            canvas.restoreState()

        documento.build(
            elementos,
            onFirstPage=pie,
            onLaterPages=pie
        )

        registrar_evento(
            "Reporte generado",
            f"Reporte PDF: {archivo}",
            usuario=nombre
        )

        return True, ruta

    except Exception as e:
        print("No se pudo generar el PDF:", e)
        return False, f"{traducir('No se pudo generar el archivo')} ({e})"


def exportar_graficas_para_imprimir(nombre=None):
    """
    El botón existente sigue usando el mismo nombre de función.
    No se crea ninguna base nueva: se genera un PDF desde usuarios.json.
    """
    persona = nombre or usuario_actual
    titulo = traducir("Reporte PDF de TALAT")

    if persona is None:
        messagebox.showinfo(
            titulo,
            traducir("Primero abre el perfil de una persona.")
        )
        return

    listo, resultado = construir_reporte_pdf(persona)

    if not listo:
        messagebox.showwarning(titulo, resultado)
        return

    abierto, motivo = _abrir_archivo_generado(resultado)

    if abierto:
        messagebox.showinfo(
            titulo,
            f"{traducir('Reporte generado correctamente.')}\n\n{resultado}"
        )
    else:
        messagebox.showinfo(
            titulo,
            f"{traducir('El archivo se guardó aquí:')}\n\n{resultado}\n\n"
            f"({traducir('No se pudo abrir solo')}: {motivo})"
        )


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
        color = NARANJA

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

    riel = ctk.CTkFrame(fila, fg_color=GRIS2, height=14, corner_radius=7)
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

            bloque = ctk.CTkFrame(scroll, fg_color=NEGRO2, corner_radius=18)
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

                tarjeta = ctk.CTkFrame(bloque, fg_color=NEGRO, corner_radius=14)
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
                            color = NARANJA

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

            ctk.CTkFrame(bloque, height=8, fg_color=NEGRO2).pack()

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

    traducir_ventana(ventana)


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
    fg_color=NEGRO2,
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
    fg_color=NEGRO2,
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
    unselected_color=GRIS2,
    command=cambiar_metrica_grafica
)

selector_metrica.set("Expresiones")

selector_metrica.pack(
    pady=(0,15)
)

grafica_canvas = ctk.CTkCanvas(
    estadisticas,
    height=240,
    bg=NEGRO2,
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
    fg_color=NEGRO2,
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
    fg_color=NEGRO2,
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

def eliminar_usuario():
    """Elimina el usuario actualmente abierto, con confirmación."""
    global usuario_actual

    if usuario_actual is None:
        return

    nombre = usuario_actual

    if nombre not in usuarios_db:
        messagebox.showerror(
            traducir("Eliminar usuario"),
            traducir("No se encontró el usuario.")
        )
        return

    if not messagebox.askyesno(
        traducir("Eliminar usuario"),
        f"{traducir('¿Seguro que quieres eliminar este usuario?')}\n\n👤 {nombre}"
    ):
        return

    try:
        usuarios_db.pop(nombre)
        guardar_usuarios(usuarios_db)
        registrar_evento(
            "Alta de la persona",
            f"Usuario eliminado: {nombre}",
            usuario=nombre
        )
        usuario_actual = None
        detener_camara()
        perfil_frame.pack_forget()
        sesion_frame.pack_forget()
        inicio_frame.pack(fill="both", expand=True)
        recargar_tarjetas_usuarios()
        messagebox.showinfo(
            traducir("Eliminar usuario"),
            traducir("El usuario fue eliminado.")
        )
    except Exception as e:
        print("Error eliminando usuario:", e)
        messagebox.showerror(
            traducir("Eliminar usuario"),
            traducir("El usuario no pudo eliminarse.")
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
    text=traducir("▶ Iniciar sesión"),
    width=220,
    fg_color=AZUL,
    hover_color=MORADO,
    command=iniciar_sesion
).grid(row=0, column=0, padx=10)

ctk.CTkButton(
    botones,
    text=traducir("✏ Editar usuario"),
    width=220,
    command=editar_usuario
).grid(row=0, column=1, padx=10)

ctk.CTkButton(
    botones,
    text=traducir("🗑 Eliminar usuario"),
    width=220,
    fg_color=ROJOOS,
    hover_color="#8B0000",
    command=eliminar_usuario
).grid(row=0, column=2, padx=10)

ctk.CTkButton(
    botones,
    text=traducir("📄 Generar reporte PDF"),
    width=220,
    fg_color=VERDE,
    text_color=NEGRO,
    hover_color=MORADO,
    command=exportar_graficas_para_imprimir
).grid(row=0, column=3, padx=10)


# ==========================================
# EJECUTAR
# ==========================================

app.mainloop()
