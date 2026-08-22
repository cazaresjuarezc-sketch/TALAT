import customtkinter as ctk
from PIL import Image
import cv2
import mediapipe as mp
import threading
from PIL import ImageTk

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
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camara = None
camara_activa = False
frame_camara = None

# ==========================================
# DETECTOR DE GESTOS T'ALAT
# ==========================================

class DetectorGestos:

    def __init__(self):

        self.emocion = "REPOSO"
        self.emoji = "😐"
        self.nota = "--"
        self.color = BLANCO
        self.mensaje = "Haz una expresión"

    def detectar(self, rostro, ancho, alto):

        # Boca
        boca_izq = rostro.landmark[61]
        boca_der = rostro.landmark[291]
        boca_sup = rostro.landmark[13]
        boca_inf = rostro.landmark[14]

        # Ojos
        ojo_izq_sup = rostro.landmark[159]
        ojo_izq_inf = rostro.landmark[145]

        ojo_der_sup = rostro.landmark[386]
        ojo_der_inf = rostro.landmark[374]

        # Cejas
        ceja_izq = rostro.landmark[52]
        ceja_izq_inicio = rostro.landmark[55]
        ceja_der_inicio = rostro.landmark[285]

        # =============================
        # MEDIDAS
        # =============================

        ancho_boca = abs(
            int(boca_der.x * ancho) -
            int(boca_izq.x * ancho)
        )

        alto_boca = abs(
            int(boca_inf.y * alto) -
            int(boca_sup.y * alto)
        )

        apertura_ojo_izq = abs(
            int(ojo_izq_inf.y * alto) -
            int(ojo_izq_sup.y * alto)
        )

        apertura_ojo_der = abs(
            int(ojo_der_inf.y * alto) -
            int(ojo_der_sup.y * alto)
        )

        altura_ceja = abs(
            int(ojo_izq_sup.y * alto) -
            int(ceja_izq.y * alto)
        )

        cercania_cejas = abs(
            int(ceja_der_inicio.x * ancho) -
            int(ceja_izq_inicio.x * ancho)
        )

        # =============================
        # REPOSO
        # =============================

        self.emocion = "REPOSO"
        self.emoji = "😐"
        self.nota = "--"
        self.color = BLANCO
        self.mensaje = "Haz una expresión"

        # =============================
        # ABURRIMIENTO
        # =============================

        if apertura_ojo_izq < 5 and apertura_ojo_der < 5:

            self.emocion = "ABURRIMIENTO"
            self.emoji = "😑"
            self.nota = "DO"
            self.color = "#BFBFBF"
            self.mensaje = "Mantén la expresión"

        # =============================
        # SORPRESA
        # =============================

        elif alto_boca > 35 and altura_ceja > 40:

            self.emocion = "SORPRESA"
            self.emoji = "😮"
            self.nota = "RE"
            self.color = "#4DA6FF"
            self.mensaje = "¡Muy bien!"

        # =============================
        # IRA
        # =============================

        elif cercania_cejas < 18:

            self.emocion = "IRA"
            self.emoji = "😠"
            self.nota = "MI"
            self.color = "#FF4444"
            self.mensaje = "Excelente"

        # =============================
        # TRISTEZA
        # =============================

        elif boca_izq.y > boca_inf.y and boca_der.y > boca_inf.y:

            self.emocion = "TRISTEZA"
            self.emoji = "🙁"
            self.nota = "FA"
            self.color = "#3B6BFF"
            self.mensaje = "Muy bien"

        # =============================
        # ALEGRÍA
        # =============================

        elif ancho_boca > 90:

            self.emocion = "ALEGRÍA"
            self.emoji = "😄"
            self.nota = "SOL"
            self.color = "#FFD93D"
            self.mensaje = "¡Excelente!"

detector = DetectorGestos()

# ==========================================
# SISTEMA DE CÁMARA T'ALAT
# ==========================================

def iniciar_camara():
    global camara
    global camara_activa

    if camara_activa:
        return

    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        print("No se pudo abrir la cámara.")
        camara = None
        return

    camara_activa = True

    actualizar_camara()

def detener_camara():

    global camara
    global camara_activa

    camara_activa = False

    if camara is not None:
        camara.release()
        camara = None

    camara_label.configure(
        image=None,
        text="📷 Cámara T'ALAT"
    )

    camara_label.image = None

def actualizar_camara():

    global camara
    global camara_activa

    if not camara_activa:
        return

    if camara is None:
            return

    ret, frame = camara.read()

    if not ret or frame is None:
        app.after(30, actualizar_camara)
        return

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
    )

    resultados = face_mesh.process(rgb)

    if resultados.multi_face_landmarks:
        rostro = resultados.multi_face_landmarks[0]

        alto, ancho, _ = frame.shape

        detector.detectar(
            rostro,
            ancho,
            alto
        )

        emoji_label.configure(
            text=detector.emoji
        )

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

        info_sesion.configure(
            border_width=4,
            border_color=detector.color
        )

    imagen = Image.fromarray(rgb)

    imagen = imagen.resize((1000, 750))

    foto = ImageTk.PhotoImage(imagen)

    camara_label.configure(
        image=foto,
        text=""
    )

    camara_label.image = foto

    app.after(
        20,
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


mensaje_vacio = ctk.CTkLabel(
    usuarios_container,
    text="Aún no hay usuarios registrados.\n\nPresiona 'Agregar usuario' para comenzar.",
    font=("Montserrat", 22),
    text_color=GRIS,
    justify="center"
)

mensaje_vacio.pack(expand=True, pady=150)


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
    text="← Perfil",
    width=140,
    height=45,
    fg_color="#222222",
    hover_color="#333333",
    command=regresar_perfil)

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
    pady=(30,10)
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
    pady=40
)

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

ctk.CTkLabel(
    info,
    text="Tiempo total: 0 min",
    font=("Montserrat",20)
).pack(anchor="w", padx=20, pady=8)

ctk.CTkLabel(
    info,
    text="Sesiones realizadas: 0",
    font=("Montserrat",20)
).pack(anchor="w", padx=20, pady=8)

ctk.CTkLabel(
    info,
    text="Última sesión: Sin sesiones",
    font=("Montserrat",20)
).pack(anchor="w", padx=20, pady=8)

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