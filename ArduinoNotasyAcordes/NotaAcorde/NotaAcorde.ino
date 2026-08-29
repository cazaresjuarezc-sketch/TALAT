/* ============================================================
   TALAT - PIANO DE RELEVADORES
   Arduino UNO + 8 relevadores

     D8 -> rele 0 DO     D3 -> rele 5 FA#
     D7 -> rele 1 RE     D2 -> rele 6 LA
     D6 -> rele 2 MI     D9 -> rele 7 SI
     D5 -> rele 3 FA
     D4 -> rele 4 SOL

   Si no has cableado los tres nuevos, ponles SIN_CONECTAR.
   Velocidad 115200. El modulo de relevadores con su propia
   fuente de 5V y los GND unidos.

   PROTOCOLO:  T:0,2,4:600   pisa reles 0,2,4 por 600 ms
               X             suelta todos
   ============================================================ */

const byte NUM_TECLAS = 8;

// Para los relevadores que todavia no estan cableados.
const byte SIN_CONECTAR = 255;

// 0=DO  1=RE  2=MI  3=FA  4=SOL  5=LA  6=FA#  7=SI
const byte PINES[NUM_TECLAS] = {
  8, 7, 6, 5, 4,
  3, 2, 9
};

// Si al encender suenan TODAS las teclas, cambia esto a false.
const bool ACTIVA_EN_BAJO = true;

// Cuanto se levanta el dedo entre dos golpes de la MISMA nota.
// Sin esto, "RE RE RE" se oye como un solo RE largo.
const unsigned long SEPARACION_REPIQUE_MS = 35;

// Soltar las notas que ya no se piden cuando llega una nota nueva.
const bool CORTAR_ANTERIOR = true;

// Lo minimo que una tecla debe quedarse pisada para que suene.
const unsigned long MINIMO_PISADO_MS = 70;

// Tope de seguridad.
const unsigned long MAXIMO_MS = 2000;

bool pisada[NUM_TECLAS];
unsigned long pisada_desde[NUM_TECLAS];
unsigned long soltar_en[NUM_TECLAS];
unsigned long repisar_en[NUM_TECLAS];

String entrada = "";

// Declaraciones adelantadas.
void pisarTecla(byte tecla, unsigned long ahora);
void soltarTecla(byte tecla);
void soltarTodo();
void leerSerial();
void atenderTiempos();
void procesar(String linea);

void setup() {
  for (byte i = 0; i < NUM_TECLAS; i++) {
    pisada[i] = false;
    pisada_desde[i] = 0;
    soltar_en[i] = 0;
    repisar_en[i] = 0;

    if (PINES[i] == SIN_CONECTAR) {
      continue;
    }

    pinMode(PINES[i], OUTPUT);
    soltarTecla(i);
  }

  Serial.begin(115200);
  entrada.reserve(48);

  Serial.println("TALAT listo");
}

void loop() {
  leerSerial();
  atenderTiempos();
}

/* ---------- comunicacion ---------- */

void leerSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {
      if (entrada.length() > 0) {
        procesar(entrada);
        entrada = "";
      }
    } else if (entrada.length() < 44) {
      entrada += c;
    }
  }
}

void procesar(String linea) {
  linea.trim();

  if (linea.length() == 0) {
    return;
  }

  if (linea.charAt(0) == 'X' || linea.charAt(0) == 'x') {
    soltarTodo();
    Serial.println("OK X");
    return;
  }

  if (linea.charAt(0) != 'T' && linea.charAt(0) != 't') {
    return;
  }

  int primeraDivision = linea.indexOf(':');
  int segundaDivision = linea.indexOf(':', primeraDivision + 1);

  if (primeraDivision < 0 || segundaDivision < 0) {
    return;
  }

  String listaTeclas = linea.substring(primeraDivision + 1, segundaDivision);
  unsigned long duracion = (unsigned long)linea.substring(segundaDivision + 1).toInt();

  if (duracion == 0) {
    duracion = 600;
  }

  if (duracion > MAXIMO_MS) {
    duracion = MAXIMO_MS;
  }

  bool pedida[NUM_TECLAS];

  for (byte i = 0; i < NUM_TECLAS; i++) {
    pedida[i] = false;
  }

  int cuantas = 0;

  while (listaTeclas.length() > 0) {
    int coma = listaTeclas.indexOf(',');
    String pedazo;

    if (coma < 0) {
      pedazo = listaTeclas;
      listaTeclas = "";
    } else {
      pedazo = listaTeclas.substring(0, coma);
      listaTeclas = listaTeclas.substring(coma + 1);
    }

    pedazo.trim();

    if (pedazo.length() == 0) {
      continue;
    }

    int tecla = pedazo.toInt();

    if (tecla >= 0 && tecla < NUM_TECLAS && !pedida[tecla]) {
      pedida[tecla] = true;
      cuantas++;
    }
  }

  unsigned long ahora = millis();

  // Soltar las notas viejas que ya no se piden.
  if (CORTAR_ANTERIOR) {
    for (byte i = 0; i < NUM_TECLAS; i++) {

      if (pedida[i] || !pisada[i]) {
        continue;
      }

      unsigned long minimo = pisada_desde[i] + MINIMO_PISADO_MS;

      soltar_en[i] = (ahora > minimo) ? ahora : minimo;
    }
  }

  // Las notas nuevas.
  for (byte i = 0; i < NUM_TECLAS; i++) {

    if (!pedida[i]) {
      continue;
    }

    if (pisada[i] || repisar_en[i] != 0) {
      soltarTecla(i);

      repisar_en[i] = ahora + SEPARACION_REPIQUE_MS;
      soltar_en[i] = repisar_en[i] + duracion;

    } else {
      pisarTecla(i, ahora);

      repisar_en[i] = 0;
      soltar_en[i] = ahora + duracion;
    }
  }

  Serial.print("OK ");
  Serial.println(cuantas);
}

/* ---------- tiempos ---------- */

void atenderTiempos() {
  unsigned long ahora = millis();

  for (byte i = 0; i < NUM_TECLAS; i++) {

    if (repisar_en[i] != 0 && (long)(ahora - repisar_en[i]) >= 0) {
      pisarTecla(i, ahora);
      repisar_en[i] = 0;
    }

    if (pisada[i] && soltar_en[i] != 0 && (long)(ahora - soltar_en[i]) >= 0) {
      soltarTecla(i);
      soltar_en[i] = 0;
    }
  }
}

/* ---------- teclas ---------- */

void pisarTecla(byte tecla, unsigned long ahora) {
  pisada[tecla] = true;
  pisada_desde[tecla] = ahora;

  if (PINES[tecla] == SIN_CONECTAR) {
    return;
  }

  digitalWrite(PINES[tecla], ACTIVA_EN_BAJO ? LOW : HIGH);
}

void soltarTecla(byte tecla) {
  pisada[tecla] = false;

  if (PINES[tecla] == SIN_CONECTAR) {
    return;
  }

  digitalWrite(PINES[tecla], ACTIVA_EN_BAJO ? HIGH : LOW);
}

void soltarTodo() {
  for (byte i = 0; i < NUM_TECLAS; i++) {
    soltarTecla(i);
    soltar_en[i] = 0;
    repisar_en[i] = 0;
  }
}
