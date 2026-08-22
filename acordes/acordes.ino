// Pines de los Relevadores (Notas físicas del piano: Do, Re, Mi, Fa, Sol, La, Si)
const int RELE_DO  = 8;
const int RELE_RE  = 7;
const int RELE_MI  = 6;
const int RELE_FA  = 5;
const int RELE_SOL = 4;
const int RELE_LA  = 3;
const int RELE_SI  = 2;

// Estructura para definir las notas de la melodía
struct NotaMusical {
  int pinRele;       // Qué relé/nota física debe activarse
  int duracionMs;    // Cuánto dura la nota en milisegundos
};

// ==========================================
// DEFINICIÓN DE MELODÍAS (Máximo 15 notas)
// ==========================================

// 1. Alegría: Melodía brillante y ascendente en la escala
const int NOTAS_ALEGRIA = 9;
NotaMusical mel_alegria[NOTAS_ALEGRIA] = {
  {RELE_DO, 200}, {RELE_MI, 200}, {RELE_SOL, 200}, {RELE_DO, 400},
  {RELE_SOL, 200}, {RELE_DO, 200}, {RELE_MI, 200}, {RELE_SOL, 200}, {RELE_DO, 600}
};

// 2. Tristeza: Melodía lenta y descendente
const int NOTAS_TRISTEZA = 6;
NotaMusical mel_tristeza[NOTAS_TRISTEZA] = {
  {RELE_SOL, 500}, {RELE_FA, 500}, {RELE_MI, 500}, {RELE_RE, 500}, {RELE_DO, 800}, {0, 400} // El 0 es un silencio
};

// 3. Sorpresa: Rápido y rítmico
const int NOTAS_SORPRESA = 5;
NotaMusical mel_sorpresa[NOTAS_SORPRESA] = {
  {RELE_DO, 150}, {RELE_SOL, 150}, {RELE_DO, 150}, {RELE_SOL, 150}, {RELE_DO, 500}
};

// 4. Ira: Notas graves y repetitivas (Tensión)
const int NOTAS_IRA = 8;
NotaMusical mel_ira[NOTAS_IRA] = {
  {RELE_RE, 150}, {RELE_RE, 150}, {RELE_RE, 300}, {RELE_RE, 150}, {RELE_RE, 150}, {RELE_RE, 300}, {RELE_FA, 300}, {RELE_RE, 500}
};

// 5. Aburrimiento: Notas largas y espaciadas
const int NOTAS_ABURRIMIENTO = 4;
NotaMusical mel_aburrimiento[NOTAS_ABURRIMIENTO] = {
  {RELE_MI, 600}, {0, 300}, {RELE_RE, 600}, {RELE_DO, 1000}
};

// Variables de control de flujo de la música
String mensajeRecibido = "";
NotaMusical* melodiaActual = NULL;
int totalNotasActual = 0;
int notaIndex = -1;
unsigned long tiempoNotaAnterior = 0;
bool notaEnProgreso = false;

void setup() {
  Serial.begin(9600);
  
  pinMode(RELE_DO, OUTPUT);
  pinMode(RELE_RE, OUTPUT);
  pinMode(RELE_MI, OUTPUT);
  pinMode(RELE_FA, OUTPUT);
  pinMode(RELE_SOL, OUTPUT);
  pinMode(RELE_LA, OUTPUT);
  pinMode(RELE_SI, OUTPUT);
  
  apagarTodosLosRelevadores();
}

void loop() {
  // 1. LEER EL PUERTO SERIAL (Siempre activo y receptivo)
  while (Serial.available() > 0) {
    char caracter = Serial.read();
    if (caracter == '\n') {
      mensajeRecibido.trim();
      cambiarMelodia(mensajeRecibido);
      mensajeRecibido = "";
    } else {
      mensajeRecibido += caracter;
    }
  }

  // 2. MÁQUINA DE ESTADOS PARA CONTROLAR LA REPRODUCCIÓN
  if (notaIndex != -1 && notaIndex < totalNotasActual) {
    unsigned long tiempoActual = millis();
    
    if (!notaEnProgreso) {
      // Tocar la siguiente nota de la lista
      apagarTodosLosRelevadores();
      int pin = melodiaActual[notaIndex].pinRele;
      
      if (pin != 0) {
        digitalWrite(pin, LOW); // Activa el relé de esa nota
      }
      
      tiempoNotaAnterior = tiempoActual;
      notaEnProgreso = true;
    } 
    else {
      // Verificar si ya terminó el tiempo de duración de la nota actual
      if (tiempoActual - tiempoNotaAnterior >= (unsigned long)melodiaActual[notaIndex].duracionMs) {
        apagarTodosLosRelevadores(); // Desactiva el relé
        notaEnProgreso = false;
        notaIndex++; // Avanza a la siguiente nota de la melodía
        
        // Si llegó al final de la melodía, se detiene
        if (notaIndex >= totalNotasActual) {
          notaIndex = -1; 
          melodiaActual = NULL;
        }
      }
    }
  }
}

// Configura las variables para iniciar la melodía correspondiente
void cambiarMelodia(String estado) {
  if (estado == "ALEGRIA") {
    melodiaActual = mel_alegria;
    totalNotasActual = NOTAS_ALEGRIA;
    iniciarSecuencia();
  } 
  else if (estado == "TRISTEZA") {
    melodiaActual = mel_tristeza;
    totalNotasActual = NOTAS_TRISTEZA;
    iniciarSecuencia();
  } 
  else if (estado == "SORPRESA") {
    melodiaActual = mel_sorpresa;
    totalNotasActual = NOTAS_SORPRESA;
    iniciarSecuencia();
  } 
  else if (estado == "IRA") {
    melodiaActual = mel_ira;
    totalNotasActual = NOTAS_IRA;
    iniciarSecuencia();
  } 
  else if (estado == "ABURRIMIENTO") {
    melodiaActual = mel_aburrimiento;
    totalNotasActual = NOTAS_ABURRIMIENTO;
    iniciarSecuencia();
  }
  else if (estado == "REPOSO") {
    // Si la cara vuelve a reposo, corta la música de golpe de forma opcional
    notaIndex = -1;
    melodiaActual = NULL;
    notaEnProgreso = false;
    apagarTodosLosRelevadores();
  }
}

void iniciarSecuencia() {
  notaIndex = 0;
  notaEnProgreso = false;
}

void apagarTodosLosRelevadores() {
  digitalWrite(RELE_DO, HIGH);
  digitalWrite(RELE_RE, HIGH);
  digitalWrite(RELE_MI, HIGH);
  digitalWrite(RELE_FA, HIGH);
  digitalWrite(RELE_SOL, HIGH);
  digitalWrite(RELE_LA, HIGH);
  digitalWrite(RELE_SI, HIGH);
}
