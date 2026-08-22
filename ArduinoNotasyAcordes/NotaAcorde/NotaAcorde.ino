// Pines de los Relevadores (Notas físicas del piano: Do, Re, Mi, Fa, Sol, La, Si)
const int RELE_DO  = 8;
const int RELE_RE  = 7;
const int RELE_MI  = 6;
const int RELE_FA  = 5;
const int RELE_SOL = 4;
const int RELE_LA  = 3; // Se mantienen para las melodías
const int RELE_SI  = 2; // Se mantienen para las melodías

// Estructura para definir las notas de la melodía
struct NotaMusical {
  int pinRele;       
  int duracionMs;    
};

// Control de Modos (True = Melodías/Acordes, False = Notas individuales)
bool modoMelodia = true; 

// ==========================================
// DEFINICIÓN DE MELODÍAS (Máximo 15 notas)
// ==========================================
const int NOTAS_ALEGRIA = 9;
NotaMusical mel_alegria[NOTAS_ALEGRIA] = {
  {RELE_DO, 200}, {RELE_MI, 200}, {RELE_SOL, 200}, {RELE_DO, 400},
  {RELE_SOL, 200}, {RELE_DO, 200}, {RELE_MI, 200}, {RELE_SOL, 200}, {RELE_DO, 600}
};

const int NOTAS_TRISTEZA = 6;
NotaMusical mel_tristeza[NOTAS_TRISTEZA] = {
  {RELE_SOL, 500}, {RELE_FA, 500}, {RELE_MI, 500}, {RELE_RE, 500}, {RELE_DO, 800}, {0, 400} 
};

const int NOTAS_SORPRESA = 5;
NotaMusical mel_sorpresa[NOTAS_SORPRESA] = {
  {RELE_DO, 150}, {RELE_SOL, 150}, {RELE_DO, 150}, {RELE_SOL, 150}, {RELE_DO, 500}
};

const int NOTAS_IRA = 8;
NotaMusical mel_ira[NOTAS_IRA] = {
  {RELE_RE, 150}, {RELE_RE, 150}, {RELE_RE, 300}, {RELE_RE, 150}, {RELE_RE, 150}, {RELE_RE, 300}, {RELE_FA, 300}, {RELE_RE, 500}
};

const int NOTAS_ABURRIMIENTO = 4;
NotaMusical mel_aburrimiento[NOTAS_ABURRIMIENTO] = {
  {RELE_MI, 600}, {0, 300}, {RELE_RE, 600}, {RELE_DO, 1000}
};

// Variables de control de flujo de la música (Modo Melodía)
String mensajeRecibido = "";
NotaMusical* melodiaActual = NULL;
int totalNotasActual = 0;
int notaIndex = -1;
unsigned long tiempoNotaAnterior = 0;
bool notaEnProgreso = false;

// Variables de control (Modo Nota Individual)
unsigned long tiempoActivacionNota = 0;
const long DURACION_NOTA = 500;  
int pinNotaActivo = -1;         

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
  // 1. LEER EL PUERTO SERIAL
  while (Serial.available() > 0) {
    char caracter = Serial.read();
    if (caracter == '\n') {
      mensajeRecibido.trim();
      
      // Cambio de modos dinámico enviado desde Python
      if (mensajeRecibido == "MODO_MELODIA") {
        modoMelodia = true;
        resetearEstados();
      } 
      else if (mensajeRecibido == "MODO_NOTA") {
        modoMelodia = false;
        resetearEstados();
      } 
      else {
        // Ejecutar la acción dependiendo del modo activo
        if (modoMelodia) {
          procesarModoMelodia(mensajeRecibido);
        } else {
          procesarModoNota(mensajeRecibido);
        }
      }
      mensajeRecibido = "";
    } else {
      mensajeRecibido += caracter;
    }
  }

  // 2. TEMPORIZADORES EN SEGUNDO PLANO
  if (modoMelodia) {
    // Máquina de estados de melodías
    if (notaIndex != -1 && notaIndex < totalNotasActual) {
      unsigned long tiempoActual = millis();
      
      if (!notaEnProgreso) {
        apagarTodosLosRelevadores();
        int pin = melodiaActual[notaIndex].pinRele;
        if (pin != 0) {
          digitalWrite(pin, LOW); 
        }
        tiempoNotaAnterior = tiempoActual;
        notaEnProgreso = true;
      } 
      else {
        if (tiempoActual - tiempoNotaAnterior >= (unsigned long)melodiaActual[notaIndex].duracionMs) {
          apagarTodosLosRelevadores(); 
          notaEnProgreso = false;
          notaIndex++; 
          
          if (notaIndex >= totalNotasActual) {
            notaIndex = -1; 
            melodiaActual = NULL;
          }
        }
      }
    }
  } else {
    // Temporizador automático de Nota Individual (500 ms)
    if (pinNotaActivo != -1 && (millis() - tiempoActivacionNota >= DURACION_NOTA)) {
      apagarTodosLosRelevadores();
    }
  }
}

// Lógica para reproducir Melodías Completas
void procesarModoMelodia(String estado) {
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
    resetearEstados();
  }
}

// Lógica para reproducir Notas Únicas de 500 ms
void procesarModoNota(String nota) {
  apagarTodosLosRelevadores(); 
  
  if (nota == "C") { digitalWrite(RELE_DO, LOW);  pinNotaActivo = RELE_DO;  tiempoActivacionNota = millis(); }
  else if (nota == "D") { digitalWrite(RELE_RE, LOW);  pinNotaActivo = RELE_RE;  tiempoActivacionNota = millis(); }
  else if (nota == "E") { digitalWrite(RELE_MI, LOW);  pinNotaActivo = RELE_MI;  tiempoActivacionNota = millis(); }
  else if (nota == "F") { digitalWrite(RELE_FA, LOW);  pinNotaActivo = RELE_FA;  tiempoActivacionNota = millis(); }
  else if (nota == "G") { digitalWrite(RELE_SOL, LOW); pinNotaActivo = RELE_SOL; tiempoActivacionNota = millis(); }
  else if (nota == "0" || nota == "REPOSO") { pinNotaActivo = -1; }
}

void iniciarSecuencia() {
  notaIndex = 0;
  notaEnProgreso = false;
}

void resetearEstados() {
  notaIndex = -1;
  melodiaActual = NULL;
  notaEnProgreso = false;
  pinNotaActivo = -1;
  apagarTodosLosRelevadores();
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
