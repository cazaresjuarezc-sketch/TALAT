// Asignación de Pines para los 5 Relevadores (Notas: Do, Re, Mi, Fa, Sol)
const int RELE_DO  = 8;  // Letra 'C' (Aburrimiento)
const int RELE_RE  = 7;  // Letra 'D' (Sorpresa)
const int RELE_MI  = 6;  // Letra 'E' (Ira)
const int RELE_FA  = 5;  // Letra 'F' (Tristeza)
const int RELE_SOL = 4;  // Letra 'G' (Alegría)

// Variables para el control de tiempo (Medio segundo)
unsigned long tiempoActivacion = 0;
const long DURACION = 1000;  // 1 segundos por nota
int pinActivo = -1;         // Guarda qué relé está encendido actualmente

void setup() {
  Serial.begin(9600);
  
  // Configurar los pines como salidas
  pinMode(RELE_DO, OUTPUT);
  pinMode(RELE_RE, OUTPUT);
  pinMode(RELE_MI, OUTPUT);
  pinMode(RELE_FA, OUTPUT);
  pinMode(RELE_SOL, OUTPUT);
  
  // Estado inicial: TODOS APAGADOS (HIGH para lógica inversa)
  apagarTodosLosRelevadores();
}

void loop() {
  // 1. LEER EL PUERTO SERIAL (Escucha de notas individuales)
  if (Serial.available() > 0) {
    char nota = Serial.read();
    
    // Saltarse los caracteres de control de línea si Python los envía
    if (nota != '\n' && nota != '\r') {
      apagarTodosLosRelevadores(); // Limpia la nota anterior de inmediato
      
      // Activar el relevador correspondiente según la letra recibida
      if (nota == 'C') { digitalWrite(RELE_DO, LOW);  pinActivo = RELE_DO;  tiempoActivacion = millis(); }
      if (nota == 'D') { digitalWrite(RELE_RE, LOW);  pinActivo = RELE_RE;  tiempoActivacion = millis(); }
      if (nota == 'E') { digitalWrite(RELE_MI, LOW);  pinActivo = RELE_MI;  tiempoActivacion = millis(); }
      if (nota == 'F') { digitalWrite(RELE_FA, LOW);  pinActivo = RELE_FA;  tiempoActivacion = millis(); }
      if (nota == 'G') { digitalWrite(RELE_SOL, LOW); pinActivo = RELE_SOL; tiempoActivacion = millis(); }
      
      // Si se recibe '0' o cualquier otro carácter de reposo, se queda apagado
      if (nota == '0') { pinActivo = -1; }
    }
  }

  // 2. TEMPORIZADOR AUTOMÁTICO (Apaga la nota después de 500 ms)
  if (pinActivo != -1 && (millis() - tiempoActivacion >= DURACION)) {
    apagarTodosLosRelevadores();
  }
}

// Función auxiliar para apagar todas las salidas de golpe
void apagarTodosLosRelevadores() {
  digitalWrite(RELE_DO, HIGH);
  digitalWrite(RELE_RE, HIGH);
  digitalWrite(RELE_MI, HIGH);
  digitalWrite(RELE_FA, HIGH);
  digitalWrite(RELE_SOL, HIGH);
  pinActivo = -1; // Resetea el rastro de relé activo
}
