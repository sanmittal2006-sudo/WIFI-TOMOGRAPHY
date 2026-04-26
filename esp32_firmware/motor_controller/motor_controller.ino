/*
 * Wi-Fi Tomography — Stepper Motor Controller
 * =============================================
 * Arduino Uno controls NEMA 17 via A4988/DRV8825.
 * Waits for serial commands from Python.
 *
 * COMMANDS (send over Serial at 9600 baud):
 *   MOVE    → rotate 22.5° clockwise, replies "DONE"
 *   HOME    → rotate back to 0°, replies "DONE"
 *   STEP:N  → rotate N steps, replies "DONE"
 *
 * Wiring:
 *   Arduino pin 2 → DIR pin on driver
 *   Arduino pin 3 → STEP pin on driver
 */

const int dirPin  = 2;
const int stepPin = 3;

// 25 half-steps = 22.5 degrees (your working value)
const int STEPS_PER_POSITION = 25;
const int STEP_DELAY_US      = 10000;  // slower = MORE TORQUE (was 5000)

int current_position = 0;  // Track which position (0-15)

void setup() {
    Serial.begin(9600);
    pinMode(dirPin, OUTPUT);
    pinMode(stepPin, OUTPUT);
    
    // Set direction clockwise
    digitalWrite(dirPin, HIGH);
    
    delay(1000);
    Serial.println("MOTOR_READY");
}

// Move N steps
void moveSteps(int steps) {
    for (int i = 0; i < steps; i++) {
        digitalWrite(stepPin, HIGH);
        delayMicroseconds(STEP_DELAY_US);
        digitalWrite(stepPin, LOW);
        delayMicroseconds(STEP_DELAY_US);
    }
}

void loop() {
    if (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();

        if (cmd == "MOVE") {
            // Move one position (22.5 degrees)
            moveSteps(STEPS_PER_POSITION);
            current_position++;
            if (current_position >= 16) current_position = 0;
            Serial.print("DONE,");
            Serial.println(current_position);
            
        } else if (cmd == "HOME") {
            // Return to position 0
            int steps_back = (16 - current_position) * STEPS_PER_POSITION;
            moveSteps(steps_back);
            current_position = 0;
            Serial.println("DONE,0");
            
        } else if (cmd.startsWith("STEP:")) {
            // Custom step count
            int n = cmd.substring(5).toInt();
            moveSteps(n);
            Serial.print("STEPPED,");
            Serial.println(n);
            
        } else if (cmd == "POS") {
            // Report current position
            Serial.print("POS,");
            Serial.println(current_position);
            
        } else if (cmd == "PING") {
            Serial.println("PONG");
        }
    }
}
