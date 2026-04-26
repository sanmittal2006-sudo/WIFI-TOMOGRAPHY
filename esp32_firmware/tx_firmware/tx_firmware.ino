/*
 * ╔══════════════════════════════════════════════════════╗
 * ║  Wi-Fi Tomography — TX (Transmitter) Firmware       ║
 * ║  ESP32-S3 DevKitC-1 N16R8                           ║
 * ╚══════════════════════════════════════════════════════╝
 *
 * WHAT THIS DOES:
 *   Creates a WiFi Access Point and continuously sends
 *   UDP broadcast packets. The RX ESP32 captures these
 *   packets and extracts CSI (Channel State Information).
 *
 * HOW TO FLASH:
 *   1. Open Arduino IDE
 *   2. Go to File → Preferences → Additional Board Manager URLs
 *      Add: https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
 *   3. Go to Tools → Board → Board Manager → Search "ESP32" → Install "esp32 by Espressif"
 *   4. Select Board: Tools → Board → ESP32S3 Dev Module
 *   5. Select Port: Tools → Port → (your TX ESP32 COM port)
 *   6. Upload this sketch
 *
 * SETTINGS IN ARDUINO IDE (Tools menu):
 *   Board:           ESP32S3 Dev Module
 *   USB CDC On Boot:  Enabled
 *   Flash Size:       16MB
 *   Partition Scheme: Default 4MB with spiffs
 *   PSRAM:            OPI PSRAM
 */

#include <WiFi.h>
#include <WiFiUdp.h>

// ═══════════════════════════════════════════════════════
//   CONFIGURATION — DO NOT CHANGE UNLESS TOLD TO
// ═══════════════════════════════════════════════════════
const char* AP_SSID     = "TOMO_TX";       // Network name
const char* AP_PASSWORD = "tomography123"; // Password (min 8 chars)
const int   WIFI_CHANNEL = 6;              // WiFi channel (1-13)
const int   TX_POWER_DBM = 20;             // Max transmit power
const int   PACKET_RATE  = 100;            // Packets per second
const int   PACKET_SIZE  = 64;             // Bytes per packet
// ═══════════════════════════════════════════════════════

WiFiUDP udp;
uint32_t packet_count = 0;
uint8_t  tx_buffer[64];

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("╔══════════════════════════════════════╗");
    Serial.println("║  Wi-Fi Tomography — TX Transmitter   ║");
    Serial.println("╚══════════════════════════════════════╝");
    Serial.println();

    // ── Start WiFi Access Point ───────────────────────────
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, AP_PASSWORD, WIFI_CHANNEL);

    // Set maximum TX power for best signal
    WiFi.setTxPower(WIFI_POWER_19_5dBm);

    Serial.print("  AP SSID:    "); Serial.println(AP_SSID);
    Serial.print("  AP IP:      "); Serial.println(WiFi.softAPIP());
    Serial.print("  Channel:    "); Serial.println(WIFI_CHANNEL);
    Serial.print("  TX Power:   "); Serial.print(TX_POWER_DBM); Serial.println(" dBm");
    Serial.print("  Packet Rate:"); Serial.print(PACKET_RATE); Serial.println(" pkt/s");
    Serial.println();
    Serial.println("  STATUS: TRANSMITTING");
    Serial.println("  Sending UDP broadcast packets...");
    Serial.println("  (RX should connect and capture CSI)");
    Serial.println();

    // Start UDP
    udp.begin(1234);

    // Fill TX buffer with known pattern
    for (int i = 0; i < PACKET_SIZE; i++) {
        tx_buffer[i] = (uint8_t)(i & 0xFF);
    }
}

void loop() {
    // Send UDP broadcast packet
    udp.beginPacket("255.255.255.255", 5555);
    
    // Put packet count in first 4 bytes
    tx_buffer[0] = (packet_count >> 24) & 0xFF;
    tx_buffer[1] = (packet_count >> 16) & 0xFF;
    tx_buffer[2] = (packet_count >> 8)  & 0xFF;
    tx_buffer[3] = (packet_count)       & 0xFF;
    
    udp.write(tx_buffer, PACKET_SIZE);
    udp.endPacket();

    packet_count++;

    // Print status every 5 seconds
    if (packet_count % (PACKET_RATE * 5) == 0) {
        Serial.printf("  TX: %u packets sent | %d clients connected\n",
                      packet_count, WiFi.softAPgetStationNum());
    }

    // Control packet rate
    delay(1000 / PACKET_RATE);
}
