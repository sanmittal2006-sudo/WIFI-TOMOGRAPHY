/*
 * ╔══════════════════════════════════════════════════════╗
 * ║  Wi-Fi Tomography — RX (Receiver) CSI Firmware      ║
 * ║  ESP32-S3 DevKitC-1 N16R8                           ║
 * ╚══════════════════════════════════════════════════════╝
 *
 * WHAT THIS DOES:
 *   Connects to the TX Access Point, enables CSI capture,
 *   and prints every CSI packet over Serial in a format
 *   that the Python script can parse.
 *
 * OUTPUT FORMAT (one line per WiFi packet received):
 *   CSI_DATA,<packet_seq>,<rssi>,<noise_floor>,<num_values>,[<val0>,<val1>,...]
 *
 *   Where values are int8 pairs: real0, imag0, real1, imag1, ...
 *   So for 52 subcarriers you get 104 values.
 *
 * HOW TO FLASH:
 *   1. Open Arduino IDE (same setup as TX)
 *   2. Select Board: Tools → Board → ESP32S3 Dev Module
 *   3. Select Port: Tools → Port → (your RX ESP32 COM port)
 *   4. Upload this sketch
 *
 * SETTINGS IN ARDUINO IDE (Tools menu):
 *   Board:           ESP32S3 Dev Module
 *   USB CDC On Boot:  Enabled
 *   Flash Size:       16MB
 *   Partition Scheme: Default 4MB with spiffs
 *   PSRAM:            OPI PSRAM
 *
 * IMPORTANT:
 *   - Flash TX first, power it on
 *   - Then flash and power on RX
 *   - RX connects to TX automatically
 *   - Open Serial Monitor at 921600 baud to see CSI data
 */

#include <WiFi.h>
#include <esp_wifi.h>
#include "esp_wifi_types.h"

// ═══════════════════════════════════════════════════════
//   CONFIGURATION — MUST MATCH TX FIRMWARE
// ═══════════════════════════════════════════════════════
const char* TX_SSID     = "TOMO_TX";       // Must match TX
const char* TX_PASSWORD = "tomography123"; // Must match TX
// ═══════════════════════════════════════════════════════

// Global packet counter
volatile uint32_t csi_packet_count = 0;
volatile uint32_t last_print_time  = 0;

// ── CSI Callback Function ─────────────────────────────
// Called automatically by ESP32 for every WiFi packet received
// This is where we extract and print the CSI data
void wifi_csi_callback(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf) {
        return;  // Invalid packet, skip
    }

    // Extract packet metadata
    int8_t  *csi_raw  = info->buf;         // Raw CSI values (int8)
    uint16_t csi_len  = info->len;         // Number of int8 values
    int      rssi     = info->rx_ctrl.rssi;
    int      noise    = info->rx_ctrl.noise_floor;

    // Skip packets with too few subcarriers
    if (csi_len < 56) {  // Need at least 28 subcarrier pairs
        return;
    }

    csi_packet_count++;

    // ── Print in machine-readable CSV format ──────────
    // Format: CSI_DATA,seq,rssi,noise,len,[val0,val1,val2,...]
    Serial.printf("CSI_DATA,%u,%d,%d,%u,[",
        csi_packet_count,
        rssi,
        noise,
        csi_len
    );

    // Print all CSI values (real, imag pairs)
    for (int i = 0; i < csi_len; i++) {
        Serial.print(csi_raw[i]);
        if (i < csi_len - 1) {
            Serial.print(",");
        }
    }
    Serial.println("]");
}

void setup() {
    // Start Serial at high baud rate (CSI generates lots of data)
    Serial.begin(921600);
    delay(1000);

    Serial.println();
    Serial.println("╔══════════════════════════════════════╗");
    Serial.println("║  Wi-Fi Tomography — RX CSI Receiver  ║");
    Serial.println("╚══════════════════════════════════════╝");
    Serial.println();

    // ── Connect to TX Access Point ────────────────────────
    Serial.printf("  Connecting to TX: %s\n", TX_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(TX_SSID, TX_PASSWORD);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
        attempts++;
        if (attempts > 40) {  // 20 second timeout
            Serial.println();
            Serial.println("  ERROR: Cannot connect to TX!");
            Serial.println("  Check:");
            Serial.println("    1. TX ESP32 is powered ON");
            Serial.println("    2. TX SSID = 'TOMO_TX'");
            Serial.println("    3. TX and RX are within range");
            Serial.println();
            Serial.println("  Restarting in 5 seconds...");
            delay(5000);
            ESP.restart();
        }
    }

    Serial.println();
    Serial.printf("  Connected! IP: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("  RSSI: %d dBm\n", WiFi.RSSI());
    Serial.println();

    // ── Configure CSI Collection ──────────────────────────
    wifi_csi_config_t csi_config;
    csi_config.lltf_en           = true;   // Long Training Field (802.11a/g)
    csi_config.htltf_en          = true;   // HT Long Training Field (802.11n)
    csi_config.stbc_htltf2_en    = true;   // STBC HT LTF2
    csi_config.ltf_merge_en      = true;   // Merge multiple LTFs
    csi_config.channel_filter_en = false;  // RAW CSI — no hardware filtering
    csi_config.manu_scale        = false;  // No manual scaling
    csi_config.shift             = false;  // No bit shifting

    // Enable CSI
    esp_err_t ret;
    ret = esp_wifi_set_csi_config(&csi_config);
    if (ret != ESP_OK) {
        Serial.printf("  ERROR: CSI config failed: %d\n", ret);
    }

    ret = esp_wifi_set_csi_rx_cb(wifi_csi_callback, NULL);
    if (ret != ESP_OK) {
        Serial.printf("  ERROR: CSI callback failed: %d\n", ret);
    }

    ret = esp_wifi_set_csi(true);
    if (ret != ESP_OK) {
        Serial.printf("  ERROR: CSI enable failed: %d\n", ret);
    }

    Serial.println("  ✓ CSI collection ENABLED");
    Serial.println();
    Serial.println("  OUTPUT FORMAT:");
    Serial.println("  CSI_DATA,seq,rssi,noise,len,[real0,imag0,real1,imag1,...]");
    Serial.println();
    Serial.println("  Waiting for packets...");
    Serial.println("  ─────────────────────────────────────");
}

void loop() {
    // Reconnect if WiFi drops
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("  WARNING: WiFi disconnected! Reconnecting...");
        WiFi.begin(TX_SSID, TX_PASSWORD);
        int attempts = 0;
        while (WiFi.status() != WL_CONNECTED && attempts < 20) {
            delay(500);
            attempts++;
        }
        if (WiFi.status() == WL_CONNECTED) {
            Serial.println("  Reconnected!");
        }
    }

    // Print heartbeat every 10 seconds
    if (millis() - last_print_time > 10000) {
        Serial.printf("# STATUS: %u CSI packets captured | RSSI: %d dBm | WiFi: %s\n",
                      csi_packet_count,
                      WiFi.RSSI(),
                      WiFi.status() == WL_CONNECTED ? "OK" : "DISCONNECTED");
        last_print_time = millis();
    }

    delay(100);
}
