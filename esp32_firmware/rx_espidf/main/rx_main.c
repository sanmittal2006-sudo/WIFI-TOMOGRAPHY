/*
 * Wi-Fi Tomography — RX CSI Receiver (ESP-IDF)
 * ===============================================
 * Connects to TX AP, captures CSI from every packet,
 * and prints data over serial for Python to read.
 *
 * OUTPUT FORMAT:
 *   CSI_DATA,<seq>,<rssi>,<noise>,<len>,[<r0>,<i0>,<r1>,<i1>,...]
 *
 * Build & Flash:
 *   idf.py set-target esp32s3
 *   idf.py build
 *   idf.py -p COM7 flash monitor
 *
 * IMPORTANT: After flashing, close the ESP-IDF monitor
 * before running Python scripts (only one program can
 * use the COM port at a time).
 */

#include <string.h>
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_netif.h"

static const char *TAG = "TOMO_RX";

// ═══════════════════════════════════════════════════════
//   CONFIGURATION — MUST MATCH TX
// ═══════════════════════════════════════════════════════
#define TX_SSID         "TOMO_TX"
#define TX_PASSWORD     "tomography123"
#define MAX_RETRY       20
// ═══════════════════════════════════════════════════════

static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT  BIT0
#define WIFI_FAIL_BIT       BIT1

static int s_retry_num = 0;
static volatile uint32_t csi_count = 0;

// ── CSI Callback ─────────────────────────────────────
// Called by the WiFi driver for EVERY received packet
static void wifi_csi_callback(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf || info->len < 56) {
        return;  // Skip invalid or too-short packets
    }

    int8_t  *buf  = info->buf;
    uint16_t len  = info->len;
    int      rssi = info->rx_ctrl.rssi;
    int      noise = info->rx_ctrl.noise_floor;

    csi_count++;

    // Print in CSV format that Python can parse
    // Format: CSI_DATA,seq,rssi,noise,len,[values...]
    printf("CSI_DATA,%lu,%d,%d,%u,[",
           (unsigned long)csi_count, rssi, noise, len);

    for (int i = 0; i < len; i++) {
        printf("%d", buf[i]);
        if (i < len - 1) {
            printf(",");
        }
    }
    printf("]\n");
}

// ── WiFi Event Handler ───────────────────────────────
static void event_handler(void *arg, esp_event_base_t event_base,
                          int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_num < MAX_RETRY) {
            esp_wifi_connect();
            s_retry_num++;
            ESP_LOGI(TAG, "Reconnecting... attempt %d/%d", s_retry_num, MAX_RETRY);
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Connected! IP: " IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_num = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

// ── WiFi Station Init ────────────────────────────────
static void wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // Register event handlers
    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL, &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL, &instance_got_ip));

    // WiFi config — connect to TX AP
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = TX_SSID,
            .password = TX_PASSWORD,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Connecting to %s ...", TX_SSID);

    // Wait for connection
    EventBits_t bits = xEventGroupWaitBits(
        s_wifi_event_group,
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE, pdFALSE, portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Connected to %s", TX_SSID);
    } else {
        ESP_LOGE(TAG, "FAILED to connect to %s", TX_SSID);
        ESP_LOGE(TAG, "Check: TX powered on? SSID correct? Within range?");
    }
}

// ── Enable CSI Collection ────────────────────────────
static void csi_init(void)
{
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,   // Legacy Long Training Field
        .htltf_en          = true,   // HT Long Training Field
        .stbc_htltf2_en    = true,   // STBC HT LTF2
        .ltf_merge_en      = true,   // Merge LTF
        .channel_filter_en = false,  // RAW CSI (no filtering)
        .manu_scale        = false,
        .shift             = false,
    };

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(TAG, "CSI collection ENABLED");
    ESP_LOGI(TAG, "FORMAT: CSI_DATA,seq,rssi,noise,len,[real0,imag0,real1,imag1,...]");
}

// ── Status Print Task ────────────────────────────────
static void status_task(void *pvParameters)
{
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));  // Every 10 seconds
        // Print as comment so Python ignores it
        printf("# STATUS: %lu CSI packets captured\n", (unsigned long)csi_count);
    }
}

// ── Main ─────────────────────────────────────────────
void app_main(void)
{
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "╔══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║  Wi-Fi Tomography — RX CSI Receiver  ║");
    ESP_LOGI(TAG, "╚══════════════════════════════════════╝");

    // 1. Connect to TX
    wifi_init_sta();

    // 2. Enable CSI capture
    csi_init();

    // 3. Start status reporting
    xTaskCreate(status_task, "status", 2048, NULL, 3, NULL);

    ESP_LOGI(TAG, "Ready! CSI data streaming over serial...");
}
