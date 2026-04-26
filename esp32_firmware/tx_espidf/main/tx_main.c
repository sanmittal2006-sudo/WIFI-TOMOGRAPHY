/*
 * Wi-Fi Tomography — TX Transmitter (ESP-IDF)
 * =============================================
 * Creates WiFi AP and sends UDP broadcast packets.
 * RX captures these packets and extracts CSI.
 *
 * Build & Flash:
 *   idf.py set-target esp32s3
 *   idf.py build
 *   idf.py -p COM5 flash monitor
 */

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "lwip/sockets.h"

static const char *TAG = "TOMO_TX";

// ═══════════════════════════════════════════════════════
//   CONFIGURATION
// ═══════════════════════════════════════════════════════
#define AP_SSID         "TOMO_TX"
#define AP_PASSWORD     "tomography123"
#define AP_CHANNEL      6
#define MAX_STA_CONN    2
#define UDP_PORT        5555
#define PACKET_RATE_MS  10    // 100 packets/sec
#define PACKET_SIZE     64
// ═══════════════════════════════════════════════════════

static uint32_t packet_count = 0;

// WiFi event handler
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_id == WIFI_EVENT_AP_STACONNECTED) {
        wifi_event_ap_staconnected_t *event = (wifi_event_ap_staconnected_t *)event_data;
        ESP_LOGI(TAG, "Station connected, AID=%d", event->aid);
    } else if (event_id == WIFI_EVENT_AP_STADISCONNECTED) {
        wifi_event_ap_stadisconnected_t *event = (wifi_event_ap_stadisconnected_t *)event_data;
        ESP_LOGI(TAG, "Station disconnected, AID=%d", event->aid);
    }
}

// Initialize WiFi as Access Point
static void wifi_init_softap(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));

    wifi_config_t wifi_config = {
        .ap = {
            .ssid = AP_SSID,
            .ssid_len = strlen(AP_SSID),
            .channel = AP_CHANNEL,
            .password = AP_PASSWORD,
            .max_connection = MAX_STA_CONN,
            .authmode = WIFI_AUTH_WPA2_PSK,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    // Set max TX power
    ESP_ERROR_CHECK(esp_wifi_set_max_tx_power(84));  // 21 dBm

    ESP_LOGI(TAG, "AP started. SSID: %s, Channel: %d", AP_SSID, AP_CHANNEL);
}

// UDP broadcast task — sends packets that RX captures as CSI
static void udp_broadcast_task(void *pvParameters)
{
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        ESP_LOGE(TAG, "Failed to create socket");
        vTaskDelete(NULL);
        return;
    }

    // Enable broadcast
    int broadcast = 1;
    setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));

    struct sockaddr_in dest_addr;
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(UDP_PORT);
    dest_addr.sin_addr.s_addr = htonl(INADDR_BROADCAST);

    uint8_t tx_buf[PACKET_SIZE];
    memset(tx_buf, 0, sizeof(tx_buf));

    ESP_LOGI(TAG, "Starting UDP broadcast on port %d", UDP_PORT);
    ESP_LOGI(TAG, "Packet rate: %d pkt/s", 1000 / PACKET_RATE_MS);

    while (1) {
        // Fill packet with counter
        tx_buf[0] = (packet_count >> 24) & 0xFF;
        tx_buf[1] = (packet_count >> 16) & 0xFF;
        tx_buf[2] = (packet_count >> 8)  & 0xFF;
        tx_buf[3] = (packet_count)       & 0xFF;

        sendto(sock, tx_buf, PACKET_SIZE, 0,
               (struct sockaddr *)&dest_addr, sizeof(dest_addr));

        packet_count++;

        if (packet_count % 500 == 0) {
            wifi_sta_list_t sta_list;
            esp_wifi_ap_get_sta_list(&sta_list);
            ESP_LOGI(TAG, "TX: %lu packets sent | %d clients", 
                     (unsigned long)packet_count, sta_list.num);
        }

        vTaskDelay(pdMS_TO_TICKS(PACKET_RATE_MS));
    }
}

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
    ESP_LOGI(TAG, "║  Wi-Fi Tomography — TX Transmitter   ║");
    ESP_LOGI(TAG, "╚══════════════════════════════════════╝");

    wifi_init_softap();

    // Start UDP broadcast task
    xTaskCreate(udp_broadcast_task, "udp_tx", 4096, NULL, 5, NULL);
}
