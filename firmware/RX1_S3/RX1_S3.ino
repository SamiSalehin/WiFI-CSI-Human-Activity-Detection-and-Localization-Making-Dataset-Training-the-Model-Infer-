/*
  ============================================================
  RX1_S3
  ESP32-S3 — CSI Receiver #1
  ============================================================

  JOBS
    1. Connect to TX_WROOM as Wi-Fi station.
    2. Capture CSI from every WROOM packet.
    3. Print raw CSI to COM9 → collector.py → rx1.csv.
    4. Continuously send 192 amplitude values to RX2 via UDP
       (RX2 only uses them when testingActive = true).

  NETWORK
    TX_NETWORK  SSID : CSI_TX_NETWORK
                PASS : csi_tx_12345678
                PORT : 3333
                RX1 static IP : 192.168.4.2
                Gateway       : 192.168.4.1

  RX2 TEST UDP
    Destination : 192.168.4.3 : 4444
    Format      : RX1TEST,<seq>,<ts_ms>,<amp0>,...,<amp191>
    amp_i       = (int)round( sqrt(I_i^2 + Q_i^2) )

  SERIAL  COM9  115200 baud

  CSI FORMAT (serial → collector.py)
    RX1,<ts_ms>,<rssi>,<seq>,<ch>,<bw>,<len>,I0,Q0,I1,Q1,...

  WROOM MAC FILTER
    Only CSI from WROOM BSSID = 84:0D:8E:E8:31:29 is kept.
  ============================================================
*/

#include <WiFi.h>
#include <WiFiUdp.h>

extern "C" {
    #include "esp_wifi.h"
}


/* ============================================================
   NETWORK CONFIGURATION
   ============================================================ */

const char*    TX_SSID = "CSI_TX_NETWORK";
const char*    TX_PASS = "csi_tx_12345678";
const uint16_t TX_PORT = 3333;

/* RX1 static IP on the TX network */
const IPAddress RX1_IP      (192, 168, 4, 2);
const IPAddress GATEWAY_IP  (192, 168, 4, 1);
const IPAddress SUBNET_MASK (255, 255, 255, 0);

/* RX2 station IP on the TX network + test UDP port */
const IPAddress RX2_TEST_IP (192, 168, 4, 3);
const uint16_t  RX2_TEST_PORT = 4444;

/* Accept CSI only from this WROOM transmitter */
const uint8_t WROOM_MAC[6] = {
    0x84, 0x0D, 0x8E, 0xE8, 0x31, 0x29
};


/* ============================================================
   DEFINES
   ============================================================ */

#define MAX_CSI_BYTES 512
#define RX_FEATURES   192     /* subcarriers per receiver */


/* ============================================================
   UDP HANDLES
   ============================================================ */

WiFiUDP udpTx;      /* receives TX heartbeat packets on TX_PORT   */
WiFiUDP udpRX2;     /* sends amplitude data to RX2 on RX2_TEST_PORT */


/* ============================================================
   CSI SHARED STATE  (written by IRAM callback, read by loop)
   ============================================================ */

volatile bool    newCSI     = false;
volatile int     csiLength  = 0;
volatile int     csiRSSI    = 0;
volatile int     csiChannel = 0;
int8_t           csiBuffer[MAX_CSI_BYTES];


/* ============================================================
   COUNTERS  (diagnostic)
   ============================================================ */

uint32_t          csiPacketCount   = 0;   /* CSI records printed    */
volatile uint32_t csiCallbackCount = 0;   /* callback invocations   */
volatile uint32_t csiAccepted      = 0;   /* passed WROOM filter    */
volatile uint32_t csiRejected      = 0;   /* rejected (not WROOM)   */
volatile uint32_t udpRxCount       = 0;   /* TX heartbeat packets   */
volatile uint32_t udpTxCount       = 0;   /* UDP packets sent to RX2 */

bool         wifiConnected  = false;
unsigned long lastDebugTime = 0;


/* ============================================================
   CSI CALLBACK   (runs in IRAM — must be very fast)
   ============================================================ */

void IRAM_ATTR wifiCSI_callback(void* ctx, wifi_csi_info_t* info)
{
    if (!info || !info->buf) return;

    csiCallbackCount++;

    /* ── WROOM MAC filter ──────────────────────────────── */
    for (int i = 0; i < 6; i++) {
        if (info->mac[i] != WROOM_MAC[i]) {
            csiRejected++;
            return;
        }
    }
    csiAccepted++;

    /* ── Copy into shared buffer ───────────────────────── */
    int len = info->len;
    if (len <= 0)              return;
    if (len > MAX_CSI_BYTES)   len = MAX_CSI_BYTES;

    for (int i = 0; i < len; i++) csiBuffer[i] = info->buf[i];

    csiLength  = len;
    csiRSSI    = info->rx_ctrl.rssi;
    csiChannel = info->rx_ctrl.channel;
    newCSI     = true;
}


/* ============================================================
   ENABLE CSI
   ============================================================ */

void enableCSI()
{
    wifi_csi_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.lltf_en           = true;
    cfg.htltf_en          = true;
    cfg.stbc_htltf2_en    = true;
    cfg.ltf_merge_en      = true;
    cfg.channel_filter_en = false;
    cfg.manu_scale        = false;
    cfg.shift             = false;

    esp_err_t err;

    err = esp_wifi_set_csi_config(&cfg);
    Serial.print("CSI_CONFIG_RESULT:");
    Serial.println((int)err);

    err = esp_wifi_set_csi_rx_cb(&wifiCSI_callback, nullptr);
    Serial.print("CSI_CB_RESULT:");
    Serial.println((int)err);

    err = esp_wifi_set_csi(true);
    Serial.print("CSI_ENABLE_RESULT:");
    Serial.println((int)err);

    if (err == ESP_OK) Serial.println("CSI_ENABLED");
    else               Serial.println("CSI_ENABLE_FAILED");
}


/* ============================================================
   SEND AMPLITUDE PACKET TO RX2
   ============================================================
   Sends 192 amplitude values (one per subcarrier) to RX2.
   RX2 buffers these in its circular buffer and uses them
   only when testingActive = true.

   Format (ASCII over UDP):
       RX1TEST,<seq>,<ts_ms>,<amp0>,<amp1>,...,<amp191>

   Each amp_i = round(sqrt(I_i^2 + Q_i^2))   (integer ≥ 0)
   ============================================================ */

void sendAmplitudesToRX2(
    const int8_t* csi,
    int           len,
    uint32_t      sequence,
    uint32_t      timestamp)
{
    /* Need at least 192 I/Q pairs = 384 bytes */
    if (len < 384) return;

    /* Build packet string */
    char packet[1600];
    int  used = snprintf(
        packet, sizeof(packet),
        "RX1TEST,%lu,%lu",
        (unsigned long)sequence,
        (unsigned long)timestamp
    );
    if (used <= 0 || used >= (int)sizeof(packet)) return;

    for (int sc = 0; sc < RX_FEATURES; sc++) {
        int   I   = (int)csi[sc * 2];
        int   Q   = (int)csi[sc * 2 + 1];
        int   mag = (int)lroundf(sqrtf((float)(I*I + Q*Q)));

        int w = snprintf(
            packet + used,
            sizeof(packet) - used,
            ",%d", mag
        );
        if (w <= 0 || w >= (int)(sizeof(packet) - used)) return;
        used += w;
    }

    udpRX2.beginPacket(RX2_TEST_IP, RX2_TEST_PORT);
    udpRX2.write((const uint8_t*)packet, (size_t)used);
    udpRX2.endPacket();

    udpTxCount++;
}


/* ============================================================
   HANDLE CSI   (called from loop — safe to use Serial)
   ============================================================ */

void handleCSI()
{
    if (!newCSI) return;

    /* ── Safe copy from ISR-shared state ──────────────── */
    int    len;
    int    rssi;
    int    channel;
    int8_t localCSI[MAX_CSI_BYTES];

    noInterrupts();
    len     = csiLength;
    rssi    = csiRSSI;
    channel = csiChannel;
    for (int i = 0; i < len; i++) localCSI[i] = csiBuffer[i];
    newCSI  = false;
    interrupts();

    if (len <= 0) return;

    uint32_t timestamp = millis();
    uint32_t sequence  = csiPacketCount++;

    /* ── Send amplitudes to RX2 for live inference ─────── */
    sendAmplitudesToRX2(localCSI, len, sequence, timestamp);

    /* ── Print raw CSI to serial → collector.py ─────────
       Format:
         RX1,<ts_ms>,<rssi>,<seq>,<ch>,<bw>,<csi_len>,I0,Q0,I1,Q1,...
    ──────────────────────────────────────────────────── */
    Serial.print("RX1,");
    Serial.print(timestamp);   Serial.print(",");
    Serial.print(rssi);        Serial.print(",");
    Serial.print(sequence);    Serial.print(",");
    Serial.print(channel);     Serial.print(",");
    Serial.print("unknown");   Serial.print(",");
    Serial.print(len);         Serial.print(",");

    for (int i = 0; i < len; i++) {
        Serial.print((int)localCSI[i]);
        if (i < len - 1) Serial.print(",");
    }
    Serial.println();
}


/* ============================================================
   PERIODIC DEBUG STATUS  (every 5 seconds)
   ============================================================ */

void printDebugStatus()
{
    if (millis() - lastDebugTime < 5000UL) return;
    lastDebugTime = millis();

    Serial.println("--- RX1 STATUS ---");
    Serial.print("WiFi        : "); Serial.println(WiFi.status());
    Serial.print("RSSI        : "); Serial.println(WiFi.RSSI());
    Serial.print("CSI cb      : "); Serial.println(csiCallbackCount);
    Serial.print("CSI accepted: "); Serial.println(csiAccepted);
    Serial.print("CSI rejected: "); Serial.println(csiRejected);
    Serial.print("CSI printed : "); Serial.println(csiPacketCount);
    Serial.print("UDP rx (TX) : "); Serial.println(udpRxCount);
    Serial.print("UDP tx (RX2): "); Serial.println(udpTxCount);
    Serial.println("------------------");
}


/* ============================================================
   SETUP
   ============================================================ */

void setup()
{
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("============================");
    Serial.println("RX1 ESP32-S3  CSI RECEIVER");
    Serial.println("============================");

    /* ── Wi-Fi station mode ────────────────────────────── */
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);

    /* Static IP so RX2 can reliably address us */
    if (!WiFi.config(RX1_IP, GATEWAY_IP, SUBNET_MASK)) {
        Serial.println("STATIC_IP_CONFIG_FAILED");
    } else {
        Serial.print("STATIC_IP:");
        Serial.println(RX1_IP);
    }

    /* ── Connect to TX WROOM AP ────────────────────────── */
    Serial.println("CONNECTING_TO_TX...");
    WiFi.begin(TX_SSID, TX_PASS);

    unsigned long t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000UL) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        wifiConnected = true;
        Serial.println("WIFI_CONNECTED");
        Serial.print("IP    : "); Serial.println(WiFi.localIP());
        Serial.print("BSSID : "); Serial.println(WiFi.BSSIDstr());
        Serial.print("Ch    : "); Serial.println(WiFi.channel());
        Serial.print("RSSI  : "); Serial.println(WiFi.RSSI());
    } else {
        wifiConnected = false;
        Serial.println("WIFI_FAILED — check TX power and SSID.");
    }

    /* ── UDP sockets ───────────────────────────────────── */

    /* Listen on TX_PORT for TX heartbeat packets */
    if (udpTx.begin(TX_PORT)) {
        Serial.print("UDP_RX_READY:");
        Serial.println(TX_PORT);
    } else {
        Serial.println("UDP_RX_FAILED");
    }

    /* Open socket for sending amplitude data to RX2 */
    /* Any local port is fine — we are the sender */
    if (udpRX2.begin(4445)) {
        Serial.print("UDP_TX_TO_RX2_READY → ");
        Serial.print(RX2_TEST_IP);
        Serial.print(":");
        Serial.println(RX2_TEST_PORT);
    } else {
        Serial.println("UDP_TX_TO_RX2_FAILED");
    }

    /* ── Enable CSI ────────────────────────────────────── */
    enableCSI();

    /* ── Ready ─────────────────────────────────────────── */
    Serial.println();
    Serial.println("FORMAT:");
    Serial.println("  RX1,ts_ms,rssi,seq,ch,bw,len,I0,Q0,I1,Q1,...");
    Serial.println();
    Serial.println("RX1_READY");
    Serial.println();
}


/* ============================================================
   LOOP
   ============================================================ */

void loop()
{
    /* ── Drain TX heartbeat UDP ────────────────────────── */
    int pktSize = udpTx.parsePacket();
    if (pktSize > 0) {
        char buf[128];
        udpTx.read(buf, sizeof(buf) - 1);
        udpRxCount++;
    }

    /* ── Process CSI (print + send to RX2) ────────────── */
    handleCSI();

    /* ── Periodic diagnostics ──────────────────────────── */
    printDebugStatus();

    delay(1);
}




