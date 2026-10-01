/*
  ============================================================
  TX_WROOM
  ESP32-WROOM — Dedicated Wi-Fi CSI Transmitter
  ============================================================

  ROLE
    This board has exactly ONE job:
    be the Wi-Fi transmitter that RX1 and RX2 lock onto
    to receive CSI.

    It creates a Wi-Fi Access Point and fires a small UDP
    packet to both receivers every 20 ms.  The receivers do
    not use the UDP payload — they only need a steady stream
    of 802.11 frames so the ESP-IDF CSI engine has something
    to measure.

  NETWORK
    Mode    : Access Point
    SSID    : CSI_TX_NETWORK
    Password: csi_tx_12345678
    Channel : 6  (fixed)
    IP      : 192.168.4.1  (AP default)
    BSSID   : printed to serial on boot

  DESTINATIONS
    RX1 : 192.168.4.2 : 3333
    RX2 : 192.168.4.3 : 3333

  UDP PACKET FORMAT
    "CSI_TX,seq=<n>,ts_ms=<t>"
    Payload is ASCII text — content is ignored by receivers.

  UDP INTERVAL
    20 ms  →  ~50 packets per second per receiver

  SERIAL
    115200 baud
    Prints BSSID, MAC, IP, channel on boot then
    a counter line every 5 seconds.

  BOARD
    ESP32-WROOM-32 / ESP32-WROOM-32D / ESP32-WROOM-32E
    (NOT an S3 — this is the dedicated TX board)

  IMPORTANT
    Do NOT run any CSI receive code on this board.
    Do NOT connect it to another AP.
    Keep it powered and running whenever you collect data
    or run live inference.
  ============================================================
*/

#include <WiFi.h>
#include <WiFiUdp.h>


/* ============================================================
   ACCESS POINT CONFIGURATION
   ============================================================ */

const char*    AP_SSID    = "CSI_TX_NETWORK";
const char*    AP_PASS    = "csi_tx_12345678";
const uint8_t  AP_CHANNEL = 6;
const uint8_t  AP_MAX_CONN = 4;    /* RX1 + RX2 + spare */


/* ============================================================
   UDP DESTINATIONS
   ============================================================
   Both receivers listen on port 3333.
   RX1 gets static IP 192.168.4.2
   RX2 gets static IP 192.168.4.3
   ============================================================ */

const uint16_t UDP_PORT = 3333;

const IPAddress RX1_IP(192, 168, 4, 2);
const IPAddress RX2_IP(192, 168, 4, 3);


/* ============================================================
   TRANSMIT INTERVAL
   ============================================================ */

const uint32_t TX_INTERVAL_MS = 20;    /* 50 Hz */


/* ============================================================
   STATE
   ============================================================ */

WiFiUDP udp;

uint32_t      sequenceNumber = 0;
unsigned long lastTxTime     = 0;
unsigned long lastDebugTime  = 0;

uint32_t totalPacketsSent = 0;


/* ============================================================
   PRINT MAC ADDRESS
   ============================================================ */

void printMAC(const uint8_t* mac)
{
    char buf[18];
    snprintf(
        buf, sizeof(buf),
        "%02X:%02X:%02X:%02X:%02X:%02X",
        mac[0], mac[1], mac[2],
        mac[3], mac[4], mac[5]
    );
    Serial.print(buf);
}


/* ============================================================
   SEND UDP PACKET TO ONE DESTINATION
   ============================================================ */

void sendPacket(const IPAddress& dest, uint32_t seq, uint32_t ts)
{
    char payload[64];
    int  len = snprintf(
        payload, sizeof(payload),
        "CSI_TX,seq=%lu,ts_ms=%lu",
        (unsigned long)seq,
        (unsigned long)ts
    );
    if (len <= 0 || len >= (int)sizeof(payload)) return;

    udp.beginPacket(dest, UDP_PORT);
    udp.write((const uint8_t*)payload, (size_t)len);
    udp.endPacket();
}


/* ============================================================
   SETUP
   ============================================================ */

void setup()
{
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("====================================");
    Serial.println("TX WROOM  —  CSI TRANSMITTER");
    Serial.println("====================================");

    /* ── Access Point ───────────────────────────────── */
    WiFi.mode(WIFI_AP);
    WiFi.setSleep(false);

    bool ok = WiFi.softAP(
        AP_SSID,
        AP_PASS,
        AP_CHANNEL,
        0,            /* not hidden */
        AP_MAX_CONN
    );

    if (ok) {
        Serial.println("AP_STARTED");
    } else {
        Serial.println("AP_FAILED — check SSID / password length.");
    }

    /* ── Print network info ─────────────────────────── */
    Serial.print("AP_SSID    : "); Serial.println(AP_SSID);
    Serial.print("AP_CHANNEL : "); Serial.println(AP_CHANNEL);
    Serial.print("AP_IP      : "); Serial.println(WiFi.softAPIP());

    /* BSSID = AP MAC address */
    uint8_t bssid[6];
    WiFi.softAPmacAddress(bssid);
    Serial.print("AP_BSSID   : ");
    printMAC(bssid);
    Serial.println();

    /* Station MAC (same chip, different address) */
    uint8_t staMac[6];
    WiFi.macAddress(staMac);
    Serial.print("STA_MAC    : ");
    printMAC(staMac);
    Serial.println();

    Serial.print("RX1_DEST   : ");
    Serial.print(RX1_IP);
    Serial.print(" : ");
    Serial.println(UDP_PORT);

    Serial.print("RX2_DEST   : ");
    Serial.print(RX2_IP);
    Serial.print(" : ");
    Serial.println(UDP_PORT);

    Serial.print("TX_INTERVAL: ");
    Serial.print(TX_INTERVAL_MS);
    Serial.println(" ms");

    /* ── UDP socket ─────────────────────────────────── */
    if (udp.begin(UDP_PORT)) {
        Serial.print("UDP_READY  : ");
        Serial.println(UDP_PORT);
    } else {
        Serial.println("UDP_FAILED");
    }

    Serial.println("====================================");
    Serial.println("TX_READY");
    Serial.println();

    lastTxTime    = millis();
    lastDebugTime = millis();
}


/* ============================================================
   LOOP
   ============================================================ */

void loop()
{
    unsigned long now = millis();

    /* ── Transmit at TX_INTERVAL_MS ─────────────────── */
    if (now - lastTxTime >= TX_INTERVAL_MS) {
        lastTxTime = now;

        uint32_t ts  = (uint32_t)now;
        uint32_t seq = sequenceNumber++;

        sendPacket(RX1_IP, seq, ts);
        sendPacket(RX2_IP, seq, ts);

        totalPacketsSent += 2;
    }

    /* ── Status print every 5 seconds ───────────────── */
    if (now - lastDebugTime >= 5000UL) {
        lastDebugTime = now;

        Serial.print("TX_STATUS  seq=");
        Serial.print(sequenceNumber);
        Serial.print("  total_sent=");
        Serial.print(totalPacketsSent);
        Serial.print("  clients=");
        Serial.println(WiFi.softAPgetStationNum());
    }
}
