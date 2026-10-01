/*
  ============================================================
  RX2_S3_HOTSPOT
  ESP32-S3 — CSI Receiver #2 + Control Interface
  ============================================================

  JOBS
    1. CSI RECEIVER  — captures Wi-Fi CSI from TX WROOM,
       prints to COM11 → collector.py → rx2.csv

    2. CONTROL HOTSPOT  — creates its own AP (CSI_CONTROL)
       and serves an HTML page for:
         • Multi-person dataset collection (0-4 persons each
           with activity + X/Y location)
         • Training trigger (runs full pipeline on laptop)
         • Testing trigger (live multi-task inference on RX2)

    3. INFERENCE ENGINE  — runs TEDNet float32 with three heads:
         • Activity  — multi-label (7 classes, multi-hot)
         • Count     — how many persons (0-4)
         • Location  — (X,Y) per predicted person

  NETWORKS
    TX_NETWORK   SSID : CSI_TX_NETWORK
                 PASS : csi_tx_12345678
                 RX2 IP (station): 192.168.4.3

    CONTROL AP   SSID : CSI_CONTROL
                 PASS : csi_control_123
                 RX2 IP (AP):      192.168.4.1

  SERIAL   RX2 → COM11   115200 baud

  MODEL BINARY FORMAT  (CSI2 version 3 — float32 only)
    Magic  "CSI2"    4 bytes
    Version 3        uint32 LE
    Tensor count N   uint32 LE
    For each tensor:
      name_len       uint16 LE
      name           UTF-8 bytes
      dtype          uint8  (1 = float32)
      ndim           uint32 LE
      dims[i]        uint32 LE × ndim
      data_size      uint64 LE (bytes = elements × 4)
      data           raw float32 bytes

  UPLOAD PROTOCOL
    PC  → "MODEL_BEGIN,<size>,3\n"
    RX2 → "MODEL_READY"
    PC  → raw bytes in chunks
    PC  → "MODEL_END\n"
    RX2 → "MODEL_LOADED"

  ACTIVITIES  (index fixed, must match Python ACTIVITY_CLASSES)
    0 empty   1 falling  2 lying   3 running
    4 sitting 5 standing 6 walking

  BUGS FIXED vs UPLOADED VERSION
    1. Serial.setRxBufferSize(4096) before Serial.begin() —
       prevents serial RX buffer overflow during 6-minute
       upload at 115200 baud, which corrupted the binary and
       caused parseModelContainer() to fail (modelLoaded=false).

    2. loop() now has a dedicated branch for modelWaitingForEnd —
       previously handleModelUpload() was not called when
       modelWaitingForEnd=true and Serial.available()==0.

    3. STATE 0 waits for Serial.available()>=10 before calling
       readStringUntil(), and uses setTimeout(200) to prevent
       blocking the loop for up to 1 second.

    4. Firmware sends "MODEL_LOADED" (not "MODEL_LOADED_PSRAM_ONLY")
       to exactly match what weightloader.py waits for.
  ============================================================
*/

#include <WiFi.h>
#include <WebServer.h>
#include <WiFiUdp.h>
#include <LittleFS.h>
#include <math.h>

struct ModelTensor;
bool parseModelContainer();

extern "C" {
    #include "esp_wifi.h"
    #include "esp_psram.h"
}


/* ============================================================
   NETWORK CONSTANTS
   ============================================================ */

const char*    TX_SSID = "CSI_TX_NETWORK";
const char*    TX_PASS = "csi_tx_12345678";
const uint16_t TX_PORT = 3333;

const uint8_t  WROOM_MAC[6] = {0x84, 0x0D, 0x8E, 0xE8, 0x31, 0x29};

const char*    AP_SSID = "CSI_CONTROL";
const char*    AP_PASS = "csi_control_123";

const uint16_t RX1_TEST_PORT = 4444;


/* ============================================================
   SERVER / UDP
   ============================================================ */

WebServer server(80);
WiFiUDP   udp;
WiFiUDP   rx1TestUDP;


/* ============================================================
   OPERATING MODE
   ============================================================ */

enum SystemMode { MODE_IDLE, MODE_TRAINING, MODE_TESTING };
volatile SystemMode currentMode = MODE_IDLE;

volatile bool collecting     = false;
volatile bool trainingActive = false;
volatile bool testingActive  = false;




String trainingStatus = "IDLE";
String testingStatus  = "NOT TESTING";

/* ── Server inference result ─────────────────────────── */
bool     serverTestActive          = false;
char     serverResultJSON[512]     = {0};
bool     serverResultReady         = false;
uint32_t serverResultTimestamp     = 0;

/* ── Deferred serial flags (Core 0 sets, Core 1 prints) ─ */
volatile bool pendingSessionStop   = false;
volatile bool pendingServerStart   = false;
volatile bool pendingServerStop    = false;

/* ============================================================
   CSI BUFFER
   ============================================================ */

#define MAX_CSI_BYTES 512
#define RX_FEATURES   192

volatile bool newCSI     = false;
volatile int  csiLength  = 0;
volatile int  csiRSSI    = 0;
volatile int  csiChannel = 0;

int8_t csiBuffer[MAX_CSI_BYTES];

volatile uint32_t csiAcceptedWroom = 0;
volatile uint32_t csiRejectedOther = 0;
uint32_t csiPacketCount = 0;
int      latestRSSI     = 0;


/* ============================================================
   RX2 AMPLITUDE  (for test frames)
   ============================================================ */

float    rx2Amplitude[RX_FEATURES];
uint32_t rx2FrameTimestamp = 0;
bool     rx2AmplitudeReady = false;


/* ============================================================
   RX1 CIRCULAR SYNC BUFFER
   ============================================================
   Stores the 3 most recent RX1 frames received via UDP.
   buildTestingFrame() picks the one closest in time to the
   current RX2 CSI frame.
   ============================================================ */

#define RX1_BUF_SIZE           3
#define MAX_SYNC_DIFFERENCE_MS 150

struct RX1Frame {
    float    amplitude[RX_FEATURES];
    uint32_t timestamp_ms;
    bool     valid;
};

RX1Frame rx1CircBuf[RX1_BUF_SIZE];
int      rx1BufHead = 0;

volatile uint32_t rx1TestPacketCount = 0;
volatile uint32_t rx1SyncRejectCount = 0;


/* ============================================================
   TEST WINDOW
   ============================================================ */

#define TEST_WINDOW_SIZE 30

float testingBuffer[TEST_WINDOW_SIZE][RX_FEATURES * 2];  /* 30 × 384 */
int   testingFrameCount = 0;


/* ============================================================
   LAST INFERENCE RESULTS
   ============================================================ */

#define N_ACTIVITIES    7
#define N_COUNT_CLASSES 5
#define MAX_PERSONS     4

const char* ACTIVITY_NAMES[N_ACTIVITIES] = {
    "empty", "falling", "lying", "running",
    "sitting", "standing", "walking"
};

bool  lastDetectedAct[N_ACTIVITIES] = {false};
int   lastPredictedCount            = 0;
float lastLocations[MAX_PERSONS][2] = {{0}};
float lastActConfidence             = 0.0f;


/* ============================================================
   MODEL STATE
   ============================================================ */

bool     modelLoaded          = false;
size_t   modelSizeBytes       = 0;
uint8_t* modelBuffer          = nullptr;
size_t   modelCapacityBytes   = 0;
size_t   modelBytesReceived   = 0;

bool     modelReceiving       = false;
bool     modelWaitingForEnd   = false;
uint32_t expectedModelBytes   = 0;
uint32_t receivedModelBytes   = 0;

const char* MODEL_FLASH_FILE = "/model.bin";
const char* MODEL_FLASH_TMP  = "/model.tmp";

#define MODEL_MAX_SIZE (8 * 1024 * 1024)


/* ============================================================
   INFERENCE CONSTANTS
   ============================================================ */

#define TED_TIME        30
#define TED_FEATURES    384
#define TED_DIM         128
#define TED_HEADS       8
#define TED_HEAD_DIM    16
#define TED_FF          512
#define TED_LAYERS      4
#define TED_HIDDEN      64
#define TED_MAX_TENSORS 320
#define TED_LN_EPS      1.0e-5f


/* ============================================================
   MODEL TENSOR
   ============================================================ */

struct ModelTensor {
    char           name[96];
    uint8_t        dtype;
    uint32_t       ndim;
    uint32_t       dims[4];
    uint64_t       bytes;
    uint32_t       elements;
    const uint8_t* data;
};

ModelTensor modelTensors[TED_MAX_TENSORS];
int         modelTensorCount = 0;

float modelFeatureMean[TED_FEATURES];
float modelFeatureStd[TED_FEATURES];
bool  modelHasNormalization = false;


/* ============================================================
   LITTLE-ENDIAN READERS
   ============================================================ */

static uint16_t readU16LE(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t readU32LE(const uint8_t* p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] <<  8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static uint64_t readU64LE(const uint8_t* p) {
    uint64_t lo = readU32LE(p);
    uint64_t hi = readU32LE(p + 4);
    return lo | (hi << 32);
}

static float readFloatLE(const uint8_t* p) {
    float v;
    memcpy(&v, p, 4);
    return v;
}


/* ============================================================
   TENSOR ACCESS
   ============================================================ */

static ModelTensor* findTensor(const char* name) {
    for (int i = 0; i < modelTensorCount; i++) {
        if (strcmp(modelTensors[i].name, name) == 0)
            return &modelTensors[i];
    }
    return nullptr;
}

static float tensorValue(const ModelTensor* t, uint32_t index) {
    if (!t || index >= t->elements) return 0.0f;
    return readFloatLE(t->data + index * 4);
}


/* ============================================================
   MATH HELPERS
   ============================================================ */

static float tedGELU(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}


/* ============================================================
   LAYER NORM
   ============================================================ */

static void tedLayerNorm(
    float* x, int rows, int cols,
    const ModelTensor* gamma, const ModelTensor* beta)
{
    for (int r = 0; r < rows; r++) {
        float mean = 0.0f;
        for (int c = 0; c < cols; c++) mean += x[r * cols + c];
        mean /= (float)cols;

        float var = 0.0f;
        for (int c = 0; c < cols; c++) {
            float d = x[r * cols + c] - mean;
            var += d * d;
        }
        var /= (float)cols;
        float inv = 1.0f / sqrtf(var + TED_LN_EPS);

        for (int c = 0; c < cols; c++) {
            float v = (x[r * cols + c] - mean) * inv;
            if (gamma) v *= tensorValue(gamma, c);
            if (beta)  v += tensorValue(beta,  c);
            x[r * cols + c] = v;
        }
    }
}


/* ============================================================
   LINEAR LAYER
   ============================================================ */

static void tedLinear(
    const float* in, float* out,
    int rows, int inDim, int outDim,
    const ModelTensor* weight, const ModelTensor* bias)
{
    for (int r = 0; r < rows; r++) {
        for (int o = 0; o < outDim; o++) {
            float sum = bias ? tensorValue(bias, o) : 0.0f;
            for (int i = 0; i < inDim; i++)
                sum += in[r * inDim + i] * tensorValue(weight, o * inDim + i);
            out[r * outDim + o] = sum;
        }
    }
}


/* ============================================================
   BATCH NORM 1D  (eval mode)
   ============================================================ */

static void tedBatchNorm1d(
    float* x, int channels, int time,
    const ModelTensor* gamma, const ModelTensor* beta,
    const ModelTensor* runningMean, const ModelTensor* runningVar)
{
    for (int c = 0; c < channels; c++) {
        float mean = tensorValue(runningMean, c);
        float var  = tensorValue(runningVar,  c);
        float inv  = 1.0f / sqrtf(var + TED_LN_EPS);
        float g    = tensorValue(gamma, c);
        float b    = tensorValue(beta,  c);
        for (int t = 0; t < time; t++) {
            int idx = c * time + t;
            x[idx] = (x[idx] - mean) * inv * g + b;
        }
    }
}


/* ============================================================
   CNN  — input [30,384] → output [30,128]
   ============================================================ */

bool tedCNN(const float* input, float* output) {
    ModelTensor* w1  = findTensor("cnn.0.weight");
    ModelTensor* b1  = findTensor("cnn.0.bias");
    ModelTensor* g1  = findTensor("cnn.1.weight");
    ModelTensor* be1 = findTensor("cnn.1.bias");
    ModelTensor* rm1 = findTensor("cnn.1.running_mean");
    ModelTensor* rv1 = findTensor("cnn.1.running_var");

    ModelTensor* w2  = findTensor("cnn.3.weight");
    ModelTensor* b2  = findTensor("cnn.3.bias");
    ModelTensor* g2  = findTensor("cnn.4.weight");
    ModelTensor* be2 = findTensor("cnn.4.bias");
    ModelTensor* rm2 = findTensor("cnn.4.running_mean");
    ModelTensor* rv2 = findTensor("cnn.4.running_var");

    if (!w1||!b1||!g1||!be1||!rm1||!rv1||
        !w2||!b2||!g2||!be2||!rm2||!rv2) {
        Serial.println("CNN_TENSOR_MISSING");
        return false;
    }

    float* a = (float*)ps_malloc(TED_TIME * TED_DIM * sizeof(float));
    float* b = (float*)ps_malloc(TED_TIME * TED_DIM * sizeof(float));
    if (!a || !b) { if (a) free(a); if (b) free(b); return false; }

    /* Conv1d #1: weight [128, 384, 3] */
    for (int oc = 0; oc < TED_DIM; oc++) {
        for (int t = 0; t < TED_TIME; t++) {
            float sum = tensorValue(b1, oc);
            for (int ic = 0; ic < TED_FEATURES; ic++) {
                for (int k = -1; k <= 1; k++) {
                    int tt = t + k;
                    if (tt < 0 || tt >= TED_TIME) continue;
                    uint32_t wi = ((uint32_t)oc * TED_FEATURES + ic) * 3 + (k + 1);
                    sum += input[tt * TED_FEATURES + ic] * tensorValue(w1, wi);
                }
            }
            a[oc * TED_TIME + t] = sum;
        }
    }

    tedBatchNorm1d(a, TED_DIM, TED_TIME, g1, be1, rm1, rv1);
    for (int i = 0; i < TED_TIME * TED_DIM; i++) a[i] = tedGELU(a[i]);

    /* Conv1d #2: weight [128, 128, 3] */
    for (int oc = 0; oc < TED_DIM; oc++) {
        for (int t = 0; t < TED_TIME; t++) {
            float sum = tensorValue(b2, oc);
            for (int ic = 0; ic < TED_DIM; ic++) {
                for (int k = -1; k <= 1; k++) {
                    int tt = t + k;
                    if (tt < 0 || tt >= TED_TIME) continue;
                    uint32_t wi = ((uint32_t)oc * TED_DIM + ic) * 3 + (k + 1);
                    sum += a[ic * TED_TIME + tt] * tensorValue(w2, wi);
                }
            }
            b[oc * TED_TIME + t] = sum;
        }
    }

    tedBatchNorm1d(b, TED_DIM, TED_TIME, g2, be2, rm2, rv2);

    /* Transpose [128,30] → [30,128] + GELU */
    for (int t = 0; t < TED_TIME; t++)
        for (int c = 0; c < TED_DIM; c++)
            output[t * TED_DIM + c] = tedGELU(b[c * TED_TIME + t]);

    free(a);
    free(b);
    return true;
}


/* ============================================================
   TRANSFORMER ENCODER LAYER
   ============================================================ */

bool tedTransformerLayer(float* x, int layer) {
    char name[128];

    snprintf(name, sizeof(name),
             "transformer.layers.%d.self_attn.in_proj_weight", layer);
    ModelTensor* inW = findTensor(name);
    snprintf(name, sizeof(name),
             "transformer.layers.%d.self_attn.in_proj_bias", layer);
    ModelTensor* inB = findTensor(name);
    snprintf(name, sizeof(name),
             "transformer.layers.%d.self_attn.out_proj.weight", layer);
    ModelTensor* outW = findTensor(name);
    snprintf(name, sizeof(name),
             "transformer.layers.%d.self_attn.out_proj.bias", layer);
    ModelTensor* outB = findTensor(name);

    snprintf(name, sizeof(name),
             "transformer.layers.%d.linear1.weight", layer);
    ModelTensor* ffW1 = findTensor(name);
    snprintf(name, sizeof(name),
             "transformer.layers.%d.linear1.bias", layer);
    ModelTensor* ffB1 = findTensor(name);
    snprintf(name, sizeof(name),
             "transformer.layers.%d.linear2.weight", layer);
    ModelTensor* ffW2 = findTensor(name);
    snprintf(name, sizeof(name),
             "transformer.layers.%d.linear2.bias", layer);
    ModelTensor* ffB2 = findTensor(name);

    snprintf(name, sizeof(name),
             "transformer.layers.%d.norm1.weight", layer);
    ModelTensor* n1g = findTensor(name);
    snprintf(name, sizeof(name),
             "transformer.layers.%d.norm1.bias", layer);
    ModelTensor* n1b = findTensor(name);
    snprintf(name, sizeof(name),
             "transformer.layers.%d.norm2.weight", layer);
    ModelTensor* n2g = findTensor(name);
    snprintf(name, sizeof(name),
             "transformer.layers.%d.norm2.bias", layer);
    ModelTensor* n2b = findTensor(name);

    if (!inW||!inB||!outW||!outB||!ffW1||!ffB1||
        !ffW2||!ffB2||!n1g||!n1b||!n2g||!n2b) {
        Serial.print("TRANSFORMER_TENSOR_MISSING_LAYER:");
        Serial.println(layer);
        return false;
    }

    float* qkv     = (float*)ps_malloc(TED_TIME * TED_DIM * 3 * sizeof(float));
    float* attnOut = (float*)ps_malloc(TED_TIME * TED_DIM     * sizeof(float));
    float* ffn     = (float*)ps_malloc(TED_TIME * TED_FF      * sizeof(float));
    float* proj    = (float*)ps_malloc(TED_TIME * TED_DIM     * sizeof(float));

    if (!qkv||!attnOut||!ffn||!proj) {
        if (qkv)     free(qkv);
        if (attnOut) free(attnOut);
        if (ffn)     free(ffn);
        if (proj)    free(proj);
        Serial.println("TRANSFORMER_WORKSPACE_FAILED");
        return false;
    }

    /* in_proj → Q K V stacked */
    tedLinear(x, qkv, TED_TIME, TED_DIM, TED_DIM * 3, inW, inB);

    /* Multi-head attention */
    const float scale = 0.25f;   /* 1/sqrt(16) */

    for (int t = 0; t < TED_TIME; t++) {
        for (int h = 0; h < TED_HEADS; h++) {
            float scores[TED_TIME];
            float maxScore = -1e30f;

            for (int s = 0; s < TED_TIME; s++) {
                float dot = 0.0f;
                for (int d = 0; d < TED_HEAD_DIM; d++) {
                    int qi = t * (TED_DIM * 3) + h * TED_HEAD_DIM + d;
                    int ki = s * (TED_DIM * 3) + TED_DIM + h * TED_HEAD_DIM + d;
                    dot += qkv[qi] * qkv[ki];
                }
                scores[s] = dot * scale;
                if (scores[s] > maxScore) maxScore = scores[s];
            }

            float denom = 0.0f;
            float probs[TED_TIME];
            for (int s = 0; s < TED_TIME; s++) {
                probs[s] = expf(scores[s] - maxScore);
                denom   += probs[s];
            }
            if (denom < 1e-20f) denom = 1.0f;

            for (int d = 0; d < TED_HEAD_DIM; d++) {
                float v = 0.0f;
                for (int s = 0; s < TED_TIME; s++) {
                    int vi = s * (TED_DIM * 3) + 2 * TED_DIM + h * TED_HEAD_DIM + d;
                    v += (probs[s] / denom) * qkv[vi];
                }
                attnOut[t * TED_DIM + h * TED_HEAD_DIM + d] = v;
            }
        }
    }

    /* Output projection → residual → norm1 */
    tedLinear(attnOut, proj, TED_TIME, TED_DIM, TED_DIM, outW, outB);
    for (int i = 0; i < TED_TIME * TED_DIM; i++) x[i] += proj[i];
    tedLayerNorm(x, TED_TIME, TED_DIM, n1g, n1b);

    /* FFN → residual → norm2 */
    tedLinear(x,   ffn,  TED_TIME, TED_DIM, TED_FF,  ffW1, ffB1);
    for (int i = 0; i < TED_TIME * TED_FF; i++) ffn[i] = tedGELU(ffn[i]);
    tedLinear(ffn, proj, TED_TIME, TED_FF,  TED_DIM, ffW2, ffB2);
    for (int i = 0; i < TED_TIME * TED_DIM; i++) x[i] += proj[i];
    tedLayerNorm(x, TED_TIME, TED_DIM, n2g, n2b);

    free(qkv);
    free(attnOut);
    free(ffn);
    free(proj);
    return true;
}


/* ============================================================
   THREE-HEAD INFERENCE
   ============================================================ */

bool runTEDNetInference(
    float  input[TED_TIME][TED_FEATURES],
    bool   detected_act[N_ACTIVITIES],
    int&   predicted_count,
    float  locations[MAX_PERSONS][2],
    float& act_confidence)
{
    if (!modelLoaded) return false;

    /* 1. Allocate workspace */
    float* x       = (float*)ps_malloc(TED_TIME * TED_FEATURES * sizeof(float));
    float* cnn_out = (float*)ps_malloc(TED_TIME * TED_DIM      * sizeof(float));

    if (!x || !cnn_out) {
        if (x)       free(x);
        if (cnn_out) free(cnn_out);
        Serial.println("INFERENCE_ALLOC_FAILED");
        return false;
    }

    /* 2. Normalize input */
    for (int t = 0; t < TED_TIME; t++) {
        for (int f = 0; f < TED_FEATURES; f++) {
            float v = input[t][f];
            if (modelHasNormalization)
                v = (v - modelFeatureMean[f]) / modelFeatureStd[f];
            x[t * TED_FEATURES + f] = v;
        }
    }

    /* 3. CNN */
    if (!tedCNN(x, cnn_out)) {
        free(x); free(cnn_out);
        return false;
    }
    free(x);
    x = cnn_out;   /* now [TED_TIME, TED_DIM] */

    /* 4. Positional embedding */
    ModelTensor* pos = findTensor("pos_embed");
    if (!pos || pos->elements < (uint32_t)(TED_TIME * TED_DIM)) {
        free(x);
        Serial.println("INFERENCE_ERROR:POS_EMBED");
        return false;
    }
    for (int t = 0; t < TED_TIME; t++)
        for (int d = 0; d < TED_DIM; d++)
            x[t * TED_DIM + d] += tensorValue(pos, t * TED_DIM + d);

    /* 5. Transformer */
    for (int layer = 0; layer < TED_LAYERS; layer++) {
        if (!tedTransformerLayer(x, layer)) {
            free(x);
            return false;
        }
    }

    /* 6. Temporal mean pool */
    float pooled[TED_DIM];
    for (int d = 0; d < TED_DIM; d++) {
        float s = 0.0f;
        for (int t = 0; t < TED_TIME; t++) s += x[t * TED_DIM + d];
        pooled[d] = s / (float)TED_TIME;
    }
    free(x);

    /* 7. Pool LayerNorm */
    ModelTensor* pnW = findTensor("pool_norm.weight");
    ModelTensor* pnB = findTensor("pool_norm.bias");
    if (!pnW || !pnB) {
        Serial.println("INFERENCE_ERROR:POOL_NORM");
        return false;
    }
    tedLayerNorm(pooled, 1, TED_DIM, pnW, pnB);

    float* hidden = (float*)ps_malloc(TED_HIDDEN * sizeof(float));
    if (!hidden) {
        Serial.println("INFERENCE_HEAD_ALLOC_FAILED");
        return false;
    }

    /* 8. Activity head */
    ModelTensor* aw0 = findTensor("head_activity.0.weight");
    ModelTensor* ab0 = findTensor("head_activity.0.bias");
    ModelTensor* aw3 = findTensor("head_activity.3.weight");
    ModelTensor* ab3 = findTensor("head_activity.3.bias");

    if (!aw0||!ab0||!aw3||!ab3) {
        free(hidden);
        Serial.println("INFERENCE_ERROR:ACTIVITY_HEAD");
        return false;
    }

    float logits_act[N_ACTIVITIES];
    tedLinear(pooled, hidden,     1, TED_DIM,   TED_HIDDEN,  aw0, ab0);
    for (int i = 0; i < TED_HIDDEN; i++) hidden[i] = tedGELU(hidden[i]);
    tedLinear(hidden, logits_act, 1, TED_HIDDEN, N_ACTIVITIES, aw3, ab3);

    act_confidence = 0.0f;
    for (int i = 0; i < N_ACTIVITIES; i++) {
        float sig = 1.0f / (1.0f + expf(-logits_act[i]));
        detected_act[i] = (sig > 0.5f);
        if (sig > act_confidence) act_confidence = sig;
    }

    /* 9. Count head */
    ModelTensor* cw0 = findTensor("head_count.0.weight");
    ModelTensor* cb0 = findTensor("head_count.0.bias");
    ModelTensor* cw3 = findTensor("head_count.3.weight");
    ModelTensor* cb3 = findTensor("head_count.3.bias");

    if (!cw0||!cb0||!cw3||!cb3) {
        free(hidden);
        Serial.println("INFERENCE_ERROR:COUNT_HEAD");
        return false;
    }

    float logits_cnt[N_COUNT_CLASSES];
    tedLinear(pooled, hidden,     1, TED_DIM,   TED_HIDDEN,     cw0, cb0);
    for (int i = 0; i < TED_HIDDEN; i++) hidden[i] = tedGELU(hidden[i]);
    tedLinear(hidden, logits_cnt, 1, TED_HIDDEN, N_COUNT_CLASSES, cw3, cb3);

    predicted_count = 0;
    float max_cnt   = logits_cnt[0];
    for (int i = 1; i < N_COUNT_CLASSES; i++) {
        if (logits_cnt[i] > max_cnt) {
            max_cnt = logits_cnt[i];
            predicted_count = i;
        }
    }

    /* 10. Location head */
    ModelTensor* lw0 = findTensor("head_location.0.weight");
    ModelTensor* lb0 = findTensor("head_location.0.bias");
    ModelTensor* lw3 = findTensor("head_location.3.weight");
    ModelTensor* lb3 = findTensor("head_location.3.bias");

    if (!lw0||!lb0||!lw3||!lb3) {
        free(hidden);
        Serial.println("INFERENCE_ERROR:LOCATION_HEAD");
        return false;
    }

    float logits_loc[MAX_PERSONS * 2];
    tedLinear(pooled, hidden,     1, TED_DIM,   TED_HIDDEN,      lw0, lb0);
    for (int i = 0; i < TED_HIDDEN; i++) hidden[i] = tedGELU(hidden[i]);
    tedLinear(hidden, logits_loc, 1, TED_HIDDEN, MAX_PERSONS * 2, lw3, lb3);

    for (int i = 0; i < MAX_PERSONS; i++) {
        locations[i][0] = logits_loc[i * 2];
        locations[i][1] = logits_loc[i * 2 + 1];
    }

    free(hidden);
    return true;
}


/* ============================================================
   BUILD + RUN TEST WINDOW
   ============================================================ */

void inferTestingWindow() {
    if (!testingActive)                       return;
    if (testingFrameCount < TEST_WINDOW_SIZE) return;

    bool  detected_act[N_ACTIVITIES];
    int   predicted_count = 0;
    float locations[MAX_PERSONS][2];
    float act_confidence  = 0.0f;

    bool ok = runTEDNetInference(
        testingBuffer,
        detected_act, predicted_count, locations, act_confidence
    );

    if (!ok) {
        testingStatus     = "INFERENCE FAILED";
        testingFrameCount = 0;
        return;
    }

    /* Store for /teststatus */
    lastPredictedCount = predicted_count;
    lastActConfidence  = act_confidence;
    for (int i = 0; i < N_ACTIVITIES; i++) lastDetectedAct[i] = detected_act[i];
    for (int i = 0; i < MAX_PERSONS; i++) {
        lastLocations[i][0] = locations[i][0];
        lastLocations[i][1] = locations[i][1];
    }

    /* Build status string */
    testingStatus = "COUNT:" + String(predicted_count) + " ACT:";
    for (int i = 0; i < N_ACTIVITIES; i++) {
        if (detected_act[i]) testingStatus += String(ACTIVITY_NAMES[i]) + " ";
    }
    testingStatus.trim();

    /* Serial */
    Serial.println("==== INFERENCE RESULT ====");
    Serial.print("PERSONS:");  Serial.println(predicted_count);
    Serial.print("ACTIVITIES:");
    for (int i = 0; i < N_ACTIVITIES; i++)
        if (detected_act[i]) { Serial.print(ACTIVITY_NAMES[i]); Serial.print(" "); }
    Serial.println();
    for (int i = 0; i < predicted_count && i < MAX_PERSONS; i++) {
        Serial.print("P"); Serial.print(i);
        Serial.print(" X="); Serial.print(locations[i][0], 2);
        Serial.print(" Y="); Serial.println(locations[i][1], 2);
    }
    Serial.print("ACT_CONF:"); Serial.println(act_confidence * 100.0f, 1);
    Serial.println("==========================");

    /* Reset for next window (tumbling window) */
    testingFrameCount = 0;
}


/* ============================================================
   BUILD TEST FRAME
   ============================================================ */

void buildTestingFrame() {
    if (!testingActive)     return;
    if (!rx2AmplitudeReady) return;

    /* Find closest RX1 frame to this RX2 timestamp */
    uint32_t best_diff = 0xFFFFFFFF;
    int      best_idx  = -1;

    for (int i = 0; i < RX1_BUF_SIZE; i++) {
        if (!rx1CircBuf[i].valid) continue;
        uint32_t t1   = rx1CircBuf[i].timestamp_ms;
        uint32_t t2   = rx2FrameTimestamp;
        uint32_t diff = (t1 > t2) ? (t1 - t2) : (t2 - t1);
        if (diff < best_diff) { best_diff = diff; best_idx = i; }
    }

    if (best_idx == -1 || best_diff > MAX_SYNC_DIFFERENCE_MS) {
        rx1SyncRejectCount++;
        rx2AmplitudeReady = false;
        return;
    }

    /* [RX1 192] ++ [RX2 192] = 384 features */
    for (int i = 0; i < RX_FEATURES; i++) {
        testingBuffer[testingFrameCount][i]              = rx1CircBuf[best_idx].amplitude[i];
        testingBuffer[testingFrameCount][RX_FEATURES + i] = rx2Amplitude[i];
    }

    testingFrameCount++;
    rx2AmplitudeReady = false;

    if (testingFrameCount >= TEST_WINDOW_SIZE) {
        Serial.println("TEST_WINDOW_READY");
        inferTestingWindow();
    }
}


/* ============================================================
   PARSE MODEL BINARY  (CSI2 version 3 — float32 only)
   ============================================================ */

bool parseModelContainer() {
    modelTensorCount = 0;

    if (modelSizeBytes < 12) {
        Serial.println("PARSER_ERROR:TOO_SMALL");
        return false;
    }

    if (modelBuffer[0]!='C' || modelBuffer[1]!='S' ||
        modelBuffer[2]!='I' || modelBuffer[3]!='2') {
        Serial.println("PARSER_ERROR:BAD_MAGIC");
        return false;
    }

    uint32_t version = readU32LE(modelBuffer + 4);
    uint32_t count   = readU32LE(modelBuffer + 8);

    Serial.print("PARSER:VERSION="); Serial.println(version);
    Serial.print("PARSER:TENSORS="); Serial.println(count);

    if (version != 3) {
        Serial.print("PARSER_ERROR:UNSUPPORTED_VERSION:");
        Serial.println(version);
        return false;
    }

    if (count == 0 || count > (uint32_t)TED_MAX_TENSORS) {
        Serial.println("PARSER_ERROR:TENSOR_COUNT_INVALID");
        return false;
    }

    size_t off = 12;

    for (uint32_t n = 0; n < count; n++) {
        if (off + 2 > modelSizeBytes) {
            Serial.println("PARSER_ERROR:NAME_LEN_OOB"); return false;
        }

        uint16_t nameLen = readU16LE(modelBuffer + off); off += 2;
        if (nameLen == 0 || nameLen >= 95) {
            Serial.println("PARSER_ERROR:NAME_LEN_BAD"); return false;
        }
        if (off + nameLen > modelSizeBytes) {
            Serial.println("PARSER_ERROR:NAME_OOB"); return false;
        }

        ModelTensor& t = modelTensors[modelTensorCount];
        memset(&t, 0, sizeof(t));
        memcpy(t.name, modelBuffer + off, nameLen);
        t.name[nameLen] = '\0';
        off += nameLen;

        if (off + 1 > modelSizeBytes) {
            Serial.println("PARSER_ERROR:DTYPE_OOB"); return false;
        }
        t.dtype = modelBuffer[off++];
        if (t.dtype != 1) {
            Serial.print("PARSER_ERROR:UNSUPPORTED_DTYPE:"); Serial.println(t.name);
            return false;
        }

        if (off + 4 > modelSizeBytes) {
            Serial.println("PARSER_ERROR:NDIM_OOB"); return false;
        }
        t.ndim = readU32LE(modelBuffer + off); off += 4;
        if (t.ndim > 4) {
            Serial.print("PARSER_ERROR:NDIM_TOO_LARGE:"); Serial.println(t.name);
            return false;
        }

        t.elements = 1;
        for (uint32_t d = 0; d < t.ndim; d++) {
            if (off + 4 > modelSizeBytes) {
                Serial.println("PARSER_ERROR:DIMS_OOB"); return false;
            }
            t.dims[d] = readU32LE(modelBuffer + off); off += 4;
            t.elements *= t.dims[d];
        }
        if (t.ndim == 0) t.elements = 1;

        if (off + 8 > modelSizeBytes) {
            Serial.println("PARSER_ERROR:DATASIZE_OOB"); return false;
        }
        t.bytes = readU64LE(modelBuffer + off); off += 8;

        if (t.bytes > (uint64_t)(modelSizeBytes - off)) {
            Serial.print("PARSER_ERROR:DATA_OOB:"); Serial.println(t.name);
            return false;
        }
        t.data = modelBuffer + off;
        off   += (size_t)t.bytes;

        modelTensorCount++;
    }

    /* Extract normalization tensors */
    ModelTensor* meanT = findTensor("feature_mean");
    ModelTensor* stdT  = findTensor("feature_std");

    if (meanT && stdT &&
        meanT->elements >= (uint32_t)TED_FEATURES &&
        stdT->elements  >= (uint32_t)TED_FEATURES) {
        for (int i = 0; i < TED_FEATURES; i++) {
            modelFeatureMean[i] = tensorValue(meanT, i);
            modelFeatureStd[i]  = tensorValue(stdT,  i);
            if (fabsf(modelFeatureStd[i]) < 1e-8f) modelFeatureStd[i] = 1.0f;
        }
        modelHasNormalization = true;
        Serial.println("NORMALIZATION:LOADED");
    } else {
        for (int i = 0; i < TED_FEATURES; i++) {
            modelFeatureMean[i] = 0.0f;
            modelFeatureStd[i]  = 1.0f;
        }
        modelHasNormalization = false;
        Serial.println("NORMALIZATION:MISSING — inference will use raw amplitudes");
    }

    modelLoaded = true;

    Serial.println("TEDNET_MODEL_PARSE_OK");
    Serial.print("TEDNET_TENSORS:"); Serial.println(modelTensorCount);
    Serial.print("TEDNET_NORM:");    Serial.println(modelHasNormalization ? "YES" : "NO");
    return true;
}


/* ============================================================
   HTML PAGE
   ============================================================ */

const char PAGE[] PROGMEM = R"HTML(
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CSI HAR Control</title>
<style>
body{font-family:Arial,sans-serif;max-width:720px;margin:20px auto;padding:0 15px}
h2{text-align:center}
h3{margin-top:25px;border-bottom:1px solid #aaa;padding-bottom:5px}
label{display:block;margin-top:10px;font-weight:bold}
input,select,button{width:100%;box-sizing:border-box;font-size:17px;padding:9px;margin-top:4px}
button{margin-top:12px;font-weight:bold;cursor:pointer}
.person-block{border:1px solid #ccc;padding:10px;margin-top:10px;border-radius:6px;background:#f9f9f9}
.person-block h4{margin:0 0 8px 0;color:#444}
#status,#trainingStatus,#testingResult{
  padding:12px;margin-top:12px;border:1px solid #999;
  text-align:center;font-weight:bold;background:#f5f5f5}
#testingResult{text-align:left;font-family:monospace;white-space:pre-wrap}
#modelStatus{padding:10px;margin-top:8px;border:1px solid #999;
  background:#f5f5f5;font-family:monospace;font-size:13px;white-space:pre-wrap}
.warn{font-size:13px;padding:8px;border:1px solid #aaa;margin-top:8px;background:#fffbe6}
</style>
</head>
<body>

<h2>Wi-Fi CSI HAR Control</h2>

<h3>Model Status</h3>
<div id="modelStatus">Loading...</div>

<h3>Data Collection</h3>

<label>Number of Persons</label>
<select id="num_persons" onchange="updatePersonFields()">
  <option value="0">0 — empty room</option>
  <option value="1" selected>1 person</option>
  <option value="2">2 persons</option>
  <option value="3">3 persons</option>
  <option value="4">4 persons</option>
</select>

<div id="person_block_0" class="person-block">
  <h4>Person 0</h4>
  <label>Activity</label>
  <select id="p0_activity">
    <option value="standing">standing</option>
    <option value="sitting">sitting</option>
    <option value="lying">lying</option>
    <option value="walking">walking</option>
    <option value="running">running</option>
    <option value="falling">falling</option>
    <option value="empty">empty</option>
  </select>
  <label>X (m)</label><input id="p0_x" type="number" step="0.1" value="0">
  <label>Y (m)</label><input id="p0_y" type="number" step="0.1" value="0">
</div>

<div id="person_block_1" class="person-block" style="display:none">
  <h4>Person 1</h4>
  <label>Activity</label>
  <select id="p1_activity">
    <option value="standing">standing</option>
    <option value="sitting">sitting</option>
    <option value="lying">lying</option>
    <option value="walking">walking</option>
    <option value="running">running</option>
    <option value="falling">falling</option>
    <option value="empty">empty</option>
  </select>
  <label>X (m)</label><input id="p1_x" type="number" step="0.1" value="0">
  <label>Y (m)</label><input id="p1_y" type="number" step="0.1" value="0">
</div>

<div id="person_block_2" class="person-block" style="display:none">
  <h4>Person 2</h4>
  <label>Activity</label>
  <select id="p2_activity">
    <option value="standing">standing</option>
    <option value="sitting">sitting</option>
    <option value="lying">lying</option>
    <option value="walking">walking</option>
    <option value="running">running</option>
    <option value="falling">falling</option>
    <option value="empty">empty</option>
  </select>
  <label>X (m)</label><input id="p2_x" type="number" step="0.1" value="0">
  <label>Y (m)</label><input id="p2_y" type="number" step="0.1" value="0">
</div>

<div id="person_block_3" class="person-block" style="display:none">
  <h4>Person 3</h4>
  <label>Activity</label>
  <select id="p3_activity">
    <option value="standing">standing</option>
    <option value="sitting">sitting</option>
    <option value="lying">lying</option>
    <option value="walking">walking</option>
    <option value="running">running</option>
    <option value="falling">falling</option>
    <option value="empty">empty</option>
  </select>
  <label>X (m)</label><input id="p3_x" type="number" step="0.1" value="0">
  <label>Y (m)</label><input id="p3_y" type="number" step="0.1" value="0">
</div>

<label>Subject ID</label>
<input id="subject" type="text" value="S01">
<label>Note</label>
<input id="note" type="text" value="">

<button onclick="startSession()" style="background:#4CAF50;color:#fff">START SESSION</button>
<button onclick="stopSession()"  style="background:#f44336;color:#fff">STOP SESSION</button>
<div id="status">IDLE</div>

<h3>Training</h3>
<div class="warn">Start training only when data collection is stopped.</div>
<button onclick="startTraining()" style="background:#2196F3;color:#fff">TRAIN</button>
<div id="trainingStatus">IDLE</div>

<h3>Live Testing</h3>
<button onclick="startTesting()" style="background:#9C27B0;color:#fff">START TESTING</button>
<button onclick="stopTesting()"  style="background:#607D8B;color:#fff">STOP TESTING</button>
<div id="testingResult">Not testing.</div>

<!-- ════════════ SERVER INFERENCE ════════════ -->
<h3>Server Inference</h3>
<div class="warn">Runs PyTorch on the laptop CPU. Each result takes ~15 seconds.</div>
<button onclick="startServerTest()" style="background:#E65100;color:#fff">START TEST ON SERVER</button>
<button onclick="stopServerTest()"  style="background:#455A64;color:#fff">STOP TEST ON SERVER</button>
<div id="serverResult" style="padding:12px;margin-top:12px;border:1px solid #999;
  text-align:left;font-family:monospace;font-size:13px;
  white-space:pre-wrap;background:#f5f5f5">Not running.</div>

<script>
function updatePersonFields() {
  const n = parseInt(document.getElementById('num_persons').value);
  for (let i = 0; i < 4; i++) {
    const el = document.getElementById('person_block_' + i);
    if (el) el.style.display = (i < n) ? 'block' : 'none';
  }
}
updatePersonFields();

function v(id)          { return document.getElementById(id).value; }
function setText(id,txt){ document.getElementById(id).textContent = txt; }

async function get(url) {
  try { const r = await fetch(url); return await r.text(); }
  catch(e) { return 'ERROR: ' + e; }
}

async function startSession() {
  const n = parseInt(v('num_persons'));
  let qs = 'num_persons=' + n;
  for (let i = 0; i < n; i++) {
    qs += '&p'+i+'_activity=' + encodeURIComponent(v('p'+i+'_activity'));
    qs += '&p'+i+'_x='        + encodeURIComponent(v('p'+i+'_x'));
    qs += '&p'+i+'_y='        + encodeURIComponent(v('p'+i+'_y'));
  }
  qs += '&subject=' + encodeURIComponent(v('subject'));
  qs += '&note='    + encodeURIComponent(v('note'));
  setText('status', await get('/start?' + qs));
}

async function stopSession()   { setText('status',         await get('/stop')); }
async function startTraining() { setText('trainingStatus', await get('/train/start')); }
async function stopTesting()   { setText('testingResult',  await get('/test/stop')); stopTestPoll(); }

let testPollTimer = null;
async function startTesting() {
  const resp = await get('/test/start');
  setText('testingResult', resp === 'TESTING STARTED'
    ? 'Testing started — waiting for first result...'
    : 'ERROR: ' + resp);
  if (resp === 'TESTING STARTED') startTestPoll();
}

function startTestPoll() {
  if (testPollTimer) return;
  testPollTimer = setInterval(pollTestStatus, 600);
}
function stopTestPoll() {
  if (testPollTimer) { clearInterval(testPollTimer); testPollTimer = null; }
}



async function pollTestStatus() {
  try {
    const r   = await fetch('/teststatus');
    const obj = await r.json();

    let txt = '';

    // ── Number of persons ───────────────────────────────
    txt += 'Number of Persons : ' + obj.count + '\n';
    txt += '─────────────────────────────────────\n';

    // ── Per-person block ────────────────────────────────
    // The model gives scene-level activities (multi-hot)
    // and per-slot locations. We pair them by index —
    // activity[0] → Person 0, activity[1] → Person 1, etc.
    for (let i = 0; i < obj.count; i++) {
      const loc = obj.locations[i];

      // Best-effort: assign detected activities by index
      // e.g. if 2 persons and activities=[sitting, standing]
      // → Person 0 = sitting, Person 1 = standing
      const act = (obj.activities.length > i)
                  ? obj.activities[i]
                  : (obj.activities.length === 1 && obj.count >= 1)
                    ? obj.activities[0]   // single activity → all persons
                    : '(unknown)';

      txt += 'Person ' + i + '\n';
      txt += '  Activity : ' + act + '\n';
      txt += '  Location : X = ' + loc.x.toFixed(2)
           + ' m    Y = ' + loc.y.toFixed(2) + ' m\n';

      if (i < obj.count - 1) txt += '\n';
    }

    // ── Empty room ──────────────────────────────────────
    if (obj.count === 0) {
      txt += 'No person detected\n';
      txt += 'Activities : ' + (obj.activities.length
             ? obj.activities.join(', ') : '(none)') + '\n';
    }

    txt += '─────────────────────────────────────\n';
    txt += 'Confidence  : ' + obj.confidence.toFixed(1) + '%\n';
    txt += 'RX1 packets : ' + obj.rx1_packets
         + '   rejects : ' + obj.rx1_rejects;

    setText('testingResult', txt);

  } catch(e) {
    setText('testingResult', 'Poll error: ' + e);
  }
}




async function updateModelStatus() {
  try {
    const r   = await fetch('/modelstatus');
    const obj = await r.json();
    let txt = 'Loaded        : ' + (obj.loaded ? 'YES' : 'NO') + '\n';
    txt += 'Tensors       : ' + obj.tensors + '\n';
    txt += 'Normalization : ' + (obj.norm ? 'YES' : 'NO') + '\n';
    txt += 'PSRAM free    : ' + obj.psram_free_kb + ' KB\n';
    txt += 'Model size    : ' + obj.model_kb + ' KB';
    document.getElementById('modelStatus').textContent = txt;
  } catch(e) {
    document.getElementById('modelStatus').textContent = 'Status error: ' + e;
  }
}

setInterval(updateModelStatus, 3000);
updateModelStatus();

/* ── Server inference ───────────────────────────────── */
async function startServerTest() {
  const r = await get('/server_test/start');
  document.getElementById('serverResult').textContent =
    'Server test started — first result in ~15 seconds...';
  startServerPoll();
}

async function stopServerTest() {
  await get('/server_test/stop');
  stopServerPoll();
  document.getElementById('serverResult').textContent = 'Stopped.';
}

let serverPollTimer = null;
function startServerPoll() {
  if (serverPollTimer) return;
  serverPollTimer = setInterval(pollServerResult, 5000);
}
function stopServerPoll() {
  if (serverPollTimer) { clearInterval(serverPollTimer); serverPollTimer = null; }
}

async function pollServerResult() {
  try {
    const r   = await fetch('/server_result');
    const obj = await r.json();

    if (!obj.ready) {
      document.getElementById('serverResult').textContent =
        'Waiting for first result...';
      return;
    }

    let txt = '';
    txt += 'Number of Persons : ' + obj.count + '\n';
    txt += '─────────────────────────────────────\n';

    for (let i = 0; i < obj.count; i++) {
      const loc = obj.locations[i];
      const act = (obj.activities.length > i)
                  ? obj.activities[i]
                  : (obj.activities.length === 1 ? obj.activities[0] : '(unknown)');
      txt += 'Person ' + i + '\n';
      txt += '  Activity : ' + act + '\n';
      txt += '  Location : X=' + loc.x.toFixed(2) + 'm  Y=' + loc.y.toFixed(2) + 'm\n';
      if (i < obj.count - 1) txt += '\n';
    }

    if (obj.count === 0) {
      txt += 'No person detected\n';
      txt += 'Activities : ' + (obj.activities.length
             ? obj.activities.join(', ') : '(none)') + '\n';
    }

    txt += '─────────────────────────────────────\n';
    txt += 'Confidence : ' + obj.confidence.toFixed(1) + '%\n';
    txt += 'Windows    : ' + obj.windows + '\n';
    txt += 'Updated    : ' + Math.round(obj.age_ms / 1000) + 's ago';

    document.getElementById('serverResult').textContent = txt;

  } catch(e) {
    document.getElementById('serverResult').textContent = 'Poll error: ' + e;
  }
}
</script>
</body>
</html>
)HTML";


/* ============================================================
   HTTP HANDLERS
   ============================================================ */

void handleRoot() {
    server.send_P(200, "text/html", PAGE);
}

void handleStart() {
    if (collecting) {
        server.send(200, "text/plain", "SESSION ALREADY RUNNING");
        return;
    }

    String numStr     = server.arg("num_persons");
    int    numPersons = numStr.toInt();
    if (numPersons < 0) numPersons = 0;
    if (numPersons > 4) numPersons = 4;

    String subject = server.arg("subject");
    String note    = server.arg("note");

    collecting = true;

    Serial.print("SESSION_START,num_persons=");
    Serial.print(numPersons);

    for (int i = 0; i < numPersons; i++) {
        String actKey   = "p" + String(i) + "_activity";
        String xKey     = "p" + String(i) + "_x";
        String yKey     = "p" + String(i) + "_y";
        String activity = server.arg(actKey);
        String xVal     = server.arg(xKey);
        String yVal     = server.arg(yKey);

        Serial.print(",p"); Serial.print(i);
        Serial.print("_activity="); Serial.print(activity);
        Serial.print(",p"); Serial.print(i);
        Serial.print("_x="); Serial.print(xVal);
        Serial.print(",p"); Serial.print(i);
        Serial.print("_y="); Serial.print(yVal);
    }

    Serial.print(",subject="); Serial.print(subject);
    Serial.print(",note=");    Serial.println(note);

    server.send(200, "text/plain", "SESSION STARTED");
}

// AFTER:
void handleStop() {
    if (!collecting) {
        server.send(200, "text/plain", "NO ACTIVE SESSION");
        return;
    }
    collecting = false;
    pendingSessionStop = true;               // ← Core 1 will print this safely
    server.send(200, "text/plain", "SESSION STOPPED");
}

void handleStartTraining() {
    if (currentMode == MODE_TESTING) {
        server.send(200, "text/plain", "STOP TESTING FIRST");
        return;
    }
    currentMode    = MODE_TRAINING;
    trainingActive = true;
    trainingStatus = "TRAINING STARTED";
    Serial.println("TRAINING_START");
    server.send(200, "text/plain", "TRAINING STARTED");
}

void handleStopTraining() {
    currentMode    = MODE_IDLE;
    trainingActive = false;
    trainingStatus = "IDLE";
    server.send(200, "text/plain", "TRAINING STOPPED");
}

void handleStartTesting() {
    if (!modelLoaded) {
        server.send(200, "text/plain", "NO MODEL LOADED");
        return;
    }
    if (currentMode == MODE_TRAINING) {
        server.send(200, "text/plain", "TRAINING IN PROGRESS");
        return;
    }

    currentMode        = MODE_TESTING;
    testingActive      = true;
    testingFrameCount  = 0;
    testingStatus      = "TESTING STARTED";
    rx1SyncRejectCount = 0;
    rx1TestPacketCount = 0;

    for (int i = 0; i < RX1_BUF_SIZE; i++) rx1CircBuf[i].valid = false;
    rx1BufHead = 0;

    Serial.println("RX1_TEST_START");
    Serial.print("RX1_TEST_PORT:"); Serial.println(RX1_TEST_PORT);

    server.send(200, "text/plain", "TESTING STARTED");
}

void handleStopTesting() {
    currentMode       = MODE_IDLE;
    testingActive     = false;
    testingFrameCount = 0;
    testingStatus     = "NOT TESTING";
    Serial.println("RX1_TEST_STOP");
    server.send(200, "text/plain", "TESTING STOPPED");
}

void handleTestStatus() {
    String json = "{";
    json += "\"count\":"      + String(lastPredictedCount);
    json += ",\"confidence\":" + String(lastActConfidence * 100.0f, 1);
    json += ",\"rx1_packets\":" + String(rx1TestPacketCount);
    json += ",\"rx1_rejects\":" + String(rx1SyncRejectCount);

    json += ",\"activities\":[";
    bool first = true;
    for (int i = 0; i < N_ACTIVITIES; i++) {
        if (lastDetectedAct[i]) {
            if (!first) json += ",";
            json += "\"" + String(ACTIVITY_NAMES[i]) + "\"";
            first = false;
        }
    }
    json += "]";

    json += ",\"locations\":[";
    for (int i = 0; i < lastPredictedCount && i < MAX_PERSONS; i++) {
        if (i > 0) json += ",";
        json += "{\"x\":" + String(lastLocations[i][0], 3)
              + ",\"y\":" + String(lastLocations[i][1], 3) + "}";
    }
    json += "]}";

    server.send(200, "application/json", json);
}

void handleModelStatus() {
    String json = "{";
    json += "\"loaded\":"         + String(modelLoaded ? "true" : "false");
    json += ",\"tensors\":"       + String(modelTensorCount);
    json += ",\"norm\":"          + String(modelHasNormalization ? "true" : "false");
    json += ",\"psram_free_kb\":" + String((int)(ESP.getFreePsram() / 1024));
    json += ",\"model_kb\":"      + String((int)(modelSizeBytes / 1024));
    json += "}";
    server.send(200, "application/json", json);
}

/* ── /server_test/start ─────────────────────────────────── */
void handleServerTestStart() {
    serverTestActive  = true;
    pendingServerStart = true;   /* Core 1 prints SERVER_TEST_START */
    server.send(200, "text/plain", "SERVER_TEST_STARTED");
}

/* ── /server_test/stop ──────────────────────────────────── */
void handleServerTestStop() {
    serverTestActive   = false;
    pendingServerStop  = true;   /* Core 1 prints SERVER_TEST_STOP */
    serverResultReady  = false;
    server.send(200, "text/plain", "SERVER_TEST_STOPPED");
}

/* ── /server_result  (JSON polled by HTML every 5s) ─────── */
void handleServerResult() {
    if (!serverResultReady) {
        server.send(200, "application/json",
            "{\"ready\":false,\"count\":0,\"activities\":[],"
            "\"locations\":[],\"confidence\":0.0,"
            "\"windows\":0,\"age_ms\":0}");
        return;
    }

    uint32_t age  = millis() - serverResultTimestamp;
    String   json = String(serverResultJSON);

    /* Inject ready + age_ms into the existing JSON object */
    if (json.endsWith("}")) {
        json = json.substring(0, json.length() - 1);
        json += ",\"ready\":true";
        json += ",\"age_ms\":"  + String(age);
        json += "}";
    }

    server.send(200, "application/json", json);
}


/* ============================================================
   CSI CALLBACK  (IRAM — no Serial allowed here)
   ============================================================ */

void IRAM_ATTR wifiCSI_callback(void* ctx, wifi_csi_info_t* info) {
    if (!info || !info->buf) return;

    for (int i = 0; i < 6; i++) {
        if (info->mac[i] != WROOM_MAC[i]) { csiRejectedOther++; return; }
    }
    csiAcceptedWroom++;

    int len = info->len;
    if (len <= 0) return;
    if (len > MAX_CSI_BYTES) len = MAX_CSI_BYTES;

    for (int i = 0; i < len; i++) csiBuffer[i] = info->buf[i];
    csiLength  = len;
    csiRSSI    = info->rx_ctrl.rssi;
    csiChannel = info->rx_ctrl.channel;
    newCSI     = true;
}


/* ============================================================
   ENABLE CSI
   ============================================================ */

void enableCSI() {
    wifi_csi_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.lltf_en           = true;
    cfg.htltf_en          = true;
    cfg.stbc_htltf2_en    = true;
    cfg.ltf_merge_en      = true;
    cfg.channel_filter_en = false;
    cfg.manu_scale        = false;
    cfg.shift             = false;

    esp_wifi_set_csi_config(&cfg);
    esp_wifi_set_csi_rx_cb(&wifiCSI_callback, nullptr);
    esp_err_t err = esp_wifi_set_csi(true);

    if (err == ESP_OK) Serial.println("CSI_ENABLED");
    else { Serial.print("CSI_ENABLE_FAILED:"); Serial.println((int)err); }
}


/* ============================================================
   PROCESS RX2 CSI  (called from loop)
   ============================================================ */

void handleCSI() {
    if (!newCSI) return;

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
    latestRSSI         = rssi;

    /* RX2 amplitude for test frames */
    if (testingActive && len >= 384) {
        for (int i = 0; i < RX_FEATURES; i++) {
            int I = (int)localCSI[i * 2];
            int Q = (int)localCSI[i * 2 + 1];
            rx2Amplitude[i] = sqrtf((float)(I * I + Q * Q));
        }
        rx2FrameTimestamp = timestamp;
        rx2AmplitudeReady = true;
    }

    /* Serial output for collector.py */
     
    if (collecting || serverTestActive) {
        Serial.print("RX2,");
        Serial.print(timestamp); Serial.print(",");
        Serial.print(rssi);      Serial.print(",");
        Serial.print(sequence);  Serial.print(",");
        Serial.print(channel);   Serial.print(",");
        Serial.print("unknown"); Serial.print(",");
        Serial.print(len);       Serial.print(",");
        for (int i = 0; i < len; i++) {
            Serial.print((int)localCSI[i]);
            if (i < len - 1) Serial.print(",");
        }
        Serial.println();
    }
}


/* ============================================================
   RECEIVE RX1 TEST DATA → CIRCULAR BUFFER
   ============================================================ */

void receiveRX1TestingData() {
    int pktSize = rx1TestUDP.parsePacket();
    if (pktSize <= 0) return;

    char buf[1700];
    int  n = rx1TestUDP.read(buf, sizeof(buf) - 1);
    if (n <= 0) return;
    buf[n] = '\0';

    if (strncmp(buf, "RX1TEST,", 8) != 0) return;

    char* p = buf + 8;

    /* skip sequence */
    char* comma = strchr(p, ',');
    if (!comma) return;
    p = comma + 1;

    /* skip timestamp */
    comma = strchr(p, ',');
    if (!comma) return;
    p = comma + 1;

    /* parse 192 amplitudes */
    int   count = 0;
    float amps[RX_FEATURES];

    while (count < RX_FEATURES && *p != '\0') {
        amps[count++] = (float)atoi(p);
        comma = strchr(p, ',');
        if (!comma) break;
        p = comma + 1;
    }

    if (count != RX_FEATURES) return;

    int slot = rx1BufHead;
    for (int i = 0; i < RX_FEATURES; i++) rx1CircBuf[slot].amplitude[i] = amps[i];
    rx1CircBuf[slot].timestamp_ms = millis();
    rx1CircBuf[slot].valid        = true;
    rx1BufHead = (rx1BufHead + 1) % RX1_BUF_SIZE;

    rx1TestPacketCount++;
}


/* ============================================================
   MODEL UPLOAD  (3-state serial receiver)
   ============================================================
   STATE 0 (modelWaitingForEnd):
       Wait for "MODEL_END\n" then parse.
   STATE 1 (!modelReceiving && !modelWaitingForEnd):
       Wait for "MODEL_BEGIN,<size>,3\n".
   STATE 2 (modelReceiving):
       Receive raw binary bytes into PSRAM.
   ============================================================ */

void handleModelUpload() {
    static uint8_t       recvBuf[1024];
    static unsigned long lastDataTime = 0;

    /* ── STATE 0: wait for MODEL_END ────────────────────────
       Wait until the full "MODEL_END\n" string (10 bytes) is
       in the RX buffer before calling readStringUntil.
       Use a short timeout so the loop stays responsive.
    ─────────────────────────────────────────────────────── */
    if (modelWaitingForEnd) {
        if (Serial.available() >= 10) {
            Serial.setTimeout(200);                    /* don't block loop */
            String end = Serial.readStringUntil('\n');
            Serial.setTimeout(1000);                   /* restore default  */
            end.trim();

            if (end == "MODEL_END") {
                modelWaitingForEnd = false;

                Serial.println("MODEL_PARSING");

                if (parseModelContainer()) {
    Serial.println("MODEL_LOADED");
    Serial.println("FLASH_SAVING...");
    if (saveModelToFlash()) {
        Serial.println("FLASH_SAVE_OK");
    } else {
        Serial.println("FLASH_SAVE_FAILED");
    }
} else {
    Serial.println("MODEL_PARSE_FAILED");
}

            }
        }
        return;
    }

    /* ── STATE 1: wait for MODEL_BEGIN ──────────────────────
       Only enter if no transfer is active.
    ─────────────────────────────────────────────────────── */
        if (!modelReceiving) {
        if (!Serial.available()) return;

        Serial.setTimeout(200);
        String cmd = Serial.readStringUntil('\n');
        Serial.setTimeout(1000);
        cmd.trim();

        /* ── SERVER_RESULT from collector.py ─────────── */
        if (cmd.startsWith("SERVER_RESULT:")) {
            String jsonPart = cmd.substring(14);
            jsonPart.toCharArray(serverResultJSON, sizeof(serverResultJSON));
            serverResultTimestamp = millis();
            serverResultReady     = true;
            return;
        }

        /* ── MODEL_BEGIN from weightloader.py ─────────── */
        if (!cmd.startsWith("MODEL_BEGIN,")) return;

        int c1 = cmd.indexOf(',');
        int c2 = cmd.indexOf(',', c1 + 1);
        if (c1 < 0 || c2 < 0) {
            Serial.println("MODEL_ERROR:BAD_HEADER");
            return;
        }

        uint32_t modelSize = (uint32_t)cmd.substring(c1 + 1, c2).toInt();
        uint32_t version   = (uint32_t)cmd.substring(c2 + 1).toInt();

        Serial.print("MODEL_SIZE:"); Serial.println(modelSize);
        Serial.print("MODEL_VER:");  Serial.println(version);

        if (version != 3) {
            Serial.println("MODEL_ERROR:UNSUPPORTED_VERSION (need 3)");
            return;
        }
        if (modelBuffer == nullptr) {
            Serial.println("MODEL_ERROR:PSRAM_NOT_READY");
            return;
        }
        if (modelSize == 0 || modelSize > modelCapacityBytes) {
            Serial.print("MODEL_ERROR:TOO_LARGE capacity=");
            Serial.println(modelCapacityBytes);
            return;
        }

        modelLoaded        = false;
        modelSizeBytes     = 0;
        modelBytesReceived = 0;
        expectedModelBytes = modelSize;
        receivedModelBytes = 0;
        modelWaitingForEnd = false;
        modelReceiving     = true;
        lastDataTime       = millis();

        Serial.print("PSRAM_FREE_BEFORE_UPLOAD:");
        Serial.println(ESP.getFreePsram());
        Serial.println("MODEL_READY");
        return;
    }

    /* ── STATE 2: receive raw bytes ─────────────────────────
       Read up to 1024 bytes per call, copy into PSRAM buffer.
       The 4096-byte serial RX buffer (set in setup) prevents
       byte loss between handleModelUpload() calls.
    ─────────────────────────────────────────────────────── */
    int avail = Serial.available();
    if (avail > 0) {
        uint32_t remaining = expectedModelBytes - receivedModelBytes;
        int toRead = min(avail, (int)sizeof(recvBuf));
        toRead     = min(toRead, (int)remaining);

        int got = Serial.read(recvBuf, toRead);
        if (got > 0) {
            memcpy(modelBuffer + receivedModelBytes, recvBuf, got);
            receivedModelBytes += (uint32_t)got;
            modelBytesReceived  = receivedModelBytes;
            lastDataTime        = millis();
        }
    }

    /* Timeout */
    if (millis() - lastDataTime > 15000UL) {
        Serial.println("MODEL_ERROR:TIMEOUT");
        modelReceiving     = false;
        modelWaitingForEnd = false;
        receivedModelBytes = 0;
        expectedModelBytes = 0;
        return;
    }

    /* All bytes received — move to STATE 0 */
    if (receivedModelBytes >= expectedModelBytes) {
        modelReceiving     = false;
        modelSizeBytes     = receivedModelBytes;
        modelWaitingForEnd = true;
        Serial.print("MODEL_BYTES_COMPLETE:");
        Serial.println(modelSizeBytes);
    }
}


/* ============================================================
   FLASH BACKUP  (LittleFS)
   ============================================================ */

bool saveModelToFlash() {
    if (!modelLoaded || modelSizeBytes == 0) {
        Serial.println("FLASH_SAVE_SKIPPED:not_loaded");
        return false;
    }

    Serial.print("FLASH_SPACE_BEFORE:");
    Serial.println(LittleFS.totalBytes() - LittleFS.usedBytes());

    if (LittleFS.exists(MODEL_FLASH_TMP)) LittleFS.remove(MODEL_FLASH_TMP);

    File f = LittleFS.open(MODEL_FLASH_TMP, "w");
    if (!f) { Serial.println("FLASH_WRITE_FAILED"); return false; }

    size_t written = f.write(modelBuffer, modelSizeBytes);
    f.close();

    if (written != modelSizeBytes) {
        LittleFS.remove(MODEL_FLASH_TMP);
        Serial.print("FLASH_WRITE_INCOMPLETE:");
        Serial.print(written); Serial.print("/"); Serial.println(modelSizeBytes);
        return false;
    }

    if (LittleFS.exists(MODEL_FLASH_FILE)) LittleFS.remove(MODEL_FLASH_FILE);

    if (!LittleFS.rename(MODEL_FLASH_TMP, MODEL_FLASH_FILE)) {
        Serial.println("FLASH_RENAME_FAILED");
        LittleFS.remove(MODEL_FLASH_TMP);
        return false;
    }

    Serial.print("FLASH_SAVED:"); Serial.println(modelSizeBytes);
    return true;
}

bool loadModelFromFlash() {
    if (!LittleFS.exists(MODEL_FLASH_FILE)) {
        Serial.println("NO_FLASH_MODEL");
        return false;
    }

    File f = LittleFS.open(MODEL_FLASH_FILE, "r");
    if (!f) {
        Serial.println("FLASH_OPEN_FAILED");
        return false;
    }

    size_t fileSize = f.size();
    Serial.print("FLASH_FILE_SIZE:"); Serial.println(fileSize);

    if (fileSize == 0 || fileSize > MODEL_MAX_SIZE) {
        Serial.println("FLASH_FILE_SIZE_INVALID");
        f.close();
        return false;
    }

    /* ── KEY FIX ──────────────────────────────────────────
       Reuse the buffer already allocated by initPSRAM()
       instead of free() + ps_malloc().

       The original code did free(modelBuffer) then tried
       ps_malloc(fileSize) which failed due to PSRAM
       fragmentation from WiFi + LittleFS buffers, leaving
       modelBuffer = nullptr and causing PSRAM_NOT_READY.

       The initPSRAM() buffer (5MB) is always larger than
       the model (~4MB) so we can read directly into it.
    ─────────────────────────────────────────────────────── */
    if (modelBuffer == nullptr) {
        /* Should not happen if initPSRAM() succeeded,
           but guard anyway */
        Serial.println("FLASH_NO_BUFFER");
        f.close();
        return false;
    }

    if (fileSize > modelCapacityBytes) {
        Serial.print("FLASH_MODEL_TOO_LARGE_FOR_BUFFER:");
        Serial.println(modelCapacityBytes);
        f.close();
        return false;
    }

    /* Read directly into the existing buffer — no reallocation */
    size_t got = f.read(modelBuffer, fileSize);
    f.close();

    if (got != fileSize) {
        Serial.print("FLASH_READ_INCOMPLETE:");
        Serial.print(got); Serial.print("/");
        Serial.println(fileSize);
        /* Do NOT free modelBuffer — still usable for next upload */
        return false;
    }

    modelSizeBytes     = fileSize;
    modelBytesReceived = fileSize;

    Serial.print("FLASH_LOADED:"); Serial.println(fileSize);

    if (!parseModelContainer()) {
        Serial.println("FLASH_MODEL_PARSE_FAILED");
        /* Do NOT free modelBuffer — still usable for next upload */
        modelSizeBytes = 0;
        return false;
    }

    Serial.println("MODEL_READY_AFTER_BOOT");
    return true;
}


/* ============================================================
   INIT PSRAM
   ============================================================ */

bool initPSRAM() {
    if (!psramFound()) {
        Serial.println("PSRAM_NOT_FOUND");
        return false;
    }

    Serial.print("PSRAM_TOTAL:"); Serial.println(ESP.getPsramSize());

    /* Allocate 5 MB — enough for the ~4.1 MB float32 model */
    modelCapacityBytes = 5UL * 1024UL * 1024UL;
    modelBuffer        = (uint8_t*)ps_malloc(modelCapacityBytes);

    if (modelBuffer == nullptr) {
        Serial.println("PSRAM_ALLOC_FAILED");
        modelCapacityBytes = 0;
        return false;
    }

    Serial.print("PSRAM_MODEL_BUFFER:"); Serial.println(modelCapacityBytes);
    Serial.print("PSRAM_FREE:");         Serial.println(ESP.getFreePsram());
    return true;
}



/* ============================================================
   HTTP SERVER TASK  (runs on Core 0 permanently)
   ============================================================
   Arduino loop() runs on Core 1.
   Putting server.handleClient() on Core 0 means its
   50-150ms WiFi TX blocking never touches the CSI loop.
   ============================================================ */

void serverTask(void* pvParameters) {
    for (;;) {
        server.handleClient();
        vTaskDelay(1);   /* yield for 1 tick (~1ms) */
    }
}



/* ============================================================
   SETUP
   ============================================================ */

void setup() {
    /*
       FIX 1: Increase serial RX buffer BEFORE Serial.begin().
       Default is 256 bytes, which overflows during the 6-minute
       model upload at 115200 baud when server.handleClient()
       briefly stalls reading. 4096 bytes gives 355 ms headroom.
    */
    Serial.setRxBufferSize(4096);
    Serial.setTxBufferSize(4096);   // ← ADD THIS — reduces print blocking from 110ms to ~20ms
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("================================");
    Serial.println("RX2 ESP32-S3 CSI + INFERENCE");
    Serial.println("================================");

    /* PSRAM */
    if (!initPSRAM()) {
        Serial.println("FATAL:PSRAM_INIT_FAILED — halting");
        while (true) delay(1000);
    }

    /* LittleFS */
    if (LittleFS.begin(true)) {
        Serial.println("LittleFS_OK");
        Serial.println("LOADING_FLASH_MODEL...");
        if (loadModelFromFlash()) {
            Serial.println("FLASH_MODEL_LOADED");
        } else {
            Serial.println("NO_FLASH_MODEL — upload via weightloader.py");
        }
    } else {
        Serial.println("LittleFS_FAILED");
    }

    /* RX1 circular buffer */
    for (int i = 0; i < RX1_BUF_SIZE; i++) rx1CircBuf[i].valid = false;

    /* Wi-Fi: AP + STA */
    WiFi.mode(WIFI_AP_STA);
    WiFi.setSleep(false);
    WiFi.softAP(AP_SSID, AP_PASS);
    Serial.print("AP_IP:"); Serial.println(WiFi.softAPIP());

    /* Static IP for STA (TX network) */
    IPAddress RX2_STA_IP(192, 168, 4, 3);
    IPAddress TX_GATEWAY(192, 168, 4, 1);
    IPAddress TX_SUBNET (255, 255, 255, 0);
    if (!WiFi.config(RX2_STA_IP, TX_GATEWAY, TX_SUBNET)) {
        Serial.println("RX2_STATIC_IP_FAILED");
    }

    /* Connect to TX */
    Serial.println("CONNECTING_TO_TX...");
    WiFi.begin(TX_SSID, TX_PASS);

    unsigned long t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
        delay(500); Serial.print(".");
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("TX_WIFI_CONNECTED");
        Serial.print("RX2_STA_IP:"); Serial.println(WiFi.localIP());
    } else {
        Serial.println("TX_WIFI_FAILED — CSI collection unavailable");
    }

    /* UDP */
    if (udp.begin(TX_PORT)) {
        Serial.print("UDP_READY:"); Serial.println(TX_PORT);
    }
    if (rx1TestUDP.begin(RX1_TEST_PORT)) {
        Serial.print("RX1_TEST_UDP_READY:"); Serial.println(RX1_TEST_PORT);
    }

    /* HTTP */
        server.on("/",                  handleRoot);
    server.on("/start",             handleStart);
    server.on("/stop",              handleStop);
    server.on("/train/start",       handleStartTraining);
    server.on("/train/stop",        handleStopTraining);
    server.on("/test/start",        handleStartTesting);
    server.on("/test/stop",         handleStopTesting);
    server.on("/teststatus",        handleTestStatus);
    server.on("/modelstatus",       handleModelStatus);
    server.on("/server_test/start", handleServerTestStart);  /* ← NEW */
    server.on("/server_test/stop",  handleServerTestStop);   /* ← NEW */
    server.on("/server_result",     handleServerResult);     /* ← NEW */
    server.begin();
    Serial.println("HTTP_SERVER_READY");

    /* Pin HTTP server to Core 0 — keeps Core 1 free for CSI */
    xTaskCreatePinnedToCore(
        serverTask,    /* function         */
        "HTTPServer",  /* task name        */
        8192,          /* stack bytes      */
        NULL,          /* parameter        */
        2,             /* priority (2>1)   */
        NULL,          /* task handle      */
        0              /* Core 0           */
    );

    /* CSI */
    enableCSI();

    Serial.println("RX2_READY");
    Serial.println();
}


/* ============================================================
   LOOP
   ============================================================ */

void loop() {
    // server.handleClient();

    /* ── Deferred serial messages (safe — only Core 1 prints) ── */
    if (pendingSessionStop) {
        pendingSessionStop = false;
        Serial.println("SESSION_STOP");
    }
    if (pendingServerStart) {
        pendingServerStart = false;
        Serial.println("SERVER_TEST_START");
    }
    if (pendingServerStop) {
        pendingServerStop = false;
        Serial.println("SERVER_TEST_STOP");
    }

    /* Drain TX heartbeat UDP */
    int pkt = udp.parsePacket();
    if (pkt > 0) { char tmp[64]; udp.read(tmp, sizeof(tmp)); }

    /* Process CSI */
    handleCSI();

    /* RX1 sync + inference (testing mode only) */
    if (testingActive) {
        receiveRX1TestingData();
        buildTestingFrame();
    }

    /* Model upload or SERVER_RESULT parsing */
       /* Model upload — handleModelUpload handles everything */
    if (modelReceiving || modelWaitingForEnd) {
        handleModelUpload();
    } else if (currentMode != MODE_TESTING && Serial.available() >= 12) {
        handleModelUpload();
    }
    delay(1);
}
