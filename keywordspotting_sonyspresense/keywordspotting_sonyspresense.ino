//#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/spresense/debug_log_callback.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "model.h"  /* quantized model */
#include <Audio.h>

#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels


#define OLED_RESET     -1 // Reset pin # (or -1 if sharing Arduino reset pin)
#define SCREEN_ADDRESS 0x3C ///< See datasheet for Address; 0x3D for 128x64, 0x3C for 128x32
#define READ_PACKET_SIZE 32000
#define BUFFER_SIZE READ_PACKET_SIZE
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

AudioClass *audio;
const char * classes[] = {"Waiting...","yes", "no", "up", "down", "left", "right", "forward", "backward"};
char buffer[BUFFER_SIZE];
tflite::ErrorReporter* error_reporter = nullptr;
tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 190 * 1024;
uint8_t tensor_arena[kTensorArenaSize];


void debug_log_printf(const char* s)
{
  Serial.print("ERROR:");
  Serial.println(s);
}
void setup() {
  Serial.begin(115200);
  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println("SSD1306 allocation failed");
    for(;;); 
  }
 pinMode(LED1, OUTPUT);

  audio = AudioClass::getInstance();

  audio->begin();

  audio->setRecorderMode(AS_SETRECDR_STS_INPUTDEVICE_MIC_A, 150);
  audio->initRecorder(AS_CODECTYPE_PCM, AS_SAMPLINGRATE_16000, AS_CHANNEL_MONO);



  tflite::InitializeTarget();
  memset(tensor_arena, 0, kTensorArenaSize * sizeof(uint8_t));

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided is schema version "
                   + String(model->version()) + " not equal "
                   + "to supported version "
                   + String(TFLITE_SCHEMA_VERSION));
    return;
  } else {
    Serial.println("Model version: " + String(model->version()));
  }

  static tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddAveragePool2D();
  resolver.AddConv2D();
  resolver.AddFullyConnected();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  RegisterDebugLogCallback(debug_log_printf);

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  } else {
    Serial.println("AllocateTensor() Success");
  }

  size_t used_size = interpreter->arena_used_bytes();
  Serial.println("Area used bytes: " + String(used_size));
  input = interpreter->input(0);
  output = interpreter->output(0);


}

void loop() {
  int16_t temp;

  uint32_t read_size = 0;
  uint32_t start_time = micros();
  audio->readFrames(buffer, BUFFER_SIZE, &read_size);
  audio->startRecorder();
  digitalWrite(LED1, HIGH);
  usleep(1000*1000);
  digitalWrite(LED1, LOW);
  audio->stopRecorder();
  Serial.println("Size");
  Serial.println(read_size);
  Serial.println("Start");
  Serial.write(buffer,read_size);
  int index = -1;
 for (int i = 0; i < read_size; i+=2) {
   temp  = (buffer[i + 1] << 8) | (buffer[i] & 0xFF );

    index++;
    input->data.f[index] = (float)(temp/32767.0);
   
 }

 
 
 TfLiteStatus invoke_status = interpreter->Invoke();
 if (invoke_status != kTfLiteOk) {
   Serial.println("Invoke failed");
   return;
 }
 

  float maxValue = output->data.f[0];
  int maxIndex = 0;
 for (int n = 0; n < 9; ++n) {
   float value = output->data.f[n];
   if (value > maxValue) {
    maxValue = value;
    maxIndex = n;
   }
 }
 uint32_t duration = micros() - start_time;
 Serial.println("Inference time = " + String(duration));
  display.clearDisplay();

  display.setTextSize(1); 
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);

  display.println("Inf= " + String(duration));
  display.display(); 
  
}
