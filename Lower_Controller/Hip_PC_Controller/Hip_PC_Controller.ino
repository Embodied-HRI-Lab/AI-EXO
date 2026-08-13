/*
 * Bilateral hip-exoskeleton Teensy torque execution layer
 * Simplified version: IMU is handled directly by PC / Raspberry Pi.
 *
 * Responsibilities kept on Teensy:
 *   1) Receive left/right desired torque over Serial3.
 *   2) Execute torque/current command at 1 kHz.
 *   3) Request motor realtime feedback at 250 Hz.
 *   4) Request motor status at 20 Hz.
 *   5) Return only left/right actual torque at 100 Hz.
 *   6) Apply command timeout, torque/current limit, slew-rate limit,
 *      velocity safety, motor-feedback safety and drive-fault safety.
 *
 * UART protocol
 * =============
 *
 * PC/RPi -> Teensy torque command, 15 bytes:
 *   A5 5A | 54 |
 *   seq:uint16 |
 *   left_desired_tau:float32 |
 *   right_desired_tau:float32 |
 *   enable:uint8 |
 *   crc8
 *
 * PC/RPi -> Teensy STOP, 4 bytes:
 *   A5 5A 50 crc8
 *
 * PC/RPi -> Teensy CLEAR_FAULT, 4 bytes:
 *   A5 5A 43 crc8
 *
 * Teensy -> PC/RPi torque feedback, 14 bytes:
 *   A5 5A | 44 |
 *   seq:uint16 |
 *   left_actual_tau:float32 |
 *   right_actual_tau:float32 |
 *   crc8
 *
 * Multi-byte values: little-endian.
 * CRC-8: polynomial 0x07, init 0x00, over CMD + payload.
 */

#include <Arduino.h>
#include <FlexCAN_T4.h>
#include <math.h>
#include <string.h>

#include "Motor_Control_Tmotor.h"

#define LINK_SERIAL Serial1

// ============================================================================
// 1. Configuration
// ============================================================================

namespace Config
{
constexpr uint8_t LEFT_MOTOR_ID  = 0x01;
constexpr uint8_t RIGHT_MOTOR_ID = 0x02;

constexpr uint32_t LINK_BAUD = 115200;
constexpr uint32_t CAN_BAUD  = 1000000;

// Main execution and feedback rates.
constexpr uint32_t CONTROL_PERIOD_US          = 1000;   // 1 kHz
constexpr uint32_t FEEDBACK_REQUEST_PERIOD_US = 4000;   // 250 Hz
constexpr uint32_t STATUS_REQUEST_PERIOD_US   = 50000;  // 20 Hz
constexpr uint32_t TELEMETRY_PERIOD_US        = 10000;  // 100 Hz

// Timeouts.
constexpr uint32_t COMMAND_TIMEOUT_US  = 200000;  // 200 ms
constexpr uint32_t FEEDBACK_TIMEOUT_US = 100000;  // 100 ms

// Torque/current safety.
constexpr float MAX_TORQUE_NM         = 5.0f;
constexpr float MAX_Q_CURRENT_A       = 10.0f;
constexpr float MAX_INPUT_TORQUE_NM   = 5.0f;
constexpr float MAX_TORQUE_SLEW_NM_S  = 50.0f;
constexpr float ZERO_EPS_NM           = 0.001f;

// Motor mounting directions.
constexpr float LEFT_TORQUE_SIGN  =  1.0f;
constexpr float RIGHT_TORQUE_SIGN = -1.0f;

// Motor velocity uses the same mechanical sign convention here.
constexpr float LEFT_VELOCITY_SIGN  =  1.0f;
constexpr float RIGHT_VELOCITY_SIGN = -1.0f;

// Keep this simple safety check because it costs almost no code/runtime.
constexpr bool ENABLE_VELOCITY_LIMIT = true;
constexpr float MAX_ABS_VELOCITY_RAD_S = 20.0f;
}

// ============================================================================
// 2. UART protocol
// ============================================================================

namespace Protocol
{
constexpr uint8_t HEADER1 = 0xA5;
constexpr uint8_t HEADER2 = 0x5A;

constexpr uint8_t CMD_TORQUE      = 0x54;
constexpr uint8_t CMD_STOP        = 0x50;
constexpr uint8_t CMD_CLEAR_FAULT = 0x43;
constexpr uint8_t CMD_STATE       = 0x44;

constexpr size_t TORQUE_PAYLOAD_SIZE = 11;  // seq2 + L4 + R4 + enable1
constexpr size_t STATE_PAYLOAD_SIZE  = 10;  // seq2 + L4 + R4
constexpr size_t STATE_FRAME_SIZE    = 14;  // header2 + cmd1 + payload10 + crc1

static_assert(STATE_FRAME_SIZE == 14, "State frame must be 14 bytes.");
}

// ============================================================================
// 3. Hardware and runtime state
// ============================================================================

Motor_Control_Tmotor left_motor(Config::LEFT_MOTOR_ID);
Motor_Control_Tmotor right_motor(Config::RIGHT_MOTOR_ID);
CAN_message_t can_message;

struct MotorFeedback
{
  float velocity = 0.0f;
  float actual_torque = 0.0f;
  uint32_t last_feedback_us = 0;
};

struct TorqueChannel
{
  float requested = 0.0f;  // latest valid command from PC/RPi
  float ramped = 0.0f;     // slew-limited joint torque
  float applied = 0.0f;    // joint-coordinate torque actually sent
};

struct Runtime
{
  MotorFeedback left_feedback;
  MotorFeedback right_feedback;

  TorqueChannel left_torque;
  TorqueChannel right_torque;

  bool command_enabled = false;
  uint16_t last_command_sequence = 0;
  uint16_t telemetry_sequence = 0;

  uint32_t last_valid_command_us = 0;

  uint32_t previous_control_us = 0;
  uint32_t previous_feedback_request_us = 0;
  uint32_t previous_status_request_us = 0;
  uint32_t previous_telemetry_us = 0;
};

Runtime rt;

// ============================================================================
// 4. Utility functions
// ============================================================================

static float clampf(float value, float min_value, float max_value)
{
  return fminf(fmaxf(value, min_value), max_value);
}

static float slew_limit(
    float target,
    float previous,
    float max_rate,
    float dt_s)
{
  const float max_delta = fmaxf(max_rate, 0.0f) * fmaxf(dt_s, 0.0f);
  return clampf(target, previous - max_delta, previous + max_delta);
}

static uint8_t crc8_update(uint8_t crc, uint8_t data)
{
  crc ^= data;

  for (uint8_t i = 0; i < 8; ++i)
  {
    if (crc & 0x80)
      crc = static_cast<uint8_t>((crc << 1) ^ 0x07);
    else
      crc <<= 1;
  }

  return crc;
}

static uint8_t crc8_compute(
    uint8_t command,
    const uint8_t* payload,
    size_t length)
{
  uint8_t crc = crc8_update(0x00, command);

  for (size_t i = 0; i < length; ++i)
    crc = crc8_update(crc, payload[i]);

  return crc;
}

static void append_bytes(
    uint8_t* buffer,
    size_t& index,
    const void* value,
    size_t size)
{
  memcpy(buffer + index, value, size);
  index += size;
}

// ============================================================================
// 5. Motor safety / readiness
// ============================================================================

static bool feedback_is_fresh(uint32_t last_feedback_us, uint32_t now_us)
{
  return last_feedback_us != 0 &&
         (uint32_t)(now_us - last_feedback_us) <= Config::FEEDBACK_TIMEOUT_US;
}

static bool motors_ready(uint32_t now_us)
{
  if (!left_motor.torque_constant_is_valid() ||
      !right_motor.torque_constant_is_valid())
    return false;

  if (!feedback_is_fresh(rt.left_feedback.last_feedback_us, now_us) ||
      !feedback_is_fresh(rt.right_feedback.last_feedback_us, now_us))
    return false;

  if (left_motor.fault_code != 0 || right_motor.fault_code != 0)
    return false;

  return true;
}

static void command_zero_current()
{
  left_motor.command_q_current_A(0.0f);
  right_motor.command_q_current_A(0.0f);

  rt.left_torque.ramped = 0.0f;
  rt.right_torque.ramped = 0.0f;
  rt.left_torque.applied = 0.0f;
  rt.right_torque.applied = 0.0f;
}

static void stop_immediately()
{
  rt.command_enabled = false;
  rt.left_torque.requested = 0.0f;
  rt.right_torque.requested = 0.0f;
  command_zero_current();
}

// ============================================================================
// 6. CAN initialization and feedback
// ============================================================================

static void request_motor_feedback()
{
  left_motor.request_realtime();
  right_motor.request_realtime();
}

static void request_motor_status()
{
  left_motor.request_status();
  right_motor.request_status();
}

static bool reply_contains_motion_feedback(uint8_t command)
{
  return command == 0xA4 ||
         command == 0xA3 ||
         command == 0xC2 ||
         command == 0xC3 ||
         command == 0xC4 ||
         command == 0xF1;
}

static bool reply_contains_torque_feedback(uint8_t command)
{
  return command == 0xC0 ||
         command == 0xA1 ||
         command == 0xA4 ||
         command == 0xF1;
}

static void drain_can_feedback()
{
  while (Motor_Control_Tmotor::Can3.read(can_message))
  {
    const uint32_t now_us = micros();
    const uint8_t reply_command =
        can_message.len > 0 ? can_message.buf[0] : 0x00;

    if (can_message.id == Config::LEFT_MOTOR_ID)
    {
      left_motor.handle_reply(can_message);

      rt.left_feedback.velocity =
          Config::LEFT_VELOCITY_SIGN * left_motor.spe;

      if (reply_contains_motion_feedback(reply_command))
        rt.left_feedback.last_feedback_us = now_us;

      if (reply_contains_torque_feedback(reply_command))
        rt.left_feedback.actual_torque =
            Config::LEFT_TORQUE_SIGN * left_motor.torque;
    }
    else if (can_message.id == Config::RIGHT_MOTOR_ID)
    {
      right_motor.handle_reply(can_message);

      rt.right_feedback.velocity =
          Config::RIGHT_VELOCITY_SIGN * right_motor.spe;

      if (reply_contains_motion_feedback(reply_command))
        rt.right_feedback.last_feedback_us = now_us;

      if (reply_contains_torque_feedback(reply_command))
        rt.right_feedback.actual_torque =
            Config::RIGHT_TORQUE_SIGN * right_motor.torque;
    }
  }
}

static bool initialize_motors()
{
  left_motor.initial_CAN(Config::CAN_BAUD);
  delay(200);

  // Clear possible power-on drive faults.
  left_motor.error_clear();
  right_motor.error_clear();
  delay(50);

  // Obtain motor parameters required for torque -> current conversion.
  const uint32_t parameter_deadline = millis() + 1000;

  while (millis() < parameter_deadline)
  {
    if (!left_motor.torque_constant_is_valid())
      left_motor.request_motor_parameters();

    if (!right_motor.torque_constant_is_valid())
      right_motor.request_motor_parameters();

    delay(5);
    drain_can_feedback();

    if (left_motor.torque_constant_is_valid() &&
        right_motor.torque_constant_is_valid())
      break;
  }

  if (!left_motor.torque_constant_is_valid() ||
      !right_motor.torque_constant_is_valid())
    return false;

  // Keep both software limits in the motor wrapper as a second layer of safety.
  left_motor.set_software_current_limit_A(Config::MAX_Q_CURRENT_A);
  right_motor.set_software_current_limit_A(Config::MAX_Q_CURRENT_A);

  left_motor.set_software_torque_limit_Nm(Config::MAX_TORQUE_NM);
  right_motor.set_software_torque_limit_Nm(Config::MAX_TORQUE_NM);

  command_zero_current();

  // Wait for the first realtime feedback from both motors.
  const uint32_t feedback_deadline = millis() + 1000;

  while (millis() < feedback_deadline)
  {
    request_motor_feedback();
    delay(5);
    drain_can_feedback();

    if (rt.left_feedback.last_feedback_us != 0 &&
        rt.right_feedback.last_feedback_us != 0)
      break;
  }

  request_motor_status();
  delay(20);
  drain_can_feedback();

  command_zero_current();

  return rt.left_feedback.last_feedback_us != 0 &&
         rt.right_feedback.last_feedback_us != 0;
}

// ============================================================================
// 7. Torque execution
// ============================================================================

static float send_joint_torque(
    Motor_Control_Tmotor& motor,
    float joint_torque_nm,
    float torque_sign)
{
  if (!motor.torque_constant_is_valid() ||
      !isfinite(motor.torque_constant) ||
      fabsf(motor.torque_constant) < 1.0e-8f)
  {
    motor.command_q_current_A(0.0f);
    return 0.0f;
  }

  const float safe_joint_torque = clampf(
      joint_torque_nm,
      -Config::MAX_TORQUE_NM,
      Config::MAX_TORQUE_NM);

  const float motor_torque = torque_sign * safe_joint_torque;

  const float target_current = clampf(
      motor_torque / motor.torque_constant,
      -Config::MAX_Q_CURRENT_A,
      Config::MAX_Q_CURRENT_A);

  if (!motor.command_q_current_A(target_current))
    return 0.0f;

  // Convert the actually commanded motor current back to joint-coordinate torque.
  return torque_sign * target_current * motor.torque_constant;
}

static float apply_velocity_safety(
    float torque,
    float velocity)
{
  if (!Config::ENABLE_VELOCITY_LIMIT)
    return torque;

  if (velocity >= Config::MAX_ABS_VELOCITY_RAD_S && torque > 0.0f)
    return 0.0f;

  if (velocity <= -Config::MAX_ABS_VELOCITY_RAD_S && torque < 0.0f)
    return 0.0f; 

  return torque;  
}

static void update_control(uint32_t now_us, float dt_s)
{
  // Communication timeout is not latched.
  // It simply disables assistance and lets slew limiting bring torque to zero.
  if (rt.command_enabled &&
      (uint32_t)(now_us - rt.last_valid_command_us) >
          Config::COMMAND_TIMEOUT_US)
  {
    rt.command_enabled = false;
    rt.left_torque.requested = 0.0f;
    rt.right_torque.requested = 0.0f;
  }  

  // Motor feedback/drive problems are treated more conservatively:
  // output is forced to zero immediately.
  if (!motors_ready(now_us))
  {
    command_zero_current();
    return;
  }

  const float left_target =
      rt.command_enabled ? rt.left_torque.requested : 0.0f;

  const float right_target =
      rt.command_enabled ? rt.right_torque.requested : 0.0f;

  rt.left_torque.ramped = slew_limit(
      left_target,
      rt.left_torque.ramped,
      Config::MAX_TORQUE_SLEW_NM_S,
      dt_s);

  rt.right_torque.ramped = slew_limit(
      right_target,
      rt.right_torque.ramped,
      Config::MAX_TORQUE_SLEW_NM_S,
      dt_s);

  float left_safe = clampf(
      rt.left_torque.ramped,
      -Config::MAX_TORQUE_NM,
      Config::MAX_TORQUE_NM);

  float right_safe = clampf(
      rt.right_torque.ramped,
      -Config::MAX_TORQUE_NM,
      Config::MAX_TORQUE_NM);

  left_safe = apply_velocity_safety(
      left_safe,
      rt.left_feedback.velocity);

  right_safe = apply_velocity_safety(
      right_safe,
      rt.right_feedback.velocity);

  if (fabsf(left_safe) < Config::ZERO_EPS_NM)
    left_safe = 0.0f;

  if (fabsf(right_safe) < Config::ZERO_EPS_NM)
    right_safe = 0.0f;

  rt.left_torque.applied = send_joint_torque(
      left_motor,
      left_safe,
      Config::LEFT_TORQUE_SIGN);

  rt.right_torque.applied = send_joint_torque(
      right_motor,
      right_safe,
      Config::RIGHT_TORQUE_SIGN);
}

// ============================================================================
// 8. UART RX parser
// ============================================================================

enum class RxState : uint8_t
{
  HEADER1,
  HEADER2,
  COMMAND,
  PAYLOAD,
  CRC
};

struct RxParser
{
  RxState state = RxState::HEADER1;
  uint8_t command = 0;
  uint8_t payload[Protocol::TORQUE_PAYLOAD_SIZE] = {};
  size_t expected_payload = 0;
  size_t index = 0;
  uint8_t crc = 0;

  void reset()
  {
    state = RxState::HEADER1;
    command = 0;
    expected_payload = 0;
    index = 0;
    crc = 0;
  }
};

RxParser rx;

static void handle_torque_packet()
{
  uint16_t sequence = 0;
  float left_tau = 0.0f;
  float right_tau = 0.0f;
  uint8_t enable = 0;

  memcpy(&sequence, rx.payload + 0, sizeof(sequence));
  memcpy(&left_tau, rx.payload + 2, sizeof(left_tau));
  memcpy(&right_tau, rx.payload + 6, sizeof(right_tau));
  memcpy(&enable, rx.payload + 10, sizeof(enable));

  if (!isfinite(left_tau) ||
      !isfinite(right_tau) ||
      fabsf(left_tau) > Config::MAX_INPUT_TORQUE_NM ||
      fabsf(right_tau) > Config::MAX_INPUT_TORQUE_NM)
  {
    return;
  }

  rt.last_command_sequence = sequence;
  rt.last_valid_command_us = micros();

  rt.command_enabled = (enable != 0);

  rt.left_torque.requested =
      rt.command_enabled ? left_tau : 0.0f;

  rt.right_torque.requested =
      rt.command_enabled ? right_tau : 0.0f;
}

static void handle_clear_fault()
{
  stop_immediately();

  left_motor.error_clear();
  right_motor.error_clear();

  delayMicroseconds(100);
  request_motor_status();
}

static void handle_complete_packet()
{
  if (rx.command == Protocol::CMD_TORQUE)
  {
    handle_torque_packet();
  }
  else if (rx.command == Protocol::CMD_STOP)
  {
    stop_immediately();
    rt.last_valid_command_us = micros();
  }
  else if (rx.command == Protocol::CMD_CLEAR_FAULT)
  {
    handle_clear_fault();
  }
}

static void process_serial_rx()
{
  while (LINK_SERIAL.available() > 0)
  {
    const uint8_t byte_in =
        static_cast<uint8_t>(LINK_SERIAL.read());

    switch (rx.state)
    {
      case RxState::HEADER1:
        if (byte_in == Protocol::HEADER1)
          rx.state = RxState::HEADER2;
        break;

      case RxState::HEADER2:
        if (byte_in == Protocol::HEADER2)
        {
          rx.state = RxState::COMMAND;
        }
        else if (byte_in != Protocol::HEADER1)
        {
          rx.state = RxState::HEADER1;
        }
        break;

      case RxState::COMMAND:
        rx.command = byte_in;
        rx.crc = crc8_update(0x00, rx.command);
        rx.index = 0;

        if (rx.command == Protocol::CMD_TORQUE)
        {
          rx.expected_payload = Protocol::TORQUE_PAYLOAD_SIZE;
          rx.state = RxState::PAYLOAD;
        }
        else if (rx.command == Protocol::CMD_STOP ||
                 rx.command == Protocol::CMD_CLEAR_FAULT)
        {
          rx.expected_payload = 0;
          rx.state = RxState::CRC;
        }
        else
        {
          rx.reset();
        }
        break;

      case RxState::PAYLOAD:
        if (rx.index >= sizeof(rx.payload))
        {
          rx.reset();
          break;
        }

        rx.payload[rx.index++] = byte_in;
        rx.crc = crc8_update(rx.crc, byte_in);

        if (rx.index >= rx.expected_payload)
          rx.state = RxState::CRC;
        break;

      case RxState::CRC:
        if (byte_in == rx.crc)
          handle_complete_packet();

        rx.reset();
        break;
    }
  }
}

// ============================================================================
// 9. Torque feedback telemetry
// ============================================================================

static void send_torque_feedback()
{
  uint8_t payload[Protocol::STATE_PAYLOAD_SIZE];
  size_t payload_index = 0;

  const uint16_t sequence = rt.telemetry_sequence++;

  append_bytes(
      payload,
      payload_index,
      &sequence,
      sizeof(sequence));

  append_bytes(
      payload,
      payload_index,
      &rt.left_feedback.actual_torque,
      sizeof(float));

  append_bytes(
      payload,
      payload_index,
      &rt.right_feedback.actual_torque,
      sizeof(float));

  uint8_t frame[Protocol::STATE_FRAME_SIZE];
  size_t frame_index = 0;

  frame[frame_index++] = Protocol::HEADER1;
  frame[frame_index++] = Protocol::HEADER2;
  frame[frame_index++] = Protocol::CMD_STATE;

  memcpy(frame + frame_index, payload, payload_index);
  frame_index += payload_index;

  frame[frame_index++] = crc8_compute(
      Protocol::CMD_STATE,
      payload,
      payload_index);

  // 14 bytes at 100 Hz = 1400 B/s. 115200 baud has plenty of headroom.
  // If the UART TX buffer is momentarily full, skip this telemetry frame
  // rather than blocking the 1 kHz control loop.
  if (LINK_SERIAL.availableForWrite() >=
      static_cast<int>(Protocol::STATE_FRAME_SIZE))
  {
    LINK_SERIAL.write(frame, Protocol::STATE_FRAME_SIZE);
  }
}

// ============================================================================
// 10. Setup / loop
// ============================================================================

void setup()
{
  // Preserve a short power-up settling time for the drives.
  delay(3000);

  LINK_SERIAL.begin(Config::LINK_BAUD);

  rt = Runtime{};

  const bool motors_initialized = initialize_motors();

  const uint32_t now_us = micros();

  rt.previous_control_us = now_us;
  rt.previous_feedback_request_us = now_us;
  rt.previous_status_request_us = now_us;
  rt.previous_telemetry_us = now_us;
  rt.last_valid_command_us = now_us;

  // Never start assistance automatically.
  rt.command_enabled = false;

  if (!motors_initialized)
    command_zero_current();
}

void loop()
{
  // Always drain communication first.
  process_serial_rx();
  drain_can_feedback();

  const uint32_t now_us = micros();

  if ((uint32_t)(now_us - rt.previous_feedback_request_us) >=
      Config::FEEDBACK_REQUEST_PERIOD_US)
  {
    rt.previous_feedback_request_us = now_us;
    request_motor_feedback();
  }

  if ((uint32_t)(now_us - rt.previous_status_request_us) >=
      Config::STATUS_REQUEST_PERIOD_US)
  {
    rt.previous_status_request_us = now_us;
    request_motor_status();
  }

  if ((uint32_t)(now_us - rt.previous_control_us) >=
      Config::CONTROL_PERIOD_US)
  {
    const uint32_t elapsed_us = now_us - rt.previous_control_us;
    rt.previous_control_us = now_us;

    const float dt_s = clampf(
        elapsed_us * 1.0e-6f,
        0.0001f,
        0.02f);

    update_control(now_us, dt_s);
  }

  if ((uint32_t)(now_us - rt.previous_telemetry_us) >=
      Config::TELEMETRY_PERIOD_US)
  {
    rt.previous_telemetry_us = now_us;
    send_torque_feedback();
  }
}