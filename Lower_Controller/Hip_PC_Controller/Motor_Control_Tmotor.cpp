#include "Motor_Control_Tmotor.h"

#include <math.h>
#include <string.h>
#include <limits.h>

FlexCAN_T4<CAN3, RX_SIZE_256, TX_SIZE_16> Motor_Control_Tmotor::Can3;

namespace
{
bool g_can_initialized = false;
constexpr float TWO_PI_F = 6.2831853071795864769f;
constexpr float COUNTS_PER_REV_F = 16384.0f;
constexpr float MIT_KP_MIN = 0.0f;
constexpr float MIT_KP_MAX = 500.0f;
constexpr float MIT_KD_MIN = 0.0f;
constexpr float MIT_KD_MAX = 5.0f;
}

Motor_Control_Tmotor::Motor_Control_Tmotor(uint8_t id, int can_id_unused)
    : pos(0.0f),
      spe(0.0f),
      torque(0.0f),
      temp(0.0f),
      q_current_A(0.0f),
      speed_rpm(0.0f),
      single_turn_pos_rad(0.0f),
      raw_position_rad(0.0f),
      bus_voltage_V(0.0f),
      bus_current_A(0.0f),
      operation_mode(0),
      fault_code(0),
      mit_status(0),
      brake_state(0xFF),
      pole_pairs(0),
      torque_constant(0.0f),
      gear_ratio(0),
      torque_constant_valid(false),
      position_kp(0.0f),
      position_ki(0.0f),
      speed_kp(0.0f),
      speed_ki(0.0f),
      MIT_P_MAX(95.5f),
      MIT_V_MAX(45.0f),
      MIT_T_MAX(18.0f),
      mit_limits_valid(false),
      last_send_ok(false),
      ID(id),
      software_zero_rad(0.0f),
      have_single_turn_sample(false),
      last_single_turn_count(0),
      unwrapped_count(0),
      software_current_limit_A(0.20f),
      software_torque_limit_Nm(0.20f)
{
    (void)can_id_unused;

    // 0x00 and 0xFF are protocol special addresses, not normal single-device IDs.
    if (ID == 0x00 || ID == 0xFF) {
        ID = 0x01;
    }

    msgW = CAN_message_t{};
}

// =============================================================================
// CAN setup / send
// =============================================================================
void Motor_Control_Tmotor::initial_CAN(uint32_t baud)
{
    if (!g_can_initialized) {
        Can3.begin();
        Can3.setBaudRate(baud);
        delay(100);
        g_can_initialized = true;

        Serial.print("CAN3 initialized, baud = ");
        Serial.println(baud);
    }
}

bool Motor_Control_Tmotor::send_CAN_message()
{
    // Old code accidentally called Can3.write() twice. This version sends once.
    last_send_ok = Can3.write(msgW);

    if (!last_send_ok) {
        Serial.println("CAN send failed");
    }

    return last_send_ok;
}

bool Motor_Control_Tmotor::send_command(uint8_t command,
                                        const uint8_t *payload,
                                        uint8_t payload_len)
{
    if (payload_len > 7) {
        last_send_ok = false;
        return false;
    }

    msgW = CAN_message_t{};

    // Host -> motor. Using 0x100 | ID makes direction distinguishable.
    msgW.id = static_cast<uint16_t>(0x100U | ID);
    msgW.len = static_cast<uint8_t>(payload_len + 1U);
    msgW.flags.extended = 0;
    msgW.flags.remote = 0;
    msgW.flags.overrun = 0;
    msgW.flags.reserved = 0;

    msgW.buf[0] = command;
    for (uint8_t i = 0; i < payload_len; ++i) {
        msgW.buf[i + 1] = payload[i];
    }

    return send_CAN_message();
}

bool Motor_Control_Tmotor::send_u32_command(uint8_t command, uint32_t value)
{
    uint8_t payload[4];
    write_u32_le(payload, value);
    return send_command(command, payload, 4);
}

bool Motor_Control_Tmotor::send_i32_command(uint8_t command, int32_t value)
{
    uint8_t payload[4];
    write_i32_le(payload, value);
    return send_command(command, payload, 4);
}

bool Motor_Control_Tmotor::send_float_command(uint8_t command, float value)
{
    uint8_t payload[4];
    write_f32_le(payload, value);
    return send_command(command, payload, 4);
}

// =============================================================================
// System/read commands
// =============================================================================
bool Motor_Control_Tmotor::reboot()
{
    const uint8_t payload[7] = {
        0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF
    };
    return send_command(0x00, payload, 7);
}

bool Motor_Control_Tmotor::request_versions()  { return send_command(0xA0); }
bool Motor_Control_Tmotor::request_q_current() { return send_command(0xA1); }
bool Motor_Control_Tmotor::request_speed()     { return send_command(0xA2); }
bool Motor_Control_Tmotor::request_position()  { return send_command(0xA3); }
bool Motor_Control_Tmotor::request_realtime()  { return send_command(0xA4); }
bool Motor_Control_Tmotor::request_status()    { return send_command(0xAE); }
bool Motor_Control_Tmotor::error_clear()       { return send_command(0xAF); }

bool Motor_Control_Tmotor::request_pos_vel()
{
    const bool ok1 = request_position();
    const bool ok2 = request_speed();
    return ok1 && ok2;
}

bool Motor_Control_Tmotor::request_torque()
{
    // New protocol reports Q-axis current. torque is estimated as Kt * Iq.
    return request_q_current();
}

// =============================================================================
// Parameter commands
// =============================================================================
bool Motor_Control_Tmotor::request_motor_parameters()
{
    return send_command(0xB0);
}

bool Motor_Control_Tmotor::set_origin()
{
    return send_command(0xB1);
}

bool Motor_Control_Tmotor::set_position_max_speed_rpm(float rpm)
{
    if (!isfinite(rpm) || rpm < 0.0f) return false;

    const double raw = static_cast<double>(rpm) * 100.0; // 0.01 rpm
    if (raw > 4294967295.0) return false;

    return send_u32_command(0xB2, static_cast<uint32_t>(raw + 0.5));
}

bool Motor_Control_Tmotor::set_max_q_current_A(float amp)
{
    if (!isfinite(amp) || amp < 0.0f) return false;

    const double raw = static_cast<double>(amp) * 1000.0; // 0.001 A
    if (raw > 4294967295.0) return false;

    return send_u32_command(0xB3, static_cast<uint32_t>(raw + 0.5));
}

bool Motor_Control_Tmotor::set_q_current_slope_Aps(float amp_per_s)
{
    if (!isfinite(amp_per_s) || amp_per_s < 0.0f) return false;

    const double raw = static_cast<double>(amp_per_s) * 1000.0; // 0.001 A/s
    if (raw > 4294967295.0) return false;

    return send_u32_command(0xB4, static_cast<uint32_t>(raw + 0.5));
}

bool Motor_Control_Tmotor::set_speed_accel_rpmps(float rpm_per_s)
{
    if (!isfinite(rpm_per_s) || rpm_per_s < 0.0f) return false;

    const double raw = static_cast<double>(rpm_per_s) * 100.0; // 0.01 rpm/s
    if (raw > 4294967295.0) return false;

    return send_u32_command(0xB5, static_cast<uint32_t>(raw + 0.5));
}

bool Motor_Control_Tmotor::request_position_kp() { return send_command(0xB6); }
bool Motor_Control_Tmotor::request_position_ki() { return send_command(0xB7); }
bool Motor_Control_Tmotor::request_speed_kp()    { return send_command(0xB8); }
bool Motor_Control_Tmotor::request_speed_ki()    { return send_command(0xB9); }

bool Motor_Control_Tmotor::set_position_kp(float kp)
{
    return isfinite(kp) && send_float_command(0xB6, kp);
}

bool Motor_Control_Tmotor::set_position_ki(float ki)
{
    return isfinite(ki) && send_float_command(0xB7, ki);
}

bool Motor_Control_Tmotor::set_speed_kp(float kp)
{
    return isfinite(kp) && send_float_command(0xB8, kp);
}

bool Motor_Control_Tmotor::set_speed_ki(float ki)
{
    return isfinite(ki) && send_float_command(0xB9, ki);
}

// =============================================================================
// Normal control
// =============================================================================
bool Motor_Control_Tmotor::command_q_current_A(float current_A)
{
    if (!isfinite(current_A)) return false;

    current_A = clampf(current_A,
                       -software_current_limit_A,
                        software_current_limit_A);

    // 0xC0: signed 32-bit, unit 0.001 A
    const double raw = static_cast<double>(current_A) * 1000.0;
    if (raw > 2147483647.0 || raw < -2147483648.0) return false;

    const int32_t value = static_cast<int32_t>(raw >= 0.0 ? raw + 0.5 : raw - 0.5);
    return send_i32_command(0xC0, value);
}

bool Motor_Control_Tmotor::command_speed_rpm(float rpm)
{
    if (!isfinite(rpm)) return false;

    // 0xC1: signed 32-bit, unit 0.01 rpm
    const double raw = static_cast<double>(rpm) * 100.0;
    if (raw > 2147483647.0 || raw < -2147483648.0) return false;

    const int32_t value = static_cast<int32_t>(raw >= 0.0 ? raw + 0.5 : raw - 0.5);
    return send_i32_command(0xC1, value);
}

bool Motor_Control_Tmotor::command_abs_position_count(int32_t count)
{
    return send_i32_command(0xC2, count);
}

bool Motor_Control_Tmotor::command_rel_position_count(int32_t count)
{
    return send_i32_command(0xC3, count);
}

bool Motor_Control_Tmotor::command_abs_position_rad(float rad)
{
    if (!isfinite(rad)) return false;

    const double raw = static_cast<double>(rad) * COUNTS_PER_REV_F / TWO_PI_F;
    if (raw > 2147483647.0 || raw < -2147483648.0) return false;

    const int32_t count = static_cast<int32_t>(raw >= 0.0 ? raw + 0.5 : raw - 0.5);
    return command_abs_position_count(count);
}

bool Motor_Control_Tmotor::command_rel_position_rad(float rad)
{
    if (!isfinite(rad)) return false;

    const double raw = static_cast<double>(rad) * COUNTS_PER_REV_F / TWO_PI_F;
    if (raw > 2147483647.0 || raw < -2147483648.0) return false;

    const int32_t count = static_cast<int32_t>(raw >= 0.0 ? raw + 0.5 : raw - 0.5);
    return command_rel_position_count(count);
}

bool Motor_Control_Tmotor::return_to_origin()
{
    return send_command(0xC4);
}

bool Motor_Control_Tmotor::set_brake(bool closed)
{
    const uint8_t payload = closed ? 0x01 : 0x00;
    return send_command(0xCE, &payload, 1);
}

bool Motor_Control_Tmotor::request_brake_state()
{
    const uint8_t payload = 0xFF;
    return send_command(0xCE, &payload, 1);
}

bool Motor_Control_Tmotor::disable_motor()
{
    return send_command(0xCF);
}

bool Motor_Control_Tmotor::command_torque_Nm(float torque_Nm)
{
    if (!isfinite(torque_Nm) ||
        !torque_constant_valid ||
        fabsf(torque_constant) < 1.0e-8f) {
        return false;
    }

    torque_Nm = clampf(torque_Nm,
                       -software_torque_limit_Nm,
                        software_torque_limit_Nm);

    // Manual: torque = torque_constant * Q-axis current.
    // gear_ratio is deliberately NOT applied automatically.
    const float current_A = torque_Nm / torque_constant;
    return command_q_current_A(current_A);
}

// =============================================================================
// MIT mode
// =============================================================================
bool Motor_Control_Tmotor::request_mit_limits()
{
    return send_command(0xF0);
}

bool Motor_Control_Tmotor::set_mit_limits(float pos_max_rad,
                                          float vel_max_rad_s,
                                          float torque_max_Nm)
{
    if (!isfinite(pos_max_rad) ||
        !isfinite(vel_max_rad_s) ||
        !isfinite(torque_max_Nm) ||
        pos_max_rad <= 0.0f ||
        vel_max_rad_s <= 0.0f ||
        torque_max_Nm <= 0.0f) {
        return false;
    }

    const long p_raw = lroundf(pos_max_rad * 10.0f);      // 0.1 rad
    const long v_raw = lroundf(vel_max_rad_s * 100.0f);  // 0.01 rad/s
    const long t_raw = lroundf(torque_max_Nm * 100.0f);  // 0.01 Nm

    if (p_raw < 1 || p_raw > 65535L ||
        v_raw < 1 || v_raw > 65535L ||
        t_raw < 1 || t_raw > 65535L) {
        return false;
    }

    uint8_t payload[6];
    write_u16_le(&payload[0], static_cast<uint16_t>(p_raw));
    write_u16_le(&payload[2], static_cast<uint16_t>(v_raw));
    write_u16_le(&payload[4], static_cast<uint16_t>(t_raw));

    return send_command(0xF0, payload, 6);
}

bool Motor_Control_Tmotor::request_mit_state()
{
    return send_command(0xF1);
}

bool Motor_Control_Tmotor::mit_ctl_cmd(float p_des,
                                           float v_des,
                                           float kp,
                                           float kd,
                                           float t_ff)
{
    if (!isfinite(p_des) || !isfinite(v_des) || !isfinite(kp) ||
        !isfinite(kd) || !isfinite(t_ff)) {
        return false;
    }

    // The manual defaults are loaded at construction; preferably read F0 first.
    const float p_max = MIT_P_MAX;
    const float v_max = MIT_V_MAX;
    const float t_max = MIT_T_MAX;

    p_des = clampf(p_des, -p_max, p_max);
    v_des = clampf(v_des, -v_max, v_max);
    kp = clampf(kp, MIT_KP_MIN, MIT_KP_MAX);
    kd = clampf(kd, MIT_KD_MIN, MIT_KD_MAX);

    // Add an extra software torque clamp for human-interaction bring-up.
    const float t_safe = fminf(t_max, software_torque_limit_Nm);
    t_ff = clampf(t_ff, -t_safe, t_safe);

    const int p_int  = float_to_uint(p_des, -p_max, p_max, 16);
    const int v_int  = float_to_uint(v_des, -v_max, v_max, 12);
    const int kp_int = float_to_uint(kp, MIT_KP_MIN, MIT_KP_MAX, 12);
    const int kd_int = float_to_uint(kd, MIT_KD_MIN, MIT_KD_MAX, 12);
    const int t_int  = float_to_uint(t_ff, -t_max, t_max, 12);

    msgW = CAN_message_t{};

    // MIT control frame: Bit[10] = 1. 0x500 also includes the 0x100 TX bit.
    msgW.id = static_cast<uint16_t>(0x500U | ID);
    msgW.len = 8;
    msgW.flags.extended = 0;
    msgW.flags.remote = 0;
    msgW.flags.overrun = 0;
    msgW.flags.reserved = 0;

    // Explicit MIT packed layout from the manual (NOT normal little-endian fields).
    msgW.buf[0] = static_cast<uint8_t>((p_int >> 8) & 0xFF);
    msgW.buf[1] = static_cast<uint8_t>(p_int & 0xFF);
    msgW.buf[2] = static_cast<uint8_t>((v_int >> 4) & 0xFF);
    msgW.buf[3] = static_cast<uint8_t>(((v_int & 0x0F) << 4) |
                                       ((kp_int >> 8) & 0x0F));
    msgW.buf[4] = static_cast<uint8_t>(kp_int & 0xFF);
    msgW.buf[5] = static_cast<uint8_t>((kd_int >> 4) & 0xFF);
    msgW.buf[6] = static_cast<uint8_t>(((kd_int & 0x0F) << 4) |
                                       ((t_int >> 8) & 0x0F));
    msgW.buf[7] = static_cast<uint8_t>(t_int & 0xFF);

    return send_CAN_message();
}

// =============================================================================
// Legacy API compatibility
// =============================================================================
bool Motor_Control_Tmotor::enter_control_mode()
{
    // Normal C0/C1/C2/C3 commands select their mode directly.
    return true;
}

bool Motor_Control_Tmotor::exit_control_mode()        { return disable_motor(); }
bool Motor_Control_Tmotor::motor_reset()          { return reboot(); }
bool Motor_Control_Tmotor::encoder_reset()        { return set_origin(); }
bool Motor_Control_Tmotor::motor_start()          { return true; }
bool Motor_Control_Tmotor::motor_end()            { return disable_motor(); }
bool Motor_Control_Tmotor::torque_ctl_mode_start(){ return true; }
bool Motor_Control_Tmotor::speed_ctl_mode_start() { return true; }

bool Motor_Control_Tmotor::mit_ctl_mode_start()
{
    // A MIT command itself enters MIT mode. Zero gains + zero feedforward torque
    // is the safest neutral entry command.
    return mit_ctl_cmd(pos, 0.0f, 0.0f, 0.0f, 0.0f);
}

bool Motor_Control_Tmotor::speed_cmd(float rpm)
{
    return command_speed_rpm(rpm);
}

bool Motor_Control_Tmotor::speed_cmd()
{
    // Old version hard-coded a nonzero speed. New compatibility overload is safe.
    return command_speed_rpm(0.0f);
}

bool Motor_Control_Tmotor::torque_cmd(float tau_Nm)
{
    return command_torque_Nm(tau_Nm);
}

// =============================================================================
// Receive dispatcher
// =============================================================================
bool Motor_Control_Tmotor::handle_reply(const CAN_message_t &msg)
{
    // Slave reply StdID is the plain device address.
    if (msg.id != ID || msg.len == 0) {
        return false;
    }

    switch (msg.buf[0]) {
        case 0xA0:
            parse_A0(msg);
            return true;

        case 0xA1:
        case 0xC0:
            parse_A1_or_C0(msg);
            return true;

        case 0xA2:
        case 0xC1:
            parse_A2_or_C1(msg);
            return true;

        case 0xA3:
        case 0xC2:
        case 0xC3:
        case 0xC4:
            parse_A3_family(msg);
            return true;

        case 0xA4:
            parse_A4(msg);
            return true;

        case 0xAE:
        case 0xCF:
            parse_AE_or_CF(msg);
            return true;

        case 0xAF:
            parse_AF(msg);
            return true;

        case 0xB0:
            parse_B0(msg);
            return true;

        case 0xB1:
            parse_B1(msg);
            return true;

        case 0xB6:
        case 0xB7:
        case 0xB8:
        case 0xB9:
            parse_B6_B9(msg);
            return true;

        case 0xCE:
            parse_CE(msg);
            return true;

        case 0xF0:
            parse_F0(msg);
            return true;

        case 0xF1:
            parse_F1(msg);
            return true;

        default:
            // B2-B5 replies are valid echoes but are not stored in dedicated fields.
            return true;
    }
}

void Motor_Control_Tmotor::unpack_reply(CAN_message_t msgR, float initial_pos)
{
    software_zero_rad = initial_pos;
    handle_reply(msgR);
}

void Motor_Control_Tmotor::unpack_pos_vel(CAN_message_t msgR, float initial_pos)
{
    software_zero_rad = initial_pos;

    if (msgR.id != ID || msgR.len == 0) return;

    switch (msgR.buf[0]) {
        case 0xA2:
        case 0xC1:
            parse_A2_or_C1(msgR);
            break;

        case 0xA3:
        case 0xC2:
        case 0xC3:
        case 0xC4:
            parse_A3_family(msgR);
            break;

        case 0xA4:
            parse_A4(msgR);
            break;

        case 0xF1:
            parse_F1(msgR);
            break;

        default:
            break;
    }
}

void Motor_Control_Tmotor::unpack_torque(CAN_message_t msgR)
{
    if (msgR.id != ID || msgR.len == 0) return;

    switch (msgR.buf[0]) {
        case 0xA1:
        case 0xC0:
            parse_A1_or_C0(msgR);
            break;

        case 0xA4:
            parse_A4(msgR);
            break;

        case 0xF1:
            parse_F1(msgR);
            break;

        default:
            break;
    }
}

// =============================================================================
// Reply parsers
// =============================================================================
void Motor_Control_Tmotor::parse_A0(const CAN_message_t &msg)
{
    // A0 reply: DLC=8. Version fields are not exposed yet.
    (void)msg;
}

void Motor_Control_Tmotor::parse_A1_or_C0(const CAN_message_t &msg)
{
    if (msg.len != 5) return;

    const int32_t raw = read_i32_le(&msg.buf[1]);
    q_current_A = static_cast<float>(raw) * 0.001f;

    if (torque_constant_valid) {
        torque = torque_constant * q_current_A;
    }
}

void Motor_Control_Tmotor::parse_A2_or_C1(const CAN_message_t &msg)
{
    if (msg.len != 5) return;

    const int32_t raw = read_i32_le(&msg.buf[1]);
    speed_rpm = static_cast<float>(raw) * 0.01f;
    spe = speed_rpm * TWO_PI_F / 60.0f;
}

void Motor_Control_Tmotor::parse_A3_family(const CAN_message_t &msg)
{
    if (msg.len != 7) return;

    const uint16_t single_count = read_u16_le(&msg.buf[1]);
    const int32_t total_count = read_i32_le(&msg.buf[3]);
    update_position_from_multiturn(single_count, total_count);
}

void Motor_Control_Tmotor::parse_A4(const CAN_message_t &msg)
{
    if (msg.len != 8) return;

    temp = static_cast<float>(msg.buf[1]);

    const int16_t current_raw = read_i16_le(&msg.buf[2]);
    const int16_t speed_raw = read_i16_le(&msg.buf[4]);
    const uint16_t single_count = read_u16_le(&msg.buf[6]);

    q_current_A = static_cast<float>(current_raw) * 0.001f;
    speed_rpm = static_cast<float>(speed_raw) * 0.01f;
    spe = speed_rpm * TWO_PI_F / 60.0f;

    if (torque_constant_valid) {
        torque = torque_constant * q_current_A;
    }

    update_position_from_single_turn(single_count);
}

void Motor_Control_Tmotor::parse_AE_or_CF(const CAN_message_t &msg)
{
    if (msg.len != 8) return;

    bus_voltage_V = static_cast<float>(read_u16_le(&msg.buf[1])) * 0.01f;
    bus_current_A = static_cast<float>(read_u16_le(&msg.buf[3])) * 0.01f;
    temp = static_cast<float>(msg.buf[5]);
    operation_mode = msg.buf[6];
    fault_code = msg.buf[7];
}

void Motor_Control_Tmotor::parse_AF(const CAN_message_t &msg)
{
    if (msg.len != 2) return;
    fault_code = msg.buf[1];
}

void Motor_Control_Tmotor::parse_B0(const CAN_message_t &msg)
{
    if (msg.len != 7) return;

    pole_pairs = msg.buf[1];
    torque_constant = read_f32_le(&msg.buf[2]);
    gear_ratio = msg.buf[6];

    torque_constant_valid =
        isfinite(torque_constant) && fabsf(torque_constant) > 1.0e-8f;
}

void Motor_Control_Tmotor::parse_B1(const CAN_message_t &msg)
{
    if (msg.len != 3) return;

    // Hardware origin changed. Restart software unwrap tracking on next sample.
    have_single_turn_sample = false;
    unwrapped_count = 0;
    raw_position_rad = 0.0f;
    software_zero_rad = 0.0f;
    pos = 0.0f;
}

void Motor_Control_Tmotor::parse_B6_B9(const CAN_message_t &msg)
{
    if (msg.len != 5) return;

    const float value = read_f32_le(&msg.buf[1]);
    switch (msg.buf[0]) {
        case 0xB6: position_kp = value; break;
        case 0xB7: position_ki = value; break;
        case 0xB8: speed_kp = value; break;
        case 0xB9: speed_ki = value; break;
        default: break;
    }
}

void Motor_Control_Tmotor::parse_CE(const CAN_message_t &msg)
{
    if (msg.len != 2) return;
    brake_state = msg.buf[1];
}

void Motor_Control_Tmotor::parse_F0(const CAN_message_t &msg)
{
    if (msg.len != 7) return;

    MIT_P_MAX = static_cast<float>(read_u16_le(&msg.buf[1])) * 0.1f;
    MIT_V_MAX = static_cast<float>(read_u16_le(&msg.buf[3])) * 0.01f;
    MIT_T_MAX = static_cast<float>(read_u16_le(&msg.buf[5])) * 0.01f;

    mit_limits_valid = MIT_P_MAX > 0.0f &&
                       MIT_V_MAX > 0.0f &&
                       MIT_T_MAX > 0.0f;
}

void Motor_Control_Tmotor::parse_F1(const CAN_message_t &msg)
{
    if (msg.len != 7) return;

    // F1: Data[0] is command 0xF1, packed state begins at Data[1].
    const int p_int =
        (static_cast<int>(msg.buf[1]) << 8) |
         static_cast<int>(msg.buf[2]);

    const int v_int =
        (static_cast<int>(msg.buf[3]) << 4) |
        (static_cast<int>(msg.buf[4]) >> 4);

    const int t_int =
        ((static_cast<int>(msg.buf[4]) & 0x0F) << 8) |
         static_cast<int>(msg.buf[5]);

    raw_position_rad = uint_to_float(p_int, -MIT_P_MAX, MIT_P_MAX, 16);
    pos = raw_position_rad - software_zero_rad;

    spe = uint_to_float(v_int, -MIT_V_MAX, MIT_V_MAX, 12);
    speed_rpm = spe * 60.0f / TWO_PI_F;

    torque = uint_to_float(t_int, -MIT_T_MAX, MIT_T_MAX, 12);
    mit_status = msg.buf[6];
}

// =============================================================================
// Position tracking
// =============================================================================
void Motor_Control_Tmotor::update_position_from_single_turn(uint16_t count)
{
    single_turn_pos_rad = static_cast<float>(count) * TWO_PI_F / COUNTS_PER_REV_F;

    if (!have_single_turn_sample) {
        last_single_turn_count = count;
        unwrapped_count = static_cast<int64_t>(count);
        have_single_turn_sample = true;
    } else {
        int32_t delta = static_cast<int32_t>(count) -
                        static_cast<int32_t>(last_single_turn_count);

        // Unwrap across 0 / 16383.
        if (delta > 8192) {
            delta -= 16384;
        } else if (delta < -8192) {
            delta += 16384;
        }

        unwrapped_count += static_cast<int64_t>(delta);
        last_single_turn_count = count;
    }

    raw_position_rad = static_cast<float>(unwrapped_count) *
                       TWO_PI_F / COUNTS_PER_REV_F;
    pos = raw_position_rad - software_zero_rad;
}

void Motor_Control_Tmotor::update_position_from_multiturn(uint16_t single_count,
                                                          int32_t total_count)
{
    single_turn_pos_rad = static_cast<float>(single_count) *
                          TWO_PI_F / COUNTS_PER_REV_F;

    last_single_turn_count = single_count;
    unwrapped_count = static_cast<int64_t>(total_count);
    have_single_turn_sample = true;

    raw_position_rad = static_cast<float>(total_count) *
                       TWO_PI_F / COUNTS_PER_REV_F;
    pos = raw_position_rad - software_zero_rad;
}

// =============================================================================
// User configuration helpers
// =============================================================================
void Motor_Control_Tmotor::set_device_id(uint8_t id)
{
    if (id >= 1 && id <= 254) ID = id;
}

void Motor_Control_Tmotor::set_software_zero_rad(float raw_zero_rad)
{
    if (!isfinite(raw_zero_rad)) return;
    software_zero_rad = raw_zero_rad;
    pos = raw_position_rad - software_zero_rad;
}

void Motor_Control_Tmotor::set_software_zero_here()
{
    software_zero_rad = raw_position_rad;
    pos = 0.0f;
}

void Motor_Control_Tmotor::set_torque_constant(float kt)
{
    if (isfinite(kt) && fabsf(kt) > 1.0e-8f) {
        torque_constant = kt;
        torque_constant_valid = true;
    } else {
        torque_constant = 0.0f;
        torque_constant_valid = false;
    }
}

void Motor_Control_Tmotor::set_software_current_limit_A(float limit_A)
{
    if (isfinite(limit_A) && limit_A > 0.0f) {
        software_current_limit_A = fabsf(limit_A);
    }
}

void Motor_Control_Tmotor::set_software_torque_limit_Nm(float limit_Nm)
{
    if (isfinite(limit_Nm) && limit_Nm > 0.0f) {
        software_torque_limit_Nm = fabsf(limit_Nm);
    }
}

// =============================================================================
// MIT numeric conversion helpers
// =============================================================================
int Motor_Control_Tmotor::float_to_uint(float x,
                                        float x_min,
                                        float x_max,
                                        uint8_t nbits) const
{
    if (nbits == 0 || nbits > 30 || x_max <= x_min) return 0;

    x = clampf(x, x_min, x_max);
    const uint32_t max_int = (1UL << nbits) - 1UL;
    const float scaled = (x - x_min) * static_cast<float>(max_int) /
                         (x_max - x_min);
    return static_cast<int>(lroundf(scaled));
}

float Motor_Control_Tmotor::uint_to_float(int x_int,
                                          float x_min,
                                          float x_max,
                                          uint8_t nbits) const
{
    if (nbits == 0 || nbits > 30 || x_max <= x_min) return x_min;

    const uint32_t max_int = (1UL << nbits) - 1UL;

    if (x_int < 0) x_int = 0;
    if (static_cast<uint32_t>(x_int) > max_int) {
        x_int = static_cast<int>(max_int);
    }

    return static_cast<float>(x_int) * (x_max - x_min) /
           static_cast<float>(max_int) + x_min;
}

// =============================================================================
// Byte helpers
// =============================================================================
float Motor_Control_Tmotor::clampf(float x, float lo, float hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

void Motor_Control_Tmotor::write_u16_le(uint8_t *dst, uint16_t value)
{
    dst[0] = static_cast<uint8_t>(value & 0xFFU);
    dst[1] = static_cast<uint8_t>((value >> 8) & 0xFFU);
}

void Motor_Control_Tmotor::write_u32_le(uint8_t *dst, uint32_t value)
{
    dst[0] = static_cast<uint8_t>(value & 0xFFUL);
    dst[1] = static_cast<uint8_t>((value >> 8) & 0xFFUL);
    dst[2] = static_cast<uint8_t>((value >> 16) & 0xFFUL);
    dst[3] = static_cast<uint8_t>((value >> 24) & 0xFFUL);
}

void Motor_Control_Tmotor::write_i32_le(uint8_t *dst, int32_t value)
{
    write_u32_le(dst, static_cast<uint32_t>(value));
}

void Motor_Control_Tmotor::write_f32_le(uint8_t *dst, float value)
{
    static_assert(sizeof(float) == 4, "This driver requires 32-bit float.");

    uint32_t raw = 0;
    memcpy(&raw, &value, sizeof(raw));
    write_u32_le(dst, raw);
}

uint16_t Motor_Control_Tmotor::read_u16_le(const uint8_t *src)
{
    return static_cast<uint16_t>(src[0]) |
           (static_cast<uint16_t>(src[1]) << 8);
}

int16_t Motor_Control_Tmotor::read_i16_le(const uint8_t *src)
{
    return static_cast<int16_t>(read_u16_le(src));
}

uint32_t Motor_Control_Tmotor::read_u32_le(const uint8_t *src)
{
    return static_cast<uint32_t>(src[0]) |
           (static_cast<uint32_t>(src[1]) << 8) |
           (static_cast<uint32_t>(src[2]) << 16) |
           (static_cast<uint32_t>(src[3]) << 24);
}

int32_t Motor_Control_Tmotor::read_i32_le(const uint8_t *src)
{
    return static_cast<int32_t>(read_u32_le(src));
}

float Motor_Control_Tmotor::read_f32_le(const uint8_t *src)
{
    const uint32_t raw = read_u32_le(src);
    float value = 0.0f;
    memcpy(&value, &raw, sizeof(value));
    return value;
}